#ifndef FAST_GICP_FAST_GICP_IMPL_HPP
#define FAST_GICP_FAST_GICP_IMPL_HPP

#include <fast_gicp/so3/so3.hpp>
/*
 * 这个头文件包含了FastGICP类的实现，和fast_gicp.hpp的关系是前者是后者的实现
 *
 * */
namespace fast_gicp {

/// @brief  模板类定义与构造函数
/// @tparam PointSource 输入的点云类型
/// @tparam PointTarget 目标的点云类型
/// @tparam SearchMethodSource 搜索源点云的方法
/// @tparam SearchMethodTarget 搜索目标点云的方法
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::FastGICP() {
//若使用 OpenMP，则 num_threads_ 设置为最大线程数，否则为1
#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  k_correspondences_ = 20;//用于近邻搜索
  reg_name_ = "FastGICP";//注册方法名称
  corr_dist_threshold_ = std::numeric_limits<float>::max();

  regularization_method_ = RegularizationMethod::PLANE;//注册匹配的方法
  search_source_.reset(new SearchMethodSource);
  search_target_.reset(new SearchMethodTarget);
}

/// @brief 析构函数
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::~FastGICP() {}

/// @brief 设置线程数
/// @param n 线程数量
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setNumThreads(int n) {
  num_threads_ = n;

#ifdef _OPENMP
  if (n == 0) {
    num_threads_ = omp_get_max_threads();
  }
#endif
}

/// @brief 设置对应点的随机性
/// @param k 每个点寻找的最近邻点的数量
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

/// @brief 设置匹配方法
/// @param method 设置的匹配方法
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

/// @brief 交换源点云和目标点云
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  search_source_.swap(search_target_);
  source_covs_.swap(target_covs_);

  correspondences_.clear();
  sq_distances_.clear();
}
/// @brief 清空源点云
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearSource() {
  input_.reset();
  source_covs_.clear();
}
/// @brief 清空目标点云
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearTarget() {
  target_.reset();
  target_covs_.clear();
}
/// @brief 设置输入源点云
/// @param cloud 点云信息
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  search_source_->setInputCloud(cloud);//点云输入
  source_covs_.clear();//清空源点云的协方差矩阵
}

/// @brief 设置输入目标点云
/// @param cloud 点云信息
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  search_target_->setInputCloud(cloud);
  target_covs_.clear();//清空目标点云的协方差矩阵
}
/// @brief 设置源点云的协方差矩阵
/// @param covs 设置源点云的协方差矩阵
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  source_covs_ = covs;
}
/// @brief 设置目标点云的协方差矩阵
/// @param covs 协方差
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  target_covs_ = covs;
}
/// @brief 计算转换矩阵,这里会在align中完成调用
/// @param output 输出点云
/// @param guess 初始猜测的转换矩阵
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  if (output.points.data() == input_->points.data() || output.points.data() == target_->points.data()) {//确保输出点云与源点云或目标点云不同
    throw std::invalid_argument("FastGICP: destination cloud cannot be identical to source or target");
  }
  if (source_covs_.size() != input_->size()) {//如果源点云或目标点云的协方差矩阵数量与点云大小不一致
    calculate_covariances(input_, *search_source_, source_covs_);//计算协方差矩阵
  }
  if (target_covs_.size() != target_->size()) {
    calculate_covariances(target_, *search_target_, target_covs_);
  }

  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);//调用父类的computeTransformation函数，这里会计算
}

/// @brief 更新对应关系
/// @param trans 位置变换
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_.size() == input_->size());
  assert(target_covs_.size() == target_->size());

  Eigen::Isometry3f trans_f = trans.cast<float>();//转换为float类型

  correspondences_.resize(input_->size());//存储每个源点对应的目标点的索引
  sq_distances_.resize(input_->size());//存储每个源点与其对应目标点之间的平方距离
  mahalanobis_.resize(input_->size());//存储每个对应关系的马氏距离矩阵

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);

#pragma omp parallel for num_threads(num_threads_) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)//并行计算，这里设置了线程数、线程间的私有变量、调度方式
  for (int i = 0; i < input_->size(); i++) {
    PointTarget pt;
    pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();//根据位置变换计算变换后的点

    search_target_->nearestKSearch(pt, 1, k_indices, k_sq_dists);//搜索最近的目标点

    sq_distances_[i] = k_sq_dists[0];//存储平方距离
    correspondences_[i] = k_sq_dists[0] < corr_dist_threshold_ * corr_dist_threshold_ ? k_indices[0] : -1;//如果距离小于阈值，则认为是对应点，否则为-1

    if (correspondences_[i] < 0) {
      continue;
    }

    const int target_index = correspondences_[i];
    const auto& cov_A = source_covs_[i];
    const auto& cov_B = target_covs_[target_index];

    Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();//根据两个点云的协方差矩阵计算马氏距离矩阵
    RCR(3, 3) = 1.0;

    mahalanobis_[i] = RCR.inverse();
    mahalanobis_[i](3, 3) = 0.0f;
  }
}

/// @brief 计算线性化,H为雅可比矩阵，b为残差,公式其实就是H /delta x = b
/// @param trans 位置变换
/// @param H 线性化矩阵
/// @param b 线性化向量
/// @return 
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
double FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  update_correspondences(trans);//根据位置变换更新对应关系

  double sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];//获取对应关系
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();//获取源点云的点
    const auto& cov_A = source_covs_[i];//获取源点云的协方差矩阵

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;//根据位置变换计算变换后的点
    const Eigen::Vector4d error = mean_B - transed_mean_A;//计算残差

    sum_errors += error.transpose() * mahalanobis_[i] * error;//计算残差的马氏距离

    if (H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());//这里是计算雅可比矩阵，并生成斜对称矩阵。这里斜对称矩阵是因为error =mean_B - transed_mean_A =(求导和mean_B没关系，对transed_mean_A求导旋转θ) R(δθ)≈I+δθ×,最终结果就是δθ×
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();//创建一个 3x3 的单位矩阵，然后取负号，即每个元素取相反数

    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = jlossexp.transpose() * mahalanobis_[i] * jlossexp;//海森矩阵计算公式：H = J^T * R * J
    Eigen::Matrix<double, 6, 1> bi = jlossexp.transpose() * mahalanobis_[i] * error;//残差计算公式：b = J^T * R * e

    Hs[omp_get_thread_num()] += Hi;
    bs[omp_get_thread_num()] += bi;
  }

  if (H && b) {
    H->setZero();
    b->setZero();
    for (int i = 0; i < num_threads_; i++) {
      (*H) += Hs[i];
      (*b) += bs[i];
    }
  }

  return sum_errors;
}

/// @brief 这个方法计算给定转换矩阵的总误差
/// @param trans 旋转矩阵
/// @return 
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
double FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::compute_error(const Eigen::Isometry3d& trans) {
  double sum_errors = 0.0;

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;//和上面的linearize函数一样，这里是计算误差

    sum_errors += error.transpose() * mahalanobis_[i] * error;
  }

  return sum_errors;
}

/// @brief 计算协方差矩阵
/// @param cloud 对应的点云特征信息
/// @param kdtree 树特征
/// @param covariances 协方差矩阵
/// @return 
template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_covariances(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);//将点云设置为搜索树的输入
  }
  covariances.resize(cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);//kd 树搜索最近的k个点

    Eigen::Matrix<double, 4, -1> neighbors(4, k_correspondences_);
    for (int j = 0; j < k_indices.size(); j++) {
      neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();//邻近点位置
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();//计算均值,并减去均值.colwise()表示对列进行操作，.rowwise()表示对行进行操作
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;//计算协方差矩阵

    if (regularization_method_ == RegularizationMethod::NONE) {//无正则化
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {// Frobenius 正则化方法
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();//协方差矩阵的前 3x3 部分加上一个 λ 倍的单位矩阵
      Eigen::Matrix3d C_inv = C.inverse();//计算其逆矩阵
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();//计算正则化后的协方差矩阵
    } else {
        //否则使用svd分解，来计算协方差矩阵
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d values;

      switch (regularization_method_) {
        default:
          std::cerr << "here must not be reached" << std::endl;
          abort();
        case RegularizationMethod::PLANE:
          values = Eigen::Vector3d(1, 1, 1e-3);//认为只有Z轴的方差很小，可以被约束
          break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);//找到svd中的值，并让奇异值调整为不小于 1e-3
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();//将奇异值归一化
          values = values.array().max(1e-3);//找到最大值,并让奇异值调整为不小于 1e-3
          break;
      }

      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();//A=U Σ V^T 
    }
  }

  return true;
}

}  // namespace fast_gicp

#endif
