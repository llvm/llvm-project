//           Copyright Maksym Zhelyenzyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_MATH_OPTIMIZATION_MINIMIZER_HPP
#define BOOST_MATH_OPTIMIZATION_MINIMIZER_HPP
#include <boost/math/optimization/detail/differentiable_opt_utilties.hpp>
#include <boost/math/optimization/gradient_optimizers.hpp>
#include <vector>
#include <chrono>
namespace boost {
namespace math {
namespace optimization {
template<typename RealType>
struct optimization_result
{
  size_t num_iter = 0;
  RealType objective_value;
  std::vector<RealType> objective_history;
  bool converged;
};

template<typename RealType>
std::ostream&
operator<<(std::ostream& os, const optimization_result<RealType>& r)
{
  os << "optimization_result {\n"
     << "  num_iter        = " << r.num_iter << "\n"
     << "  objective_value = " << r.objective_value << "\n"
     << "  converged       = " << std::boolalpha << r.converged << "\n"
     << "  objective_history = [";

  for (std::size_t i = 0; i < r.objective_history.size(); ++i) {
    os << r.objective_history[i];
    if (i + 1 < r.objective_history.size()) {
      os << ", ";
    }
  }
  os << "]\n}\n";
  return os;
}
/*****************************************************************************************/
template<typename RealType>
struct gradient_norm_convergence_policy
{
  RealType tol_;
  explicit gradient_norm_convergence_policy(RealType tol)
    : tol_(tol)
  {
  }

  template<class GradientContainer>
  bool operator()(const GradientContainer& g, RealType /*objective_v*/) const
  {
    return norm_2(g) < tol_;
  }
};

template<typename RealType>
struct objective_tol_convergence_policy
{
  RealType tol_;
  mutable RealType last_value_;
  mutable bool first_call_;

  explicit objective_tol_convergence_policy(RealType tol)
    : tol_(tol)
    , last_value_(0)
    , first_call_(true)
  {
  }

  template<class GradientContainer>
  bool operator()(const GradientContainer&, RealType objective_v) const
  {
    if (first_call_) {
      last_value_ = objective_v;
      first_call_ = false;
      return false;
    }
    RealType diff = abs(objective_v - last_value_);
    last_value_ = objective_v;
    return diff < tol_;
  }
};

template<typename RealType>
struct relative_objective_tol_policy
{
  RealType rel_tol_;
  mutable RealType last_value_;
  mutable bool first_call_;

  explicit relative_objective_tol_policy(RealType rel_tol)
    : rel_tol_(rel_tol)
    , last_value_(0)
    , first_call_(true)
  {
  }

  template<class GradientContainer>
  bool operator()(const GradientContainer&, RealType objective_v) const
  {
    if (first_call_) {
      last_value_ = objective_v;
      first_call_ = false;
      return false;
    }
    RealType denom = max<RealType>(1, abs(last_value_));
    RealType rel_diff = abs(objective_v - last_value_) / denom;
    last_value_ = objective_v;
    return rel_diff < rel_tol_;
  }
};

template<class Policy1, class Policy2>
struct combined_convergence_policy
{
  Policy1 p1_;
  Policy2 p2_;

  combined_convergence_policy(Policy1 p1, Policy2 p2)
    : p1_(p1)
    , p2_(p2)
  {
  }

  template<class GradientContainer, class RealType>
  bool operator()(const GradientContainer& g, RealType obj) const
  {
    return p1_(g, obj) || p2_(g, obj);
  }
};

/*****************************************************************************************/
struct max_iter_termination_policy
{
  size_t max_iter_;
  max_iter_termination_policy(size_t max_iter)
    : max_iter_(max_iter) {};
  bool operator()(size_t iter)
  {
    if (iter < max_iter_) {
      return false;
    }
    return true;
  }
};

struct wallclock_termination_policy
{
  std::chrono::steady_clock::time_point start_;
  std::chrono::milliseconds max_time_;

  explicit wallclock_termination_policy(std::chrono::milliseconds max_time)
    : start_(std::chrono::steady_clock::now())
    , max_time_(max_time)
  {
  }

  bool operator()(size_t /*iter*/) const
  {
    return std::chrono::steady_clock::now() - start_ > max_time_;
  }
};

/*****************************************************************************************/
template<typename ArgumentContainer>
struct unconstrained_policy
{
  void operator()(ArgumentContainer&) {}
};

template<typename ArgumentContainer, typename RealType>
struct box_constraints
{
  RealType min_, max_;
  box_constraints(RealType min, RealType max)
    : min_(min)
    , max_(max) {};
  void operator()(ArgumentContainer& x)
  {
    for (auto& xi : x) {
      xi = (xi < min_) ? min_ : (max_ < xi) ? max_ : xi;
    }
  }
};

template<typename ArgumentContainer, typename RealType>
struct nonnegativity_constraint
{
  void operator()(ArgumentContainer& x) const
  {
    for (auto& xi : x) {
      if (xi < RealType{ 0 })
        xi = RealType{ 0 };
    }
  }
};
template<typename ArgumentContainer, typename RealType>
struct l2_ball_constraint
{
  RealType radius_;

  explicit l2_ball_constraint(RealType radius)
    : radius_(radius)
  {
  }

  void operator()(ArgumentContainer& x) const
  {
    RealType norm2v = norm_2(x);
    if (norm2v > radius_) {
      RealType scale = radius_ / norm2v;
      for (auto& xi : x)
        xi *= scale;
    }
  }
};

template<typename ArgumentContainer, typename RealType>
struct l1_ball_constraint
{
  RealType radius_;

  explicit l1_ball_constraint(RealType radius)
    : radius_(radius)
  {
  }

  void operator()(ArgumentContainer& x) const
  {
    RealType norm1v = norm_1(x);

    if (norm1v > radius_) {
      RealType scale = radius_ / norm1v;
      for (auto& xi : x)
        xi *= scale;
    }
  }
};
template<typename ArgumentContainer, typename RealType>
struct simplex_constraint
{
  void operator()(ArgumentContainer& x) const
  {
    RealType sum = RealType{ 0 };
    for (auto& xi : x) {
      if (xi < RealType{ 0 })
        xi = RealType{ 0 }; // clip negatives
      sum += xi;
    }
    if (sum > RealType{ 0 }) {
      for (auto& xi : x)
        xi /= sum;
    }
  }
};

template<typename ArgumentContainer>
struct function_constraint
{
  using func_t = void (*)(ArgumentContainer&);

  func_t f_;

  explicit function_constraint(func_t f)
    : f_(f)
  {
  }

  void operator()(ArgumentContainer& x) const { f_(x); }
};
template<typename ArgumentContainer, typename RealType>
struct unit_sphere_constraint
{
  void operator()(ArgumentContainer& x) const
  {
    RealType norm = norm_2(x);
    if (norm > RealType{ 0 }) {
      for (auto& xi : x)
        xi /= norm;
    }
  }
};
/*****************************************************************************************/

template<class Optimizer,
         class ConstraintPolicy,
         class ConvergencePolicy,
         class TerminationPolicy>
auto
minimize_impl(Optimizer& opt,
              ConstraintPolicy project,
              ConvergencePolicy converged,
              TerminationPolicy terminate,
              bool history)
{
  optimization_result<typename Optimizer::real_type_t> result;
  size_t iter = 0;
  do {
    opt.step();
    project(opt.arguments());
    ++iter;
    if (history) {
      result.objective_history.push_back(opt.objective_value());
    }

  } while (!converged(opt.gradients(), opt.objective_value()) &&
           !terminate(iter));
  result.num_iter = iter;
  result.objective_value = opt.objective_value();
  result.converged = converged(opt.gradients(), opt.objective_value());
  return result;
}
template<class Optimizer,
         class ConstraintPolicy =
           unconstrained_policy<typename Optimizer::argument_container_t>,
         class ConvergencePolicy =
           gradient_norm_convergence_policy<typename Optimizer::real_type_t>,
         class TerminationPolicy = max_iter_termination_policy>
auto
minimize(Optimizer& opt,
         ConstraintPolicy project = ConstraintPolicy{},
         ConvergencePolicy converged =
           ConvergencePolicy{
             static_cast<typename Optimizer::real_type_t>(1e-3) },
         TerminationPolicy terminate = TerminationPolicy(100000),
         bool history = true)
{
  return minimize_impl(opt, project, converged, terminate, history);
}
} // namespace optimization
} // namespace math
} // namespace boost
#endif
