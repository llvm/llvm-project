//           Copyright Maksym Zhelyenzyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_MATH_OPTIMIZATION_DETAIL_GRADIENT_OPT_BASE_HPP
#define BOOST_MATH_OPTIMIZATION_DETAIL_GRADIENT_OPT_BASE_HPP
#include <boost/math/differentiation/autodiff_reverse.hpp>

namespace boost {
namespace math {
namespace optimization {

namespace rdiff = boost::math::differentiation::reverse_mode;

/**
 * @brief The abstract_optimizer class implementing common variables
 * and methods across optimizers
 *
 * @tparam> ArgumentContainer
 * @tparam> RealType
 * @tparam> Objective
 * @tparam> InitializationPolicy
 *
 */

template<typename ArgumentContainer,
         typename RealType,
         class Objective,
         class InitializationPolicy,
         class ObjectiveEvalPolicy,
         class GradEvalPolicy,
         class UpdatePolicy,
         typename DerivedOptimizer>
class abstract_optimizer
{
protected:
  Objective objective_;          // obj function
  ArgumentContainer& x_;         // arguments to objective function
  std::vector<RealType> g_;      // container of references to gradients
  ObjectiveEvalPolicy obj_eval_; // how to evaluate your funciton
  GradEvalPolicy grad_eval_;     // how to evaluate/bind gradients
  InitializationPolicy init_;    // how to initialize the problem
  UpdatePolicy update_;          // update step
  RealType obj_v_;               // objective value (for history)
  // access derived class
  DerivedOptimizer& derived() { return static_cast<DerivedOptimizer&>(*this); }
  const DerivedOptimizer& derived() const
  {
    return static_cast<const DerivedOptimizer&>(*this);
  }

  void step_impl()
  {
    grad_eval_(objective_, x_, obj_eval_, obj_v_, g_);
    for (size_t i = 0; i < x_.size(); ++i) {
      update_(x_[i], g_[i]);
    }
  };

public:
  using argument_container_t = ArgumentContainer;
  using real_type_t = RealType;

  abstract_optimizer(Objective&& objective,
                     ArgumentContainer& x,
                     InitializationPolicy&& ip,
                     ObjectiveEvalPolicy&& oep,
                     GradEvalPolicy&& gep,
                     UpdatePolicy&& up)
    : objective_(std::forward<Objective>(objective))
    , x_(x)
    , obj_eval_(std::forward<ObjectiveEvalPolicy>(oep))
    , grad_eval_(std::forward<GradEvalPolicy>(gep))
    , init_(std::forward<InitializationPolicy>(ip))
    , update_(std::forward<UpdatePolicy>(up))
  {
    init_(x_);            // initialize your problem
    g_.resize(x_.size()); // initialize space for gradients
  }

  ArgumentContainer& arguments() { return derived().x_; }
  const ArgumentContainer& arguments() const { return derived().x_; }

  RealType& objective_value() { return derived().obj_v_; }
  const RealType& objective_value() const { return derived().obj_v_; }
  std::vector<RealType>& gradients() { return derived().g_; }
  const std::vector<RealType>& gradients() const { return derived().g_; }
};
} // namespace optimization
} // namespace math
} // namespace boost
#endif
