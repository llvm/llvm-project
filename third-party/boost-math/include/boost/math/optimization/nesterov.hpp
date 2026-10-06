//           Copyright Maksym Zhelyenzyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_MATH_OPTIMIZATION_NESTEROV_HPP
#define BOOST_MATH_OPTIMIZATION_NESTEROV_HPP
#include <boost/math/optimization/detail/differentiable_opt_utilties.hpp>
#include <boost/math/optimization/detail/gradient_opt_base.hpp>
#include <boost/math/optimization/detail/rdiff_optimization_policies.hpp>
#include <vector>

namespace boost {
namespace math {
namespace optimization {

namespace rdiff = boost::math::differentiation::reverse_mode;

/**
 * @brief The nesterov_update_policy class
 */
template<typename RealType>
struct nesterov_update_policy
{
  RealType lr_, mu_;
  nesterov_update_policy(RealType lr, RealType mu)
    : lr_(lr)
    , mu_(mu) {};

  template<typename ArgumentType,
           typename = typename std::enable_if<
             boost::math::differentiation::reverse_mode::detail::is_expression<
               ArgumentType>::value>::type>
  void operator()(ArgumentType& x, RealType& g, RealType& v)
  {
    RealType v_prev = v;
    v = mu_ * v - lr_ * g;
    x.get_value() += -mu_ * v_prev + (static_cast<RealType>(1) + mu_) * v;
  }
  template<typename ArgumentType,
           typename std::enable_if<!boost::math::differentiation::reverse_mode::
                                     detail::is_expression<ArgumentType>::value,
                                   int>::type = 0>
  void operator()(ArgumentType& x, RealType& g, RealType& v) const
  {
    const RealType v_prev = v;
    v = mu_ * v - lr_ * g;
    x += -mu_ * v_prev + (static_cast<RealType>(1) + mu_) * v;
  }
  RealType lr() const noexcept { return lr_; }
  RealType mu() const noexcept { return mu_; }
};

/**
 * @brief The nesterov_accelerated_gradient class
 *
 * https://jlmelville.github.io/mize/nesterov.html
 */
template<typename ArgumentContainer,
         typename RealType,
         class Objective,
         class InitializationPolicy,
         class ObjectiveEvalPolicy,
         class GradEvalPolicy>
class nesterov_accelerated_gradient
  : public abstract_optimizer<
      ArgumentContainer,
      RealType,
      Objective,
      InitializationPolicy,
      ObjectiveEvalPolicy,
      GradEvalPolicy,
      nesterov_update_policy<RealType>,
      nesterov_accelerated_gradient<ArgumentContainer,
                                    RealType,
                                    Objective,
                                    InitializationPolicy,
                                    ObjectiveEvalPolicy,
                                    GradEvalPolicy>>
{
  using base_opt =
    abstract_optimizer<ArgumentContainer,
                       RealType,
                       Objective,
                       InitializationPolicy,
                       ObjectiveEvalPolicy,
                       GradEvalPolicy,
                       nesterov_update_policy<RealType>,
                       nesterov_accelerated_gradient<ArgumentContainer,
                                                     RealType,
                                                     Objective,
                                                     InitializationPolicy,
                                                     ObjectiveEvalPolicy,
                                                     GradEvalPolicy>>;
  std::vector<RealType> v_;

public:
  using base_opt::base_opt;
  nesterov_accelerated_gradient(Objective&& objective,
                                ArgumentContainer& x,
                                InitializationPolicy&& ip,
                                ObjectiveEvalPolicy&& oep,
                                GradEvalPolicy&& gep,
                                nesterov_update_policy<RealType>&& up)
    : base_opt(std::forward<Objective>(objective),
               x,
               std::forward<InitializationPolicy>(ip),
               std::forward<ObjectiveEvalPolicy>(oep),
               std::forward<GradEvalPolicy>(gep),
               std::forward<nesterov_update_policy<RealType>>(up))
    , v_(x.size(), RealType(0))
  {
  }

  void step()
  {
    auto& x = this->arguments();
    auto& g = this->gradients();
    auto& obj = this->objective_value();
    auto& obj_eval = this->obj_eval_;
    auto& grad_eval = this->grad_eval_;
    auto& objective = this->objective_;
    auto& update = this->update_;

    grad_eval(objective, x, obj_eval, obj, g);

    for (size_t i = 0; i < x.size(); ++i) {
      update(x[i], g[i], v_[i]);
    }
  }
};
template<class Objective, typename ArgumentContainer, typename RealType>
auto
make_nag(Objective&& obj,
         ArgumentContainer& x,
         RealType lr = RealType{ 0.01 },
         RealType mu = RealType{ 0.95 })
{
  return nesterov_accelerated_gradient<
    ArgumentContainer,
    RealType,
    std::decay_t<Objective>,
    tape_initializer_rvar<RealType>,
    reverse_mode_function_eval_policy<RealType>,
    reverse_mode_gradient_evaluation_policy<RealType>>(
    std::forward<Objective>(obj),
    x,
    tape_initializer_rvar<RealType>{},
    reverse_mode_function_eval_policy<RealType>{},
    reverse_mode_gradient_evaluation_policy<RealType>{},
    nesterov_update_policy<RealType>(lr, mu));
}
template<class Objective,
         typename ArgumentContainer,
         typename RealType,
         class InitializationPolicy>
auto
make_nag(Objective&& obj,
         ArgumentContainer& x,
         RealType lr,
         RealType mu,
         InitializationPolicy&& ip)
{
  return nesterov_accelerated_gradient<
    ArgumentContainer,
    RealType,
    std::decay_t<Objective>,
    InitializationPolicy,
    reverse_mode_function_eval_policy<RealType>,
    reverse_mode_gradient_evaluation_policy<RealType>>(
    std::forward<Objective>(obj),
    x,
    std::forward<InitializationPolicy>(ip),
    reverse_mode_function_eval_policy<RealType>{},
    reverse_mode_gradient_evaluation_policy<RealType>{},
    nesterov_update_policy<RealType>(lr, mu));
}
template<typename ArgumentContainer,
         typename RealType,
         class Objective,
         class InitializationPolicy,
         class ObjectiveEvalPolicy,
         class GradEvalPolicy>
auto
make_nag(Objective&& obj,
         ArgumentContainer& x,
         RealType lr,
         RealType mu,
         InitializationPolicy&& ip,
         ObjectiveEvalPolicy&& oep,
         GradEvalPolicy&& gep)
{
  return nesterov_accelerated_gradient<ArgumentContainer,
                                       RealType,
                                       std::decay_t<Objective>,
                                       InitializationPolicy,
                                       ObjectiveEvalPolicy,
                                       GradEvalPolicy>(
    std::forward<Objective>(obj),
    x,
    std::forward<InitializationPolicy>(ip),
    std::forward<ObjectiveEvalPolicy>(oep),
    std::forward<GradEvalPolicy>(gep),
    nesterov_update_policy<RealType>{ lr, mu });
}
} // namespace optimization
} // namespace math
} // namespace boost
#endif
