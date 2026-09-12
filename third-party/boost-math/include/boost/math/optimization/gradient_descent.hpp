//           Copyright Maksym Zhelyenzyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_MATH_OPTIMIZATION_GRADIENT_DESCENT_HPP
#define BOOST_MATH_OPTIMIZATION_GRADIENT_DESCENT_HPP
#include <boost/math/optimization/detail/differentiable_opt_utilties.hpp>
#include <boost/math/optimization/detail/gradient_opt_base.hpp>
#include <boost/math/optimization/detail/rdiff_optimization_policies.hpp>

namespace boost {
namespace math {
namespace optimization {

template<typename RealType>
struct gradient_descent_update_policy
{
  RealType lr_;
  gradient_descent_update_policy(RealType lr)
    : lr_(lr) {};

  template<typename ArgumentType,
           typename = typename std::enable_if<
             boost::math::differentiation::reverse_mode::detail::is_expression<
               ArgumentType>::value>::type>
  void operator()(ArgumentType& x, RealType& g)
  {
    x.get_value() -= lr_ * g;
  }
  template<typename ArgumentType,
           typename std::enable_if<!boost::math::differentiation::reverse_mode::
                                     detail::is_expression<ArgumentType>::value,
                                   int>::type = 0>
  void operator()(ArgumentType& x, RealType& g) const
  {
    x -= lr_ * g;
  }
};

/**
 * @brief> Gradient Descent Optimizer.
 *
 * This class implements a gradient descent optimization strategy using the
 * policies provided for initialization, objective evaluation, and gradient
 * computation. It inherits from `abstract_optimizer`, which provides the
 * general optimization framework.
 *
 * @tparam> ArgumentContainer Type of the parameter container (e.g.
 *          std::vector<SomeDifferentiableType or RealType>).
 * @tparam> RealType          Floating-point type
 * @tparam> Objective         Objective function type (functor or callable).
 * @tparam> InitializationPolicy Policy controlling initialization of
 *          differentiable variables.
 * @tparam> ObjectiveEvalPolicy  Policy defining how the objective is evaluated.
 * @tparam> GradEvalPolicy       Policy defining how the gradient is computed.
 */

template<typename ArgumentContainer,
         typename RealType,
         class Objective,
         class InitializationPolicy,
         class ObjectiveEvalPolicy,
         class GradEvalPolicy>
class gradient_descent
  : public abstract_optimizer<ArgumentContainer,
                              RealType,
                              Objective,
                              InitializationPolicy,
                              ObjectiveEvalPolicy,
                              GradEvalPolicy,
                              gradient_descent_update_policy<RealType>,
                              gradient_descent<ArgumentContainer,
                                               RealType,
                                               Objective,
                                               InitializationPolicy,
                                               ObjectiveEvalPolicy,
                                               GradEvalPolicy>>
{
  using base_opt = abstract_optimizer<ArgumentContainer,
                                      RealType,
                                      Objective,
                                      InitializationPolicy,
                                      ObjectiveEvalPolicy,
                                      GradEvalPolicy,
                                      gradient_descent_update_policy<RealType>,
                                      gradient_descent<ArgumentContainer,
                                                       RealType,
                                                       Objective,
                                                       InitializationPolicy,
                                                       ObjectiveEvalPolicy,
                                                       GradEvalPolicy>>;

public:
  using base_opt::base_opt;
  /**
   * @brief Perform one optimization step.
   *
   * defaults to a regular update inside abstract optimizer
   * x -= lr * grad(f)
   */
  void step() { this->step_impl(); }
};

/**
 * @brief Create a gradient descent optimizer with default reverse-mode autodiff
 policies.
 *
 * make_gradient_descent(objective, x)
 *      constructs gradient descent with objective function, and parameters x
 *
 *      learning rate set to 0.01 by default
 *
 *      initialization strategy : use specified, gradient tape taken care of by
 *              optimizer
 *      function eval policy : rvar function eval policy, to be used with
 *              boost::math::differentiation::reverse_mode::rvar
 *
 *      gradient eval policy : rvar gradient evaluation policy
 *
 *      gradient descent update policy :
 *              basically x -= lr * grad(f);
 *
 * make_gradient_descent(objective, x, lr)
 *      custom learning rate
*/

template<class Objective, typename ArgumentContainer, typename RealType>
auto
make_gradient_descent(Objective&& obj,
                      ArgumentContainer& x,
                      RealType lr = RealType{ 0.01 })
{
  return gradient_descent<ArgumentContainer,
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
    gradient_descent_update_policy<RealType>(lr));
}

template<class Objective,
         typename ArgumentContainer,
         typename RealType,
         class InitializationPolicy>
auto
make_gradient_descent(Objective&& obj,
                      ArgumentContainer& x,
                      RealType lr,
                      InitializationPolicy&& ip)
{
  return gradient_descent<ArgumentContainer,
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
    gradient_descent_update_policy<RealType>(lr));
}

template<typename ArgumentContainer,
         typename RealType,
         class Objective,
         class InitializationPolicy,
         class ObjectiveEvalPolicy,
         class GradEvalPolicy>
auto
make_gradient_descent(Objective&& obj,
                      ArgumentContainer& x,
                      RealType& lr,
                      InitializationPolicy&& ip,
                      ObjectiveEvalPolicy&& oep,
                      GradEvalPolicy&& gep)
{
  return gradient_descent<ArgumentContainer,
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
    gradient_descent_update_policy<RealType>{ lr });
}

} // namespace optimization
} // namespace math
} // namespace boost
#endif
