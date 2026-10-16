//           Copyright Maksym Zhelyenzyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_MATH_OPTIMIZATION_DETAIL_RDIFF_OPTIMIZATION_POLICIES_HPP
#define BOOST_MATH_OPTIMIZATION_DETAIL_RDIFF_OPTIMIZATION_POLICIES_HPP

#include <boost/math/differentiation/autodiff_reverse.hpp>
#include <random>
#include <type_traits>

namespace boost {
namespace math {
namespace optimization {

namespace rdiff = boost::math::differentiation::reverse_mode;

/******************************************************************/
/**
 * @brief> function evaluation policy for reverse mode autodiff
 * @arg objective> objective function to evaluate
 * @arg x> argument list
 */
template<typename RealType>
struct reverse_mode_function_eval_policy
{
  template<typename Objective, class ArgumentContainer>
  rdiff::rvar<RealType, 1> operator()(Objective&& objective,
                                      ArgumentContainer& x)
  {
    auto& tape = rdiff::get_active_tape<RealType, 1>();
    tape.zero_grad();
    tape.rewind_to_last_checkpoint();

    return objective(x);
  }
};
/******************************************************************/

/**
 * @brief> gradient evaluation policy
 * @arg obj_f> objective
 * @arg x> argument list
 * @arg f_eval_olicy> funciton evaluation policy. These need to be
 *                    done in tandem
 * @arg obj_v> reference to variable inside gradient class
 */
template<typename RealType>
struct reverse_mode_gradient_evaluation_policy
{
  template<class Objective,
           class ArgumentContainer,
           class FunctionEvaluationPolicy>
  void operator()(Objective&& obj_f,
                  ArgumentContainer& x,
                  FunctionEvaluationPolicy&& f_eval_pol,
                  RealType& obj_v,
                  std::vector<RealType>& g)
  {
    // compute objective via eval policy that takes care of tape
    rdiff::rvar<RealType, 1> v = f_eval_pol(obj_f, x);
    v.backward();
    obj_v = v.item();
    g.resize(x.size());

    // copy gradients into gradient vector
    for (size_t i = 0; i < x.size(); ++i) {
      g[i] = x[i].adjoint();
    }
  }
};
/******************************************************************
 * init policies
 */
template<typename RealType>
struct tape_initializer_rvar
{
    template<class ArgumentContainer>
    void operator()(ArgumentContainer& x) const noexcept
    {
        static_assert(std::is_same<typename ArgumentContainer::value_type,
                                   rdiff::rvar<RealType, 1>>::value,
                      "ArgumentContainer::value_type must be rdiff::rvar<RealType,1>");
        auto& tape = rdiff::get_active_tape<RealType, 1>();
        tape.add_checkpoint();
    }
};

template<typename RealType>
struct random_uniform_initializer_rvar
{
  RealType low_, high_;
  size_t seed_;
  random_uniform_initializer_rvar(RealType low = 0,
                                  RealType high = 1,
                                  size_t seed = std::random_device{}())
    : low_(low)
    , high_(high)
    , seed_(seed) {};

  template<class ArgumentContainer>
  void operator()(ArgumentContainer& x) const
  {
    static std::mt19937 gen{ std::random_device{}() };
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& xi : x) {
      xi = rdiff::rvar<RealType, 1>(static_cast<RealType>(dist(gen)));
    }
    auto& tape = rdiff::get_active_tape<RealType, 1>();
    tape.add_checkpoint();
  }
};

template<typename RealType>
struct costant_initializer_rvar
{
  RealType constant;
  explicit costant_initializer_rvar(RealType v = 0)
    : constant(v) {};
  template<class ArgumentContainer>
  void operator()(ArgumentContainer& x) const
  {
    for (auto& xi : x) {
      xi = rdiff::rvar<RealType, 1>(constant);
    }
    auto& tape = rdiff::get_active_tape<RealType, 1>();
    tape.add_checkpoint();
  }
};
} // namespace optimization
} // namespace math
} // namespace boost

#endif
