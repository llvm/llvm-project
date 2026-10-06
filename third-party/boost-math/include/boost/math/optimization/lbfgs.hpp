//           Copyright Maksym Zhelyenzyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_MATH_OPTIMIZATION_LBFGS_HPP
#define BOOST_MATH_OPTIMIZATION_LBFGS_HPP
#include <boost/math/optimization/detail/differentiable_opt_utilties.hpp>
#include <boost/math/optimization/detail/gradient_opt_base.hpp>
#include <boost/math/optimization/detail/rdiff_optimization_policies.hpp>
#include <vector>

#include <boost/math/optimization/detail/line_search_policies.hpp>
#include <deque>

namespace boost {
namespace math {
namespace optimization {

/** @brief> Helper struct for L-BFGS
 *
 * stores state of L-BFGS optimizer
 *  @param> m = 10 -> history (how far back to look)
 *  @param> S - > x_k - x_{k-1} ( m states )
 *  @param> Y - > g_k - g_{k-1} (m states)
 *  @param> rho - > 1/(y^T y)
 *  @param> g_prev, x_prev - > previous state of argument and gradient
 *  @param -> f_prev - > previous function value
 *
 *  https://en.wikipedia.org/wiki/Limited-memory_BFGS
 *
 *  Jorge Nocedal and Stephen J. Wright,
 *  Numerical Optimization, 2nd Edition,
 *  Springer, 2006.
 *
 *  pages 176-180
 *  algorithms 7.4/7.5
 *  */
template<typename RealType>
struct lbfgs_optimizer_state
{
  size_t m = 10; // default history length
  std::deque<std::vector<RealType>> S, Y;
  std::deque<RealType> rho;
  std::vector<RealType> g_prev, x_prev;
  RealType f_prev = std::numeric_limits<RealType>::quiet_NaN();
  const RealType EPS = std::numeric_limits<RealType>::epsilon();

  template<typename ArgumentContainer>
  void update_state(ArgumentContainer& x,
                    std::vector<RealType>& g_k,
                    RealType fk)
  {

    // iteration 0
    if (g_prev.empty()) {
      g_prev.assign(g_k.begin(), g_k.end());
      x_prev.resize(x.size());
      std::transform(x.begin(), x.end(), x_prev.begin(), [](const auto& xi) {
        return static_cast<RealType>(xi);
      });
      f_prev = fk;
      return;
    }

    std::vector<RealType> s_k(x.size()), y_k(g_k.size());
    for (size_t i = 0; i < x.size(); ++i) {
      s_k[i] = static_cast<RealType>(x[i]) - x_prev[i];
      y_k[i] = g_k[i] - g_prev[i];
    }

    RealType ys = dot(y_k, s_k);
    RealType sn = sqrt(dot(s_k, s_k));
    RealType yn = sqrt(dot(y_k, y_k));

    const RealType threshold = EPS * sn * yn;
    if (ys > threshold && ys > RealType(0)) { // check if curvature if non-zero
      if (S.size() == m) {                    // iteration > m
        S.pop_front();
        Y.pop_front();
        rho.pop_front();
      }
      S.push_back(std::move(s_k));
      Y.push_back(std::move(y_k));
      rho.push_back(RealType(1) / ys);
    }

    g_prev.assign(g_k.begin(), g_k.end());
    // safely cast to realtype
    std::transform(x.begin(), x.end(), x_prev.begin(), [](const auto& xi) {
      return static_cast<RealType>(xi);
    });
    f_prev = fk;
  }
};

/** @brief> helper update for l-bfgs
 * x +=  alpha  * search direction
 * */
template<typename RealType>
struct lbfgs_update_policy
{
  template<typename ArgumentType,
           typename = typename std::enable_if<
             boost::math::differentiation::reverse_mode::detail::is_expression<
               ArgumentType>::value>::type>
  void operator()(ArgumentType& x, RealType pk, RealType alpha)
  {
      x.get_value() += alpha * pk;
  }
  template<typename ArgumentType,
           typename std::enable_if<!boost::math::differentiation::reverse_mode::
                                     detail::is_expression<ArgumentType>::value,
                                   int>::type = 0>
  void operator()(ArgumentType& x, RealType pk, RealType alpha)
  {
    x += alpha * pk;
  }
};

/**
 *
 * @brief Limited-memory BFGS (L-BFGS) optimizer
 *
 * The `lbfgs` class implements the Limited-memory BFGS optimization algorithm,
 * a quasi-Newton method that approximates the inverse Hessian using a rolling
 * window of the last `m` updates. It is suitable for medium- to large-scale
 * optimization problems where full Hessian storage is infeasible.
 *
 * @tparam> ArgumentContainer: container type for parameters, e.g.
 * std::vector<RealType>
 * @tparam> RealType scalar floating type (e.g. double, float)
 * @tparam> Objective: objective function. must support "f(x)" evaluation
 * @tparam> InitializationPolicy: policy for initializing x
 * @tparam> ObjectiveEvalPolicy: policy for computing the objective value
 * @tparam> GradEvalPolicy: policy for computing gradients
 * @tparam> LineaSearchPolicy: e.g. Armijo, StrongWolfe
 *
 * https://en.wikipedia.org/wiki/Limited-memory_BFGS
 */

template<typename ArgumentContainer,
         typename RealType,
         class Objective,
         class InitializationPolicy,
         class ObjectiveEvalPolicy,
         class GradEvalPolicy,
         class LineSearchPolicy>
class lbfgs
  : public abstract_optimizer<ArgumentContainer,
                              RealType,
                              Objective,
                              InitializationPolicy,
                              ObjectiveEvalPolicy,
                              GradEvalPolicy,
                              lbfgs_update_policy<RealType>,
                              lbfgs<ArgumentContainer,
                                    RealType,
                                    Objective,
                                    InitializationPolicy,
                                    ObjectiveEvalPolicy,
                                    GradEvalPolicy,
                                    LineSearchPolicy>>
{
  using base_opt = abstract_optimizer<ArgumentContainer,
                                      RealType,
                                      Objective,
                                      InitializationPolicy,
                                      ObjectiveEvalPolicy,
                                      GradEvalPolicy,
                                      lbfgs_update_policy<RealType>,
                                      lbfgs<ArgumentContainer,
                                            RealType,
                                            Objective,
                                            InitializationPolicy,
                                            ObjectiveEvalPolicy,
                                            GradEvalPolicy,
                                            LineSearchPolicy>>;

  const RealType EPS = std::numeric_limits<RealType>::epsilon();
  lbfgs_optimizer_state<RealType> state_;
  LineSearchPolicy line_search_;

  std::vector<RealType> compute_direction(const std::vector<RealType>& gk)
  {
    const size_t n = gk.size();
    const size_t L = state_.S.size(); // since S changes when iter < m

    if (L == 0) {
      std::vector<RealType> p(n);
      std::transform(
        gk.begin(), gk.end(), p.begin(), [](RealType gi) { return -gi; });
      return p;
    }

    std::vector<RealType> q = gk;
    std::vector<RealType> alpha(L, RealType(0));
    for (size_t t = 0; t < L; ++t) {
      const std::size_t i = L - 1 - t; // newest first
      const RealType sTq = dot(state_.S[i], q);
      alpha[i] = state_.rho[i] * sTq;
      axpy(-alpha[i], state_.Y[i], q);
    }

    const RealType sTy = dot(state_.S.back(), state_.Y.back());
    const RealType yTy = dot(state_.Y.back(), state_.Y.back());
    const RealType gamma = (yTy > RealType(0)) ? (sTy / yTy) : RealType(1);

    std::vector<RealType> r = q;
    scale(r, gamma);

    for (std::size_t i = 0; i < L; ++i) {
      const RealType yTr = dot(state_.Y[i], r);
      const RealType beta = state_.rho[i] * yTr;
      axpy(alpha[i] - beta, state_.S[i], r);
    }
    scale(r, RealType{ -1 });
    return r;
  }

public:
  using base_opt::base_opt;
  lbfgs(Objective&& objective,
        ArgumentContainer& x,
        size_t m,
        InitializationPolicy&& ip,
        ObjectiveEvalPolicy&& oep,
        GradEvalPolicy&& gep,
        lbfgs_update_policy<RealType>&& up,
        LineSearchPolicy&& lsp)
    : base_opt(std::forward<Objective>(objective),
               x,
               std::forward<InitializationPolicy>(ip),
               std::forward<ObjectiveEvalPolicy>(oep),
               std::forward<GradEvalPolicy>(gep),
               std::forward<lbfgs_update_policy<RealType>>(up))
    , line_search_(lsp)
  {
    state_.m = m;

    state_.S.clear();
    state_.Y.clear();
    state_.rho.clear();
    state_.g_prev.clear();
    state_.f_prev = std::numeric_limits<RealType>::quiet_NaN();
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
    state_.update_state(x, g, obj);
    std::vector<RealType> p = compute_direction(g);
    RealType alpha = line_search_(objective, obj_eval, grad_eval, x, g, p, obj);
    for (size_t i = 0; i < x.size(); ++i) {
      update(x[i], p[i], alpha);
    }
  }
};

template<class Objective, typename ArgumentContainer>
auto
make_lbfgs(Objective&& obj, ArgumentContainer& x, std::size_t m = 10)
{
    using RealType = typename argument_container_t<ArgumentContainer>::type;
    return lbfgs<ArgumentContainer,
                 RealType,
                 std::decay_t<Objective>,
                 tape_initializer_rvar<RealType>,
                 reverse_mode_function_eval_policy<RealType>,
                 reverse_mode_gradient_evaluation_policy<RealType>,
                 strong_wolfe_line_search_policy<RealType>>(
        std::forward<Objective>(obj),
        x,
        m,
        tape_initializer_rvar<RealType>{},
        reverse_mode_function_eval_policy<RealType>{},
        reverse_mode_gradient_evaluation_policy<RealType>{},
        lbfgs_update_policy<RealType>{},
        strong_wolfe_line_search_policy<RealType>{});
}

template<class Objective,
         typename ArgumentContainer,
         class InitializationPolicy>
auto
make_lbfgs(Objective&& obj,
           ArgumentContainer& x,
           std::size_t m,
           InitializationPolicy&& ip)
{
  using RealType = typename argument_container_t<ArgumentContainer>::type;

  return lbfgs<ArgumentContainer,
               RealType,
               std::decay_t<Objective>,
               InitializationPolicy,
               reverse_mode_function_eval_policy<RealType>,
               reverse_mode_gradient_evaluation_policy<RealType>,
               strong_wolfe_line_search_policy<RealType>>(
    std::forward<Objective>(obj),
    x,
    m,
    std::forward<InitializationPolicy>(ip),
    reverse_mode_function_eval_policy<RealType>{},
    reverse_mode_gradient_evaluation_policy<RealType>{},
    lbfgs_update_policy<RealType>{},
    strong_wolfe_line_search_policy<RealType>{});
}

template<class Objective,
         typename ArgumentContainer,
         class InitializationPolicy,
         class LineSearchPolicy>
auto
make_lbfgs(Objective&& obj,
           ArgumentContainer& x,
           std::size_t m,
           InitializationPolicy&& ip,
           LineSearchPolicy&& lsp)
{
  using RealType = typename argument_container_t<ArgumentContainer>::type;

  return lbfgs<ArgumentContainer,
               RealType,
               std::decay_t<Objective>,
               InitializationPolicy,
               reverse_mode_function_eval_policy<RealType>,
               reverse_mode_gradient_evaluation_policy<RealType>,
               LineSearchPolicy>(
    std::forward<Objective>(obj),
    x,
    m,
    std::forward<InitializationPolicy>(ip),
    reverse_mode_function_eval_policy<RealType>{},
    reverse_mode_gradient_evaluation_policy<RealType>{},
    lbfgs_update_policy<RealType>{},
    std::forward<LineSearchPolicy>(lsp));
}

template<class Objective,
         typename ArgumentContainer,
         class InitializationPolicy,
         class FunctionEvalPolicy,
         class GradientEvalPolicy,
         class LineSearchPolicy>
auto
make_lbfgs(Objective&& obj,
           ArgumentContainer& x,
           std::size_t m,
           InitializationPolicy&& ip,
           FunctionEvalPolicy&& fep,
           GradientEvalPolicy&& gep,
           LineSearchPolicy&& lsp)
{
  using RealType = typename argument_container_t<ArgumentContainer>::type;
  return lbfgs<ArgumentContainer,
               RealType,
               std::decay_t<Objective>,
               InitializationPolicy,
               FunctionEvalPolicy,
               GradientEvalPolicy,
               LineSearchPolicy>(std::forward<Objective>(obj),
                                 x,
                                 m,
                                 std::forward<InitializationPolicy>(ip),
                                 std::forward<FunctionEvalPolicy>(fep),
                                 std::forward<GradientEvalPolicy>(gep),
                                 lbfgs_update_policy<RealType>{},
                                 std::forward<LineSearchPolicy>(lsp));
}

} // namespace optimization
} // namespace math
} // namespace boost
#endif
