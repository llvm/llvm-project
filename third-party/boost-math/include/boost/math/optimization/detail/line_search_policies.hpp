//           Copyright Maksym Zhelyenzyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_OPTIMIZATION_DETAIL_LINE_SEARCH_POLICIES_HPP
#define BOOST_MATH_OPTIMIZATION_DETAIL_LINE_SEARCH_POLICIES_HPP

#include <boost/math/optimization/detail/differentiable_opt_utilties.hpp>
#include <cmath>
#include <vector>
namespace boost {
namespace math {
namespace optimization {

/**
 * @brief> Armijo condition backtracking line search
 * https://en.wikipedia.org/wiki/Backtracking_line_search
 *
 *  f(x+alpha p) <= f(x) + alpha * c * grad(f)^T
 *
 * Jorge Nocedal and Stephen J. Wright,
 *  Numerical Optimization, 2nd Edition,
 *  Springer, 2006.
 *
 *  Algorithm 3.1: Backtracking Line Search
 *  (Page 37)
 */
template<typename RealType>
class armijo_line_search_policy
{
private:
  RealType alpha0_; // initial step size
  RealType c_;      // sufficient decrease constant
  RealType rho_;    // backtracking factor
  int max_iter_;    // maximum backtracking steps

public:
  armijo_line_search_policy(RealType alpha0 = 1.0,
                            RealType c = 1e-4,
                            RealType rho = 0.5,
                            int max_iter = 20)
    : alpha0_(alpha0)
    , c_(c)
    , rho_(rho)
    , max_iter_(max_iter)
  {
  }

  template<class Objective,
           class ObjectiveEvalPolicy,
           class GradientEvalPolicy,
           class ArgumentContainer>
  RealType operator()(Objective& objective,
                      ObjectiveEvalPolicy& obj_eval,
                      GradientEvalPolicy& grad_eval,
                      ArgumentContainer& x,
                      const std::vector<RealType>& g,
                      const std::vector<RealType>& p,
                      RealType f_x) const
  {
    /** @brief> line search
     * */
    RealType alpha = alpha0_;
    ArgumentContainer x_trial; // = x; // copy
    const RealType gTp = dot(g, p);

    for (int iter = 0; iter < max_iter_; ++iter) {
      x_trial = x;
      axpy(alpha, p, x_trial);
      auto f_trial = obj_eval(objective, x_trial);
      if (f_trial <=
          f_x + c_ * alpha * gTp) // check if armijo condition is satisfied
        return alpha;
      alpha *= rho_;
    }
    return alpha;
  }
};

/** @brief> Strong-Wolfe line search:
 *  Jorge Nocedal and Stephen J. Wright,
 *  Numerical Optimization, 2nd Edition,
 *  Springer, 2006.
 *
 *  Algorithm 3.5  Line Search Algorithm (Strong Wolfe Conditions)
 *  pages 60-61
 */
template<typename RealType>
class strong_wolfe_line_search_policy
{
private:
  RealType alpha0_; // initial step size
  RealType c1_;     // Armijo constant (sufficient decrease)
  RealType c2_;     // curvature constant
  RealType rho_;    // backtracking factor
  int max_iter_;    // maximum iterations
public:
  strong_wolfe_line_search_policy(RealType alpha0 = 1.0,
                                  RealType c1 = 1e-4,
                                  RealType c2 = 0.9,
                                  RealType rho = 2.0,
                                  int max_iter = 20)
    : alpha0_(alpha0)
    , c1_(c1)
    , c2_(c2)
    , rho_(rho)
    , max_iter_(max_iter)
  {
  }
  template<class Objective,
           class ObjectiveEvalPolicy,
           class GradientEvalPolicy,
           class ArgumentContainer>
  RealType operator()(Objective& objective,
                      ObjectiveEvalPolicy& obj_eval,
                      GradientEvalPolicy& grad_eval,
                      ArgumentContainer& x,
                      const std::vector<RealType>& g,
                      const std::vector<RealType>& p,
                      RealType f_x) const
  {
    RealType gTp0 = dot(g, p);
    RealType alpha_prev{0};
    RealType f_prev = f_x;
    RealType alpha = alpha0_;

    ArgumentContainer x_trial;
    std::vector<RealType> g_trial(g.size());
    for (int i = 0; i < max_iter_; ++i) {
      x_trial = x; // explicit copy
      axpy(alpha, p, x_trial);
      RealType f_trial = static_cast<RealType>(obj_eval(objective, x_trial));
      if ((f_trial > f_x + c1_ * alpha * gTp0) ||
          (i > 0 && f_trial >= f_prev)) {
        return zoom(
          objective, obj_eval, grad_eval, x, p, f_x, gTp0, alpha_prev, alpha);
      }
      grad_eval(objective, x_trial, obj_eval, f_trial, g_trial);
      RealType gTp = dot(g_trial, p);

      if (fabs(gTp) <= c2_ * fabs(gTp0)) {
        return alpha;
      }

      if (gTp >= 0) {
        return zoom(
          objective, obj_eval, grad_eval, x, p, f_x, gTp0, alpha, alpha_prev);
      }
      alpha_prev = alpha;
      f_prev = f_trial;
      alpha *= rho_;
    }

    return alpha;
  }

private:
  template<class Objective,
           class ObjectiveEvalPolicy,
           class GradientEvalPolicy,
           class ArgumentContainer>
  RealType zoom(Objective& objective,
                ObjectiveEvalPolicy& obj_eval,
                GradientEvalPolicy& grad_eval,
                ArgumentContainer& x,
                const std::vector<RealType>& p,
                RealType f_x,
                RealType gTp0,
                RealType alpha_lo,
                RealType alpha_hi) const
  {
    ArgumentContainer x_trial;
    std::vector<RealType> g_trial(p.size());

    for (int iter = 0; iter < max_iter_; ++iter) {
      const RealType alpha_mid = 0.5 * (alpha_lo + alpha_hi);
      x_trial = x;
      axpy(alpha_mid, p, x_trial);
      RealType f_mid;
      grad_eval(objective, x_trial, obj_eval, f_mid, g_trial);
      RealType gTp_mid = dot(g_trial, p);
      ArgumentContainer x_lo = x;
      axpy(alpha_lo, p, x_lo);
      RealType f_lo = static_cast<RealType>(obj_eval(objective, x_lo));
      if ((f_mid > f_x + c1_ * alpha_mid * gTp0) || (f_mid >= f_lo)) {
        alpha_hi = alpha_mid;
      } else {
        if (fabs(gTp_mid) <= c2_ * fabs(gTp0)) {
          return alpha_mid;
        }
        if (gTp_mid * (alpha_hi - alpha_lo) >= 0) {
          alpha_hi = alpha_lo;
        }
        alpha_lo = alpha_mid;
      }
    }

    return 0.5 * (alpha_lo + alpha_hi);
  }
};

} // namespace optimization
} // namespace math
} // namespace boost
#endif // LINE_SEARCH_POLICIES_HPP
