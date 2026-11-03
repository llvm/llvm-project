//           Copyright Maksym Zhelyeznyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)
#ifndef REVERSE_MODE_AUTODOFF_BASIC_OPERATOR_OVERLOADS_HPP
#define REVERSE_MODE_AUTODOFF_BASIC_OPERATOR_OVERLOADS_HPP

#include <boost/math/differentiation/detail/reverse_mode_autodiff_expression_template_base.hpp>

namespace boost {
namespace math {
namespace differentiation {
namespace reverse_mode {
/****************************************************************************************************************/
template<typename RealType, size_t DerivativeOrder, typename LHS, typename RHS>
struct add_expr : public abstract_binary_expression<RealType,
                                                    DerivativeOrder,
                                                    LHS,
                                                    RHS,
                                                    add_expr<RealType, DerivativeOrder, LHS, RHS>>
{
    /* @brief addition
   * rvar+rvar
   * */
    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;
    // Explicitly define constructor to forward to base class
    explicit add_expr(const expression<RealType, DerivativeOrder, LHS> &left_hand_expr,
                      const expression<RealType, DerivativeOrder, RHS> &right_hand_expr)
        : abstract_binary_expression<RealType,
                                     DerivativeOrder,
                                     LHS,
                                     RHS,
                                     add_expr<RealType, DerivativeOrder, LHS, RHS>>(left_hand_expr,
                                                                                    right_hand_expr)
    {}

    inner_t              evaluate() const { return this->lhs.evaluate() + this->rhs.evaluate(); }
    static const inner_t left_derivative(const inner_t & /*l*/,
                                         const inner_t & /*r*/,
                                         const inner_t & /*v*/)
    {
        return inner_t(static_cast<RealType>(1.0));
    }
    static const inner_t right_derivative(const inner_t & /*l*/,
                                          const inner_t & /*r*/,
                                          const inner_t & /*v*/)
    {
        return inner_t(static_cast<RealType>(1.0));
    }
};
template<typename RealType, size_t DerivativeOrder, typename ARG>
struct add_const_expr
    : public abstract_unary_expression<RealType,
                                       DerivativeOrder,
                                       ARG,
                                       add_const_expr<RealType, DerivativeOrder, ARG>>
{
    /* @brief
   * rvar+float or float+rvar
   * */
    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;
    explicit add_const_expr(const expression<RealType, DerivativeOrder, ARG> &arg_expr,
                            const RealType                                    v)
        : abstract_unary_expression<RealType,
                                    DerivativeOrder,
                                    ARG,
                                    add_const_expr<RealType, DerivativeOrder, ARG>>(arg_expr, v){};
    inner_t              evaluate() const { return this->arg.evaluate() + inner_t(this->constant); }
    static const inner_t derivative(const inner_t & /*argv*/,
                                    const inner_t & /*v*/,
                                    const RealType & /*constant*/)
    {
        return inner_t(static_cast<RealType>(1.0));
    }
};
/****************************************************************************************************************/
template<typename RealType, size_t DerivativeOrder, typename LHS, typename RHS>
struct mult_expr : public abstract_binary_expression<RealType,
                                                     DerivativeOrder,
                                                     LHS,
                                                     RHS,
                                                     mult_expr<RealType, DerivativeOrder, LHS, RHS>>
{
    /* @brief multiplication
   * rvar * rvar
   * */
    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;
    explicit mult_expr(const expression<RealType, DerivativeOrder, LHS> &left_hand_expr,
                       const expression<RealType, DerivativeOrder, RHS> &right_hand_expr)
        : abstract_binary_expression<RealType,
                                     DerivativeOrder,
                                     LHS,
                                     RHS,
                                     mult_expr<RealType, DerivativeOrder, LHS, RHS>>(left_hand_expr,
                                                                                     right_hand_expr)
    {}

    inner_t              evaluate() const { return this->lhs.evaluate() * this->rhs.evaluate(); };
    static const inner_t left_derivative(const inner_t & /*l*/,
                                         const inner_t &r,
                                         const inner_t & /*v*/) noexcept
    {
        return r;
    };
    static const inner_t right_derivative(const inner_t &l,
                                          const inner_t & /*r*/,
                                          const inner_t & /*v*/) noexcept
    {
        return l;
    };
};
template<typename RealType, size_t DerivativeOrder, typename ARG>
struct mult_const_expr
    : public abstract_unary_expression<RealType,
                                       DerivativeOrder,
                                       ARG,
                                       mult_const_expr<RealType, DerivativeOrder, ARG>>
{
    /* @brief
   * rvar+float or float+rvar
   * */
    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;

    explicit mult_const_expr(const expression<RealType, DerivativeOrder, ARG> &arg_expr,
                             const RealType                                    v)
        : abstract_unary_expression<RealType,
                                    DerivativeOrder,
                                    ARG,
                                    mult_const_expr<RealType, DerivativeOrder, ARG>>(arg_expr, v){};

    inner_t              evaluate() const { return this->arg.evaluate() * inner_t(this->constant); }
    static const inner_t derivative(const inner_t & /*argv*/,
                                    const inner_t & /*v*/,
                                    const RealType &constant)
    {
        return inner_t(constant);
    }
};
/****************************************************************************************************************/
template<typename RealType, size_t DerivativeOrder, typename LHS, typename RHS>
struct sub_expr : public abstract_binary_expression<RealType,
                                                    DerivativeOrder,
                                                    LHS,
                                                    RHS,
                                                    sub_expr<RealType, DerivativeOrder, LHS, RHS>>
{
    /* @brief addition
   * rvar-rvar
   * */
    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;
    // Explicitly define constructor to forward to base class
    explicit sub_expr(const expression<RealType, DerivativeOrder, LHS> &left_hand_expr,
                      const expression<RealType, DerivativeOrder, RHS> &right_hand_expr)
        : abstract_binary_expression<RealType,
                                     DerivativeOrder,
                                     LHS,
                                     RHS,
                                     sub_expr<RealType, DerivativeOrder, LHS, RHS>>(left_hand_expr,
                                                                                    right_hand_expr)
    {}

    inner_t              evaluate() const { return this->lhs.evaluate() - this->rhs.evaluate(); }
    static const inner_t left_derivative(const inner_t & /*l*/,
                                         const inner_t & /*r*/,
                                         const inner_t & /*v*/)
    {
        return inner_t(static_cast<RealType>(1.0));
    }
    static const inner_t right_derivative(const inner_t & /*l*/,
                                          const inner_t & /*r*/,
                                          const inner_t & /*v*/)
    {
        return inner_t(static_cast<RealType>(-1.0));
    }
};

/****************************************************************************************************************/
template<typename RealType, size_t DerivativeOrder, typename LHS, typename RHS>
struct div_expr : public abstract_binary_expression<RealType,
                                                    DerivativeOrder,
                                                    LHS,
                                                    RHS,
                                                    div_expr<RealType, DerivativeOrder, LHS, RHS>>
{
    /* @brief multiplication
   * rvar / rvar
   * */
    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;
    // Explicitly define constructor to forward to base class
    explicit div_expr(const expression<RealType, DerivativeOrder, LHS> &left_hand_expr,
                      const expression<RealType, DerivativeOrder, RHS> &right_hand_expr)
        : abstract_binary_expression<RealType,
                                     DerivativeOrder,
                                     LHS,
                                     RHS,
                                     div_expr<RealType, DerivativeOrder, LHS, RHS>>(left_hand_expr,
                                                                                    right_hand_expr)
    {}

    inner_t              evaluate() const { return this->lhs.evaluate() / this->rhs.evaluate(); };
    static const inner_t left_derivative(const inner_t & /*l*/,
                                         const inner_t &r,
                                         const inner_t & /*v*/)
    {
        return static_cast<RealType>(1.0) / r;
    };
    static const inner_t right_derivative(const inner_t &l, const inner_t &r, const inner_t & /*v*/)
    {
        return -l / (r * r);
    };
};
template<typename RealType, size_t DerivativeOrder, typename ARG>
struct div_by_const_expr
    : public abstract_unary_expression<RealType,
                                       DerivativeOrder,
                                       ARG,
                                       div_by_const_expr<RealType, DerivativeOrder, ARG>>
{
    /* @brief
   * rvar/float
   * */
    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;

    explicit div_by_const_expr(const expression<RealType, DerivativeOrder, ARG> &arg_expr,
                               const RealType                                    v)
        : abstract_unary_expression<RealType,
                                    DerivativeOrder,
                                    ARG,
                                    div_by_const_expr<RealType, DerivativeOrder, ARG>>(arg_expr,
                                                                                       v){};

    inner_t              evaluate() const { return this->arg.evaluate() / inner_t(this->constant); }
    static const inner_t derivative(const inner_t & /*argv*/,
                                    const inner_t & /*v*/,
                                    const RealType &constant)
    {
        return inner_t(1.0 / constant);
    }
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
struct const_div_by_expr
    : public abstract_unary_expression<RealType,
                                       DerivativeOrder,
                                       ARG,
                                       const_div_by_expr<RealType, DerivativeOrder, ARG>>
{
    /** @brief
    * float/rvar
    * */
    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;

    explicit const_div_by_expr(const expression<RealType, DerivativeOrder, ARG> &arg_expr,
                               const RealType                                    v)
        : abstract_unary_expression<RealType,
                                    DerivativeOrder,
                                    ARG,
                                    const_div_by_expr<RealType, DerivativeOrder, ARG>>(arg_expr,
                                                                                       v){};

    inner_t              evaluate() const { return inner_t(this->constant) / this->arg.evaluate(); }
    static const inner_t derivative(const inner_t &argv,
                                    const inner_t & /*v*/,
                                    const RealType &constant)
    {
        return -inner_t{constant} / (argv * argv);
    }
};
/****************************************************************************************************************/

} // namespace reverse_mode
} // namespace differentiation
} // namespace math
} // namespace boost

#endif
