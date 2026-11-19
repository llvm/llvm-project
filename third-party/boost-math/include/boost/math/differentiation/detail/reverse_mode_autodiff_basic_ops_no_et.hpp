//           Copyright Maksym Zhelyeznyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)
#ifndef REVERSE_MODE_AUTODIFF_BASIC_OPS_NO_ET_HPP
#define REVERSE_MODE_AUTODIFF_BASIC_OPS_NO_ET_HPP
#include <boost/math/differentiation/detail/reverse_mode_autodiff_basic_operator_expressions.hpp>
namespace boost {
namespace math {
namespace differentiation {
namespace reverse_mode {

template<typename RealType, size_t DerivativeOrder, typename LHS, typename RHS>
rvar<RealType, DerivativeOrder> operator*(const expression<RealType, DerivativeOrder, LHS> &lhs,
                                          const expression<RealType, DerivativeOrder, RHS> &rhs)
{
    return mult_expr<RealType, DerivativeOrder, LHS, RHS>(lhs, rhs);
}

/** @brief type promotion is handled by casting the numeric type to
 *  the type inside expression. This is to avoid converting the
 *  entire tape in case you have something like double * rvar<float>
 *  */
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         typename ARG,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
rvar<RealType1, DerivativeOrder> operator*(const expression<RealType1, DerivativeOrder, ARG> &arg,
                                           const RealType2                                   &v)
{
    return mult_const_expr<RealType1, DerivativeOrder, ARG>(arg, static_cast<RealType1>(v));
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         typename ARG,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
rvar<RealType1, DerivativeOrder> operator*(const RealType2                                   &v,
                                           const expression<RealType1, DerivativeOrder, ARG> &arg)
{
    return mult_const_expr<RealType1, DerivativeOrder, ARG>(arg, static_cast<RealType1>(v));
}
/****************************************************************************************************************/
/* + */
template<typename RealType, size_t DerivativeOrder, typename LHS, typename RHS>
rvar<RealType, DerivativeOrder> operator+(const expression<RealType, DerivativeOrder, LHS> &lhs,
                                          const expression<RealType, DerivativeOrder, RHS> &rhs)
{
    return add_expr<RealType, DerivativeOrder, LHS, RHS>(lhs, rhs);
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         typename ARG,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
rvar<RealType1, DerivativeOrder> operator+(const expression<RealType1, DerivativeOrder, ARG> &arg,
                                           const RealType2                                   &v)
{
    return add_const_expr<RealType1, DerivativeOrder, ARG>(arg, static_cast<RealType1>(v));
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         typename ARG,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
rvar<RealType1, DerivativeOrder> operator+(const RealType2                                   &v,
                                           const expression<RealType1, DerivativeOrder, ARG> &arg)
{
    return add_const_expr<RealType1, DerivativeOrder, ARG>(arg, static_cast<RealType1>(v));
}
/****************************************************************************************************************/
/* - overload */
/** @brief
 *  negation (-1.0*rvar) */
template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> operator-(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return mult_const_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(-1.0));
}

/** @brief
 *  subtraction rvar-rvar */
template<typename RealType, size_t DerivativeOrder, typename LHS, typename RHS>
rvar<RealType, DerivativeOrder> operator-(const expression<RealType, DerivativeOrder, LHS> &lhs,
                                          const expression<RealType, DerivativeOrder, RHS> &rhs)
{
    return sub_expr<RealType, DerivativeOrder, LHS, RHS>(lhs, rhs);
}

/** @brief
 *  subtraction float - rvar */
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         typename ARG,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
rvar<RealType1, DerivativeOrder> operator-(const expression<RealType1, DerivativeOrder, ARG> &arg,
                                           const RealType2                                   &v)
{
    /* rvar - float = rvar + (-float) */
    return add_const_expr<RealType1, DerivativeOrder, ARG>(arg, -static_cast<RealType1>(v));
}

/** @brief
 *   subtraction float - rvar
 *  @return add_expr<neg_expr<ARG>>
 */
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         typename ARG,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
rvar<RealType1, DerivativeOrder> operator-(const RealType2                                   &v,
                                           const expression<RealType1, DerivativeOrder, ARG> &arg)
{
    auto neg = -arg;
    return neg + static_cast<RealType1>(v);
}
/****************************************************************************************************************/
/* / */
template<typename RealType, size_t DerivativeOrder, typename LHS, typename RHS>
rvar<RealType, DerivativeOrder> operator/(const expression<RealType, DerivativeOrder, LHS> &lhs,
                                          const expression<RealType, DerivativeOrder, RHS> &rhs)
{
    return div_expr<RealType, DerivativeOrder, LHS, RHS>(lhs, rhs);
}

template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         typename ARG,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
rvar<RealType1, DerivativeOrder> operator/(const RealType2                                   &v,
                                           const expression<RealType1, DerivativeOrder, ARG> &arg)
{
    return const_div_by_expr<RealType1, DerivativeOrder, ARG>(arg, static_cast<RealType1>(v));
}

template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         typename ARG,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
rvar<RealType1, DerivativeOrder> operator/(const expression<RealType1, DerivativeOrder, ARG> &arg,
                                           const RealType2                                   &v)
{
    return div_by_const_expr<RealType1, DerivativeOrder, ARG>(arg, static_cast<RealType1>(v));
}
} // namespace reverse_mode
} // namespace differentiation
} // namespace math
} // namespace boost
#endif
