//           Copyright Maksym Zhelyeznyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#ifndef REVERSE_MODE_AUTODIFF_COMPARISON_OPERATOR_OVERLOADS_HPP
#define REVERSE_MODE_AUTODIFF_COMPARISON_OPERATOR_OVERLOADS_HPP
#include <boost/math/differentiation/detail/reverse_mode_autodiff_expression_template_base.hpp>
namespace boost {
namespace math {
namespace differentiation {
namespace reverse_mode {
template<typename RealType, size_t DerivativeOrder1, size_t DerivativeOrder2, class LhsExpr, class RhsExpr>
bool operator==(const expression<RealType, DerivativeOrder1, LhsExpr> &lhs,
                const expression<RealType, DerivativeOrder2, RhsExpr> &rhs)
{
    return lhs.evaluate() == rhs.evaluate();
}

template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         class ArgExpr,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
bool operator==(const expression<RealType1, DerivativeOrder, ArgExpr> &lhs, const RealType2 &rhs)
{
    return lhs.evaluate() == static_cast<RealType1>(rhs);
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         class ArgExpr,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
bool operator==(const RealType2 &lhs, const expression<RealType1, DerivativeOrder, ArgExpr> &rhs)
{
    return lhs == rhs.evaluate();
}

template<typename RealType, size_t DerivativeOrder1, size_t DerivativeOrder2, class LhsExpr, class RhsExpr>
bool operator!=(const expression<RealType, DerivativeOrder1, LhsExpr> &lhs,
                const expression<RealType, DerivativeOrder2, RhsExpr> &rhs)
{
    return lhs.evaluate() != rhs.evaluate();
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         class ArgExpr,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
bool operator!=(const expression<RealType1, DerivativeOrder, ArgExpr> &lhs, const RealType2 &rhs)
{
    return lhs.evaluate() != rhs;
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         class ArgExpr,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
bool operator!=(const RealType2 &lhs, const expression<RealType1, DerivativeOrder, ArgExpr> &rhs)
{
    return lhs != rhs.evaluate();
}

template<typename RealType, size_t DerivativeOrder1, size_t DerivativeOrder2, class LhsExpr, class RhsExpr>
bool operator<(const expression<RealType, DerivativeOrder1, LhsExpr> &lhs,
               const expression<RealType, DerivativeOrder2, RhsExpr> &rhs)
{
    return lhs.evaluate() < rhs.evaluate();
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         class ArgExpr,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
bool operator<(const expression<RealType1, DerivativeOrder, ArgExpr> &lhs, const RealType2 &rhs)
{
    return lhs.evaluate() < rhs;
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         class ArgExpr,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
bool operator<(const RealType2 &lhs, const expression<RealType1, DerivativeOrder, ArgExpr> &rhs)
{
    return lhs < rhs.evaluate();
}

template<typename RealType, size_t DerivativeOrder1, size_t DerivativeOrder2, class LhsExpr, class RhsExpr>
bool operator>(const expression<RealType, DerivativeOrder1, LhsExpr> &lhs,
               const expression<RealType, DerivativeOrder2, RhsExpr> &rhs)
{
    return lhs.evaluate() > rhs.evaluate();
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         class ArgExpr,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
bool operator>(const expression<RealType1, DerivativeOrder, ArgExpr> &lhs, const RealType2 &rhs)
{
    return lhs.evaluate() > rhs;
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         class ArgExpr,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
bool operator>(const RealType2 &lhs, const expression<RealType1, DerivativeOrder, ArgExpr> &rhs)
{
    return lhs > rhs.evaluate();
}

template<typename RealType, size_t DerivativeOrder1, size_t DerivativeOrder2, class LhsExpr, class RhsExpr>
bool operator<=(const expression<RealType, DerivativeOrder1, LhsExpr> &lhs,
                const expression<RealType, DerivativeOrder2, RhsExpr> &rhs)
{
    return lhs.evaluate() <= rhs.evaluate();
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         class ArgExpr,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
bool operator<=(const expression<RealType1, DerivativeOrder, ArgExpr> &lhs, const RealType2 &rhs)
{
    return lhs.evaluate() <= rhs;
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         class ArgExpr,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
bool operator<=(const RealType2 &lhs, const expression<RealType1, DerivativeOrder, ArgExpr> &rhs)
{
    return lhs <= rhs.evaluate();
}

template<typename RealType, size_t DerivativeOrder1, size_t DerivativeOrder2, class LhsExpr, class RhsExpr>
bool operator>=(const expression<RealType, DerivativeOrder1, LhsExpr> &lhs,
                const expression<RealType, DerivativeOrder2, RhsExpr> &rhs)
{
    return lhs.evaluate() >= rhs.evaluate();
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         class ArgExpr,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
bool operator>=(const expression<RealType1, DerivativeOrder, ArgExpr> &lhs, const RealType2 &rhs)
{
    return lhs.evaluate() >= rhs;
}
template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         class ArgExpr,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
bool operator>=(const RealType2 &lhs, const expression<RealType1, DerivativeOrder, ArgExpr> &rhs)
{
    return lhs >= rhs.evaluate();
}
} // namespace reverse_mode
} // namespace differentiation
} // namespace math
} // namespace boost
#endif // REVERSE_MODE_AUTODIFF_COMPARISON_OPERATOR_OVERLOADS_HPP
