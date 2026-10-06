//           Copyright Maksym Zhelyeznyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#ifndef REVERSE_MODE_AUTODIFF_EXPRESSION_TEMPLATE_BASE_HPP
#define REVERSE_MODE_AUTODIFF_EXPRESSION_TEMPLATE_BASE_HPP
#include <cstddef>
#include <type_traits>
namespace boost {
namespace math {
namespace differentiation {
namespace reverse_mode {

/* forward declarations for utitlity functions */
struct expression_base
{};

template<typename RealType, size_t DerivativeOrder, class DerivedExpression>
struct expression;

template<typename RealType, size_t DerivativeOrder>
class rvar;

template<typename RealType,
         size_t DerivativeOrder,
         typename LHS,
         typename RHS,
         typename ConcreteBinaryOperation>
struct abstract_binary_expression;

template<typename RealType, size_t DerivativeOrder, typename ARG, typename ConcreteUnaryOperation>

struct abstract_unary_expression;

template<typename RealType, size_t DerivativeOrder>
class gradient_node; // forward declaration for tape

namespace detail {
template<typename...>
using void_t = void;
// Check if T has a 'value_type' alias
template<typename T, typename Enable = void>
struct has_value_type : std::false_type
{};
template<typename T>
struct has_value_type<T, void_t<typename T::value_type>> : std::true_type
{};
template<typename T, typename Enable = void>
struct has_binary_sub_types : std::false_type
{};
template<typename T>
struct has_binary_sub_types<T, void_t<typename T::lhs_type, typename T::rhs_type>> : std::true_type
{};
template<typename T, typename Enable = void>
struct has_unary_sub_type : std::false_type
{};
template<typename T>
struct has_unary_sub_type<T, void_t<typename T::arg_type>> : std::true_type
{};

template<typename T, size_t order, typename Enable = void>
struct count_rvar_impl
{
    static constexpr std::size_t value = 0;
};
template<typename RealType, size_t DerivativeOrder>
struct count_rvar_impl<rvar<RealType, DerivativeOrder>, DerivativeOrder>
{
    static constexpr std::size_t value = 1;
};

template<typename RealType, std::size_t DerivativeOrder>
struct count_rvar_impl<
    RealType,
    DerivativeOrder,
    std::enable_if_t<has_binary_sub_types<RealType>::value
                     && !std::is_same<RealType, rvar<typename RealType::value_type, DerivativeOrder>>::value
                     && !has_unary_sub_type<RealType>::value>>
{
    static constexpr std::size_t value
        = count_rvar_impl<typename RealType::lhs_type, DerivativeOrder>::value
          + count_rvar_impl<typename RealType::rhs_type, DerivativeOrder>::value;
};

template<typename RealType, size_t DerivativeOrder>
struct count_rvar_impl<
    RealType,
    DerivativeOrder,
    typename std::enable_if_t<
        has_unary_sub_type<RealType>::value
        && !std::is_same<RealType, rvar<typename RealType::value_type, DerivativeOrder>>::value
        && !has_binary_sub_types<RealType>::value>>
{
    static constexpr std::size_t value
        = count_rvar_impl<typename RealType::arg_type, DerivativeOrder>::value;
};
template<typename RealType, size_t DerivativeOrder>
constexpr std::size_t count_rvars = detail::count_rvar_impl<RealType, DerivativeOrder>::value;

template<typename T>
struct is_expression : std::is_base_of<expression_base, typename std::decay<T>::type>
{};

template<typename RealType, size_t N>
struct rvar_type_impl
{
    using type = rvar<RealType, N>;
};

template<typename RealType>
struct rvar_type_impl<RealType, 0>
{
    using type = RealType;
};

} // namespace detail

template<typename T, size_t N>
using rvar_t = typename detail::rvar_type_impl<T, N>::type;

template<typename RealType, size_t DerivativeOrder, class DerivedExpression>
struct expression : expression_base
{
    /* @brief
   * base expression class
   * */

    using value_type                = RealType;
    static constexpr size_t order_v = DerivativeOrder;
    using derived_type              = DerivedExpression;

    static constexpr size_t num_literals = 0;
    using inner_t                        = rvar_t<RealType, DerivativeOrder - 1>;
    inner_t evaluate() const { return static_cast<const DerivedExpression *>(this)->evaluate(); }

    template<size_t arg_index>
    void propagatex(gradient_node<RealType, DerivativeOrder> *node, inner_t adj) const
    {
        return static_cast<const DerivedExpression *>(this)->template propagatex<arg_index>(node,
                                                                                            adj);
    }
};

template<typename RealType,
         size_t DerivativeOrder,
         typename LHS,
         typename RHS,
         typename ConcreteBinaryOperation>
struct abstract_binary_expression
    : public expression<
          RealType,
          DerivativeOrder,
          abstract_binary_expression<RealType, DerivativeOrder, LHS, RHS, ConcreteBinaryOperation>>
{
    using lhs_type   = LHS;
    using rhs_type   = RHS;
    using value_type = RealType;
    using inner_t    = rvar_t<RealType, DerivativeOrder - 1>;
    const lhs_type lhs;
    const rhs_type rhs;

    explicit abstract_binary_expression(
        const expression<RealType, DerivativeOrder, LHS> &left_hand_expr,
        const expression<RealType, DerivativeOrder, RHS> &right_hand_expr)
        : lhs(static_cast<const LHS &>(left_hand_expr))
        , rhs(static_cast<const RHS &>(right_hand_expr)){};

    inner_t evaluate() const
    {
        return static_cast<const ConcreteBinaryOperation *>(this)->evaluate();
    };

    template<size_t arg_index>
    void propagatex(gradient_node<RealType, DerivativeOrder> *node, inner_t adj) const
    {
        const inner_t lv        = lhs.evaluate();
        const inner_t rv        = rhs.evaluate();
        const inner_t v         = evaluate();
        const inner_t partial_l = ConcreteBinaryOperation::left_derivative(lv, rv, v);
        const inner_t partial_r = ConcreteBinaryOperation::right_derivative(lv, rv, v);

        constexpr size_t num_lhs_args = detail::count_rvars<LHS, DerivativeOrder>;
        constexpr size_t num_rhs_args = detail::count_rvars<RHS, DerivativeOrder>;

        propagate_lhs<num_lhs_args, arg_index>(node, adj * partial_l);
        propagate_rhs<num_rhs_args, arg_index + num_lhs_args>(node, adj * partial_r);
    }

private:
    /* everything here just emulates c++17 if constexpr */

    template<std::size_t num_args,
             std::size_t arg_index_,
             typename std::enable_if<(num_args > 0), int>::type = 0>
    void propagate_lhs(gradient_node<RealType, DerivativeOrder> *node, inner_t adj) const
    {
        lhs.template propagatex<arg_index_>(node, adj);
    }

    template<std::size_t num_args,
             std::size_t arg_index_,
             typename std::enable_if<(num_args == 0), int>::type = 0>
    void propagate_lhs(gradient_node<RealType, DerivativeOrder> *, inner_t) const
    {}

    template<std::size_t num_args,
             std::size_t arg_index_,
             typename std::enable_if<(num_args > 0), int>::type = 0>
    void propagate_rhs(gradient_node<RealType, DerivativeOrder> *node, inner_t adj) const
    {
        rhs.template propagatex<arg_index_>(node, adj);
    }

    template<std::size_t num_args,
             std::size_t arg_index_,
             typename std::enable_if<(num_args == 0), int>::type = 0>
    void propagate_rhs(gradient_node<RealType, DerivativeOrder> *, inner_t) const
    {}
};
template<typename RealType, size_t DerivativeOrder, typename ARG, typename ConcreteUnaryOperation>

struct abstract_unary_expression
    : public expression<
          RealType,
          DerivativeOrder,
          abstract_unary_expression<RealType, DerivativeOrder, ARG, ConcreteUnaryOperation>>
{
    using arg_type   = ARG;
    using value_type = RealType;
    using inner_t    = rvar_t<RealType, DerivativeOrder - 1>;
    const arg_type arg;
    const RealType constant;
    explicit abstract_unary_expression(const expression<RealType, DerivativeOrder, ARG> &arg_expr,
                                       const RealType                                   &constant)
        : arg(static_cast<const ARG &>(arg_expr))
        , constant(constant){};
    inner_t evaluate() const
    {
        return static_cast<const ConcreteUnaryOperation *>(this)->evaluate();
    };

    template<size_t arg_index>
    void propagatex(gradient_node<RealType, DerivativeOrder> *node, inner_t adj) const
    {
        inner_t argv        = arg.evaluate();
        inner_t v           = evaluate();
        inner_t partial_arg = ConcreteUnaryOperation::derivative(argv, v, constant);

        arg.template propagatex<arg_index>(node, adj * partial_arg);
    }
};
} // namespace reverse_mode
} // namespace differentiation
} // namespace math
} // namespace boost

#endif // REVERSE_MODE_AUTODIFF_EXPRESSION_TEMPLATE_BASE_HPP
