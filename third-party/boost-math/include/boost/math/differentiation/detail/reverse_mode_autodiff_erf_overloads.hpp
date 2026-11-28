//           Copyright Maksym Zhelyeznyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)
#ifndef REVERSE_MODE_AUTODIFF_ERF_OVERLOADS_HPP
#define REVERSE_MODE_AUTODIFF_ERF_OVERLOADS_HPP

#include <boost/math/constants/constants.hpp>
#include <boost/math/differentiation/detail/reverse_mode_autodiff_expression_template_base.hpp>

#ifdef BOOST_MATH_REVERSE_MODE_ET_ON
#include <boost/math/differentiation/detail/reverse_mode_autodiff_stl_et.hpp>
#else
#include <boost/math/differentiation/detail/reverse_mode_autodiff_stl_no_et.hpp>
#endif

#include <boost/math/differentiation/detail/reverse_mode_autodiff_utilities.hpp>
#include <boost/math/special_functions/erf.hpp>

namespace boost {
namespace math {
namespace differentiation {
namespace reverse_mode {

template<typename RealType, size_t DerivativeOrder, typename ARG>
struct erf_expr;

template<typename RealType, size_t DerivativeOrder, typename ARG>
struct erfc_expr;

template<typename RealType, size_t DerivativeOrder, typename ARG>
struct erf_inv_expr;

template<typename RealType, size_t DerivativeOrder, typename ARG>
struct erfc_inv_expr;

template<typename RealType, size_t DerivativeOrder, typename ARG>
erf_expr<RealType, DerivativeOrder, ARG> erf(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return erf_expr<RealType, DerivativeOrder, ARG>(arg, 0.0);
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
erfc_expr<RealType, DerivativeOrder, ARG> erfc(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return erfc_expr<RealType, DerivativeOrder, ARG>(arg, 0.0);
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
erf_inv_expr<RealType, DerivativeOrder, ARG> erf_inv(
    const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return erf_inv_expr<RealType, DerivativeOrder, ARG>(arg, 0.0);
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
erfc_inv_expr<RealType, DerivativeOrder, ARG> erfc_inv(
    const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return erfc_inv_expr<RealType, DerivativeOrder, ARG>(arg, 0.0);
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
struct erf_expr : public abstract_unary_expression<RealType,
                                                   DerivativeOrder,
                                                   ARG,
                                                   erf_expr<RealType, DerivativeOrder, ARG>>
{
    /** @brief erf(x)
    *
    * d/dx erf(x) = 2*exp(x^2)/sqrt(pi)
    *
    * */
    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;

    explicit erf_expr(const expression<RealType, DerivativeOrder, ARG> &arg_expr, const RealType v)
        : abstract_unary_expression<RealType,
                                    DerivativeOrder,
                                    ARG,
                                    erf_expr<RealType, DerivativeOrder, ARG>>(arg_expr, v){};

    inner_t evaluate() const
    {
        return detail::if_functional_dispatch<(DerivativeOrder > 1)>(
            [this](auto &&x) { return reverse_mode::erf(std::forward<decltype(x)>(x)); },
            [this](auto &&x) { return boost::math::erf(std::forward<decltype(x)>(x)); },
            this->arg.evaluate());
    }
    static const inner_t derivative(const inner_t &argv,
                                    const inner_t & /*v*/,
                                    const RealType & /*constant*/)
    {
        BOOST_MATH_STD_USING
        return static_cast<RealType>(2.0) * exp(-argv * argv) / sqrt(constants::pi<RealType>());
    }
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
struct erfc_expr : public abstract_unary_expression<RealType,
                                                    DerivativeOrder,
                                                    ARG,
                                                    erfc_expr<RealType, DerivativeOrder, ARG>>
{
    /** @brief erfc(x)
    *
    * d/dx erf(x) = -2*exp(x^2)/sqrt(pi)
    *
    * */

    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;

    explicit erfc_expr(const expression<RealType, DerivativeOrder, ARG> &arg_expr, const RealType v)
        : abstract_unary_expression<RealType,
                                    DerivativeOrder,
                                    ARG,
                                    erfc_expr<RealType, DerivativeOrder, ARG>>(arg_expr, v){};

    inner_t evaluate() const
    {
        return detail::if_functional_dispatch<((DerivativeOrder > 1))>(
            [this](auto &&x) { return reverse_mode::erfc(std::forward<decltype(x)>(x)); },
            [this](auto &&x) { return boost::math::erfc(std::forward<decltype(x)>(x)); },
            this->arg.evaluate());
    }
    static const inner_t derivative(const inner_t &argv,
                                    const inner_t & /*v*/,
                                    const RealType & /*constant*/)
    {
        BOOST_MATH_STD_USING
        return static_cast<RealType>(-2.0) * exp(-argv * argv) / sqrt(constants::pi<RealType>());
    }
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
struct erf_inv_expr : public abstract_unary_expression<RealType,
                                                       DerivativeOrder,
                                                       ARG,
                                                       erf_inv_expr<RealType, DerivativeOrder, ARG>>
{
    /** @brief erf(x)
    *
    * d/dx erf(x) = 2*exp(x^2)/sqrt(pi)
    *
    * */

    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;

    explicit erf_inv_expr(const expression<RealType, DerivativeOrder, ARG> &arg_expr,
                          const RealType                                    v)
        : abstract_unary_expression<RealType,
                                    DerivativeOrder,
                                    ARG,
                                    erf_inv_expr<RealType, DerivativeOrder, ARG>>(arg_expr, v){};

    inner_t evaluate() const
    {
        return detail::if_functional_dispatch<((DerivativeOrder > 1))>(
            [this](auto &&x) { return reverse_mode::erf_inv(std::forward<decltype(x)>(x)); },
            [this](auto &&x) { return boost::math::erf_inv(std::forward<decltype(x)>(x)); },
            this->arg.evaluate());
    }
    static const inner_t derivative(const inner_t &argv,
                                    const inner_t & /*v*/,
                                    const RealType & /*constant*/)
    {
        BOOST_MATH_STD_USING
        return detail::if_functional_dispatch<((DerivativeOrder > 1))>(
            [](auto &&x) {
                return static_cast<RealType>(0.5) * sqrt(constants::pi<RealType>())
                       * reverse_mode::exp(
                           reverse_mode::pow(reverse_mode::erf_inv(x), static_cast<RealType>(2.0)));
            },
            [](auto &&x) {
                return static_cast<RealType>(0.5) * sqrt(constants::pi<RealType>())
                       * exp(pow(boost::math::erf_inv(x), static_cast<RealType>(2.0)));
            },
            argv);
    }
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
struct erfc_inv_expr
    : public abstract_unary_expression<RealType,
                                       DerivativeOrder,
                                       ARG,
                                       erfc_inv_expr<RealType, DerivativeOrder, ARG>>
{
    /** @brief erfc(x)
    *
    * d/dx erf(x) = -2*exp(x^2)/sqrt(pi)
    *
    * */

    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;

    explicit erfc_inv_expr(const expression<RealType, DerivativeOrder, ARG> &arg_expr,
                           const RealType                                    v)
        : abstract_unary_expression<RealType,
                                    DerivativeOrder,
                                    ARG,
                                    erfc_inv_expr<RealType, DerivativeOrder, ARG>>(arg_expr, v){};

    inner_t evaluate() const
    {
        return detail::if_functional_dispatch<((DerivativeOrder > 1))>(
            [this](auto &&x) { return reverse_mode::erfc_inv(std::forward<decltype(x)>(x)); },
            [this](auto &&x) { return boost::math::erfc_inv(std::forward<decltype(x)>(x)); },
            this->arg.evaluate());
    }
    static const inner_t derivative(const inner_t &argv,
                                    const inner_t & /*v*/,
                                    const RealType & /*constant*/)
    {
        BOOST_MATH_STD_USING
        return detail::if_functional_dispatch<((DerivativeOrder > 1))>(
            [](auto &&x) {
                return static_cast<RealType>(-0.5) * sqrt(constants::pi<RealType>())
                       * reverse_mode::exp(reverse_mode::pow(reverse_mode::erfc_inv(x),
                                                             static_cast<RealType>(2.0)));
            },
            [](auto &&x) {
                return static_cast<RealType>(-0.5) * sqrt(constants::pi<RealType>())
                       * exp(pow(boost::math::erfc_inv(x), static_cast<RealType>(2.0)));
            },
            argv);
    }
};

} // namespace reverse_mode
} // namespace differentiation
} // namespace math
} // namespace boost

#endif // REVERSE_MODE_AUTODIFF_ERF_OVERLOADS_HPP
