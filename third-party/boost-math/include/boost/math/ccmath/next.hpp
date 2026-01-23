//  (C) Copyright John Maddock 2008 - 2022.
//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_NEXT_HPP
#define BOOST_MATH_CCMATH_NEXT_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/next.hpp> can only be used in C++17 and later."
#endif

#include <stdexcept>
#include <cfloat>
#include <cstdint>
#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/tools/assert.hpp>
#include <boost/math/tools/config.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/tools/traits.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/ccmath/ilogb.hpp>
#include <boost/math/ccmath/ldexp.hpp>
#include <boost/math/ccmath/scalbln.hpp>
#include <boost/math/ccmath/round.hpp>
#include <boost/math/ccmath/fabs.hpp>
#include <boost/math/ccmath/fpclassify.hpp>
#include <boost/math/ccmath/isfinite.hpp>
#include <boost/math/ccmath/fmod.hpp>

namespace boost::math::ccmath {

namespace detail {

// Forward Declarations
template <typename T, typename result_type = tools::promote_args_t<T>>
constexpr result_type float_prior(const T& val);

template <typename T, typename result_type = tools::promote_args_t<T>>
constexpr result_type float_next(const T& val);

template <typename T>
struct has_hidden_guard_digits;
template <>
struct has_hidden_guard_digits<float> : public std::false_type {};
template <>
struct has_hidden_guard_digits<double> : public std::false_type {};
template <>
struct has_hidden_guard_digits<long double> : public std::false_type {};
#ifdef BOOST_HAS_FLOAT128
template <>
struct has_hidden_guard_digits<__float128> : public std::false_type {};
#endif

template <typename T, bool b>
struct has_hidden_guard_digits_10 : public std::false_type {};
template <typename T>
struct has_hidden_guard_digits_10<T, true> : public std::integral_constant<bool, (std::numeric_limits<T>::digits10 != std::numeric_limits<T>::max_digits10)> {};

template <typename T>
struct has_hidden_guard_digits 
    : public has_hidden_guard_digits_10<T, 
    std::numeric_limits<T>::is_specialized
    && (std::numeric_limits<T>::radix == 10) >
{};

template <typename T>
constexpr T normalize_value(const T& val, const std::false_type&) { return val; }
template <typename T>
constexpr T normalize_value(const T& val, const std::true_type&) 
{
    static_assert(std::numeric_limits<T>::is_specialized, "Type T must be specialized.");
    static_assert(std::numeric_limits<T>::radix != 2, "Type T must be specialized.");

    std::intmax_t shift = static_cast<std::intmax_t>(std::numeric_limits<T>::digits) - static_cast<std::intmax_t>(boost::math::ccmath::ilogb(val)) - 1;
    T result = boost::math::ccmath::scalbn(val, shift);
    result = boost::math::ccmath::round(result);
    return boost::math::ccmath::scalbn(result, -shift); 
}

template <typename T>
constexpr T get_smallest_value(const std::true_type&)
{
    //
    // numeric_limits lies about denorms being present - particularly
    // when this can be turned on or off at runtime, as is the case
    // when using the SSE2 registers in DAZ or FTZ mode.
    //
    constexpr T m = std::numeric_limits<T>::denorm_min();
    return ((tools::min_value<T>() / 2) == 0) ? tools::min_value<T>() : m;
}

template <typename T>
constexpr T get_smallest_value(const std::false_type&)
{
    return tools::min_value<T>();
}

template <typename T>
constexpr T get_smallest_value()
{
    return get_smallest_value<T>(std::integral_constant<bool, std::numeric_limits<T>::is_specialized>());
}

template <typename T>
constexpr T calc_min_shifted(const std::true_type&)
{
   return boost::math::ccmath::ldexp(tools::min_value<T>(), tools::digits<T>() + 1);
}

template <typename T>
constexpr T calc_min_shifted(const std::false_type&)
{
   static_assert(std::numeric_limits<T>::is_specialized, "Type T must be specialized.");
   static_assert(std::numeric_limits<T>::radix != 2, "Type T must be specialized.");

   return boost::math::ccmath::scalbn(tools::min_value<T>(), std::numeric_limits<T>::digits + 1);
}

template <typename T>
constexpr T get_min_shift_value()
{
   const T val = calc_min_shifted<T>(std::integral_constant<bool, !std::numeric_limits<T>::is_specialized || std::numeric_limits<T>::radix == 2>());
   return val;
}

template <typename T, bool b = boost::math::tools::detail::has_backend_type_v<T>>
struct exponent_type
{
    using type = int;
};

template <typename T>
struct exponent_type<T, true>
{
    using type = typename T::backend_type::exponent_type;
};

template <typename T, bool b = boost::math::tools::detail::has_backend_type_v<T>>
using exponent_type_t = typename exponent_type<T>::type;

template <typename T>
constexpr T float_next_imp(const T& val, const std::true_type&)
{
    using exponent_type = exponent_type_t<T>;
    
    exponent_type expon {};

    int fpclass = boost::math::ccmath::fpclassify(val);

    if (fpclass == FP_NAN)
    {
        return val;
    }
    else if (fpclass == FP_INFINITE)
    {
        return val;
    }
    else if (val <= -tools::max_value<T>())
    {
        return val;
    }

    if (val == 0)
    {
        return detail::get_smallest_value<T>();
    }

    if ((fpclass != FP_SUBNORMAL) && (fpclass != FP_ZERO) 
        && (boost::math::ccmath::fabs(val) < detail::get_min_shift_value<T>()) 
        && (val != -tools::min_value<T>()))
    {
        //
        // Special case: if the value of the least significant bit is a denorm, and the result
        // would not be a denorm, then shift the input, increment, and shift back.
        // This avoids issues with the Intel SSE2 registers when the FTZ or DAZ flags are set.
        //
        return boost::math::ccmath::ldexp(boost::math::ccmath::detail::float_next(static_cast<T>(boost::math::ccmath::ldexp(val, 2 * tools::digits<T>()))), -2 * tools::digits<T>());
    }

    if (-0.5f == boost::math::ccmath::frexp(val, &expon))
    {
        --expon; // reduce exponent when val is a power of two, and negative.
    }
    T diff = boost::math::ccmath::ldexp(static_cast<T>(1), expon - tools::digits<T>());
    if(diff == 0)
    {
        diff = detail::get_smallest_value<T>();
    }
    return val + diff;
}

//
// Special version for some base other than 2:
//
template <typename T>
constexpr T float_next_imp(const T& val, const std::false_type&)
{
    using exponent_type = exponent_type_t<T>;

    static_assert(std::numeric_limits<T>::is_specialized, "Type T must be specialized.");
    static_assert(std::numeric_limits<T>::radix != 2, "Type T must be specialized.");

    exponent_type expon {};

    int fpclass = boost::math::ccmath::fpclassify(val);

    if (fpclass == FP_NAN)
    {
        return val;
    }
    else if (fpclass == FP_INFINITE)
    {
        return val;
    }
    else if (val <= -tools::max_value<T>())
    {
        return val;
    }

    if (val == 0)
    {
        return detail::get_smallest_value<T>();
    }

    if ((fpclass != FP_SUBNORMAL) && (fpclass != FP_ZERO) 
        && (boost::math::ccmath::fabs(val) < detail::get_min_shift_value<T>()) 
        && (val != -tools::min_value<T>()))
    {
        //
        // Special case: if the value of the least significant bit is a denorm, and the result
        // would not be a denorm, then shift the input, increment, and shift back.
        // This avoids issues with the Intel SSE2 registers when the FTZ or DAZ flags are set.
        //
        return boost::math::ccmath::scalbn(boost::math::ccmath::detail::float_next(static_cast<T>(boost::math::ccmath::scalbn(val, 2 * std::numeric_limits<T>::digits))), -2 * std::numeric_limits<T>::digits);
    }

    expon = 1 + boost::math::ccmath::ilogb(val);
    if(-1 == boost::math::ccmath::scalbn(val, -expon) * std::numeric_limits<T>::radix)
    {
        --expon; // reduce exponent when val is a power of base, and negative.
    }

    T diff = boost::math::ccmath::scalbn(static_cast<T>(1), expon - std::numeric_limits<T>::digits);
    if(diff == 0)
    {
        diff = detail::get_smallest_value<T>();
    }

    return val + diff;
}

template <typename T, typename result_type>
constexpr result_type float_next(const T& val)
{
    return detail::float_next_imp(detail::normalize_value(static_cast<result_type>(val), typename detail::has_hidden_guard_digits<result_type>::type()), std::integral_constant<bool, !std::numeric_limits<result_type>::is_specialized || (std::numeric_limits<result_type>::radix == 2)>());
}

template <typename T>
constexpr T float_prior_imp(const T& val, const std::true_type&)
{
    using exponent_type = exponent_type_t<T>;

    exponent_type expon {};

    int fpclass = boost::math::ccmath::fpclassify(val);

    if (fpclass == FP_NAN)
    {
        return val;
    }
    else if (fpclass == FP_INFINITE)
    {
        return val;
    }
    else if (val <= -tools::max_value<T>())
    {
        return val;
    }

    if (val == 0)
    {
        return -detail::get_smallest_value<T>();
    }

    if ((fpclass != FP_SUBNORMAL) && (fpclass != FP_ZERO) 
        && (boost::math::ccmath::fabs(val) < detail::get_min_shift_value<T>()) 
        && (val != tools::min_value<T>()))
    {
        //
        // Special case: if the value of the least significant bit is a denorm, and the result
        // would not be a denorm, then shift the input, increment, and shift back.
        // This avoids issues with the Intel SSE2 registers when the FTZ or DAZ flags are set.
        //
        return boost::math::ccmath::ldexp(boost::math::ccmath::detail::float_prior(static_cast<T>(boost::math::ccmath::ldexp(val, 2 * tools::digits<T>()))), -2 * tools::digits<T>());
    }

    if(T remain = boost::math::ccmath::frexp(val, &expon); remain == 0.5f)
    {
        --expon; // when val is a power of two we must reduce the exponent
    }

    T diff = boost::math::ccmath::ldexp(static_cast<T>(1), expon - tools::digits<T>());
    if(diff == 0)
    {
        diff = detail::get_smallest_value<T>();
    }

    return val - diff;
}

//
// Special version for bases other than 2:
//
template <typename T>
constexpr T float_prior_imp(const T& val, const std::false_type&)
{
    using exponent_type = exponent_type_t<T>;

    static_assert(std::numeric_limits<T>::is_specialized, "Type T must be specialized.");
    static_assert(std::numeric_limits<T>::radix != 2, "Type T must be specialized.");

    exponent_type expon {};

    int fpclass = boost::math::ccmath::fpclassify(val);

    if (fpclass == FP_NAN)
    {
        return val;
    }
    else if (fpclass == FP_INFINITE)
    {
        return val;
    }
    else if (val <= -tools::max_value<T>())
    {
        return val;
    }

    if (val == 0)
    {
        return -detail::get_smallest_value<T>();
    }

    if ((fpclass != FP_SUBNORMAL) && (fpclass != FP_ZERO) 
        && (boost::math::ccmath::fabs(val) < detail::get_min_shift_value<T>()) 
        && (val != tools::min_value<T>()))
    {
        //
        // Special case: if the value of the least significant bit is a denorm, and the result
        // would not be a denorm, then shift the input, increment, and shift back.
        // This avoids issues with the Intel SSE2 registers when the FTZ or DAZ flags are set.
        //
        return boost::math::ccmath::scalbn(boost::math::ccmath::detail::float_prior(static_cast<T>(boost::math::ccmath::scalbn(val, 2 * std::numeric_limits<T>::digits))), -2 * std::numeric_limits<T>::digits);
    }

    expon = 1 + boost::math::ccmath::ilogb(val);
    
    if (T remain = boost::math::ccmath::scalbn(val, -expon); remain * std::numeric_limits<T>::radix == 1)
    {
        --expon; // when val is a power of two we must reduce the exponent
    }

    T diff = boost::math::ccmath::scalbn(static_cast<T>(1), expon - std::numeric_limits<T>::digits);
    if (diff == 0)
    {
        diff = detail::get_smallest_value<T>();
    }
    return val - diff;
} // float_prior_imp

template <typename T, typename result_type>
constexpr result_type float_prior(const T& val)
{
    return detail::float_prior_imp(detail::normalize_value(static_cast<result_type>(val), typename detail::has_hidden_guard_digits<result_type>::type()), std::integral_constant<bool, !std::numeric_limits<result_type>::is_specialized || (std::numeric_limits<result_type>::radix == 2)>());
}

} // namespace detail

template <typename T, typename U, typename result_type = tools::promote_args_t<T, U>>
constexpr result_type nextafter(const T& val, const U& direction)
{
    if (BOOST_MATH_IS_CONSTANT_EVALUATED(val))
    {
        if (boost::math::ccmath::isnan(val))
        {
            return val;
        }
        else if (boost::math::ccmath::isnan(direction))
        {
            return direction;
        }
        else if (val < direction)
        {
            return boost::math::ccmath::detail::float_next(val);
        }
        else if (val == direction)
        {
            // IEC 60559 recommends that from is returned whenever from == to. These functions return to instead, 
            // which makes the behavior around zero consistent: std::nextafter(-0.0, +0.0) returns +0.0 and 
            // std::nextafter(+0.0, -0.0) returns -0.0.
            return direction;
        }

        return boost::math::ccmath::detail::float_prior(val);
    }
    else
    {
        using std::nextafter;
        return nextafter(static_cast<result_type>(val), static_cast<result_type>(direction));
    }
}

constexpr float nextafterf(float val, float direction)
{
    return boost::math::ccmath::nextafter(val, direction);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS

constexpr long double nextafterl(long double val, long double direction)
{
    return boost::math::ccmath::nextafter(val, direction);
}

template <typename T, typename result_type = tools::promote_args_t<T, long double>, typename return_type = std::conditional_t<std::is_integral_v<T>, double, T>>
constexpr return_type nexttoward(T val, long double direction)
{
    if (BOOST_MATH_IS_CONSTANT_EVALUATED(val))
    {
        return static_cast<return_type>(boost::math::ccmath::nextafter(static_cast<result_type>(val), direction));
    }
    else
    {
        using std::nexttoward;
        return nexttoward(val, direction);
    }
}

constexpr float nexttowardf(float val, long double direction)
{
    return boost::math::ccmath::nexttoward(val, direction);
}

constexpr long double nexttowardl(long double val, long double direction)
{
    return boost::math::ccmath::nexttoward(val, direction);
}

#endif

} // Namespaces

#endif // BOOST_MATH_SPECIAL_NEXT_HPP
