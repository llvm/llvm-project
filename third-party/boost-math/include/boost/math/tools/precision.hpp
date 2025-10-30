//  Copyright John Maddock 2005-2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_PRECISION_INCLUDED
#define BOOST_MATH_TOOLS_PRECISION_INCLUDED

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/assert.hpp>
#include <boost/math/tools/type_traits.hpp>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/policies/policy.hpp>

#ifndef BOOST_MATH_HAS_NVRTC
#include <type_traits>
#include <limits>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cfloat> // LDBL_MANT_DIG
#endif

namespace boost{ namespace math
{
namespace tools
{
// If T is not specialized, the functions digits, max_value and min_value,
// all get synthesised automatically from std::numeric_limits.
// However, if numeric_limits is not specialised for type RealType,
// for example with NTL::RR type, then you will get a compiler error
// when code tries to use these functions, unless you explicitly specialise them.

// For example if the precision of RealType varies at runtime,
// then numeric_limits support may not be appropriate,
// see boost/math/tools/ntl.hpp  for examples like
// template <> NTL::RR max_value<NTL::RR> ...
// See  Conceptual Requirements for Real Number Types.

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr int digits(BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(T)) noexcept
{
   static_assert( ::boost::math::numeric_limits<T>::is_specialized, "Type T must be specialized");
   static_assert( ::boost::math::numeric_limits<T>::radix == 2 || ::boost::math::numeric_limits<T>::radix == 10, "Type T must have a radix of 2 or 10");

   return boost::math::numeric_limits<T>::radix == 2
      ? boost::math::numeric_limits<T>::digits
      : ((boost::math::numeric_limits<T>::digits + 1) * 1000L) / 301L;
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T max_value(BOOST_MATH_EXPLICIT_TEMPLATE_TYPE(T))  noexcept(boost::math::is_floating_point<T>::value)
{
   static_assert( ::boost::math::numeric_limits<T>::is_specialized, "Type T must be specialized");
   return (boost::math::numeric_limits<T>::max)();
} // Also used as a finite 'infinite' value for - and +infinity, for example:
// -max_value<double> = -1.79769e+308, max_value<double> = 1.79769e+308.

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T min_value(BOOST_MATH_EXPLICIT_TEMPLATE_TYPE(T)) noexcept(boost::math::is_floating_point<T>::value)
{
   static_assert( ::boost::math::numeric_limits<T>::is_specialized, "Type T must be specialized");

   return (boost::math::numeric_limits<T>::min)();
}

namespace detail{
//
// Logarithmic limits come next, note that although
// we can compute these from the log of the max value
// that is not in general thread safe (if we cache the value)
// so it's better to specialise these:
//
// For type float first:
//
template <class T>
BOOST_MATH_GPU_ENABLED constexpr T log_max_value(const boost::math::integral_constant<int, 128>& BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(T)) noexcept(boost::math::is_floating_point<T>::value)
{
   return 88.0f;
}

template <class T>
BOOST_MATH_GPU_ENABLED constexpr T log_min_value(const boost::math::integral_constant<int, 128>& BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(T)) noexcept(boost::math::is_floating_point<T>::value)
{
   return -87.0f;
}
//
// Now double:
//
template <class T>
BOOST_MATH_GPU_ENABLED constexpr T log_max_value(const boost::math::integral_constant<int, 1024>& BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(T)) noexcept(boost::math::is_floating_point<T>::value)
{
   return 709.0;
}

template <class T>
BOOST_MATH_GPU_ENABLED constexpr T log_min_value(const boost::math::integral_constant<int, 1024>& BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(T)) noexcept(boost::math::is_floating_point<T>::value)
{
   return -708.0;
}
//
// 80 and 128-bit long doubles:
//
template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T log_max_value(const boost::math::integral_constant<int, 16384>& BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(T)) noexcept(boost::math::is_floating_point<T>::value)
{
   return 11356.0L;
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T log_min_value(const boost::math::integral_constant<int, 16384>& BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(T)) noexcept(boost::math::is_floating_point<T>::value)
{
   return -11355.0L;
}

template <class T>
BOOST_MATH_GPU_ENABLED inline T log_max_value(const boost::math::integral_constant<int, 0>& BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(T))
{
   BOOST_MATH_STD_USING
#ifdef __SUNPRO_CC
   static const T m = boost::math::tools::max_value<T>();
   static const T val = log(m);
#else
   static const T val = log(boost::math::tools::max_value<T>());
#endif
   return val;
}

template <class T>
BOOST_MATH_GPU_ENABLED inline T log_min_value(const boost::math::integral_constant<int, 0>& BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(T))
{
   BOOST_MATH_STD_USING
#ifdef __SUNPRO_CC
   static const T m = boost::math::tools::min_value<T>();
   static const T val = log(m);
#else
   static const T val = log(boost::math::tools::min_value<T>());
#endif
   return val;
}

template <class T>
BOOST_MATH_GPU_ENABLED constexpr T epsilon(const boost::math::true_type& BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(T)) noexcept(boost::math::is_floating_point<T>::value)
{
   return boost::math::numeric_limits<T>::epsilon();
}

#if defined(__GNUC__) && ((LDBL_MANT_DIG == 106) || (__LDBL_MANT_DIG__ == 106))
template <>
BOOST_MATH_GPU_ENABLED inline constexpr long double epsilon<long double>(const boost::math::true_type& BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(long double)) noexcept(boost::math::is_floating_point<long double>::value)
{
   // numeric_limits on Darwin (and elsewhere) tells lies here:
   // the issue is that long double on a few platforms is
   // really a "double double" which has a non-contiguous
   // mantissa: 53 bits followed by an unspecified number of
   // zero bits, followed by 53 more bits.  Thus the apparent
   // precision of the type varies depending where it's been.
   // Set epsilon to the value that a 106 bit fixed mantissa
   // type would have, as that will give us sensible behaviour everywhere.
   //
   // This static assert fails for some unknown reason, so
   // disabled for now...
   // static_assert(std::numeric_limits<long double>::digits == 106);
   return 2.4651903288156618919116517665087e-32L;
}
#endif

template <class T>
BOOST_MATH_GPU_ENABLED inline T epsilon(const boost::math::false_type& BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(T))
{
   // Note: don't cache result as precision may vary at runtime:
   BOOST_MATH_STD_USING  // for ADL of std names
   return ldexp(static_cast<T>(1), 1-policies::digits<T, policies::policy<> >());
}

template <class T>
struct log_limit_traits
{
   typedef typename boost::math::conditional<
      (boost::math::numeric_limits<T>::radix == 2) &&
      (
         (     boost::math::numeric_limits<T>::max_exponent == 128
            || boost::math::numeric_limits<T>::max_exponent == 1024
            || boost::math::numeric_limits<T>::max_exponent == 16384
         )
         && (-boost::math::numeric_limits<T>::min_exponent10 + 1 == boost::math::numeric_limits<T>::max_exponent10)
      ),
      boost::math::integral_constant<int, (boost::math::numeric_limits<T>::max_exponent > (boost::math::numeric_limits<int>::max)() ? (boost::math::numeric_limits<int>::max)() : static_cast<int>(boost::math::numeric_limits<T>::max_exponent))>,
      boost::math::integral_constant<int, 0>
   >::type tag_type;
   static constexpr bool value = (tag_type::value != 0);
   static_assert(::boost::math::numeric_limits<T>::is_specialized || !value, "Type T must be specialized or equal to 0");
};

template <class T, bool b> struct log_limit_noexcept_traits_imp : public log_limit_traits<T> {};
template <class T> struct log_limit_noexcept_traits_imp<T, false> : public boost::math::integral_constant<bool, false> {};

template <class T>
struct log_limit_noexcept_traits : public log_limit_noexcept_traits_imp<T, boost::math::is_floating_point<T>::value> {};

} // namespace detail

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4309)
#endif

template <class T>
BOOST_MATH_GPU_ENABLED inline T log_max_value(BOOST_MATH_EXPLICIT_TEMPLATE_TYPE(T)) noexcept(detail::log_limit_noexcept_traits<T>::value)
{
#ifndef BOOST_MATH_HAS_NVRTC
   #ifndef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
      return detail::log_max_value<T>(typename detail::log_limit_traits<T>::tag_type());
   #else
      BOOST_MATH_ASSERT(::boost::math::numeric_limits<T>::is_specialized);
      BOOST_MATH_STD_USING
      static const T val = log((boost::math::numeric_limits<T>::max)());
      return val;
   #endif
#else
   return log((boost::math::numeric_limits<T>::max)());
#endif
}

template <class T>
BOOST_MATH_GPU_ENABLED inline T log_min_value(BOOST_MATH_EXPLICIT_TEMPLATE_TYPE(T)) noexcept(detail::log_limit_noexcept_traits<T>::value)
{
#ifndef BOOST_MATH_HAS_NVRTC
   #ifndef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
      return detail::log_min_value<T>(typename detail::log_limit_traits<T>::tag_type());
   #else
      BOOST_MATH_ASSERT(::boost::math::numeric_limits<T>::is_specialized);
      BOOST_MATH_STD_USING
      static const T val = log((boost::math::numeric_limits<T>::min)());
      return val;
   #endif
#else
   return log((boost::math::numeric_limits<T>::min)());
#endif
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

template <class T>
BOOST_MATH_GPU_ENABLED constexpr T epsilon(BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(T)) noexcept(boost::math::is_floating_point<T>::value)
{
   // NVRTC does not like this dispatching method so we just skip to where we want to go
#ifndef BOOST_MATH_HAS_NVRTC
   #ifndef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
      return detail::epsilon<T>(boost::math::integral_constant<bool, ::boost::math::numeric_limits<T>::is_specialized>());
   #else
      return ::boost::math::numeric_limits<T>::is_specialized ?
         detail::epsilon<T>(boost::math::true_type()) :
         detail::epsilon<T>(boost::math::false_type());
   #endif
#else
   return boost::math::numeric_limits<T>::epsilon();
#endif
}

namespace detail{

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T root_epsilon_imp(const boost::math::integral_constant<int, 24>&) noexcept(boost::math::is_floating_point<T>::value)
{
   return static_cast<T>(0.00034526698300124390839884978618400831996329879769945L);
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T root_epsilon_imp(const T*, const boost::math::integral_constant<int, 53>&) noexcept(boost::math::is_floating_point<T>::value)
{
   return static_cast<T>(0.1490116119384765625e-7L);
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T root_epsilon_imp(const T*, const boost::math::integral_constant<int, 64>&) noexcept(boost::math::is_floating_point<T>::value)
{
   return static_cast<T>(0.32927225399135962333569506281281311031656150598474e-9L);
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T root_epsilon_imp(const T*, const boost::math::integral_constant<int, 113>&) noexcept(boost::math::is_floating_point<T>::value)
{
   return static_cast<T>(0.1387778780781445675529539585113525390625e-16L);
}

template <class T, class Tag>
BOOST_MATH_GPU_ENABLED inline T root_epsilon_imp(const T*, const Tag&)
{
   BOOST_MATH_STD_USING
   BOOST_MATH_STATIC_LOCAL_VARIABLE const T r_eps = sqrt(tools::epsilon<T>());
   return r_eps;
}

template <class T>
BOOST_MATH_GPU_ENABLED inline T root_epsilon_imp(const T*, const boost::math::integral_constant<int, 0>&)
{
   BOOST_MATH_STD_USING
   return sqrt(tools::epsilon<T>());
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T cbrt_epsilon_imp(const boost::math::integral_constant<int, 24>&) noexcept(boost::math::is_floating_point<T>::value)
{
   return static_cast<T>(0.0049215666011518482998719164346805794944150447839903L);
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T cbrt_epsilon_imp(const T*, const boost::math::integral_constant<int, 53>&) noexcept(boost::math::is_floating_point<T>::value)
{
   return static_cast<T>(6.05545445239333906078989272793696693569753008995e-6L);
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T cbrt_epsilon_imp(const T*, const boost::math::integral_constant<int, 64>&) noexcept(boost::math::is_floating_point<T>::value)
{
   return static_cast<T>(4.76837158203125e-7L);
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T cbrt_epsilon_imp(const T*, const boost::math::integral_constant<int, 113>&) noexcept(boost::math::is_floating_point<T>::value)
{
   return static_cast<T>(5.7749313854154005630396773604745549542403508090496e-12L);
}

template <class T, class Tag>
BOOST_MATH_GPU_ENABLED inline T cbrt_epsilon_imp(const T*, const Tag&)
{
   BOOST_MATH_STD_USING;
   static const T cbrt_eps = pow(tools::epsilon<T>(), T(1) / 3);
   return cbrt_eps;
}

template <class T>
BOOST_MATH_GPU_ENABLED inline T cbrt_epsilon_imp(const T*, const boost::math::integral_constant<int, 0>&)
{
   BOOST_MATH_STD_USING;
   return pow(tools::epsilon<T>(), T(1) / 3);
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T forth_root_epsilon_imp(const T*, const boost::math::integral_constant<int, 24>&) noexcept(boost::math::is_floating_point<T>::value)
{
   return static_cast<T>(0.018581361171917516667460937040007436176452688944747L);
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T forth_root_epsilon_imp(const T*, const boost::math::integral_constant<int, 53>&) noexcept(boost::math::is_floating_point<T>::value)
{
   return static_cast<T>(0.0001220703125L);
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T forth_root_epsilon_imp(const T*, const boost::math::integral_constant<int, 64>&) noexcept(boost::math::is_floating_point<T>::value)
{
   return static_cast<T>(0.18145860519450699870567321328132261891067079047605e-4L);
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T forth_root_epsilon_imp(const T*, const boost::math::integral_constant<int, 113>&) noexcept(boost::math::is_floating_point<T>::value)
{
   return static_cast<T>(0.37252902984619140625e-8L);
}

template <class T, class Tag>
BOOST_MATH_GPU_ENABLED inline T forth_root_epsilon_imp(const T*, const Tag&)
{
   BOOST_MATH_STD_USING
   static const T r_eps = sqrt(sqrt(tools::epsilon<T>()));
   return r_eps;
}

template <class T>
BOOST_MATH_GPU_ENABLED inline T forth_root_epsilon_imp(const T*, const boost::math::integral_constant<int, 0>&)
{
   BOOST_MATH_STD_USING
   return sqrt(sqrt(tools::epsilon<T>()));
}

template <class T>
struct root_epsilon_traits
{
   typedef boost::math::integral_constant<int, (::boost::math::numeric_limits<T>::radix == 2) && (::boost::math::numeric_limits<T>::digits != (boost::math::numeric_limits<int>::max)()) ? boost::math::numeric_limits<T>::digits : 0> tag_type;
   static constexpr bool has_noexcept = (tag_type::value == 113) || (tag_type::value == 64) || (tag_type::value == 53) || (tag_type::value == 24);
};

}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T root_epsilon() noexcept(boost::math::is_floating_point<T>::value && detail::root_epsilon_traits<T>::has_noexcept)
{
   return detail::root_epsilon_imp(static_cast<T const*>(nullptr), typename detail::root_epsilon_traits<T>::tag_type());
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T cbrt_epsilon() noexcept(boost::math::is_floating_point<T>::value && detail::root_epsilon_traits<T>::has_noexcept)
{
   return detail::cbrt_epsilon_imp(static_cast<T const*>(nullptr), typename detail::root_epsilon_traits<T>::tag_type());
}

template <class T>
BOOST_MATH_GPU_ENABLED inline constexpr T forth_root_epsilon() noexcept(boost::math::is_floating_point<T>::value && detail::root_epsilon_traits<T>::has_noexcept)
{
   return detail::forth_root_epsilon_imp(static_cast<T const*>(nullptr), typename detail::root_epsilon_traits<T>::tag_type());
}

} // namespace tools
} // namespace math
} // namespace boost

#endif // BOOST_MATH_TOOLS_PRECISION_INCLUDED

