// boost\math\tools\promotion.hpp

// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2006.
// Copyright Matt Borland 2023.
// Copyright Ryan Elandt 2023.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Promote arguments functions to allow math functions to have arguments
// provided as integer OR real (floating-point, built-in or UDT)
// (called ArithmeticType in functions that use promotion)
// that help to reduce the risk of creating multiple instantiations.
// Allows creation of an inline wrapper that forwards to a foo(RT, RT) function,
// so you never get to instantiate any mixed foo(RT, IT) functions.

#ifndef BOOST_MATH_PROMOTION_HPP
#define BOOST_MATH_PROMOTION_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/type_traits.hpp>

namespace boost
{
  namespace math
  {
    namespace tools
    {
      ///// This promotion system works as follows:
      // 
      // Rule<T1> (one argument promotion rule):
      //   - Promotes `T` to `double` if `T` is an integer type as identified by
      //     `std::is_integral`, otherwise is `T`
      //
      // Rule<T1, T2_to_TN...> (two or more argument promotion rule):
      //   - 1. Calculates type using applying Rule<T1>.
      //   - 2. Calculates type using applying Rule<T2_to_TN...> 
      //   - If the type calculated in 1 and 2 are both floating point types, as
      //     identified by `std::is_floating_point`, then return the type
      //     determined by `std::common_type`. Otherwise return the type using
      //     an asymmetric convertibility rule.
      //
      ///// Discussion:
      //
      // If either T1 or T2 is an integer type,
      // pretend it was a double (for the purposes of further analysis).
      // Then pick the wider of the two floating-point types
      // as the actual signature to forward to.
      // For example:
      //    foo(int, short) -> double foo(double, double);  // ***NOT*** float foo(float, float)
      //    foo(int, float) -> double foo(double, double);  // ***NOT*** float foo(float, float)
      //    foo(int, double) -> foo(double, double);
      //    foo(double, float) -> double foo(double, double);
      //    foo(double, float) -> double foo(double, double);
      //    foo(any-int-or-float-type, long double) -> foo(long double, long double);
      // ONLY float foo(float, float) is unchanged, so the only way to get an
      // entirely float version is to call foo(1.F, 2.F). But since most (all?) the
      // math functions convert to double internally, probably there would not be the
      // hoped-for gain by using float here.
      //
      // This follows the C-compatible conversion rules of pow, etc
      // where pow(int, float) is converted to pow(double, double).


      // Promotes a single argument to double if it is an integer type
      template <class T>
      struct promote_arg {
         using type = typename boost::math::conditional<boost::math::is_integral<T>::value, double, T>::type;
      };


      // Promotes two arguments, neither of which is an integer type using an asymmetric
      // convertibility rule.
      template <class T1, class T2, bool = (boost::math::is_floating_point<T1>::value && boost::math::is_floating_point<T2>::value)>
      struct pa2_integral_already_removed {
         using type = typename boost::math::conditional<
            !boost::math::is_floating_point<T2>::value && boost::math::is_convertible<T1, T2>::value, 
            T2, T1>::type;
      };
      // For two floating point types, promotes using `std::common_type` functionality 
      template <class T1, class T2>
      struct pa2_integral_already_removed<T1, T2, true> {
         using type = boost::math::common_type_t<T1, T2, float>;
      };


      // Template definition for promote_args_permissive
      template <typename... Args>
      struct promote_args_permissive;
      // Specialization for one argument
      template <typename T>
      struct promote_args_permissive<T> {
         using type = typename promote_arg<typename boost::math::remove_cv<T>::type>::type;
      };
      // Specialization for two or more arguments
      template <typename T1, typename... T2_to_TN>
      struct promote_args_permissive<T1, T2_to_TN...> {
         using type = typename pa2_integral_already_removed<
                  typename promote_args_permissive<T1>::type,
                  typename promote_args_permissive<T2_to_TN...>::type
               >::type;
      };

      template <class... Args>
      using promote_args_permissive_t = typename promote_args_permissive<Args...>::type;


      // Same as `promote_args_permissive` but with a static assertion that the promoted type
      // is not `long double` if `BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS` is defined
      template <class... Args>
      struct promote_args {
         using type = typename promote_args_permissive<Args...>::type;
#if defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
         //
         // Guard against use of long double if it's not supported:
         //
         static_assert((0 == boost::math::is_same<type, long double>::value), "Sorry, but this platform does not have sufficient long double support for the special functions to be reliably implemented.");
#endif
      };

      template <class... Args>
      using promote_args_t = typename promote_args<Args...>::type;

    } // namespace tools
  } // namespace math
} // namespace boost

#endif // BOOST_MATH_PROMOTION_HPP
