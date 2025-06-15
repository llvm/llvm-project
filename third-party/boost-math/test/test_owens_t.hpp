//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#include <boost/math/concepts/real_concept.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"
#include "owens_t_T7.hpp"


template <class RealType>
void test_spot(
   RealType h,    //
   RealType a,    //
   RealType tol)   // Test tolerance
{
   BOOST_CHECK_CLOSE_FRACTION(owens_t(h, a), 3.89119302347013668966224771378e-2L, tol);
}

template <class RealType> // Any floating-point type RealType.
void test_spots(RealType)
{
   using namespace std;
   // Basic sanity checks, test data is as accurate as long double,
   // so set tolerance to a few epsilon expressed as a fraction.
   RealType tolerance = boost::math::tools::epsilon<RealType>() * 30; // most OK with 3 eps tolerance.
   cout << "Tolerance = " << tolerance << "." << endl;

   using  ::boost::math::owens_t;
   using ::boost::math::normal_distribution;
   BOOST_MATH_STD_USING // ADL of std names.

   // Checks of six sub-methods T1 to T6.
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(0.0625L), static_cast<RealType>(0.25L)), static_cast<RealType>(3.89119302347013668966224771378499505568e-2L), tolerance);  // T1
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(6.5L), static_cast<RealType>(0.4375L)), static_cast<RealType>(2.00057730485083154100907167684918851101649922551817956120806662022118025e-11L), tolerance); // T2
   if (boost::math::tools::digits<RealType>() < 100) // too large error for 128 bit long double
      BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(7L), static_cast<RealType>(0.96875L)), static_cast<RealType>(6.3990627193898685308321991442891601376479719094145923322318222572484602e-13L), tolerance); // T3
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(4.78125L), static_cast<RealType>(0.0625L)), static_cast<RealType>(1.06329748046874638058307112826015825291136503488102191050906959246644943e-7L), tolerance); // T4
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(2.L), static_cast<RealType>(0.5L)), static_cast<RealType>(8.6250779855215071311348831915463718787564119039085429110080944948781288e-3L), tolerance); // T5
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(1.L), static_cast<RealType>(0.9999975L)), static_cast<RealType>(6.6741808978228592771558982240461689232406934240709035854119334966793020e-2L), tolerance); // T6
   //BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(L), static_cast<RealType>(L)), static_cast<RealType>(L), tolerance);

   //   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(L), static_cast<RealType>(L)), static_cast<RealType>(L), tolerance);

   // Spots values using Mathematica
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(6.5L), static_cast<RealType>(0.4375L)), static_cast<RealType>(2.00057730485083154100907167684918851101649922551817956120806662022118024594547E-11L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(0.4375L), static_cast<RealType>(6.5L)), static_cast<RealType>(0.16540130125449396247498691826626273249659241838438244251206819782787761751256L), tolerance);
   if (boost::math::tools::digits<RealType>() < 100) // too large error for 128 bit long double
      BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(7.L), static_cast<RealType>(0.96875L)), static_cast<RealType>(6.39906271938986853083219914428916013764797190941459233223182225724846022843930e-13L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(0.96875L), static_cast<RealType>(7.L)), static_cast<RealType>(0.08316748474602973770533230453272140919966614259525787470390475393923633179072L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(4.78125L), static_cast<RealType>(0.0625L)), static_cast<RealType>(1.06329748046874638058307112826015825291136503488102191050906959246644942646701e-7L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(0.0625L), static_cast<RealType>(4.78125L)), static_cast<RealType>(0.21571185819897989857261253680409017017649352928888660746045361855686569265171L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(2.L), static_cast<RealType>(0.5L)), static_cast<RealType>(0.00862507798552150713113488319154637187875641190390854291100809449487812876461L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(0.5L), static_cast<RealType>(2L)), static_cast<RealType>(0.14158060365397839346662819588111542648867283386549027383784843786494855594607L), tolerance);
   // check basic properties
   BOOST_CHECK_EQUAL(owens_t(static_cast<RealType>(0.5L), static_cast<RealType>(2L)), owens_t(static_cast<RealType>(-0.5L), static_cast<RealType>(2L)));
   BOOST_CHECK_EQUAL(owens_t(static_cast<RealType>(0.5L), static_cast<RealType>(2L)), -owens_t(static_cast<RealType>(0.5L), static_cast<RealType>(-2L)));
   BOOST_CHECK_EQUAL(owens_t(static_cast<RealType>(0.5L), static_cast<RealType>(2L)), -owens_t(static_cast<RealType>(-0.5L), static_cast<RealType>(-2L)));

   // Special relations from Owen's original paper:
   BOOST_CHECK_EQUAL(owens_t(static_cast<RealType>(0.5), static_cast<RealType>(0)), static_cast<RealType>(0));
   BOOST_CHECK_EQUAL(owens_t(static_cast<RealType>(10), static_cast<RealType>(0)), static_cast<RealType>(0));
   BOOST_CHECK_EQUAL(owens_t(static_cast<RealType>(10000), static_cast<RealType>(0)), static_cast<RealType>(0));

   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(0), static_cast<RealType>(2L)), atan(static_cast<RealType>(2L)) / (boost::math::constants::pi<RealType>() * 2), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(0), static_cast<RealType>(0.5L)), atan(static_cast<RealType>(0.5L)) / (boost::math::constants::pi<RealType>() * 2), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(0), static_cast<RealType>(2000L)), atan(static_cast<RealType>(2000L)) / (boost::math::constants::pi<RealType>() * 2), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(5), static_cast<RealType>(1)), cdf(normal_distribution<RealType>(), 5) * cdf(complement(normal_distribution<RealType>(), 5)) / 2, tolerance);
   BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(0.125), static_cast<RealType>(1)), cdf(normal_distribution<RealType>(), 0.125) * cdf(complement(normal_distribution<RealType>(), 0.125)) / 2, tolerance);
   if(std::numeric_limits<RealType>::has_infinity)
   {
      BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(0.125), std::numeric_limits<RealType>::infinity()), cdf(complement(normal_distribution<RealType>(), 0.125)) / 2, tolerance);
      BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(5), std::numeric_limits<RealType>::infinity()), cdf(complement(normal_distribution<RealType>(), 5)) / 2, tolerance);
      BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(-0.125), std::numeric_limits<RealType>::infinity()), cdf(normal_distribution<RealType>(), -0.125) / 2, tolerance);
      BOOST_CHECK_CLOSE_FRACTION(owens_t(static_cast<RealType>(-5), std::numeric_limits<RealType>::infinity()), cdf(normal_distribution<RealType>(), -5) / 2, tolerance);
   }
} // template <class RealType>void test_spots(RealType)

template <class RealType> // Any floating-point type RealType.
void check_against_T7(RealType)
{
   using namespace std;

   if (!std::numeric_limits<RealType>::digits || (std::numeric_limits<RealType>::digits > 100))
      return;  // Can't be precise enough for this to work here

   // Basic sanity checks, test data is as accurate as long double,
   // so set tolerance to a few epsilon expressed as a fraction.
   RealType tolerance = boost::math::tools::epsilon<RealType>() * 150; // most OK with 3 eps tolerance.
   cout << "Tolerance = " << tolerance << "." << endl;

   using  ::boost::math::owens_t;
   using namespace std; // ADL of std names.

   // apply log scale because points near zero are more interesting
   for(RealType a = static_cast<RealType>(-10.0l); a < static_cast<RealType>(3l); a += static_cast<RealType>(0.2l))
      for(RealType h = static_cast<RealType>(-10.0l); h < static_cast<RealType>(3.5l); h += static_cast<RealType>(0.2l))
      {
         const RealType expa = exp(a);
         const RealType exph = exp(h);
         const RealType t = boost::math::owens_t(exph, expa);
         RealType t7 = boost::math::owens_t_T7(exph, expa);
         //if(!(boost::math::isnormal)(t) || !(boost::math::isnormal)(t7))
         //   std::cout << "a = " << expa << " h = " << exph << " t = " << t << " t7 = " << t7 << std::endl;
         BOOST_CHECK_CLOSE_FRACTION(t, t7, tolerance);
      }

} // template <class RealType>void test_spots(RealType)

template <class Real, class T>
void do_test_owens_t(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(OWENS_T_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type(*pg)(value_type, value_type);
#ifdef OWENS_T_FUNCTION_TO_TEST
   pg funcp = OWENS_T_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::owens_t<value_type>;
#else
   pg funcp = boost::math::owens_t;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test owens_t against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1),
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "owens_t", test_name);

   std::cout << std::endl;
#endif
}

template <class T>
void test_owens_t(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // three items, input value a, input value b and erf(a, b):
   //
#  include "owens_t.ipp"

   do_test_owens_t<T>(owens_t, name, "Owens T (medium small values)");

   if (!std::numeric_limits<T>::digits || (std::numeric_limits<T>::digits > 100))
      return; // can't be precise enough for next test

#include "owens_t_large_data.ipp"

   do_test_owens_t<T>(owens_t_large_data, name, "Owens T (large and diverse values)");
}
