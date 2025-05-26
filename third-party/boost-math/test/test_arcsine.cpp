// test_arcsine_dist.cpp

// Copyright John Maddock 2014.
// Copyright  Paul A. Bristow 2014.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Tests for the arcsine Distribution.

#ifndef SYCL_LANGUAGE_VERSION
#include <pch.hpp> // Must be 1st include, and include_directory /libs/math/src/tr1/ is needed.
#endif

#ifdef _MSC_VER
#  pragma warning(disable: 4127) // Conditional expression is constant.
#  pragma warning (disable : 4996) // POSIX name for this item is deprecated.
#  pragma warning (disable : 4224) // Nonstandard extension used : formal parameter 'arg' was previously defined as a type.
#endif

#include <boost/math/concepts/real_concept.hpp> // for real_concept.
using ::boost::math::concepts::real_concept;

#include <boost/math/distributions/arcsine.hpp> // for arcsine_distribution.
using boost::math::arcsine_distribution;

#include <boost/math/constants/constants.hpp>
using boost::math::constants::one_div_root_two;

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE_FRACTION

#include <cmath>

#include "test_out_of_range.hpp"

#include <iostream>
using std::cout;
using std::endl;
#include <limits>
using std::numeric_limits;

#if defined(BOOST_CHECK_THROW) && defined(BOOST_MATH_NO_EXCEPTIONS)
#  undef BOOST_CHECK_THROW
#  define BOOST_CHECK_THROW(x, y)
#endif

template <class RealType>
void test_ignore_policy(RealType)
{
  // Check on returns when errors are ignored.
  if ((typeid(RealType) != typeid(boost::math::concepts::real_concept))
    && std::numeric_limits<RealType>::has_infinity
    && std::numeric_limits<RealType>::has_quiet_NaN
    )
  { // Ordinary floats only.

    using namespace boost::math;
    //   RealType inf = std::numeric_limits<RealType>::infinity();
    RealType nan = std::numeric_limits<RealType>::quiet_NaN();

    using boost::math::policies::policy;
    // Types of error whose action can be altered by policies:.
    //using boost::math::policies::evaluation_error;
    //using boost::math::policies::domain_error;
    //using boost::math::policies::overflow_error;
    //using boost::math::policies::underflow_error;
    //using boost::math::policies::domain_error;
    //using boost::math::policies::pole_error;

    //// Actions on error (in enum error_policy_type):
    //using boost::math::policies::errno_on_error;
    //using boost::math::policies::ignore_error;
    //using boost::math::policies::throw_on_error;
    //using boost::math::policies::denorm_error;
    //using boost::math::policies::pole_error;
    //using boost::math::policies::user_error;

    typedef policy<
      boost::math::policies::domain_error<boost::math::policies::ignore_error>,
      boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
      boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
      boost::math::policies::denorm_error<boost::math::policies::ignore_error>,
      boost::math::policies::pole_error<boost::math::policies::ignore_error>,
      boost::math::policies::evaluation_error<boost::math::policies::ignore_error>
    > ignore_all_policy;

    typedef arcsine_distribution<RealType, ignore_all_policy> ignore_error_arcsine;

    // Only test NaN and infinity if type has these features (realconcept returns zero).
    // Integers are always converted to RealType,
    // others requires static cast to RealType from long double.

    if (std::numeric_limits<RealType>::has_quiet_NaN)
    {
      // Demonstrate output of PDF with infinity,
      // but string output from NaN is platform dependent, so can't use BOOST_CHECK.
      if (std::numeric_limits<RealType>::has_infinity)
      {
        //std::cout << "pdf(ignore_error_arcsine(-1, +1), std::numeric_limits<RealType>::infinity()) = " << pdf(ignore_error_arcsine(-1, +1), std::numeric_limits<RealType>::infinity()) << std::endl;
        //  Outputs:  pdf(ignore_error_arcsine(-1, +1), std::numeric_limits<RealType>::infinity()) = 1.#QNAN
      }
      BOOST_CHECK((boost::math::isnan)(pdf(ignore_error_arcsine(0, 1), std::numeric_limits<RealType>::infinity()))); // x == infinity
      BOOST_CHECK((boost::math::isnan)(pdf(ignore_error_arcsine(-1, 1), std::numeric_limits<RealType>::infinity()))); // x == infinity
      BOOST_CHECK((boost::math::isnan)(pdf(ignore_error_arcsine(0, 1), static_cast <RealType>(-2))));  // x < xmin
      BOOST_CHECK((boost::math::isnan)(pdf(ignore_error_arcsine(-1, 1), static_cast <RealType>(-2))));  // x < xmin
      BOOST_CHECK((boost::math::isnan)(pdf(ignore_error_arcsine(0, 1), static_cast <RealType>(+2))));  // x > x_max
      BOOST_CHECK((boost::math::isnan)(pdf(ignore_error_arcsine(-1, 1), static_cast <RealType>(+2)))); // x > x_max

      // Logpdf
      BOOST_CHECK((boost::math::isnan)(logpdf(ignore_error_arcsine(0, 1), std::numeric_limits<RealType>::infinity()))); // x == infinity
      BOOST_CHECK((boost::math::isnan)(logpdf(ignore_error_arcsine(-1, 1), std::numeric_limits<RealType>::infinity()))); // x == infinity
      BOOST_CHECK((boost::math::isnan)(logpdf(ignore_error_arcsine(0, 1), static_cast <RealType>(-2))));  // x < xmin
      BOOST_CHECK((boost::math::isnan)(logpdf(ignore_error_arcsine(-1, 1), static_cast <RealType>(-2))));  // x < xmin
      BOOST_CHECK((boost::math::isnan)(logpdf(ignore_error_arcsine(0, 1), static_cast <RealType>(+2))));  // x > x_max
      BOOST_CHECK((boost::math::isnan)(logpdf(ignore_error_arcsine(-1, 1), static_cast <RealType>(+2)))); // x > x_max

      // Mean
      BOOST_CHECK((boost::math::isnan)(mean(ignore_error_arcsine(-nan, 0))));
      BOOST_CHECK((boost::math::isnan)(mean(ignore_error_arcsine(+nan, 0))));

      if (std::numeric_limits<RealType>::has_infinity)
      {
        //BOOST_CHECK((boost::math::isnan)(mean(ignore_error_arcsine(-std::numeric_limits<RealType>::infinity(), 0))));
        // std::cout << "arcsine(-inf,+1) mean " << mean(ignore_error_arcsine(-std::numeric_limits<RealType>::infinity())) << std::endl;
        //BOOST_CHECK((boost::math::isnan)(mean(ignore_error_arcsine(std::numeric_limits<RealType>::infinity(), 0))));
      }

      // NaN constructors.
      BOOST_CHECK((boost::math::isnan)(mean(ignore_error_arcsine(2, nan))));
      BOOST_CHECK((boost::math::isnan)(mean(ignore_error_arcsine(nan, nan))));
      BOOST_CHECK((boost::math::isnan)(mean(ignore_error_arcsine(nan, 2))));

      // Variance
      BOOST_CHECK((boost::math::isnan)(variance(ignore_error_arcsine(nan, 0))));
      BOOST_CHECK((boost::math::isnan)(variance(ignore_error_arcsine(1, nan))));
      BOOST_CHECK((boost::math::isnan)(variance(ignore_error_arcsine(2, nan))));
      BOOST_CHECK((boost::math::isnan)(variance(ignore_error_arcsine(0, 0))));
      BOOST_CHECK((boost::math::isnan)(variance(ignore_error_arcsine(1, 0))));
      BOOST_CHECK((boost::math::isnan)(variance(ignore_error_arcsine(static_cast<RealType>(1.7L), 0))));
      BOOST_CHECK((boost::math::isnan)(variance(ignore_error_arcsine(2, 0))));

      // Skewness
      BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_arcsine(nan, 0))));
      BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_arcsine(-1, nan))));
      BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_arcsine(0, 0))));
      BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_arcsine(1, 0))));
      BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_arcsine(2, 0))));
      BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_arcsine(3, 0))));

      // Kurtosis
      BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_arcsine(nan, 0))));
      BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_arcsine(-1, nan))));
      BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_arcsine(0, 0))));
      BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_arcsine(1, 0))));
      BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_arcsine(2, 0))));
      BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_arcsine(static_cast<RealType>(2.0001L), 0))));
      BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_arcsine(3, 0))));
      BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_arcsine(4, 0))));

      // Kurtosis excess
      BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_arcsine(nan, 0))));
      BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_arcsine(-1, nan))));
      BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_arcsine(0, 0))));
      BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_arcsine(1, 0))));
      BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_arcsine(2, 0))));
      BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_arcsine(static_cast<RealType>(2.0001L), 0))));
      BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_arcsine(3, 0))));
      BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_arcsine(4, 0))));
    } // has_quiet_NaN

    //
    BOOST_CHECK(boost::math::isfinite(mean(ignore_error_arcsine(0, std::numeric_limits<RealType>::epsilon()))));

    check_support<arcsine_distribution<RealType> >(arcsine_distribution<RealType>(0, 1));
  } // ordinary floats.
} // template <class RealType> void test_ignore_policy(RealType)


template <class RealType>
RealType informax()
{ //! \return Infinity else max_value.
  return ((std::numeric_limits<RealType>::has_infinity) ?
     std::numeric_limits<RealType>::infinity() : boost::math::tools::max_value<RealType>());
}

template <class RealType>
void test_spot(
  RealType a,    // alpha a or lo or x_min
  RealType b,    // arcsine b or hi or x_maz
  RealType x,    // Probability
  RealType P,    // CDF of arcsine(a, b)
  RealType Q,    // Complement of CDF of arcsine (a, b)
  RealType tol)  // Test tolerance.
{
  boost::math::arcsine_distribution<RealType> anarcsine(a, b);
  BOOST_CHECK_CLOSE_FRACTION(cdf(anarcsine, x), P, tol);
  if ((P < 0.99) && (Q < 0.99))
  { // We can only check this if P is not too close to 1,
    // so that we can guarantee that Q is free of error,
    // (and similarly for Q).
    BOOST_CHECK_CLOSE_FRACTION(cdf(complement(anarcsine, x)), Q, tol);
    if (x != 0)
    {
      BOOST_CHECK_CLOSE_FRACTION(
        quantile(anarcsine, P), x, tol);
    }
    else
    {
      // Just check quantile is very small:
      if ((std::numeric_limits<RealType>::max_exponent <= std::numeric_limits<double>::max_exponent)
        && (boost::is_floating_point<RealType>::value))
      {
        // Limit where this is checked: if exponent range is very large we may
        // run out of iterations in our root finding algorithm.
        BOOST_CHECK(quantile(anarcsine, P) < boost::math::tools::epsilon<RealType>() * 10);
      }
    } // if k
    if (x != 0)
    {
      BOOST_CHECK_CLOSE_FRACTION(quantile(complement(anarcsine, Q)), x, tol * 10);
    }
    else
    {  // Just check quantile is very small:
      if ((std::numeric_limits<RealType>::max_exponent <= std::numeric_limits<double>::max_exponent) && (boost::is_floating_point<RealType>::value))
      { // Limit where this is checked: if exponent range is very large we may
        // run out of iterations in our root finding algorithm.
        BOOST_CHECK(quantile(complement(anarcsine, Q)) < boost::math::tools::epsilon<RealType>() * 10);
      }
    } // if x
  }
} // template <class RealType> void test_spot

template <class RealType> // Any floating-point type RealType.
void test_spots(RealType)
{
    // Basic sanity checks with 'known good' values.
    // so set tolerance to a few eps expressed as a fraction, or
    // few eps of type double expressed as a fraction,
    // whichever is the larger.

    RealType tolerance = (std::max)
      (boost::math::tools::epsilon<RealType>(),
      static_cast<RealType>(std::numeric_limits<double>::epsilon())); // 0 if real_concept.

    tolerance *= 2; // Note: NO * 100 because tolerance is a fraction, NOT %.
    cout << "tolerance = " << tolerance << endl;

    using boost::math::arcsine_distribution;
    using  ::boost::math::cdf;
    using  ::boost::math::pdf;
    using  ::boost::math::logpdf;
    using  ::boost::math::complement;
    using  ::boost::math::quantile;

    // Basic sanity-check spot values.

    // Test values from Wolfram alpha, for example:
    // http://www.wolframalpha.com/input/?i=+N%5BPDF%5Barcsinedistribution%5B0%2C+1%5D%2C+0.5%5D%2C+50%5D
    // N[PDF[arcsinedistribution[0, 1], 0.5], 50]
    // 0.63661977236758134307553505349005744813783858296183

    arcsine_distribution<RealType> arcsine_01; // (Our) Standard arcsine.
    // Member functions.
    BOOST_CHECK_EQUAL(arcsine_01.x_min(), 0);
    BOOST_CHECK_EQUAL(arcsine_01.x_max(), 1);

    // Derived functions.
    BOOST_CHECK_EQUAL(mean(arcsine_01), 0.5); // 1 / (1 + 1) = 1/2 exactly.
    BOOST_CHECK_EQUAL(median(arcsine_01), 0.5); // 1 / (1 + 1) = 1/2 exactly.
    BOOST_CHECK_EQUAL(variance(arcsine_01), 0.125); // 1/8 = 0.125
    BOOST_CHECK_CLOSE_FRACTION(standard_deviation(arcsine_01), one_div_root_two<double>() / 2, tolerance); // 1/ sqrt(s) = 0.35355339059327379
    BOOST_CHECK_EQUAL(skewness(arcsine_01), 0); //
    BOOST_CHECK_EQUAL(kurtosis_excess(arcsine_01), -1.5); // 3/2
    BOOST_CHECK_EQUAL(support(arcsine_01).first, 0); //
    BOOST_CHECK_EQUAL(range(arcsine_01).first, 0); //
    BOOST_CHECK_THROW(mode(arcsine_01), std::domain_error); //  Two modes at x_min and x_max, so throw instead.

    // PDF
    // pdf of x = 1/4 is same as reflected value at x = 3/4.
    // N[PDF[arcsinedistribution[0, 1], 0.25], 50]
    // N[PDF[arcsinedistribution[0, 1], 0.75], 50]
    // 0.73510519389572273268176866441729258852984864048885

    BOOST_CHECK_CLOSE_FRACTION(pdf(arcsine_01, 0.000001), static_cast<RealType>(318.31004533885312973989414360099118178698415543136L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(pdf(arcsine_01, 0.000005), static_cast<RealType>(142.35286456604168061345817902422241622116338936911L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(pdf(arcsine_01, 0.05), static_cast<RealType>(1.4605059227421865250256574657088244053723856445614L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(pdf(arcsine_01, 0.5), static_cast<RealType>(0.63661977236758134307553505349005744813783858296183L), tolerance);
    // Note loss of significance when x is near x_max.
    BOOST_CHECK_CLOSE_FRACTION(pdf(arcsine_01, 0.95), static_cast<RealType>(1.4605059227421865250256574657088244053723856445614L), 8 * tolerance); // Less accurate.
    BOOST_CHECK_CLOSE_FRACTION(pdf(arcsine_01, 0.999995), static_cast<RealType>(142.35286456604168061345817902422241622116338936911L), 50000 * tolerance); // Much less accurate.
    BOOST_CHECK_CLOSE_FRACTION(pdf(arcsine_01, 0.999999), static_cast<RealType>(318.31004533885312973989414360099118178698415543136L), 100000 * tolerance);// Even less accurate.

    // Extreme x.
    #ifndef BOOST_MATH_ENABLE_SYCL
    if (std::numeric_limits<RealType>::has_infinity)
    { //
      BOOST_CHECK_EQUAL(pdf(arcsine_01, 0), informax<RealType>()); //
      BOOST_CHECK_EQUAL(pdf(arcsine_01, 1), informax<RealType>()); //
    }
    #endif

    BOOST_CHECK_CLOSE_FRACTION(pdf(arcsine_01, tolerance),
      1 /(sqrt(tolerance) * boost::math::constants::pi<RealType>()), 2 * tolerance); //
    BOOST_CHECK_CLOSE_FRACTION(pdf(arcsine_01, static_cast<RealType>(1) - tolerance),
      1 /(sqrt(tolerance) * boost::math::constants::pi<RealType>()), 2 * tolerance); //

    // Log PDF
    BOOST_CHECK_CLOSE_FRACTION(logpdf(arcsine_01, 0.000001), static_cast<RealType>(5.7630258931329868780772138043668005779060097243996L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(logpdf(arcsine_01, 0.000005), static_cast<RealType>(4.9583089369219367114435788047327747268154560240604L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(logpdf(arcsine_01, 0.05), static_cast<RealType>(0.37878289812137058928728250884555529541061717942415L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(logpdf(arcsine_01, 0.5), static_cast<RealType>(-0.45158270528945486472619522989488214357179467855506L), tolerance);
    // Note loss of significance when x is near x_max.
    BOOST_CHECK_CLOSE_FRACTION(logpdf(arcsine_01, 0.95), static_cast<RealType>(0.37878289812137058928728250884555529541061717942415L), 8 * tolerance); // Less accurate.
    BOOST_CHECK_CLOSE_FRACTION(logpdf(arcsine_01, 0.999995), static_cast<RealType>(4.9583089369219367114435788047327747268154560240604L), 50000 * tolerance); // Much less accurate.
    BOOST_CHECK_CLOSE_FRACTION(logpdf(arcsine_01, 0.999999), static_cast<RealType>(5.7630258931329868780772138043668005779060097243996L), 100000 * tolerance);// Even less accurate.

    // CDF
    BOOST_CHECK_CLOSE_FRACTION(cdf(arcsine_01, 0.000001), static_cast<RealType>(0.00063661987847092448418377367957384866092127786060574L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cdf(arcsine_01, 0.000005), static_cast<RealType>(0.0014235262731079289297302426454125318201831474507326L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cdf(arcsine_01, 0.05), static_cast<RealType>(0.14356629312870627075094188477505571882161519989741L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cdf(arcsine_01, 0.5), static_cast<RealType>(0.5L), tolerance); // Exact.
    BOOST_CHECK_CLOSE_FRACTION(cdf(arcsine_01, 0.95), static_cast<RealType>(0.85643370687129372924905811522494428117838480010259L), 2 * tolerance);
    // Values near unity should use the cdf complemented for better accuracy,
    BOOST_CHECK_CLOSE_FRACTION(cdf(arcsine_01, 0.999995), static_cast<RealType>(0.99857647372689207107026975735458746817981685254927L), 100 * tolerance); // Less accurate.
    BOOST_CHECK_CLOSE_FRACTION(cdf(arcsine_01, 0.999999), static_cast<RealType>(0.99936338012152907551581622632042615133907872213939L), 1000 * tolerance); // Less accurate.

    //  Complement CDF
    BOOST_CHECK_CLOSE_FRACTION(cdf(complement(arcsine_01, 0.000001)), static_cast<RealType>(1 - 0.00063661987847092448418377367957384866092127786060574L), 2 * tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cdf(complement(arcsine_01, 0.000001)), static_cast<RealType>(0.99936338012152907551581622632043L), 2 * tolerance); //
    BOOST_CHECK_CLOSE_FRACTION(cdf(complement(arcsine_01, 0.05)), static_cast<RealType>(0.85643370687129372924905811522494428117838480010259L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cdf(complement(arcsine_01, 0.5)), static_cast<RealType>(0.5L), tolerance); // Exact.
    // Some values near unity when complement is expected to be less accurate.
    BOOST_CHECK_CLOSE_FRACTION(cdf(complement(arcsine_01, 0.95)), static_cast<RealType>(0.14356629312870627075094188477505571882161519989741L), 8 * tolerance); // 2 for asin
    BOOST_CHECK_CLOSE_FRACTION(cdf(complement(arcsine_01, 0.999999)), static_cast<RealType>(1 - 0.99936338012152907551581622632042615133907872213939L), 1000000 * tolerance); // 10000 for asin, 1000000 for acos.

    // Quantile.

    // Check 1st, 2nd and 3rd quartiles.
    BOOST_CHECK_CLOSE_FRACTION(quantile(arcsine_01, static_cast<RealType>(0.25L)), static_cast<RealType>(0.14644660940672624L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(quantile(arcsine_01, static_cast<RealType>(0.5L)), 0.5, 2 * tolerance);  // probability = 0.5, x = 0.5
    BOOST_CHECK_CLOSE_FRACTION(quantile(arcsine_01, static_cast<RealType>(0.75L)), static_cast<RealType>(0.85355339059327373L), tolerance);

    // N[CDF[arcsinedistribution[0, 1], 0.05], 50]  == 0.14356629312870627075094188477505571882161519989741
    BOOST_CHECK_CLOSE_FRACTION(quantile(arcsine_01, static_cast<RealType>(0.14356629312870627075094188477505571882161519989741L)), 0.05, tolerance);

    // Quantile of complement.
    // N[1-CDF[arcsinedistribution[0, 1], 0.05], 50] == 0.85643370687129372924905811522494428117838480010259
    BOOST_CHECK_CLOSE_FRACTION(quantile(complement(arcsine_01, static_cast<RealType>(0.85643370687129372924905811522494428117838480010259L))), 0.05, tolerance * 2);
    // N[sin^2[0.75 * pi/2],50] == 0.85355339059327376220042218105242451964241796884424
    BOOST_CHECK_CLOSE_FRACTION(quantile(complement(arcsine_01, static_cast<RealType>(0.25L))), static_cast<RealType>(0.85355339059327376220042218105242451964241796884424L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(quantile(complement(arcsine_01, static_cast<RealType>(0.5L))), 0.5, 2 * tolerance);  // probability = 0.5, x = 0.5
    BOOST_CHECK_CLOSE_FRACTION(quantile(complement(arcsine_01, static_cast<RealType>(0.75L))), static_cast<RealType>(0.14644660940672623779957781894757548035758203115576L), 2 * tolerance); // Less accurate.

    // N[CDF[arcsinedistribution[0, 1], 0.25], 5
    // 0.33333333333333333333333333333333333333333333333333
    BOOST_CHECK_CLOSE_FRACTION(quantile(arcsine_01, static_cast<RealType>(1) / 3), static_cast<RealType>(0.25L), 2 * tolerance);
    BOOST_CHECK_CLOSE_FRACTION(quantile(arcsine_01, static_cast<RealType>(0.5L)), 0.5, 2 * tolerance);  // probability = 0.5, x = 0.5
    BOOST_CHECK_CLOSE_FRACTION(quantile(arcsine_01, static_cast<RealType>(2) / 3), static_cast<RealType>(0.75L), tolerance);

    // Arcsine(-1, +1)    xmin = -1, x_max = +1  symmetric about zero.
    arcsine_distribution<RealType> as_m11(-1, +1);

    BOOST_CHECK_EQUAL(as_m11.x_min(), -1); //
    BOOST_CHECK_EQUAL(as_m11.x_max(), +1);
    BOOST_CHECK_EQUAL(mean(as_m11), 0); //
    BOOST_CHECK_EQUAL(median(as_m11), 0); //
    BOOST_CHECK_CLOSE_FRACTION(standard_deviation(as_m11), one_div_root_two<RealType>(),  tolerance * 2); //

    BOOST_CHECK_EQUAL(variance(as_m11), 0.5); // 1 - (-1) = 2 ^ 2 = 4 /8 = 0.5
    BOOST_CHECK_EQUAL(skewness(as_m11), 0); //
    BOOST_CHECK_EQUAL(kurtosis_excess(as_m11), -1.5); // 3/2


    BOOST_CHECK_CLOSE_FRACTION(pdf(as_m11, 0.05), static_cast<RealType>(0.31870852113797122803869876869296281629727218095644L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(pdf(as_m11, 0.5), static_cast<RealType>(0.36755259694786136634088433220864629426492432024443L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(pdf(as_m11, 0.95), static_cast<RealType>(1.0194074882503562519812229448639426942621591013381L), 2 * tolerance); // Less accurate.

    BOOST_CHECK_CLOSE_FRACTION(logpdf(as_m11, 0.05), static_cast<RealType>(-1.1434783207403409089630164813372974217316704642782L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(logpdf(as_m11, 0.5), static_cast<RealType>(-1.0008888496235097104238178483561449958955399574664L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(logpdf(as_m11, 0.95), static_cast<RealType>(0.019221564639767605567429885545559909302927558782238L), 100 * tolerance); // Less accurate.

    BOOST_CHECK_CLOSE_FRACTION(cdf(as_m11, 0.05), static_cast<RealType>(0.51592213323666034437274347433261364289389772737836L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cdf(as_m11, 0.5), static_cast<RealType>(0.66666666666666666666666666666666666666666666666667L), 2 * tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cdf(as_m11, 0.95), static_cast<RealType>(0.89891737589574013042121018491729701360300248368629L), tolerance); //  Not less accurate.

    // Quantile
    BOOST_CHECK_CLOSE_FRACTION(quantile(as_m11, static_cast<RealType>(1) / 3), -static_cast<RealType>(0.5L), 2 * tolerance); // p = 1/3 x = -0.5
    BOOST_CHECK_SMALL(quantile(as_m11, static_cast<RealType>(0.5L)), 2 * tolerance);                             // p = 0.5, x = 0
    BOOST_CHECK_CLOSE_FRACTION(quantile(as_m11, static_cast<RealType>(2) / 3), +static_cast<RealType>(0.5L), 4 * tolerance);     // p = 2/3, x = +0.5

    //  Loop back tests.
    test_spot(
      static_cast<RealType>(0),   // lo or a
      static_cast<RealType>(1),   // hi or b
      static_cast<RealType>(0.05), // Random variate  x
      static_cast<RealType>(0.14356629312870627075094188477505571882161519989741L), // Probability of result (CDF of arcsine), P
      static_cast<RealType>(0.85643370687129372924905811522494428117838480010259L),  // Complement of CDF Q = 1 - P
      tolerance); // Test tolerance.

    test_spot(
      static_cast<RealType>(0),   // lo or a
      static_cast<RealType>(1),   // hi or b
      static_cast<RealType>(0.95), // Random variate  x
      static_cast<RealType>(0.85643370687129372924905811522494428117838480010259L), // Probability of result (CDF of arcsine), P
      static_cast<RealType>(0.14356629312870627075094188477505571882161519989741L),  // Complement of CDF Q = 1 - P
      tolerance * 4); // Test tolerance (slightly increased compared to x < 0.5 above).

    test_spot(
      static_cast<RealType>(0),   // lo or a
      static_cast<RealType>(1),   // hi or b
      static_cast<RealType>(static_cast<RealType>(0.5L)), // Random variate  x
      static_cast<RealType>(static_cast<RealType>(0.5L)), // Probability of result (CDF of arcsine), P
      static_cast<RealType>(static_cast<RealType>(0.5L)),  // Complement of CDF Q = 1 - P
      tolerance * 4); // Test tolerance.

    // Arcsine(-2, -1) xmin = -2, x_max = -1  - Asymmetric both negative.
    arcsine_distribution<RealType> as_m2m1(-2, -1);

    BOOST_CHECK_EQUAL(as_m2m1.x_min(), -2); //
    BOOST_CHECK_EQUAL(as_m2m1.x_max(), -1);
    BOOST_CHECK_EQUAL(mean(as_m2m1), -1.5); // 1 / (1 + 1) = 1/2 exactly.
    BOOST_CHECK_EQUAL(median(as_m2m1), -1.5); // 1 / (1 + 1) = 1/2 exactly.
    BOOST_CHECK_EQUAL(variance(as_m2m1), 0.125);
    BOOST_CHECK_EQUAL(skewness(as_m2m1), 0); //
    BOOST_CHECK_EQUAL(kurtosis_excess(as_m2m1), -1.5); // 3/2

    BOOST_CHECK_CLOSE_FRACTION(pdf(as_m2m1, -1.95), static_cast<RealType>(1.4605059227421865250256574657088244053723856445614L), 4 * tolerance);
    BOOST_CHECK_CLOSE_FRACTION(pdf(as_m2m1, -1.5), static_cast<RealType>(0.63661977236758134307553505349005744813783858296183L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(pdf(as_m2m1, -1.05), static_cast<RealType>(1.4605059227421865250256574657088244053723856445614L), 4 * tolerance); // Less accurate.

    BOOST_CHECK_CLOSE_FRACTION(cdf(as_m2m1, -1.05), static_cast<RealType>(0.85643370687129372924905811522494428117838480010259L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cdf(as_m2m1, -1.5), static_cast<RealType>(0.5L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cdf(as_m2m1, -1.95), static_cast<RealType>(0.14356629312870627075094188477505571882161519989741L), 8 * tolerance); //  Not much less accurate.

    // Quantile
    BOOST_CHECK_CLOSE_FRACTION(quantile(as_m2m1, static_cast<RealType>(0.85643370687129372924905811522494428117838480010259L)), -static_cast<RealType>(1.05L), 2 * tolerance); //
    BOOST_CHECK_CLOSE_FRACTION(quantile(as_m2m1, static_cast<RealType>(0.5L)), -static_cast<RealType>(1.5L), 2 * tolerance);                             //
    BOOST_CHECK_CLOSE_FRACTION(quantile(as_m2m1, static_cast<RealType>(0.14356629312870627075094188477505571882161519989741L)), -static_cast<RealType>(1.95L), 4 * tolerance);     //

    BOOST_CHECK_CLOSE_FRACTION(quantile(complement(as_m2m1, static_cast<RealType>(0.14356629312870627075094188477505571882161519989741L))), -static_cast<RealType>(1.05L), 2 * tolerance); //
    BOOST_CHECK_CLOSE_FRACTION(quantile(as_m2m1, static_cast<RealType>(0.5L)), -static_cast<RealType>(1.5L), 2 * tolerance);                             //
    BOOST_CHECK_CLOSE_FRACTION(quantile(complement(as_m2m1, static_cast<RealType>(0.85643370687129372924905811522494428117838480010259L))), -static_cast<RealType>(1.95L), 4 * tolerance);

    // Tests that should throw:
    BOOST_CHECK_THROW(mode(arcsine_distribution<RealType>(static_cast<RealType>(0), static_cast<RealType>(1))), std::domain_error);
    // mode is undefined, and must throw domain_error!


    BOOST_CHECK_THROW( // For various bad arguments.
      pdf(
      arcsine_distribution<RealType>(static_cast<RealType>(+1), static_cast<RealType>(-1)), // min_x > max_x
      static_cast<RealType>(1)), std::domain_error);

    BOOST_CHECK_THROW(
      pdf(
      arcsine_distribution<RealType>(static_cast<RealType>(1), static_cast<RealType>(0)), // bad constructor parameters.
      static_cast<RealType>(1)), std::domain_error);

    BOOST_CHECK_THROW(
      pdf(
      arcsine_distribution<RealType>(static_cast<RealType>(1), static_cast<RealType>(-1)), // bad constructor parameters.
      static_cast<RealType>(1)), std::domain_error);

    BOOST_CHECK_THROW(
      pdf(
      arcsine_distribution<RealType>(static_cast<RealType>(1), static_cast<RealType>(1)), // equal constructor parameters.
      static_cast<RealType>(-1)), std::domain_error);

    BOOST_CHECK_THROW(
      pdf(
      arcsine_distribution<RealType>(static_cast<RealType>(0), static_cast<RealType>(1)), // bad x > 1.
      static_cast<RealType>(999)), std::domain_error);

    BOOST_CHECK_THROW( // For various bad arguments.
      logpdf(
      arcsine_distribution<RealType>(static_cast<RealType>(+1), static_cast<RealType>(-1)), // min_x > max_x
      static_cast<RealType>(1)), std::domain_error);

    BOOST_CHECK_THROW(
      logpdf(
      arcsine_distribution<RealType>(static_cast<RealType>(1), static_cast<RealType>(0)), // bad constructor parameters.
      static_cast<RealType>(1)), std::domain_error);

    BOOST_CHECK_THROW(
      logpdf(
      arcsine_distribution<RealType>(static_cast<RealType>(1), static_cast<RealType>(-1)), // bad constructor parameters.
      static_cast<RealType>(1)), std::domain_error);

    BOOST_CHECK_THROW(
      logpdf(
      arcsine_distribution<RealType>(static_cast<RealType>(1), static_cast<RealType>(1)), // equal constructor parameters.
      static_cast<RealType>(-1)), std::domain_error);

    BOOST_CHECK_THROW(
      logpdf(
      arcsine_distribution<RealType>(static_cast<RealType>(0), static_cast<RealType>(1)), // bad x > 1.
      static_cast<RealType>(999)), std::domain_error);

    // Checks on things that are errors.

    // Construction with 'bad' parameters.
    BOOST_CHECK_THROW(arcsine_distribution<RealType>(+1, -1), std::domain_error); // max < min.
    BOOST_CHECK_THROW(arcsine_distribution<RealType>(+1, 0), std::domain_error);  // max < min.

    arcsine_distribution<> dist;
    BOOST_CHECK_THROW(pdf(dist, -1), std::domain_error);
    BOOST_CHECK_THROW(logpdf(dist, -1), std::domain_error);
    BOOST_CHECK_THROW(cdf(dist, -1), std::domain_error);
    BOOST_CHECK_THROW(cdf(complement(dist, -1)), std::domain_error);
    BOOST_CHECK_THROW(quantile(dist, -1), std::domain_error);
    BOOST_CHECK_THROW(quantile(complement(dist, -1)), std::domain_error);
    BOOST_CHECK_THROW(quantile(dist, -1), std::domain_error);
    BOOST_CHECK_THROW(quantile(complement(dist, -1)), std::domain_error);

    // Various combinations of bad constructor and member function parameters.
    BOOST_CHECK_THROW(pdf(boost::math::arcsine_distribution<RealType>(0, 1), -1), std::domain_error);
    BOOST_CHECK_THROW(pdf(boost::math::arcsine_distribution<RealType>(-1, 1), +2), std::domain_error);
    BOOST_CHECK_THROW(logpdf(boost::math::arcsine_distribution<RealType>(0, 1), -1), std::domain_error);
    BOOST_CHECK_THROW(logpdf(boost::math::arcsine_distribution<RealType>(-1, 1), +2), std::domain_error);
    BOOST_CHECK_THROW(quantile(boost::math::arcsine_distribution<RealType>(1, 1), -1), std::domain_error);
    BOOST_CHECK_THROW(quantile(boost::math::arcsine_distribution<RealType>(1, 1), 2), std::domain_error);

    // No longer allow any parameter to be NaN or inf, so all these tests should throw.
    if (std::numeric_limits<RealType>::has_quiet_NaN)
    {
      // Attempt to construct from non-finite parameters should throw.
      RealType nan = std::numeric_limits<RealType>::quiet_NaN();
#ifndef BOOST_NO_EXCEPTIONS
      BOOST_CHECK_THROW(arcsine_distribution<RealType> w(nan), std::domain_error);
      BOOST_CHECK_THROW(arcsine_distribution<RealType> w(1, nan), std::domain_error);
      BOOST_CHECK_THROW(arcsine_distribution<RealType> w(nan, 1), std::domain_error);
#else
      BOOST_CHECK_THROW(arcsine_distribution<RealType>(nan), std::domain_error);
      BOOST_CHECK_THROW(arcsine_distribution<RealType>(1, nan), std::domain_error);
      BOOST_CHECK_THROW(arcsine_distribution<RealType>(nan, 1), std::domain_error);
#endif

      arcsine_distribution<RealType> w(RealType(-1), RealType(+1));
      // NaN parameters to member functions should throw.
      BOOST_CHECK_THROW(pdf(w, +nan), std::domain_error); // x = NaN
      BOOST_CHECK_THROW(logpdf(w, +nan), std::domain_error); // x = NaN
      BOOST_CHECK_THROW(cdf(w, +nan), std::domain_error); // x = NaN
      BOOST_CHECK_THROW(cdf(complement(w, +nan)), std::domain_error); // x = + nan
      BOOST_CHECK_THROW(quantile(w, +nan), std::domain_error); // p = + nan
      BOOST_CHECK_THROW(quantile(complement(w, +nan)), std::domain_error); // p = + nan
    } // has_quiet_NaN

    if (std::numeric_limits<RealType>::has_infinity)
    {
      // Attempt to construct from non-finite should throw.
      RealType inf = std::numeric_limits<RealType>::infinity();
#ifndef BOOST_NO_EXCEPTIONS
      BOOST_CHECK_THROW(arcsine_distribution<RealType> w(inf), std::domain_error);
      BOOST_CHECK_THROW(arcsine_distribution<RealType> w(1, inf), std::domain_error);
#else
      BOOST_CHECK_THROW(arcsine_distribution<RealType>(inf), std::domain_error);
      BOOST_CHECK_THROW(arcsine_distribution<RealType>(1, inf), std::domain_error);
#endif
      // Infinite parameters to member functions should throw.
      arcsine_distribution<RealType> w(RealType(0), RealType(1));
#ifndef BOOST_NO_EXCEPTIONS
      BOOST_CHECK_THROW(arcsine_distribution<RealType> w(inf), std::domain_error);
      BOOST_CHECK_THROW(arcsine_distribution<RealType> w(1, inf), std::domain_error);
#else
      BOOST_CHECK_THROW(arcsine_distribution<RealType>(inf), std::domain_error);
      BOOST_CHECK_THROW(arcsine_distribution<RealType>(1, inf), std::domain_error);
#endif
      BOOST_CHECK_THROW(pdf(w, +inf), std::domain_error); // x = inf
      BOOST_CHECK_THROW(logpdf(w, +inf), std::domain_error); // x = inf
      BOOST_CHECK_THROW(cdf(w, +inf), std::domain_error); // x = inf
      BOOST_CHECK_THROW(cdf(complement(w, +inf)), std::domain_error); // x = + inf
      BOOST_CHECK_THROW(quantile(w, +inf), std::domain_error); // p = + inf
      BOOST_CHECK_THROW(quantile(complement(w, +inf)), std::domain_error); // p = + inf
    } // has_infinity

    // Error handling checks:
    check_out_of_range<boost::math::arcsine_distribution<RealType> >(-1, +1); // (All) valid constructor parameter values.
    // and range and non-finite.

    test_ignore_policy(static_cast<RealType>(0));

  } // template <class RealType>void test_spots(RealType)

  BOOST_AUTO_TEST_CASE(test_main)
  {
    BOOST_MATH_CONTROL_FP;

    // Check that can generate arcsine distribution using convenience method:
    using boost::math::arcsine;

    arcsine_distribution<> arcsine_01; // Using default RealType double.
    // Note: NOT arcsine01() - or compiler will assume a function.

    arcsine as; // Using typedef for default standard arcsine.

    //
    BOOST_CHECK_EQUAL(as.x_min(), 0); //
    BOOST_CHECK_EQUAL(as.x_max(), 1);
    BOOST_CHECK_EQUAL(mean(as), 0.5); // 1 / (1 + 1) = 1/2 exactly.
    BOOST_CHECK_EQUAL(median(as), 0.5); // 1 / (1 + 1) = 1/2 exactly.
    BOOST_CHECK_EQUAL(variance(as), 0.125); //0.125
    BOOST_CHECK_CLOSE_FRACTION(standard_deviation(as), one_div_root_two<double>() / 2, std::numeric_limits<double>::epsilon()); // 0.353553
    BOOST_CHECK_EQUAL(skewness(as), 0); //
    BOOST_CHECK_EQUAL(kurtosis_excess(as), -1.5); // 3/2
    BOOST_CHECK_EQUAL(support(as).first, 0); //
    BOOST_CHECK_EQUAL(range(as).first, 0); //
    BOOST_CHECK_THROW(mode(as), std::domain_error); //  Two modes at x_min and x_max, so throw instead.

    // (Parameter value, arbitrarily zero, only communicates the floating point type).
    test_spots(0.0F); // Test float.
    test_spots(0.0); // Test double.
    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
      test_spots(0.0L); // Test long double.
      #if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
        test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
      #endif
    #endif
  /*    */
  } // BOOST_AUTO_TEST_CASE( test_main )

  /*


Microsoft Visual Studio Professional 2013
Version 12.0.30110.00 Update 1

  1>  Description: Autorun "J:\Cpp\MathToolkit\test\Math_test\Debug\test_arcsine.exe"
  1>  Running 1 test case...
  1>  Platform: Win32
  1>  Compiler: Microsoft Visual C++ version 12.0  ???? MSVC says 2013
  1>  STL     : Dinkumware standard library version 610
  1>  Boost   : 1.56.0

  Sample Output is:

  1>  Description: Autorun "J:\Cpp\MathToolkit\test\Math_test\Debug\test_arcsine.exe"
  1>  Running 1 test case...
  1>  Platform: Win32
  1>  Compiler: Microsoft Visual C++ version 12.0
  1>  STL     : Dinkumware standard library version 610
  1>  Boost   : 1.56.0
  1>  tolerance = 2.38419e-007
  1>  tolerance = 4.44089e-016
  1>  tolerance = 4.44089e-016
  1>  tolerance = 4.44089e-016
  1>
  1>  *** No errors detected

  GCC 4.9.1

  Running 1 test case...
  tolerance = 2.38419e-007
  tolerance = 4.44089e-016
  tolerance = 4.44089e-016
  tolerance = 4.44089e-016

  *** No errors detected

  RUN SUCCESSFUL (total time: 141ms)

  */
