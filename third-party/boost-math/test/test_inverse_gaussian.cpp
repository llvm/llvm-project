// Copyright Paul A. Bristow 2010.
// Copyright John Maddock 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224) // nonstandard extension used : formal parameter 'type' was previously defined as a type
// in Boost.test and lexical_cast
#  pragma warning (disable : 4310) // cast truncates constant value
#  pragma warning (disable : 4512) // assignment operator could not be generated

#endif

//#include <pch.hpp> // include directory libs/math/src/tr1/ is needed.

#include <boost/math/tools/config.hpp>
#include "../include_private/boost/math/tools/test.hpp"

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#endif

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/math/distributions/inverse_gaussian.hpp>
using boost::math::inverse_gaussian_distribution;
using boost::math::inverse_gaussian;

#include "test_out_of_range.hpp"

#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
using std::setprecision;
#include <limits>
using std::numeric_limits;
#include <cmath>
using std::log;

template <class RealType>
void check_inverse_gaussian(RealType mean, RealType scale, RealType x, RealType p, RealType q, RealType tol)
{
 using boost::math::inverse_gaussian_distribution;

  BOOST_CHECK_CLOSE_FRACTION(
    ::boost::math::cdf(   // Check cdf
    inverse_gaussian_distribution<RealType>(mean, scale),      // distribution.
    x),    // random variable.
    p,     // probability.
    tol);   // tolerance.
  BOOST_CHECK_CLOSE_FRACTION(
    ::boost::math::cdf( // Check cdf complement
    complement( 
    inverse_gaussian_distribution<RealType>(mean, scale),   // distribution.
    x)),   // random variable.
    q,      // probability complement.
    tol);    // %tolerance.
  BOOST_CHECK_CLOSE_FRACTION(
    ::boost::math::quantile( // Check quantile
    inverse_gaussian_distribution<RealType>(mean, scale),    // distribution.
    p),   // probability.
    x,   // random variable.
    tol);   // tolerance.
  BOOST_CHECK_CLOSE_FRACTION(
    ::boost::math::quantile( // Check quantile complement
    complement(
    inverse_gaussian_distribution<RealType>(mean, scale),   // distribution.
    q)),   // probability complement.
    x,     // random variable.
    tol);  // tolerance.

   inverse_gaussian_distribution<RealType> dist (mean, scale);

   if((p < 0.999) && (q < 0.999))
   {  // We can only check this if P is not too close to 1,
      // so that we can guarantee Q is accurate:
      BOOST_CHECK_CLOSE_FRACTION(
        cdf(complement(dist, x)), q, tol); // 1 - cdf
      BOOST_CHECK_CLOSE_FRACTION(
        quantile(dist, p), x, tol); // quantile(cdf) = x
      BOOST_CHECK_CLOSE_FRACTION(
        quantile(complement(dist, q)), x, tol); // quantile(complement(1 - cdf)) = x
   }
}

template <class RealType>
void test_spots(RealType)
{
  // Basic sanity checks
  RealType tolerance = static_cast<RealType>(1e-4L); // 
  cout << "Tolerance for type " << typeid(RealType).name()  << " is " << tolerance << endl;

  // Check some bad parameters to the distribution,
#ifndef BOOST_NO_EXCEPTIONS
  BOOST_MATH_CHECK_THROW(boost::math::inverse_gaussian_distribution<RealType> nbad1(0, 0), std::domain_error); // zero scale
  BOOST_MATH_CHECK_THROW(boost::math::inverse_gaussian_distribution<RealType> nbad1(0, -1), std::domain_error); // negative scale
#else
  BOOST_MATH_CHECK_THROW(boost::math::inverse_gaussian_distribution<RealType>(0, 0), std::domain_error); // zero scale
  BOOST_MATH_CHECK_THROW(boost::math::inverse_gaussian_distribution<RealType>(0, -1), std::domain_error); // negative scale
#endif

  inverse_gaussian_distribution<RealType> w11;

  // Error tests:
  check_out_of_range<inverse_gaussian_distribution<RealType> >(0.25, 1);
  
  // Check complements.

    BOOST_CHECK_CLOSE_FRACTION(
     cdf(complement(w11, 1.)), static_cast<RealType>(1) - cdf(w11, 1.), tolerance); // cdf complement
    // cdf(complement = 1 - cdf  - but if cdf near unity, then loss of accuracy in cdf,
    // but cdf complement is near zero but more accurate.

     BOOST_CHECK_CLOSE_FRACTION( // quantile(complement p) == quantile(1 - p)
     quantile(complement(w11, static_cast<RealType>(0.5))), 
     quantile(w11, 1 - static_cast<RealType>(0.5)),
     tolerance); // cdf complement

  check_inverse_gaussian(
     static_cast<RealType>(2),
     static_cast<RealType>(3),
     static_cast<RealType>(1),
     static_cast<RealType>(0.28738674440477374),
     static_cast<RealType>(1 - 0.28738674440477374),
     tolerance);

  RealType tolfeweps = boost::math::tools::epsilon<RealType>() * 5;

  inverse_gaussian_distribution<RealType> dist(2, 3);

  using namespace std; // ADL of std names.
  // mean:
  BOOST_CHECK_CLOSE_FRACTION(mean(dist),
    static_cast<RealType>(2), tolfeweps);
  BOOST_CHECK_CLOSE_FRACTION(scale(dist),
    static_cast<RealType>(3), tolfeweps);

  // variance:
  BOOST_CHECK_CLOSE_FRACTION(variance(dist),
    static_cast<RealType>(2.6666666666666666666666666666666666666666666666666666666667L), 1000*tolfeweps);
  // std deviation:
  BOOST_CHECK_CLOSE_FRACTION(standard_deviation(dist), 
    static_cast<RealType>(1.632993L), 1000 * tolerance);
  //// hazard:
  //BOOST_CHECK_CLOSE_FRACTION(hazard(dist, x),
  //  pdf(dist, x) / cdf(complement(dist, x)), tolerance);
  //// cumulative hazard:
  //BOOST_CHECK_CLOSE_FRACTION(chf(dist, x),
  //  -log(cdf(complement(dist, x))), tolerance);
  // coefficient_of_variation:
  BOOST_CHECK_CLOSE_FRACTION(coefficient_of_variation(dist),
    standard_deviation(dist) / mean(dist), tolerance);
  // mode:
  BOOST_CHECK_CLOSE_FRACTION(mode(dist),
    static_cast<RealType>(0.8284271L), tolerance);

  // median
  BOOST_CHECK_CLOSE_FRACTION(median(dist),
    static_cast<RealType>(1.5122506636053668L), tolerance);
  // Fails for real_concept - because std::numeric_limits<RealType>::digits = 0

  // skewness:
  BOOST_CHECK_CLOSE_FRACTION(skewness(dist),
    static_cast<RealType>(2.449490L), tolerance);
  // kurtosis:
  BOOST_CHECK_CLOSE_FRACTION(kurtosis(dist),
    static_cast<RealType>(10-3), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(kurtosis_excess(dist),
    static_cast<RealType>(10), tolerance);
} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{
  using boost::math::inverse_gaussian;
  using boost::math::inverse_gaussian_distribution;

  //int precision = 17; // std::numeric_limits<double::max_digits10;
  double tolfeweps = numeric_limits<double>::epsilon() * 5;
  //double tol6decdigits = numeric_limits<float>::epsilon() * 2;
  // Check that can generate inverse_gaussian distribution using the two convenience methods:
  boost::math::inverse_gaussian w12(1., 2); // Using typedef
  inverse_gaussian_distribution<> w23(2., 3); // Using default RealType double.
  boost::math::inverse_gaussian w11; // Use default unity values for mean and scale.
  // Note NOT myn01() as the compiler will interpret as a function!
  BOOST_CHECK_EQUAL(w11.mean(), 1);
  BOOST_CHECK_EQUAL(w11.scale(), 1);
  BOOST_CHECK_EQUAL(w23.mean(), 2);
  BOOST_CHECK_EQUAL(w23.scale(), 3);
  BOOST_CHECK_EQUAL(w23.shape(), 1.5L);

  // Check the synonyms, provided to allow generic use of find_location and find_scale.
  BOOST_CHECK_EQUAL(w11.mean(), w11.location());
  BOOST_CHECK_EQUAL(w11.scale(), w11.scale());

  BOOST_CHECK_CLOSE_FRACTION(mean(w11), static_cast<double>(1), tolfeweps); // Default mean == unity
  BOOST_CHECK_CLOSE_FRACTION(scale(w11), static_cast<double>(1), tolfeweps); // Default mean == unity

  // median
  // (test double because fails for real_concept because numeric_limits<real_concept>::digits = 0)
  BOOST_CHECK_CLOSE_FRACTION(median(w11),
    static_cast<double>(0.67584130569523893), tolfeweps);
  BOOST_CHECK_CLOSE_FRACTION(median(w23),
    static_cast<double>(1.5122506636053668), tolfeweps);
  
  // Initial spot tests using double values from R.
  // library(SuppDists)
  // formatC(SuppDists::dinverse_gaussian(1, 1, 1), digits=17) ...
  BOOST_CHECK_CLOSE_FRACTION( //  x = 1
    pdf(w11, 1.), static_cast<double>(0.3989422804014327), tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION( //  x = 1
    logpdf(w11, 1.), static_cast<double>(log(0.3989422804014327)), tolfeweps); // logpdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 1.), static_cast<double>(0.66810200122317065), 10 * tolfeweps); // cdf

  BOOST_CHECK_CLOSE_FRACTION(
    pdf(w11, 0.1), static_cast<double>(0.21979480031862672), tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w11, 0.1), static_cast<double>(log(0.21979480031862672)), tolfeweps); // logpdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 0.1), static_cast<double>(0.0040761113207110162), 10 * tolfeweps); // cdf

  BOOST_CHECK_CLOSE_FRACTION( // small x
    pdf(w11, 0.01), static_cast<double>(2.0811768202028392e-19), tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION( // small x
    logpdf(w11, 0.01), static_cast<double>(log(2.0811768202028392e-19)), tolfeweps); // logpdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 0.01), static_cast<double>(4.122313403318778e-23), 10 * tolfeweps); // cdf

  BOOST_CHECK_CLOSE_FRACTION( // smaller x
    pdf(w11, 0.001), static_cast<double>(2.4420044378793562e-213),  tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION( // smaller x
    logpdf(w11, 0.001), static_cast<double>(log(2.4420044378793562e-213)),  tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 0.001), static_cast<double>(4.8791443010851493e-219), 1000 * tolfeweps); // cdf
  // 4.8791443010859224e-219 versus 4.8791443010851493e-219 so still 14 decimal digits.

  BOOST_CHECK_CLOSE_FRACTION(
    quantile(w11, 0.66810200122317065), static_cast<double>(1.), 1 * tolfeweps); // cdf
  BOOST_CHECK_CLOSE_FRACTION(
    quantile(w11, 0.0040761113207110162), static_cast<double>(0.1), 1 * tolfeweps); // cdf
  BOOST_CHECK_CLOSE_FRACTION(
    quantile(w11, 4.122313403318778e-23), 0.01, 1 * tolfeweps); // quantile
  BOOST_CHECK_CLOSE_FRACTION(
    quantile(w11, 2.4420044378793562e-213), 0.001, 0.03); // quantile
  // quantile 0.001026926242348481 compared to expected 0.001, so much less accurate,
  // but better than R that gives up completely!
  // R Error in SuppDists::qinverse_gaussian(4.87914430108515e-219, 1, 1) : Infinite value in NewtonRoot()

  BOOST_CHECK_CLOSE_FRACTION(
    pdf(w11, 0.5), static_cast<double>(0.87878257893544476), tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w11, 0.5), static_cast<double>(log(0.87878257893544476)), tolfeweps); // logpdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 0.5), static_cast<double>(0.3649755481729598), tolfeweps); // cdf

  BOOST_CHECK_CLOSE_FRACTION(
    pdf(w11, 2), static_cast<double>(0.10984782236693059), tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w11, 2), static_cast<double>(log(0.10984782236693059)), tolfeweps); // logpdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 2), static_cast<double>(.88547542598600637), tolfeweps); // cdf

  BOOST_CHECK_CLOSE_FRACTION(
    pdf(w11, 10), static_cast<double>(0.00021979480031862676), tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w11, 10), static_cast<double>(log(0.00021979480031862676)), tolfeweps); // logpdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 10), static_cast<double>(0.99964958546279115), tolfeweps); // cdf

  BOOST_CHECK_CLOSE_FRACTION(
    pdf(w11, 100), static_cast<double>(2.0811768202028246e-25), tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w11, 100), static_cast<double>(log(2.0811768202028246e-25)), tolfeweps); // logpdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 100), static_cast<double>(1), tolfeweps); // cdf
  BOOST_CHECK_CLOSE_FRACTION(
    pdf(w11, 1000), static_cast<double>(2.4420044378793564e-222), 10 * tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w11, 1000), static_cast<double>(log(2.4420044378793564e-222)), 10 * tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 1000), static_cast<double>(1.), tolfeweps); // cdf

  // A few more misc tests, probably not very useful.  
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 1.), static_cast<double>(0.66810200122317065), tolfeweps); // cdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 0.1), static_cast<double>(0.0040761113207110162), tolfeweps * 5); // cdf
  // 0.0040761113207110162   0.0040761113207110362
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 0.2), static_cast<double>(0.063753567519976254), tolfeweps * 5); // cdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 0.5), static_cast<double>(0.3649755481729598), tolfeweps); // cdf

  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 0.9), static_cast<double>(0.62502320258649202), tolfeweps); // cdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 0.99), static_cast<double>(0.66408247396139031), tolfeweps); // cdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 0.999), static_cast<double>(0.66770275955311675), tolfeweps); // cdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 10.), static_cast<double>(0.99964958546279115), tolfeweps); // cdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w11, 50.), static_cast<double>(0.99999999999992029), tolfeweps); // cdf

  BOOST_CHECK_CLOSE_FRACTION(
    quantile(w11, 0.3649755481729598), static_cast<double>(0.5), tolfeweps); // quantile
  BOOST_CHECK_CLOSE_FRACTION(
    quantile(w11, 0.62502320258649202), static_cast<double>(0.9), tolfeweps); // quantile
  BOOST_CHECK_CLOSE_FRACTION(
    quantile(w11, 0.0040761113207110162), static_cast<double>(0.1), tolfeweps); // quantile

  // Wald(2,3) tests
  // ===================
  BOOST_CHECK_CLOSE_FRACTION( // formatC(SuppDists::dinvGauss(1, 2, 3), digits=17) "0.47490884963330904"
    pdf(w23, 1.), static_cast<double>(0.47490884963330904), tolfeweps ); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w23, 1.), static_cast<double>(log(0.47490884963330904)), tolfeweps ); // logpdf
  BOOST_CHECK_CLOSE_FRACTION(
    pdf(w23, 0.1), static_cast<double>(2.8854207087665401e-05), tolfeweps * 2); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w23, 0.1), static_cast<double>(log(2.8854207087665401e-05)), tolfeweps * 2); // logpdf
  //2.8854207087665452e-005 2.8854207087665401e-005
  BOOST_CHECK_CLOSE_FRACTION(
    pdf(w23, 10.), static_cast<double>(0.0019822751498574636), tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w23, 10.), static_cast<double>(log(0.0019822751498574636)), tolfeweps); // logpdf
  BOOST_CHECK_CLOSE_FRACTION(
    pdf(w23, 10.), static_cast<double>(0.0019822751498574636), tolfeweps); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w23, 10.), static_cast<double>(log(0.0019822751498574636)), tolfeweps); // logpdf

  // Bigger changes in mean and scale.

  inverse_gaussian w012(0.1, 2);
  BOOST_CHECK_CLOSE_FRACTION(
    pdf(w012, 1.), static_cast<double>(3.7460367141230404e-36), tolfeweps ); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w012, 1.), static_cast<double>(log(3.7460367141230404e-36)), tolfeweps ); // logpdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w012, 1.), static_cast<double>(1), tolfeweps ); // pdf

  inverse_gaussian w0110(0.1, 10);
  BOOST_CHECK_CLOSE_FRACTION(
    pdf(w0110, 1.), static_cast<double>(1.6279643678071011e-176), 100 * tolfeweps ); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w0110, 1.), static_cast<double>(log(1.6279643678071011e-176)), 100 * tolfeweps ); // logpdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w0110, 1.), static_cast<double>(1), tolfeweps ); // cdf
  BOOST_CHECK_CLOSE_FRACTION(
     cdf(complement(w0110, 1.)), static_cast<double>(3.2787685715328683e-179), 1e6 * tolfeweps ); // cdf complement
  // Differs because of loss of accuracy.

  BOOST_CHECK_CLOSE_FRACTION(
    pdf(w0110, 0.1), static_cast<double>(39.894228040143268), tolfeweps ); // pdf
  BOOST_CHECK_CLOSE_FRACTION(
    logpdf(w0110, 0.1), static_cast<double>(log(39.894228040143268)), tolfeweps ); // logpdf
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(w0110, 0.1), static_cast<double>(0.51989761564832704), 10 * tolfeweps ); // cdf

    // Basic sanity-check spot values for all floating-point types..
  // (Parameter value, arbitrarily zero, only communicates the floating point type).
  test_spots(0.0F); // Test float. OK at decdigits = 0 tolerance = 0.0001 %
  test_spots(0.0); // Test double. OK at decdigits 7, tolerance = 1e07 %
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  test_spots(0.0L); // Test long double.
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
  test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#else
  std::cout << "<note>The long double tests have been disabled on this platform "
    "either because the long double overloads of the usual math functions are "
    "not available at all, or because they are too inaccurate for these tests "
    "to pass.</note>" << std::endl;
#endif
  /*      */
  
} // BOOST_AUTO_TEST_CASE( test_main )

/*

Output:


*/


