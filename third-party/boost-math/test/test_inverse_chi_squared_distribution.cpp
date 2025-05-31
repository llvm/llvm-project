// test_inverse_chi_squared.cpp

// Copyright Paul A. Bristow 2010.
// Copyright John Maddock 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4310) // cast truncates constant value.
#endif

// http://www.wolframalpha.com/input/?i=inverse+chisquare+distribution

#include <boost/math/tools/config.hpp>
#include "../include_private/boost/math/tools/test.hpp"

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp> // for real_concept
using ::boost::math::concepts::real_concept;
#endif

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE_FRACTION
#include "test_out_of_range.hpp"

#include <boost/math/distributions/inverse_chi_squared.hpp> // for inverse_chisquared_distribution
using boost::math::inverse_chi_squared_distribution;
using boost::math::cdf;
using boost::math::pdf;

// Use Inverse Gamma distribution to check their relationship:
// inverse_chi_squared<>(v) == inverse_gamma<>(v / 2., 0.5)
#include <boost/math/distributions/inverse_gamma.hpp> // for inverse_gamma_distribution
using boost::math::inverse_gamma_distribution;
using boost::math::inverse_gamma;
//  using  ::boost::math::cdf;
//  using  ::boost::math::pdf;

#include <boost/math/special_functions/gamma.hpp> 
using boost::math::tgamma; // for naive pdf.

#include <iostream>
using std::cout;
using std::endl;
#include <limits>
using std::numeric_limits; // for epsilon.

template <class RealType>
RealType naive_pdf(RealType df, RealType scale, RealType x)
{ // Formula from Wikipedia
   using namespace std; // For ADL of std functions.
   using boost::math::tgamma;
   RealType result = pow(scale * df/2, df/2) * exp(-df * scale/(2 * x));
   result /= tgamma(df/2) * pow(x, 1 + df/2);
   return result;
}

// Test using a spot value from some other reference source,
// in this case test values from output from R provided by Thomas Mang,
// and Wolfram Mathematica by Mark Coleman.

template <class RealType>
void test_spot(
     RealType degrees_of_freedom, // degrees_of_freedom,
     RealType scale, // scale,
     RealType x, // random variate x,
     RealType pd, // expected pdf,
     RealType P, // expected CDF,
     RealType Q, // expected complement of CDF,
     RealType tol) // test tolerance.
{
   boost::math::inverse_chi_squared_distribution<RealType> dist(degrees_of_freedom, scale);

   BOOST_CHECK_CLOSE_FRACTION
      ( // Compare to expected PDF.
      pdf(dist, x), // calculated.
      pd, // expected
      tol);

   BOOST_CHECK_CLOSE_FRACTION( // Compare to naive pdf formula (probably less accurate).
      pdf(dist, x), naive_pdf(dist.degrees_of_freedom(), dist.scale(), x), tol);

   BOOST_CHECK_CLOSE_FRACTION( // Compare to expected CDF.
      cdf(dist, x), P, tol);

   if((P < 0.999) && (Q < 0.999))
   {  // We can only check this if P is not too close to 1,
      // so that we can guarantee Q is accurate:
      BOOST_CHECK_CLOSE_FRACTION(
        cdf(complement(dist, x)), Q, tol); // 1 - cdf
      BOOST_CHECK_CLOSE_FRACTION(
        quantile(dist, P), x, tol); // quantile(cdf) = x
      BOOST_CHECK_CLOSE_FRACTION(
        quantile(complement(dist, Q)), x, tol); // quantile(complement(1 - cdf)) = x
   }
} // test_spot

template <class RealType> // Any floating-point type RealType.
void test_spots(RealType)
{
  // Basic sanity checks, some test data is to six decimal places only,
  // so set tolerance to 0.000001 (expressed as a percentage = 0.0001%).

  RealType tolerance = 0.000001f;
  cout << "Tolerance = " << tolerance * 100 << "%." << endl;

// This test values from output from geoR (17 decimal digits) guided by Thomas Mang.
  test_spot(static_cast<RealType>(2), static_cast<RealType>(1./2.),
    // degrees_of_freedom, default scale = 1/df.
  static_cast<RealType>(1.L), // x.
  static_cast<RealType>(0.30326532985631671L), // pdf.
  static_cast<RealType>(0.60653065971263365L), // cdf.
  static_cast<RealType>(1 - 0.606530659712633657L), // cdf complement.
  tolerance  // tol
  );

// Tests from Mark Coleman & Georgi Boshnakov using Wolfram Mathematica.
  test_spot(static_cast<RealType>(10), static_cast<RealType>(0.1L), // degrees_of_freedom, scale
  static_cast<RealType>(0.2), // x
  static_cast<RealType>(1.6700235722635659824529759616528281217001163943570L), // pdf
  static_cast<RealType>(0.89117801891415124234834646836872197623907651175353L), // cdf
  static_cast<RealType>(1 - 0.89117801891415127L), // cdf complement
  tolerance  // tol
  );

  test_spot(static_cast<RealType>(10), static_cast<RealType>(0.1L), // degrees_of_freedom, scale
  static_cast<RealType>(0.5), // x
  static_cast<RealType>(0.03065662009762021L), // pdf
  static_cast<RealType>(0.99634015317265628765454354418728984933240514654437L), // cdf
  static_cast<RealType>(1 - 0.99634015317265628765454354418728984933240514654437L), // cdf complement
  tolerance  // tol
  );


  test_spot(static_cast<RealType>(10), static_cast<RealType>(2), // degrees_of_freedom, scale
  static_cast<RealType>(0.5), // x
  static_cast<RealType>(0.00054964096598361569L), // pdf
  static_cast<RealType>(0.000016944743930067383903707995865261004246785511612700L), // cdf
  static_cast<RealType>(1 - 0.000016944743930067383903707995865261004246785511612700L), // cdf complement
  tolerance  // tol
  );
  
  // Check some bad parameters to the distribution cause expected exception to be thrown.
#ifndef BOOST_NO_EXCEPTIONS
  BOOST_MATH_CHECK_THROW(boost::math::inverse_chi_squared_distribution<RealType> ichsqbad1(-1), std::domain_error); // negative degrees_of_freedom.
  BOOST_MATH_CHECK_THROW(boost::math::inverse_chi_squared_distribution<RealType> ichsqbad2(1, -1), std::domain_error); // negative scale.
  BOOST_MATH_CHECK_THROW(boost::math::inverse_chi_squared_distribution<RealType> ichsqbad3(-1, -1), std::domain_error); // negative scale and degrees_of_freedom.
#else
  BOOST_MATH_CHECK_THROW(boost::math::inverse_chi_squared_distribution<RealType>(-1), std::domain_error); // negative degrees_of_freedom.
  BOOST_MATH_CHECK_THROW(boost::math::inverse_chi_squared_distribution<RealType>(1, -1), std::domain_error); // negative scale.
  BOOST_MATH_CHECK_THROW(boost::math::inverse_chi_squared_distribution<RealType>(-1, -1), std::domain_error); // negative scale and degrees_of_freedom.
#endif
  check_out_of_range<boost::math::inverse_chi_squared_distribution<RealType> >(1, 1);

  inverse_chi_squared_distribution<RealType> ichsq;

  if(std::numeric_limits<RealType>::has_infinity)
  {
    BOOST_MATH_CHECK_THROW(pdf(ichsq, +std::numeric_limits<RealType>::infinity()), std::domain_error); // x = + infinity, pdf = 0
    BOOST_MATH_CHECK_THROW(pdf(ichsq, -std::numeric_limits<RealType>::infinity()),  std::domain_error); // x = - infinity, pdf = 0
    BOOST_MATH_CHECK_THROW(cdf(ichsq, +std::numeric_limits<RealType>::infinity()),std::domain_error ); // x = + infinity, cdf = 1
    BOOST_MATH_CHECK_THROW(cdf(ichsq, -std::numeric_limits<RealType>::infinity()), std::domain_error); // x = - infinity, cdf = 0
    BOOST_MATH_CHECK_THROW(cdf(complement(ichsq, +std::numeric_limits<RealType>::infinity())), std::domain_error); // x = + infinity, c cdf = 0
    BOOST_MATH_CHECK_THROW(cdf(complement(ichsq, -std::numeric_limits<RealType>::infinity())), std::domain_error); // x = - infinity, c cdf = 1
#ifndef BOOST_NO_EXCEPTIONS
    BOOST_MATH_CHECK_THROW(boost::math::inverse_chi_squared_distribution<RealType> nbad1(std::numeric_limits<RealType>::infinity(), static_cast<RealType>(1)), std::domain_error); // +infinite mean
    BOOST_MATH_CHECK_THROW(boost::math::inverse_chi_squared_distribution<RealType> nbad1(-std::numeric_limits<RealType>::infinity(),  static_cast<RealType>(1)), std::domain_error); // -infinite mean
    BOOST_MATH_CHECK_THROW(boost::math::inverse_chi_squared_distribution<RealType> nbad1(static_cast<RealType>(0), std::numeric_limits<RealType>::infinity()), std::domain_error); // infinite sd
#else
    BOOST_MATH_CHECK_THROW(boost::math::inverse_chi_squared_distribution<RealType>(std::numeric_limits<RealType>::infinity(), static_cast<RealType>(1)), std::domain_error); // +infinite mean
    BOOST_MATH_CHECK_THROW(boost::math::inverse_chi_squared_distribution<RealType>(-std::numeric_limits<RealType>::infinity(),  static_cast<RealType>(1)), std::domain_error); // -infinite mean
    BOOST_MATH_CHECK_THROW(boost::math::inverse_chi_squared_distribution<RealType>(static_cast<RealType>(0), std::numeric_limits<RealType>::infinity()), std::domain_error); // infinite sd
#endif
  }

  if (std::numeric_limits<RealType>::has_quiet_NaN)
  { // If no longer allow x or p to be NaN, then these tests should throw.
    BOOST_MATH_CHECK_THROW(pdf(ichsq, +std::numeric_limits<RealType>::quiet_NaN()), std::domain_error); // x = NaN
    BOOST_MATH_CHECK_THROW(cdf(ichsq, +std::numeric_limits<RealType>::quiet_NaN()), std::domain_error); // x = NaN
    BOOST_MATH_CHECK_THROW(cdf(complement(ichsq, +std::numeric_limits<RealType>::quiet_NaN())), std::domain_error); // x = + infinity
    BOOST_MATH_CHECK_THROW(quantile(ichsq, std::numeric_limits<RealType>::quiet_NaN()), std::domain_error); // p = + quiet_NaN
    BOOST_MATH_CHECK_THROW(quantile(complement(ichsq, std::numeric_limits<RealType>::quiet_NaN())), std::domain_error); // p = + quiet_NaN
  }
    // Spot check for pdf using 'naive pdf' function
  for(RealType x = 0.5; x < 5; x += 0.5)
  {
    BOOST_CHECK_CLOSE_FRACTION(
      pdf(inverse_chi_squared_distribution<RealType>(5, 6), x),
      naive_pdf(RealType(5), RealType(6), x),
      tolerance);
  }   // Spot checks for parameters:

  RealType tol_2eps = boost::math::tools::epsilon<RealType>() * 2; // 2 eps as a fraction.
  inverse_chi_squared_distribution<RealType> dist51(5, 1);
  inverse_chi_squared_distribution<RealType> dist52(5, 2);
  inverse_chi_squared_distribution<RealType> dist31(3, 1);
  inverse_chi_squared_distribution<RealType> dist111(11, 1);
  // 11 mean 0.10000000000000001, variance  0.0011111111111111111, sd 0.033333333333333333

  using namespace std; // ADL of std names.
  using namespace boost::math;
  
  inverse_chi_squared_distribution<RealType> dist10(10);
  //  mean, variance etc
  BOOST_CHECK_CLOSE_FRACTION(mean(dist10), static_cast<RealType>(0.125), tol_2eps);
  BOOST_CHECK_CLOSE_FRACTION(variance(dist10), static_cast<RealType>(0.0052083333333333333333333333333333333333333333333333L), tol_2eps);
  BOOST_CHECK_CLOSE_FRACTION(mode(dist10), static_cast<RealType>(0.08333333333333333333333333333333333333333333333L), tol_2eps);
  BOOST_CHECK_CLOSE_FRACTION(median(dist10), static_cast<RealType>(0.10704554778227709530244586234274024205738435512468L), tol_2eps);
  BOOST_CHECK_CLOSE_FRACTION(cdf(dist10, median(dist10)), static_cast<RealType>(0.5L), 4 * tol_2eps);
  BOOST_CHECK_CLOSE_FRACTION(skewness(dist10), static_cast<RealType>(3.4641016151377545870548926830117447338856105076208L), tol_2eps);
  BOOST_CHECK_CLOSE_FRACTION(kurtosis(dist10), static_cast<RealType>(45), tol_2eps);
  BOOST_CHECK_CLOSE_FRACTION(kurtosis_excess(dist10), static_cast<RealType>(45-3), tol_2eps);

  tol_2eps = boost::math::tools::epsilon<RealType>() * 2; // 2 eps as a percentage.

  // Special and limit cases:

  RealType mx = (std::numeric_limits<RealType>::max)();
  RealType mi = (std::numeric_limits<RealType>::min)();

  BOOST_CHECK_EQUAL(
  pdf(inverse_chi_squared_distribution<RealType>(1),
    static_cast<RealType>(mx)), // max()
    static_cast<RealType>(0)
    );

  BOOST_CHECK_EQUAL(
  pdf(inverse_chi_squared_distribution<RealType>(1),
    static_cast<RealType>(mi)), // min()
    static_cast<RealType>(0)
    );

  BOOST_CHECK_EQUAL(
    pdf(inverse_chi_squared_distribution<RealType>(1), static_cast<RealType>(0)), static_cast<RealType>(0));
  BOOST_CHECK_EQUAL(
    pdf(inverse_chi_squared_distribution<RealType>(3), static_cast<RealType>(0))
    , static_cast<RealType>(0.0f));
  BOOST_CHECK_EQUAL(
    cdf(inverse_chi_squared_distribution<RealType>(1), static_cast<RealType>(0))
    , static_cast<RealType>(0.0f));
  BOOST_CHECK_EQUAL(
    cdf(inverse_chi_squared_distribution<RealType>(2), static_cast<RealType>(0))
    , static_cast<RealType>(0.0f));
  BOOST_CHECK_EQUAL(
    cdf(inverse_chi_squared_distribution<RealType>(3L), static_cast<RealType>(0L))
    , static_cast<RealType>(0));
  BOOST_CHECK_EQUAL(
    cdf(complement(inverse_chi_squared_distribution<RealType>(1), static_cast<RealType>(0)))
    , static_cast<RealType>(1));
  BOOST_CHECK_EQUAL(
    cdf(complement(inverse_chi_squared_distribution<RealType>(2), static_cast<RealType>(0)))
    , static_cast<RealType>(1));
  BOOST_CHECK_EQUAL(
    cdf(complement(inverse_chi_squared_distribution<RealType>(3), static_cast<RealType>(0)))
    , static_cast<RealType>(1));

  BOOST_MATH_CHECK_THROW(
    pdf(
    inverse_chi_squared_distribution<RealType>(static_cast<RealType>(-1)), // degrees_of_freedom negative.
    static_cast<RealType>(1)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    pdf(
    inverse_chi_squared_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(-1)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    cdf(
    inverse_chi_squared_distribution<RealType>(static_cast<RealType>(-1)),
    static_cast<RealType>(1)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    cdf(
    inverse_chi_squared_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(-1)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    cdf(complement(
    inverse_chi_squared_distribution<RealType>(static_cast<RealType>(-1)),
    static_cast<RealType>(1))), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    cdf(complement(
    inverse_chi_squared_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(-1))), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    quantile(
    inverse_chi_squared_distribution<RealType>(static_cast<RealType>(-1)),
    static_cast<RealType>(0.5)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    quantile(
    inverse_chi_squared_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(-1)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    quantile(
    inverse_chi_squared_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(1.1)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    quantile(complement(
    inverse_chi_squared_distribution<RealType>(static_cast<RealType>(-1)),
    static_cast<RealType>(0.5))), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    quantile(complement(
    inverse_chi_squared_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(-1))), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    quantile(complement(
    inverse_chi_squared_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(1.1))), std::domain_error
    );
} // template <class RealType>void test_spots(RealType)


BOOST_AUTO_TEST_CASE( test_main )
{
  BOOST_MATH_CONTROL_FP;

  double tol_few_eps = numeric_limits<double>::epsilon() * 4;
  
  // Check that can generate inverse_chi_squared distribution using the two convenience methods:
  // inverse_chi_squared_distribution; // with default parameters, degrees_of_freedom = 1, scale - 1
  using boost::math::inverse_chi_squared;
   
  // Some constructor tests using default double.
  double tol4eps = boost::math::tools::epsilon<double>() * 4; // 4 eps as a fraction.

  inverse_chi_squared ichsqdef; // Using typedef and both default parameters.

  BOOST_CHECK_EQUAL(ichsqdef.degrees_of_freedom(), 1.); // df == 1
  BOOST_CHECK_EQUAL(ichsqdef.scale(), 1); // scale == 1./df
  BOOST_CHECK_CLOSE_FRACTION(pdf(ichsqdef, 1), 0.24197072451914330, tol4eps);
  BOOST_CHECK_CLOSE_FRACTION(pdf(ichsqdef, 9), 0.013977156581221969, tol4eps);
  
  inverse_chi_squared_distribution<double> ichisq102(10., 2); // Both parameters specified.
  BOOST_CHECK_EQUAL(ichisq102.degrees_of_freedom(), 10.); // Check both parameters stored OK.
  BOOST_CHECK_EQUAL(ichisq102.scale(), 2.); // Check both parameters stored OK.

  inverse_chi_squared_distribution<double> ichisq10(10.); // Only df parameter specified (unscaled).
  BOOST_CHECK_EQUAL(ichisq10.degrees_of_freedom(), 10.); // Check  parameter stored.
  BOOST_CHECK_EQUAL(ichisq10.scale(), 0.1); // Check default scale = 1/df = 1/10 = 0.1
  BOOST_CHECK_CLOSE_FRACTION(pdf(ichisq10, 1),  0.00078975346316749169, tol4eps);
  BOOST_CHECK_CLOSE_FRACTION(pdf(ichisq10, 10), 0.0000000012385799798186384, tol4eps);

  BOOST_CHECK_CLOSE_FRACTION(mode(ichisq10), 0.0833333333333333333333333333333333333333, tol4eps);
  // nu * xi / nu + 2 = 10 * 0.1 / (10 + 2) = 1/12 =  0.0833333...
  // mode is not defined in Mathematica.
  // See Discussion section http://en.wikipedia.org/wiki/Talk:Scaled-inverse-chi-square_distribution
  // for origin of this formula.

  inverse_chi_squared_distribution<double> ichisq5(5.); // // Only df parameter specified.
  BOOST_CHECK_EQUAL(ichisq5.degrees_of_freedom(), 5.); // check  parameter stored.
  BOOST_CHECK_EQUAL(ichisq5.scale(), 1./5.); // check default is 1/df
  BOOST_CHECK_CLOSE_FRACTION(pdf(ichisq5, 0.2), 3.0510380337346841, tol4eps);
  BOOST_CHECK_CLOSE_FRACTION(cdf(ichisq5, 0.5), 0.84914503608460956, tol4eps);
  BOOST_CHECK_CLOSE_FRACTION(cdf(complement(ichisq5, 0.5)), 1 - 0.84914503608460956, tol4eps);

  BOOST_CHECK_CLOSE_FRACTION(quantile(ichisq5, 0.84914503608460956), 0.5, tol4eps*100);
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(ichisq5, 1. - 0.84914503608460956)), 0.5, tol4eps*100);

  // Check mean, etc spot values.
  inverse_chi_squared_distribution<double> ichisq81(8., 1.); // degrees_of_freedom = 5, scale = 1
  BOOST_CHECK_CLOSE_FRACTION(mean(ichisq81),1.33333333333333333333333333333333333333333, tol4eps);
  BOOST_CHECK_CLOSE_FRACTION(variance(ichisq81), 0.888888888888888888888888888888888888888888888, tol4eps);
  BOOST_CHECK_CLOSE_FRACTION(skewness(ichisq81), 2 * std::sqrt(8.), tol4eps);
  inverse_chi_squared_distribution<double> ichisq21(2., 1.);
  BOOST_CHECK_CLOSE_FRACTION(mode(ichisq21), 0.5, tol4eps);
  BOOST_CHECK_CLOSE_FRACTION(median(ichisq21), 1.4426950408889634, tol4eps);

  inverse_chi_squared ichsq4(4.); // Using typedef and degrees_of_freedom parameter (and default scale = 1/df).
  BOOST_CHECK_EQUAL(ichsq4.degrees_of_freedom(), 4.); // df == 4.
  BOOST_CHECK_EQUAL(ichsq4.scale(), 0.25); // scale  == 1 /df == 1/4.

  inverse_chi_squared ichsq32(3, 2);
  BOOST_CHECK_EQUAL(ichsq32.degrees_of_freedom(), 3.); // df == 3.
  BOOST_CHECK_EQUAL(ichsq32.scale(), 2); // scale  == 2
  
  inverse_chi_squared ichsq11(1, 1); // Using explicit degrees_of_freedom parameter, and default scale = 1).
  BOOST_CHECK_CLOSE_FRACTION(mode(ichsq11), 0.3333333333333333333333333333333333333333, tol4eps);
  // (1 * 1)/ (1 + 2) = 1/3 using Wikipedia nu * xi /(nu + 2)
  BOOST_CHECK_EQUAL(ichsq11.degrees_of_freedom(), 1.); // df == 1 (default).
  BOOST_CHECK_EQUAL(ichsq11.scale(), 1.); // scale == 1.
  /*
  // Used to find some 'exact' values for testing mean, variance ...
  // First with scale fixed at unity (Wikipedia definition 1)
  cout << "df      scale            mean            variance              sd              median" << endl;
  for (int degrees_of_freedom = 8; degrees_of_freedom < 30; degrees_of_freedom++)
  {
    inverse_chi_squared ichisq(degrees_of_freedom, 1);
    cout.precision(17);
    cout << degrees_of_freedom << "    "  << 1 << "  " << mean(ichisq) << ' ' 
      << variance(ichisq) << ' ' << standard_deviation(ichisq)
      << ' ' << median(ichisq) << endl;
  }

  // Default scale = 1 / df
  cout << "|\n" << "df           scale          mean            variance              sd              median" << endl;
  for (int degrees_of_freedom = 8; degrees_of_freedom < 30; degrees_of_freedom++)
  {
    inverse_chi_squared ichisq(degrees_of_freedom);
    cout.precision(17);
    cout << degrees_of_freedom << "    "  << 1./degrees_of_freedom << "  " << mean(ichisq) << ' ' 
      << variance(ichisq) << ' ' << standard_deviation(ichisq)
      << ' ' << median(ichisq) << endl;
  }
  */
  inverse_chi_squared_distribution<> ichisq14(14, 1); // Using default RealType double.
  BOOST_CHECK_CLOSE_FRACTION(mean(ichisq14), 1.166666666666666666666666666666666666666666666, tol4eps);
  BOOST_CHECK_CLOSE_FRACTION(variance(ichisq14), 0.272222222222222222222222222222222222222222222, tol4eps);

  inverse_chi_squared_distribution<> ichisq121(12); // Using default RealType double.
  BOOST_CHECK_CLOSE_FRACTION(mean(ichisq121),  0.1, tol4eps);
  BOOST_CHECK_CLOSE_FRACTION(variance(ichisq121), 0.0025, tol4eps);
  BOOST_CHECK_CLOSE_FRACTION(standard_deviation(ichisq121), 0.05, tol4eps);

  // and "using boost::math::inverse_chi_squared_distribution;".
  inverse_chi_squared_distribution<> ichsq23(2., 3.); // Using default RealType double.
  BOOST_CHECK_EQUAL(ichsq23.degrees_of_freedom(), 2.); //
  BOOST_CHECK_EQUAL(ichsq23.scale(), 3.); //
  BOOST_MATH_CHECK_THROW(mean(ichsq23), std::domain_error); // Degrees of freedom (nu) must be > 2
  BOOST_MATH_CHECK_THROW(variance(ichsq23), std::domain_error); // Degrees of freedom (nu) must be > 4
  BOOST_MATH_CHECK_THROW(skewness(ichsq23), std::domain_error); // Degrees of freedom (nu) must be > 6
  BOOST_MATH_CHECK_THROW(kurtosis_excess(ichsq23), std::domain_error); // Degrees of freedom (nu) must be > 8

  { // Check relationship between inverse gamma and inverse chi_squared distributions.
  using boost::math::inverse_gamma_distribution;

  double df = 2.;
  double scale = 1.;
  double alpha = df/2; // aka inv_gamma shape
  double beta = scale /2; // inv_gamma scale.
 
  inverse_gamma_distribution<> ig(alpha, beta); 

  inverse_chi_squared_distribution<> ichsq(df, 1./df); // == default scale.
  BOOST_CHECK_EQUAL(pdf(ichsq, 0), 0); // Special case of zero x.

  double x = 0.5;
  BOOST_CHECK_EQUAL(pdf(ig, x), pdf(ichsq, x)); // inv_gamma compared to inv_chisq
  BOOST_CHECK_EQUAL(cdf(ichsq, 0), 0); // Special case of zero.
  BOOST_CHECK_EQUAL(cdf(ig, x), cdf(ichsq, x)); // invgamma == invchisq

  // Test pdf by comparing using naive_pdf with relation to inverse gamma distribution
  // wikipedia http://en.wikipedia.org/wiki/Scaled-inverse-chi-square_distribution related distributions.
  // So if naive_pdf is correct, inverse_chi_squared_distribution should agree.
  df = 1.; scale = 1.;
  BOOST_CHECK_CLOSE_FRACTION(naive_pdf(df, scale, x), pdf(ichsq11, x), tol_few_eps);

  //inverse_gamma_distribution<> igd(df/2, (df * scale)/2); 
  inverse_gamma_distribution<> igd11(df/2, df * scale/2);
  BOOST_CHECK_CLOSE_FRACTION(naive_pdf(df, scale, x), pdf(igd11, x), tol_few_eps);
  BOOST_CHECK_CLOSE_FRACTION(naive_pdf(df, scale, x), pdf(ichsq11, x), tol_few_eps);

  df = 2; scale = 1;
  inverse_gamma_distribution<> igd21(df/2, df * scale/2);
  inverse_chi_squared_distribution<> ichsq21(df, scale);
  BOOST_CHECK_CLOSE_FRACTION(naive_pdf(df, scale, x), pdf(igd21, x), tol_few_eps); // 0.54134113294645081 OK 
  BOOST_CHECK_CLOSE_FRACTION(naive_pdf(df, scale, x), pdf(ichsq21, x), tol_few_eps); 

  df = 2; scale = 2;
  inverse_gamma_distribution<> igd22(df/2, df * scale/2);
  inverse_chi_squared_distribution<> ichsq22(df, scale);
  BOOST_CHECK_CLOSE_FRACTION(naive_pdf(df, scale, x), pdf(igd22, x), tol_few_eps);
  BOOST_CHECK_CLOSE_FRACTION(naive_pdf(df, scale, x), pdf(ichsq22, x), tol_few_eps);
  }

  // Check using float.
  inverse_chi_squared_distribution<float> igf23(1.f, 2.f); // Using explicit RealType float.
  BOOST_CHECK_EQUAL(igf23.degrees_of_freedom(), 1.f); //
  BOOST_CHECK_EQUAL(igf23.scale(), 2.f); //
  
  // Check throws from bad parameters.
  inverse_chi_squared ig051(0.5, 1.); // degrees_of_freedom < 1, so wrong for mean.
  BOOST_MATH_CHECK_THROW(mean(ig051), std::domain_error);
  inverse_chi_squared ig191(1.9999, 1.); // degrees_of_freedom < 2, so wrong for variance.
  BOOST_MATH_CHECK_THROW(variance(ig191), std::domain_error);
  inverse_chi_squared ig291(2.9999, 1.); // degrees_of_freedom < 3, so wrong for skewness.
  BOOST_MATH_CHECK_THROW(skewness(ig291), std::domain_error);
  inverse_chi_squared ig391(3.9999, 1.); // degrees_of_freedom < 1, so wrong for kurtosis and kurtosis_excess.
  BOOST_MATH_CHECK_THROW(kurtosis(ig391), std::domain_error);
  BOOST_MATH_CHECK_THROW(kurtosis_excess(ig391), std::domain_error);
  
  inverse_chi_squared ig102(10, 2); // Wolfram.com/ page 2, quantile = 2.96859.
  //http://reference.wolfram.com/mathematica/ref/InverseChiSquareDistribution.html
  BOOST_CHECK_CLOSE_FRACTION(quantile(ig102, 0.75), 2.96859, 0.000001); 
  BOOST_CHECK_CLOSE_FRACTION(cdf(ig102, 2.96859), 0.75 , 0.000001); 
  BOOST_CHECK_CLOSE_FRACTION(cdf(complement(ig102, 2.96859)), 1 - 0.75 , 0.00001); 
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(ig102, 1 - 0.75)), 2.96859, 0.000001); 
 
  // Basic sanity-check spot values.
  // (Parameter value, arbitrarily zero, only communicates the floating point type).
  test_spots(0.0F); // Test float. OK at decdigits = 0 tolerance = 0.0001 %
  test_spots(0.0); // Test double. OK at decdigits 7, tolerance = 1e07 %
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  test_spots(0.0L); // Test long double.
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x0582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
  test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#else
  std::cout << "<note>The long double tests have been disabled on this platform "
    "either because the long double overloads of the usual math functions are "
    "not available at all, or because they are too inaccurate for these tests "
    "to pass.</note>" << std::endl;
#endif

 /*    */
  
} // BOOST_AUTO_TEST_CASE( test_main )

/*

Output:




*/



