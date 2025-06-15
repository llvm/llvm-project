// test_nc_t.cpp

// Copyright John Maddock 2008, 2012.
// Copyright Paul A. Bristow 2012.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch.hpp> // Need to include lib/math/test in path.

#ifdef _MSC_VER
#pragma warning (disable:4127 4512)
#endif

#if !defined(TEST_FLOAT) && !defined(TEST_DOUBLE) && !defined(TEST_LDOUBLE) && !defined(TEST_REAL_CONCEPT)
#  define TEST_FLOAT
#  define TEST_DOUBLE
#  define TEST_LDOUBLE
#  define TEST_REAL_CONCEPT
#endif

#include <boost/math/tools/test.hpp>
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#include <boost/math/distributions/non_central_t.hpp> // for chi_squared_distribution.
#include <boost/math/distributions/normal.hpp> // for normal distribution (for comparison).

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE

#include "functor.hpp"
#include "handle_test_result.hpp"
#include "table_type.hpp"
#include "test_nc_t.hpp"

#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
#include <limits>
using std::numeric_limits;


void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //
   const char* largest_type;
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   if(boost::math::policies::digits<double, boost::math::policies::policy<> >() == boost::math::policies::digits<long double, boost::math::policies::policy<> >())
   {
      largest_type = "(long\\s+)?double|real_concept";
   }
   else
   {
      largest_type = "long double|real_concept";
   }
#else
   largest_type = "(long\\s+)?double|real_concept";
#endif

   //
   // Catch all cases come last:
   //
   if(std::numeric_limits<long double>::digits > 54)
   {
      add_expected_result(
         "[^|]*",                          // compiler
         "[^|]*",                          // stdlib
         "[^|]*",                          // platform
         largest_type,                     // test type(s)
         "[^|]*large[^|]*",                // test data group
         "[^|]*", 2000000, 200000);        // test function
      add_expected_result(
         "[^|]*",                          // compiler
         "[^|]*",                          // stdlib
         "[^|]*",                          // platform
         "double",                         // test type(s)
         "[^|]*large[^|]*",                // test data group
         "[^|]*", 500, 100);               // test function
   }
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "real_concept",                   // test type(s)
      "[^|]*",                          // test data group
      "[^|]*", 300000, 100000);                // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*large[^|]*",                // test data group
      "[^|]*", 1500, 300);              // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*small[^|]*",                // test data group
      "[^|]*", 400, 100);              // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      ".*Solaris.*",                    // platform
      largest_type,                     // test type(s)
      "[^|]*",                          // test data group
      "[^|]*", 400, 100);               // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "double",                         // test type(s)
      "[^|]*PDF",                  // test data group
      "[^|]*", static_cast<std::uintmax_t>(1 / boost::math::tools::root_epsilon<double>()), static_cast<std::uintmax_t>(1 / boost::math::tools::root_epsilon<double>())); // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "long double",                         // test type(s)
      "[^|]*PDF",                  // test data group
      "[^|]*", static_cast<std::uintmax_t>(1 / boost::math::tools::root_epsilon<long double>()), static_cast<std::uintmax_t>(1 / boost::math::tools::root_epsilon<long double>())); // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*",                          // test data group
      "[^|]*", 250, 50);                // test function

   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", " 
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}


BOOST_AUTO_TEST_CASE( test_main )
{
  BOOST_MATH_CONTROL_FP;
   // Basic sanity-check spot values.
   expected_results();

   // (Parameter value, arbitrarily zero, only communicates the floating point type).
#ifdef TEST_FLOAT
   test_spots(0.0F); // Test float.
#endif
#ifdef TEST_DOUBLE
   test_spots(0.0); // Test double.
#endif
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#ifdef TEST_LDOUBLE
   test_spots(0.0L); // Test long double.
#endif
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#ifdef TEST_REAL_CONCEPT
   test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#endif
#endif
  
#ifdef TEST_FLOAT
   test_accuracy(0.0F, "float"); // Test float.
   test_big_df(0.F); // float
#endif
#ifdef TEST_DOUBLE
   test_accuracy(0.0, "double"); // Test double.
   test_big_df(0.); // double
   test_ignore_policy(0.0);
#endif
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#ifdef TEST_LDOUBLE
   test_accuracy(0.0L, "long double"); // Test long double.
#endif
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#ifdef TEST_REAL_CONCEPT
   test_accuracy(boost::math::concepts::real_concept(0.), "real_concept"); // Test real concept.
#endif
#endif
#endif
  /* */

   
} // BOOST_AUTO_TEST_CASE( test_main )

/*

Output:

  Description: Autorun "J:\Cpp\MathToolkit\test\Math_test\Debug\test_nc_t.exe"
  Running 1 test case...
  Tests run with Microsoft Visual C++ version 10.0, Dinkumware standard library version 520, Win32
  Tolerance = 0.000596046%.
  Tolerance = 5e-010%.
  Tolerance = 5e-010%.
  Tolerance = 1e-008%.
  Testing: Non Central T
  CDF<float> Max = 0 RMS Mean=0
  
  CCDF<float> Max = 0 RMS Mean=0
  
  
  Testing: float quantile sanity check, with tests Non Central T
  Testing: Non Central T (small non-centrality)
  CDF<float> Max = 0 RMS Mean=0
  
  CCDF<float> Max = 0 RMS Mean=0
  
  
  Testing: float quantile sanity check, with tests Non Central T (small non-centrality)
  Testing: Non Central T (large parameters)
  CDF<float> Max = 0 RMS Mean=0
  
  CCDF<float> Max = 0 RMS Mean=0
  
  
  Testing: float quantile sanity check, with tests Non Central T (large parameters)
  Testing: Non Central T
  CDF<double> Max = 137.7 RMS Mean=31.5
      worst case at row: 181
      { 188.01481628417969, -282.022216796875, -298.02532958984375, 0.1552789395983287, 0.84472106040167128 }
  
  CCDF<double> Max = 150.4 RMS Mean=32.32
      worst case at row: 184
      { 191.43339538574219, 765.73358154296875, 820.14422607421875, 0.89943076553533785, 0.10056923446466212 }
  
  
  Testing: double quantile sanity check, with tests Non Central T
  Testing: Non Central T (small non-centrality)
  CDF<double> Max = 3.605 RMS Mean=1.031
      worst case at row: 42
      { 7376104448, 7.3761043495323975e-007, -1.3614851236343384, 0.086680099352107118, 0.91331990064789292 }
  
  CCDF<double> Max = 5.207 RMS Mean=1.432
      worst case at row: 38
      { 1524088576, 1.5240885886669275e-007, 1.3784774541854858, 0.91597201432644526, 0.084027985673554725 }
  
  
  Testing: double quantile sanity check, with tests Non Central T (small non-centrality)
  Testing: Non Central T (large parameters)
  CDF<double> Max = 286.4 RMS Mean=62.79
      worst case at row: 24
      { 1.3091821180254421e+019, 1309.18212890625, 1308.01171875, 0.12091797523015677, 0.87908202476984321 }
  
  CCDF<double> Max = 226.9 RMS Mean=50.41
      worst case at row: 23
      { 7.9217674231144776e+018, 792.1767578125, 793.54827880859375, 0.91489369852628, 0.085106301473719961 }
  
  
  Testing: double quantile sanity check, with tests Non Central T (large parameters)
  Testing: Non Central T
  CDF<long double> Max = 137.7 RMS Mean=31.5
      worst case at row: 181
      { 188.01481628417969, -282.022216796875, -298.02532958984375, 0.1552789395983287, 0.84472106040167128 }
  
  CCDF<long double> Max = 150.4 RMS Mean=32.32
      worst case at row: 184
      { 191.43339538574219, 765.73358154296875, 820.14422607421875, 0.89943076553533785, 0.10056923446466212 }
  
  
  Testing: long double quantile sanity check, with tests Non Central T
  Testing: Non Central T (small non-centrality)
  CDF<long double> Max = 3.605 RMS Mean=1.031
      worst case at row: 42
      { 7376104448, 7.3761043495323975e-007, -1.3614851236343384, 0.086680099352107118, 0.91331990064789292 }
  
  CCDF<long double> Max = 5.207 RMS Mean=1.432
      worst case at row: 38
      { 1524088576, 1.5240885886669275e-007, 1.3784774541854858, 0.91597201432644526, 0.084027985673554725 }
  
  
  Testing: long double quantile sanity check, with tests Non Central T (small non-centrality)
  Testing: Non Central T (large parameters)
  CDF<long double> Max = 286.4 RMS Mean=62.79
      worst case at row: 24
      { 1.3091821180254421e+019, 1309.18212890625, 1308.01171875, 0.12091797523015677, 0.87908202476984321 }
  
  CCDF<long double> Max = 226.9 RMS Mean=50.41
      worst case at row: 23
      { 7.9217674231144776e+018, 792.1767578125, 793.54827880859375, 0.91489369852628, 0.085106301473719961 }
  
  
  Testing: long double quantile sanity check, with tests Non Central T (large parameters)
  Testing: Non Central T
  CDF<real_concept> Max = 2.816e+005 RMS Mean=2.029e+004
      worst case at row: 185
      { 191.50137329101562, -957.5068359375, -1035.4078369140625, 0.072545502958829097, 0.92745449704117089 }
  
  CCDF<real_concept> Max = 1.304e+005 RMS Mean=1.529e+004
      worst case at row: 184
      { 191.43339538574219, 765.73358154296875, 820.14422607421875, 0.89943076553533785, 0.10056923446466212 }
  
  
  cdf(n10, 11)  = 0.84134471416473389 0.15865525603294373
  cdf(n10, 9)  = 0.15865525603294373 0.84134471416473389
  cdf(maxdf10, 11)  = 0.84134477376937866 0.15865525603294373
  cdf(infdf10, 11)  = 0.84134477376937866 0.15865525603294373
  cdf(n10, 11)  = 0.84134474606854293 0.15865525393145707
  cdf(n10, 9)  = 0.15865525393145707 0.84134474606854293
  cdf(maxdf10, 11)  = 0.84134474606854293 0.15865525393145707
  cdf(infdf10, 11)  = 0.84134474606854293 0.15865525393145707
  
  *** No errors detected

    Description: Autorun "J:\Cpp\MathToolkit\test\Math_test\Debug\test_nc_t.exe"
  Running 1 test case...
  Tests run with Microsoft Visual C++ version 10.0, Dinkumware standard library version 520, Win32
  Tolerance = 0.000596046%.
  Tolerance = 5e-010%.
  Tolerance = 5e-010%.
  Tolerance = 1e-008%.
  Testing: Non Central T
  CDF<float> Max = 0 RMS Mean=0
  
  CCDF<float> Max = 0 RMS Mean=0
  
  
  Testing: float quantile sanity check, with tests Non Central T
  Testing: Non Central T (small non-centrality)
  CDF<float> Max = 0 RMS Mean=0
  
  CCDF<float> Max = 0 RMS Mean=0
  
  
  Testing: float quantile sanity check, with tests Non Central T (small non-centrality)
  Testing: Non Central T (large parameters)
  CDF<float> Max = 0 RMS Mean=0
  
  CCDF<float> Max = 0 RMS Mean=0
  
  
  Testing: float quantile sanity check, with tests Non Central T (large parameters)
  Testing: Non Central T
  CDF<double> Max = 137.7 RMS Mean=31.5
      worst case at row: 181
      { 188.01481628417969, -282.022216796875, -298.02532958984375, 0.1552789395983287, 0.84472106040167128 }
  
  CCDF<double> Max = 150.4 RMS Mean=32.32
      worst case at row: 184
      { 191.43339538574219, 765.73358154296875, 820.14422607421875, 0.89943076553533785, 0.10056923446466212 }
  
  
  Testing: double quantile sanity check, with tests Non Central T
  Testing: Non Central T (small non-centrality)
  CDF<double> Max = 3.605 RMS Mean=1.031
      worst case at row: 42
      { 7376104448, 7.3761043495323975e-007, -1.3614851236343384, 0.086680099352107118, 0.91331990064789292 }
  
  CCDF<double> Max = 5.207 RMS Mean=1.432
      worst case at row: 38
      { 1524088576, 1.5240885886669275e-007, 1.3784774541854858, 0.91597201432644526, 0.084027985673554725 }
  
  
  Testing: double quantile sanity check, with tests Non Central T (small non-centrality)
  Testing: Non Central T (large parameters)
  CDF<double> Max = 286.4 RMS Mean=62.79
      worst case at row: 24
      { 1.3091821180254421e+019, 1309.18212890625, 1308.01171875, 0.12091797523015677, 0.87908202476984321 }
  
  CCDF<double> Max = 226.9 RMS Mean=50.41
      worst case at row: 23
      { 7.9217674231144776e+018, 792.1767578125, 793.54827880859375, 0.91489369852628, 0.085106301473719961 }
  
  
  Testing: double quantile sanity check, with tests Non Central T (large parameters)
  Testing: Non Central T
  CDF<long double> Max = 137.7 RMS Mean=31.5
      worst case at row: 181
      { 188.01481628417969, -282.022216796875, -298.02532958984375, 0.1552789395983287, 0.84472106040167128 }
  
  CCDF<long double> Max = 150.4 RMS Mean=32.32
      worst case at row: 184
      { 191.43339538574219, 765.73358154296875, 820.14422607421875, 0.89943076553533785, 0.10056923446466212 }
  
  
  Testing: long double quantile sanity check, with tests Non Central T
  Testing: Non Central T (small non-centrality)
  CDF<long double> Max = 3.605 RMS Mean=1.031
      worst case at row: 42
      { 7376104448, 7.3761043495323975e-007, -1.3614851236343384, 0.086680099352107118, 0.91331990064789292 }
  
  CCDF<long double> Max = 5.207 RMS Mean=1.432
      worst case at row: 38
      { 1524088576, 1.5240885886669275e-007, 1.3784774541854858, 0.91597201432644526, 0.084027985673554725 }
  
  
  Testing: long double quantile sanity check, with tests Non Central T (small non-centrality)
  Testing: Non Central T (large parameters)
  CDF<long double> Max = 286.4 RMS Mean=62.79
      worst case at row: 24
      { 1.3091821180254421e+019, 1309.18212890625, 1308.01171875, 0.12091797523015677, 0.87908202476984321 }
  
  CCDF<long double> Max = 226.9 RMS Mean=50.41
      worst case at row: 23
      { 7.9217674231144776e+018, 792.1767578125, 793.54827880859375, 0.91489369852628, 0.085106301473719961 }
  
  
  Testing: long double quantile sanity check, with tests Non Central T (large parameters)
  Testing: Non Central T
  CDF<real_concept> Max = 2.816e+005 RMS Mean=2.029e+004
      worst case at row: 185
      { 191.50137329101562, -957.5068359375, -1035.4078369140625, 0.072545502958829097, 0.92745449704117089 }
  
  CCDF<real_concept> Max = 1.304e+005 RMS Mean=1.529e+004
      worst case at row: 184
      { 191.43339538574219, 765.73358154296875, 820.14422607421875, 0.89943076553533785, 0.10056923446466212 }
  
  
  
  *** No errors detected


*/



/*

Temporary stuff from student's t version.


   // Calculate 1 / eps, the point where student's t should change to normal distribution.
    RealType limit = 1 / boost::math::tools::epsilon<RealType>();

    using namespace boost::math::policies;
    typedef policy<digits10<17> > accurate_policy; // 17 = max_digits10 where available.
    limit = 1 / policies::get_epsilon<RealType, accurate_policy>();

    BOOST_CHECK_CLOSE_FRACTION(limit, static_cast<RealType>(1) / std::numeric_limits<RealType>::epsilon(), tolerance);
    // Default policy to get full accuracy.
    // std::cout << "Switch over to normal if df > " << limit << std::endl;
    // float Switch over to normal if df > 8.38861e+006
    // double Switch over to normal if df > 4.5036e+015
    // Can't test real_concept - doesn't converge.

    boost::math::normal_distribution<RealType> n01(0, 1); // 
    boost::math::normal_distribution<RealType> n10(10, 1); // 
    non_central_t_distribution<RealType> nct(boost::math::tools::max_value<RealType>(), 0); // Well over the switchover point,
    non_central_t_distribution<RealType> nct2(limit /5, 0); // Just below the switchover point,
    non_central_t_distribution<RealType> nct3(limit /100, 0); // Well below the switchover point,
    non_central_t_distribution<RealType> nct4(limit, 10); // Well below the switchover point, and 10 non-centrality.

    // PDF
    BOOST_CHECK_CLOSE_FRACTION(pdf(nct, 0), pdf(n01, 0.), tolerance); // normal and non-central t should be nearly equal.
    BOOST_CHECK_CLOSE_FRACTION(pdf(nct2, 0), pdf(n01, 0.), tolerance); // should be very close to normal.
    BOOST_CHECK_CLOSE_FRACTION(pdf(nct3, 0), pdf(n01, 0.), tolerance * 10); // should be close to normal.
 //   BOOST_CHECK_CLOSE_FRACTION(pdf(nct4, 10), pdf(n10, 0.), tolerance * 100); // should be fairly close to normal tolerance.

    RealType delta = 10; // non-centrality.
    RealType nu = static_cast<RealType>(limit); // df
    boost::math::normal_distribution<RealType> nl(delta, 1); // Normal distribution that nct tends to for big df. 
    non_central_t_distribution<RealType> nct5(nu, delta); //
    RealType x = delta;
  //  BOOST_CHECK_CLOSE_FRACTION(pdf(nct5, x), pdf(nl, x), tolerance * 10 ); // nu = 1e15
  //  BOOST_CHECK_CLOSE_FRACTION(pdf(nct5, x), pdf(nl, x), tolerance * 1000 ); // nu = 1e14
  //  BOOST_CHECK_CLOSE_FRACTION(pdf(nct5, x), pdf(nl, x), tolerance * 10000 ); // nu = 1e13
  //  BOOST_CHECK_CLOSE_FRACTION(pdf(nct5, x), pdf(nl, x), tolerance * 100000 ); // nu = 1e12
    BOOST_CHECK_CLOSE_FRACTION(pdf(nct5, x), pdf(nl, x), tolerance * 5  ); // nu = 1/eps

  // Increasing the non-centrality delta increases the difference too because increases asymmetry.
  // For example, with non-centrality = 100, need tolerance * 500 

      // CDF
    BOOST_CHECK_CLOSE_FRACTION(cdf(nct, 0), cdf(n01, 0.), tolerance); // should be exactly equal.
    BOOST_CHECK_CLOSE_FRACTION(cdf(nct2, 0), cdf(n01, 0.), tolerance); // should be very close to normal.

    BOOST_CHECK_CLOSE_FRACTION(cdf(complement(n10, 11)), 1 - cdf(n10, 11), tolerance); // 
    //   cdf(n10, 10)  = 0.841345 0.158655
    BOOST_CHECK_CLOSE_FRACTION(cdf(complement(n10, 9)), 1 - cdf(n10, 9), tolerance); // 
    std::cout.precision(17);
    std::cout  << "cdf(n10, 11)  = " << cdf(n10, 11) << ' ' << cdf(complement(n10, 11)) << endl;
    std::cout  << "cdf(n10, 9)  = " << cdf(n10, 9) << ' ' << cdf(complement(n10, 9)) << endl;

  std::cout << std::numeric_limits<double>::max_digits10 << std::endl;
   std::cout.precision(17);

   using boost::math::tools::max_value;

   double eps = std::numeric_limits<double>::epsilon();
   // Use policies so that if policy requests lower precision, 
   // then get the normal distribution approximation earlier.
   //limit = static_cast<double>(1) / limit; // 1/eps
   double delta = 1e2;
   double df = 
   delta / (4 * eps);

    std::cout << df << std::endl; // df = 1.125899906842624e+018
     
   {
     boost::math::non_central_t_distribution<double> dist(df, delta);

      std::cout <<"mean " << mean(dist) << std::endl; // mean 1000
      std::cout <<"variance " << variance(dist) << std::endl; // variance 1
      std::cout <<"skewness " << skewness(dist) << std::endl; //  skewness 8.8817841970012523e-010
      std::cout <<"kurtosis_excess " << kurtosis_excess(dist) << std::endl; // kurtosis_excess 3.0001220703125
  //1.125899906842624e+017
  //mean 100
  //variance 1
  //skewness 8.8817841970012523e-012
  //kurtosis_excess 3


   }



  */
