// Copyright Paul A. Bristow 2006, 2017.
// Copyright John Maddock 2006.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// test_students_t.cpp

// http://en.wikipedia.org/wiki/Student%27s_t_distribution
// http://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm

// Basic sanity test for Student's t probability (quantile) (0. < p < 1).
// and Student's t probability Quantile (0. < p < 1).

#ifdef _MSC_VER
#  pragma warning (disable :4127) // conditional expression is constant.
#endif

#include <boost/math/tools/config.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/next.hpp>  // for has_denorm_now

#include "../include_private/boost/math/tools/test.hpp"

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#endif

#include "test_out_of_range.hpp"
#include <boost/math/distributions/students_t.hpp>
    using boost::math::students_t_distribution;

#include <iostream>
   using std::cout;
   using std::endl;
   using std::setprecision;
#include <limits>
  using std::numeric_limits;
#include <type_traits>

template <class RealType>
RealType naive_pdf(RealType v, RealType t)
{
   // Calculate the pdf of the students t in a deliberately
   // naive way, using equation (5) from
   // http://mathworld.wolfram.com/Studentst-Distribution.html
   // This is equivalent to, but a different method
   // to the one in the actual implementation, so can be used as
   // a very basic sanity check.  However some published values
   // would be nice....

   using namespace std;  // for ADL
   using boost::math::beta;

   //return pow(v / (v + t*t), (1+v) / 2) / (sqrt(v) * beta(v/2, RealType(0.5f)));
   RealType result = boost::math::tgamma_ratio((v+1)/2, v/2);
   result /= sqrt(v * boost::math::constants::pi<RealType>());
   result /= pow(1 + t*t/v, (v+1)/2);
   return result;
}

template <class RealType>
void test_spots(RealType)
{
  // Basic sanity checks

   RealType tolerance = static_cast<RealType>(1e-4); // 1e-6 (as %)
   // Some tests only pass at 1e-5 because probability value is less accurate,
   // a digit in 6th decimal place, although calculated using
   // a t-distribution generator (claimed 6 decimal digits) at
  // http://faculty.vassar.edu/lowry/VassarStats.html
   // http://faculty.vassar.edu/lowry/tsamp.html
   // df = 5, +/-t = 2.0, 1-tailed = 0.050970, 2-tailed = 0.101939

   cout << "Tolerance for type " << typeid(RealType).name()  << " is " << tolerance << " %" << endl;

   // http://en.wikipedia.org/wiki/Student%27s_t_distribution#Table_of_selected_values
  // Using tabulated value of t = 3.182 for 0.975, 3 df, one-sided.

   // http://www.mth.kcl.ac.uk/~shaww/web_page/papers/Tdistribution06.pdf refers to:

   // A lookup table of quantiles of the RealType distribution
  // for 1 to 25 in steps of 0.1 is provided in CSV form at:
  // www.mth.kcl.ac.uk/~shaww/web_page/papers/Tsupp/tquantiles.csv
   // gives accurate t of -3.1824463052837 and 3 degrees of freedom.
   // Values below are from this source, saved as tquantiles.xls.
   // DF are across the columns, probabilities down the rows
   // and the t- values (quantiles) are shown.
   // These values are probably accurate to nearly 64-bit double
  // (perhaps 14 decimal digits).

   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         students_t_distribution<RealType>(2),       // degrees_of_freedom
         static_cast<RealType>(-6.96455673428326)),  // t
         static_cast<RealType>(0.01),                // probability.
         tolerance); // %

   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         students_t_distribution<RealType>(5),       // degrees_of_freedom
         static_cast<RealType>(-3.36492999890721)),  // t
         static_cast<RealType>(0.01),                // probability.
         tolerance);

   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         students_t_distribution<RealType>(1),      // degrees_of_freedom
         static_cast<RealType>(-31830.988607907)),  // t
         static_cast<RealType>(0.00001),            // probability.
         tolerance);

   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         students_t_distribution<RealType>(25.),    // degrees_of_freedom
         static_cast<RealType>(-5.2410429995425)),  // t
         static_cast<RealType>(0.00001),            // probability.
         tolerance);

   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         students_t_distribution<RealType>(1),   // degrees_of_freedom
         static_cast<RealType>(-63661.97723)),   // t
         static_cast<RealType>(0.000005),        // probability.
         tolerance);

    BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         students_t_distribution<RealType>(5.),  // degrees_of_freedom
         static_cast<RealType>(-17.89686614)),   // t
         static_cast<RealType>(0.000005),        // probability.
         tolerance);

    BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         students_t_distribution<RealType>(25.),  // degrees_of_freedom
         static_cast<RealType>(-5.510848412)),    // t
         static_cast<RealType>(0.000005),         // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         students_t_distribution<RealType>(10.),  // degrees_of_freedom
         static_cast<RealType>(-1.812461123)),    // t
         static_cast<RealType>(0.05),             // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         students_t_distribution<RealType>(10),  // degrees_of_freedom
         static_cast<RealType>(1.812461123)),    // t
         static_cast<RealType>(0.95),            // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(
            students_t_distribution<RealType>(10),  // degrees_of_freedom
            static_cast<RealType>(1.812461123))),    // t
         static_cast<RealType>(0.05),            // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         students_t_distribution<RealType>(10),  // degrees_of_freedom
         static_cast<RealType>(9.751995491)),    // t
         static_cast<RealType>(0.999999),        // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         students_t_distribution<RealType>(10.),  // degrees_of_freedom - for ALL degrees_of_freedom!
         static_cast<RealType>(0.)),              // t
         static_cast<RealType>(0.5),              // probability.
         tolerance);


   // Student's t Inverse function tests.
  // Special cases

  BOOST_MATH_CHECK_THROW(boost::math::quantile(
         students_t_distribution<RealType>(1.),  // degrees_of_freedom (ignored).
         static_cast<RealType>(0)), std::overflow_error); // t == -infinity.

  BOOST_MATH_CHECK_THROW(boost::math::quantile(
         students_t_distribution<RealType>(1.),  // degrees_of_freedom (ignored).
         static_cast<RealType>(1)), std::overflow_error); // t == +infinity.

  BOOST_CHECK_EQUAL(boost::math::quantile(
         students_t_distribution<RealType>(1.),  // degrees_of_freedom (ignored).
         static_cast<RealType>(0.5)),  //  probability == half - special case.
         static_cast<RealType>(0)); // t == zero.

  BOOST_CHECK_EQUAL(boost::math::quantile(
         complement(
            students_t_distribution<RealType>(1.),  // degrees_of_freedom (ignored).
            static_cast<RealType>(0.5))),  //  probability == half - special case.
         static_cast<RealType>(0)); // t == zero.

  BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(1.),  // degrees_of_freedom (ignored).
         static_cast<RealType>(0.5)),  //  probability == half - special case.
         static_cast<RealType>(0), // t == zero.
         tolerance);

   BOOST_CHECK_CLOSE( // Tests of p middling.
      ::boost::math::cdf(
         students_t_distribution<RealType>(5.),  // degrees_of_freedom
         static_cast<RealType>(-0.559429644)),  // t
         static_cast<RealType>(0.3), // probability.
         tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         students_t_distribution<RealType>(5.),  // degrees_of_freedom
         static_cast<RealType>(0.3)),  // probability.
         static_cast<RealType>(-0.559429644), // t
         tolerance);

   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(
            students_t_distribution<RealType>(5.),  // degrees_of_freedom
            static_cast<RealType>(0.7))),  // probability.
         static_cast<RealType>(-0.559429644), // t
         tolerance);

   BOOST_CHECK_CLOSE( // Tests of p high.
      ::boost::math::cdf(
         students_t_distribution<RealType>(5.),  // degrees_of_freedom
         static_cast<RealType>(1.475884049)),  // t
         static_cast<RealType>(0.9), // probability.
         tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         students_t_distribution<RealType>(5.),  // degrees_of_freedom
         static_cast<RealType>(0.9)),  // probability.
         static_cast<RealType>(1.475884049), // t
         tolerance);

   BOOST_CHECK_CLOSE( // Tests of p low.
      ::boost::math::cdf(
         students_t_distribution<RealType>(5.),  // degrees_of_freedom
         static_cast<RealType>(-1.475884049)),  // t
         static_cast<RealType>(0.1), // probability.
         tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         students_t_distribution<RealType>(5.),  // degrees_of_freedom
         static_cast<RealType>(0.1)),  // probability.
         static_cast<RealType>(-1.475884049), // t
         tolerance);

   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         students_t_distribution<RealType>(2.),  // degrees_of_freedom
         static_cast<RealType>(-6.96455673428326)),  // t
         static_cast<RealType>(0.01), // probability.
         tolerance);

   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         students_t_distribution<RealType>(2.),  // degrees_of_freedom
         static_cast<RealType>(0.01)),  // probability.
         static_cast<RealType>(-6.96455673428326), // t
         tolerance);

      //
      // Some special tests to exercise the double-precision approximations
      // to the quantile:
      //
      // tolerance is 50 eps expressed as a percent:
      //
      tolerance = boost::math::tools::epsilon<RealType>() * 5000;
      //
      // But higher error rates at 128 bit precision?
      //
      if (boost::math::tools::digits<RealType>() > 100)
         tolerance *= 500;

      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(2.00390625L),                     // degrees_of_freedom.
         static_cast<RealType>(0.5625L)),                                    //  probability.
         static_cast<RealType>(0.178133131573788108465134803511798566L),     // t.
         tolerance);      
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(1L),                              // degrees_of_freedom.
         static_cast<RealType>(0.03125L)),                                   //  probability.
         static_cast<RealType>(-10.1531703876088604621071476634194722L),     // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(1L),                            // degrees_of_freedom.
         static_cast<RealType>(0.875L)),                                   //  probability.
         static_cast<RealType>(2.41421356237309504880168872421390942L),    // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(2L),                              // degrees_of_freedom.
         static_cast<RealType>(0.03125L)),                                   //  probability.
         static_cast<RealType>(-3.81000381000571500952501666878143315L),     // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(2L),                            // degrees_of_freedom.
         static_cast<RealType>(0.875L)),                                   //  probability.
         static_cast<RealType>(1.60356745147454630810732088527854144L),    // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(4L),                              // degrees_of_freedom.
         static_cast<RealType>(0.03125L)),                                   //  probability.
         static_cast<RealType>(-2.56208431914409044861223047927635034L),     // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(4L),                            // degrees_of_freedom.
         static_cast<RealType>(0.875L)),                                   //  probability.
         static_cast<RealType>(1.34439755550909142430681981315923574L),    // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(6L),                              // degrees_of_freedom.
         static_cast<RealType>(0.03125L)),                                   //  probability.
         static_cast<RealType>(-2.28348667906973065861212495010082952L),     // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(6L),                            // degrees_of_freedom.
         static_cast<RealType>(0.875L)),                                   //  probability.
         static_cast<RealType>(1.27334930914664286821103236660071906L),    // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(8L),                              // degrees_of_freedom.
         static_cast<RealType>(0.03125L)),                                   //  probability.
         static_cast<RealType>(-2.16296475406014719458642055768894376L),     // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(8L),                            // degrees_of_freedom.
         static_cast<RealType>(0.875L)),                                   //  probability.
         static_cast<RealType>(1.24031826078267310637634677726479038L),    // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(10L),                             // degrees_of_freedom.
         static_cast<RealType>(0.03125L)),                                   //  probability.
         static_cast<RealType>(-2.09596136475109350926340169211429572L),     // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(10L),                         // degrees_of_freedom.
         static_cast<RealType>(0.875L)),                                 //  probability.
         static_cast<RealType>(1.2212553950039221407185188573696834L),   // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(2.125L),                          // degrees_of_freedom.
         static_cast<RealType>(0.03125L)),                                   //  probability.
         static_cast<RealType>(-3.62246031671091980110493455859296532L),     // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(2.125L),                        // degrees_of_freedom.
         static_cast<RealType>(0.875L)),                                   //  probability.
         static_cast<RealType>(1.56905270993307293450392958697861969L),    // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(3L),                              // degrees_of_freedom.
         static_cast<RealType>(0.03125L)),                                   //  probability.
         static_cast<RealType>(-2.90004411882995814036141778367917946L),     // t.
         tolerance);
      BOOST_CHECK_CLOSE(boost::math::quantile(
         students_t_distribution<RealType>(3L),                            // degrees_of_freedom.
         static_cast<RealType>(0.875L)),                                   //  probability.
         static_cast<RealType>(1.42262528146180931868169289781115099L),    // t.
         tolerance);

      if(boost::is_floating_point<RealType>::value)
      {
         BOOST_CHECK_CLOSE(boost::math::cdf(
            students_t_distribution<RealType>(1e30f), 
               boost::math::quantile(
                  students_t_distribution<RealType>(1e30f), static_cast<RealType>(0.25f))), 
            static_cast<RealType>(0.25f), tolerance);
         BOOST_CHECK_CLOSE(boost::math::cdf(
            students_t_distribution<RealType>(1e20f), 
               boost::math::quantile(
                  students_t_distribution<RealType>(1e20f), static_cast<RealType>(0.25f))), 
            static_cast<RealType>(0.25f), tolerance);
         BOOST_CHECK_CLOSE(boost::math::cdf(
            students_t_distribution<RealType>(static_cast<RealType>(0x7FFFFFFF)), 
               boost::math::quantile(
                  students_t_distribution<RealType>(static_cast<RealType>(0x7FFFFFFF)), static_cast<RealType>(0.25f))), 
            static_cast<RealType>(0.25f), tolerance);
         BOOST_CHECK_CLOSE(boost::math::cdf(
            students_t_distribution<RealType>(static_cast<RealType>(0x10000000)), 
               boost::math::quantile(
                  students_t_distribution<RealType>(static_cast<RealType>(0x10000000)), static_cast<RealType>(0.25f))), 
            static_cast<RealType>(0.25f), tolerance);
         BOOST_CHECK_CLOSE(boost::math::cdf(
            students_t_distribution<RealType>(static_cast<RealType>(0x0fffffff)), 
               boost::math::quantile(
                  students_t_distribution<RealType>(static_cast<RealType>(0x0fffffff)), static_cast<RealType>(0.25f))), 
            static_cast<RealType>(0.25f), tolerance);
         //
         // Bug cases:
         //
         if (std::numeric_limits<RealType>::is_specialized && boost::math::detail::has_denorm_now<RealType>())
         {
            BOOST_CHECK_THROW(boost::math::quantile(students_t_distribution<RealType>((std::numeric_limits<RealType>::min)() / 2), static_cast<RealType>(0.0025f)), std::overflow_error);
         }
      }

  // Student's t pdf tests.
  // for PDF checks, use 100 eps tolerance expressed as a percent:
   tolerance = boost::math::tools::epsilon<RealType>() * 10000;

   for(unsigned i = 1; i < 20; i += 3)
   {
      for(RealType r = -10; r < 10; r += 0.125)
      {
         //std::cout << "df=" << i << " t=" << r << std::endl;
         BOOST_CHECK_CLOSE(
            boost::math::pdf(
               students_t_distribution<RealType>(static_cast<RealType>(i)),
               r),
            naive_pdf<RealType>(static_cast<RealType>(i), r),
            tolerance);
      }
   }

    RealType tol2 = boost::math::tools::epsilon<RealType>() * 5;
    students_t_distribution<RealType> dist(8);
    RealType x = static_cast<RealType>(0.125);
    using namespace std; // ADL of std names.
    // mean:
    BOOST_CHECK_CLOSE(
       mean(dist)
       , static_cast<RealType>(0), tol2);
    // variance:
 //   BOOST_CHECK_CLOSE(
 //      variance(dist)
 //      , static_cast<RealType>(13.0L / 6.0L), tol2);
 //// was     , static_cast<RealType>(8.0L / 6.0L), tol2);
    // std deviation:
    BOOST_CHECK_CLOSE(
       standard_deviation(dist)
       , static_cast<RealType>(sqrt(8.0L / 6.0L)), tol2);
    // hazard:
    BOOST_CHECK_CLOSE(
       hazard(dist, x)
       , pdf(dist, x) / cdf(complement(dist, x)), tol2);
    // cumulative hazard:
    BOOST_CHECK_CLOSE(
       chf(dist, x)
       , -log(cdf(complement(dist, x))), tol2);
    // coefficient_of_variation:
    BOOST_MATH_CHECK_THROW(
       coefficient_of_variation(dist),
       std::overflow_error);
    // mode:
    BOOST_CHECK_CLOSE(
       mean(dist)
       , static_cast<RealType>(0), tol2);
    // median:
    BOOST_CHECK_CLOSE(
       median(dist)
       , static_cast<RealType>(0), tol2);
    // skewness:
    BOOST_CHECK_CLOSE(
       skewness(dist)
       , static_cast<RealType>(0), tol2);
    // kurtosis:
    BOOST_CHECK_CLOSE(
       kurtosis(dist)
       , static_cast<RealType>(4.5), tol2);
    // kurtosis excess:
    BOOST_CHECK_CLOSE(
       kurtosis_excess(dist)
       , static_cast<RealType>(1.5), tol2);

    using std::log;
    using std::sqrt;
    RealType expected_entropy = (RealType(9)/2)*(boost::math::digamma(RealType(9)/2) - boost::math::digamma(RealType(4))) + log(sqrt(RealType(8))*boost::math::beta(RealType(4), RealType(1)/2));
    BOOST_CHECK_CLOSE(
       entropy(dist)
       , expected_entropy, 300*tol2);

    // Parameter estimation. These results are close to but
    // not identical to those reported on the NIST website at
    // http://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm
    // the NIST results appear to be calculated using a normal
    // approximation, which slightly under-estimates the degrees of
    // freedom required, particularly when the result is small.
    //
    BOOST_CHECK_EQUAL(
       ceil(students_t_distribution<RealType>::find_degrees_of_freedom(
         static_cast<RealType>(0.5),
         static_cast<RealType>(0.005),
         static_cast<RealType>(0.01),
         static_cast<RealType>(1.0))),
         99);
    BOOST_CHECK_EQUAL(
       ceil(students_t_distribution<RealType>::find_degrees_of_freedom(
         static_cast<RealType>(1.5),
         static_cast<RealType>(0.005),
         static_cast<RealType>(0.01),
         static_cast<RealType>(1.0))),
         14);
    BOOST_CHECK_EQUAL(
       ceil(students_t_distribution<RealType>::find_degrees_of_freedom(
         static_cast<RealType>(0.5),
         static_cast<RealType>(0.025),
         static_cast<RealType>(0.01),
         static_cast<RealType>(1.0))),
         76);
    BOOST_CHECK_EQUAL(
       ceil(students_t_distribution<RealType>::find_degrees_of_freedom(
         static_cast<RealType>(1.5),
         static_cast<RealType>(0.025),
         static_cast<RealType>(0.01),
         static_cast<RealType>(1.0))),
         11);
    BOOST_CHECK_EQUAL(
       ceil(students_t_distribution<RealType>::find_degrees_of_freedom(
         static_cast<RealType>(0.5),
         static_cast<RealType>(0.05),
         static_cast<RealType>(0.01),
         static_cast<RealType>(1.0))),
         65);
    BOOST_CHECK_EQUAL(
       ceil(students_t_distribution<RealType>::find_degrees_of_freedom(
         static_cast<RealType>(1.5),
         static_cast<RealType>(0.05),
         static_cast<RealType>(0.01),
         static_cast<RealType>(1.0))),
         9);

    // Test for large degrees of freedom when should be same as normal.
    RealType inf = std::numeric_limits<RealType>::infinity();
    RealType nan = std::numeric_limits<RealType>::quiet_NaN();

    std::string type = typeid(RealType).name();
//    if (type != "class boost::math::concepts::real_concept") fails for gcc

    #ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
    BOOST_MATH_IF_CONSTEXPR(!std::is_same<RealType, boost::math::concepts::real_concept>::value)
    #endif
    { // Ordinary floats only.
      RealType limit = 1/ boost::math::tools::epsilon<RealType>();
      // Default policy to get full accuracy.
      // std::cout << "Switch over to normal if df > " << limit << std::endl;
      // float Switch over to normal if df > 8.38861e+006
      // double Switch over to normal if df > 4.5036e+015
      // Can't test real_concept - doesn't converge.

      boost::math::normal_distribution<RealType> n(0, 1); // 
      students_t_distribution<RealType> st(boost::math::tools::max_value<RealType>()); // Well over the switchover point,
      // PDF
      BOOST_CHECK_EQUAL(pdf(st, 0), pdf(n, 0.)); // Should be exactly equal.

      students_t_distribution<RealType> st2(limit /5 ); // Just below the switchover point,
      BOOST_CHECK_CLOSE_FRACTION(pdf(st2, 0), pdf(n, 0.), tolerance); // Should be very close to normal.
      // CDF
      BOOST_CHECK_EQUAL(cdf(st, 0), cdf(n, 0.)); // Should be exactly equal.
      BOOST_CHECK_CLOSE_FRACTION(cdf(st2, 0), cdf(n, 0.), tolerance); // Should be very close to normal.

      // Tests for df = infinity.
      students_t_distribution<RealType> infdf(inf);
      BOOST_CHECK_EQUAL(infdf.degrees_of_freedom(), inf);
      BOOST_CHECK_EQUAL(mean(infdf), 0); // OK.
#ifndef BOOST_NO_EXCEPTIONS
      BOOST_MATH_CHECK_THROW(students_t_distribution<RealType> minfdf(-inf), std::domain_error);
      BOOST_MATH_CHECK_THROW(students_t_distribution<RealType> minfdf(nan), std::domain_error);
      BOOST_MATH_CHECK_THROW(students_t_distribution<RealType> minfdf(-nan), std::domain_error);
#endif
      BOOST_CHECK_EQUAL(pdf(infdf, -inf), 0);
      BOOST_CHECK_EQUAL(pdf(infdf, +inf), 0);
      BOOST_CHECK_EQUAL(cdf(infdf, -inf), 0);
      BOOST_CHECK_EQUAL(cdf(infdf, +inf), 1);

     // BOOST_CHECK_CLOSE_FRACTION(pdf(infdf, 0), static_cast<RealType>(0.3989422804014326779399460599343818684759L), tolerance);
      BOOST_CHECK_CLOSE_FRACTION(pdf(infdf, 0),boost::math::constants::one_div_root_two_pi<RealType>() , tolerance);
      BOOST_CHECK_CLOSE_FRACTION(cdf(infdf, 0),boost::math::constants::half<RealType>() , tolerance);

    // Checks added for Trac #7717 report by Thomas Mang.

    BOOST_MATH_CHECK_THROW(quantile(dist, -1), std::domain_error);
    BOOST_MATH_CHECK_THROW(quantile(dist, 2), std::domain_error);
    BOOST_MATH_CHECK_THROW(pdf(students_t_distribution<RealType>(0), 0), std::domain_error);
    BOOST_MATH_CHECK_THROW(pdf(students_t_distribution<RealType>(-1), 0), std::domain_error);
  
    // Check on df for mean (moment k = 1)
    BOOST_MATH_CHECK_THROW(mean(students_t_distribution<RealType>(nan)), std::domain_error);
//    BOOST_MATH_CHECK_THROW(mean(students_t_distribution<RealType>(inf)), std::domain_error); inf is now OK
    BOOST_MATH_CHECK_THROW(mean(students_t_distribution<RealType>(-1)), std::domain_error);
    BOOST_MATH_CHECK_THROW(mean(students_t_distribution<RealType>(0)), std::domain_error);
    BOOST_MATH_CHECK_THROW(mean(students_t_distribution<RealType>(1)), std::domain_error); // df == k
    BOOST_CHECK_EQUAL(mean(students_t_distribution<RealType>(2)), 0); // OK.
    BOOST_CHECK_EQUAL(mean(students_t_distribution<RealType>(inf)), 0); // OK.

    // Check on df for variance (moment 2)
    BOOST_MATH_CHECK_THROW(variance(students_t_distribution<RealType>(nan)), std::domain_error);
//    BOOST_MATH_CHECK_THROW(variance(students_t_distribution<RealType>(inf)), std::domain_error); // inf is now OK.
    BOOST_MATH_CHECK_THROW(variance(students_t_distribution<RealType>(-1)), std::domain_error);
    BOOST_MATH_CHECK_THROW(variance(students_t_distribution<RealType>(0)), std::domain_error);
    BOOST_MATH_CHECK_THROW(variance(students_t_distribution<RealType>(1)), std::domain_error);
    BOOST_MATH_CHECK_THROW(variance(students_t_distribution<RealType>(static_cast<RealType>(1.99999L))), std::domain_error);
    BOOST_MATH_CHECK_THROW(variance(students_t_distribution<RealType>(static_cast<RealType>(1.99999L))), std::domain_error);
    BOOST_MATH_CHECK_THROW(variance(students_t_distribution<RealType>(2)), std::domain_error); // df == 
    BOOST_CHECK_EQUAL(variance(students_t_distribution<RealType>(2.5)), 5); // OK.
    BOOST_CHECK_EQUAL(variance(students_t_distribution<RealType>(3)), 3); // OK.
    BOOST_CHECK_EQUAL(variance(students_t_distribution<RealType>(inf)), 1); // OK.

    // Check on df for skewness (moment 3)
    BOOST_MATH_CHECK_THROW(skewness(students_t_distribution<RealType>(nan)), std::domain_error);
    BOOST_MATH_CHECK_THROW(skewness(students_t_distribution<RealType>(-1)), std::domain_error);
    BOOST_MATH_CHECK_THROW(skewness(students_t_distribution<RealType>(0)), std::domain_error);
    BOOST_MATH_CHECK_THROW(skewness(students_t_distribution<RealType>(1)), std::domain_error);
    BOOST_MATH_CHECK_THROW(skewness(students_t_distribution<RealType>(1.5L)), std::domain_error);
    BOOST_MATH_CHECK_THROW(skewness(students_t_distribution<RealType>(2)), std::domain_error); 
    BOOST_MATH_CHECK_THROW(skewness(students_t_distribution<RealType>(3)), std::domain_error); // df == k
    BOOST_CHECK_EQUAL(skewness(students_t_distribution<RealType>(3.5)), 0); // OK.
    BOOST_CHECK_EQUAL(skewness(students_t_distribution<RealType>(4)), 0); // OK.
    BOOST_CHECK_EQUAL(skewness(students_t_distribution<RealType>(inf)), 0); // OK.

    // Check on df for kurtosis_excess (moment 4)
    BOOST_MATH_CHECK_THROW(kurtosis_excess(students_t_distribution<RealType>(nan)), std::domain_error);
    BOOST_MATH_CHECK_THROW(kurtosis_excess(students_t_distribution<RealType>(-1)), std::domain_error);
    BOOST_MATH_CHECK_THROW(kurtosis_excess(students_t_distribution<RealType>(0)), std::domain_error);
    BOOST_MATH_CHECK_THROW(kurtosis_excess(students_t_distribution<RealType>(1)), std::domain_error);
    BOOST_MATH_CHECK_THROW(kurtosis_excess(students_t_distribution<RealType>(1.5L)), std::domain_error);
    BOOST_MATH_CHECK_THROW(kurtosis_excess(students_t_distribution<RealType>(2)), std::domain_error); 
    BOOST_MATH_CHECK_THROW(kurtosis(students_t_distribution<RealType>(static_cast<RealType>(2.1))), std::domain_error);
    BOOST_MATH_CHECK_THROW(kurtosis_excess(students_t_distribution<RealType>(3)), std::domain_error);
    BOOST_MATH_CHECK_THROW(kurtosis_excess(students_t_distribution<RealType>(4)), std::domain_error); // df == k
    BOOST_CHECK_EQUAL(kurtosis_excess(students_t_distribution<RealType>(5)), 6); // OK.
    BOOST_CHECK_EQUAL(kurtosis_excess(students_t_distribution<RealType>(inf)), 0); // OK.

    // Check on df for kurtosis (moment 4)
    BOOST_MATH_CHECK_THROW(kurtosis(students_t_distribution<RealType>(nan)), std::domain_error);
    BOOST_MATH_CHECK_THROW(kurtosis(students_t_distribution<RealType>(-1)), std::domain_error);
    BOOST_MATH_CHECK_THROW(kurtosis(students_t_distribution<RealType>(0)), std::domain_error);
    BOOST_MATH_CHECK_THROW(kurtosis(students_t_distribution<RealType>(1)), std::domain_error); 
    BOOST_MATH_CHECK_THROW(kurtosis(students_t_distribution<RealType>(2)), std::domain_error); 
    BOOST_MATH_CHECK_THROW(kurtosis(students_t_distribution<RealType>(static_cast<RealType>(2.0001L))), std::domain_error);
    BOOST_MATH_CHECK_THROW(kurtosis(students_t_distribution<RealType>(3)), std::domain_error);
    BOOST_MATH_CHECK_THROW(kurtosis(students_t_distribution<RealType>(4)), std::domain_error); // df == k
    BOOST_CHECK_EQUAL(kurtosis(students_t_distribution<RealType>(5)), 9); // OK.
    BOOST_CHECK_EQUAL(kurtosis(students_t_distribution<RealType>(inf)), 3); // OK.

   }


    // Use a new distribution ignore_error_students_t with a custom policy to ignore all errors,
    // and check returned values are as expected.

    /* 
     Sandia-darwin-intel-12.0 - math - test_students_t / intel-darwin-12.0
    ../libs/math/test/test_students_t.cpp(544): error: "domain_error" has already been declared in the current scope
    using boost::math::policies::domain_error;

../libs/math/test/test_students_t.cpp(552): error: "pole_error" has already been declared in the current scope
    using boost::math::policies::pole_error;

    Unclear where previous declaration is. 
    Does not seem to be in student_t.hpp or any included files???

    So to avoid this perceived problem by this compiler,
    the ignore policy below uses fully specified names.
    */

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
              > my_ignore_policy;

  typedef students_t_distribution<RealType, my_ignore_policy> ignore_error_students_t;



  // Only test NaN and infinity if type has these features (realconcept returns zero).
  // Integers are always converted to RealType,
  // others requires static cast to RealType from long double.

  if(std::numeric_limits<RealType>::has_quiet_NaN)
  {
  // Mean
    BOOST_CHECK((boost::math::isnan)(mean(ignore_error_students_t(-1))));
    BOOST_CHECK((boost::math::isnan)(mean(ignore_error_students_t(0))));
    BOOST_CHECK((boost::math::isnan)(mean(ignore_error_students_t(1))));

    // Variance
    BOOST_CHECK((boost::math::isnan)(variance(ignore_error_students_t(std::numeric_limits<RealType>::quiet_NaN()))));
    BOOST_CHECK((boost::math::isnan)(variance(ignore_error_students_t(-1))));
    BOOST_CHECK((boost::math::isnan)(variance(ignore_error_students_t(0))));
    BOOST_CHECK((boost::math::isnan)(variance(ignore_error_students_t(1))));
    BOOST_CHECK((boost::math::isnan)(variance(ignore_error_students_t(static_cast<RealType>(1.7L)))));
    BOOST_CHECK((boost::math::isnan)(variance(ignore_error_students_t(2))));

  // Skewness
    BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_students_t(std::numeric_limits<RealType>::quiet_NaN()))));
    BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_students_t(-1))));
    BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_students_t(0))));
    BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_students_t(1))));
    BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_students_t(2))));
    BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_students_t(3))));

  // Kurtosis 
    BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_students_t(std::numeric_limits<RealType>::quiet_NaN()))));
    BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_students_t(-1))));
    BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_students_t(0))));
    BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_students_t(1))));
    BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_students_t(2))));
    BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_students_t(static_cast<RealType>(2.0001L)))));
    BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_students_t(3))));
    BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_students_t(4))));
 
    // Kurtosis excess
    BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_students_t(std::numeric_limits<RealType>::quiet_NaN()))));
    BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_students_t(-1))));
    BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_students_t(0))));
    BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_students_t(1))));
    BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_students_t(2))));
    BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_students_t(static_cast<RealType>(2.0001L)))));
    BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_students_t(3))));
    BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_students_t(4))));
  } // has_quiet_NaN

  BOOST_CHECK(boost::math::isfinite(mean(ignore_error_students_t(1 + std::numeric_limits<RealType>::epsilon()))));
  BOOST_CHECK(boost::math::isfinite(variance(ignore_error_students_t(2 + 2 * std::numeric_limits<RealType>::epsilon()))));
  BOOST_CHECK(boost::math::isfinite(variance(ignore_error_students_t(static_cast<RealType>(2.0001L)))));
  BOOST_CHECK(boost::math::isfinite(variance(ignore_error_students_t(2 + 2 * std::numeric_limits<RealType>::epsilon()))));
  BOOST_CHECK(boost::math::isfinite(skewness(ignore_error_students_t(3 + 3 * std::numeric_limits<RealType>::epsilon()))));
  BOOST_CHECK(boost::math::isfinite(kurtosis(ignore_error_students_t(4 + 4 * std::numeric_limits<RealType>::epsilon()))));
  BOOST_CHECK(boost::math::isfinite(kurtosis(ignore_error_students_t(static_cast<RealType>(4.0001L)))));

  // check_out_of_range<students_t_distribution<RealType> >(1);
  // Cannot be used because fails "exception std::domain_error is expected but not raised" 
  // if df = +infinity is allowed, must use new version that allows skipping infinity tests.
  // Infinite == true

  check_support<students_t_distribution<RealType> >(students_t_distribution<RealType>(1), true);

} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{
  // Check that can construct students_t distribution using the two convenience methods:
  using namespace boost::math;
  students_t myst1(2); // Using typedef
  students_t_distribution<> myst2(2); // Using default RealType double.
   //students_t_distribution<double> myst3(2); // Using explicit RealType double.

   // Basic sanity-check spot values.
   // (Parameter value, arbitrarily zero, only communicates the floating point type).
  test_spots(0.0F); // Test float. OK at decdigits = 0 tolerance = 0.0001 %
  test_spots(0.0); // Test double. OK at decdigits 7, tolerance = 1e07 %
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  test_spots(0.0L); // Test long double.
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
  test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif


   
} // BOOST_AUTO_TEST_CASE( test_main )

/*

Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\test_students_t.exe"
Running 1 test case...
Tolerance for type float is 0.0001 %
Tolerance for type double is 0.0001 %
Tolerance for type long double is 0.0001 %
Tolerance for type class boost::math::concepts::real_concept is 0.0001 %
*** No errors detected

*/


