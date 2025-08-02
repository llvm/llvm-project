// test_binomial.cpp

// Copyright John Maddock 2006.
// Copyright  Paul A. Bristow 2007.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Basic sanity test for Binomial Cumulative Distribution Function.

#define BOOST_MATH_DISCRETE_QUANTILE_POLICY real

#if !defined(TEST_FLOAT) && !defined(TEST_DOUBLE) && !defined(TEST_LDOUBLE) && !defined(TEST_REAL_CONCEPT)
#  define TEST_FLOAT
#  define TEST_DOUBLE
#  define TEST_LDOUBLE
#  define TEST_REAL_CONCEPT
#endif

#ifdef _MSC_VER
#  pragma warning(disable: 4127) // conditional expression is constant.
#  pragma warning(disable: 4100) // unreferenced formal parameter.
// Seems an entirely spurious warning - formal parameter T IS used - get error if /* T */
//#  pragma warning(disable: 4535) // calling _set_se_translator() requires /EHa (in Boost.test)
// Enable C++ Exceptions Yes With SEH Exceptions (/EHa) prevents warning 4535.
#endif

#include <boost/math/concepts/real_concept.hpp> // for real_concept
using ::boost::math::concepts::real_concept;

#include <boost/math/distributions/binomial.hpp> // for binomial_distribution
using boost::math::binomial_distribution;

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE
#include "table_type.hpp"

#include "test_out_of_range.hpp"

#include <iostream>
using std::cout;
using std::endl;
#include <limits>
using std::numeric_limits;

template <class RealType>
void test_spot(
     RealType N,    // Number of trials
     RealType k,    // Number of successes
     RealType p,    // Probability of success
     RealType P,    // CDF
     RealType Q,    // Complement of CDF
     RealType tol)  // Test tolerance
{
   boost::math::binomial_distribution<RealType> bn(N, p);
   BOOST_CHECK_CLOSE(
      cdf(bn, k), P, tol);
   if((P < 0.99) && (Q < 0.99))
   {
      //
      // We can only check this if P is not too close to 1,
      // so that we can guarantee Q is free of error:
      //
      BOOST_CHECK_CLOSE(
         cdf(complement(bn, k)), Q, tol);
      if(k != 0)
      {
         BOOST_CHECK_CLOSE(
            quantile(bn, P), k, tol);
      }
      else
      {
         // Just check quantile is very small:
         if((std::numeric_limits<RealType>::max_exponent <= std::numeric_limits<double>::max_exponent) && (boost::is_floating_point<RealType>::value))
         {
            // Limit where this is checked: if exponent range is very large we may
            // run out of iterations in our root finding algorithm.
            BOOST_CHECK(quantile(bn, P) < boost::math::tools::epsilon<RealType>() * 10);
         }
      }
      if(k != 0)
      {
         BOOST_CHECK_CLOSE(
            quantile(complement(bn, Q)), k, tol);
      }
      else
      {
         // Just check quantile is very small:
         if((std::numeric_limits<RealType>::max_exponent <= std::numeric_limits<double>::max_exponent) && (boost::is_floating_point<RealType>::value))
         {
            // Limit where this is checked: if exponent range is very large we may
            // run out of iterations in our root finding algorithm.
            BOOST_CHECK(quantile(complement(bn, Q)) < boost::math::tools::epsilon<RealType>() * 10);
         }
      }
      if(k > 0)
      {
         // estimate success ratio:
         // Note lower bound uses a different formula internally
         // from upper bound, have to adjust things to prevent
         // fencepost errors:
         BOOST_CHECK_CLOSE(
            binomial_distribution<RealType>::find_lower_bound_on_p(
               N, k+1, Q),
            p, tol);
         BOOST_CHECK_CLOSE(
            binomial_distribution<RealType>::find_upper_bound_on_p(
               N, k, P),
            p, tol);

         if(Q < P)
         {
            // Default method (Clopper Pearson)
            BOOST_CHECK(
               binomial_distribution<RealType>::find_lower_bound_on_p(
                  N, k, Q)
                  <=
               binomial_distribution<RealType>::find_upper_bound_on_p(
                  N, k, Q)
                  );
            BOOST_CHECK((
               binomial_distribution<RealType>::find_lower_bound_on_p(
                  N, k, Q)
                  <= k/N) && (k/N <=
               binomial_distribution<RealType>::find_upper_bound_on_p(
                  N, k, Q))
                  );
            // Bayes Method (Jeffreys Prior)
            BOOST_CHECK(
               binomial_distribution<RealType>::find_lower_bound_on_p(
               N, k, Q, binomial_distribution<RealType>::jeffreys_prior_interval)
                  <=
               binomial_distribution<RealType>::find_upper_bound_on_p(
                  N, k, Q, binomial_distribution<RealType>::jeffreys_prior_interval)
                  );
            BOOST_CHECK((
               binomial_distribution<RealType>::find_lower_bound_on_p(
                  N, k, Q, binomial_distribution<RealType>::jeffreys_prior_interval)
                  <= k/N) && (k/N <=
               binomial_distribution<RealType>::find_upper_bound_on_p(
                  N, k, Q, binomial_distribution<RealType>::jeffreys_prior_interval))
                  );
         }
         else
         {
            // Default method (Clopper Pearson)
            BOOST_CHECK(
               binomial_distribution<RealType>::find_lower_bound_on_p(
                  N, k, P)
                  <=
               binomial_distribution<RealType>::find_upper_bound_on_p(
                  N, k, P)
                  );
            BOOST_CHECK(
               (binomial_distribution<RealType>::find_lower_bound_on_p(
                  N, k, P)
                  <= k / N) && (k/N <=
               binomial_distribution<RealType>::find_upper_bound_on_p(
                  N, k, P))
                  );
            // Bayes Method (Jeffreys Prior)
            BOOST_CHECK(
               binomial_distribution<RealType>::find_lower_bound_on_p(
                  N, k, P, binomial_distribution<RealType>::jeffreys_prior_interval)
                  <=
               binomial_distribution<RealType>::find_upper_bound_on_p(
                  N, k, P, binomial_distribution<RealType>::jeffreys_prior_interval)
                  );
            BOOST_CHECK(
               (binomial_distribution<RealType>::find_lower_bound_on_p(
                  N, k, P, binomial_distribution<RealType>::jeffreys_prior_interval)
                  <= k / N) && (k/N <=
               binomial_distribution<RealType>::find_upper_bound_on_p(
                  N, k, P, binomial_distribution<RealType>::jeffreys_prior_interval))
                  );
         }
      }
      //
      // estimate sample size:
      //
      BOOST_CHECK_CLOSE(
         binomial_distribution<RealType>::find_minimum_number_of_trials(
            k, p, P),
         N, tol);
      BOOST_CHECK_CLOSE(
         binomial_distribution<RealType>::find_maximum_number_of_trials(
            k, p, Q),
         N, tol);
   }

   // Double check consistency of CDF and PDF by computing
   // the finite sum:
   RealType sum = 0;
   for(unsigned i = 0; i <= k; ++i)
      sum += pdf(bn, RealType(i));
   BOOST_CHECK_CLOSE(
      sum, P, tol);
   // And complement as well:
   sum = 0;
   for(RealType i = N; i > k; i -= 1)
      sum += pdf(bn, i);
   if(P < 0.99)
   {
      BOOST_CHECK_CLOSE(
         sum, Q, tol);
   }
   else
   {
      // Not enough information content in P for Q to be meaningful
      RealType tol = (std::max)(2 * Q, boost::math::tools::epsilon<RealType>());
      BOOST_CHECK(sum < tol);
   }
}

template <class RealType> // Any floating-point type RealType.
void test_spots(RealType T)
{
  // Basic sanity checks, test data is to double precision only
  // so set tolerance to 100eps expressed as a percent, or
  // 100eps of type double expressed as a percent, whichever
  // is the larger.

  RealType tolerance = (std::max)
      (boost::math::tools::epsilon<RealType>(),
      static_cast<RealType>(std::numeric_limits<double>::epsilon()));
  tolerance *= 100 * 1000;
  RealType tol2 = boost::math::tools::epsilon<RealType>() * 5 * 100;  // 5 eps as a percent

  cout << "Tolerance for type " << typeid(T).name()  << " is " << tolerance << " %" << endl;


  // Sources of spot test values:

  // MathCAD defines pbinom(k, n, p)
  // returns pr(X ,=k) when random variable X has the binomial distribution with parameters n and p.
  // 0 <= k ,= n
  // 0 <= p <= 1
  // P = pbinom(30, 500, 0.05) = 0.869147702104609

  using boost::math::binomial_distribution;
  using  ::boost::math::cdf;
  using  ::boost::math::pdf;

#if !defined(TEST_ROUNDING) || (TEST_ROUNDING == 0)
  // Test binomial using cdf spot values from MathCAD.
  // These test quantiles and complements as well.
  test_spot(
     static_cast<RealType>(500),                     // Sample size, N
     static_cast<RealType>(30),                      // Number of successes, k
     static_cast<RealType>(0.05),                    // Probability of success, p
     static_cast<RealType>(0.869147702104609),       // Probability of result (CDF), P
     static_cast<RealType>(1 - 0.869147702104609),   // Q = 1 - P
     tolerance);

  test_spot(
     static_cast<RealType>(500),                     // Sample size, N
     static_cast<RealType>(250),                     // Number of successes, k
     static_cast<RealType>(0.05),                    // Probability of success, p
     static_cast<RealType>(1),                       // Probability of result (CDF), P
     static_cast<RealType>(0),   // Q = 1 - P
     tolerance);

  test_spot(
     static_cast<RealType>(500),                     // Sample size, N
     static_cast<RealType>(470),                     // Number of successes, k
     static_cast<RealType>(0.95),                    // Probability of success, p
     static_cast<RealType>(0.176470742656766),       // Probability of result (CDF), P
     static_cast<RealType>(1 - 0.176470742656766),   // Q = 1 - P
     tolerance * 10);                                // Note higher tolerance on this test!

  test_spot(
     static_cast<RealType>(500),                       // Sample size, N
     static_cast<RealType>(400),                       // Number of successes, k
     static_cast<RealType>(0.05),                      // Probability of success, p
     static_cast<RealType>(1),                         // Probability of result (CDF), P
     static_cast<RealType>(0),                         // Q = 1 - P
     tolerance);

  test_spot(
     static_cast<RealType>(500),                       // Sample size, N
     static_cast<RealType>(400),                       // Number of successes, k
     static_cast<RealType>(0.9),                       // Probability of success, p
     static_cast<RealType>(1.80180425681923E-11),      // Probability of result (CDF), P
     static_cast<RealType>(1 - 1.80180425681923E-11),  // Q = 1 - P
     tolerance);

  test_spot(
     static_cast<RealType>(500),                       // Sample size, N
     static_cast<RealType>(5),                         // Number of successes, k
     static_cast<RealType>(0.05),                      // Probability of success, p
     static_cast<RealType>(9.181808267643E-7),         // Probability of result (CDF), P
     static_cast<RealType>(1 - 9.181808267643E-7),     // Q = 1 - P
     tolerance);

  test_spot(
     static_cast<RealType>(2),                       // Sample size, N
     static_cast<RealType>(1),                       // Number of successes, k
     static_cast<RealType>(0.5),                     // Probability of success, p
     static_cast<RealType>(0.75),                    // Probability of result (CDF), P
     static_cast<RealType>(0.25),                    // Q = 1 - P
     tolerance);

  test_spot(
     static_cast<RealType>(8),                       // Sample size, N
     static_cast<RealType>(3),                       // Number of successes, k
     static_cast<RealType>(0.25),                    // Probability of success, p
     static_cast<RealType>(0.8861846923828125),      // Probability of result (CDF), P
     static_cast<RealType>(1 - 0.8861846923828125),  // Q = 1 - P
     tolerance);

  test_spot(
     static_cast<RealType>(8),                       // Sample size, N
     static_cast<RealType>(0),                       // Number of successes, k
     static_cast<RealType>(0.25),                    // Probability of success, p
     static_cast<RealType>(0.1001129150390625),      // Probability of result (CDF), P
     static_cast<RealType>(1 - 0.1001129150390625),  // Q = 1 - P
     tolerance);

  test_spot(
     static_cast<RealType>(8),                       // Sample size, N
     static_cast<RealType>(1),                       // Number of successes, k
     static_cast<RealType>(0.25),                    // Probability of success, p
     static_cast<RealType>(0.36708068847656244),     // Probability of result (CDF), P
     static_cast<RealType>(1 - 0.36708068847656244), // Q = 1 - P
     tolerance);

  test_spot(
     static_cast<RealType>(8),                       // Sample size, N
     static_cast<RealType>(4),                       // Number of successes, k
     static_cast<RealType>(0.25),                    // Probability of success, p
     static_cast<RealType>(0.9727020263671875),      // Probability of result (CDF), P
     static_cast<RealType>(1 - 0.9727020263671875),  // Q = 1 - P
     tolerance);

  test_spot(
     static_cast<RealType>(8),                       // Sample size, N
     static_cast<RealType>(7),                       // Number of successes, k
     static_cast<RealType>(0.25),                    // Probability of success, p
     static_cast<RealType>(0.9999847412109375),      // Probability of result (CDF), P
     static_cast<RealType>(1 - 0.9999847412109375),  // Q = 1 - P
     tolerance);

  // Tests on PDF follow:
  BOOST_CHECK_CLOSE(
     pdf(binomial_distribution<RealType>(static_cast<RealType>(20), static_cast<RealType>(0.75)),
     static_cast<RealType>(10)),  // k.
     static_cast<RealType>(0.00992227527967770583927631378173), // 0.00992227527967770583927631378173
     tolerance);

  BOOST_CHECK_CLOSE(
    pdf(binomial_distribution<RealType>(static_cast<RealType>(20), static_cast<RealType>(0.5)),
    static_cast<RealType>(10)),  // k.
    static_cast<RealType>(0.17619705200195312500000000000000000000), // get k=10 0.049611376398388612 p = 0.25
    tolerance);

  // Binomial pdf Test values from
  // http://www.adsciengineering.com/bpdcalc/index.php  for example
  // http://www.adsciengineering.com/bpdcalc/index.php?n=20&p=0.25&start=0&stop=20&Submit=Generate
  // Appears to use at least 80-bit long double for 32 decimal digits accuracy,
  // but loses accuracy of display if leading zeros?
  // (if trailings zero then are exact values?)
  // so useful for testing 64-bit double accuracy.
  // P = 0.25, n = 20, k = 0 to 20

  //0   C(20,0) * 0.25^0 * 0.75^20   0.00317121193893399322405457496643
  //1   C(20,1) * 0.25^1 * 0.75^19   0.02114141292622662149369716644287
  //2   C(20,2) * 0.25^2 * 0.75^18   0.06694780759971763473004102706909
  //3   C(20,3) * 0.25^3 * 0.75^17   0.13389561519943526946008205413818
  //4   C(20,4) * 0.25^4 * 0.75^16   0.18968545486586663173511624336242
  //5   C(20,5) * 0.25^5 * 0.75^15   0.20233115185692440718412399291992
  //6   C(20,6) * 0.25^6 * 0.75^14   0.16860929321410367265343666076660
  //7   C(20,7) * 0.25^7 * 0.75^13   0.11240619547606911510229110717773
  //8   C(20,8) * 0.25^8 * 0.75^12   0.06088668921620410401374101638793
  //9   C(20,9) * 0.25^9 * 0.75^11   0.02706075076275737956166267395019
  //10   C(20,10) * 0.25^10 * 0.75^10   0.00992227527967770583927631378173
  //11   C(20,11) * 0.25^11 * 0.75^9   0.00300675008475081995129585266113
  //12   C(20,12) * 0.25^12 * 0.75^8   0.00075168752118770498782396316528
  //13   C(20,13) * 0.25^13 * 0.75^7   0.00015419231203850358724594116210
  //14   C(20,14) * 0.25^14 * 0.75^6   0.00002569871867308393120765686035
  //15   C(20,15) * 0.25^15 * 0.75^5   0.00000342649582307785749435424804
  //16   C(20,16) * 0.25^16 * 0.75^4   0.00000035692664823727682232856750
  //17   C(20,17) * 0.25^17 * 0.75^3   0.00000002799424692057073116302490
  //18   C(20,18) * 0.25^18 * 0.75^2   0.00000000155523594003170728683471
  //19   C(20,19) * 0.25^19 * 0.75^1   0.00000000005456968210637569427490
  //20   C(20,20) * 0.25^20 * 0.75^0   0.00000000000090949470177292823791


    BOOST_CHECK_CLOSE(
    pdf(binomial_distribution<RealType>(static_cast<RealType>(20), static_cast<RealType>(0.25)),
    static_cast<RealType>(10)),  // k.
    static_cast<RealType>(0.00992227527967770583927631378173), // k=10  p = 0.25
    tolerance);

    BOOST_CHECK_CLOSE( // k = 0 use different formula - only exp so more accurate.
    pdf(binomial_distribution<RealType>(static_cast<RealType>(20), static_cast<RealType>(0.25)),
    static_cast<RealType>(0)),  // k.
    static_cast<RealType>(0.00317121193893399322405457496643), // k=0  p = 0.25
    tolerance);

    BOOST_CHECK_CLOSE( // k = 20 use different formula - only exp so more accurate.
    pdf(binomial_distribution<RealType>(static_cast<RealType>(20), static_cast<RealType>(0.25)),
    static_cast<RealType>(20)),  // k == n.
    static_cast<RealType>(0.00000000000090949470177292823791), // k=20  p = 0.25
    tolerance);

    BOOST_CHECK_CLOSE( // k = 1.
    pdf(binomial_distribution<RealType>(static_cast<RealType>(20), static_cast<RealType>(0.25)),
    static_cast<RealType>(1)),  // k.
    static_cast<RealType>(0.02114141292622662149369716644287), // k=1  p = 0.25
    tolerance);

    // Some exact (probably) values.
    BOOST_CHECK_CLOSE(
    pdf(binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
    static_cast<RealType>(0)),  // k.
    static_cast<RealType>(0.10011291503906250000000000000000), // k=0  p = 0.25
    tolerance);

    BOOST_CHECK_CLOSE( // k = 1.
    pdf(binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
    static_cast<RealType>(1)),  // k.
    static_cast<RealType>(0.26696777343750000000000000000000), // k=1  p = 0.25
    tolerance);

    BOOST_CHECK_CLOSE( // k = 2.
    pdf(binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
    static_cast<RealType>(2)),  // k.
    static_cast<RealType>(0.31146240234375000000000000000000), // k=2  p = 0.25
    tolerance);

    BOOST_CHECK_CLOSE( // k = 3.
    pdf(binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
    static_cast<RealType>(3)),  // k.
    static_cast<RealType>(0.20764160156250000000000000000000), // k=3  p = 0.25
    tolerance);

    BOOST_CHECK_CLOSE( // k = 7.
    pdf(binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
    static_cast<RealType>(7)),  // k.
    static_cast<RealType>(0.00036621093750000000000000000000), // k=7  p = 0.25
    tolerance);

    BOOST_CHECK_CLOSE( // k = 8.
    pdf(binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
    static_cast<RealType>(8)),  // k = n.
    static_cast<RealType>(0.00001525878906250000000000000000), // k=8  p = 0.25
    tolerance);

    binomial_distribution<RealType> dist(static_cast<RealType>(8), static_cast<RealType>(0.25));
    RealType x = static_cast<RealType>(0.125);
    using namespace std; // ADL of std names.
    // mean:
    BOOST_CHECK_CLOSE(
       mean(dist)
       , static_cast<RealType>(8 * 0.25), tol2);
    // variance:
    BOOST_CHECK_CLOSE(
       variance(dist)
       , static_cast<RealType>(8 * 0.25 * 0.75), tol2);
    // std deviation:
    BOOST_CHECK_CLOSE(
       standard_deviation(dist)
       , static_cast<RealType>(sqrt(8 * 0.25L * 0.75L)), tol2);
    // hazard:
    BOOST_CHECK_CLOSE(
       hazard(dist, x)
       , pdf(dist, x) / cdf(complement(dist, x)), tol2);
    // cumulative hazard:
    BOOST_CHECK_CLOSE(
       chf(dist, x)
       , -log(cdf(complement(dist, x))), tol2);
    // coefficient_of_variation:
    BOOST_CHECK_CLOSE(
       coefficient_of_variation(dist)
       , standard_deviation(dist) / mean(dist), tol2);
    // mode:
    BOOST_CHECK_CLOSE(
       mode(dist)
       , static_cast<RealType>(std::floor(9 * 0.25)), tol2);
    // skewness:
    BOOST_CHECK_CLOSE(
       skewness(dist)
       , static_cast<RealType>(0.40824829046386301636621401245098L), (std::max)(tol2, static_cast<RealType>(5e-29))); // test data has 32 digits only.
    // kurtosis:
    BOOST_CHECK_CLOSE(
       kurtosis(dist)
       , static_cast<RealType>(2.916666666666666666666666666666666666L), tol2);
    // kurtosis excess:
    BOOST_CHECK_CLOSE(
       kurtosis_excess(dist)
       , static_cast<RealType>(-0.08333333333333333333333333333333333333L), tol2);
    // Check kurtosis_excess == kurtosis -3;
      BOOST_CHECK_EQUAL(kurtosis(dist), static_cast<RealType>(3) + kurtosis_excess(dist));

    // special cases for PDF:
    BOOST_CHECK_EQUAL(
       pdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0)),
          static_cast<RealType>(0)), static_cast<RealType>(1)
       );
    BOOST_CHECK_EQUAL(
       pdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0)),
          static_cast<RealType>(0.0001)), static_cast<RealType>(0)
       );
    BOOST_CHECK_EQUAL(
       pdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(1)),
          static_cast<RealType>(0.001)), static_cast<RealType>(0)
       );
    BOOST_CHECK_EQUAL(
       pdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(1)),
          static_cast<RealType>(8)), static_cast<RealType>(1)
       );
    BOOST_CHECK_EQUAL(
       pdf(
          binomial_distribution<RealType>(static_cast<RealType>(0), static_cast<RealType>(0.25)),
          static_cast<RealType>(0)), static_cast<RealType>(1)
       );
    BOOST_CHECK_THROW(
       pdf(
          binomial_distribution<RealType>(static_cast<RealType>(-1), static_cast<RealType>(0.25)),
          static_cast<RealType>(0)), std::domain_error
       );
    BOOST_CHECK_THROW(
       pdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(-0.25)),
          static_cast<RealType>(0)), std::domain_error
       );
    BOOST_CHECK_THROW(
       pdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(1.25)),
          static_cast<RealType>(0)), std::domain_error
       );
    BOOST_CHECK_THROW(
       pdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
          static_cast<RealType>(-1)), std::domain_error
       );
    BOOST_CHECK_THROW(
       pdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
          static_cast<RealType>(9)), std::domain_error
       );
    BOOST_CHECK_THROW(
       cdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
          static_cast<RealType>(-1)), std::domain_error
       );
    BOOST_CHECK_THROW(
       cdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
          static_cast<RealType>(9)), std::domain_error
       );
    BOOST_CHECK_THROW(
       cdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(-0.25)),
          static_cast<RealType>(0)), std::domain_error
       );
    BOOST_CHECK_THROW(
       cdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(1.25)),
          static_cast<RealType>(0)), std::domain_error
       );
    BOOST_CHECK_THROW(
       quantile(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(-0.25)),
          static_cast<RealType>(0)), std::domain_error
       );
    BOOST_CHECK_THROW(
       quantile(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(1.25)),
          static_cast<RealType>(0)), std::domain_error
       );

    BOOST_CHECK_EQUAL(
       quantile(
          binomial_distribution<RealType>(static_cast<RealType>(16), static_cast<RealType>(0.25)),
          static_cast<RealType>(0.01)), // Less than cdf == pdf(binomial_distribution<RealType>(16, 0.25), 0)
          static_cast<RealType>(0) // so expect zero as best approximation.
       );

    BOOST_CHECK_EQUAL(
       cdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
          static_cast<RealType>(8)), static_cast<RealType>(1)
       );
    BOOST_CHECK_EQUAL(
       cdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0)),
          static_cast<RealType>(7)), static_cast<RealType>(1)
       );
    BOOST_CHECK_EQUAL(
       cdf(
          binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(1)),
          static_cast<RealType>(7)), static_cast<RealType>(0)
       );

#endif

  {
    // This is a visual sanity check that everything is OK:
    binomial_distribution<RealType> my8dist(8., 0.25); // Note: double values (matching the distribution definition) avoid the need for any casting.
    //cout << "mean(my8dist) = " << boost::math::mean(my8dist) << endl; // mean(my8dist) = 2
    //cout << "my8dist.trials() = " << my8dist.trials()  << endl; // my8dist.trials() = 8
    //cout << "my8dist.success_fraction() = " << my8dist.success_fraction()  << endl; // my8dist.success_fraction() = 0.25
    BOOST_CHECK_CLOSE(my8dist.trials(), static_cast<RealType>(8), tol2);
    BOOST_CHECK_CLOSE(my8dist.success_fraction(), static_cast<RealType>(0.25), tol2);

   //{
   //   int n = static_cast<int>(boost::math::tools::real_cast<double>(my8dist.trials()));
   //   RealType sumcdf = 0.;
   //   for (int k = 0; k <= n; k++)
   //   {
   //     cout << k << ' ' << pdf(my8dist, static_cast<RealType>(k));
   //     sumcdf += pdf(my8dist, static_cast<RealType>(k));
   //     cout  << ' '  << sumcdf;
   //     cout << ' ' << cdf(my8dist, static_cast<RealType>(k));
   //     cout << ' ' << sumcdf - cdf(my8dist, static_cast<RealType>(k)) << endl;
   //   } // for k
   // }
    // n = 8, p =0.25
    //k         pdf              cdf
    //0 0.1001129150390625 0.1001129150390625
    //1 0.26696777343749994 0.36708068847656244
    //2 0.31146240234375017 0.67854309082031261
    //3 0.20764160156249989 0.8861846923828125
    //4 0.086517333984375 0.9727020263671875
    //5 0.023071289062499997 0.9957733154296875
    //6 0.0038452148437500009 0.9996185302734375
    //7 0.00036621093749999984 0.9999847412109375
    //8 1.52587890625e-005 1 1 0
  }
#define T RealType
#include "binomial_quantile.ipp"

  for(unsigned i = 0; i < binomial_quantile_data.size(); ++i)
  {
     using namespace boost::math::policies;
     RealType tol = boost::math::tools::epsilon<RealType>() * 500;
     if(!boost::is_floating_point<RealType>::value)
        tol *= 10;  // no lanczos approximation implies less accuracy
     RealType x;
#if !defined(TEST_ROUNDING) || (TEST_ROUNDING == 1)
     //
     // Check full real value first:
     //
     typedef policy<discrete_quantile<boost::math::policies::real> > P1;
     binomial_distribution<RealType, P1> p1(binomial_quantile_data[i][0], binomial_quantile_data[i][1]);
     x = quantile(p1, binomial_quantile_data[i][2]);
     BOOST_CHECK_CLOSE_FRACTION(x, (RealType)binomial_quantile_data[i][3], tol);
     x = quantile(complement(p1, (RealType)binomial_quantile_data[i][2]));
     BOOST_CHECK_CLOSE_FRACTION(x, (RealType)binomial_quantile_data[i][4], tol);
#endif
#if !defined(TEST_ROUNDING) || (TEST_ROUNDING == 2)
     //
     // Now with round down to integer:
     //
     typedef policy<discrete_quantile<integer_round_down> > P2;
     binomial_distribution<RealType, P2> p2(binomial_quantile_data[i][0], binomial_quantile_data[i][1]);
     x = quantile(p2, binomial_quantile_data[i][2]);
     BOOST_CHECK_EQUAL(x, (RealType)floor(binomial_quantile_data[i][3]));
     x = quantile(complement(p2, binomial_quantile_data[i][2]));
     BOOST_CHECK_EQUAL(x, (RealType)floor(binomial_quantile_data[i][4]));
#endif
#if !defined(TEST_ROUNDING) || (TEST_ROUNDING == 3)
     //
     // Now with round up to integer:
     //
     typedef policy<discrete_quantile<integer_round_up> > P3;
     binomial_distribution<RealType, P3> p3(binomial_quantile_data[i][0], binomial_quantile_data[i][1]);
     x = quantile(p3, binomial_quantile_data[i][2]);
     BOOST_CHECK_EQUAL(x, (RealType)ceil(binomial_quantile_data[i][3]));
     x = quantile(complement(p3, binomial_quantile_data[i][2]));
     BOOST_CHECK_EQUAL(x, (RealType)ceil(binomial_quantile_data[i][4]));
#endif
#if !defined(TEST_ROUNDING) || (TEST_ROUNDING == 4)
     //
     // Now with round to integer "outside":
     //
     typedef policy<discrete_quantile<integer_round_outwards> > P4;
     binomial_distribution<RealType, P4> p4(binomial_quantile_data[i][0], binomial_quantile_data[i][1]);
     x = quantile(p4, binomial_quantile_data[i][2]);
     BOOST_CHECK_EQUAL(x, (RealType)(binomial_quantile_data[i][2] < 0.5f ? floor(binomial_quantile_data[i][3]) : ceil(binomial_quantile_data[i][3])));
     x = quantile(complement(p4, binomial_quantile_data[i][2]));
     BOOST_CHECK_EQUAL(x, (RealType)(binomial_quantile_data[i][2] < 0.5f ? ceil(binomial_quantile_data[i][4]) : floor(binomial_quantile_data[i][4])));
#endif
#if !defined(TEST_ROUNDING) || (TEST_ROUNDING == 5)
     //
     // Now with round to integer "inside":
     //
     typedef policy<discrete_quantile<integer_round_inwards> > P5;
     binomial_distribution<RealType, P5> p5(binomial_quantile_data[i][0], binomial_quantile_data[i][1]);
     x = quantile(p5, binomial_quantile_data[i][2]);
     BOOST_CHECK_EQUAL(x, (RealType)(binomial_quantile_data[i][2] < 0.5f ? ceil(binomial_quantile_data[i][3]) : floor(binomial_quantile_data[i][3])));
     x = quantile(complement(p5, binomial_quantile_data[i][2]));
     BOOST_CHECK_EQUAL(x, (RealType)(binomial_quantile_data[i][2] < 0.5f ? floor(binomial_quantile_data[i][4]) : ceil(binomial_quantile_data[i][4])));
#endif
#if !defined(TEST_ROUNDING) || (TEST_ROUNDING == 6)
     //
     // Now with round to nearest integer:
     //
     typedef policy<discrete_quantile<integer_round_nearest> > P6;
     binomial_distribution<RealType, P6> p6(binomial_quantile_data[i][0], binomial_quantile_data[i][1]);
     x = quantile(p6, binomial_quantile_data[i][2]);
     BOOST_CHECK_EQUAL(x, (RealType)(floor(binomial_quantile_data[i][3] + 0.5f)));
     x = quantile(complement(p6, binomial_quantile_data[i][2]));
     BOOST_CHECK_EQUAL(x, (RealType)(floor(binomial_quantile_data[i][4] + 0.5f)));
#endif
  }

   check_out_of_range<boost::math::binomial_distribution<RealType> >(1, 1); // (All) valid constructor parameter values.

#if !(defined(__clang__) && defined( __APPLE__))
   // See bug reported here: https://github.com/boostorg/math/pull/1007
   //
   // This test case is so extreme that the incomplete beta basically gets
   // no digits in the result correct, as a result expect some failures on
   // some platforms.
   {
      using namespace boost::math::policies;
      typedef policy<discrete_quantile<integer_round_outwards> > Policy;
      binomial_distribution<RealType, Policy> dist(9079765771874083840, 0.561815);
      // Accuracy is not too important here; the main purpose is to
      // make sure it is not stuck.
      BOOST_CHECK_CLOSE(quantile(dist, 0.0365346), 5101148604445670400, 1e12);
   }
#endif
} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{
   BOOST_MATH_CONTROL_FP;
   // Check that can generate binomial distribution using one convenience methods:
   binomial_distribution<> mybn2(1., 0.5); // Using default RealType double.
  // but that
   // boost::math::binomial mybn1(1., 0.5); // Using typedef fails
  // error C2039: 'binomial' : is not a member of 'boost::math'

  // Basic sanity-check spot values.

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
#if !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
#ifdef TEST_REAL_CONCEPT
  test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif

} // BOOST_AUTO_TEST_CASE( test_main )

/*

Output is:

  Description: Autorun "J:\Cpp\MathToolkit\test\Math_test\Debug\test_binomial.exe"
  Running 1 test case...
  Tolerance for type float is 0.0119209 %
  Tolerance for type double is 2.22045e-011 %
  Tolerance for type long double is 2.22045e-011 %
  Tolerance for type class boost::math::concepts::real_concept is 2.22045e-011 %

  *** No errors detected

========== Build: 1 succeeded, 0 failed, 0 up-to-date, 0 skipped ==========


*/
