//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_OVERFLOW_ERROR_POLICY
#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#endif

#include <boost/math/concepts/real_concept.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/distributions/non_central_t.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"
#include "test_out_of_range.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#define BOOST_CHECK_CLOSE_EX(a, b, prec, i) \
      {\
      unsigned int failures = boost::unit_test::results_collector.results( boost::unit_test::framework::current_test_case().p_id ).p_assertions_failed;\
      BOOST_CHECK_CLOSE(a, b, prec); \
      if(failures != boost::unit_test::results_collector.results( boost::unit_test::framework::current_test_case().p_id ).p_assertions_failed)\
            {\
         std::cerr << "Failure was at row " << i << std::endl;\
         std::cerr << std::setprecision(35); \
         std::cerr << "{ " << data[i][0] << " , " << data[i][1] << " , " << data[i][2];\
         std::cerr << " , " << data[i][3] << " , " << data[i][4] << " } " << std::endl;\
            }\
      }

#define BOOST_CHECK_EX(a, i) \
      {\
      unsigned int failures = boost::unit_test::results_collector.results( boost::unit_test::framework::current_test_case().p_id ).p_assertions_failed;\
      BOOST_CHECK(a); \
      if(failures != boost::unit_test::results_collector.results( boost::unit_test::framework::current_test_case().p_id ).p_assertions_failed)\
            {\
         std::cerr << "Failure was at row " << i << std::endl;\
         std::cerr << std::setprecision(35); \
         std::cerr << "{ " << data[i][0] << " , " << data[i][1] << " , " << data[i][2];\
         std::cerr << " , " << data[i][3] << " , " << data[i][4] << " } " << std::endl;\
            }\
      }

template <class RealType>
RealType naive_pdf(RealType, RealType, RealType)
{
}

template <class RealType>
RealType naive_mean(RealType v, RealType delta)
{
   using boost::math::tgamma;
   return delta * sqrt(v / 2) * tgamma((v - 1) / 2) / tgamma(v / 2);
}

float naive_mean(float v, float delta)
{
   return (float)naive_mean((double)v, (double)delta);
}

template <class RealType>
RealType naive_variance(RealType v, RealType delta)
{
   using boost::math::tgamma;
   RealType r = tgamma((v - 1) / 2) / tgamma(v / 2);
   r *= r;
   r *= -delta * delta * v / 2;
   r += (1 + delta * delta) * v / (v - 2);
   return r;
}

float naive_variance(float v, float delta)
{
   return (float)naive_variance((double)v, (double)delta);
}

template <class RealType>
RealType naive_skewness(RealType v, RealType delta)
{
   using boost::math::tgamma;
   RealType tgr = tgamma((v - 1) / 2) / tgamma(v / 2);
   RealType r = delta * sqrt(v) * tgamma((v - 1) / 2)
      * (v * (-3 + delta * delta + 2 * v) / ((-3 + v) * (-2 + v))
      - 2 * ((1 + delta * delta) * v / (-2 + v) - delta * delta * v * tgr * tgr / 2));
   r /= boost::math::constants::root_two<RealType>()
      * pow(((1 + delta*delta) * v / (-2 + v) - delta*delta*v*tgr*tgr / 2), RealType(1.5f))
      * tgamma(v / 2);
   return r;
}

float naive_skewness(float v, float delta)
{
   return (float)naive_skewness((double)v, (double)delta);
}

template <class RealType>
RealType naive_kurtosis_excess(RealType v, RealType delta)
{
   using boost::math::tgamma;
   RealType tgr = tgamma((v - 1) / 2) / tgamma(v / 2);
   RealType r = -delta * delta * v * tgr * tgr / 2;
   r *= v * (delta * delta * (1 + v) + 3 * (-5 + 3 * v)) / ((-3 + v)*(-2 + v))
      - 3 * ((1 + delta * delta) * v / (-2 + v) - delta * delta * v * tgr * tgr / 2);
   r += (3 + 6 * delta * delta + delta * delta * delta * delta)* v * v
      / ((-4 + v) * (-2 + v));
   r /= (1 + delta*delta)*v / (-2 + v) - delta*delta*v *tgr*tgr / 2;
   r /= (1 + delta*delta)*v / (-2 + v) - delta*delta*v *tgr*tgr / 2;
   return r - static_cast<RealType>(3);
}

float naive_kurtosis_excess(float v, float delta)
{
   return (float)naive_kurtosis_excess((double)v, (double)delta);
}

template <class RealType>
void test_spot(
   RealType df,    // Degrees of freedom
   RealType ncp,   // non-centrality param
   RealType t,     // T statistic
   RealType P,     // CDF
   RealType Q,     // Complement of CDF
   RealType tol)   // Test tolerance
{
   // An extra fudge factor for real_concept which has a less accurate tgamma:
   RealType tolerance_tgamma_extra = std::numeric_limits<RealType>::is_specialized ? 1 : 5;

   boost::math::non_central_t_distribution<RealType> dist(df, ncp);
   BOOST_CHECK_CLOSE(
      cdf(dist, t), P, tol);
#ifndef BOOST_NO_EXCEPTIONS
   try{
      BOOST_CHECK_CLOSE(
         mean(dist), naive_mean(df, ncp), tol);
      BOOST_CHECK_CLOSE(
         variance(dist), naive_variance(df, ncp), tol);
      BOOST_CHECK_CLOSE(
         skewness(dist), naive_skewness(df, ncp), tol * 10 * tolerance_tgamma_extra);
      BOOST_CHECK_CLOSE(
         kurtosis_excess(dist), naive_kurtosis_excess(df, ncp), tol * 350 * tolerance_tgamma_extra);
      BOOST_CHECK_CLOSE(
         kurtosis(dist), 3 + naive_kurtosis_excess(df, ncp), tol * 50 * tolerance_tgamma_extra);
   }
   catch(const std::domain_error&)
   {
   }
#endif
   /*
   BOOST_CHECK_CLOSE(
   pdf(dist, t), naive_pdf(dist.degrees_of_freedom(), ncp, t), tol * 50);
   */
   if((P < 0.99) && (Q < 0.99))
   {
      //
      // We can only check this if P is not too close to 1,
      // so that we can guarantee Q is reasonably free of error:
      //
      BOOST_CHECK_CLOSE(
         cdf(complement(dist, t)), Q, tol);
      BOOST_CHECK_CLOSE(
         quantile(dist, P), t, tol * 10);
      BOOST_CHECK_CLOSE(
         quantile(complement(dist, Q)), t, tol * 10);
      /*  Removed because can give more than one solution.
      BOOST_CHECK_CLOSE(
      dist.find_degrees_of_freedom(ncp, t, P), df, tol * 10);
      BOOST_CHECK_CLOSE(
      dist.find_degrees_of_freedom(boost::math::complement(ncp, t, Q)), df, tol * 10);
      BOOST_CHECK_CLOSE(
      dist.find_non_centrality(df, t, P), ncp, tol * 10);
      BOOST_CHECK_CLOSE(
      dist.find_non_centrality(boost::math::complement(df, t, Q)), ncp, tol * 10);
      */
   }
}

template <class RealType> // Any floating-point type RealType.
void test_spots(RealType)
{
   using namespace std;
   //
   // Approx limit of test data is 12 digits expressed here as a percentage:
   //
   RealType tolerance = (std::max)(
      boost::math::tools::epsilon<RealType>(),
      (RealType)5e-12f) * 100;
   //
   // At float precision we need to up the tolerance, since
   // the input values are rounded off to inexact quantities
   // the results get thrown off by a noticeable amount.
   //
   if(boost::math::tools::digits<RealType>() < 50)
      tolerance *= 50;
   if(boost::is_floating_point<RealType>::value != 1)
      tolerance *= 20; // real_concept special functions are less accurate

   cout << "Tolerance = " << tolerance << "%." << endl;

   //
   // Test data is taken from:
   //
   // Computing discrete mixtures of continuous
   // distributions: noncentral chisquare, noncentral t
   // and the distribution of the square of the sample
   // multiple correlation coefficient.
   // Denise Benton, K. Krishnamoorthy.
   // Computational Statistics & Data Analysis 43 (2003) 249 - 267
   //
   test_spot(
      static_cast<RealType>(3),   // degrees of freedom
      static_cast<RealType>(1),   // non centrality
      static_cast<RealType>(2.34),   // T
      static_cast<RealType>(0.801888999613917),       // Probability of result (CDF), P
      static_cast<RealType>(1 - 0.801888999613917),           // Q = 1 - P
      tolerance);
   test_spot(
      static_cast<RealType>(126),   // degrees of freedom
      static_cast<RealType>(-2),   // non centrality
      static_cast<RealType>(-4.33),   // T
      static_cast<RealType>(1.252846196792878e-2),       // Probability of result (CDF), P
      static_cast<RealType>(1 - 1.252846196792878e-2),           // Q = 1 - P
      tolerance);
   test_spot(
      static_cast<RealType>(20),   // degrees of freedom
      static_cast<RealType>(23),   // non centrality
      static_cast<RealType>(23),   // T
      static_cast<RealType>(0.460134400391924),       // Probability of result (CDF), P
      static_cast<RealType>(1 - 0.460134400391924),           // Q = 1 - P
      tolerance);
   test_spot(
      static_cast<RealType>(20),   // degrees of freedom
      static_cast<RealType>(33),   // non centrality
      static_cast<RealType>(34),   // T
      static_cast<RealType>(0.532008386378725),       // Probability of result (CDF), P
      static_cast<RealType>(1 - 0.532008386378725),           // Q = 1 - P
      tolerance);
   test_spot(
      static_cast<RealType>(12),   // degrees of freedom
      static_cast<RealType>(38),   // non centrality
      static_cast<RealType>(39),   // T
      static_cast<RealType>(0.495868184917805),       // Probability of result (CDF), P
      static_cast<RealType>(1 - 0.495868184917805),           // Q = 1 - P
      tolerance);
   test_spot(
      static_cast<RealType>(12),   // degrees of freedom
      static_cast<RealType>(39),   // non centrality
      static_cast<RealType>(39),   // T
      static_cast<RealType>(0.446304024668836),       // Probability of result (CDF), P
      static_cast<RealType>(1 - 0.446304024668836),           // Q = 1 - P
      tolerance);
   test_spot(
      static_cast<RealType>(200),   // degrees of freedom
      static_cast<RealType>(38),   // non centrality
      static_cast<RealType>(39),   // T
      static_cast<RealType>(0.666194209961795),       // Probability of result (CDF), P
      static_cast<RealType>(1 - 0.666194209961795),           // Q = 1 - P
      tolerance);
   test_spot(
      static_cast<RealType>(200),   // degrees of freedom
      static_cast<RealType>(42),   // non centrality
      static_cast<RealType>(40),   // T
      static_cast<RealType>(0.179292265426085),       // Probability of result (CDF), P
      static_cast<RealType>(1 - 0.179292265426085),           // Q = 1 - P
      tolerance);

   // From https://svn.boost.org/trac/boost/ticket/10480.
   // Test value from Mathematica N[CDF[NoncentralStudentTDistribution[2, 4], 5], 35]:
   test_spot(
      static_cast<RealType>(2),   // degrees of freedom
      static_cast<RealType>(4),   // non centrality
      static_cast<RealType>(5),   // T
      static_cast<RealType>(0.53202069866995310466912357978934321L),       // Probability of result (CDF), P
      static_cast<RealType>(1 - 0.53202069866995310466912357978934321L),           // Q = 1 - P
      tolerance);

   /* This test fails
   "Result of tgamma is too large to represent" at naive_mean check for max and infinity.
   if (std::numeric_limits<RealType>::has_infinity)
   {
   test_spot(
   //static_cast<RealType>(std::numeric_limits<RealType>::infinity()),   // degrees of freedom
   static_cast<RealType>((std::numeric_limits<RealType>::max)()),   // degrees of freedom
   static_cast<RealType>(10),   // non centrality
   static_cast<RealType>(11),   // T
   static_cast<RealType>(0.84134474606854293),       // Probability of result (CDF), P
   static_cast<RealType>(0.15865525393145707),           // Q = 1 - P
   tolerance);
   }
   */

   boost::math::non_central_t_distribution<RealType> dist(static_cast<RealType>(8), static_cast<RealType>(12));
   BOOST_CHECK_CLOSE(pdf(dist, 12), static_cast<RealType>(1.235329715425894935157684607751972713457e-1L), tolerance);
   BOOST_CHECK_CLOSE(pdf(boost::math::non_central_t_distribution<RealType>(126, -2), -4), static_cast<RealType>(5.797932289365814702402873546466798025787e-2L), tolerance);
   BOOST_CHECK_CLOSE(pdf(boost::math::non_central_t_distribution<RealType>(126, 2), 4), static_cast<RealType>(5.797932289365814702402873546466798025787e-2L), tolerance);
   BOOST_CHECK_CLOSE(pdf(boost::math::non_central_t_distribution<RealType>(126, 2), 0), static_cast<RealType>(5.388394890639957139696546086044839573749e-2L), tolerance);

   // Tests ultimately derived from https://github.com/scipy/scipy/issues/20693
   BOOST_CHECK_CLOSE(pdf(boost::math::non_central_t_distribution<RealType>(8, 16), 0), static_cast<RealType>(9.9467084610854116569233495190046171e-57L), tolerance);
   BOOST_CHECK_CLOSE(pdf(boost::math::non_central_t_distribution<RealType>(8, 16), boost::math::tools::min_value<RealType>()), static_cast<RealType>(9.9467084610854116569233495190046171e-57L), tolerance);
   BOOST_CHECK_CLOSE(pdf(boost::math::non_central_t_distribution<RealType>(8, 16), -boost::math::tools::min_value<RealType>()), static_cast<RealType>(9.9467084610854116569233495190046171e-57L), tolerance);
   BOOST_CHECK_CLOSE(pdf(boost::math::non_central_t_distribution<RealType>(8, 16), -0.125), static_cast<RealType>(1.4095889399390926611629593059778035e-57L), tolerance);
   BOOST_CHECK_CLOSE(pdf(boost::math::non_central_t_distribution<RealType>(8, 16), -1e-16), static_cast<RealType>(9.9467084610853952383141848485633491e-57L), tolerance);
   BOOST_CHECK_CLOSE(pdf(boost::math::non_central_t_distribution<RealType>(8, 16), 1e-16), static_cast<RealType>(9.9467084610854280755325141894744198e-57L), tolerance);
   BOOST_CHECK_CLOSE(pdf(boost::math::non_central_t_distribution<RealType>(8, 16), 0.125), static_cast<RealType>(8.7874127030564572234218759603362e-56L), tolerance);
   BOOST_CHECK_CLOSE(cdf(boost::math::non_central_t_distribution<RealType>(8, 16), 0), static_cast<RealType>(6.3887544005380872812754825749176666e-58L), tolerance);
   BOOST_CHECK_CLOSE(cdf(boost::math::non_central_t_distribution<RealType>(8, 16), boost::math::tools::min_value<RealType>()), static_cast<RealType>(6.3887544005380872812754825749176666e-58L), tolerance);
   BOOST_CHECK_CLOSE(cdf(boost::math::non_central_t_distribution<RealType>(8, 16), -boost::math::tools::min_value<RealType>()), static_cast<RealType>(6.3887544005380872812754825749176666e-58L), tolerance);
   BOOST_CHECK_CLOSE(cdf(boost::math::non_central_t_distribution<RealType>(8, 16), -0.125), static_cast<RealType>(1.0189377690928162394097857383628309e-58L), tolerance);
   BOOST_CHECK_CLOSE(cdf(boost::math::non_central_t_distribution<RealType>(8, 16), -1e-16), static_cast<RealType>(6.3887544005380773345670214895144269e-58L), tolerance);
   BOOST_CHECK_CLOSE(cdf(boost::math::non_central_t_distribution<RealType>(8, 16), 1e-16), static_cast<RealType>(6.3887544005380972279839436603373249e-58L), tolerance);
   BOOST_CHECK_CLOSE(cdf(boost::math::non_central_t_distribution<RealType>(8, 16), 0.125), static_cast<RealType>(5.0299048839141484925784179651886214e-57L), tolerance);

   // Error handling checks:
   //check_out_of_range<boost::math::non_central_t_distribution<RealType> >(1, 1);  // Fails one check because df for this distribution *can* be infinity.
   BOOST_MATH_CHECK_THROW(pdf(boost::math::non_central_t_distribution<RealType>(0, 1), 0), std::domain_error);
   BOOST_MATH_CHECK_THROW(pdf(boost::math::non_central_t_distribution<RealType>(-1, 1), 0), std::domain_error);
   BOOST_MATH_CHECK_THROW(quantile(boost::math::non_central_t_distribution<RealType>(1, 1), -1), std::domain_error);
   BOOST_MATH_CHECK_THROW(quantile(boost::math::non_central_t_distribution<RealType>(1, 1), 2), std::domain_error);
   //
   // Some special error handling tests, if the non-centrality param is too large
   // then we have no evaluation method and should get a domain_error:
   //
   using std::ldexp;
   using distro1 = boost::math::non_central_t_distribution<RealType>;
   using distro2 = boost::math::non_central_t_distribution<RealType, boost::math::policies::policy<boost::math::policies::domain_error<boost::math::policies::ignore_error>>>;
   using de = std::domain_error;
   BOOST_MATH_CHECK_THROW(distro1(2, ldexp(RealType(1), 100)), de);
   if (std::numeric_limits<RealType>::has_quiet_NaN)
   {
      distro2 d2(2, ldexp(RealType(1), 100));
      BOOST_CHECK(boost::math::isnan(pdf(d2, 0.5)));
      BOOST_CHECK(boost::math::isnan(cdf(d2, 0.5)));
   }

   // Bug cases, 
   // https://github.com/scipy/scipy/issues/19348
   //
   {
      distro1 d(8.0f, 8.5f);
      BOOST_CHECK_CLOSE(pdf(d, -1), static_cast<RealType>(6.1747948083757028903541988987716621647020752431287e-20), 2e-5);  // Can we do better on accuracy here?
   }

} // template <class RealType>void test_spots(RealType)

template <class T>
T nct_cdf(T df, T nc, T x)
{
   return cdf(boost::math::non_central_t_distribution<T>(df, nc), x);
}

template <class T>
T nct_pdf(T df, T nc, T x)
{
   return pdf(boost::math::non_central_t_distribution<T>(df, nc), x);
}

template <class T>
T nct_ccdf(T df, T nc, T x)
{
   return cdf(complement(boost::math::non_central_t_distribution<T>(df, nc), x));
}

template <typename Real, typename T>
void do_test_nc_t(T& data, const char* type_name, const char* test)
{
   typedef Real                   value_type;

   std::cout << "Testing: " << test << std::endl;

#ifdef NC_T_CDF_FUNCTION_TO_TEST
   value_type(*fp1)(value_type, value_type, value_type) = NC_T_CDF_FUNCTION_TO_TEST;
#else
   value_type(*fp1)(value_type, value_type, value_type) = nct_cdf;
#endif
   boost::math::tools::test_result<value_type> result;

#if !(defined(ERROR_REPORTING_MODE) && !defined(NC_T_CDF_FUNCTION_TO_TEST))
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(fp1, 0, 1, 2),
      extract_result<Real>(3));
   handle_test_result(result, data[result.worst()], result.worst(),
      type_name, "non central t CDF", test);
#endif

#if !(defined(ERROR_REPORTING_MODE) && !defined(NC_T_CCDF_FUNCTION_TO_TEST))
#ifdef NC_T_CCDF_FUNCTION_TO_TEST
   fp1 = NC_T_CCDF_FUNCTION_TO_TEST;
#else
   fp1 = nct_ccdf;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(fp1, 0, 1, 2),
      extract_result<Real>(4));
   handle_test_result(result, data[result.worst()], result.worst(),
      type_name, "non central t CDF complement", test);

   std::cout << std::endl;
#endif
}

template <typename Real, typename T>
void do_test_nc_t_pdf(T& data, const char* type_name, const char* test)
{
   typedef Real                   value_type;

   std::cout << "Testing: " << test << std::endl;

   value_type(*fp1)(value_type, value_type, value_type) = nct_pdf;

   boost::math::tools::test_result<value_type> result;

   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(fp1, 0, 1, 2),
      extract_result<Real>(3));
   handle_test_result(result, data[result.worst()], result.worst(),
      type_name, "non central t PDF", test);

   std::cout << std::endl;
}

template <typename Real, typename T>
void quantile_sanity_check(T& data, const char* type_name, const char* test)
{
#ifndef ERROR_REPORTING_MODE
   typedef Real                   value_type;

   //
   // Tests with type real_concept take rather too long to run, so
   // for now we'll disable them:
   //
   if(!boost::is_floating_point<value_type>::value)
      return;

   std::cout << "Testing: " << type_name << " quantile sanity check, with tests " << test << std::endl;

   //
   // These sanity checks test for a round trip accuracy of one half
   // of the bits in T, unless T is type float, in which case we check
   // for just one decimal digit.  The problem here is the sensitivity
   // of the functions, not their accuracy.  This test data was generated
   // for the forward functions, which means that when it is used as
   // the input to the inverses then it is necessarily inexact.  This rounding
   // of the input is what makes the data unsuitable for use as an accuracy check,
   // and also demonstrates that you can't in general round-trip these functions.
   // It is however a useful sanity check.
   //
   value_type precision = static_cast<value_type>(ldexp(1.0, 1 - boost::math::policies::digits<value_type, boost::math::policies::policy<> >() / 2)) * 100;
   if(boost::math::policies::digits<value_type, boost::math::policies::policy<> >() < 50)
      precision = 1;   // 1% or two decimal digits, all we can hope for when the input is truncated to float

   for(unsigned i = 0; i < data.size(); ++i)
   {
      if(data[i][3] == 0)
      {
         BOOST_CHECK(0 == quantile(boost::math::non_central_t_distribution<value_type>(data[i][0], data[i][1]), data[i][3]));
      }
      else if(data[i][3] < 0.9999f)
      {
         value_type p = quantile(boost::math::non_central_t_distribution<value_type>(data[i][0], data[i][1]), data[i][3]);
         value_type pt = data[i][2];
         BOOST_CHECK_CLOSE_EX(pt, p, precision, i);
      }
      if(data[i][4] == 0)
      {
         BOOST_CHECK(0 == quantile(complement(boost::math::non_central_t_distribution<value_type>(data[i][0], data[i][1]), data[i][3])));
      }
      else if(data[i][4] < 0.9999f)
      {
         value_type p = quantile(complement(boost::math::non_central_t_distribution<value_type>(data[i][0], data[i][1]), data[i][4]));
         value_type pt = data[i][2];
         BOOST_CHECK_CLOSE_EX(pt, p, precision, i);
      }
      if(boost::math::tools::digits<value_type>() > 50)
      {
         //
         // Sanity check mode, the accuracy of
         // the mode is at *best* the square root of the accuracy of the PDF:
         //
#ifndef BOOST_NO_EXCEPTIONS
         try{
            value_type m = mode(boost::math::non_central_t_distribution<value_type>(data[i][0], data[i][1]));
            value_type p = pdf(boost::math::non_central_t_distribution<value_type>(data[i][0], data[i][1]), m);
            value_type delta = (std::max)(fabs(m * sqrt(precision) * 50), sqrt(precision) * 50);
            BOOST_CHECK_EX(pdf(boost::math::non_central_t_distribution<value_type>(data[i][0], data[i][1]), m + delta) <= p, i);
            BOOST_CHECK_EX(pdf(boost::math::non_central_t_distribution<value_type>(data[i][0], data[i][1]), m - delta) <= p, i);
         }
         catch(const boost::math::evaluation_error&) {}
#endif
#if 0
         //
         // Sanity check degrees-of-freedom finder, don't bother at float
         // precision though as there's not enough data in the probability
         // values to get back to the correct degrees of freedom or
         // non-centrality parameter:
         //
         try{
            if((data[i][3] < 0.99) && (data[i][3] != 0))
            {
               BOOST_CHECK_CLOSE_EX(
                  boost::math::non_central_t_distribution<value_type>::find_degrees_of_freedom(data[i][1], data[i][2], data[i][3]),
                  data[i][0], precision, i);
               BOOST_CHECK_CLOSE_EX(
                  boost::math::non_central_t_distribution<value_type>::find_non_centrality(data[i][0], data[i][2], data[i][3]),
                  data[i][1], precision, i);
            }
            if((data[i][4] < 0.99) && (data[i][4] != 0))
            {
               BOOST_CHECK_CLOSE_EX(
                  boost::math::non_central_t_distribution<value_type>::find_degrees_of_freedom(boost::math::complement(data[i][1], data[i][2], data[i][4])),
                  data[i][0], precision, i);
               BOOST_CHECK_CLOSE_EX(
                  boost::math::non_central_t_distribution<value_type>::find_non_centrality(boost::math::complement(data[i][0], data[i][2], data[i][4])),
                  data[i][1], precision, i);
            }
         }
         catch(const std::exception& e)
         {
            BOOST_ERROR(e.what());
         }
#endif
      }
   }
#endif
}

template <typename T>
void test_accuracy(T, const char* type_name)
{
#include "nct.ipp"
   do_test_nc_t<T>(nct, type_name, "Non Central T");
   quantile_sanity_check<T>(nct, type_name, "Non Central T");
   if(std::numeric_limits<T>::is_specialized)
   {
      //
      // Don't run these tests for real_concept: they take too long and don't converge
      // without numeric_limits and lanczos support:
      //
#include "nct_small_delta.ipp"
      do_test_nc_t<T>(nct_small_delta, type_name, "Non Central T (small non-centrality)");
      quantile_sanity_check<T>(nct_small_delta, type_name, "Non Central T (small non-centrality)");
#include "nct_asym.ipp"
      do_test_nc_t<T>(nct_asym, type_name, "Non Central T (large parameters)");
      quantile_sanity_check<T>(nct_asym, type_name, "Non Central T (large parameters)");

#include "nc_t_pdf_data.ipp"
      do_test_nc_t_pdf<T>(nc_t_pdf_data, type_name, "Non Central T PDF");
   }
}


template <class RealType>
void test_big_df(RealType)
{
   using namespace boost::math;

   if(typeid(RealType) != typeid(boost::math::concepts::real_concept))
   { // Ordinary floats only.
      // Could also test if (std::numeric_limits<RealType>::is_specialized);

      RealType tolerance = 10 * boost::math::tools::epsilon<RealType>(); // static_cast<RealType>(1e-14); //
      std::cout.precision(17); // Note: need to reset after calling BOOST_CHECK_s
      // due to buglet in Boost.test that fails to restore precision correctly.

      // Test for large degrees of freedom when should be same as normal.
      RealType inf =
         (std::numeric_limits<RealType>::has_infinity) ?
         std::numeric_limits<RealType>::infinity()
         :
         boost::math::tools::max_value<RealType>();
      RealType nan = std::numeric_limits<RealType>::quiet_NaN();

      // Tests for df = max_value and infinity.
      RealType max_val = boost::math::tools::max_value<RealType>();
      non_central_t_distribution<RealType> maxdf(max_val, 0);
      BOOST_CHECK_EQUAL(maxdf.degrees_of_freedom(), max_val);

      non_central_t_distribution<RealType> infdf(inf, 0);
      BOOST_CHECK_EQUAL(infdf.degrees_of_freedom(), inf);
      BOOST_CHECK_EQUAL(mean(infdf), 0);
      BOOST_CHECK_EQUAL(mean(maxdf), 0);
      BOOST_CHECK_EQUAL(variance(infdf), 1);
      BOOST_CHECK_EQUAL(variance(maxdf), 1);
      BOOST_CHECK_EQUAL(skewness(infdf), 0);
      BOOST_CHECK_EQUAL(skewness(maxdf), 0);
      BOOST_CHECK_EQUAL(kurtosis_excess(infdf), 1);
      BOOST_CHECK_CLOSE_FRACTION(kurtosis_excess(maxdf), static_cast<RealType>(1), tolerance);

      // Bad df examples.
#ifndef BOOST_NO_EXCEPTIONS
      BOOST_MATH_CHECK_THROW(non_central_t_distribution<RealType> minfdf(-inf, 0), std::domain_error);
      BOOST_MATH_CHECK_THROW(non_central_t_distribution<RealType> minfdf(nan, 0), std::domain_error);
      BOOST_MATH_CHECK_THROW(non_central_t_distribution<RealType> minfdf(-nan, 0), std::domain_error);
#else
      BOOST_MATH_CHECK_THROW(non_central_t_distribution<RealType>(-inf, 0), std::domain_error);
      BOOST_MATH_CHECK_THROW(non_central_t_distribution<RealType>(nan, 0), std::domain_error);
      BOOST_MATH_CHECK_THROW(non_central_t_distribution<RealType>(-nan, 0), std::domain_error);
#endif


      // BOOST_CHECK_CLOSE_FRACTION(pdf(infdf, 0), static_cast<RealType>(0.3989422804014326779399460599343818684759L), tolerance);
      BOOST_CHECK_CLOSE_FRACTION(pdf(maxdf, 0), boost::math::constants::one_div_root_two_pi<RealType>(), tolerance);
      BOOST_CHECK_CLOSE_FRACTION(pdf(infdf, 0), boost::math::constants::one_div_root_two_pi<RealType>(), tolerance);
      BOOST_CHECK_CLOSE_FRACTION(cdf(infdf, 0), boost::math::constants::half<RealType>(), tolerance);
      BOOST_CHECK_CLOSE_FRACTION(cdf(maxdf, 0), boost::math::constants::half<RealType>(), tolerance);

      // non-centrality delta = 10
      // Degrees of freedom = Max value and  = infinity should be very close.
      non_central_t_distribution<RealType> maxdf10(max_val, 10);
      non_central_t_distribution<RealType> infdf10(inf, 10);
      BOOST_CHECK_EQUAL(infdf10.degrees_of_freedom(), inf);
      BOOST_CHECK_EQUAL(infdf10.non_centrality(), 10);
      BOOST_CHECK_EQUAL(mean(infdf10), 10);
      BOOST_CHECK_CLOSE_FRACTION(mean(maxdf10), static_cast<RealType>(10), tolerance);

      BOOST_CHECK_CLOSE_FRACTION(pdf(infdf10, 11), pdf(maxdf10, 11), tolerance); //

      BOOST_CHECK_CLOSE_FRACTION(cdf(complement(infdf10, 11)), 1 - cdf(infdf10, 11), tolerance); //
      BOOST_CHECK_CLOSE_FRACTION(cdf(complement(maxdf10, 11)), 1 - cdf(maxdf10, 11), tolerance); //
      BOOST_CHECK_CLOSE_FRACTION(cdf(complement(infdf10, 11)), 1 - cdf(maxdf10, 11), tolerance); //
      std::cout.precision(17);
      //std::cout  << "cdf(maxdf10, 11)  = " << cdf(maxdf10, 11) << ' ' << cdf(complement(maxdf10, 11)) << endl;
      //std::cout  << "cdf(infdf10, 11)  = " << cdf(infdf10, 11) << ' ' << cdf(complement(infdf10, 11)) << endl;
      //std::cout  << "quantile(maxdf10, 0.5)  = " << quantile(maxdf10, 0.5) << std::endl; // quantile(maxdf10, 0.5)  = 10.000000000000004
      //std::cout  << "quantile(infdf10, 0.5) = " << ' ' << quantile(infdf10, 0.5) << std::endl; // quantile(infdf10, 0.5) =  10

      BOOST_CHECK_CLOSE_FRACTION(quantile(infdf10, 0.5), static_cast<RealType>(10), tolerance);
      BOOST_CHECK_CLOSE_FRACTION(quantile(maxdf10, 0.5), static_cast<RealType>(10), tolerance);

      BOOST_TEST_MESSAGE("non_central_t_distribution<RealType> infdf100(inf, 100);");
      non_central_t_distribution<RealType> infdf100(inf, 100);
      BOOST_TEST_MESSAGE("non_central_t_distribution<RealType> maxdf100(max_val, 100);");
      non_central_t_distribution<RealType> maxdf100(max_val, 100);
      BOOST_TEST_MESSAGE("BOOST_CHECK_CLOSE_FRACTION(quantile(infdf100, 0.5), static_cast<RealType>(100), tolerance);");
      BOOST_CHECK_CLOSE_FRACTION(quantile(infdf100, 0.5), static_cast<RealType>(100), tolerance);
      BOOST_TEST_MESSAGE("BOOST_CHECK_CLOSE_FRACTION(quantile(maxdf100, 0.5), static_cast<RealType>(100), tolerance);");
      BOOST_CHECK_CLOSE_FRACTION(quantile(maxdf100, 0.5), static_cast<RealType>(100), tolerance);
      { // Loop back.
         RealType p = static_cast<RealType>(0.01);
         RealType x = quantile(infdf10, p);
         RealType c = cdf(infdf10, x);
         BOOST_CHECK_CLOSE_FRACTION(c, p, tolerance);
      }
    {
       RealType q = static_cast<RealType>(0.99);
       RealType x = quantile(complement(infdf10, q));
       RealType c = cdf(complement(infdf10, x));
       BOOST_CHECK_CLOSE_FRACTION(c, q, tolerance);
    }
    { // Loop back.
       RealType p = static_cast<RealType>(0.99);
       RealType x = quantile(infdf10, p);
       RealType c = cdf(infdf10, x);
       BOOST_CHECK_CLOSE_FRACTION(c, p, tolerance);
    }
    {
       RealType q = static_cast<RealType>(0.01);
       RealType x = quantile(complement(infdf10, q));
       RealType c = cdf(complement(infdf10, x));
       BOOST_CHECK_CLOSE_FRACTION(c, q, tolerance * 2); // c{0.0100000128} and q{0.00999999978}
    }

    //RealType cinf = quantile(infdf10, 0.25);
    //std::cout << cinf << ' ' << cdf(infdf10, cinf) << std::endl; // 9.32551 0.25

    //RealType cmax = quantile(maxdf10, 0.25);
    //std::cout << cmax << ' ' << cdf(maxdf10, cmax) << std::endl; //  9.32551 0.25

    //RealType cinfc = quantile(complement(infdf10, 0.75));
    //std::cout << cinfc << ' ' << cdf(infdf10, cinfc) << std::endl; // 9.32551 0.25

    //RealType cmaxc = quantile(complement(maxdf10, 0.75));
    //std::cout << cmaxc << ' ' << cdf(maxdf10, cmaxc) << std::endl; // 9.32551 0.25

    BOOST_CHECK_CLOSE_FRACTION(quantile(infdf10, 0.5), quantile(maxdf10, 0.5), tolerance); //
    BOOST_CHECK_CLOSE_FRACTION(quantile(infdf10, 0.2), quantile(maxdf10, 0.2), tolerance); //
    BOOST_CHECK_CLOSE_FRACTION(quantile(infdf10, 0.8), quantile(maxdf10, 0.8), tolerance); //

    BOOST_CHECK_CLOSE_FRACTION(quantile(infdf10, 0.25), quantile(complement(infdf10, 0.75)), tolerance); //
    BOOST_CHECK_CLOSE_FRACTION(quantile(complement(infdf10, 0.5)), quantile(complement(maxdf10, 0.5)), tolerance); //

    BOOST_CHECK_CLOSE_FRACTION(quantile(maxdf10, 0.25), quantile(complement(maxdf10, 0.75)), tolerance); //

    BOOST_CHECK_CLOSE_FRACTION(quantile(infdf10, 0.99), quantile(complement(infdf10, 0.01)), tolerance); //
    BOOST_CHECK_CLOSE_FRACTION(quantile(infdf10, 0.4), quantile(complement(infdf10, 0.6)), tolerance); //
    BOOST_CHECK_CLOSE_FRACTION(quantile(infdf10, 0.01), quantile(complement(infdf10, 1 - 0.01)), tolerance); //
   }
} // void test_big_df(RealType)

template <class RealType>
void test_ignore_policy(RealType)
{
   // Check on returns when errors are ignored.
   if((typeid(RealType) != typeid(boost::math::concepts::real_concept))
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

      typedef non_central_t_distribution<RealType, ignore_all_policy> ignore_error_non_central_t;

      // Only test NaN and infinity if type has these features (realconcept returns zero).
      // Integers are always converted to RealType,
      // others requires static cast to RealType from long double.

      if(std::numeric_limits<RealType>::has_quiet_NaN)
      {
         // Mean
         BOOST_CHECK((boost::math::isnan)(mean(ignore_error_non_central_t(-nan, 0))));
         BOOST_CHECK((boost::math::isnan)(mean(ignore_error_non_central_t(+nan, 0))));
         BOOST_CHECK((boost::math::isnan)(mean(ignore_error_non_central_t(-1, 0))));
         BOOST_CHECK((boost::math::isnan)(mean(ignore_error_non_central_t(0, 0))));
         BOOST_CHECK((boost::math::isnan)(mean(ignore_error_non_central_t(1, 0))));
         BOOST_CHECK((boost::math::isnan)(mean(ignore_error_non_central_t(2, nan))));
         BOOST_CHECK((boost::math::isnan)(mean(ignore_error_non_central_t(nan, nan))));
         BOOST_CHECK(boost::math::isfinite(mean(ignore_error_non_central_t(2, 0)))); // OK

         // Variance
         BOOST_CHECK((boost::math::isnan)(variance(ignore_error_non_central_t(nan, 0))));
         BOOST_CHECK((boost::math::isnan)(variance(ignore_error_non_central_t(1, nan))));
         BOOST_CHECK((boost::math::isnan)(variance(ignore_error_non_central_t(2, nan))));
         BOOST_CHECK((boost::math::isnan)(variance(ignore_error_non_central_t(-1, 0))));
         BOOST_CHECK((boost::math::isnan)(variance(ignore_error_non_central_t(0, 0))));
         BOOST_CHECK((boost::math::isnan)(variance(ignore_error_non_central_t(1, 0))));
         BOOST_CHECK((boost::math::isnan)(variance(ignore_error_non_central_t(static_cast<RealType>(1.7L), 0))));
         BOOST_CHECK((boost::math::isnan)(variance(ignore_error_non_central_t(2, 0))));

         // Skewness
         BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_non_central_t(std::numeric_limits<RealType>::quiet_NaN(), 0))));
         BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_non_central_t(-1, 0))));
         BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_non_central_t(0, 0))));
         BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_non_central_t(1, 0))));
         BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_non_central_t(2, 0))));
         BOOST_CHECK((boost::math::isnan)(skewness(ignore_error_non_central_t(3, 0))));

         // Kurtosis
         BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_non_central_t(std::numeric_limits<RealType>::quiet_NaN(), 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_non_central_t(-1, 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_non_central_t(0, 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_non_central_t(1, 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_non_central_t(2, 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_non_central_t(static_cast<RealType>(2.0001L), 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_non_central_t(3, 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis(ignore_error_non_central_t(4, 0))));

         // Kurtosis excess
         BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_non_central_t(std::numeric_limits<RealType>::quiet_NaN(), 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_non_central_t(-1, 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_non_central_t(0, 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_non_central_t(1, 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_non_central_t(2, 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_non_central_t(static_cast<RealType>(2.0001L), 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_non_central_t(3, 0))));
         BOOST_CHECK((boost::math::isnan)(kurtosis_excess(ignore_error_non_central_t(4, 0))));
      } // has_quiet_NaN
      BOOST_CHECK(boost::math::isfinite(mean(ignore_error_non_central_t(1 + std::numeric_limits<RealType>::epsilon(), 0))));
      BOOST_CHECK(boost::math::isfinite(variance(ignore_error_non_central_t(2 + 2 * std::numeric_limits<RealType>::epsilon(), 0))));
      BOOST_CHECK(boost::math::isfinite(variance(ignore_error_non_central_t(static_cast<RealType>(2.0001L), 0))));
      BOOST_CHECK(boost::math::isfinite(variance(ignore_error_non_central_t(2 + 2 * std::numeric_limits<RealType>::epsilon(), 0))));
      BOOST_CHECK(boost::math::isfinite(skewness(ignore_error_non_central_t(3 + 3 * std::numeric_limits<RealType>::epsilon(), 0))));
      BOOST_CHECK(boost::math::isfinite(kurtosis(ignore_error_non_central_t(4 + 4 * std::numeric_limits<RealType>::epsilon(), 0))));
      BOOST_CHECK(boost::math::isfinite(kurtosis(ignore_error_non_central_t(static_cast<RealType>(4.0001L), 0))));

      // check_out_of_range<non_central_t_distribution<RealType> >(1, 0); // Fails one check because allows df = infinity.
      check_support<non_central_t_distribution<RealType> >(non_central_t_distribution<RealType>(1, 0));
   } // ordinary floats.
} // template <class RealType> void test_ignore_policy(RealType)
