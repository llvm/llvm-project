//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_OVERFLOW_ERROR_POLICY
#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#endif
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#endif
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/distributions/non_central_beta.hpp> 
#include <boost/math/distributions/poisson.hpp> 
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

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

template <class T>
T nc_beta_cdf(T a, T b, T nc, T x)
{
#ifdef NC_BETA_CDF_FUNCTION_TO_TEST
   return NC_BETA_CDF_FUNCTION_TO_TEST(a, b, nc, x);
#else
   return cdf(boost::math::non_central_beta_distribution<T>(a, b, nc), x);
#endif
}

template <class T>
T nc_beta_ccdf(T a, T b, T nc, T x)
{
#ifdef NC_BETA_CCDF_FUNCTION_TO_TEST
   return NC_BETA_CCDF_FUNCTION_TO_TEST(a, b, nc, x);
#else
   return cdf(complement(boost::math::non_central_beta_distribution<T>(a, b, nc), x));
#endif
}

template <typename Real, typename T>
void do_test_nc_chi_squared(T& data, const char* type_name, const char* test)
{
   typedef Real                   value_type;

   std::cout << "Testing: " << test << std::endl;

   value_type(*fp1)(value_type, value_type, value_type, value_type) = nc_beta_cdf;
   boost::math::tools::test_result<value_type> result;

#if !(defined(ERROR_REPORTING_MODE) && !defined(NC_BETA_CDF_FUNCTION_TO_TEST))
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(fp1, 0, 1, 2, 3),
      extract_result<Real>(4));
   handle_test_result(result, data[result.worst()], result.worst(),
      type_name, "non central beta CDF", test);
#endif
#if !(defined(ERROR_REPORTING_MODE) && !defined(NC_BETA_CCDF_FUNCTION_TO_TEST))
   fp1 = nc_beta_ccdf;
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(fp1, 0, 1, 2, 3),
      extract_result<Real>(5));
   handle_test_result(result, data[result.worst()], result.worst(),
      type_name, "non central beta CDF complement", test);
#endif
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
      //
      // Test case 493 fails at float precision: not enough bits to get
      // us back where we started:
      //
      if((i == 493) && boost::is_same<float, value_type>::value)
         continue;

      if(data[i][4] == 0)
      {
         BOOST_CHECK(0 == quantile(boost::math::non_central_beta_distribution<value_type>(data[i][0], data[i][1], data[i][2]), data[i][4]));
      }
      else if(data[i][4] < 0.9999f)
      {
         value_type p = quantile(boost::math::non_central_beta_distribution<value_type>(data[i][0], data[i][1], data[i][2]), data[i][4]);
         value_type pt = data[i][3];
         BOOST_CHECK_CLOSE_EX(pt, p, precision, i);
      }
      if(data[i][5] == 0)
      {
         BOOST_CHECK(1 == quantile(complement(boost::math::non_central_beta_distribution<value_type>(data[i][0], data[i][1], data[i][2]), data[i][5])));
      }
      else if(data[i][5] < 0.9999f)
      {
         value_type p = quantile(complement(boost::math::non_central_beta_distribution<value_type>(data[i][0], data[i][1], data[i][2]), data[i][5]));
         value_type pt = data[i][3];
         BOOST_CHECK_CLOSE_EX(pt, p, precision, i);
      }
      if(boost::math::tools::digits<value_type>() > 50)
      {
         //
         // Sanity check mode, accuracy of
         // the mode is at *best* the square root of the accuracy of the PDF:
         //
         value_type m = mode(boost::math::non_central_beta_distribution<value_type>(data[i][0], data[i][1], data[i][2]));
         if((m == 1) || (m == 0))
            break;
         value_type p = pdf(boost::math::non_central_beta_distribution<value_type>(data[i][0], data[i][1], data[i][2]), m);
         if(m * (1 + sqrt(precision) * 10) < 1)
         {
            BOOST_CHECK_EX(pdf(boost::math::non_central_beta_distribution<value_type>(data[i][0], data[i][1], data[i][2]), m * (1 + sqrt(precision) * 10)) <= p, i);
         }
         if(m * (1 - sqrt(precision)) * 10 > boost::math::tools::min_value<value_type>())
         {
            BOOST_CHECK_EX(pdf(boost::math::non_central_beta_distribution<value_type>(data[i][0], data[i][1], data[i][2]), m * (1 - sqrt(precision)) * 10) <= p, i);
         }
      }
   }
#endif
}

template <typename T>
void test_accuracy(T, const char* type_name)
{
#if !defined(TEST_DATA) || (TEST_DATA == 1)
#include "ncbeta.ipp"
   do_test_nc_chi_squared<T>(ncbeta, type_name, "Non Central Beta, medium parameters");
   quantile_sanity_check<T>(ncbeta, type_name, "Non Central Beta, medium parameters");
#endif
#if !defined(TEST_DATA) || (TEST_DATA == 2)
#include "ncbeta_big.ipp"
   do_test_nc_chi_squared<T>(ncbeta_big, type_name, "Non Central Beta, large parameters");
   // Takes too long to run:
   // quantile_sanity_check(ncbeta_big, type_name, "Non Central Beta, large parameters");
#endif
}
