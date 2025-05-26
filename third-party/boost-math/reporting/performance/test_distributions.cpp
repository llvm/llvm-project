//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#define DISTRIBUTIONS_TEST

#include <boost/math/distributions.hpp>
#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>
#include "../../test/table_type.hpp"
#include "table_helper.hpp"
#include "performance.hpp"
#include <iostream>

#ifdef TEST_GSL
#include <gsl/gsl_cdf.h>
#endif

class distribution_tester
{
   std::string distro_name;
   static const double quantiles[19];
   double sum;

   struct param_info
   {
      std::vector<double> params;
      std::vector<double> x_values;
   };
   std::vector<param_info> tests;
   double sanitize_x(double x)
   {
      if(x > boost::math::tools::max_value<float>() / 2)
         return boost::math::tools::max_value<float>() / 2;
      if(x < -boost::math::tools::max_value<float>() / 2)
         return -boost::math::tools::max_value<float>() / 2;
      return x;
   }
public:
   distribution_tester(const char* name) : distro_name(name), sum(0) {}

   template <class F>
   void add_test_case(F f)
   {
      tests.push_back(param_info());
      for(unsigned i = 0; i < sizeof(quantiles) / sizeof(quantiles[0]); ++i)
      {
         tests.back().x_values.push_back(sanitize_x(f(quantiles[i])));
      }
   }
   template <class F>
   void add_test_case(double p1, F f)
   {
      tests.push_back(param_info());
      tests.back().params.push_back(p1);
      for(unsigned i = 0; i < sizeof(quantiles) / sizeof(quantiles[0]); ++i)
      {
         tests.back().x_values.push_back(sanitize_x(f(p1, quantiles[i])));
      }
   }
   template <class F>
   void add_test_case(double p1, double p2, F f)
   {
      tests.push_back(param_info());
      tests.back().params.push_back(p1);
      tests.back().params.push_back(p2);
      for(unsigned i = 0; i < sizeof(quantiles) / sizeof(quantiles[0]); ++i)
      {
         tests.back().x_values.push_back(sanitize_x(f(p1, p2, quantiles[i])));
      }
   }
   template <class F>
   void add_test_case(double p1, double p2, double p3, F f)
   {
      tests.push_back(param_info());
      tests.back().params.push_back(p1);
      tests.back().params.push_back(p2);
      tests.back().params.push_back(p3);
      for(unsigned i = 0; i < sizeof(quantiles) / sizeof(quantiles[0]); ++i)
      {
         tests.back().x_values.push_back(sanitize_x(f(p1, p2, p3, quantiles[i])));
      }
   }

   enum
   {
      main_table = 1,
      boost_only_table = 2,
      both_tables = 3
   };

   template <class F>
   void run_timed_tests(F f, std::string sub_name, std::string column, bool p_value = false, int where = main_table)
   {
      std::cout << "Testing " << distro_name + " (" + std::string(sub_name) + ")" << " with library " << column << std::endl;
      try{
         double t = 0;
         unsigned repeats = 1;
         unsigned data_size;
         do{
            data_size = 0;
            stopwatch<boost::chrono::high_resolution_clock> w;

            for(unsigned count = 0; count < repeats; ++count)
            {
               for(unsigned i = 0; i < tests.size(); ++i)
               {
                  for(unsigned j = 0; j < tests[i].x_values.size(); ++j)
                  {
                     if((boost::math::isfinite)(tests[i].x_values[j]))
                        sum += f(tests[i].params, p_value ? quantiles[j] : tests[i].x_values[j]);
                     ++data_size;
                  }
               }
            }

            t = boost::chrono::duration_cast<boost::chrono::duration<double>>(w.elapsed()).count();
            if(t < 0.5)
               repeats *= 2;
         } while(t < 0.5);

         static const std::string main_table_name = std::string("Distribution performance comparison with ") + compiler_name() + std::string(" on ") + platform_name();
         static const std::string boost_table_name = std::string("Distribution performance comparison for different performance options with ") + compiler_name() + std::string(" on ") + platform_name();

         if (where & 1)
         {
            report_execution_time(
               t / data_size,
               main_table_name,
               distro_name + " (" + std::string(sub_name) + ")",
               column);
         }
         if (where & 2)
         {
            report_execution_time(
               t / data_size,
               boost_table_name,
               distro_name + " (" + std::string(sub_name) + ")",
               column);
         }
      }
      catch(const std::exception& e)
      {
         std::cerr << "Aborting due to exception: " << e.what() << std::endl;
         std::cerr << "In " << distro_name + " (" + std::string(sub_name) + ")" << std::endl;
         report_execution_time(
            (std::numeric_limits<std::uintmax_t>::max)(),
            std::string("Distribution performance comparison with ") + compiler_name() + std::string(" on ") + platform_name(),
            distro_name + " (" + std::string(sub_name) + ")",
            column);
      }
   }
};

const double distribution_tester::quantiles[19] = 
{
   0.000001,
   0.00001,
   0.0001,
   0.001,
   0.01,
   0.1,
   0.2,
   0.3,
   0.4,
   0.5,
   0.6,
   0.7,
   0.8,
   0.9,
   0.99,
   0.999,
   0.9999,
   0.99999,
   0.999999
};

template <class D>
struct three_param_quantile
{
   template <class T, class U, class V, class X>
   double operator()(T x, U y, V z, X q)const
   {
      return quantile(D(x, y, z), q);
   }
};

template <class D>
struct two_param_quantile
{
   template <class T, class U, class V>
   double operator()(T x, U y, V q)const
   {
      return quantile(D(x, y), q);
   }
};

template <class D>
struct one_param_quantile
{
   template <class T, class V>
   double operator()(T x, V q)const
   {
      return quantile(D(x), q);
   }
};

template <template <class T, class U> class D>
void test_boost_1_param(distribution_tester& tester)
{
   //
   // Define some custom policies to test:
   //
   typedef boost::math::policies::policy<> default_policy;
   typedef boost::math::policies::policy<boost::math::policies::promote_double<false> > no_promote_double_policy;
   typedef boost::math::policies::policy<boost::math::policies::promote_double<false>, boost::math::policies::digits10<10> > no_promote_double_10_digits_policy;
   typedef boost::math::policies::policy<boost::math::policies::promote_float<false> > no_promote_float_policy;

   tester.run_timed_tests([](const std::vector<double>& v, double x){  return pdf(D<double, default_policy>(v[0]), x); }, "PDF", boost_name(), false, distribution_tester::both_tables);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return cdf(D<double, default_policy>(v[0]), x); }, "CDF", boost_name(), false, distribution_tester::both_tables);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return quantile(D<double, default_policy>(v[0]), x); }, "quantile", boost_name(), true, distribution_tester::both_tables);
   if(sizeof(double) != sizeof(long double))
   {
      tester.run_timed_tests([](const std::vector<double>& v, double x){  return pdf(D<double, no_promote_double_policy>(v[0]), x); }, "PDF", "Boost[br]promote_double<false>", false, distribution_tester::both_tables);
      tester.run_timed_tests([](const std::vector<double>& v, double x){  return cdf(D<double, no_promote_double_policy>(v[0]), x); }, "CDF", "Boost[br]promote_double<false>", false, distribution_tester::both_tables);
      tester.run_timed_tests([](const std::vector<double>& v, double x){  return quantile(D<double, no_promote_double_policy>(v[0]), x); }, "quantile", "Boost[br]promote_double<false>", true, distribution_tester::both_tables);
   }
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return pdf(D<double, no_promote_double_10_digits_policy>(v[0]), x); }, "PDF", "Boost[br]promote_double<false>[br]digits10<10>", false, distribution_tester::boost_only_table);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return cdf(D<double, no_promote_double_10_digits_policy>(v[0]), x); }, "CDF", "Boost[br]promote_double<false>[br]digits10<10>", false, distribution_tester::boost_only_table);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return quantile(D<double, no_promote_double_10_digits_policy>(v[0]), x); }, "quantile", "Boost[br]promote_double<false>[br]digits10<10>", true, distribution_tester::boost_only_table);

   tester.run_timed_tests([](const std::vector<double>& v, double x){  return pdf(D<float, no_promote_float_policy>(static_cast<float>(v[0])), static_cast<float>(x)); }, "PDF", "Boost[br]float[br]promote_float<false>", false, distribution_tester::boost_only_table);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return cdf(D<float, no_promote_float_policy>(static_cast<float>(v[0])), static_cast<float>(x)); }, "CDF", "Boost[br]float[br]promote_float<false>", false, distribution_tester::boost_only_table);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return quantile(D<float, no_promote_float_policy>(static_cast<float>(v[0])), static_cast<float>(x)); }, "quantile", "Boost[br]float[br]promote_float<false>", true, distribution_tester::boost_only_table);
}

template <template <class T, class U> class D>
void test_boost_2_param(distribution_tester& tester)
{
   //
   // Define some custom policies to test:
   //
   typedef boost::math::policies::policy<> default_policy;
   typedef boost::math::policies::policy<boost::math::policies::promote_double<false> > no_promote_double_policy;
   typedef boost::math::policies::policy<boost::math::policies::promote_double<false>, boost::math::policies::digits10<10> > no_promote_double_10_digits_policy;
   typedef boost::math::policies::policy<boost::math::policies::promote_float<false> > no_promote_float_policy;

   tester.run_timed_tests([](const std::vector<double>& v, double x){  return pdf(D<double, default_policy>(v[0], v[1]), x); }, "PDF", boost_name(), false, distribution_tester::both_tables);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return cdf(D<double, default_policy>(v[0], v[1]), x); }, "CDF", boost_name(), false, distribution_tester::both_tables);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return quantile(D<double, default_policy>(v[0], v[1]), x); }, "quantile", boost_name(), true, distribution_tester::both_tables);
   if(sizeof(double) != sizeof(long double))
   {
      tester.run_timed_tests([](const std::vector<double>& v, double x){  return pdf(D<double, no_promote_double_policy>(v[0], v[1]), x); }, "PDF", "Boost[br]promote_double<false>", false, distribution_tester::both_tables);
      tester.run_timed_tests([](const std::vector<double>& v, double x){  return cdf(D<double, no_promote_double_policy>(v[0], v[1]), x); }, "CDF", "Boost[br]promote_double<false>", false, distribution_tester::both_tables);
      tester.run_timed_tests([](const std::vector<double>& v, double x){  return quantile(D<double, no_promote_double_policy>(v[0], v[1]), x); }, "quantile", "Boost[br]promote_double<false>", true, distribution_tester::both_tables);
   }
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return pdf(D<double, no_promote_double_10_digits_policy>(v[0], v[1]), x); }, "PDF", "Boost[br]promote_double<false>[br]digits10<10>", false, distribution_tester::boost_only_table);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return cdf(D<double, no_promote_double_10_digits_policy>(v[0], v[1]), x); }, "CDF", "Boost[br]promote_double<false>[br]digits10<10>", false, distribution_tester::boost_only_table);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return quantile(D<double, no_promote_double_10_digits_policy>(v[0], v[1]), x); }, "quantile", "Boost[br]promote_double<false>[br]digits10<10>", true, distribution_tester::boost_only_table);

   tester.run_timed_tests([](const std::vector<double>& v, double x){  return pdf(D<float, no_promote_float_policy>(static_cast<float>(v[0]), static_cast<float>(v[1])), static_cast<float>(x)); }, "PDF", "Boost[br]float[br]promote_float<false>", false, distribution_tester::boost_only_table);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return cdf(D<float, no_promote_float_policy>(static_cast<float>(v[0]), static_cast<float>(v[1])), static_cast<float>(x)); }, "CDF", "Boost[br]float[br]promote_float<false>", false, distribution_tester::boost_only_table);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return quantile(D<float, no_promote_float_policy>(static_cast<float>(v[0]), static_cast<float>(v[1])), static_cast<float>(x)); }, "quantile", "Boost[br]float[br]promote_float<false>", true, distribution_tester::boost_only_table);
}

template <template <class T, class U> class D>
void test_boost_3_param(distribution_tester& tester)
{
   //
   // Define some custom policies to test:
   //
   typedef boost::math::policies::policy<> default_policy;
   typedef boost::math::policies::policy<boost::math::policies::promote_double<false> > no_promote_double_policy;
   typedef boost::math::policies::policy<boost::math::policies::promote_double<false>, boost::math::policies::digits10<10> > no_promote_double_10_digits_policy;
   typedef boost::math::policies::policy<boost::math::policies::promote_float<false> > no_promote_float_policy;

   tester.run_timed_tests([](const std::vector<double>& v, double x){  return pdf(D<double, default_policy>(v[0], v[1], v[2]), x); }, "PDF", boost_name(), false, distribution_tester::both_tables);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return cdf(D<double, default_policy>(v[0], v[1], v[2]), x); }, "CDF", boost_name(), false, distribution_tester::both_tables);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return quantile(D<double, default_policy>(v[0], v[1], v[2]), x); }, "quantile", boost_name(), true, distribution_tester::both_tables);
   if(sizeof(double) != sizeof(long double))
   {
      tester.run_timed_tests([](const std::vector<double>& v, double x){  return pdf(D<double, no_promote_double_policy>(v[0], v[1], v[2]), x); }, "PDF", "Boost[br]promote_double<false>", false, distribution_tester::both_tables);
      tester.run_timed_tests([](const std::vector<double>& v, double x){  return cdf(D<double, no_promote_double_policy>(v[0], v[1], v[2]), x); }, "CDF", "Boost[br]promote_double<false>", false, distribution_tester::both_tables);
      tester.run_timed_tests([](const std::vector<double>& v, double x){  return quantile(D<double, no_promote_double_policy>(v[0], v[1], v[2]), x); }, "quantile", "Boost[br]promote_double<false>", true, distribution_tester::both_tables);
   }
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return pdf(D<double, no_promote_double_10_digits_policy>(v[0], v[1], v[2]), x); }, "PDF", "Boost[br]promote_double<false>[br]digits10<10>", false, distribution_tester::boost_only_table);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return cdf(D<double, no_promote_double_10_digits_policy>(v[0], v[1], v[2]), x); }, "CDF", "Boost[br]promote_double<false>[br]digits10<10>", false, distribution_tester::boost_only_table);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return quantile(D<double, no_promote_double_10_digits_policy>(v[0], v[1], v[2]), x); }, "quantile", "Boost[br]promote_double<false>[br]digits10<10>", true, distribution_tester::boost_only_table);

   tester.run_timed_tests([](const std::vector<double>& v, double x){  return pdf(D<float, no_promote_float_policy>(static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])), static_cast<float>(x)); }, "PDF", "Boost[br]float[br]promote_float<false>", false, distribution_tester::boost_only_table);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return cdf(D<float, no_promote_float_policy>(static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])), static_cast<float>(x)); }, "CDF", "Boost[br]float[br]promote_float<false>", false, distribution_tester::boost_only_table);
   tester.run_timed_tests([](const std::vector<double>& v, double x){  return quantile(D<float, no_promote_float_policy>(static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])), static_cast<float>(x)); }, "quantile", "Boost[br]float[br]promote_float<false>", true, distribution_tester::boost_only_table);
}

int main()
{
   try {
      //
      // Normal:
      //
      distribution_tester n("Normal");
      n.add_test_case(0, 1, two_param_quantile<boost::math::normal_distribution<> >());
      n.add_test_case(20, 20, two_param_quantile<boost::math::normal_distribution<> >());
      n.add_test_case(-20, 0.0125, two_param_quantile<boost::math::normal_distribution<> >());

      test_boost_2_param<boost::math::normal_distribution>(n);

      distribution_tester arcsine("ArcSine");
      arcsine.add_test_case(0, 1, two_param_quantile<boost::math::arcsine_distribution<> >());
      arcsine.add_test_case(20, 500, two_param_quantile<boost::math::arcsine_distribution<> >());
      arcsine.add_test_case(-20, 100000, two_param_quantile<boost::math::arcsine_distribution<> >());

      test_boost_2_param<boost::math::arcsine_distribution>(arcsine);

      distribution_tester beta("Beta");
      beta.add_test_case(1, 4, two_param_quantile<boost::math::beta_distribution<> >());
      beta.add_test_case(20, 500, two_param_quantile<boost::math::beta_distribution<> >());
      beta.add_test_case(0.1, 0.01, two_param_quantile<boost::math::beta_distribution<> >());

      test_boost_2_param<boost::math::beta_distribution>(beta);

      distribution_tester binomial("Binomial");
      binomial.add_test_case(5, 0.125, two_param_quantile<boost::math::binomial_distribution<> >());
      binomial.add_test_case(200, 0.75, two_param_quantile<boost::math::binomial_distribution<> >());
      binomial.add_test_case(2000, 0.5, two_param_quantile<boost::math::binomial_distribution<> >());
      binomial.add_test_case(20000, 0.001, two_param_quantile<boost::math::binomial_distribution<> >());
      binomial.add_test_case(200000, 0.99, two_param_quantile<boost::math::binomial_distribution<> >());

      test_boost_2_param<boost::math::binomial_distribution>(binomial);

      distribution_tester cauchy("Cauchy");
      cauchy.add_test_case(0, 1, two_param_quantile<boost::math::cauchy_distribution<> >());
      cauchy.add_test_case(20, 20, two_param_quantile<boost::math::cauchy_distribution<> >());
      cauchy.add_test_case(-20, 0.0125, two_param_quantile<boost::math::cauchy_distribution<> >());

      test_boost_2_param<boost::math::cauchy_distribution>(cauchy);

      distribution_tester chi_squared("ChiSquared");
      chi_squared.add_test_case(3, one_param_quantile<boost::math::chi_squared_distribution<> >());
      chi_squared.add_test_case(20, one_param_quantile<boost::math::chi_squared_distribution<> >());
      chi_squared.add_test_case(200, one_param_quantile<boost::math::chi_squared_distribution<> >());
      chi_squared.add_test_case(2000, one_param_quantile<boost::math::chi_squared_distribution<> >());
      chi_squared.add_test_case(20000, one_param_quantile<boost::math::chi_squared_distribution<> >());
      chi_squared.add_test_case(200000, one_param_quantile<boost::math::chi_squared_distribution<> >());

      test_boost_1_param<boost::math::chi_squared_distribution>(chi_squared);

      distribution_tester exponential("Exponential");
      exponential.add_test_case(0.001, one_param_quantile<boost::math::exponential_distribution<> >());
      exponential.add_test_case(0.01, one_param_quantile<boost::math::exponential_distribution<> >());
      exponential.add_test_case(0.1, one_param_quantile<boost::math::exponential_distribution<> >());
      exponential.add_test_case(1, one_param_quantile<boost::math::exponential_distribution<> >());
      exponential.add_test_case(10, one_param_quantile<boost::math::exponential_distribution<> >());
      exponential.add_test_case(100, one_param_quantile<boost::math::exponential_distribution<> >());
      exponential.add_test_case(1000, one_param_quantile<boost::math::exponential_distribution<> >());

      test_boost_1_param<boost::math::exponential_distribution>(exponential);

      distribution_tester extreme_value("ExtremeValue");
      extreme_value.add_test_case(0, 1, two_param_quantile<boost::math::extreme_value_distribution<> >());
      extreme_value.add_test_case(20, 20, two_param_quantile<boost::math::extreme_value_distribution<> >());
      extreme_value.add_test_case(-20, 0.0125, two_param_quantile<boost::math::extreme_value_distribution<> >());

      test_boost_2_param<boost::math::extreme_value_distribution>(extreme_value);

      distribution_tester fisher("F");
      for (unsigned i = 2; i <= 200000; i *= 10)
      {
         for (unsigned j = 2; j <= 200000; j *= 10)
         {
            fisher.add_test_case(i, j, two_param_quantile<boost::math::fisher_f_distribution<> >());
         }
      }
      test_boost_2_param<boost::math::fisher_f_distribution>(fisher);

      distribution_tester gamma("Gamma");
      gamma.add_test_case(0.1, 1, two_param_quantile<boost::math::gamma_distribution<> >());
      gamma.add_test_case(20, 20, two_param_quantile<boost::math::gamma_distribution<> >());
      gamma.add_test_case(200, 0.0125, two_param_quantile<boost::math::gamma_distribution<> >());
      gamma.add_test_case(2000, 500, two_param_quantile<boost::math::gamma_distribution<> >());

      test_boost_2_param<boost::math::gamma_distribution>(gamma);

      distribution_tester geometric("Geometric");
      geometric.add_test_case(0.001, one_param_quantile<boost::math::geometric_distribution<> >());
      geometric.add_test_case(0.01, one_param_quantile<boost::math::geometric_distribution<> >());
      geometric.add_test_case(0.1, one_param_quantile<boost::math::geometric_distribution<> >());
      geometric.add_test_case(0.5, one_param_quantile<boost::math::geometric_distribution<> >());
      geometric.add_test_case(0.9, one_param_quantile<boost::math::geometric_distribution<> >());
      geometric.add_test_case(0.99, one_param_quantile<boost::math::geometric_distribution<> >());
      geometric.add_test_case(0.999, one_param_quantile<boost::math::geometric_distribution<> >());

      test_boost_1_param<boost::math::geometric_distribution>(geometric);

      distribution_tester hypergeometric("Hypergeometric");
      hypergeometric.add_test_case(10, 5, 100, three_param_quantile<boost::math::hypergeometric_distribution<> >());
      hypergeometric.add_test_case(50, 75, 100, three_param_quantile<boost::math::hypergeometric_distribution<> >());
      hypergeometric.add_test_case(30, 20, 100, three_param_quantile<boost::math::hypergeometric_distribution<> >());
      hypergeometric.add_test_case(100, 50, 1000000, three_param_quantile<boost::math::hypergeometric_distribution<> >());
      hypergeometric.add_test_case(500000, 3000, 1000000, three_param_quantile<boost::math::hypergeometric_distribution<> >());
      hypergeometric.add_test_case(20000, 800000, 1000000, three_param_quantile<boost::math::hypergeometric_distribution<> >());
      hypergeometric.add_test_case(100, 5, 1000, three_param_quantile<boost::math::hypergeometric_distribution<> >());
      hypergeometric.add_test_case(500, 50, 1000, three_param_quantile<boost::math::hypergeometric_distribution<> >());
      hypergeometric.add_test_case(2, 25, 1000, three_param_quantile<boost::math::hypergeometric_distribution<> >());
      hypergeometric.add_test_case(1, 5, 1000, three_param_quantile<boost::math::hypergeometric_distribution<> >());
      hypergeometric.add_test_case(100, 500, 1000, three_param_quantile<boost::math::hypergeometric_distribution<> >());

      test_boost_3_param<boost::math::hypergeometric_distribution>(hypergeometric);

      distribution_tester inverse_chi_squared("InverseChiSquared");
      inverse_chi_squared.add_test_case(5, 0.125, two_param_quantile<boost::math::inverse_chi_squared_distribution<> >());
      inverse_chi_squared.add_test_case(200, 0.75, two_param_quantile<boost::math::inverse_chi_squared_distribution<> >());
      inverse_chi_squared.add_test_case(2000, 1, two_param_quantile<boost::math::inverse_chi_squared_distribution<> >());
      inverse_chi_squared.add_test_case(20000, 10, two_param_quantile<boost::math::inverse_chi_squared_distribution<> >());
      inverse_chi_squared.add_test_case(200000, 100, two_param_quantile<boost::math::inverse_chi_squared_distribution<> >());

      test_boost_2_param<boost::math::inverse_chi_squared_distribution>(inverse_chi_squared);

      distribution_tester inverse_gamma("InverseGamma");
      inverse_gamma.add_test_case(0.1, 1, two_param_quantile<boost::math::inverse_gamma_distribution<> >());
      inverse_gamma.add_test_case(20, 20, two_param_quantile<boost::math::inverse_gamma_distribution<> >());
      inverse_gamma.add_test_case(200, 0.0125, two_param_quantile<boost::math::inverse_gamma_distribution<> >());
      inverse_gamma.add_test_case(2000, 500, two_param_quantile<boost::math::inverse_gamma_distribution<> >());

      test_boost_2_param<boost::math::inverse_gamma_distribution>(inverse_gamma);

      distribution_tester inverse_gaussian("InverseGaussian");
      inverse_gaussian.add_test_case(0.001, 1, two_param_quantile<boost::math::inverse_gaussian_distribution<> >());
      inverse_gaussian.add_test_case(20, 20, two_param_quantile<boost::math::inverse_gaussian_distribution<> >());

      test_boost_2_param<boost::math::inverse_gaussian_distribution>(inverse_gaussian);

      distribution_tester kolmogorov("KolmogorovSmirnov");
      kolmogorov.add_test_case(3, one_param_quantile<boost::math::kolmogorov_smirnov_distribution<> >());
      kolmogorov.add_test_case(20, one_param_quantile<boost::math::kolmogorov_smirnov_distribution<> >());
      kolmogorov.add_test_case(200, one_param_quantile<boost::math::kolmogorov_smirnov_distribution<> >());
      kolmogorov.add_test_case(2000, one_param_quantile<boost::math::kolmogorov_smirnov_distribution<> >());
      kolmogorov.add_test_case(20000, one_param_quantile<boost::math::kolmogorov_smirnov_distribution<> >());
      kolmogorov.add_test_case(200000, one_param_quantile<boost::math::kolmogorov_smirnov_distribution<> >());

      test_boost_1_param<boost::math::kolmogorov_smirnov_distribution>(kolmogorov);

      distribution_tester laplace("Laplace");
      laplace.add_test_case(0, 1, two_param_quantile<boost::math::laplace_distribution<> >());
      laplace.add_test_case(20, 20, two_param_quantile<boost::math::laplace_distribution<> >());
      laplace.add_test_case(-20, 0.0125, two_param_quantile<boost::math::laplace_distribution<> >());

      test_boost_2_param<boost::math::laplace_distribution>(laplace);

      distribution_tester logistic("Logistic");
      logistic.add_test_case(0, 1, two_param_quantile<boost::math::logistic_distribution<> >());
      logistic.add_test_case(20, 20, two_param_quantile<boost::math::logistic_distribution<> >());
      logistic.add_test_case(-20, 0.0125, two_param_quantile<boost::math::logistic_distribution<> >());

      test_boost_2_param<boost::math::logistic_distribution>(logistic);

      distribution_tester lognormal("LogNormal");
      lognormal.add_test_case(0, 1, two_param_quantile<boost::math::lognormal_distribution<> >());
      lognormal.add_test_case(20, 20, two_param_quantile<boost::math::lognormal_distribution<> >());
      lognormal.add_test_case(-20, 0.0125, two_param_quantile<boost::math::lognormal_distribution<> >());

      test_boost_2_param<boost::math::lognormal_distribution>(lognormal);

      distribution_tester negative_binomial("NegativeBinomial");
      negative_binomial.add_test_case(5, 0.125, two_param_quantile<boost::math::negative_binomial_distribution<> >());
      negative_binomial.add_test_case(200, 0.75, two_param_quantile<boost::math::negative_binomial_distribution<> >());
      negative_binomial.add_test_case(2000, 0.001, two_param_quantile<boost::math::negative_binomial_distribution<> >());
      negative_binomial.add_test_case(20000, 0.5, two_param_quantile<boost::math::negative_binomial_distribution<> >());
      negative_binomial.add_test_case(200000, 0.99, two_param_quantile<boost::math::negative_binomial_distribution<> >());

      test_boost_2_param<boost::math::negative_binomial_distribution>(negative_binomial);

      distribution_tester non_central_beta("NonCentralBeta");
      non_central_beta.add_test_case(2, 5, 2.1, three_param_quantile<boost::math::non_central_beta_distribution<> >());
      non_central_beta.add_test_case(0.25, 0.01, 20, three_param_quantile<boost::math::non_central_beta_distribution<> >());
      non_central_beta.add_test_case(20, 3, 30, three_param_quantile<boost::math::non_central_beta_distribution<> >());
      non_central_beta.add_test_case(100, 200, 400, three_param_quantile<boost::math::non_central_beta_distribution<> >());
      non_central_beta.add_test_case(100, 0.25, 20, three_param_quantile<boost::math::non_central_beta_distribution<> >());

      test_boost_3_param<boost::math::non_central_beta_distribution>(non_central_beta);

      distribution_tester non_central_chi_squared("NonCentralChiSquared");
      non_central_chi_squared.add_test_case(5, 0.5, two_param_quantile<boost::math::non_central_chi_squared_distribution<> >());
      non_central_chi_squared.add_test_case(200, 2, two_param_quantile<boost::math::non_central_chi_squared_distribution<> >());
      non_central_chi_squared.add_test_case(2000, 20, two_param_quantile<boost::math::non_central_chi_squared_distribution<> >());
      non_central_chi_squared.add_test_case(20000, 10, two_param_quantile<boost::math::non_central_chi_squared_distribution<> >());
      non_central_chi_squared.add_test_case(200000, 50, two_param_quantile<boost::math::non_central_chi_squared_distribution<> >());

      test_boost_2_param<boost::math::non_central_chi_squared_distribution>(non_central_chi_squared);

      distribution_tester non_central_f("NonCentralF");
      non_central_f.add_test_case(20, 20, 3, three_param_quantile<boost::math::non_central_f_distribution<> >());
      non_central_f.add_test_case(20, 50, 20, three_param_quantile<boost::math::non_central_f_distribution<> >());
      non_central_f.add_test_case(100, 20, 30, three_param_quantile<boost::math::non_central_f_distribution<> >());
      non_central_f.add_test_case(100, 200, 100, three_param_quantile<boost::math::non_central_f_distribution<> >());
      non_central_f.add_test_case(1000, 100000, 20, three_param_quantile<boost::math::non_central_f_distribution<> >());

      test_boost_3_param<boost::math::non_central_f_distribution>(non_central_f);

      distribution_tester non_central_t("NonCentralT");
      non_central_t.add_test_case(5, 0.5, two_param_quantile<boost::math::non_central_t_distribution<> >());
      non_central_t.add_test_case(200, 2, two_param_quantile<boost::math::non_central_t_distribution<> >());
      non_central_t.add_test_case(2000, 20, two_param_quantile<boost::math::non_central_t_distribution<> >());
      non_central_t.add_test_case(20000, 10, two_param_quantile<boost::math::non_central_t_distribution<> >());
      non_central_t.add_test_case(200000, 50, two_param_quantile<boost::math::non_central_t_distribution<> >());

      test_boost_2_param<boost::math::non_central_t_distribution>(non_central_t);

      distribution_tester pareto("Pareto");
      pareto.add_test_case(0.1, 1, two_param_quantile<boost::math::pareto_distribution<> >());
      pareto.add_test_case(20, 20, two_param_quantile<boost::math::pareto_distribution<> >());
      pareto.add_test_case(200, 0.0125, two_param_quantile<boost::math::pareto_distribution<> >());
      pareto.add_test_case(2000, 500, two_param_quantile<boost::math::pareto_distribution<> >());

      test_boost_2_param<boost::math::pareto_distribution>(pareto);

      distribution_tester poisson("Poisson");
      poisson.add_test_case(0.001, one_param_quantile<boost::math::poisson_distribution<> >());
      poisson.add_test_case(0.01, one_param_quantile<boost::math::poisson_distribution<> >());
      poisson.add_test_case(0.1, one_param_quantile<boost::math::poisson_distribution<> >());
      poisson.add_test_case(1, one_param_quantile<boost::math::poisson_distribution<> >());
      poisson.add_test_case(10, one_param_quantile<boost::math::poisson_distribution<> >());
      poisson.add_test_case(100, one_param_quantile<boost::math::poisson_distribution<> >());
      poisson.add_test_case(1000, one_param_quantile<boost::math::poisson_distribution<> >());

      test_boost_1_param<boost::math::poisson_distribution>(poisson);

      distribution_tester rayleigh("Rayleigh");
      rayleigh.add_test_case(0.001, one_param_quantile<boost::math::rayleigh_distribution<> >());
      rayleigh.add_test_case(0.01, one_param_quantile<boost::math::rayleigh_distribution<> >());
      rayleigh.add_test_case(0.1, one_param_quantile<boost::math::rayleigh_distribution<> >());
      rayleigh.add_test_case(1, one_param_quantile<boost::math::rayleigh_distribution<> >());
      rayleigh.add_test_case(10, one_param_quantile<boost::math::rayleigh_distribution<> >());
      rayleigh.add_test_case(100, one_param_quantile<boost::math::rayleigh_distribution<> >());
      rayleigh.add_test_case(1000, one_param_quantile<boost::math::rayleigh_distribution<> >());

      test_boost_1_param<boost::math::rayleigh_distribution>(rayleigh);

      distribution_tester skew_norm("SkewNormal");
      skew_norm.add_test_case(0, 1, 0.1, three_param_quantile<boost::math::skew_normal_distribution<> >());
      skew_norm.add_test_case(20, 20, 30, three_param_quantile<boost::math::skew_normal_distribution<> >());
      skew_norm.add_test_case(-20, 0.0125, 10, three_param_quantile<boost::math::skew_normal_distribution<> >());

      test_boost_3_param<boost::math::skew_normal_distribution>(skew_norm);

      distribution_tester students_t("StudentsT");
      students_t.add_test_case(3, one_param_quantile<boost::math::students_t_distribution<> >());
      students_t.add_test_case(20, one_param_quantile<boost::math::students_t_distribution<> >());
      students_t.add_test_case(200, one_param_quantile<boost::math::students_t_distribution<> >());
      students_t.add_test_case(2000, one_param_quantile<boost::math::students_t_distribution<> >());
      students_t.add_test_case(20000, one_param_quantile<boost::math::students_t_distribution<> >());
      students_t.add_test_case(200000, one_param_quantile<boost::math::students_t_distribution<> >());

      test_boost_1_param<boost::math::students_t_distribution>(students_t);

      distribution_tester weibull("Weibull");
      weibull.add_test_case(0.1, 1, two_param_quantile<boost::math::weibull_distribution<> >());
      weibull.add_test_case(20, 20, two_param_quantile<boost::math::weibull_distribution<> >());
      weibull.add_test_case(200, 0.0125, two_param_quantile<boost::math::weibull_distribution<> >());
      weibull.add_test_case(2000, 500, two_param_quantile<boost::math::weibull_distribution<> >());

      test_boost_2_param<boost::math::weibull_distribution>(weibull);

#ifdef TEST_GSL
      // normal, note no location param
      n.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_gaussian_P(x, v[1]); }, "CDF", "GSL");
      n.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_gaussian_Pinv(x, v[1]); }, "quantile", "GSL", true);
      // exponential:
      exponential.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_exponential_P(x, 1 / v[0]); }, "CDF", "GSL");
      exponential.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_exponential_Pinv(x, 1 / v[0]); }, "quantile", "GSL", true);
      // laplace, note no location param:
      laplace.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_laplace_P(x, v[1]); }, "CDF", "GSL");
      laplace.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_laplace_Pinv(x, v[1]); }, "quantile", "GSL", true);
      // cauchy, note no location param:
      cauchy.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_cauchy_P(x, v[1]); }, "CDF", "GSL");
      cauchy.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_cauchy_Pinv(x, v[1]); }, "quantile", "GSL", true);
      // rayleigh:
      rayleigh.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_rayleigh_P(x, v[0]); }, "CDF", "GSL");
      rayleigh.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_rayleigh_Pinv(x, v[0]); }, "quantile", "GSL", true);
      // gamma:
      gamma.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_gamma_P(x, v[0], v[1]); }, "CDF", "GSL");
      gamma.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_gamma_Pinv(x, v[0], v[1]); }, "quantile", "GSL", true);
      // lognormal:
      lognormal.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_lognormal_P(x, v[0], v[1]); }, "CDF", "GSL");
      lognormal.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_lognormal_Pinv(x, v[0], v[1]); }, "quantile", "GSL", true);
      // chi squared:
      chi_squared.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_chisq_P(x, v[0]); }, "CDF", "GSL");
      chi_squared.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_chisq_Pinv(x, v[0]); }, "quantile", "GSL", true);
      // F:
      fisher.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_fdist_P(x, v[0], v[1]); }, "CDF", "GSL");
      fisher.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_fdist_Pinv(x, v[0], v[1]); }, "quantile", "GSL", true);
      // T:
      students_t.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_tdist_P(x, v[0]); }, "CDF", "GSL");
      students_t.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_tdist_Pinv(x, v[0]); }, "quantile", "GSL", true);
      // beta:
      beta.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_beta_P(x, v[0], v[1]); }, "CDF", "GSL");
      beta.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_beta_Pinv(x, v[0], v[1]); }, "quantile", "GSL", true);
      // logistic, note no location param
      logistic.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_logistic_P(x, v[1]); }, "CDF", "GSL");
      logistic.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_logistic_Pinv(x, v[1]); }, "quantile", "GSL", true);
      // pareto:
      pareto.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_pareto_P(x, v[1], v[0]); }, "CDF", "GSL");
      pareto.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_pareto_Pinv(x, v[1], v[0]); }, "quantile", "GSL", true);
      // weibull:
      weibull.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_weibull_P(x, v[1], v[0]); }, "CDF", "GSL");
      weibull.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_weibull_Pinv(x, v[1], v[0]); }, "quantile", "GSL", true);
      // poisson:
      poisson.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_poisson_P(x, v[0]); }, "CDF", "GSL");
      //poisson.run_timed_tests([](const std::vector<double>& v, double x){  return gsl_cdf_poisson_Pinv(x, v[0]); }, "quantile", "GSL", true);
      // binomial:
      binomial.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_binomial_P(x, v[1], v[0]); }, "CDF", "GSL");
      //binomial.run_timed_tests([](const std::vector<double>& v, double x){  return gsl_cdf_binomial_Pinv(x, v[1], v[0]); }, "quantile", "GSL", true);
      // negative_binomial:
      negative_binomial.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_negative_binomial_P(x, v[1], v[0]); }, "CDF", "GSL");
      //negative_binomial.run_timed_tests([](const std::vector<double>& v, double x){  return gsl_cdf_negative_binomial_Pinv(x, v[1], v[0]); }, "quantile", "GSL", true);
      // geometric:
      geometric.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_geometric_P(x + 1, v[0]); }, "CDF", "GSL");
      //geometric.run_timed_tests([](const std::vector<double>& v, double x){  return gsl_cdf_geometric_Pinv(x, v[0]) - 1; }, "quantile", "GSL", true);
      // hypergeometric:
      hypergeometric.run_timed_tests([](const std::vector<double>& v, double x) {  return gsl_cdf_hypergeometric_P(x, v[0], v[2] - v[0], v[1]); }, "CDF", "GSL");
      //hypergeometric.run_timed_tests([](const std::vector<double>& v, double x){  return gsl_cdf_hypergeometric_Pinv(x, v[0], v[2] - v[0], v[1]); }, "quantile", "GSL", true);
#endif

#ifdef TEST_RMATH
   // beta
      beta.run_timed_tests([](const std::vector<double>& v, double x) {  return dbeta(x, v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      beta.run_timed_tests([](const std::vector<double>& v, double x) {  return pbeta(x, v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      beta.run_timed_tests([](const std::vector<double>& v, double x) {  return qbeta(x, v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // non-central beta
      non_central_beta.run_timed_tests([](const std::vector<double>& v, double x) {  return dnbeta(x, v[0], v[1], v[2], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      non_central_beta.run_timed_tests([](const std::vector<double>& v, double x) {  return pnbeta(x, v[0], v[1], v[2], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      non_central_beta.run_timed_tests([](const std::vector<double>& v, double x) {  return qnbeta(x, v[0], v[1], v[2], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // binomial
      binomial.run_timed_tests([](const std::vector<double>& v, double x) {  return dbinom(x, v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      binomial.run_timed_tests([](const std::vector<double>& v, double x) {  return pbinom(x, v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      binomial.run_timed_tests([](const std::vector<double>& v, double x) {  return qbinom(x, v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // cauchy
      cauchy.run_timed_tests([](const std::vector<double>& v, double x) {  return dcauchy(x, v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      cauchy.run_timed_tests([](const std::vector<double>& v, double x) {  return pcauchy(x, v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      cauchy.run_timed_tests([](const std::vector<double>& v, double x) {  return qcauchy(x, v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // chi squared
      chi_squared.run_timed_tests([](const std::vector<double>& v, double x) {  return dchisq(x, v[0], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      chi_squared.run_timed_tests([](const std::vector<double>& v, double x) {  return pchisq(x, v[0], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      chi_squared.run_timed_tests([](const std::vector<double>& v, double x) {  return qchisq(x, v[0], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // non central chi squared
      non_central_chi_squared.run_timed_tests([](const std::vector<double>& v, double x) {  return dnchisq(x, v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      non_central_chi_squared.run_timed_tests([](const std::vector<double>& v, double x) {  return pnchisq(x, v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      non_central_chi_squared.run_timed_tests([](const std::vector<double>& v, double x) {  return qnchisq(x, v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // exponential
      exponential.run_timed_tests([](const std::vector<double>& v, double x) {  return dexp(x, 1 / v[0], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      exponential.run_timed_tests([](const std::vector<double>& v, double x) {  return pexp(x, 1 / v[0], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      exponential.run_timed_tests([](const std::vector<double>& v, double x) {  return qexp(x, 1 / v[0], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // F
      fisher.run_timed_tests([](const std::vector<double>& v, double x) {  return df(x, v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      fisher.run_timed_tests([](const std::vector<double>& v, double x) {  return pf(x, v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      fisher.run_timed_tests([](const std::vector<double>& v, double x) {  return qf(x, v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // non central F
      non_central_f.run_timed_tests([](const std::vector<double>& v, double x) {  return dnf(x, v[0], v[1], v[2], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      non_central_f.run_timed_tests([](const std::vector<double>& v, double x) {  return pnf(x, v[0], v[1], v[2], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      non_central_f.run_timed_tests([](const std::vector<double>& v, double x) {  return qnf(x, v[0], v[1], v[2], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // gamma
      gamma.run_timed_tests([](const std::vector<double>& v, double x) {  return dgamma(x, v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      gamma.run_timed_tests([](const std::vector<double>& v, double x) {  return pgamma(x, v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      gamma.run_timed_tests([](const std::vector<double>& v, double x) {  return qgamma(x, v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // geometric
      geometric.run_timed_tests([](const std::vector<double>& v, double x) {  return dgeom(x, v[0], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      geometric.run_timed_tests([](const std::vector<double>& v, double x) {  return pgeom(x, v[0], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      geometric.run_timed_tests([](const std::vector<double>& v, double x) {  return qgeom(x, v[0], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // hypergeometric
      hypergeometric.run_timed_tests([](const std::vector<double>& v, double x) {  return dhyper(x, v[0], v[2] - v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      hypergeometric.run_timed_tests([](const std::vector<double>& v, double x) {  return phyper(x, v[0], v[2] - v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      hypergeometric.run_timed_tests([](const std::vector<double>& v, double x) {  return qhyper(x, v[0], v[2] - v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // logistic
      logistic.run_timed_tests([](const std::vector<double>& v, double x) {  return dlogis(x, v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      logistic.run_timed_tests([](const std::vector<double>& v, double x) {  return plogis(x, v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      logistic.run_timed_tests([](const std::vector<double>& v, double x) {  return qlogis(x, v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // lognormal
      lognormal.run_timed_tests([](const std::vector<double>& v, double x) {  return dlnorm(x, v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      lognormal.run_timed_tests([](const std::vector<double>& v, double x) {  return plnorm(x, v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      lognormal.run_timed_tests([](const std::vector<double>& v, double x) {  return qlnorm(x, v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // negative_binomial
      negative_binomial.run_timed_tests([](const std::vector<double>& v, double x) {  return dnbinom(x, v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      negative_binomial.run_timed_tests([](const std::vector<double>& v, double x) {  return pnbinom(x, v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      negative_binomial.run_timed_tests([](const std::vector<double>& v, double x) {  return qnbinom(x, v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // normal
      n.run_timed_tests([](const std::vector<double>& v, double x) {  return dnorm(x, v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      n.run_timed_tests([](const std::vector<double>& v, double x) {  return pnorm(x, v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      n.run_timed_tests([](const std::vector<double>& v, double x) {  return qnorm(x, v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // poisson
      poisson.run_timed_tests([](const std::vector<double>& v, double x) {  return dpois(x, v[0], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      poisson.run_timed_tests([](const std::vector<double>& v, double x) {  return ppois(x, v[0], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      poisson.run_timed_tests([](const std::vector<double>& v, double x) {  return qpois(x, v[0], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // T
      students_t.run_timed_tests([](const std::vector<double>& v, double x) {  return dt(x, v[0], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      students_t.run_timed_tests([](const std::vector<double>& v, double x) {  return pt(x, v[0], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      students_t.run_timed_tests([](const std::vector<double>& v, double x) {  return qt(x, v[0], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // non central T
      non_central_t.run_timed_tests([](const std::vector<double>& v, double x) {  return dnt(x, v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      non_central_t.run_timed_tests([](const std::vector<double>& v, double x) {  return pnt(x, v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      non_central_t.run_timed_tests([](const std::vector<double>& v, double x) {  return qnt(x, v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);
      // weibull
      weibull.run_timed_tests([](const std::vector<double>& v, double x) {  return dweibull(x, v[0], v[1], 0); }, "PDF", "Rmath "  R_VERSION_STRING);
      weibull.run_timed_tests([](const std::vector<double>& v, double x) {  return pweibull(x, v[0], v[1], 1, 0); }, "CDF", "Rmath "  R_VERSION_STRING);
      weibull.run_timed_tests([](const std::vector<double>& v, double x) {  return qweibull(x, v[0], v[1], 1, 0); }, "quantile", "Rmath "  R_VERSION_STRING, true);

#endif

#ifdef TEST_DCDFLIB
      n.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_norm_cdf(x, v[0], v[1]); }, "CDF", "DCDFLIB");
      n.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_norm_quantile(x, v[0], v[1]); }, "quantile", "DCDFLIB", true);

      beta.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_beta_cdf(x, v[0], v[1]); }, "CDF", "DCDFLIB");
      beta.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_beta_quantile(x, v[0], v[1]); }, "quantile", "DCDFLIB", true);

      binomial.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_binomial_cdf(x, v[0], v[1]); }, "CDF", "DCDFLIB");
      binomial.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_binomial_quantile(x, v[0], v[1]); }, "quantile", "DCDFLIB", true);

      chi_squared.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_chi_cdf(x, v[0]); }, "CDF", "DCDFLIB");
      chi_squared.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_chi_quantile(x, v[0]); }, "quantile", "DCDFLIB", true);

      non_central_chi_squared.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_chi_n_cdf(x, v[0], v[1]); }, "CDF", "DCDFLIB");
      non_central_chi_squared.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_chi_n_quantile(x, v[0], v[1]); }, "quantile", "DCDFLIB", true);

      fisher.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_f_cdf(x, v[0], v[1]); }, "CDF", "DCDFLIB");
      fisher.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_f_quantile(x, v[0], v[1]); }, "quantile", "DCDFLIB", true);

      non_central_f.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_f_n_cdf(x, v[0], v[1], v[2]); }, "CDF", "DCDFLIB");
      non_central_f.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_f_n_quantile(x, v[0], v[1], v[2]); }, "quantile", "DCDFLIB", true);

      gamma.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_gamma_cdf(x, v[0], v[1]); }, "CDF", "DCDFLIB");
      gamma.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_gamma_quantile(x, v[0], v[1]); }, "quantile", "DCDFLIB", true);

      negative_binomial.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_nbin_cdf(x, v[0], v[1]); }, "CDF", "DCDFLIB");
      negative_binomial.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_nbin_quantile(x, v[0], v[1]); }, "quantile", "DCDFLIB", true);

      poisson.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_poisson_cdf(x, v[0]); }, "CDF", "DCDFLIB");
      poisson.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_poisson_quantile(x, v[0]); }, "quantile", "DCDFLIB", true);

      students_t.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_t_cdf(x, v[0]); }, "CDF", "DCDFLIB");
      students_t.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_t_quantile(x, v[0]); }, "quantile", "DCDFLIB", true);

      //non_central_t.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_t_n_cdf(x, v[0], v[1]); }, "CDF", "DCDFLIB");
      //non_central_t.run_timed_tests([](const std::vector<double>& v, double x) {  return dcdflib_t_n_quantile(x, v[0], v[1]); }, "quantile", "DCDFLIB", true);
#endif

   }
   catch(const std::exception& e)
   {
      std::cout << "Test run aborted due to thrown exception: " << e.what() << std::endl;
      return 1;
   }
   return 0;
}

