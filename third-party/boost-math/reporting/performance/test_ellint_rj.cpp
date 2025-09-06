//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/math/special_functions/ellint_rj.hpp>
#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>
#include "../../test/table_type.hpp"
#include "table_helper.hpp"
#include "performance.hpp"
#include <iostream>

typedef double T;
#define SC_(x) static_cast<double>(x)

int main()
{
#include "ellint_rj_data.ipp"
#include "ellint_rj_e4.ipp"
#include "ellint_rj_e3.ipp"
#include "ellint_rj_e2.ipp"
#include "ellint_rj_zp.ipp"

   add_data(ellint_rj_data);
   add_data(ellint_rj_e4);
   add_data(ellint_rj_e3);
   add_data(ellint_rj_e2);
   add_data(ellint_rj_zp);

   unsigned data_total = data.size();

   screen_data([](const std::vector<double>& v){  return boost::math::ellint_rj(v[0], v[1], v[2], v[3]);  }, [](const std::vector<double>& v){ return v[4];  });


#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return gsl_sf_ellint_RJ(v[0], v[1], v[2], v[3], GSL_PREC_DOUBLE);  }, [](const std::vector<double>& v){ return v[4];  });
#endif

   unsigned data_used = data.size();
   std::string function = "ellint_rj[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "ellint_rj";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::ellint_rj(v[0], v[1], v[2], v[3]);  });
   std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH))
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name());
#endif
   report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name());
   //
   // Boost again, but with promotion to long double turned off:
   //
#if !defined(COMPILER_COMPARISON_TABLES)
   if(sizeof(long double) != sizeof(double))
   {
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::ellint_rj(v[0], v[1], v[2], v[3], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_ellint_RJ(v[0], v[1], v[2], v[3], GSL_PREC_DOUBLE);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif

   return 0;
}

