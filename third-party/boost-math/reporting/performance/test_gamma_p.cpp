//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/math/special_functions/gamma.hpp>
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
#  include "igamma_med_data.ipp"
#  include "igamma_small_data.ipp"
#  include "igamma_big_data.ipp"
#  include "igamma_int_data.ipp"

   add_data(igamma_med_data);
   add_data(igamma_small_data);
   add_data(igamma_big_data);
   add_data(igamma_int_data);

   unsigned data_total = data.size();

   screen_data([](const std::vector<double>& v){  return boost::math::gamma_p(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[5];  });


#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return gsl_sf_gamma_inc_P(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[5];  });
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return pgamma(v[1], v[0], 1.0, 1, 0);  }, [](const std::vector<double>& v){ return v[5];  });
#endif

   unsigned data_used = data.size();
   std::string function = "gamma_p[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "gamma_p";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::gamma_p(v[0], v[1]);  });
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
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::gamma_p(v[0], v[1], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_gamma_inc_P(v[0], v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return pgamma(v[1], v[0], 1.0, 1, 0);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "Rmath "  R_VERSION_STRING);
#endif

   return 0;
}

