//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/math/special_functions/beta.hpp>
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
#  include "ibeta_inv_data.ipp"

   add_data(ibeta_inv_data);

   unsigned data_total = data.size();

   std::cout << "Screening boost data:\n";
   screen_data([](const std::vector<double>& v){  return boost::math::ibetac_inv(v[0], v[1], v[2]);  }, [](const std::vector<double>& v){ return v[4];  });


#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening libstdc++ data:\n";
      screen_data([](const std::vector<double>& v){  return ::qbeta(v[2], v[0], v[1], 0, 0);  }, [](const std::vector<double>& v){ return v[4];  });
#endif

   unsigned data_used = data.size();
   std::string function = "ibetac_inv[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "ibetac_inv";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::ibetac_inv(v[0], v[1], v[2]);  });
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
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::ibetac_inv(v[0], v[1], v[2], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return ::qbeta(v[2], v[0], v[1], 0, 0);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "Rmath "  R_VERSION_STRING);
#endif

   return 0;
}

