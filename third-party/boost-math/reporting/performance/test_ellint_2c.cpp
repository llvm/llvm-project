//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/math/special_functions/ellint_2.hpp>
#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>
#include "../../test/table_type.hpp"
#include "table_helper.hpp"
#include "performance.hpp"
#include <iostream>

typedef double T;
#define SC_(x) static_cast<double>(x)
static const std::array<std::array<T, 2>, 10> data2 = { {
   { { SC_(-1.0), SC_(1.0) } },
   { { SC_(0.0), SC_(1.5707963267948966192313216916397514420985846996876) } },
   { { T(100) / 1024, SC_(1.5670445330545086723323795143598956428788609133377) } },
   { { T(200) / 1024, SC_(1.5557071588766556854463404816624361127847775545087) } },
   { { T(300) / 1024, SC_(1.5365278991162754883035625322482669608948678755743) } },
   { { T(400) / 1024, SC_(1.5090417763083482272165682786143770446401437564021) } },
   { { SC_(-0.5), SC_(1.4674622093394271554597952669909161360253617523272) } },
   { { T(-600) / 1024, SC_(1.4257538571071297192428217218834579920545946473778) } },
   { { T(-800) / 1024, SC_(1.2927868476159125056958680222998765985004489572909) } },
   { { T(-900) / 1024, SC_(1.1966864890248739524112920627353824133420353430982) } },
   } };

int main()
{
#include "ellint_e_data.ipp"

   add_data(data2);
   add_data(ellint_e_data);

   unsigned data_total = data.size();

   screen_data([](const std::vector<double>& v){  return boost::math::ellint_2(v[0]);  }, [](const std::vector<double>& v){ return v[1];  });


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return std::tr1::comp_ellint_2(v[0]);  }, [](const std::vector<double>& v){ return v[1];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return gsl_sf_ellint_Ecomp(v[0], GSL_PREC_DOUBLE);  }, [](const std::vector<double>& v){ return v[1];  });
#endif

   unsigned data_used = data.size();
   std::string function = "ellint_2 (complete)[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "ellint_2 (complete)";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::ellint_2(v[0]);  });
   std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name());
#endif
   report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name());
   //
   // Boost again, but with promotion to long double turned off:
   //
#if !defined(COMPILER_COMPARISON_TABLES)
   if(sizeof(long double) != sizeof(double))
   {
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::ellint_2(v[0], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::comp_ellint_2(v[0]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_ellint_Ecomp(v[0], GSL_PREC_DOUBLE);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif

   return 0;
}

