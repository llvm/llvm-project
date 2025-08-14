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
static const std::array<std::array<T, 3>, 10> data1 = { {
   { { SC_(0.0), SC_(0.0), SC_(0.0) } },
   { { SC_(-10.0), SC_(0.0), SC_(-10.0) } },
   { { SC_(-1.0), SC_(-1.0), SC_(-0.84147098480789650665250232163029899962256306079837) } },
   { { SC_(-4.0), T(900) / 1024, SC_(-3.1756145986492562317862928524528520686391383168377) } },
   { { SC_(8.0), T(-600) / 1024, SC_(7.2473147180505693037677015377802777959345489333465) } },
   { { SC_(1e-05), T(800) / 1024, SC_(9.999999999898274739584436515967055859383969942432E-6) } },
   { { SC_(1e+05), T(100) / 1024, SC_(99761.153306972066658135668386691227343323331995888) } },
   { { SC_(1e+10), SC_(-0.5), SC_(9.3421545766487137036576748555295222252286528414669e9) } },
   { { static_cast<T>(ldexp(T(1), 66)), T(400) / 1024, SC_(7.0886102721911705466476846969992069994308167515242e19) } },
   { { static_cast<T>(ldexp(T(1), 166)), T(900) / 1024, SC_(7.1259011068364515942912094521783688927118026465790e49) } },
   } };

int main()
{
#include "ellint_e2_data.ipp"

   add_data(data1);
   add_data(ellint_e2_data);

   unsigned data_total = data.size();

   screen_data([](const std::vector<double>& v){  return boost::math::ellint_2(v[1], v[0]);  }, [](const std::vector<double>& v){ return v[2];  });


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return std::tr1::ellint_2(v[1], v[0]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return gsl_sf_ellint_E(v[0], v[1], GSL_PREC_DOUBLE);  }, [](const std::vector<double>& v){ return v[2];  });
#endif

   unsigned data_used = data.size();
   std::string function = "ellint_2[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "ellint_2";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::ellint_2(v[1], v[0]);  });
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
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::ellint_2(v[1], v[0], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::ellint_2(v[1], v[0]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_ellint_E(v[0], v[1], GSL_PREC_DOUBLE);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif

   return 0;
}

