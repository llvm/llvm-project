//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/math/special_functions/bessel.hpp>
#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>
#include "../../test/table_type.hpp"
#include "table_helper.hpp"
#include "performance.hpp"
#include <iostream>

typedef double T;
#define SC_(x) static_cast<double>(x)
static const std::array<std::array<T, 3>, 11> yv_data = { {
      //SC_(2.25), {{ SC_(1.0) / 1024, SC_(-1.01759203636941035147948317764932151601257765988969544340275e7) }},
   { { SC_(0.5), T(1) / (1024 * 1024), SC_(-817.033790261762580469303126467917092806755460418223776544122) } },
   { { SC_(5.5), SC_(3.125), SC_(-2.61489440328417468776474188539366752698192046890955453259866) } },
   { { SC_(-5.5), SC_(3.125), SC_(-0.0274994493896489729948109971802244976377957234563871795364056) } },
   { { SC_(-5.5), SC_(1e+04), SC_(-0.00759343502722670361395585198154817047185480147294665270646578) } },
   { { T(-10486074) / (1024 * 1024), T(1) / 1024, SC_(-1.50382374389531766117868938966858995093408410498915220070230e38) } },
   { { T(-10486074) / (1024 * 1024), SC_(1e+02), SC_(0.0583041891319026009955779707640455341990844522293730214223545) } },
   { { SC_(141.75), SC_(1e+02), SC_(-5.38829231428696507293191118661269920130838607482708483122068e9) } },
   { { SC_(141.75), SC_(2e+04), SC_(-0.00376577888677186194728129112270988602876597726657372330194186) } },
   { { SC_(-141.75), SC_(1e+02), SC_(-3.81009803444766877495905954105669819951653361036342457919021e9) } },
   { { SC_(8.5), boost::math::constants::pi<T>() * 4, SC_(0.257086543428224355151772807588810984369026142375675714560864) } },
   { { SC_(-8.5), boost::math::constants::pi<T>() * 4, SC_(0.0436807946352780974532519564114026730332781693877984686758680) } },
   } };
static const std::array<std::array<T, 3>, 7> yv_large_data = { {
      // Bug report https://svn.boost.org/trac/boost/ticket/5560:
   { { SC_(0.5), static_cast<T>(std::ldexp(0.5, -683)), SC_(-7.14823099969225685526188875418476476336424046896822867989728e102) } },
   { { SC_(-0.5), static_cast<T>(std::ldexp(0.5, -683)), SC_(8.90597649117647254543282704099383321071493400182381039079219e-104) } },
   { { SC_(0.0), static_cast<T>(std::ldexp(1.0, -53)), SC_(-23.4611779112897561252987257324561640034037313549011724328997) } },
   { { SC_(1.0), static_cast<T>(std::ldexp(1.0, -53)), SC_(-5.73416113922265864550047623401604244038331542638719289100990e15) } },
   { { SC_(2.0), static_cast<T>(std::ldexp(1.0, -53)), SC_(-1.03297463879542177245046832533417970379386617249046560049244e32) } },
   { { SC_(3.0), static_cast<T>(std::ldexp(1.0, -53)), SC_(-3.72168335868978735639260528876490232745489151562358712422544e48) } },
   { { SC_(10.0), static_cast<T>(std::ldexp(1.0, -53)), SC_(-4.15729476804920974669173904282420477878640623992500096231384e167) } },
   } };

int main()
{
#include "bessel_yv_data.ipp"

   add_data(yv_data);
   add_data(yv_large_data);
   add_data(bessel_yv_data);

   unsigned data_total = data.size();

   std::cout << "Screening boost data:\n";
   screen_data([](const std::vector<double>& v){  return boost::math::cyl_neumann(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[2];  });


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening libstdc++ data:\n";
   screen_data([](const std::vector<double>& v){  return std::tr1::cyl_neumann(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening GSL data:\n";
   screen_data([](const std::vector<double>& v){  return gsl_sf_bessel_Ynu(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening Rmath data:\n";
   screen_data([](const std::vector<double>& v){  return bessel_y(v[1], v[0]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif

   unsigned data_used = data.size();
   std::string function = "cyl_neumann[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "cyl_neumann";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_neumann(v[0], v[1]);  });
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
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_neumann(v[0], v[1], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::cyl_neumann(v[0], v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_bessel_Ynu(v[0], v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return bessel_y(v[1], v[0]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "Rmath "  R_VERSION_STRING);
#endif

   return 0;
}

