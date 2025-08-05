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
static const std::array<std::array<T, 3>, 21> jv_data = { {
      //SC_(-2.4), {{ SC_(0.0), std::numeric_limits<T>::infinity() }},
   { { T(22.5), T(0.0), SC_(0.0) } },
   { { T(2457.0) / 1024, T(1.0) / 1024, SC_(3.80739920118603335646474073457326714709615200130620574875292e-9) } },
   { { SC_(5.5), T(3217) / 1024, SC_(0.0281933076257506091621579544064767140470089107926550720453038) } },
   { { SC_(-5.5), T(3217) / 1024, SC_(-2.55820064470647911823175836997490971806135336759164272675969) } },
   { { SC_(-5.5), SC_(1e+04), SC_(2.449843111985605522111159013846599118397e-03) } },
   { { SC_(5.5), SC_(1e+04), SC_(0.00759343502722670361395585198154817047185480147294665270646578) } },
   { { SC_(5.5), SC_(1e+06), SC_(-0.000747424248595630177396350688505919533097973148718960064663632) } },
   { { SC_(5.125), SC_(1e+06), SC_(-0.000776600124835704280633640911329691642748783663198207360238214) } },
   { { SC_(5.875), SC_(1e+06), SC_(-0.000466322721115193071631008581529503095819705088484386434589780) } },
   { { SC_(0.5), SC_(101.0), SC_(0.0358874487875643822020496677692429287863419555699447066226409) } },
   { { SC_(-5.5), SC_(1e+04), SC_(0.00244984311198560552211115901384659911839737686676766460822577) } },
   { { SC_(-5.5), SC_(1e+06), SC_(0.000279243200433579511095229508894156656558211060453622750659554) } },
   { { SC_(-0.5), SC_(101.0), SC_(0.0708184798097594268482290389188138201440114881159344944791454) } },
   { { T(-10486074) / (1024 * 1024), T(1) / 1024, SC_(1.41474013160494695750009004222225969090304185981836460288562e35) } },
   { { T(-10486074) / (1024 * 1024), SC_(15.0), SC_(-0.0902239288885423309568944543848111461724911781719692852541489) } },
   { { T(10486074) / (1024 * 1024), SC_(1e+02), SC_(-0.0547064914615137807616774867984047583596945624129838091326863) } },
   { { T(10486074) / (1024 * 1024), SC_(2e+04), SC_(-0.00556783614400875611650958980796060611309029233226596737701688) } },
   { { T(-10486074) / (1024 * 1024), SC_(1e+02), SC_(-0.0547613660316806551338637153942604550779513947674222863858713) } },
   // Bug report https://svn.boost.org/trac/boost/ticket/4812:
   { { SC_(1.5), T(8034) / 1024, SC_(0.0339477646369710610146236955872928005087352629422508823945264) } },
   { { SC_(8.5), boost::math::constants::pi<T>() * 4, SC_(0.0436807946352780974532519564114026730332781693877984686758680) } },
   { { SC_(-8.5), boost::math::constants::pi<T>() * 4, SC_(-0.257086543428224355151772807588810984369026142375675714560864) } },
   } };

int main()
{
#include "bessel_j_data.ipp"
#include "bessel_j_large_data.ipp"

   add_data(jv_data);
   add_data(bessel_j_data);
   add_data(bessel_j_large_data);

   unsigned data_total = data.size();

   std::cout << "Screening boost data:\n";
   screen_data([](const std::vector<double>& v){  return boost::math::cyl_bessel_j(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[2];  });

#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening libstdc++ data:\n";
   screen_data([](const std::vector<double>& v){  return std::tr1::cyl_bessel_j(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening libstdc++ data:\n";
   screen_data([](const std::vector<double>& v){  return gsl_sf_bessel_Jnu(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening libstdc++ data:\n";
   screen_data([](const std::vector<double>& v){  return bessel_j(v[1], v[0]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif

   unsigned data_used = data.size();
   std::string function = "cyl_bessel_j[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "cyl_bessel_j";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_bessel_j(v[0], v[1]);  });
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
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_bessel_j(v[0], v[1], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::cyl_bessel_j(v[0], v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_bessel_Jnu(v[0], v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return bessel_j(v[1], v[0]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "Rmath "  R_VERSION_STRING);
#endif

   return 0;
}

