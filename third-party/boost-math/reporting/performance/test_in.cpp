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
static const std::array<std::array<T, 3>, 10> i0_data = { {
   { { SC_(0.0), SC_(0.0), SC_(1.0) } },
   { { SC_(0.0), SC_(1.0), SC_(1.26606587775200833559824462521471753760767031135496220680814) } },
   { { SC_(0.0), SC_(-2.0), SC_(2.27958530233606726743720444081153335328584110278545905407084) } },
   { { SC_(0.0), SC_(4.0), SC_(11.3019219521363304963562701832171024974126165944353377060065) } },
   { { SC_(0.0), SC_(-7.0), SC_(168.593908510289698857326627187500840376522679234531714193194) } },
   { { SC_(0.0), T(1) / 1024, SC_(1.00000023841859331241759166109699567801556273303717896447683) } },
   { { SC_(0.0), T(SC_(1.0)) / (1024 * 1024), SC_(1.00000000000022737367544324498417583090700894607432256476338) } },
   { { SC_(0.0), SC_(-1.0), SC_(1.26606587775200833559824462521471753760767031135496220680814) } },
   { { SC_(0.0), SC_(100.0), SC_(1.07375170713107382351972085760349466128840319332527279540154e42) } },
   { { SC_(0.0), SC_(200.0), SC_(2.03968717340972461954167312677945962233267573614834337894328e85) } },
   } };
static const std::array<std::array<T, 3>, 10> i1_data = { {
   { { SC_(1.0), SC_(0.0), SC_(0.0) } },
   { { SC_(1.0), SC_(1.0), SC_(0.565159103992485027207696027609863307328899621621092009480294) } },
   { { SC_(1.0), SC_(-2.0), SC_(-1.59063685463732906338225442499966624795447815949553664713229) } },
   { { SC_(1.0), SC_(4.0), SC_(9.75946515370444990947519256731268090005597033325296730692753) } },
   { { SC_(1.0), SC_(-8.0), SC_(-399.873136782560098219083086145822754889628443904067647306574) } },
   { { SC_(1.0), T(SC_(1.0)) / 1024, SC_(0.000488281308207663226432087816784315537514225208473395063575150) } },
   { { SC_(1.0), T(SC_(1.0)) / (1024 * 1024), SC_(4.76837158203179210108624277276025646653133998635956784292029E-7) } },
   { { SC_(1.0), SC_(-10.0), SC_(-2670.98830370125465434103196677215254914574515378753771310849) } },
   { { SC_(1.0), SC_(100.0), SC_(1.06836939033816248120614576322429526544612284405623226965918e42) } },
   { { SC_(1.0), SC_(200.0), SC_(2.03458154933206270342742797713906950389661161681122964159220e85) } },
   } };
static const std::array<std::array<T, 3>, 11> in_data = { {
   { { SC_(-2.0), SC_(0.0), SC_(0.0) } },
   { { SC_(2.0), T(SC_(1.0)) / (1024 * 1024), SC_(1.13686837721624646204093977095674566928522671779753217215467e-13) } },
   { { SC_(5.0), SC_(10.0), SC_(777.188286403259959907293484802339632852674154572666041953297) } },
   { { SC_(-5.0), SC_(100.0), SC_(9.47009387303558124618275555002161742321578485033007130107740e41) } },
   { { SC_(-5.0), SC_(-1.0), SC_(-0.000271463155956971875181073905153777342383564426758143634974124) } },
   { { SC_(10.0), SC_(20.0), SC_(3.54020020901952109905289138244985607057267103782948493874391e6) } },
   { { SC_(10.0), SC_(-5.0), SC_(0.00458004441917605126118647027872016953192323139337073320016447) } },
   { { SC_(1e+02), SC_(9.0), SC_(2.74306601746058997093587654668959071522869282506446891736820e-93) } },
   { { SC_(1e+02), SC_(80.0), SC_(4.65194832850610205318128191404145885093970505338730540776711e8) } },
   { { SC_(-100.0), SC_(-200.0), SC_(4.35275044972702191438729017441198257508190719030765213981307e74) } },
   { { SC_(10.0), SC_(1e-100), SC_(2.69114445546737213403880070546737213403880070546737213403880e-1010) } },
   } };

int main()
{
#include "bessel_i_int_data.ipp"

   add_data(i0_data);
   add_data(i1_data);
   add_data(in_data);
   add_data(bessel_i_int_data);

   unsigned data_total = data.size();

   screen_data([](const std::vector<double>& v){  return boost::math::cyl_bessel_i(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });

#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return std::tr1::cyl_bessel_i(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return gsl_sf_bessel_In(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return bessel_i(v[1], static_cast<int>(v[0]), 1);  }, [](const std::vector<double>& v){ return v[2];  });
#endif

   unsigned data_used = data.size();
   std::string function = "cyl_bessel_i (integer order)[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "cyl_bessel_i (integer order)";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_bessel_i(static_cast<int>(v[0]), v[1]);  });
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
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_bessel_i(static_cast<int>(v[0]), v[1], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::cyl_bessel_i(static_cast<int>(v[0]), v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_bessel_In(static_cast<int>(v[0]), v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return bessel_i(v[1], static_cast<int>(v[0]), 1);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "Rmath "  R_VERSION_STRING);
#endif

   return 0;
}

