//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>
#include "../../test/table_type.hpp"
#include "table_helper.hpp"
#include "performance.hpp"
#include <iostream>

typedef double T;
#define SC_(x) static_cast<double>(x)
static const std::array<std::array<T, 3>, 19> data1 = { {
   { { SC_(0.0), SC_(0.0), SC_(0.0) } },
   { { SC_(-10.0), SC_(0.0), SC_(-10.0) } },
   { { SC_(-1.0), SC_(-1.0), SC_(-1.2261911708835170708130609674719067527242483502207) } },
   { { SC_(-4.0), SC_(0.875), SC_(-5.3190556182262405182189463092940736859067548232647) } },
   { { SC_(8.0), SC_(-0.625), SC_(9.0419973860310100524448893214394562615252527557062) } },
   { { SC_(1e-05), SC_(0.875), SC_(0.000010000000000127604166668510945638036143355898993088) } },
   { { SC_(1e+05), T(10) / 1024, SC_(100002.38431454899771096037307519328741455615271038) } },
   { { SC_(1e-20), SC_(1.0), SC_(1.0000000000000000000000000000000000000000166666667e-20) } },
   { { SC_(1e-20), SC_(1e-20), SC_(1.000000000000000e-20) } },
   { { SC_(1e+20), T(400) / 1024, SC_(1.0418143796499216839719289963154558027005142709763e20) } },
   { { SC_(1e+50), SC_(0.875), SC_(1.3913251718238765549409892714295358043696028445944e50) } },
   { { SC_(2.0), SC_(0.5), SC_(2.1765877052210673672479877957388515321497888026770) } },
   { { SC_(4.0), SC_(0.5), SC_(4.2543274975235836861894752787874633017836785640477) } },
   { { SC_(6.0), SC_(0.5), SC_(6.4588766202317746302999080620490579800463614807916) } },
   { { SC_(10.0), SC_(0.5), SC_(10.697409951222544858346795279378531495869386960090) } },
   { { SC_(-2.0), SC_(0.5), SC_(-2.1765877052210673672479877957388515321497888026770) } },
   { { SC_(-4.0), SC_(0.5), SC_(-4.2543274975235836861894752787874633017836785640477) } },
   { { SC_(-6.0), SC_(0.5), SC_(-6.4588766202317746302999080620490579800463614807916) } },
   { { SC_(-10.0), SC_(0.5), SC_(-10.697409951222544858346795279378531495869386960090) } },
   } };

int main()
{
#include "ellint_f_data.ipp"

   add_data(data1);
   add_data(ellint_f_data);

   unsigned data_total = data.size();

   screen_data([](const std::vector<double>& v){  return boost::math::ellint_1(v[1], v[0]);  }, [](const std::vector<double>& v){ return v[2];  });


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return std::tr1::ellint_1(v[1], v[0]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return gsl_sf_ellint_F(v[0], v[1], GSL_PREC_DOUBLE);  }, [](const std::vector<double>& v){ return v[2];  });
#endif

   unsigned data_used = data.size();
   std::string function = "ellint_1[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "ellint_1";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::ellint_1(v[1], v[0]);  });
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
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::ellint_1(v[1], v[0], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::ellint_1(v[1], v[0]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_ellint_F(v[0], v[1], GSL_PREC_DOUBLE);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif

   return 0;
}

