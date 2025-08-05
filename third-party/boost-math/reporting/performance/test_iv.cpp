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
static const std::array<std::array<T, 3>, 10> iv_data = { {
   { { SC_(2.25), T(1) / (1024 * 1024), SC_(2.34379212133481347189068464680335815256364262507955635911656e-15) } },
   { { SC_(5.5), SC_(3.125), SC_(0.0583514045989371500460946536220735787163510569634133670181210) } },
   { { T(-5) + T(1) / 1024, SC_(2.125), SC_(0.0267920938009571023702933210070984416052633027166975342895062) } },
   { { SC_(-5.5), SC_(10.0), SC_(597.577606961369169607937419869926705730305175364662688426534) } },
   { { SC_(-5.5), SC_(100.0), SC_(9.22362906144706871737354069133813819358704200689067071415379e41) } },
   { { T(-10486074) / (1024 * 1024), T(1) / 1024, SC_(1.41474005665181350367684623930576333542989766867888186478185e35) } },
   { { T(-10486074) / (1024 * 1024), SC_(50.0), SC_(1.07153277202900671531087024688681954238311679648319534644743e20) } },
   { { T(144794) / 1024, SC_(100.0), SC_(2066.27694757392660413922181531984160871678224178890247540320) } },
   { { T(144794) / 1024, SC_(200.0), SC_(2.23699739472246928794922868978337381373643889659337595319774e64) } },
   { { T(-144794) / 1024, SC_(100.0), SC_(2066.27694672763190927440969155740243346136463461655104698748) } },
   } };
static const std::array<std::array<T, 3>, 5> iv_large_data = { {
      // Bug report https://svn.boost.org/trac/boost/ticket/5560:
   { { SC_(-1.0), static_cast<T>(ldexp(0.5, -512)), SC_(1.86458518280005168582274132886573345934411788365010172356788e-155) } },
   { { SC_(1.0), static_cast<T>(ldexp(0.5, -512)), SC_(1.86458518280005168582274132886573345934411788365010172356788e-155) } },
   { { SC_(-1.125), static_cast<T>(ldexp(0.5, -512)), SC_(-1.34963720853101363690381585556234820027343435206156667634081e173) } },
   { { SC_(1.125), static_cast<T>(ldexp(0.5, -512)), SC_(8.02269390325932403421158766283366891170783955777638875887348e-175) } },
   { { SC_(0.5), static_cast<T>(ldexp(0.5, -683)), SC_(8.90597649117647254543282704099383321071493400182381039079219e-104) } },
   } };

int main()
{
#include "bessel_i_data.ipp"

   add_data(iv_data);
   add_data(iv_large_data);
   add_data(bessel_i_data);

   unsigned data_total = data.size();

   std::cout << "Screening boost data:\n";
   screen_data([](const std::vector<double>& v){  return boost::math::cyl_bessel_i(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[2];  });

#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening libstdc++ data:\n";
   screen_data([](const std::vector<double>& v){  return std::tr1::cyl_bessel_i(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening GSL data:\n";
   screen_data([](const std::vector<double>& v){  return gsl_sf_bessel_Inu(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening GSL data:\n";
   screen_data([](const std::vector<double>& v){  return bessel_i(v[1], v[0], 1);  }, [](const std::vector<double>& v){ return v[2];  });
#endif

   unsigned data_used = data.size();
   std::string function = "cyl_bessel_i[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "cyl_bessel_i";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_bessel_i(v[0], v[1]);  });
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
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_bessel_i(v[0], v[1], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::cyl_bessel_i(v[0], v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_bessel_Inu(v[0], v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return bessel_i(v[1], v[0], 1);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "Rmath "  R_VERSION_STRING);
#endif

   return 0;
}

