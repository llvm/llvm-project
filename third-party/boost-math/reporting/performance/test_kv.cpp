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
static const std::array<std::array<T, 3>, 11> kv_data = { {
   { { SC_(0.5), SC_(0.875), SC_(0.558532231646608646115729767013630967055657943463362504577189) } },
   { { SC_(0.5), SC_(1.125), SC_(0.383621010650189547146769320487006220295290256657827220786527) } },
   { { SC_(2.25), T(std::ldexp(1.0, -30)), SC_(5.62397392719283271332307799146649700147907612095185712015604e20) } },
   { { SC_(5.5), T(3217) / 1024, SC_(1.30623288775012596319554857587765179889689223531159532808379) } },
   { { SC_(-5.5), SC_(10.0), SC_(0.0000733045300798502164644836879577484533096239574909573072142667) } },
   { { SC_(-5.5), SC_(100.0), SC_(5.41274555306792267322084448693957747924412508020839543293369e-45) } },
   { { T(10240) / 1024, T(1) / 1024, SC_(2.35522579263922076203415803966825431039900000000993410734978e38) } },
   { { T(10240) / 1024, SC_(10.0), SC_(0.00161425530039067002345725193091329085443750382929208307802221) } },
   { { T(144793) / 1024, SC_(100.0), SC_(1.39565245860302528069481472855619216759142225046370312329416e-6) } },
   { { T(144793) / 1024, SC_(200.0), SC_(9.11950412043225432171915100042647230802198254567007382956336e-68) } },
   { { T(-144793) / 1024, SC_(50.0), SC_(1.30185229717525025165362673848737761549946548375142378172956e42) } },
   } };
static const std::array<std::array<T, 3>, 5> kv_large_data = { {
      // Bug report https://svn.boost.org/trac/boost/ticket/5560:
   { { SC_(-1.0), static_cast<T>(ldexp(0.5, -512)), SC_(2.68156158598851941991480499964116922549587316411847867554471e154) } },
   { { SC_(1.0), static_cast<T>(ldexp(0.5, -512)), SC_(2.68156158598851941991480499964116922549587316411847867554471e154) } },
   { { SC_(-1.125), static_cast<T>(ldexp(0.5, -512)), SC_(5.53984048006472105611199242328122729730752165907526178753978e173) } },
   { { SC_(1.125), static_cast<T>(ldexp(0.5, -512)), SC_(5.53984048006472105611199242328122729730752165907526178753978e173) } },
   { { SC_(0.5), static_cast<T>(ldexp(0.5, -683)), SC_(1.12284149973980088540335945247019177715948513804063794284101e103) } },
   } };

int main()
{
#include "bessel_k_data.ipp"

   add_data(kv_data);
   add_data(kv_large_data);
   add_data(bessel_k_data);

   unsigned data_total = data.size();

   std::cout << "Screening boost data:\n";
   screen_data([](const std::vector<double>& v){  return boost::math::cyl_bessel_k(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[2];  });

#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening libstdc++ data:\n";
   screen_data([](const std::vector<double>& v){  return std::tr1::cyl_bessel_k(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening libstdc++ data:\n";
   screen_data([](const std::vector<double>& v){  return gsl_sf_bessel_Knu(v[0], v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening libstdc++ data:\n";
   screen_data([](const std::vector<double>& v){  return bessel_k(v[1], v[0], 1);  }, [](const std::vector<double>& v){ return v[2];  });
#endif

   unsigned data_used = data.size();
   std::string function = "cyl_bessel_k[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "cyl_bessel_k";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_bessel_k(v[0], v[1]);  });
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
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_bessel_k(v[0], v[1], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::cyl_bessel_k(v[0], v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_bessel_Knu(v[0], v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return bessel_k(v[1], v[0], 1);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "Rmath "  R_VERSION_STRING);
#endif

   return 0;
}

