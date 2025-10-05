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
static const std::array<std::array<T, 3>, 9> k0_data = { {
   { { SC_(0.0), SC_(1.0), SC_(0.421024438240708333335627379212609036136219748226660472298970) } },
   { { SC_(0.0), SC_(2.0), SC_(0.113893872749533435652719574932481832998326624388808882892530) } },
   { { SC_(0.0), SC_(4.0), SC_(0.0111596760858530242697451959798334892250090238884743405382553) } },
   { { SC_(0.0), SC_(8.0), SC_(0.000146470705222815387096584408698677921967305368833759024089154) } },
   { { SC_(0.0), T(std::ldexp(1.0, -15)), SC_(10.5131392267382037062459525561594822400447325776672021972753) } },
   { { SC_(0.0), T(std::ldexp(1.0, -30)), SC_(20.9103469324567717360787328239372191382743831365906131108531) } },
   { { SC_(0.0), T(std::ldexp(1.0, -60)), SC_(41.7047623492551310138446473188663682295952219631968830346918) } },
   { { SC_(0.0), SC_(50.0), SC_(3.41016774978949551392067551235295223184502537762334808993276e-23) } },
   { { SC_(0.0), SC_(100.0), SC_(4.65662822917590201893900528948388635580753948544211387402671e-45) } },
   } };
static const std::array<std::array<T, 3>, 9> k1_data = { {
   { { SC_(1.0), SC_(1.0), SC_(0.601907230197234574737540001535617339261586889968106456017768) } },
   { { SC_(1.0), SC_(2.0), SC_(0.139865881816522427284598807035411023887234584841515530384442) } },
   { { SC_(1.0), SC_(4.0), SC_(0.0124834988872684314703841799808060684838415849886258457917076) } },
   { { SC_(1.0), SC_(8.0), SC_(0.000155369211805001133916862450622474621117065122872616157079566) } },
   { { SC_(1.0), T(std::ldexp(1.0, -15)), SC_(32767.9998319528316432647441316539139725104728341577594326513) } },
   { { SC_(1.0), T(std::ldexp(1.0, -30)), SC_(1.07374182399999999003003028572687332810353799544215073362305e9) } },
   { { SC_(1.0), T(std::ldexp(1.0, -60)), SC_(1.15292150460684697599999999999999998169660198868126604634036e18) } },
   { { SC_(1.0), SC_(50.0), SC_(3.44410222671755561259185303591267155099677251348256880221927e-23) } },
   { { SC_(1.0), SC_(100.0), SC_(4.67985373563690928656254424202433530797494354694335352937465e-45) } },
   } };
static const std::array<std::array<T, 3>, 9> kn_data = { {
   { { SC_(2.0), T(std::ldexp(1.0, -30)), SC_(2.30584300921369395150000000000000000234841952009593636868109e18) } },
   { { SC_(5.0), SC_(10.0), SC_(0.0000575418499853122792763740236992723196597629124356739596921536) } },
   { { SC_(-5.0), SC_(100.0), SC_(5.27325611329294989461777188449044716451716555009882448801072e-45) } },
   { { SC_(10.0), SC_(10.0), SC_(0.00161425530039067002345725193091329085443750382929208307802221) } },
   { { SC_(10.0), T(std::ldexp(1.0, -30)), SC_(3.78470202927236255215249281534478864916684072926050665209083e98) } },
   { { SC_(-10.0), SC_(1.0), SC_(1.80713289901029454691597861302340015908245782948536080022119e8) } },
   { { SC_(100.0), SC_(5.0), SC_(7.03986019306167654653386616796116726248616158936088056952477e115) } },
   { { SC_(100.0), SC_(80.0), SC_(8.39287107246490782848985384895907681748152272748337807033319e-12) } },
   { { SC_(-129.0), SC_(200.0), SC_(3.61744436315860678558682169223740584132967454950379795115566e-71) } },
   } };

int main()
{
#include "bessel_k_int_data.ipp"

   add_data(k0_data);
   add_data(k1_data);
   add_data(kn_data);
   add_data(bessel_k_int_data);

   unsigned data_total = data.size();

   screen_data([](const std::vector<double>& v){  return boost::math::cyl_bessel_k(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });

#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return std::tr1::cyl_bessel_k(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return gsl_sf_bessel_Kn(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return bessel_k(v[1], static_cast<int>(v[0]), 1);  }, [](const std::vector<double>& v){ return v[2];  });
#endif

   unsigned data_used = data.size();
   std::string function = "cyl_bessel_k (integer order)[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "cyl_bessel_k (integer order)";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_bessel_k(static_cast<int>(v[0]), v[1]);  });
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
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_bessel_k(static_cast<int>(v[0]), v[1], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::cyl_bessel_k(static_cast<int>(v[0]), v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_bessel_Kn(static_cast<int>(v[0]), v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return bessel_k(v[1], static_cast<int>(v[0]), 1);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "Rmath "  R_VERSION_STRING);
#endif

   return 0;
}

