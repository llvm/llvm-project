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
static const std::array<std::array<typename table_type<T>::type, 3>, 9> y0_data = { {
   { { SC_(0.0), SC_(1.0), SC_(0.0882569642156769579829267660235151628278175230906755467110438) } },
   { { SC_(0.0), SC_(2.0), SC_(0.510375672649745119596606592727157873268139227085846135571839) } },
   { { SC_(0.0), SC_(4.0), SC_(-0.0169407393250649919036351344471532182404925898980149027169321) } },
   { { SC_(0.0), SC_(8.0), SC_(0.223521489387566220527323400498620359274814930781423577578334) } },
   { { SC_(0.0), SC_(1e-05), SC_(-7.40316028370197013259676050746759072070960287586102867247159) } },
   { { SC_(0.0), SC_(1e-10), SC_(-14.7325162726972420426916696426209144888762342592762415255386) } },
   { { SC_(0.0), SC_(1e-20), SC_(-29.3912282502857968601858410375186700783698345615477536431464) } },
   { { SC_(0.0), SC_(1e+03), SC_(0.00471591797762281339977326146566525500985900489680197718528000) } },
   { { SC_(0.0), SC_(1e+05), SC_(0.00184676615886506410434074102431546125884886798090392516843524) } }
   } };
static const std::array<std::array<typename table_type<T>::type, 3>, 9> y1_data = { {
   { { SC_(1.0), SC_(1.0), SC_(-0.781212821300288716547150000047964820549906390716444607843833) } },
   { { SC_(1.0), SC_(2.0), SC_(-0.107032431540937546888370772277476636687480898235053860525795) } },
   { { SC_(1.0), SC_(4.0), SC_(0.397925710557100005253979972450791852271189181622908340876586) } },
   { { SC_(1.0), SC_(8.0), SC_(-0.158060461731247494255555266187483550327344049526705737651263) } },
   { { SC_(1.0), SC_(1e-10), SC_(-6.36619772367581343150789184284462611709080831190542841855708e9) } },
   { { SC_(1.0), SC_(1e-20), SC_(-6.36619772367581343075535053490057448139324059868649274367256e19) } },
   { { SC_(1.0), SC_(1e+01), SC_(0.249015424206953883923283474663222803260416543069658461246944) } },
   { { SC_(1.0), SC_(1e+03), SC_(-0.0247843312923517789148623560971412909386318548648705287583490) } },
   { { SC_(1.0), SC_(1e+05), SC_(0.00171921035008825630099494523539897102954509504993494957572726) } }
   } };
static const std::array<std::array<typename table_type<T>::type, 3>, 10> yn_data = { {
   { { SC_(2.0), SC_(1e-20), SC_(-1.27323954473516268615107010698011489627570899691226996904849e40) } },
   { { SC_(5.0), SC_(10.0), SC_(0.135403047689362303197029014762241709088405766746419538495983) } },
   { { SC_(-5.0), SC_(1e+06), SC_(0.000331052088322609048503535570014688967096938338061796192422114) } },
   { { SC_(10.0), SC_(10.0), SC_(-0.359814152183402722051986577343560609358382147846904467526222) } },
   { { SC_(10.0), SC_(1e-10), SC_(-1.18280490494334933900960937719565669877576135140014365217993e108) } },
   { { SC_(-10.0), SC_(1e+06), SC_(0.000725951969295187086245251366365393653610914686201194434805730) } },
   { { SC_(1e+02), SC_(5.0), SC_(-5.08486391602022287993091563093082035595081274976837280338134e115) } },
   { { SC_(1e+03), SC_(1e+05), SC_(0.00217254919137684037092834146629212647764581965821326561261181) } },
   { { SC_(-1e+03), SC_(7e+02), SC_(-1.88753109980945889960843803284345261796244752396992106755091e77) } },
   { { SC_(-25.0), SC_(8.0), SC_(3.45113613777297661997458045843868931827873456761831907587263e8) } }
   } };

int main()
{
#include "bessel_y01_data.ipp"
#include "bessel_yn_data.ipp"

   add_data(y0_data);
   add_data(y1_data);
   add_data(yn_data);
   add_data(bessel_y01_data);
   add_data(bessel_yn_data);

   unsigned data_total = data.size();

   screen_data([](const std::vector<double>& v){  return boost::math::cyl_neumann(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });


#if defined(TEST_C99) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return ::yn(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return std::tr1::cyl_neumann(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return gsl_sf_bessel_Yn(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return bessel_y(v[1], static_cast<int>(v[0]));  }, [](const std::vector<double>& v){ return v[2];  });
#endif

   unsigned data_used = data.size();
   std::string function = "cyl_neumann (integer order)[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "cyl_neumann (integer order)";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_neumann(static_cast<int>(v[0]), v[1]);  });
   std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_C99) || defined(TEST_LIBSTDCXX))
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name());
#endif
   report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name());
   //
   // Boost again, but with promotion to long double turned off:
   //
#if !defined(COMPILER_COMPARISON_TABLES)
   if(sizeof(long double) != sizeof(double))
   {
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_neumann(static_cast<int>(v[0]), v[1], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_C99) || defined(TEST_LIBSTDCXX))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_C99) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return ::yn(static_cast<int>(v[0]), v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "math.h");
#endif
#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::cyl_neumann(static_cast<int>(v[0]), v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_bessel_Yn(static_cast<int>(v[0]), v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return bessel_y(v[1], static_cast<int>(v[0]));  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "Rmath "  R_VERSION_STRING);
#endif

   return 0;
}

