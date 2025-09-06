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
static const std::array<std::array<typename table_type<T>::type, 3>, 8> j0_data = { {
   { { SC_(0.0), SC_(0.0), SC_(1.0) } },
   { { SC_(0.0), SC_(1.0), SC_(0.7651976865579665514497175261026632209093) } },
   { { SC_(0.0), SC_(-2.0), SC_(0.2238907791412356680518274546499486258252) } },
   { { SC_(0.0), SC_(4.0), SC_(-0.3971498098638473722865907684516980419756) } },
   { { SC_(0.0), SC_(-8.0), SC_(0.1716508071375539060908694078519720010684) } },
   { { SC_(0.0), SC_(1e-05), SC_(0.999999999975000000000156249999999565972) } },
   { { SC_(0.0), SC_(1e-10), SC_(0.999999999999999999997500000000000000000) } },
   { { SC_(0.0), SC_(-1e+01), SC_(-0.2459357644513483351977608624853287538296) } },
   } };
static const std::array<std::array<T, 3>, 6> j0_tricky = { {
      // Big numbers make the accuracy of std::sin the limiting factor:
   { { SC_(0.0), SC_(1e+03), SC_(0.02478668615242017456133073111569370878617) } },
   { { SC_(0.0), SC_(1e+05), SC_(-0.001719201116235972192570601477073201747532) } },
   // test at the roots:
   { { SC_(0.0), T(2521642.0) / (1024 * 1024), SC_(1.80208819970046790002973759410972422387259992955354630042138e-7) } },
   { { SC_(0.0), T(5788221.0) / (1024 * 1024), SC_(-1.37774249380686777043369399806210229535671843632174587432454e-7) } },
   { { SC_(0.0), T(9074091.0) / (1024 * 1024), SC_(1.03553057441100845081018471279571355857520645127532785991335e-7) } },
   { { SC_(0.0), T(12364320.0) / (1024 * 1024), SC_(-3.53017140778223781420794006033810387155048392363051866610931e-9) } }
   } };

static const std::array<std::array<typename table_type<T>::type, 3>, 8> j1_data = { {
   { { SC_(1.0), SC_(0.0), SC_(0.0) } },
   { { SC_(1.0), SC_(1.0), SC_(0.4400505857449335159596822037189149131274) } },
   { { SC_(1.0), SC_(-2.0), SC_(-0.5767248077568733872024482422691370869203) } },
   { { SC_(1.0), SC_(4.0), SC_(-6.604332802354913614318542080327502872742e-02) } },
   { { SC_(1.0), SC_(-8.0), SC_(-0.2346363468539146243812766515904546115488) } },
   { { SC_(1.0), SC_(1e-05), SC_(4.999999999937500000000260416666666124132e-06) } },
   { { SC_(1.0), SC_(1e-10), SC_(4.999999999999999999993750000000000000000e-11) } },
   { { SC_(1.0), SC_(-1e+01), SC_(-4.347274616886143666974876802585928830627e-02) } },
   } };
static const std::array<std::array<T, 3>, 5> j1_tricky = { {
      // Big numbers make the accuracy of std::sin the limiting factor:
   { { SC_(1.0), SC_(1e+03), SC_(4.728311907089523917576071901216916285418e-03) } },
   { { SC_(1.0), SC_(1e+05), SC_(1.846757562882567716362123967114215743694e-03) } },
   // test zeros:
   { { SC_(1.0), T(4017834) / (1024 * 1024), SC_(3.53149033321258645807835062770856949751958513973522222203044e-7) } },
   { { SC_(1.0), T(7356375) / (1024 * 1024), SC_(-2.31227973111067286051984021150135526024117175836722748404342e-7) } },
   { { SC_(1.0), T(10667654) / (1024 * 1024), SC_(1.24591331097191900488116495350277530373473085499043086981229e-7) } },
   } };

static const std::array<std::array<typename table_type<T>::type, 3>, 17> jn_data = { {
      // This first one is a modified test case from https://svn.boost.org/trac/boost/ticket/2733
   { { SC_(-1.0), SC_(1.25), SC_(-0.510623260319880467069474837274910375352924050139633057168856) } },
   { { SC_(2.0), SC_(0.0), SC_(0.0) } },
   { { SC_(-2.0), SC_(0.0), SC_(0.0) } },
   { { SC_(2.0), SC_(1e-02), SC_(1.249989583365885362413250958437642113452e-05) } },
   { { SC_(5.0), SC_(10.0), SC_(-0.2340615281867936404436949416457777864635) } },
   { { SC_(5.0), SC_(-10.0), SC_(0.2340615281867936404436949416457777864635) } },
   { { SC_(-5.0), SC_(1e+06), SC_(7.259643842453285052375779970433848914846e-04) } },
   { { SC_(5.0), SC_(1e+06), SC_(-0.000725964384245328505237577997043384891484649290328285235308619) } },
   { { SC_(-5.0), SC_(-1.0), SC_(2.497577302112344313750655409880451981584e-04) } },
   { { SC_(10.0), SC_(10.0), SC_(0.2074861066333588576972787235187534280327) } },
   { { SC_(10.0), SC_(-10.0), SC_(0.2074861066333588576972787235187534280327) } },
   { { SC_(10.0), SC_(-5.0), SC_(1.467802647310474131107532232606627020895e-03) } },
   { { SC_(-10.0), SC_(1e+06), SC_(-3.310793117604488741264958559035744460210e-04) } },
   { { SC_(10.0), SC_(1e+06), SC_(-0.000331079311760448874126495855903574446020957243277028930713243) } },
   { { SC_(1e+02), SC_(8e+01), SC_(4.606553064823477354141298259169874909670e-06) } },
   { { SC_(1e+03), SC_(1e+05), SC_(1.283178112502480365195139312635384057363e-03) } },
   { { SC_(10.0), SC_(1e-100), SC_(2.69114445546737213403880070546737213403880070546737213403880e-1010) } },
   } };


int main()
{
#include "bessel_j_int_data.ipp"

   add_data(j0_data);
   add_data(j0_tricky);
   add_data(j1_data);
   add_data(j1_tricky);
   add_data(jn_data);
   add_data(bessel_j_int_data);

   unsigned data_total = data.size();

   screen_data([](const std::vector<double>& v){  return boost::math::cyl_bessel_j(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });


#if defined(TEST_C99) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return ::jn(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return std::tr1::cyl_bessel_j(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return gsl_sf_bessel_Jn(static_cast<int>(v[0]), v[1]);  }, [](const std::vector<double>& v){ return v[2];  });
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return bessel_j(v[1], static_cast<int>(v[0]));  }, [](const std::vector<double>& v){ return v[2];  });
#endif

   unsigned data_used = data.size();
   std::string function = "cyl_bessel_j (integer order)[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "cyl_bessel_j (integer order)";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_bessel_j(static_cast<int>(v[0]), v[1]);  });
   std::cout << time << std::endl;
#if defined(COMPILER_COMPARISON_TABLES)
   report_execution_time(time, std::string("Compiler Option Comparison on ") + platform_name(), "boost::math::cyl_bessel_j (integer orders)", get_compiler_options_name());
#else
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_C99) || defined(TEST_LIBSTDCXX))
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name());
#endif
   report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name());
#endif
   //
   // Boost again, but with promotion to long double turned off:
   //
#if !defined(COMPILER_COMPARISON_TABLES)
   if(sizeof(long double) != sizeof(double))
   {
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::cyl_bessel_j(static_cast<int>(v[0]), v[1], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_C99) || defined(TEST_LIBSTDCXX))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_C99) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return ::jn(static_cast<int>(v[0]), v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "math.h");
#endif
#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::cyl_bessel_j(static_cast<int>(v[0]), v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_bessel_Jn(static_cast<int>(v[0]), v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return bessel_j(v[1], static_cast<int>(v[0]));  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "Rmath "  R_VERSION_STRING);
#endif


   return 0;
}

