//  (C) Copyright John Maddock 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// The purpose of this performance test is to probe the performance
// the of polynomial approximation to Elliptic integrals K and E.
//
// In the original paper:
// "Fast computation of complete elliptic integrals and Jacobian elliptic functions",
// Celestial Mechanics and Dynamical Astronomy, April 2012.
// they claim performance comparable to std::log, so we add comparison to log here
// as a reference "known good".
//
// We also measure the effect of disabling overflow errors, and of a "bare-bones"
// implementation which lacks a lot of our usual boilerplate.
//
// Note in the performance test routines, we had to sum the results of the function calls
// otherwise some msvc results were unrealistically fast, despite the use of
// benchmark::DoNotOptimize.
//
// Some sample results from msvc, and taking log(double) as a score of 1:
//
// ellint_1_performance<double>                     1.7
// ellint_1_performance_no_overflow_check<double>   1.6
// ellint_2_performance<double>                     1.8
// ellint_2_performance_no_overflow_check<double>   1.6
// basic_ellint_rational_performance<double>        1.6
//
// We can in fact get basic_ellint_rational_performance to much the same performance as log
// ONLY if we remove all error handling for cases with m > 0.9.  In particular the code appears
// to be ultra-sensitive to the presence of "if" statements which significantly hamper optimisation.
// 
// Performance with gcc-cygwin appears to be broadly similar.
//
#include <vector>
#include <benchmark/benchmark.h>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <boost/math/tools/random_vector.hpp>

using boost::math::generate_random_uniform_vector;

template <typename Real>
const std::vector<Real>& get_test_data()
{
   static const std::vector<Real> data = generate_random_uniform_vector<Real>(1024, 12345, 0.0, 0.9);
   return data;
}
template <typename Real>
Real& get_result_data()
{
   static Real data = 0;
   return data;
}


template <class T>
BOOST_FORCEINLINE T basic_ellint_rational(T k)
{
   using namespace boost::math::tools;

   static const char* function = "boost::math::ellint_k<%1%>(%1%)";

   T m = k * k;
   switch (static_cast<int>(m * 20))
   {
   case 0:
   case 1:
      //if (m < 0.1)
   {
      constexpr T coef[] =
      {
         1.591003453790792180,
         0.416000743991786912,
         0.245791514264103415,
         0.179481482914906162,
         0.144556057087555150,
         0.123200993312427711,
         0.108938811574293531,
         0.098853409871592910,
         0.091439629201749751,
         0.085842591595413900,
         0.081541118718303215,
         0.078199656811256481910
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.05);
   }
   case 2:
   case 3:
      //else if (m < 0.2)
   {
      constexpr T coef[] =
      {
         1.635256732264579992,
         0.471190626148732291,
         0.309728410831499587,
         0.252208311773135699,
         0.226725623219684650,
         0.215774446729585976,
         0.213108771877348910,
         0.216029124605188282,
         0.223255831633057896,
         0.234180501294209925,
         0.248557682972264071,
         0.266363809892617521
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.15);
   }
   case 4:
   case 5:
      //else if (m < 0.3)
   {
      constexpr T coef[] =
      {
         1.685750354812596043,
         0.541731848613280329,
         0.401524438390690257,
         0.369642473420889090,
         0.376060715354583645,
         0.405235887085125919,
         0.453294381753999079,
         0.520518947651184205,
         0.609426039204995055,
         0.724263522282908870,
         0.871013847709812357,
         1.057652872753547036
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.25);
   }
   case 6:
   case 7:
      //else if (m < 0.4)
   {
      constexpr T coef[] =
      {
         1.744350597225613243,
         0.634864275371935304,
         0.539842564164445538,
         0.571892705193787391,
         0.670295136265406100,
         0.832586590010977199,
         1.073857448247933265,
         1.422091460675497751,
         1.920387183402304829,
         2.632552548331654201,
         3.652109747319039160,
         5.115867135558865806,
         7.224080007363877411
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.35);
   }
   case 8:
   case 9:
      //else if (m < 0.5)
   {
      constexpr T coef[] =
      {
         1.813883936816982644,
         0.763163245700557246,
         0.761928605321595831,
         0.951074653668427927,
         1.315180671703161215,
         1.928560693477410941,
         2.937509342531378755,
         4.594894405442878062,
         7.330071221881720772,
         11.87151259742530180,
         19.45851374822937738,
         32.20638657246426863,
         53.73749198700554656,
         90.27388602940998849
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.45);
   }
   case 10:
   case 11:
      //else if (m < 0.6)
   {
      constexpr T coef[] =
      {
         1.898924910271553526,
         0.950521794618244435,
         1.151077589959015808,
         1.750239106986300540,
         2.952676812636875180,
         5.285800396121450889,
         9.832485716659979747,
         18.78714868327559562,
         36.61468615273698145,
         72.45292395127771801,
         145.1079577347069102,
         293.4786396308497026,
         598.3851815055010179,
         1228.420013075863451,
         2536.529755382764488
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.55);
   }
   case 12:
   case 13:
      //else if (m < 0.7)
   {
      constexpr T coef[] =
      {
         2.007598398424376302,
         1.248457231212347337,
         1.926234657076479729,
         3.751289640087587680,
         8.119944554932045802,
         18.66572130873555361,
         44.60392484291437063,
         109.5092054309498377,
         274.2779548232413480,
         697.5598008606326163,
         1795.716014500247129,
         4668.381716790389910,
         12235.76246813664335,
         32290.17809718320818,
         85713.07608195964685,
         228672.1890493117096,
         612757.2711915852774
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.65);
   }
   case 14:
   case 15:
      //else if (m < 0.8)
   {
      constexpr T coef[] =
      {
         2.156515647499643235,
         1.791805641849463243,
         3.826751287465713147,
         10.38672468363797208,
         31.40331405468070290,
         100.9237039498695416,
         337.3268282632272897,
         1158.707930567827917,
         4060.990742193632092,
         14454.00184034344795,
         52076.66107599404803,
         189493.6591462156887,
         695184.5762413896145,
         2567994.048255284686,
         9541921.966748386322,
         35634927.44218076174,
         133669298.4612040871,
         503352186.6866284541,
         1901975729.538660119,
         7208915015.330103756
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.75);
   }
   case 16:
      //else if (m < 0.85)
   {
      constexpr T coef[] =
      {
         2.318122621712510589,
         2.616920150291232841,
         7.897935075731355823,
         30.50239715446672327,
         131.4869365523528456,
         602.9847637356491617,
         2877.024617809972641,
         14110.51991915180325,
         70621.44088156540229,
         358977.2665825309926,
         1847238.263723971684,
         9600515.416049214109,
         50307677.08502366879,
         265444188.6527127967,
         1408862325.028702687,
         7515687935.373774627
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.825);
   }
   case 17:
      //else if (m < 0.90)
   {
      constexpr T coef[] =
      {
         2.473596173751343912,
         3.727624244118099310,
         15.60739303554930496,
         84.12850842805887747,
         506.9818197040613935,
         3252.277058145123644,
         21713.24241957434256,
         149037.0451890932766,
         1043999.331089990839,
         7427974.817042038995,
         53503839.67558661151,
         389249886.9948708474,
         2855288351.100810619,
         21090077038.76684053,
         156699833947.7902014,
         1170222242422.439893,
         8777948323668.937971,
         66101242752484.95041,
         499488053713388.7989,
         37859743397240299.20
      };
      return boost::math::tools::evaluate_polynomial(coef, m - 0.875);
   }
   case 18:
   case 19:
      return boost::math::detail::ellint_k_imp(k, boost::math::policies::policy<>(), std::integral_constant<int, 2>());
   default:
      if (m == 1)
      {
         return boost::math::policies::raise_overflow_error<T>(function, nullptr, boost::math::policies::policy<>());
      }
      else
         return boost::math::policies::raise_domain_error<T>(function,
            "Got k = %1%, function requires |k| <= 1", k, boost::math::policies::policy<>());
   }
}

template <typename Real>
void basic_ellint_rational_performance(benchmark::State& state)
{
    std::vector<Real>const& test_set = get_test_data<Real>();
    Real& r = get_result_data<Real>();

    for(auto _ : state)
    {
        for(unsigned i = 0; i < test_set.size(); ++i)
           benchmark::DoNotOptimize(r += basic_ellint_rational(test_set[i]));
    }
}

template <typename Real>
void basic_ellint_rational_performance_no_error_check(benchmark::State& state)
{
    std::vector<Real>const& test_set = get_test_data<Real>();
    Real& r = get_result_data<Real>();

    for(auto _ : state)
    {
        for(unsigned i = 0; i < test_set.size(); ++i)
           benchmark::DoNotOptimize(r += basic_ellint_rational_no_error_checks(test_set[i]));
    }
}

template <typename Real>
void ellint_1_performance(benchmark::State& state)
{
    std::vector<Real>const& test_set = get_test_data<Real>();
    Real& r = get_result_data<Real>();

    for(auto _ : state)
    {
        for(unsigned i = 0; i < test_set.size(); ++i)
           benchmark::DoNotOptimize(r += boost::math::ellint_1(test_set[i]));
    }
}

template <typename Real>
void ellint_1_performance_no_overflow_check(benchmark::State& state)
{
    std::vector<Real>const& test_set = get_test_data<Real>();
    Real& r = get_result_data<Real>();

    for(auto _ : state)
    {
        for(unsigned i = 0; i < test_set.size(); ++i)
           benchmark::DoNotOptimize(r += boost::math::ellint_1(test_set[i], boost::math::policies::make_policy(boost::math::policies::overflow_error<boost::math::policies::ignore_error>())));
    }
}

template <typename Real>
void ellint_2_performance(benchmark::State& state)
{
    std::vector<Real>const& test_set = get_test_data<Real>();
    Real& r = get_result_data<Real>();

    for(auto _ : state)
    {
        for(unsigned i = 0; i < test_set.size(); ++i)
           benchmark::DoNotOptimize(r += boost::math::ellint_2(test_set[i]));
    }
}

template <typename Real>
void ellint_2_performance_no_overflow_check(benchmark::State& state)
{
    std::vector<Real>const& test_set = get_test_data<Real>();
    Real& r = get_result_data<Real>();

    for(auto _ : state)
    {
        for(unsigned i = 0; i < test_set.size(); ++i)
           benchmark::DoNotOptimize(r += boost::math::ellint_2(test_set[i], boost::math::policies::make_policy(boost::math::policies::overflow_error<boost::math::policies::ignore_error>())));
    }
}

template <typename Real>
void log_performance(benchmark::State& state)
{
    std::vector<Real>const& test_set = get_test_data<Real>();
    Real& r = get_result_data<Real>();

    for(auto _ : state)
    {
        for(unsigned i = 0; i < test_set.size(); ++i)
           benchmark::DoNotOptimize(r += log(test_set[i]));
    }
}

BENCHMARK_TEMPLATE(ellint_1_performance, float);
BENCHMARK_TEMPLATE(ellint_1_performance, double);
BENCHMARK_TEMPLATE(ellint_1_performance, long double);
BENCHMARK_TEMPLATE(ellint_1_performance_no_overflow_check, float);
BENCHMARK_TEMPLATE(ellint_1_performance_no_overflow_check, double);
BENCHMARK_TEMPLATE(ellint_1_performance_no_overflow_check, long double);
BENCHMARK_TEMPLATE(ellint_2_performance, float);
BENCHMARK_TEMPLATE(ellint_2_performance, double);
BENCHMARK_TEMPLATE(ellint_2_performance, long double);
BENCHMARK_TEMPLATE(ellint_2_performance_no_overflow_check, float);
BENCHMARK_TEMPLATE(ellint_2_performance_no_overflow_check, double);
BENCHMARK_TEMPLATE(ellint_2_performance_no_overflow_check, long double);
BENCHMARK_TEMPLATE(log_performance, float);
BENCHMARK_TEMPLATE(log_performance, double);
BENCHMARK_TEMPLATE(log_performance, long double);
BENCHMARK_TEMPLATE(basic_ellint_rational_performance, float);
BENCHMARK_TEMPLATE(basic_ellint_rational_performance, double);
BENCHMARK_TEMPLATE(basic_ellint_rational_performance, long double);

BENCHMARK_MAIN();
