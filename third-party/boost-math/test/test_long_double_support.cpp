// Copyright John Maddock 2009

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <array>
#include <math.h>
#include <limits.h>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <iostream>
#include <iomanip>
   using std::cout;
   using std::endl;
   using std::setprecision;

#include <boost/array.hpp>
#include "functor.hpp"
#include "handle_test_result.hpp"
#include "table_type.hpp"

#include <boost/math/tools/config.hpp>

void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //
   const char* largest_type;
   largest_type = "(long\\s+)?double";

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*",                          // test data group
      ".*", 50, 20);                 // test function
}

template <class R, class Proc, class Arg>
struct simple_binder
{
   simple_binder(Proc p, Arg a) : m_proc(p), m_arg(a) {}
   R operator()(R arg) { return m_proc(arg, m_arg); }
   operator bool() { return m_proc ? true : false; }
private:
   Proc m_proc;
   Arg m_arg;
};

template <class A, class Proc>
void do_test_std_function(const A& data, const char* type_name, const char* function_name, const char* test_name, Proc proc, const char* inv_function_name = 0, Proc inv_proc = 0)
{
   // warning suppression:
   (void)data;
   (void)type_name;
   (void)test_name;
   typedef typename A::value_type row_type;
   typedef typename row_type::value_type value_type;

   boost::math::tools::test_result<value_type> result;

   //
   // test against data:
   //
   result = boost::math::tools::test(
      data,
      bind_func<value_type>(proc, 0),
      extract_result<value_type>(1));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, function_name, test_name);
   if(inv_proc)
   {
      result = boost::math::tools::test(
         data,
         bind_func<value_type>(inv_proc, 1),
         extract_result<value_type>(0));
      handle_test_result(result, data[result.worst()], result.worst(), type_name, inv_function_name, test_name);
   }
}

template <class A>
void do_test_std_function_2(const A& data, const char* type_name, const char* function_name, const char* test_name, long double(*proc)(long double, long double), const char* inv_function_name = 0, long double(*inv_proc)(long double, long double) = 0)
{
   // warning suppression:
   (void)data;
   (void)type_name;
   (void)test_name;
   typedef typename A::value_type row_type;
   typedef typename row_type::value_type value_type;

   boost::math::tools::test_result<value_type> result;

   //
   // test against data:
   //
   result = boost::math::tools::test(
      data,
      bind_func<value_type>(proc, 0, 1),
      extract_result<value_type>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, function_name, test_name);
   if(inv_proc)
   {
      result = boost::math::tools::test(
         data,
         bind_func<value_type>(inv_proc, 2, 1),
         extract_result<value_type>(0));
      handle_test_result(result, data[result.worst()], result.worst(), type_name, inv_function_name, test_name);
   }
}

long double std_inv_pow(long double a, long double b)
{
   return std::pow(a, 1 / b);
}
long double std_inv_powl(long double a, long double b)
{
   return ::powl(a, 1 / b);
}


void test_spots()
{
   // Basic sanity checks.
   // Test data taken from functions.wolfram.com
   long double (*unary_proc)(long double);
   long double (*inv_unary_proc)(long double);
   long double(*binary_proc)(long double, long double);
   long double(*inv_binary_proc)(long double, long double);
   //
   // COS:
   //
   std::array<std::array<long double, 2>, 4> cos_test_data = {{
      {{ 0, 1, }},
      {{ 0.125L, 0.992197667229329053149096907788250869543327304736601263468910L, }},
      {{ 1.125L, 0.431176516798666176551969042921689826840697850225767471037314L, }},
      {{ 1.75L, -0.178246055649492090382676943942631358969920851291548272886063L, }},
   }};
   unary_proc = std::cos;
   inv_unary_proc = std::acos;
   do_test_std_function(cos_test_data, "long double", "std::cos", "Mathematica data", unary_proc, "std::acos", inv_unary_proc);
   unary_proc = ::cosl;
   inv_unary_proc = ::acosl;
   do_test_std_function(cos_test_data, "long double", "::cosl", "Mathematica data", unary_proc, "::acosl", inv_unary_proc);
   //
   // SIN:
   //
   std::array<std::array<long double, 2>, 6> sin_test_data = {{
      {{ 0, 0, }},
      {{ 0.125L, 0.124674733385227689957442708712108467587834905641679257885515L, }},
      {{ -0.125L, -0.124674733385227689957442708712108467587834905641679257885515L, }},
      {{ 1.125L, 0.902267594099095162918416128654829100758989018716070814389152L, }},
#if LDBL_MAX_EXP > DBL_MAX_EXP
      {{ 1e-500L, 1e-500L, }},
      {{ 1e-1500L, 1e-1500L, }},
#else
      {{ 0, 0, }},
      {{ 0, 0, }},
#endif
   }};
   unary_proc = std::sin;
   inv_unary_proc = std::asin;
   do_test_std_function(sin_test_data, "long double", "std::sin", "Mathematica data", unary_proc, "std::asin", inv_unary_proc);
   unary_proc = ::sinl;
   inv_unary_proc = ::asinl;
   do_test_std_function(sin_test_data, "long double", "::sinl", "Mathematica data", unary_proc, "::asinl", inv_unary_proc);
   //
   // TAN:
   //
   std::array<std::array<long double, 2>, 6> tan_test_data = {{
      {{ 0, 0, }},
      {{ 0.125L, 0.125655136575130967792678218629774000758665763892225542668867L, }},
      {{ -0.125L, -0.125655136575130967792678218629774000758665763892225542668867L, }},
      {{ 1.125L, 2.09257127637217900442373398123488678225994171614872057291399L, }},
#if LDBL_MAX_EXP > DBL_MAX_EXP
      {{ 1e-500L, 1e-500L, }},
      {{ 1e-1500L, 1e-1500L, }},
#else
      {{ 0, 0, }},
      {{ 0, 0, }},
#endif
   }};
   unary_proc = std::tan;
   inv_unary_proc = std::atan;
   do_test_std_function(tan_test_data, "long double", "std::tan", "Mathematica data", unary_proc, "std::atan", inv_unary_proc);
   unary_proc = ::tanl;
   inv_unary_proc = ::atanl;
   do_test_std_function(tan_test_data, "long double", "::tanl", "Mathematica data", unary_proc, "::atanl", inv_unary_proc);
   //
   // EXP:
   //
   std::array<std::array<long double, 2>, 16> exp_test_data = {{
      {{ 0, 1, }},
      {{ 0.125L, 1.13314845306682631682900722781179387256550313174518162591282L, }},
      {{ -0.125L, 0.882496902584595402864892143229050736222004824990650741770309L, }},
      {{ 1.125L, 3.08021684891803124500466787877703957705899375982613074033239L, }},
      {{ 4.125L, 61.867809250367886509615357042290375913816094769196382159019381470L, }},
      {{ 23.125L, 1.10423089505572219567123524435791450127334161017155506895150e10L, }},
      {{ 230.125L, 8.75019331781009802320104206800466428512912084177754608785284e99L, }},
      {{ -230.125, 1.142831893741827617501612517500406182641114997360625464302e-100L, }},
      {{ -23.125, 9.0560769896728668382056793264648840991620764503768889249675e-11L, }},
      {{ -4.125, 0.0161634945881658751056030474082387107949421381136604400486095906L, }},
#if LDBL_MAX_EXP > DBL_MAX_EXP
      {{ 1151.5L, 1.2305404990401821581032984969433567556706865150466892644862e500L, }},
      {{ 2302.5, 9.1842687219959504902800771503994260058344776082873013828008e999L, }},
      {{ 11351.5, 7.830893628960601829810725204586653440595788354171367740730e4929L, }},
      {{ -11351.5, 1.27699346636729947157842192471351824733268709642976649973e-4930L, }},
      {{ -2302.5, 1.08881831561073624042773252180262334451975661825567070484e-1000L, }},
      {{ -1151.5, 8.126510267479997242743361503071571891855687212592001273854e-501L, }},
#else
      {{ 0, 1, }},
      {{ 0, 1, }},
      {{ 0, 1, }},
      {{ 0, 1, }},
      {{ 0, 1, }},
      {{ 0, 1, }},
#endif
   }};
   unary_proc = std::exp;
   inv_unary_proc = std::log;
   do_test_std_function(exp_test_data, "long double", "std::exp", "Mathematica data", unary_proc, "std::log", inv_unary_proc);
   unary_proc = ::expl;
   inv_unary_proc = ::logl;
   do_test_std_function(exp_test_data, "long double", "::expl", "Mathematica data", unary_proc, "::logl", inv_unary_proc);
   //
   // SQRT:
   //
   std::array<std::array<long double, 2>, 8> sqrt_test_data = {{
      {{ 1, 1, }},
      {{ 0.125L, 0.353553390593273762200422181052424519642417968844237018294170L, }},
      {{ 1.125L, 1.06066017177982128660126654315727355892725390653271105488251L, }},
      {{ 1e10L, 1e5L, }},
      {{ 1e100L, 1e50L, }},
#if LDBL_MAX_EXP > DBL_MAX_EXP
      {{ 1e500L, 1e250L, }},
      {{ 1e1000L, 1e500L, }},
      {{ 1e4930L, 1e2465L }},
#else
      {{ 1, 1, }},
      {{ 1, 1, }},
      {{ 1, 1, }},
#endif
   }};
   unary_proc = std::sqrt;
   inv_unary_proc = 0;
   do_test_std_function(sqrt_test_data, "long double", "std::sqrt", "Mathematica data", unary_proc, "", inv_unary_proc);
   unary_proc = ::sqrtl;
   do_test_std_function(sqrt_test_data, "long double", "::sqrtl", "Mathematica data", unary_proc, "", inv_unary_proc);
   //
   // POW:
   //
   std::array<std::array<long double, 3>, 40> pow_test_data = { {
      {{ 0.66666666666666666666666666666666667L, 10.5L, 0.014159299884333205600738476477156445L }}, {{ 0.40000000000000000000000000000000000L, 10.5L, 0.000066317769195774370528601435944941645L }}, {{ 0.28571428571428571428571428571428571L, 10.5L, 1.9376955162420093656912416026996235e-6L }}, {{ 0.22222222222222222222222222222222222L, 10.5L, 1.3844223610486906136767179806965939e-7L }}, {{ 0.18181818181818181818181818181818182L, 10.5L, 1.6834172004858801495594195366987548e-8L }}, {{ 0.15384615384615384615384615384615385L, 10.5L, 2.9134646649328940997378075111892189e-9L }}, {{ 0.13333333333333333333333333333333333L, 10.5L, 6.4842049648996264411393424327833257e-10L }}, {{ 0.11764705882352941176470588235294118L, 10.5L, 1.7422131202561313794251961087695707e-10L }}, {{ 0.10526315789473684210526315789473684L, 10.5L, 5.4187878014379968862785381990028068e-11L }}, {{ 0.095238095238095238095238095238095238L, 10.5L, 1.8945774321493250860060262734435905e-11L }}, {{ 0.086956521739130434782608695652173913L, 10.5L, 7.2890793204362496660639940633060965e-12L }}, {{ 0.080000000000000000000000000000000000L, 10.5L, 3.0370004999760496924513885300263083e-12L }}, {{ 0.074074074074074074074074074074074074L, 10.5L, 1.3536158492499429228631254734846878e-12L }}, {{ 0.068965517241379310344827586206896552L, 10.5L, 6.3919883760311690617057708564070401e-13L }}, {{ 0.064516129032258064516129032258064516L, 10.5L, 3.1733441149827305559623332316304054e-13L }}, {{ 0.060606060606060606060606060606060606L, 10.5L, 1.6459573809191842515197160159747426e-13L }}, {{ 0.057142857142857142857142857142857143L, 10.5L, 8.8736130949400182344750668262188042e-14L }}, {{ 0.054054054054054054054054054054054054L, 10.5L, 4.9510447505703867036532738058014010e-14L }}, {{ 0.051282051282051282051282051282051282L, 10.5L, 2.8486335222839832002073312496999392e-14L }}, {{ 0.048780487804878048780487804878048780L, 10.5L, 1.6849400716758403447644197639952420e-14L }},
      {{ 1.5000000000000000000000000000000000L, 10.5L, 70.624960850392521250220423499662198L }}, {{ 2.5000000000000000000000000000000000L, 10.5L, 15078.914929239174518579929086841195L }}, {{ 3.5000000000000000000000000000000000L, 10.5L, 516076.95410237226543172579761452715L }}, {{ 4.5000000000000000000000000000000000L, 10.5L, 7.2232291830544165404082939248515132e6L }}, {{ 5.5000000000000000000000000000000000L, 10.5L, 5.9402981014532387277725120300837238e7L }}, {{ 6.5000000000000000000000000000000000L, 10.5L, 3.4323395510377093368928553895198363e8L }}, {{ 7.5000000000000000000000000000000000L, 10.5L, 1.5422091149388577381859115936567576e9L }}, {{ 8.5000000000000000000000000000000000L, 10.5L, 5.7398259051853831024096483301864439e9L }}, {{ 9.5000000000000000000000000000000000L, 10.5L, 1.8454311861679240697429074147628415e10L }}, {{ 10.500000000000000000000000000000000L, 10.5L, 5.2782218505872232242456985697751839e10L }}, {{ 11.500000000000000000000000000000000L, 10.5L, 1.3719153764678064136017469971586606e11L }}, {{ 12.500000000000000000000000000000000L, 10.5L, 3.2927225399135962333569506281281311e11L }}, {{ 13.500000000000000000000000000000000L, 10.5L, 7.3876203544315301346698877141070794e11L }}, {{ 14.500000000000000000000000000000000L, 10.5L, 1.5644584144580486482323422555543492e12L }}, {{ 15.500000000000000000000000000000000L, 10.5L, 3.1512497975828317415327223169183800e12L }}, {{ 16.500000000000000000000000000000000L, 10.5L, 6.0754914531356236794145301221027031e12L }}, {{ 17.500000000000000000000000000000000L, 10.5L, 1.1269366708925227994916228749680770e13L }}, {{ 18.500000000000000000000000000000000L, 10.5L, 2.0197757248806823614685377151546893e13L }}, {{ 19.500000000000000000000000000000000L, 10.5L, 3.5104550732037231618759085175272827e13L }}, {{ 20.500000000000000000000000000000000L, 10.5L, 5.9349291812224551314681152356459565e13L }},
   } };
   binary_proc = std::pow;
   inv_binary_proc = std_inv_pow;
   do_test_std_function_2(pow_test_data, "long double", "std::pow", "Mathematica data", binary_proc, "std::pow", inv_binary_proc);
   binary_proc = ::powl;
   inv_binary_proc = std_inv_powl;
   do_test_std_function_2(pow_test_data, "long double", "::powl", "Mathematica data", binary_proc, "::pow", inv_binary_proc);
   //
   // LDEXP:
   //
   std::array<std::array<long double, 2>, 20> ld_data = { {
      {{ 0.66666666666666666666666666666666667L, 8.4510040015215293433113547025066667e29L }}, {{ 0.40000000000000000000000000000000000L, 5.0706024009129176059868128215040000e29L }}, {{ 0.28571428571428571428571428571428571L, 3.6218588577949411471334377296457143e29L }}, {{ 0.22222222222222222222222222222222222L, 2.8170013338405097811037849008355556e29L }}, {{ 0.18181818181818181818181818181818182L, 2.3048192731422352754485512825018182e29L }}, {{ 0.15384615384615384615384615384615385L, 1.9502316926588144638410818544246154e29L }}, {{ 0.13333333333333333333333333333333333L, 1.6902008003043058686622709405013333e29L }}, {{ 0.11764705882352941176470588235294118L, 1.4913536473273287076431802416188235e29L }}, {{ 0.10526315789473684210526315789473684L, 1.3343690528718204226281086372378947e29L }}, {{ 0.095238095238095238095238095238095238L, 1.2072862859316470490444792432152381e29L }}, {{ 0.086956521739130434782608695652173913L, 1.1023048697636777404319158307617391e29L }}, {{ 0.080000000000000000000000000000000000L, 1.0141204801825835211973625643008000e29L }}, {{ 0.074074074074074074074074074074074074L, 9.3900044461350326036792830027851852e28L }}, {{ 0.068965517241379310344827586206896552L, 8.7424179326084786310117462439724138e28L }}, {{ 0.064516129032258064516129032258064516L, 8.1783909692143832354626013250064516e28L }}, {{ 0.060606060606060606060606060606060606L, 7.6827309104741175848285042750060606e28L }}, {{ 0.057142857142857142857142857142857143L, 7.2437177155898822942668754592914286e28L }}, {{ 0.054054054054054054054054054054054054L, 6.8521654066390778459281254344648649e28L }}, {{ 0.051282051282051282051282051282051282L, 6.5007723088627148794702728480820513e28L }}, {{ 0.048780487804878048780487804878048780L, 6.1836614645279482999839180750048780e28L }}
   }};
   using namespace boost;
   long double(*pld)(long double, int) = std::ldexp;
   do_test_std_function(ld_data, "long double", "std::ldexp", "Mathematica data", simple_binder<long double, long double(*)(long double, int), int>(pld, 100), "std::ldexp", simple_binder<long double, long double(*)(long double, int), int>(pld, -100));
   pld = ::ldexpl;
   do_test_std_function(ld_data, "long double", "::ldexp", "Mathematica data", simple_binder<long double, long double(*)(long double, int), int>(pld, 100), "::ldexp", simple_binder<long double, long double(*)(long double, int), int>(pld, -100));
   //
   // Sinh:
   //
   std::array<std::array<long double, 2>, 20> sinh_data = { {
      {{ 1.0L, 1.1752011936438014568823818505956008L }}, {{ 2.0L, 3.6268604078470187676682139828012617L }}, {{ 3.0L, 10.017874927409901898974593619465828L }}, {{ 4.0L, 27.289917197127752448908271590793819L }}, {{ 5.0L, 74.203210577788758977009471996064566L }}, {{ 6.0L, 201.71315737027922812498206768797873L }}, {{ 7.0L, 548.31612327324652237375611757601851L }}, {{ 8.0L, 1490.4788257895501861158766390318814L }}, {{ 9.0L, 4051.5419020827899605152235958980346L }}, {{ 10.L, 11013.232874703393377236524554846364L }}, {{ 11.L, 29937.070849248058832540413239811132L }}, {{ 12.L, 81377.395706429854227338497569902237L }}, {{ 13.L, 221206.69600333008695956086081165150L }}, {{ 14.L, 601302.14208197262451506660144189773L }}, {{ 15.L, 1.6345086862359023684906766101516749e6L }}, {{ 16.L, 4.4430552602538800507941522408334683e6L }}, {{ 17.L, 1.2077476376787628407699123664359614e7L }}, {{ 18.L, 3.2829984568665247954403379273215799e7L }}, {{ 19.L, 8.9241150481593627621056798125727582e7L }}, {{ 20.L, 2.4258259770489513795397660405149137e8L }},
   } };
   unary_proc = &std::sinh;
#if __cplusplus >= 201103
   inv_unary_proc = &std::asinh;
#else
   inv_unary_proc = 0;
#endif
   do_test_std_function(sinh_data, "long double", "std::sinh", "Mathematica data", unary_proc, "std::asinh", inv_unary_proc);
   unary_proc = ::sinhl;
#if __cplusplus >= 201103
   inv_unary_proc = ::asinhl;
#endif
   do_test_std_function(sinh_data, "long double", "::sinhl", "Mathematica data", unary_proc, "::asinhl", inv_unary_proc);
   //
   // Cosh:
   //
   std::array<std::array<long double, 2L>, 20L> cosh_data = { {
      {{ 1.0L, 1.5430806348152437784779056207570617L }}, {{ 2.0L, 3.7621956910836314595622134777737461L }}, {{ 3.0L, 10.067661995777765841953936035115890L }}, {{ 4.0L, 27.308232836016486629201989612067060L }}, {{ 5.0L, 74.209948524787844444106108044487714L }}, {{ 6.0L, 201.71563612245589448340511285540955L }}, {{ 7.0L, 548.31703515521207688996412071210292L }}, {{ 8.0L, 1490.4791612521780886277154604210072L }}, {{ 9.0L, 4051.5420254925940471947730935347253L }}, {{ 10.L, 11013.232920103323139721376090437880L }}, {{ 11.L, 29937.070865949759622786072552446649L }}, {{ 12.L, 81377.395712574066580666707328584546L }}, {{ 13.L, 221206.69600559041636654191513743678L }}, {{ 14.L, 601302.14208280415323417016932596172L }}, {{ 15.L, 1.6345086862362082708111784359400464e6L }}, {{ 16.L, 4.4430552602539925859688714999479821e6L }}, {{ 17.L, 1.2077476376787669807076311516026210e7L }}, {{ 18.L, 3.2829984568665263184383123985844235e7L }}, {{ 19.L, 8.9241150481593633223853235662995122e7L }}, {{ 20.L, 2.4258259770489514001513022649004919e8L }},
   } };
   unary_proc = &std::cosh;
#if __cplusplus >= 201103
   inv_unary_proc = &std::acosh;
#endif
   do_test_std_function(cosh_data, "long double", "std::cosh", "Mathematica data", unary_proc, "std::acosh", inv_unary_proc);
   unary_proc = ::coshl;
#if __cplusplus >= 201103
   inv_unary_proc = ::acoshl;
#endif
   do_test_std_function(cosh_data, "long double", "::cosh", "Mathematica data", unary_proc, "::acosh", inv_unary_proc);
   //
   // Tanh:
   //
   std::array<std::array<long double, 2L>, 20L> tanh_data = { {
      {{ 2.0000000000000000000000000000000000L, 0.96402758007581688394641372410092315L }}, {{ 1.0000000000000000000000000000000000L, 0.76159415595576488811945828260479359L }}, {{ 0.66666666666666666666666666666666667L, 0.58278294534791012006763998724863620L }}, {{ 0.50000000000000000000000000000000000L, 0.46211715726000975850231848364367255L }}, {{ 0.40000000000000000000000000000000000L, 0.37994896225522488526774812389687331L }}, {{ 0.33333333333333333333333333333333333L, 0.32151273753163434471940622242520647L }}, {{ 0.28571428571428571428571428571428571L, 0.27818549032570244047180008724146611L }}, {{ 0.25000000000000000000000000000000000L, 0.24491866240370912927780113149101696L }}, {{ 0.22222222222222222222222222222222222L, 0.21863508368712133408473136585229335L }}, {{ 0.20000000000000000000000000000000000L, 0.19737532022490400073815731881101567L }}, {{ 0.18181818181818181818181818181818182L, 0.17984081852510791219962261245781186L }}, {{ 0.16666666666666666666666666666666667L, 0.16514041292462935373278922792245912L }}, {{ 0.15384615384615384615384615384615385L, 0.15264375981490485028417961210708858L }}, {{ 0.14285714285714285714285714285714286L, 0.14189319376693254602300070386766884L }}, {{ 0.13333333333333333333333333333333333L, 0.13254878839087838732054090217452509L }}, {{ 0.12500000000000000000000000000000000L, 0.12435300177159620805464727580589271L }}, {{ 0.11764705882352941176470588235294118L, 0.11710726941545656349019174731833940L }}, {{ 0.11111111111111111111111111111111111L, 0.11065611052473799138171921474515056L }}, {{ 0.10526315789473684210526315789473684L, 0.10487608974842188468874887218357955L }}, {{ 0.10000000000000000000000000000000000L, 0.099667994624955817118305083678352184L }}
   } };
   unary_proc = &std::tanh;
#if __cplusplus >= 201103
   inv_unary_proc = &std::atanh;
#endif
   do_test_std_function(tanh_data, "long double", "std::tanh", "Mathematica data", unary_proc, "std::atanh", inv_unary_proc);
   unary_proc = ::tanhl;
#if __cplusplus >= 201103
   inv_unary_proc = ::atanhl;
#endif
   do_test_std_function(tanh_data, "long double", "::tanh", "Mathematica data", unary_proc, "::atanh", inv_unary_proc);
   //
   // ABS, FABS:
   //
   std::array<std::array<long double, 2L>, 20L> abs_data = { {
      {{ -2.0000000000000000000000000000000000L, 2.0000000000000000000000000000000000L }}, {{ -1.0000000000000000000000000000000000L, 1.0000000000000000000000000000000000L }}, {{ -0.66666666666666666666666666666666667L, 0.66666666666666666666666666666666667L }}, {{ -0.50000000000000000000000000000000000L, 0.50000000000000000000000000000000000L }}, {{ -0.40000000000000000000000000000000000L, 0.40000000000000000000000000000000000L }}, {{ -0.33333333333333333333333333333333333L, 0.33333333333333333333333333333333333L }}, {{ -0.28571428571428571428571428571428571L, 0.28571428571428571428571428571428571L }}, {{ -0.25000000000000000000000000000000000L, 0.25000000000000000000000000000000000L }}, {{ -0.22222222222222222222222222222222222L, 0.22222222222222222222222222222222222L }}, {{ -0.20000000000000000000000000000000000L, 0.20000000000000000000000000000000000L }}, {{ -0.18181818181818181818181818181818182L, 0.18181818181818181818181818181818182L }}, {{ -0.16666666666666666666666666666666667L, 0.16666666666666666666666666666666667L }}, {{ -0.15384615384615384615384615384615385L, 0.15384615384615384615384615384615385L }}, {{ -0.14285714285714285714285714285714286L, 0.14285714285714285714285714285714286L }}, {{ -0.13333333333333333333333333333333333L, 0.13333333333333333333333333333333333L }}, {{ -0.12500000000000000000000000000000000L, 0.12500000000000000000000000000000000L }}, {{ -0.11764705882352941176470588235294118L, 0.11764705882352941176470588235294118L }}, {{ -0.11111111111111111111111111111111111L, 0.11111111111111111111111111111111111L }}, {{ -0.10526315789473684210526315789473684L, 0.10526315789473684210526315789473684L }}, {{ -0.10000000000000000000000000000000000L, 0.10000000000000000000000000000000000L }}
   } };
   unary_proc = &std::abs;
   inv_unary_proc = 0;
   do_test_std_function(abs_data, "long double", "std::abs", "Mathematica data", unary_proc, "std::abs", inv_unary_proc);
   unary_proc = &std::fabs;
   inv_unary_proc = 0;
   do_test_std_function(abs_data, "long double", "std::fabs", "Mathematica data", unary_proc, "std::abs", inv_unary_proc);
   unary_proc = ::fabsl;
   do_test_std_function(abs_data, "long double", "::fabs", "Mathematica data", unary_proc, "::abs", inv_unary_proc);
}


BOOST_AUTO_TEST_CASE( test_main )
{
   expected_results();
   std::cout << "Running tests with BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS "
#ifdef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
      "defined."
#else
      "not defined."
#endif
      << std::endl;
   // Basic sanity-check spot values.
   // (Parameter value, arbitrarily zero, only communicates the floating point type).
   test_spots(); // Test long double.


} // BOOST_AUTO_TEST_CASE( test_main )

