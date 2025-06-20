//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include "test_autodiff.hpp"
#include <boost/utility/identity_type.hpp>
#include <boost/math/tools/test_value.hpp>

BOOST_AUTO_TEST_SUITE(test_autodiff_3)

BOOST_AUTO_TEST_CASE_TEMPLATE(atanh_test, T, all_float_types) {
  using boost::math::atanh;
  const T eps = 3000 * test_constants_t<T>::pct_epsilon(); // percent
  constexpr unsigned m = 5;
  const T cx = T(0.5);
  auto x = make_fvar<T, m>(cx);
  auto y = atanh(x);
  // BOOST_CHECK_EQUAL(y.derivative(0) , atanh(cx)); // fails due to overload
  BOOST_CHECK_CLOSE(y.derivative(0u), atanh(static_cast<T>(x)), eps);
  BOOST_CHECK_CLOSE(y.derivative(1u), static_cast<T>(4) / 3, eps);
  BOOST_CHECK_CLOSE(y.derivative(2u), static_cast<T>(16) / 9, eps);
  BOOST_CHECK_CLOSE(y.derivative(3u), static_cast<T>(224) / 27, eps);
  BOOST_CHECK_CLOSE(y.derivative(4u), static_cast<T>(1280) / 27, eps);
  BOOST_CHECK_CLOSE(y.derivative(5u), static_cast<T>(31232) / 81, eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atan_test, T, all_float_types) {
  BOOST_MATH_STD_USING
  using namespace boost;

  const T cx = 1;
  constexpr unsigned m = 5;
  const auto x = make_fvar<T, m>(cx);
  auto y = atan(x);
  const auto eps = boost::math::tools::epsilon<T>() * 200; // 2eps as a percentage
  BOOST_CHECK_CLOSE(y.derivative(0u), boost::math::constants::pi<T>() / 4, eps);
  BOOST_CHECK_CLOSE(y.derivative(1u), T(0.5), eps);
  BOOST_CHECK_CLOSE(y.derivative(2u), T(-0.5), eps);
  BOOST_CHECK_CLOSE(y.derivative(3u), T(0.5), eps);
  BOOST_CHECK_CLOSE(y.derivative(4u), T(0), eps);
  BOOST_CHECK_CLOSE(y.derivative(5u), T(-3), eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(erf_test, T, all_float_types) {
  BOOST_MATH_STD_USING
  using namespace boost;
  using boost::math::erf;

  const T eps = 300 * 100 * boost::math::tools::epsilon<T>(); // percent
  const T cx = 1;
  constexpr unsigned m = 5;
  const auto x = make_fvar<T, m>(cx);
  auto y = erf(x);
  BOOST_CHECK_CLOSE(y.derivative(0u), erf(static_cast<T>(x)), eps);
  BOOST_CHECK_CLOSE(
      y.derivative(1u),
      T(2) / (math::constants::e<T>() * math::constants::root_pi<T>()), eps);
  BOOST_CHECK_CLOSE(
      y.derivative(2u),
      T(-4) / (math::constants::e<T>() * math::constants::root_pi<T>()), eps);
  BOOST_CHECK_CLOSE(
      y.derivative(3u),
      T(4) / (math::constants::e<T>() * math::constants::root_pi<T>()), eps);
  BOOST_CHECK_CLOSE(
      y.derivative(4u),
      T(8) / (math::constants::e<T>() * math::constants::root_pi<T>()), eps);
  BOOST_CHECK_CLOSE(
      y.derivative(5u),
      T(-40) / (math::constants::e<T>() * math::constants::root_pi<T>()), eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sinc_test, T, bin_float_types) {
  BOOST_MATH_STD_USING
  const T eps = 20000 * boost::math::tools::epsilon<T>(); // percent
  const T cx = 1;
  constexpr unsigned m = 5;
  auto x = make_fvar<T, m>(cx);
  auto y = sinc(x);
  BOOST_CHECK_CLOSE(y.derivative(0u), sin(cx), eps);
  BOOST_CHECK_CLOSE(y.derivative(1u), cos(cx) - sin(cx), eps);
  BOOST_CHECK_CLOSE(y.derivative(2u), sin(cx) - 2 * cos(cx), eps);
  BOOST_CHECK_CLOSE(y.derivative(3u), T(5) * cos(cx) - T(3) * sin(cx), eps);
  BOOST_CHECK_CLOSE(y.derivative(4u), T(13) * sin(cx) - T(20) * cos(cx), eps);
  BOOST_CHECK_CLOSE(y.derivative(5u), T(101) * cos(cx) - T(65) * sin(cx), eps);
  // Test at x = 0
  auto y2 = sinc(make_fvar<T, 10>(0));
  BOOST_CHECK_CLOSE(y2.derivative(0u), T(1), eps);
  BOOST_CHECK_CLOSE(y2.derivative(1u), T(0), eps);
  BOOST_CHECK_CLOSE(y2.derivative(2u), -cx / T(3), eps);
  BOOST_CHECK_CLOSE(y2.derivative(3u), T(0), eps);
  BOOST_CHECK_CLOSE(y2.derivative(4u), cx / T(5), eps);
  BOOST_CHECK_CLOSE(y2.derivative(5u), T(0), eps);
  BOOST_CHECK_CLOSE(y2.derivative(6u), -cx / T(7), eps);
  BOOST_CHECK_CLOSE(y2.derivative(7u), T(0), eps);
  BOOST_CHECK_CLOSE(y2.derivative(8u), cx / T(9), eps);
  BOOST_CHECK_CLOSE(y2.derivative(9u), T(0), eps);
  BOOST_CHECK_CLOSE(y2.derivative(10u), -cx / T(11), eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sinh_and_cosh, T, bin_float_types) {
  BOOST_MATH_STD_USING
  const T eps = 300 * boost::math::tools::epsilon<T>(); // percent
  const T cx = 1;
  constexpr unsigned m = 5;
  auto x = make_fvar<T, m>(cx);
  auto s = sinh(x);
  auto c = cosh(x);
  BOOST_CHECK_CLOSE(s.derivative(0u), sinh(static_cast<T>(x)), eps);
  BOOST_CHECK_CLOSE(c.derivative(0u), cosh(static_cast<T>(x)), eps);
  for (auto i : boost::irange(m + 1)) {
    BOOST_CHECK_CLOSE(s.derivative(i), static_cast<T>(i % 2 == 1 ? c : s), eps);
    BOOST_CHECK_CLOSE(c.derivative(i), static_cast<T>(i % 2 == 1 ? s : c), eps);
  }
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
BOOST_AUTO_TEST_CASE_TEMPLATE(tanh_test, T, all_float_types) {
  using bmp::fabs;
  using bmp::tanh;
  using detail::fabs;
  using detail::tanh;
  using std::fabs;
  using std::tanh;
  std::array<T, 6> tanh_derivatives{
      {BOOST_MATH_TEST_VALUE(T, 0.76159415595576488811945828260479359041276859725793655159681050012195324457663848345894752167367671442190275970155),
       BOOST_MATH_TEST_VALUE(T, 0.4199743416140260693944967390417014449171867282307709547133114402445898995240483056156940088623187260),
       BOOST_MATH_TEST_VALUE(T, -0.6397000084492245001884917693038439532192113630607991449429985631870206934885434644440069533372017992),
       BOOST_MATH_TEST_VALUE(T, 0.6216266807712962631065304287222233996757241175544541856396870633581620622188951465548376863495698762),
       BOOST_MATH_TEST_VALUE(T, 0.6650910447505016777350714809210623499275713283320312544881492938309646347626843278089998045994094537),
       BOOST_MATH_TEST_VALUE(T, -5.556893558473719797604582902316972009873833721162934560195313423947089897942786231796317250984197038)}};
  const T cx = 1;
  constexpr std::size_t m = 5;
  auto x = make_fvar<T, m>(cx);
  auto t = tanh(x);
  for (auto i : boost::irange(tanh_derivatives.size())) {
    BOOST_TEST_WARN(isNearZero(t.derivative(i) - tanh_derivatives[i]));
  }
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(tan_test, T, bin_float_types) {
  BOOST_MATH_STD_USING
  const T eps = 800 * boost::math::tools::epsilon<T>(); // percent
  const T cx = boost::math::constants::third_pi<T>();
  const T root_three = boost::math::constants::root_three<T>();
  constexpr unsigned m = 5;
  const auto x = make_fvar<T, m>(cx);
  auto y = tan(x);
  BOOST_CHECK_CLOSE(y.derivative(0u), root_three, eps);
  BOOST_CHECK_CLOSE(y.derivative(1u), T(4), eps);
  BOOST_CHECK_CLOSE(y.derivative(2u), T(8) * root_three, eps);
  BOOST_CHECK_CLOSE(y.derivative(3u), T(80), eps);
  BOOST_CHECK_CLOSE(y.derivative(4u), T(352) * root_three, eps);
  BOOST_CHECK_CLOSE(y.derivative(5u), T(5824), eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fmod_test, T, bin_float_types) {
  BOOST_MATH_STD_USING
  constexpr unsigned m = 3;
  const T cx = T(3.25);
  const T cy = T(0.5);
  auto x = make_fvar<T, m>(cx);
  auto y = fmod(x, autodiff_fvar<T, m>(cy));
  BOOST_CHECK_EQUAL(y.derivative(0u), T(0.25));
  BOOST_CHECK_EQUAL(y.derivative(1u), T(1));
  BOOST_CHECK_EQUAL(y.derivative(2u), T(0));
  BOOST_CHECK_EQUAL(y.derivative(3u), T(0));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(round_and_trunc, T, all_float_types) {
  BOOST_MATH_STD_USING
  constexpr unsigned m = 3;
  const T cx = T(3.25);
  auto x = make_fvar<T, m>(cx);
  auto y = round(x);
  BOOST_CHECK_EQUAL(y.derivative(0u), round(cx));
  BOOST_CHECK_EQUAL(y.derivative(1u), T(0));
  BOOST_CHECK_EQUAL(y.derivative(2u), T(0));
  BOOST_CHECK_EQUAL(y.derivative(3u), T(0));
  y = trunc(x);
  BOOST_CHECK_EQUAL(y.derivative(0u), trunc(cx));
  BOOST_CHECK_EQUAL(y.derivative(1u), T(0));
  BOOST_CHECK_EQUAL(y.derivative(2u), T(0));
  BOOST_CHECK_EQUAL(y.derivative(3u), T(0));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(iround_and_itrunc, T, all_float_types) {
  BOOST_MATH_STD_USING
  using namespace boost::math;
  constexpr unsigned m = 3;
  const T cx = T(3.25);
  auto x = make_fvar<T, m>(cx);
  int y = iround(x);
  BOOST_CHECK_EQUAL(y, iround(cx));
  y = itrunc(x);
  BOOST_CHECK_EQUAL(y, itrunc(cx));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(lambert_w0_test, T, all_float_types) {
  const T eps = 1000 * boost::math::tools::epsilon<T>(); // percent
  constexpr unsigned m = 10;
  const T cx = 3;
  // Mathematica: N[Table[D[ProductLog[x], {x, n}], {n, 0, 10}] /. x -> 3, 52]
  std::array<T, m + 1> answers{
      {BOOST_MATH_TEST_VALUE(T, 1.049908894964039959988697070552897904589466943706341),
       BOOST_MATH_TEST_VALUE(T, 0.1707244807388472968312949774415522047470762509741737),
       BOOST_MATH_TEST_VALUE(T, -0.04336545501146252734105411312976167858858970875797718),
       BOOST_MATH_TEST_VALUE(T, 0.02321456264324789334313200360870492961288748451791104),
       BOOST_MATH_TEST_VALUE(T, -0.01909049778427783072663170526188353869136655225133878),
       BOOST_MATH_TEST_VALUE(T, 0.02122935002563637629500975949987796094687564718834156),
       BOOST_MATH_TEST_VALUE(T, -0.02979093848448877259041971538394953658978044986784643),
       BOOST_MATH_TEST_VALUE(T, 0.05051290266216717699803334605370337985567016837482099),
       BOOST_MATH_TEST_VALUE(T, -0.1004503154972645060971099914384090562800544486549660),
       BOOST_MATH_TEST_VALUE(T, 0.2292464437392250211967939182075930820454464472006425),
       BOOST_MATH_TEST_VALUE(T, -0.5905839053125614593682763387470654123192290838719517)}};
  auto x = make_fvar<T, m>(cx);
  auto y = lambert_w0(x);
  for (auto i : boost::irange(m + 1)) {
    const T answer = answers[i];
    BOOST_CHECK_CLOSE(y.derivative(i), answer, eps);
  }
  // const T cx0 = -1 / boost::math::constants::e<T>();
  // auto edge = lambert_w0(make_fvar<T,m>(cx0));
  // std::cout << "edge = " << edge << std::endl;
  // edge = depth(1)(-1,inf,-inf,inf,-inf,inf,-inf,inf,-inf,inf,-inf)
  // edge = depth(1)(-1,inf,-inf,inf,-inf,inf,-inf,inf,-inf,inf,-inf)
  // edge =
  // depth(1)(-1,3.68935e+19,-9.23687e+57,4.62519e+96,-2.89497e+135,2.02945e+174,-1.52431e+213,1.19943e+252,-9.75959e+290,8.14489e+329,-6.93329e+368)
}

BOOST_AUTO_TEST_CASE_TEMPLATE(digamma_test, T, all_float_types) {
  const T eps = 1000 * boost::math::tools::epsilon<T>(); // percent
  constexpr unsigned m = 10;
  const T cx = 3;
  // Mathematica: N[Table[PolyGamma[n, 3], {n, 0, 10}], 52]
  std::array<T, m + 1> answers{
    {BOOST_MATH_TEST_VALUE(T, 0.9227843350984671393934879099175975689578406640600764)
    ,BOOST_MATH_TEST_VALUE(T, 0.3949340668482264364724151666460251892189499012067984)
    ,BOOST_MATH_TEST_VALUE(T, -0.1541138063191885707994763230228999815299725846809978)
    ,BOOST_MATH_TEST_VALUE(T, 0.1189394022668291490960221792470074166485057115123614)
    ,BOOST_MATH_TEST_VALUE(T, -0.1362661234408782319527716749688200333699420680459075)
    ,BOOST_MATH_TEST_VALUE(T, 0.2061674381338967657421515749104633482180988039424274)
    ,BOOST_MATH_TEST_VALUE(T, -0.3864797149844353246542358918536669119017636069718686)
    ,BOOST_MATH_TEST_VALUE(T, 0.8623752376394704685736020836084249051623848752441025)
    ,BOOST_MATH_TEST_VALUE(T, -2.228398747634885327823655450854278779627928241914664)
    ,BOOST_MATH_TEST_VALUE(T, 6.536422382626807143525565747764891144367614117601463)
    ,BOOST_MATH_TEST_VALUE(T, -21.4366066287129906188428320541054572790340793874298)}};
  auto x = make_fvar<T, m>(cx);
  auto y = digamma(x);
  for (auto i : boost::irange(m + 1)) {
    const T answer = answers[i];
    BOOST_CHECK_CLOSE(y.derivative(i), answer, eps);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(lgamma_test, T, all_float_types) {
  const T eps = 1000 * boost::math::tools::epsilon<T>(); // percent
  constexpr unsigned m = 10;
  const T cx = 3;
  // Mathematica: N[Table[D[LogGamma[x],{x,n}] /. x->3, {n, 0, 10}], 52]
  std::array<T, m + 1> answers{
    {BOOST_MATH_TEST_VALUE(T, 0.6931471805599453094172321214581765680755001343602553)
    ,BOOST_MATH_TEST_VALUE(T, 0.9227843350984671393934879099175975689578406640600764)
    ,BOOST_MATH_TEST_VALUE(T, 0.3949340668482264364724151666460251892189499012067984)
    ,BOOST_MATH_TEST_VALUE(T, -0.1541138063191885707994763230228999815299725846809978)
    ,BOOST_MATH_TEST_VALUE(T, 0.1189394022668291490960221792470074166485057115123614)
    ,BOOST_MATH_TEST_VALUE(T, -0.1362661234408782319527716749688200333699420680459075)
    ,BOOST_MATH_TEST_VALUE(T, 0.2061674381338967657421515749104633482180988039424274)
    ,BOOST_MATH_TEST_VALUE(T, -0.3864797149844353246542358918536669119017636069718686)
    ,BOOST_MATH_TEST_VALUE(T, 0.8623752376394704685736020836084249051623848752441025)
    ,BOOST_MATH_TEST_VALUE(T, -2.228398747634885327823655450854278779627928241914664)
    ,BOOST_MATH_TEST_VALUE(T, 6.536422382626807143525565747764891144367614117601463)}};
  auto x = make_fvar<T, m>(cx);
  auto y = lgamma(x);
  for (auto i : boost::irange(m + 1)) {
    const T answer = answers[i];
    BOOST_CHECK_CLOSE(y.derivative(i), answer, eps);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(tgamma_test, T, all_float_types) {
  const T eps = 1000 * boost::math::tools::epsilon<T>(); // percent
  constexpr unsigned m = 10;
  const T cx = 3;
  // Mathematica: N[Table[D[Gamma[x],{x,n}] /. x->3, {n, 0, 10}], 52]
  std::array<T, m + 1> answers{
    {BOOST_MATH_TEST_VALUE(T, 2.0)
    ,BOOST_MATH_TEST_VALUE(T, 1.845568670196934278786975819835195137915681328120153)
    ,BOOST_MATH_TEST_VALUE(T, 2.492929991902693057942510065508124245503778067273315)
    ,BOOST_MATH_TEST_VALUE(T, 3.449965013523673365279327178241708777509009968597547)
    ,BOOST_MATH_TEST_VALUE(T, 5.521798578098737512443417699412265532987916790978887)
    ,BOOST_MATH_TEST_VALUE(T, 8.845805593922864253981346455183370214190789096412155)
    ,BOOST_MATH_TEST_VALUE(T, 15.86959874461221647760760269963155031595848150772695)
    ,BOOST_MATH_TEST_VALUE(T, 27.46172054213435946038727460195592342721862288816812)
    ,BOOST_MATH_TEST_VALUE(T, 54.64250508485402729556251663145824730270508661240771)
    ,BOOST_MATH_TEST_VALUE(T, 96.08542140594972502872131946513104238293824803599579)
    ,BOOST_MATH_TEST_VALUE(T, 222.0936743583156040996433943128676567542497584689499)}};
  auto x = make_fvar<T, m>(cx);
  auto y = tgamma(x);
  for (auto i : boost::irange(m + 1)) {
    const T answer = answers[i];
    BOOST_CHECK_CLOSE(y.derivative(i), answer, eps);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(tgamma2_test, T, all_float_types) {
  //const T eps = 5000 * boost::math::tools::epsilon<T>(); // ok for non-multiprecision
  const T eps = 500000 * boost::math::tools::epsilon<T>(); // percent
  constexpr unsigned m = 10;
  const T cx = T(-1.5);
  // Mathematica: N[Table[D[Gamma[x],{x,n}] /. x->-3/2, {n, 0, 10}], 52]
  std::array<T, m + 1> answers{
    {BOOST_MATH_TEST_VALUE(T, 2.363271801207354703064223311121526910396732608163183)
    ,BOOST_MATH_TEST_VALUE(T, 1.661750260668596505586468565464938761014714509096807)
    ,BOOST_MATH_TEST_VALUE(T, 23.33417984355457252918927856618603412638766668207679)
    ,BOOST_MATH_TEST_VALUE(T, 47.02130025080143055642555842335081335790754507072526)
    ,BOOST_MATH_TEST_VALUE(T, 1148.336052788822231948472800239024335856568111484074)
    ,BOOST_MATH_TEST_VALUE(T, 3831.214710988836934569706027888431190714054814541186)
    ,BOOST_MATH_TEST_VALUE(T, 138190.9008816865362698874238213771413807566436072179)
    ,BOOST_MATH_TEST_VALUE(T, 644956.0066517306036921195893233874126907491308967028)
    ,BOOST_MATH_TEST_VALUE(T, 3.096453684470713902448094810299787572782887316764214e7)
    ,BOOST_MATH_TEST_VALUE(T, 1.857893143852025058151037296906468662709947415219451e8)
    ,BOOST_MATH_TEST_VALUE(T, 1.114762466163487983067783853825224537320312784955935e10)}};
  auto x = make_fvar<T, m>(cx);
  auto y = tgamma(x);
  for (auto i : boost::irange(m + 1)) {
    const T answer = answers[i];
    BOOST_CHECK_CLOSE(y.derivative(i), answer, eps);
  }
}

BOOST_AUTO_TEST_SUITE_END()
