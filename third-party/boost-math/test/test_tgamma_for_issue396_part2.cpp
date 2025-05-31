///////////////////////////////////////////////////////////////////
//  Copyright Christopher Kormanyos 2020.
//  Copyright John Maddock 2020.
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <boost/math/tools/config.hpp>
#ifndef BOOST_MATH_NO_MP_TESTS

#if 0
#define BOOST_TEST_MODULE test_tgamma_for_issue396
#include <boost/test/included/unit_test.hpp>
#else
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#endif

#include <array>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

// In issue 396, a bug regarding overflow was traced back to the tgamma function.
// This test file tests the fix in 433 and its corresponding original bug report code.

const char* issue396_control_string0 = "0.88622692545275801364908374167057259139877472806119356410690389492645564229551609068747532836927233270811341181214128533311807643286221130126254685480139353423101884932655256142496258651447541311446604768963398140008731950767573986025835009509261700929272348724745632015696088776295310820270966625045319920380686673873757671683399489468292591820439772558258086938002953369671589566640492742312409245102732742609780662578082373375752136938052805399806355360503018602224183618264830685404716174941583421211";
const char* issue396_control_string1 = "1.1332783889487855673345741655888924755602983082751597766087234145294833900560041537176305387276072906583502717008932373348895801731780765775979953796646009714415152490764416630481375706606053932396039541459764525989187023837695167161085523804417015113740063535865261183579508922972990386756543208549178543857406373798865630303794109491220205170302558277398183764099268751365861892723863412249690833216320407918186480305202146014474770321625907339955121137559264239090240758401696425720048012081453338360E6";
const char* issue396_control_string2 = "9.3209631040827166083491098091419104379064970381623611540161175194120765977611623552218076053836060223609993676387199220631835256331102029826429784793420637988460945604451237342972023988743201341318701614328454618664952897316247603329530308777063116667275003586843755354841307657702809317290363831151480295446074722690100652644579131609996151999119113967501099655433566352849645431012667388627160383486515144610582794470005796689975604764040892168183647321540427819244511610500074895473959438490375652158E156";
const char* issue396_control_string3 = "1.2723011956950554641822441803774445695066347098655278283939929838804808618389143636393314317333622154343715992535881414698586440455330620652019981627229614973177953241634213768203151670660953863412381880742653187501307209325406338924004280546485392703623101051957976224599412003938216329590158926122017907280168159527761842471509358725974702333390709735919152262756462872191402491961250987725812831155116532550035967994387094267848607390288008530653715254376729558412833771092612838971719786622446726968E2566";

template<class BigFloatType>
bool test_tgamma_for_issue396_value_checker()
{
  typedef BigFloatType floating_point_type;

  // Table[N[Gamma[(1/2) + (10^n)], 503], {n, 0, 3, 1}]

  const std::array<floating_point_type, 4U> control =
  {{
    floating_point_type(issue396_control_string0),
    floating_point_type(issue396_control_string1),
    floating_point_type(issue396_control_string2),
    floating_point_type(issue396_control_string3)
  }};

  std::uint32_t ten_pow_n = (std::uint32_t) (1);

  const floating_point_type tol = std::numeric_limits<floating_point_type>::epsilon() * (std::uint32_t) (5000);

  bool result_is_ok = true;

  for(typename std::array<floating_point_type, 4U>::size_type i = 0U; i < control.size(); ++i)
  {
    const floating_point_type g = boost::math::tgamma(boost::math::constants::half<floating_point_type>() + ten_pow_n);

    ten_pow_n *= (std::uint32_t) (10);

    const floating_point_type closeness = fabs(1 - (g / control[i]));

    result_is_ok &= (closeness < tol);
  }

  return result_is_ok;
}

template<const unsigned BigFloatDigits>
bool test_tgamma_for_issue396()
{
  typedef boost::multiprecision::number<boost::multiprecision::cpp_bin_float<BigFloatDigits>, boost::multiprecision::et_off> cpp_bin_float_type;
  typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<BigFloatDigits>, boost::multiprecision::et_off> cpp_dec_float_type;

  const bool bin_is_ok = test_tgamma_for_issue396_value_checker<cpp_bin_float_type>();
  const bool dec_is_ok = test_tgamma_for_issue396_value_checker<cpp_dec_float_type>();

  const bool result_is_ok = (bin_is_ok && dec_is_ok);

  return result_is_ok;
}

bool test_tgamma_for_issue396_cpp_dec_float_101_through_501()
{
  const bool b_101 = test_tgamma_for_issue396<101U>();
  const bool b_501 = test_tgamma_for_issue396<501U>();

  const bool result_is_ok = (b_101 && b_501);

  return result_is_ok;
}

BOOST_AUTO_TEST_CASE(test_tgamma_for_issue396_part2_tag)
{
  const bool b_101_through_501_is_ok = test_tgamma_for_issue396_cpp_dec_float_101_through_501();

  BOOST_CHECK(b_101_through_501_is_ok == true);
}
#else // No mp tests
int main(void) { return 0; }
#endif
