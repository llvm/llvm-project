// Copyright 2024 Christopher Kormanyos
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/math/special_functions/gamma.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <random>

namespace local
{
  std::mt19937 eng(static_cast<typename std::mt19937::result_type>(UINT8_C(42)));
  std::uniform_int_distribution<int> dst_one(1, 1);

  template<typename NumericType>
  auto is_close_fraction(const NumericType& a,
                         const NumericType& b,
                         const NumericType& tol) noexcept -> bool
  {
    using std::fabs;

    auto result_is_ok = bool { };

    if(b == static_cast<NumericType>(0))
    {
      result_is_ok = (fabs(a - b) < tol);
    }
    else
    {
      const auto delta = fabs(1 - (a / b));

      result_is_ok = (delta < tol);
    }

    return result_is_ok;
  }

  auto tgamma_under_cbrt_epsilon() -> void
  {
    // This test is intended to hit the lines:

    // template <class T, class Policy>
    // T gamma_imp(T z, const Policy& pol, const lanczos::undefined_lanczos&)
    // ...
    // ...near the comment:
    //      Special case for ultra-small z:

    using local_float_type = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<250>, boost::multiprecision::et_off>;

    static_assert(   (std::numeric_limits<local_float_type>::digits10 >= 248)
                  && (std::numeric_limits<local_float_type>::digits10 <= 252), "Error: Multiprecision wrong number of digits");

    // Table[N[Gamma[n (10^-84)], 260], {n, 1, 10, 1}]
    using local_data_array_type = std::array<local_float_type, static_cast<std::size_t>(UINT8_C(10))>;

    const local_data_array_type ctrl_data =
    {{
      static_cast<local_float_type>("9.9999999999999999999999999999999999999999999999999999999999999999999999999999999999942278433509846713939348790991759756895784066406007640119423276511513227322233532906404199270358122304076394840169355222697867950624167125421902178604121309579597843800360126781E+83"),
      static_cast<local_float_type>("4.9999999999999999999999999999999999999999999999999999999999999999999999999999999999942278433509846713939348790991759756895784066406007640119423276511513227322233532906503104869890919559615934405319418693491786302696988534466221756972734972784545720974832491521E+83"),
      static_cast<local_float_type>("3.3333333333333333333333333333333333333333333333333333333333333333333333333333333333275611766843180047272682124325093090229117399739340973452756609844846560655566866239935343802757050148488807303802815497619037988103143276843874668674681969322826931482456693779E+83"),
      static_cast<local_float_type>("2.4999999999999999999999999999999999999999999999999999999999999999999999999999999999942278433509846713939348790991759756895784066406007640119423276511513227322233532906700916068956514070695013535619545635079623006842631352554860913709962299194441475323232733555E+83"),
      static_cast<local_float_type>("1.9999999999999999999999999999999999999999999999999999999999999999999999999999999999942278433509846713939348790991759756895784066406007640119423276511513227322233532906799821668489311326234553100769609105873541358915452761599180492078575962399389352497160610849E+83"),
      static_cast<local_float_type>("1.6666666666666666666666666666666666666666666666666666666666666666666666666666666666608945100176513380606015457658426423562450733072674306786089943178179893988900199573565393934688775248440759332586339243334126377654940837310166737113856292271003896337573658994E+83"),
      static_cast<local_float_type>("1.4285714285714285714285714285714285714285714285714285714285714285714285714285714285656564147795560999653634505277474042610069780691721925833708990797227513036519247192711918581840620123027917945355450333175663777346809865402105363101517574523570821130186163706E+83"),
      static_cast<local_float_type>("1.2499999999999999999999999999999999999999999999999999999999999999999999999999999999942278433509846713939348790991759756895784066406007640119423276511513227322233532907096538467087703092853171796219799518255296415133916988732139227184416952014232984017855267840E+83"),
      static_cast<local_float_type>("1.1111111111111111111111111111111111111111111111111111111111111111111111111111111111053389544620957825050459902102870868006895177517118751230534387622624338433344644018306555177731611459503822472480974100160325878317849508887569916664141726330291972302168272984E+83"),
      static_cast<local_float_type>("9.9999999999999999999999999999999999999999999999999999999999999999999999999999999999422784335098467139393487909917597568957840664060076401194232765115132273222335329072943496661532976039322509265199264598431331192795598068207783839216442784241287383640775600912E+82"),
    }};

    unsigned index = 1U;

    const local_float_type little { "1E-84" };
    const local_float_type my_tol { std::numeric_limits<local_float_type>::epsilon() * 256 };

    for(const auto& ctrl : ctrl_data)
    {
      const auto x_small = static_cast<local_float_type>(static_cast<local_float_type>(index) * little);

      ++index;

      const auto g_val   = boost::math::tgamma(x_small);

      const auto result_tgamma_x_small_is_ok = is_close_fraction(g_val, ctrl, my_tol);

      BOOST_TEST(result_tgamma_x_small_is_ok);

      if(!result_tgamma_x_small_is_ok)
      {
        break; // LCOV_EXCL_LINE
      }
    }
  }

  auto tgamma_undefined_lanczos_known_error() -> void
  {
    // This test is intended to hit the lines:

    // template <class T, class Policy>
    // T gamma_imp(T z, const Policy& pol, const lanczos::undefined_lanczos&)
    // ...
    // ...for edge cases that raise errors such as domain error.

    using local_float_type = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<250>, boost::multiprecision::et_off>;

    {
      const local_float_type my_tol { std::numeric_limits<local_float_type>::epsilon() * 256 };

      for(auto index = static_cast<unsigned>(UINT8_C(0)); index < static_cast<unsigned>(UINT8_C(3)); ++index)
      {
        static_cast<void>(index);

        const local_float_type zero_ctrl { 0 };

        local_float_type zero { 0 };

        zero *= dst_one(eng);

        const auto result_zero_is_ok = is_close_fraction(zero, zero_ctrl, my_tol);

        BOOST_TEST(result_zero_is_ok);
      }
    }

    for(auto index = static_cast<unsigned>(UINT8_C(0)); index < static_cast<unsigned>(UINT8_C(3)); ++index)
    {
      static_cast<void>(index);

      local_float_type zero { 0 };

      zero *= dst_one(eng);

      bool domain_error_is_ok { false };

      try
      {
        boost::math::tgamma(zero);
      }
      catch(std::domain_error& err)
      {
        static_cast<void>(err.what());

        domain_error_is_ok = true;
      }

      BOOST_TEST(domain_error_is_ok);
    }

    for(auto index = static_cast<unsigned>(UINT8_C(0)); index < static_cast<unsigned>(UINT8_C(3)); ++index)
    {
      static_cast<void>(index);

      local_float_type my_nan = std::numeric_limits<local_float_type>::quiet_NaN();

      my_nan *= dst_one(eng);

      bool domain_error_is_ok { false };

      try
      {
        boost::math::tgamma(my_nan);
      }
      catch(std::domain_error& err)
      {
        static_cast<void>(err.what());

        domain_error_is_ok = true;
      }

      BOOST_TEST(domain_error_is_ok);
    }

    for(auto index = static_cast<unsigned>(UINT8_C(0)); index < static_cast<unsigned>(UINT8_C(3)); ++index)
    {
      static_cast<void>(index);

      local_float_type my_inf = -std::numeric_limits<local_float_type>::infinity();

      my_inf *= dst_one(eng);

      bool domain_error_is_ok { false };

      try
      {
        boost::math::tgamma(my_inf);
      }
      catch(std::domain_error& err)
      {
        static_cast<void>(err.what());

        domain_error_is_ok = true;
      }

      BOOST_TEST(domain_error_is_ok);
    }
  }

  auto lgamma_big_asymp() -> void
  {
    // This test is intended to hit the asymptotic log-gamma expansion for multiprecision.

    using local_float_type = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<250>, boost::multiprecision::et_off>;

    static_assert(   (std::numeric_limits<local_float_type>::digits10 >= 248)
                  && (std::numeric_limits<local_float_type>::digits10 <= 252), "Error: Multiprecision wrong number of digits");

    local_float_type big_arg_numerator { 1234567L };

    // Table[N[Log[Gamma[(1234567 + n)/1000]], 260], {n, 0, 3, 1}]
    const local_float_type ctrl0 { "7551.0278099842760398085493506933061185258592164059260052791257174648102458654516760859347475429811747227042884941464597963128452844941163716092798494933305452087249880911022309522317482008162381529082884245980549740815352929296384544778543502768128060636123031" };
    const local_float_type ctrl1 { "7551.0349280552065308610373629214633349814110368633190642156085097598877230874250481117271260334496206128158535271616589000730715715804390525860149840442193710637326207809853649225510544815601053550751028151966244578864039961973124357117676870769851159530881598" };
    const local_float_type ctrl2 { "7551.0420461269473499872464481408311466395059218944034402487444941839649424484440812974771163653823079980811183096059838617975730358237216549153177603277276543261570202589028877022053234067330484829335430150182956719650949866427225508960918047040961544342635607" };
    const local_float_type ctrl3 { "7551.0491641994984965305455873077287163417673805051480753266140277293097406691212685736787897808049319210175552316063915681371903149757892763598947756743285838029800201937242960777121582844482027617050641997959480250230315703285501155035159657216828260204447321" };

    const local_float_type my_tol { std::numeric_limits<local_float_type>::epsilon() * 256 };

    const local_float_type lg_big0 { boost::math::lgamma(big_arg_numerator / 1000) }; ++big_arg_numerator;
    const local_float_type lg_big1 { boost::math::lgamma(big_arg_numerator / 1000) }; ++big_arg_numerator;
    const local_float_type lg_big2 { boost::math::lgamma(big_arg_numerator / 1000) }; ++big_arg_numerator;
    const local_float_type lg_big3 { boost::math::lgamma(big_arg_numerator / 1000) };

    const auto result_lgamma_big0_is_ok = is_close_fraction(lg_big0, ctrl0, my_tol);
    const auto result_lgamma_big1_is_ok = is_close_fraction(lg_big1, ctrl1, my_tol);
    const auto result_lgamma_big2_is_ok = is_close_fraction(lg_big2, ctrl2, my_tol);
    const auto result_lgamma_big3_is_ok = is_close_fraction(lg_big3, ctrl3, my_tol);

    BOOST_TEST(result_lgamma_big0_is_ok);
    BOOST_TEST(result_lgamma_big1_is_ok);
    BOOST_TEST(result_lgamma_big2_is_ok);
    BOOST_TEST(result_lgamma_big3_is_ok);
  }

  auto lgamma_undefined_lanczos_known_error() -> void
  {
    // This test is intended to hit the lines:

    // template <class T, class Policy>
    // T lgamma_imp(T z, const Policy& pol, const lanczos::undefined_lanczos&, int* sign)    // ...
    // ...
    // ...for edge cases that raise errors such as domain error.

    using local_float_type = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<250>, boost::multiprecision::et_off>;

    for(auto index = static_cast<unsigned>(UINT8_C(0)); index < static_cast<unsigned>(UINT8_C(3)); ++index)
    {
      static_cast<void>(index);

      local_float_type zero { 0 };

      zero *= dst_one(eng);

      bool domain_error_is_ok { false };

      try
      {
        boost::math::lgamma(zero);
      }
      catch(std::domain_error& err)
      {
        static_cast<void>(err.what());

        domain_error_is_ok = true;
      }

      BOOST_TEST(domain_error_is_ok);
    }

    for(auto index = static_cast<unsigned>(UINT8_C(0)); index < static_cast<unsigned>(UINT8_C(3)); ++index)
    {
      static_cast<void>(index);

      local_float_type my_nan = std::numeric_limits<local_float_type>::quiet_NaN();

      my_nan *= dst_one(eng);

      bool domain_error_is_ok { false };

      try
      {
        boost::math::lgamma(my_nan);
      }
      catch(std::domain_error& err)
      {
        static_cast<void>(err.what());

        domain_error_is_ok = true;
      }

      BOOST_TEST(domain_error_is_ok);
    }

    for(auto index = static_cast<unsigned>(UINT8_C(0)); index < static_cast<unsigned>(UINT8_C(3)); ++index)
    {
      static_cast<void>(index);

      local_float_type my_inf = -std::numeric_limits<local_float_type>::infinity();

      my_inf *= dst_one(eng);

      bool overflow_error_is_ok { false };

      try
      {
        boost::math::lgamma(my_inf);
      }
      catch(std::overflow_error& err)
      {
        static_cast<void>(err.what());

        overflow_error_is_ok = true;
      }

      BOOST_TEST(overflow_error_is_ok);
    }
  }
}

auto main() -> int
{
  local::tgamma_under_cbrt_epsilon();
  local::tgamma_undefined_lanczos_known_error();
  local::lgamma_big_asymp();
  local::lgamma_undefined_lanczos_known_error();

  const auto result_is_ok = (boost::report_errors() == 0);

  return (result_is_ok ? 0 : -1);
}
