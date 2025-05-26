// Copyright Paul Bristow 2007, 2011.
// Copyright John Maddock 2006, 2011.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Check values of constants are drawn from an independent source, or calculated.
// Both must be at long double precision for the most precise compilers floating-point implementation.
// So all values use static_cast<RealType>() of values at least 40 decimal digits
// and that have suffix L to ensure floating-point type is long double.

// Steve Moshier's command interpreter V1.3 100 digits calculator used for some values.

#ifdef _MSC_VER
#  pragma warning(disable : 4127) // conditional expression is constant.
#endif
#include "math_unit_test.hpp"
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#ifdef BOOST_MATH_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif
#include <type_traits>
#include <limits>
#include <cmath>

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900)
#include <boost/math/tools/agm.hpp>
// Check at compile time that the construction method for constants of type float, is "construct from a float", or "construct from a double", ...
static_assert((std::is_same<boost::math::constants::construction_traits<float, boost::math::policies::policy<>>::type, std::integral_constant<int, boost::math::constants::construct_from_float>>::value), "Need to be able to construct from float");
static_assert((std::is_same<boost::math::constants::construction_traits<double, boost::math::policies::policy<>>::type, std::integral_constant<int, boost::math::constants::construct_from_double>>::value), "Need to be able to construct from double");
static_assert((std::is_same<boost::math::constants::construction_traits<long double, boost::math::policies::policy<>>::type, std::integral_constant<int, (sizeof(double) == sizeof(long double) ? boost::math::constants::construct_from_double : boost::math::constants::construct_from_long_double)>>::value), "Need to be able to construct from long double");
static_assert((std::is_same<boost::math::constants::construction_traits<boost::math::concepts::real_concept, boost::math::policies::policy<>>::type, std::integral_constant<int, 0>>::value), "Need to be able to construct from real_concept");

// Policy to set precision at maximum possible using long double.
using real_concept_policy_1 = boost::math::policies::policy<boost::math::policies::digits2<std::numeric_limits<long double>::digits>>;
// Policy with precision +2 (could be any reasonable value),
// forces the precision of the policy to be greater than
// that of a long double, and therefore triggers different code (construct from string).
#ifdef BOOST_MATH_USE_FLOAT128
using real_concept_policy_2 = boost::math::policies::policy<boost::math::policies::digits2<115>>;
#else
using real_concept_policy_2 = boost::math::policies::policy<boost::math::policies::digits2<std::numeric_limits<long double>::digits + 2>>;
#endif
// Policy with precision greater than the string representations, forces computation of values (i.e. different code path):
using real_concept_policy_3 = boost::math::policies::policy<boost::math::policies::digits2<400>>;

static_assert((std::is_same<boost::math::constants::construction_traits<boost::math::concepts::real_concept, real_concept_policy_1 >::type, std::integral_constant<int, (sizeof(double) == sizeof(long double) ? boost::math::constants::construct_from_double : boost::math::constants::construct_from_long_double) >>::value), "Need to be able to construct from long double");
static_assert((std::is_same<boost::math::constants::construction_traits<boost::math::concepts::real_concept, real_concept_policy_2 >::type, std::integral_constant<int, boost::math::constants::construct_from_string>>::value), "Need to be able to construct integer from string");
static_assert((boost::math::constants::construction_traits<boost::math::concepts::real_concept, real_concept_policy_3>::type::value >= 5), "Need 5 digits");
#endif // C++11

// We need to declare a conceptual type whose precision is unknown at
// compile time, and is so enormous when checked at runtime,
// that we're forced to calculate the values of the constants ourselves.

namespace boost{ namespace math{ namespace concepts{

class big_real_concept : public real_concept
{
public:
   big_real_concept() {}
   template <typename T>
   big_real_concept(const T& t, typename std::enable_if<std::is_convertible<T, real_concept>::value, bool>::type = false) : real_concept(t) {}
};

inline int itrunc(const big_real_concept& val)
{
   BOOST_MATH_STD_USING
   return itrunc(val.value());
}

}
namespace tools{

template <>
inline constexpr int digits<concepts::big_real_concept>(BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(T)) noexcept
{
   return 2 * boost::math::constants::max_string_digits;
}

}}}

template <typename RealType>
void test_spots(RealType)
{
   // Basic sanity checks for constants,
   // where template parameter RealType can be float, double, long double,
   // or real_concept, a prototype for user-defined floating-point types.

   // Parameter RealType is only used to communicate the RealType,
   // and is an arbitrary zero for all tests.

   //typedef typename boost::math::constants::construction_traits<RealType, boost::math::policies::policy<>>::type construction_type;
   using namespace boost::math::constants;
   BOOST_MATH_STD_USING

   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L, pi<RealType>(), 2);
   CHECK_ULP_CLOSE(sqrt(3.14159265358979323846264338327950288419716939937510L), root_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(sqrt(3.14159265358979323846264338327950288419716939937510L/2), root_half_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(sqrt(3.14159265358979323846264338327950288419716939937510L * 2), root_two_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(sqrt(log(4.0L)), root_ln_four<RealType>(), 2);
   CHECK_ULP_CLOSE(2.71828182845904523536028747135266249775724709369995L, e<RealType>(), 2);
   CHECK_ULP_CLOSE(0.5L, half<RealType>(), 2);
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104259335L, euler<RealType>(), 2);
   CHECK_ULP_CLOSE(sqrt(2.0L), root_two<RealType>(), 2);
   CHECK_ULP_CLOSE(log(2.0L), ln_two<RealType>(), 2);
   CHECK_ULP_CLOSE(log(10.0L), ln_ten<RealType>(), 2);
   CHECK_ULP_CLOSE(log(log(2.0L)), ln_ln_two<RealType>(), 2);

   CHECK_ULP_CLOSE(static_cast<long double>(1)/3, third<RealType>(), 2);
   CHECK_ULP_CLOSE(static_cast<long double>(2)/3, twothirds<RealType>(), 2);
   CHECK_ULP_CLOSE(0.14159265358979323846264338327950288419716939937510L, pi_minus_three<RealType>(), 2);
   CHECK_ULP_CLOSE(4.L - 3.14159265358979323846264338327950288419716939937510L, four_minus_pi<RealType>(), 2);
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   CHECK_ULP_CLOSE(pow((3.14159265358979323846264338327950288419716939937510L), 2.71828182845904523536028747135266249775724709369995L), pi_pow_e<RealType>(), 2);
   CHECK_ULP_CLOSE(pow((3.14159265358979323846264338327950288419716939937510L), 0.33333333333333333333333333333333333333333333333333L), cbrt_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(exp(-0.5L), exp_minus_half<RealType>(), 2);
   CHECK_ULP_CLOSE(pow(2.71828182845904523536028747135266249775724709369995L, 3.14159265358979323846264338327950288419716939937510L), e_pow_pi<RealType>(), 3);


#else // Only double, so no suffix L.
   CHECK_ULP_CLOSE(pow((3.14159265358979323846264338327950288419716939937510), 2.71828182845904523536028747135266249775724709369995), pi_pow_e<RealType>(), 2);
   CHECK_ULP_CLOSE(pow((3.14159265358979323846264338327950288419716939937510), 0.33333333333333333333333333333333333333333333333333), cbrt_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(exp(-0.5), exp_minus_half<RealType>(), 2);
#endif
   // Rational fractions.
   CHECK_ULP_CLOSE(0.333333333333333333333333333333333333333L, third<RealType>(), 2);
   CHECK_ULP_CLOSE(0.666666666666666666666666666666666666667L, two_thirds<RealType>(), 2);
   CHECK_ULP_CLOSE(0.75L, three_quarters<RealType>(), 2);
   CHECK_ULP_CLOSE(0.1666666666666666666666666666666666666667L, sixth<RealType>(), 2);

   // Two and related.
   CHECK_ULP_CLOSE(sqrt(2.L), root_two<RealType>(), 2);
   CHECK_ULP_CLOSE(sqrt(3.L), root_three<RealType>(), 2);
   CHECK_ULP_CLOSE(sqrt(2.L)/2, half_root_two<RealType>(), 2);
   CHECK_ULP_CLOSE(log(2.L), ln_two<RealType>(), 2);
   CHECK_ULP_CLOSE(log(log(2.0L)), ln_ln_two<RealType>(), 2);
   CHECK_ULP_CLOSE(sqrt(log(4.0L)), root_ln_four<RealType>(), 2);
   CHECK_ULP_CLOSE(1/sqrt(2.0L), one_div_root_two<RealType>(), 2);

   // pi.
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L, pi<RealType>(), 2);
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L/2, half_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L/4, quarter_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L/3, third_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L/4, quarter_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L/6, sixth_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(2 * 3.14159265358979323846264338327950288419716939937510L, two_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(2 * 3.14159265358979323846264338327950288419716939937510L, tau<RealType>(), 2);
   CHECK_ULP_CLOSE(3 * 3.14159265358979323846264338327950288419716939937510L / 4, three_quarters_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(4 * 3.14159265358979323846264338327950288419716939937510L / 3, four_thirds_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(1 / (3.14159265358979323846264338327950288419716939937510L), one_div_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(2 / (3.14159265358979323846264338327950288419716939937510L), two_div_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(1 / (2 * 3.14159265358979323846264338327950288419716939937510L), one_div_two_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(sqrt(3.14159265358979323846264338327950288419716939937510L), root_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(sqrt(3.14159265358979323846264338327950288419716939937510L / 2), root_half_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(sqrt(2 * 3.14159265358979323846264338327950288419716939937510L), root_two_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(1 / sqrt(3.14159265358979323846264338327950288419716939937510L), one_div_root_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(2 / sqrt(3.14159265358979323846264338327950288419716939937510L), two_div_root_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(1 / sqrt(2 * 3.14159265358979323846264338327950288419716939937510L), one_div_root_two_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(sqrt(1. / 3.14159265358979323846264338327950288419716939937510L), root_one_div_pi<RealType>(), 2);
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L - 3.L, pi_minus_three<RealType>(), 4 * 4 ); // 4 * 2 because of cancellation loss.
   CHECK_ULP_CLOSE(4.L - 3.14159265358979323846264338327950288419716939937510L, four_minus_pi<RealType>(), 4 );
   //
   CHECK_ULP_CLOSE(pow((3.14159265358979323846264338327950288419716939937510L), 2.71828182845904523536028747135266249775724709369995L), pi_pow_e<RealType>(), 2);  // See above.
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L, pi_sqr<RealType>(), 2);  // See above.
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L/6, pi_sqr_div_six<RealType>(), 2);  // See above.
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L, pi_cubed<RealType>(), 2);  // See above.

   // CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L, cbrt_pi<RealType>(), 2);  // See above.
   CHECK_ULP_CLOSE(cbrt_pi<RealType>() * cbrt_pi<RealType>() * cbrt_pi<RealType>(), pi<RealType>(), 2);
   CHECK_ULP_CLOSE((1)/cbrt_pi<RealType>(), one_div_cbrt_pi<RealType>(), 2);

   // Euler
   CHECK_ULP_CLOSE(2.71828182845904523536028747135266249775724709369995L, e<RealType>(), 2);
   //CHECK_ULP_CLOSE(exp(-0.5L), exp_minus_half<RealType>(), 2);  // See above.
   CHECK_ULP_CLOSE(exp(-1.L), exp_minus_one<RealType>(), 2);
   CHECK_ULP_CLOSE(pow(e<RealType>(), pi<RealType>()), e_pow_pi<RealType>(), 3); // See also above.
   CHECK_ULP_CLOSE(sqrt(e<RealType>()), root_e<RealType>(), 2);
   CHECK_ULP_CLOSE(log10(e<RealType>()), log10_e<RealType>(), 2);
   CHECK_ULP_CLOSE(1/log10(e<RealType>()), one_div_log10_e<RealType>(), 2);
   CHECK_ULP_CLOSE((1/ln_two<RealType>()), log2_e<RealType>(), 2);

   // Trigonometric
   CHECK_ULP_CLOSE(pi<RealType>()/180, degree<RealType>(), 2);
   CHECK_ULP_CLOSE(180 / pi<RealType>(), radian<RealType>(), 2);
   CHECK_ULP_CLOSE(sin(1.L), sin_one<RealType>(), 2);
   CHECK_ULP_CLOSE(cos(1.L), cos_one<RealType>(), 2);
   CHECK_ULP_CLOSE(sinh(1.L), sinh_one<RealType>(), 2);
   CHECK_ULP_CLOSE(cosh(1.L), cosh_one<RealType>(), 2);

   // Phi
   CHECK_ULP_CLOSE((1.L + sqrt(5.L)) /2, phi<RealType>(), 2);
   CHECK_ULP_CLOSE(log((1.L + sqrt(5.L)) /2), ln_phi<RealType>(), 2);
   CHECK_ULP_CLOSE(1.L / log((1.L + sqrt(5.L)) /2), one_div_ln_phi<RealType>(), 2);

   //Euler's Gamma
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992L, euler<RealType>(), 2); // (sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(1.L/ 0.57721566490153286060651209008240243104215933593992L, one_div_euler<RealType>(), 2); // (from sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992L * 0.57721566490153286060651209008240243104215933593992L, euler_sqr<RealType>(), 2); // (from sequence A001620 in OEIS).

   // Misc
   CHECK_ULP_CLOSE(1.644934066848226436472415166646025189218949901206L, zeta_two<RealType>(), 2); // A013661 as a constant (usually base 10) in OEIS.
   CHECK_ULP_CLOSE(1.20205690315959428539973816151144999076498629234049888179227L, zeta_three<RealType>(), 2); // (sequence A002117 in OEIS)
   CHECK_ULP_CLOSE(.91596559417721901505460351493238411077414937428167213L, catalan<RealType>(), 2); // A006752 as a constant in OEIS.
   CHECK_ULP_CLOSE(1.1395470994046486574927930193898461120875997958365518247216557100852480077060706857071875468869385150L, extreme_value_skewness<RealType>(), 2); //  Mathematica: N[12 Sqrt[6]  Zeta[3]/Pi^3, 1101]
   CHECK_ULP_CLOSE(0.6311106578189371381918993515442277798440422031347194976580945856929268196174737254599050270325373067L, rayleigh_skewness<RealType>(), 2); // Mathematica: N[2 Sqrt[Pi] (Pi - 3)/((4 - Pi)^(3/2)), 1100]
   CHECK_ULP_CLOSE(2.450893006876380628486604106197544154e-01L, rayleigh_kurtosis_excess<RealType>(), 4 * 2);
   CHECK_ULP_CLOSE(2.68545200106530644530971483548179569382038229399446295305115234555721885953715200280114117493184769799515L, khinchin<RealType>(), 4 ); // A002210 as a constant https://oeis.org/A002210/constant
   CHECK_ULP_CLOSE(1.2824271291006226368753425688697917277676889273250011L, glaisher<RealType>(), 4 ); // https://oeis.org/A074962/constant

   //
   // Last of all come the test cases that behave differently if we're calculating the constants on the fly:
   //
   if(boost::math::tools::digits<RealType>() > boost::math::constants::max_string_digits)
   {
      // This suffers from cancellation error, so increased 4:
      CHECK_ULP_CLOSE(static_cast<RealType>(4. - 3.14159265358979323846264338327950288419716939937510L), four_minus_pi<RealType>(), 4 * 3);
      CHECK_ULP_CLOSE(static_cast<RealType>(0.14159265358979323846264338327950288419716939937510L), pi_minus_three<RealType>(), 4 * 3);
   }
   else
   {
      CHECK_ULP_CLOSE(static_cast<RealType>(4. - 3.14159265358979323846264338327950288419716939937510L), four_minus_pi<RealType>(), 2);
      CHECK_ULP_CLOSE(static_cast<RealType>(0.14159265358979323846264338327950288419716939937510L), pi_minus_three<RealType>(), 2);
   }
} // template <typename RealType>void test_spots(RealType)

void test_float_spots()
{
   // Basic sanity checks for constants in boost::math::float_constants::
   // for example: boost::math::float_constants::pi
   // (rather than boost::math::constants::pi<float>() ).
   using namespace boost::math::float_constants;
   BOOST_MATH_STD_USING

   CHECK_ULP_CLOSE(static_cast<float>(3.14159265358979323846264338327950288419716939937510F), pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(3.14159265358979323846264338327950288419716939937510F)), root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(3.14159265358979323846264338327950288419716939937510F/2)), root_half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(3.14159265358979323846264338327950288419716939937510F * 2)), root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(log(4.0F))), root_ln_four, 2);
   CHECK_ULP_CLOSE(static_cast<float>(2.71828182845904523536028747135266249775724709369995F), e, 2);
   CHECK_ULP_CLOSE(static_cast<float>(0.5), half, 2);
   CHECK_ULP_CLOSE(static_cast<float>(0.57721566490153286060651209008240243104259335F), euler, 2);
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(2.0F)), root_two, 2);
   CHECK_ULP_CLOSE(static_cast<float>(log(2.0F)), ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<float>(log(log(2.0F))), ln_ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<float>(1)/3, third, 2);
   CHECK_ULP_CLOSE(static_cast<float>(2)/3, twothirds, 2);
   CHECK_ULP_CLOSE(static_cast<float>(0.14159265358979323846264338327950288419716939937510F), pi_minus_three, 2);
   CHECK_ULP_CLOSE(static_cast<float>(4.F - 3.14159265358979323846264338327950288419716939937510F), four_minus_pi, 2);
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   CHECK_ULP_CLOSE(static_cast<float>(pow((3.14159265358979323846264338327950288419716939937510F), 2.71828182845904523536028747135266249775724709369995F)), pi_pow_e, 2);
   CHECK_ULP_CLOSE(static_cast<float>(pow((3.14159265358979323846264338327950288419716939937510F), 0.33333333333333333333333333333333333333333333333333F)), cbrt_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(exp(-0.5F)), exp_minus_half, 2);
   CHECK_ULP_CLOSE(static_cast<float>(pow(2.71828182845904523536028747135266249775724709369995F, 3.14159265358979323846264338327950288419716939937510F)), e_pow_pi, 2);


#else // Only double, so no suffix F.
   CHECK_ULP_CLOSE(static_cast<float>(pow((3.14159265358979323846264338327950288419716939937510), 2.71828182845904523536028747135266249775724709369995)), pi_pow_e, 2);
   CHECK_ULP_CLOSE(static_cast<float>(pow((3.14159265358979323846264338327950288419716939937510), 0.33333333333333333333333333333333333333333333333333)), cbrt_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(exp(-0.5)), exp_minus_half, 2);
#endif
   // Rational fractions.
   CHECK_ULP_CLOSE(static_cast<float>(0.333333333333333333333333333333333333333F), third, 2);
   CHECK_ULP_CLOSE(static_cast<float>(0.666666666666666666666666666666666666667F), two_thirds, 2);
   CHECK_ULP_CLOSE(static_cast<float>(0.75F), three_quarters, 2);
   // Two and related.
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(2.F)), root_two, 2);
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(3.F)), root_three, 2);
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(2.F)/2), half_root_two, 2);
   CHECK_ULP_CLOSE(static_cast<float>(log(2.F)), ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<float>(log(log(2.0F))), ln_ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(log(4.0F))), root_ln_four, 2);
   CHECK_ULP_CLOSE(static_cast<float>(1/sqrt(2.0F)), one_div_root_two, 2);

   // pi.
   CHECK_ULP_CLOSE(static_cast<float>(3.14159265358979323846264338327950288419716939937510F), pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(3.14159265358979323846264338327950288419716939937510F/2), half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(3.14159265358979323846264338327950288419716939937510F/4), quarter_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(3.14159265358979323846264338327950288419716939937510F/3), third_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(3.14159265358979323846264338327950288419716939937510F/6), sixth_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(2 * 3.14159265358979323846264338327950288419716939937510F), two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(3 * 3.14159265358979323846264338327950288419716939937510F / 4), three_quarters_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(4 * 3.14159265358979323846264338327950288419716939937510F / 3), four_thirds_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(1 / (3.14159265358979323846264338327950288419716939937510F)), one_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(2 / (3.14159265358979323846264338327950288419716939937510F)), two_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(1 / (2 * 3.14159265358979323846264338327950288419716939937510F)), one_div_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(3.14159265358979323846264338327950288419716939937510F)), root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(3.14159265358979323846264338327950288419716939937510F / 2)), root_half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(2 * 3.14159265358979323846264338327950288419716939937510F)), root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(1 / sqrt(3.14159265358979323846264338327950288419716939937510F)), one_div_root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(2 / sqrt(3.14159265358979323846264338327950288419716939937510F)), two_div_root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(1 / sqrt(2 * 3.14159265358979323846264338327950288419716939937510F)), one_div_root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(sqrt(1. / 3.14159265358979323846264338327950288419716939937510F)), root_one_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<float>(3.14159265358979323846264338327950288419716939937510L - 3.L), pi_minus_three, 4 * 2 ); // 4 * 2 because of cancellation loss.
   CHECK_ULP_CLOSE(static_cast<float>(4.L - 3.14159265358979323846264338327950288419716939937510L), four_minus_pi, 4 );
   //
   CHECK_ULP_CLOSE(static_cast<float>(pow((3.14159265358979323846264338327950288419716939937510F), 2.71828182845904523536028747135266249775724709369995F)), pi_pow_e, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<float>(3.14159265358979323846264338327950288419716939937510F * 3.14159265358979323846264338327950288419716939937510F), pi_sqr, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<float>(3.14159265358979323846264338327950288419716939937510F * 3.14159265358979323846264338327950288419716939937510F/6), pi_sqr_div_six, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<float>(3.14159265358979323846264338327950288419716939937510F * 3.14159265358979323846264338327950288419716939937510F * 3.14159265358979323846264338327950288419716939937510F), pi_cubed, 2);  // See above.

   // CHECK_ULP_CLOSE(static_cast<float>(3.14159265358979323846264338327950288419716939937510F * 3.14159265358979323846264338327950288419716939937510F), cbrt_pi, 2);  // See above.
   CHECK_ULP_CLOSE(cbrt_pi * cbrt_pi * cbrt_pi, pi, 2);
   CHECK_ULP_CLOSE((static_cast<float>(1)/cbrt_pi), one_div_cbrt_pi, 2);

   // Euler
   CHECK_ULP_CLOSE(static_cast<float>(2.71828182845904523536028747135266249775724709369995F), e, 2);

   //CHECK_ULP_CLOSE(static_cast<float>(exp(-0.5F)), exp_minus_half, 2);  // See above.
   CHECK_ULP_CLOSE(pow(e, pi), e_pow_pi, 2); // See also above.
   CHECK_ULP_CLOSE(sqrt(e), root_e, 2);
   CHECK_ULP_CLOSE(log10(e), log10_e, 2);
   CHECK_ULP_CLOSE(static_cast<float>(1)/log10(e), one_div_log10_e, 2);

   // Trigonometric
   CHECK_ULP_CLOSE(pi/180, degree, 2);
   CHECK_ULP_CLOSE(180 / pi, radian, 2);
   CHECK_ULP_CLOSE(sin(1.F), sin_one, 2);
   CHECK_ULP_CLOSE(cos(1.F), cos_one, 2);
   CHECK_ULP_CLOSE(sinh(1.F), sinh_one, 2);
   CHECK_ULP_CLOSE(cosh(1.F), cosh_one, 2);

   // Phi
   CHECK_ULP_CLOSE((1.F + sqrt(5.F)) /2, phi, 2);
   CHECK_ULP_CLOSE(log((1.F + sqrt(5.F)) /2), ln_phi, 2);
   CHECK_ULP_CLOSE(1.F / log((1.F + sqrt(5.F)) /2), one_div_ln_phi, 2);

   // Euler's Gamma
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992F, euler, 2); // (sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(1.F/ 0.57721566490153286060651209008240243104215933593992F, one_div_euler, 2); // (from sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992F * 0.57721566490153286060651209008240243104215933593992F, euler_sqr, 2); // (from sequence A001620 in OEIS).

   // Misc
   CHECK_ULP_CLOSE(1.644934066848226436472415166646025189218949901206F, zeta_two, 2); // A013661 as a constant (usually base 10) in OEIS.
   CHECK_ULP_CLOSE(1.20205690315959428539973816151144999076498629234049888179227F, zeta_three, 2); // (sequence A002117 in OEIS)
   CHECK_ULP_CLOSE(.91596559417721901505460351493238411077414937428167213F, catalan, 2); // A006752 as a constant in OEIS.
   CHECK_ULP_CLOSE(1.1395470994046486574927930193898461120875997958365518247216557100852480077060706857071875468869385150F, extreme_value_skewness, 2); //  Mathematica: N[12 Sqrt[6]  Zeta[3]/Pi^3, 1101]
   CHECK_ULP_CLOSE(0.6311106578189371381918993515442277798440422031347194976580945856929268196174737254599050270325373067F, rayleigh_skewness, 2); // Mathematica: N[2 Sqrt[Pi] (Pi - 3)/((4 - Pi)^(3/2)), 1100]
   CHECK_ULP_CLOSE(2.450893006876380628486604106197544154e-01F, rayleigh_kurtosis_excess, 2);
   CHECK_ULP_CLOSE(2.68545200106530644530971483548179569382038229399446295305115234555721885953715200280114117493184769799515F, khinchin, 4 ); // A002210 as a constant https://oeis.org/A002210/constant
   CHECK_ULP_CLOSE(1.2824271291006226368753425688697917277676889273250011F, glaisher, 4 ); // https://oeis.org/A074962/constant
   CHECK_ULP_CLOSE(4.66920160910299067185320382046620161725F, first_feigenbaum, 1);
} // template <typename RealType>void test_spots(RealType)

#ifdef __STDCPP_FLOAT32_T__

void test_f32_spots()
{
   // Basic sanity checks for constants in boost::math::float_constants::
   // for example: boost::math::float_constants::pi
   // (rather than boost::math::constants::pi<float>() ).
   using namespace boost::math::float_constants;
   BOOST_MATH_STD_USING

   CHECK_ULP_CLOSE(static_cast<std::float32_t>(3.14159265358979323846264338327950288419716939937510F32), pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(3.14159265358979323846264338327950288419716939937510F32)), root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(3.14159265358979323846264338327950288419716939937510F32/2)), root_half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(3.14159265358979323846264338327950288419716939937510F32 * 2)), root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(log(4.0F32))), root_ln_four, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(2.71828182845904523536028747135266249775724709369995F32), e, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(0.5), half, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(0.57721566490153286060651209008240243104259335F32), euler, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(2.0F32)), root_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(log(2.0F32)), ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(log(log(2.0F32))), ln_ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(1)/3, third, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(2)/3, twothirds, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(0.14159265358979323846264338327950288419716939937510F32), pi_minus_three, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(4.F32 - 3.14159265358979323846264338327950288419716939937510F32), four_minus_pi, 2);
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(pow((3.14159265358979323846264338327950288419716939937510F32), 2.71828182845904523536028747135266249775724709369995F32)), pi_pow_e, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(pow((3.14159265358979323846264338327950288419716939937510F32), 0.33333333333333333333333333333333333333333333333333F32)), cbrt_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(exp(-0.5F32)), exp_minus_half, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(pow(2.71828182845904523536028747135266249775724709369995F32, 3.14159265358979323846264338327950288419716939937510F32)), e_pow_pi, 2);


#else // Only double, so no suffix F32.
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(pow((3.14159265358979323846264338327950288419716939937510), 2.71828182845904523536028747135266249775724709369995)), pi_pow_e, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(pow((3.14159265358979323846264338327950288419716939937510), 0.33333333333333333333333333333333333333333333333333)), cbrt_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(exp(-0.5)), exp_minus_half, 2);
#endif
   // Rational fractions.
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(0.333333333333333333333333333333333333333F32), third, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(0.666666666666666666666666666666666666667F32), two_thirds, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(0.75F32), three_quarters, 2);
   // Two and related.
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(2.F32)), root_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(3.F32)), root_three, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(2.F32)/2), half_root_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(log(2.F32)), ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(log(log(2.0F32))), ln_ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(log(4.0F32))), root_ln_four, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(1/sqrt(2.0F32)), one_div_root_two, 2);

   // pi.
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(3.14159265358979323846264338327950288419716939937510F32), pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(3.14159265358979323846264338327950288419716939937510F32/2), half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(3.14159265358979323846264338327950288419716939937510F32/4), quarter_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(3.14159265358979323846264338327950288419716939937510F32/3), third_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(3.14159265358979323846264338327950288419716939937510F32/6), sixth_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(2 * 3.14159265358979323846264338327950288419716939937510F32), two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(3 * 3.14159265358979323846264338327950288419716939937510F32 / 4), three_quarters_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(4 * 3.14159265358979323846264338327950288419716939937510F32 / 3), four_thirds_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(1 / (3.14159265358979323846264338327950288419716939937510F32)), one_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(2 / (3.14159265358979323846264338327950288419716939937510F32)), two_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(1 / (2 * 3.14159265358979323846264338327950288419716939937510F32)), one_div_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(3.14159265358979323846264338327950288419716939937510F32)), root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(3.14159265358979323846264338327950288419716939937510F32 / 2)), root_half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(2 * 3.14159265358979323846264338327950288419716939937510F32)), root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(1 / sqrt(3.14159265358979323846264338327950288419716939937510F32)), one_div_root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(2 / sqrt(3.14159265358979323846264338327950288419716939937510F32)), two_div_root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(1 / sqrt(2 * 3.14159265358979323846264338327950288419716939937510F32)), one_div_root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(sqrt(1. / 3.14159265358979323846264338327950288419716939937510F32)), root_one_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(3.14159265358979323846264338327950288419716939937510L - 3.L), pi_minus_three, 4 * 2 ); // 4 * 2 because of cancellation loss.
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(4.L - 3.14159265358979323846264338327950288419716939937510L), four_minus_pi, 4 );
   //
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(pow((3.14159265358979323846264338327950288419716939937510F32), 2.71828182845904523536028747135266249775724709369995F32)), pi_pow_e, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(3.14159265358979323846264338327950288419716939937510F32 * 3.14159265358979323846264338327950288419716939937510F32), pi_sqr, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(3.14159265358979323846264338327950288419716939937510F32 * 3.14159265358979323846264338327950288419716939937510F32/6), pi_sqr_div_six, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(3.14159265358979323846264338327950288419716939937510F32 * 3.14159265358979323846264338327950288419716939937510F32 * 3.14159265358979323846264338327950288419716939937510F32), pi_cubed, 2);  // See above.

   // CHECK_ULP_CLOSE(static_cast<std::float32_t>(3.14159265358979323846264338327950288419716939937510F32 * 3.14159265358979323846264338327950288419716939937510F32), cbrt_pi, 2);  // See above.
   CHECK_ULP_CLOSE(cbrt_pi * cbrt_pi * cbrt_pi, pi, 2);
   CHECK_ULP_CLOSE((static_cast<std::float32_t>(1)/cbrt_pi), one_div_cbrt_pi, 2);

   // Euler
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(2.71828182845904523536028747135266249775724709369995F32), e, 2);

   //CHECK_ULP_CLOSE(static_cast<std::float32_t>(exp(-0.5F32)), exp_minus_half, 2);  // See above.
   CHECK_ULP_CLOSE(pow(e, pi), e_pow_pi, 2); // See also above.
   CHECK_ULP_CLOSE(sqrt(e), root_e, 2);
   CHECK_ULP_CLOSE(log10(e), log10_e, 2);
   CHECK_ULP_CLOSE(static_cast<std::float32_t>(1)/log10(e), one_div_log10_e, 2);

   // Trigonometric
   CHECK_ULP_CLOSE(pi/180, degree, 2);
   CHECK_ULP_CLOSE(180 / pi, radian, 2);
   CHECK_ULP_CLOSE(sin(1.F32), sin_one, 2);
   CHECK_ULP_CLOSE(cos(1.F32), cos_one, 2);
   CHECK_ULP_CLOSE(sinh(1.F32), sinh_one, 2);
   CHECK_ULP_CLOSE(cosh(1.F32), cosh_one, 2);

   // Phi
   CHECK_ULP_CLOSE((1.F32 + sqrt(5.F32)) /2, phi, 2);
   CHECK_ULP_CLOSE(log((1.F32 + sqrt(5.F32)) /2), ln_phi, 2);
   CHECK_ULP_CLOSE(1.F32 / log((1.F32 + sqrt(5.F32)) /2), one_div_ln_phi, 2);

   // Euler's Gamma
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992F32, euler, 2); // (sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(1.F32/ 0.57721566490153286060651209008240243104215933593992F32, one_div_euler, 2); // (from sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992F32 * 0.57721566490153286060651209008240243104215933593992F32, euler_sqr, 2); // (from sequence A001620 in OEIS).

   // Misc
   CHECK_ULP_CLOSE(1.644934066848226436472415166646025189218949901206F32, zeta_two, 2); // A013661 as a constant (usually base 10) in OEIS.
   CHECK_ULP_CLOSE(1.20205690315959428539973816151144999076498629234049888179227F32, zeta_three, 2); // (sequence A002117 in OEIS)
   CHECK_ULP_CLOSE(.91596559417721901505460351493238411077414937428167213F32, catalan, 2); // A006752 as a constant in OEIS.
   CHECK_ULP_CLOSE(1.1395470994046486574927930193898461120875997958365518247216557100852480077060706857071875468869385150F32, extreme_value_skewness, 2); //  Mathematica: N[12 Sqrt[6]  Zeta[3]/Pi^3, 1101]
   CHECK_ULP_CLOSE(0.6311106578189371381918993515442277798440422031347194976580945856929268196174737254599050270325373067F32, rayleigh_skewness, 2); // Mathematica: N[2 Sqrt[Pi] (Pi - 3)/((4 - Pi)^(3/2)), 1100]
   CHECK_ULP_CLOSE(2.450893006876380628486604106197544154e-01F32, rayleigh_kurtosis_excess, 2);
   CHECK_ULP_CLOSE(2.68545200106530644530971483548179569382038229399446295305115234555721885953715200280114117493184769799515F32, khinchin, 4 ); // A002210 as a constant https://oeis.org/A002210/constant
   CHECK_ULP_CLOSE(1.2824271291006226368753425688697917277676889273250011F32, glaisher, 4 ); // https://oeis.org/A074962/constant
   CHECK_ULP_CLOSE(4.66920160910299067185320382046620161725F32, first_feigenbaum, 1);
} // template <typename RealType>void test_spots(RealType)

#endif

void test_double_spots()
{
   // Basic sanity checks for constants in boost::math::double_constants::
   // for example: boost::math::double_constants::pi
   // (rather than boost::math::constants::pi<double>() ).
   using namespace boost::math::double_constants;
   BOOST_MATH_STD_USING

   CHECK_ULP_CLOSE(static_cast<double>(3.14159265358979323846264338327950288419716939937510), pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(3.14159265358979323846264338327950288419716939937510)), root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(3.14159265358979323846264338327950288419716939937510/2)), root_half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(3.14159265358979323846264338327950288419716939937510 * 2)), root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(log(4.0))), root_ln_four, 2);
   CHECK_ULP_CLOSE(static_cast<double>(2.71828182845904523536028747135266249775724709369995), e, 2);
   CHECK_ULP_CLOSE(static_cast<double>(0.5), half, 2);
   CHECK_ULP_CLOSE(static_cast<double>(0.57721566490153286060651209008240243104259335), euler, 2);
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(2.0)), root_two, 2);
   CHECK_ULP_CLOSE(static_cast<double>(log(2.0)), ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<double>(log(log(2.0))), ln_ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<double>(1)/3, third, 2);
   CHECK_ULP_CLOSE(static_cast<double>(2)/3, twothirds, 2);
   CHECK_ULP_CLOSE(static_cast<double>(0.14159265358979323846264338327950288419716939937510), pi_minus_three, 2);
   CHECK_ULP_CLOSE(static_cast<double>(4. - 3.14159265358979323846264338327950288419716939937510), four_minus_pi, 2);
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   CHECK_ULP_CLOSE(static_cast<double>(pow((3.14159265358979323846264338327950288419716939937510), 2.71828182845904523536028747135266249775724709369995)), pi_pow_e, 2);
   CHECK_ULP_CLOSE(static_cast<double>(pow((3.14159265358979323846264338327950288419716939937510), 0.33333333333333333333333333333333333333333333333333)), cbrt_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(exp(-0.5)), exp_minus_half, 2);
   CHECK_ULP_CLOSE(static_cast<double>(pow(2.71828182845904523536028747135266249775724709369995, 3.14159265358979323846264338327950288419716939937510)), e_pow_pi, 2);


#else // Only double, so no suffix .
   CHECK_ULP_CLOSE(static_cast<double>(pow((3.14159265358979323846264338327950288419716939937510), 2.71828182845904523536028747135266249775724709369995)), pi_pow_e, 2);
   CHECK_ULP_CLOSE(static_cast<double>(pow((3.14159265358979323846264338327950288419716939937510), 0.33333333333333333333333333333333333333333333333333)), cbrt_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(exp(-0.5)), exp_minus_half, 2);
#endif
   // Rational fractions.
   CHECK_ULP_CLOSE(static_cast<double>(0.333333333333333333333333333333333333333), third, 2);
   CHECK_ULP_CLOSE(static_cast<double>(0.666666666666666666666666666666666666667), two_thirds, 2);
   CHECK_ULP_CLOSE(static_cast<double>(0.75), three_quarters, 2);
   // Two and related.
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(2.)), root_two, 2);
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(3.)), root_three, 2);
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(2.)/2), half_root_two, 2);
   CHECK_ULP_CLOSE(static_cast<double>(log(2.)), ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<double>(log(log(2.0))), ln_ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(log(4.0))), root_ln_four, 2);
   CHECK_ULP_CLOSE(static_cast<double>(1/sqrt(2.0)), one_div_root_two, 2);

   // pi.
   CHECK_ULP_CLOSE(static_cast<double>(3.14159265358979323846264338327950288419716939937510), pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(3.14159265358979323846264338327950288419716939937510/2), half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(3.14159265358979323846264338327950288419716939937510/4), quarter_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(3.14159265358979323846264338327950288419716939937510/3), third_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(3.14159265358979323846264338327950288419716939937510/6), sixth_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(2 * 3.14159265358979323846264338327950288419716939937510), two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(3 * 3.14159265358979323846264338327950288419716939937510 / 4), three_quarters_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(4 * 3.14159265358979323846264338327950288419716939937510 / 3), four_thirds_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(1 / (3.14159265358979323846264338327950288419716939937510)), one_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(2 / (3.14159265358979323846264338327950288419716939937510)), two_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(1 / (2 * 3.14159265358979323846264338327950288419716939937510)), one_div_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(3.14159265358979323846264338327950288419716939937510)), root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(3.14159265358979323846264338327950288419716939937510 / 2)), root_half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(2 * 3.14159265358979323846264338327950288419716939937510)), root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(1 / sqrt(3.14159265358979323846264338327950288419716939937510)), one_div_root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(2 / sqrt(3.14159265358979323846264338327950288419716939937510)), two_div_root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(1 / sqrt(2 * 3.14159265358979323846264338327950288419716939937510)), one_div_root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(sqrt(1. / 3.14159265358979323846264338327950288419716939937510)), root_one_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<double>(3.14159265358979323846264338327950288419716939937510 - 3.), pi_minus_three, 4 * 2 ); // 4 * 2 because of cancellation loss.
   CHECK_ULP_CLOSE(static_cast<double>(4. - 3.14159265358979323846264338327950288419716939937510), four_minus_pi, 4 );
   //
   CHECK_ULP_CLOSE(static_cast<double>(pow((3.14159265358979323846264338327950288419716939937510), 2.71828182845904523536028747135266249775724709369995)), pi_pow_e, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<double>(3.14159265358979323846264338327950288419716939937510 * 3.14159265358979323846264338327950288419716939937510), pi_sqr, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<double>(3.14159265358979323846264338327950288419716939937510 * 3.14159265358979323846264338327950288419716939937510/6), pi_sqr_div_six, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<double>(3.14159265358979323846264338327950288419716939937510 * 3.14159265358979323846264338327950288419716939937510 * 3.14159265358979323846264338327950288419716939937510), pi_cubed, 2);  // See above.

   // CHECK_ULP_CLOSE(static_cast<double>(3.14159265358979323846264338327950288419716939937510 * 3.14159265358979323846264338327950288419716939937510), cbrt_pi, 2);  // See above.
   CHECK_ULP_CLOSE(cbrt_pi * cbrt_pi * cbrt_pi, pi, 2);
   CHECK_ULP_CLOSE((static_cast<double>(1)/cbrt_pi), one_div_cbrt_pi, 2);

   // Euler
   CHECK_ULP_CLOSE(static_cast<double>(2.71828182845904523536028747135266249775724709369995), e, 2);

   //CHECK_ULP_CLOSE(static_cast<double>(exp(-0.5)), exp_minus_half, 2);  // See above.
   CHECK_ULP_CLOSE(pow(e, pi), e_pow_pi, 2); // See also above.
   CHECK_ULP_CLOSE(sqrt(e), root_e, 2);
   CHECK_ULP_CLOSE(log10(e), log10_e, 2);
   CHECK_ULP_CLOSE(static_cast<double>(1)/log10(e), one_div_log10_e, 2);

   // Trigonometric
   CHECK_ULP_CLOSE(pi/180, degree, 2);
   CHECK_ULP_CLOSE(180 / pi, radian, 2);
   CHECK_ULP_CLOSE(sin(1.), sin_one, 2);
   CHECK_ULP_CLOSE(cos(1.), cos_one, 2);
   CHECK_ULP_CLOSE(sinh(1.), sinh_one, 2);
   CHECK_ULP_CLOSE(cosh(1.), cosh_one, 2);

   // Phi
   CHECK_ULP_CLOSE((1. + sqrt(5.)) /2, phi, 2);
   CHECK_ULP_CLOSE(log((1. + sqrt(5.)) /2), ln_phi, 2);
   CHECK_ULP_CLOSE(1. / log((1. + sqrt(5.)) /2), one_div_ln_phi, 2);

   //Euler's Gamma
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992, euler, 2); // (sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(1./ 0.57721566490153286060651209008240243104215933593992, one_div_euler, 2); // (from sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992 * 0.57721566490153286060651209008240243104215933593992, euler_sqr, 2); // (from sequence A001620 in OEIS).

   // Misc
   CHECK_ULP_CLOSE(1.644934066848226436472415166646025189218949901206, zeta_two, 2); // A013661 as a constant (usually base 10) in OEIS.
   CHECK_ULP_CLOSE(1.20205690315959428539973816151144999076498629234049888179227, zeta_three, 2); // (sequence A002117 in OEIS)
   CHECK_ULP_CLOSE(.91596559417721901505460351493238411077414937428167213, catalan, 2); // A006752 as a constant in OEIS.
   CHECK_ULP_CLOSE(1.1395470994046486574927930193898461120875997958365518247216557100852480077060706857071875468869385150, extreme_value_skewness, 2); //  Mathematica: N[12 Sqrt[6]  Zeta[3]/Pi^3, 1101]
   CHECK_ULP_CLOSE(0.6311106578189371381918993515442277798440422031347194976580945856929268196174737254599050270325373067, rayleigh_skewness, 2); // Mathematica: N[2 Sqrt[Pi] (Pi - 3)/((4 - Pi)^(3/2)), 1100]
   CHECK_ULP_CLOSE(2.450893006876380628486604106197544154e-01, rayleigh_kurtosis_excess, 2);
   CHECK_ULP_CLOSE(2.68545200106530644530971483548179569382038229399446295305115234555721885953715200280114117493184769799515, khinchin, 4 ); // A002210 as a constant https://oeis.org/A002210/constant
   CHECK_ULP_CLOSE(1.2824271291006226368753425688697917277676889273250011, glaisher, 4 ); // https://oeis.org/A074962/constant

} // template <typename RealType>void test_spots(RealType)

#ifdef __STDCPP_FLOAT64_T__
void test_f64_spots()
{
   // Basic sanity checks for constants in boost::math::double_constants::
   // for example: boost::math::double_constants::pi
   // (rather than boost::math::constants::pi<double>() ).
   using namespace boost::math::double_constants;
   BOOST_MATH_STD_USING

   CHECK_ULP_CLOSE(static_cast<std::float64_t>(3.14159265358979323846264338327950288419716939937510F64), pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(3.14159265358979323846264338327950288419716939937510F64)), root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(3.14159265358979323846264338327950288419716939937510F64/2)), root_half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(3.14159265358979323846264338327950288419716939937510F64 * 2)), root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(log(4.0F64))), root_ln_four, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(2.71828182845904523536028747135266249775724709369995F64), e, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(0.5F64), half, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(0.57721566490153286060651209008240243104259335F64), euler, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(2.0F64)), root_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(log(2.0F64)), ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(log(log(2.0F64))), ln_ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(1)/3, third, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(2)/3, twothirds, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(0.14159265358979323846264338327950288419716939937510F64), pi_minus_three, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(4.F64 - 3.14159265358979323846264338327950288419716939937510F64), four_minus_pi, 2);
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(pow((3.14159265358979323846264338327950288419716939937510F64), 2.71828182845904523536028747135266249775724709369995F64)), pi_pow_e, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(pow((3.14159265358979323846264338327950288419716939937510F64), 0.33333333333333333333333333333333333333333333333333F64)), cbrt_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(exp(-0.5F64)), exp_minus_half, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(pow(2.71828182845904523536028747135266249775724709369995F64, 3.14159265358979323846264338327950288419716939937510F64)), e_pow_pi, 2);


#else // Only double, so no suffix .
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(pow((3.14159265358979323846264338327950288419716939937510F64), 2.71828182845904523536028747135266249775724709369995F64)), pi_pow_e, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(pow((3.14159265358979323846264338327950288419716939937510F64), 0.33333333333333333333333333333333333333333333333333F64)), cbrt_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(exp(-0.5)), exp_minus_half, 2);
#endif
   // Rational fractions.
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(0.333333333333333333333333333333333333333F64), third, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(0.666666666666666666666666666666666666667F64), two_thirds, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(0.75), three_quarters, 2);
   // Two and related.
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(2.F64)), root_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(3.F64)), root_three, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(2.F64)/2), half_root_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(log(2.F64)), ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(log(log(2.0F64))), ln_ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(log(4.0F64))), root_ln_four, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(1/sqrt(2.0F64)), one_div_root_two, 2);

   // pi.
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(3.14159265358979323846264338327950288419716939937510F64), pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(3.14159265358979323846264338327950288419716939937510F64/2), half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(3.14159265358979323846264338327950288419716939937510F64/4), quarter_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(3.14159265358979323846264338327950288419716939937510F64/3), third_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(3.14159265358979323846264338327950288419716939937510F64/6), sixth_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(2 * 3.14159265358979323846264338327950288419716939937510F64), two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(3 * 3.14159265358979323846264338327950288419716939937510F64 / 4), three_quarters_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(4 * 3.14159265358979323846264338327950288419716939937510F64 / 3), four_thirds_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(1 / (3.14159265358979323846264338327950288419716939937510F64)), one_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(2 / (3.14159265358979323846264338327950288419716939937510F64)), two_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(1 / (2 * 3.14159265358979323846264338327950288419716939937510F64)), one_div_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(3.14159265358979323846264338327950288419716939937510F64)), root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(3.14159265358979323846264338327950288419716939937510F64 / 2)), root_half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(2 * 3.14159265358979323846264338327950288419716939937510F64)), root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(1 / sqrt(3.14159265358979323846264338327950288419716939937510F64)), one_div_root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(2 / sqrt(3.14159265358979323846264338327950288419716939937510F64)), two_div_root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(1 / sqrt(2 * 3.14159265358979323846264338327950288419716939937510F64)), one_div_root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(sqrt(1. / 3.14159265358979323846264338327950288419716939937510F64)), root_one_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(3.14159265358979323846264338327950288419716939937510F64 - 3.F64), pi_minus_three, 4 * 2 ); // 4 * 2 because of cancellation loss.
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(4.F64 - 3.14159265358979323846264338327950288419716939937510F64), four_minus_pi, 4 );
   //
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(pow((3.14159265358979323846264338327950288419716939937510F64), 2.71828182845904523536028747135266249775724709369995F64)), pi_pow_e, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(3.14159265358979323846264338327950288419716939937510F64 * 3.14159265358979323846264338327950288419716939937510F64), pi_sqr, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(3.14159265358979323846264338327950288419716939937510F64 * 3.14159265358979323846264338327950288419716939937510F64/6), pi_sqr_div_six, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(3.14159265358979323846264338327950288419716939937510F64 * 3.14159265358979323846264338327950288419716939937510F64 * 3.14159265358979323846264338327950288419716939937510F64), pi_cubed, 2);  // See above.

   // CHECK_ULP_CLOSE(static_cast<std::float64_t>(3.14159265358979323846264338327950288419716939937510 * 3.14159265358979323846264338327950288419716939937510), cbrt_pi, 2);  // See above.
   CHECK_ULP_CLOSE(cbrt_pi * cbrt_pi * cbrt_pi, pi, 2);
   CHECK_ULP_CLOSE((static_cast<std::float64_t>(1)/cbrt_pi), one_div_cbrt_pi, 2);

   // Euler
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(2.71828182845904523536028747135266249775724709369995), e, 2);

   //CHECK_ULP_CLOSE(static_cast<std::float64_t>(exp(-0.5)), exp_minus_half, 2);  // See above.
   CHECK_ULP_CLOSE(pow(e, pi), e_pow_pi, 2); // See also above.
   CHECK_ULP_CLOSE(sqrt(e), root_e, 2);
   CHECK_ULP_CLOSE(log10(e), log10_e, 2);
   CHECK_ULP_CLOSE(static_cast<std::float64_t>(1)/log10(e), one_div_log10_e, 2);

   // Trigonometric
   CHECK_ULP_CLOSE(pi/180, degree, 2);
   CHECK_ULP_CLOSE(180 / pi, radian, 2);
   CHECK_ULP_CLOSE(sin(1.F64), sin_one, 2);
   CHECK_ULP_CLOSE(cos(1.F64), cos_one, 2);
   CHECK_ULP_CLOSE(sinh(1.F64), sinh_one, 2);
   CHECK_ULP_CLOSE(cosh(1.F64), cosh_one, 2);

   // Phi
   CHECK_ULP_CLOSE((1.F64 + sqrt(5.F64)) /2, phi, 2);
   CHECK_ULP_CLOSE(log((1.F64 + sqrt(5.F64)) /2), ln_phi, 2);
   CHECK_ULP_CLOSE(1.F64 / log((1.F64 + sqrt(5.F64)) /2), one_div_ln_phi, 2);

   //Euler's Gamma
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992F64, euler, 2); // (sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(1.F64/ 0.57721566490153286060651209008240243104215933593992F64, one_div_euler, 2); // (from sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992F64 * 0.57721566490153286060651209008240243104215933593992F64, euler_sqr, 2); // (from sequence A001620 in OEIS).

   // Misc
   CHECK_ULP_CLOSE(1.644934066848226436472415166646025189218949901206F64, zeta_two, 2); // A013661 as a constant (usually base 10) in OEIS.
   CHECK_ULP_CLOSE(1.20205690315959428539973816151144999076498629234049888179227F64, zeta_three, 2); // (sequence A002117 in OEIS)
   CHECK_ULP_CLOSE(.91596559417721901505460351493238411077414937428167213F64, catalan, 2); // A006752 as a constant in OEIS.
   CHECK_ULP_CLOSE(1.1395470994046486574927930193898461120875997958365518247216557100852480077060706857071875468869385150F64, extreme_value_skewness, 2); //  Mathematica: N[12 Sqrt[6]  Zeta[3]/Pi^3, 1101]
   CHECK_ULP_CLOSE(0.6311106578189371381918993515442277798440422031347194976580945856929268196174737254599050270325373067F64, rayleigh_skewness, 2); // Mathematica: N[2 Sqrt[Pi] (Pi - 3)/((4 - Pi)^(3/2)), 1100]
   CHECK_ULP_CLOSE(2.450893006876380628486604106197544154e-01F64, rayleigh_kurtosis_excess, 2);
   CHECK_ULP_CLOSE(2.68545200106530644530971483548179569382038229399446295305115234555721885953715200280114117493184769799515F64, khinchin, 4 ); // A002210 as a constant https://oeis.org/A002210/constant
   CHECK_ULP_CLOSE(1.2824271291006226368753425688697917277676889273250011F64, glaisher, 4 ); // https://oeis.org/A074962/constant

} // template <typename RealType>void test_spots(RealType)
#endif

void test_long_double_spots()
{
   // Basic sanity checks for constants in boost::math::long double_constants::
   // for example: boost::math::long_double_constants::pi
   // (rather than boost::math::constants::pi<long double>() ).

   // All constants are tested here using at least long double precision
   // with independent calculated or listed values,
   // or calculations using long double (sometime a little less accurate).
   using namespace boost::math::long_double_constants;
   BOOST_MATH_STD_USING

   CHECK_ULP_CLOSE(static_cast<long double>(3.14159265358979323846264338327950288419716939937510L), pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(3.14159265358979323846264338327950288419716939937510L)), root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(3.14159265358979323846264338327950288419716939937510L/2)), root_half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(3.14159265358979323846264338327950288419716939937510L * 2)), root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(log(4.0L))), root_ln_four, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(2.71828182845904523536028747135266249775724709369995L), e, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(0.5), half, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(0.57721566490153286060651209008240243104259335L), euler, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(2.0L)), root_two, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(log(2.0L)), ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(log(log(2.0L))), ln_ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(1)/3, third, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(2)/3, twothirds, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(0.14159265358979323846264338327950288419716939937510L), pi_minus_three, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(4.L - 3.14159265358979323846264338327950288419716939937510L), four_minus_pi, 2);
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   CHECK_ULP_CLOSE(static_cast<long double>(pow((3.14159265358979323846264338327950288419716939937510L), 2.71828182845904523536028747135266249775724709369995L)), pi_pow_e, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(pow((3.14159265358979323846264338327950288419716939937510L), 0.33333333333333333333333333333333333333333333333333L)), cbrt_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(exp(-0.5L)), exp_minus_half, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(pow(2.71828182845904523536028747135266249775724709369995L, 3.14159265358979323846264338327950288419716939937510L)), e_pow_pi, 3);


#else // Only double, so no suffix L.
   CHECK_ULP_CLOSE(static_cast<long double>(pow((3.14159265358979323846264338327950288419716939937510), 2.71828182845904523536028747135266249775724709369995)), pi_pow_e, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(pow((3.14159265358979323846264338327950288419716939937510), 0.33333333333333333333333333333333333333333333333333)), cbrt_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(exp(-0.5)), exp_minus_half, 2);
#endif
   // Rational fractions.
   CHECK_ULP_CLOSE(static_cast<long double>(0.333333333333333333333333333333333333333L), third, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(0.666666666666666666666666666666666666667L), two_thirds, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(0.75L), three_quarters, 2);
   // Two and related.
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(2.L)), root_two, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(3.L)), root_three, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(2.L)/2), half_root_two, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(log(2.L)), ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(log(log(2.0L))), ln_ln_two, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(log(4.0L))), root_ln_four, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(1/sqrt(2.0L)), one_div_root_two, 2);

   // pi.
   CHECK_ULP_CLOSE(static_cast<long double>(3.14159265358979323846264338327950288419716939937510L), pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(3.14159265358979323846264338327950288419716939937510L/2), half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(3.14159265358979323846264338327950288419716939937510L/4), quarter_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(3.14159265358979323846264338327950288419716939937510L/3), third_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(3.14159265358979323846264338327950288419716939937510L/6), sixth_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(2 * 3.14159265358979323846264338327950288419716939937510L), two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(3 * 3.14159265358979323846264338327950288419716939937510L / 4), three_quarters_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(4 * 3.14159265358979323846264338327950288419716939937510L / 3), four_thirds_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(1 / (3.14159265358979323846264338327950288419716939937510L)), one_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(2 / (3.14159265358979323846264338327950288419716939937510L)), two_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(1 / (2 * 3.14159265358979323846264338327950288419716939937510L)), one_div_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(3.14159265358979323846264338327950288419716939937510L)), root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(3.14159265358979323846264338327950288419716939937510L / 2)), root_half_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(2 * 3.14159265358979323846264338327950288419716939937510L)), root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(1 / sqrt(3.14159265358979323846264338327950288419716939937510L)), one_div_root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(2 / sqrt(3.14159265358979323846264338327950288419716939937510L)), two_div_root_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(1 / sqrt(2 * 3.14159265358979323846264338327950288419716939937510L)), one_div_root_two_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(sqrt(1. / 3.14159265358979323846264338327950288419716939937510L)), root_one_div_pi, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(3.14159265358979323846264338327950288419716939937510L - 3.L), pi_minus_three, 4 * 4 ); // 4 * 2 because of cancellation loss.
   CHECK_ULP_CLOSE(static_cast<long double>(4.L - 3.14159265358979323846264338327950288419716939937510L), four_minus_pi, 4 );
   //
   CHECK_ULP_CLOSE(static_cast<long double>(pow((3.14159265358979323846264338327950288419716939937510L), 2.71828182845904523536028747135266249775724709369995L)), pi_pow_e, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<long double>(3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L), pi_sqr, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<long double>(3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L/6), pi_sqr_div_six, 2);  // See above.
   CHECK_ULP_CLOSE(static_cast<long double>(3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L), pi_cubed, 2);  // See above.

   // CHECK_ULP_CLOSE(static_cast<long double>(3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L), cbrt_pi, 2);  // See above.
   CHECK_ULP_CLOSE(cbrt_pi * cbrt_pi * cbrt_pi, pi, 2);
   CHECK_ULP_CLOSE((static_cast<long double>(1)/cbrt_pi), one_div_cbrt_pi, 2);

   CHECK_ULP_CLOSE(static_cast<long double>(6.366197723675813430755350534900574481378385829618257E-1L), two_div_pi, 4 * 3);  // 2/pi
   CHECK_ULP_CLOSE(static_cast<long double>(7.97884560802865355879892119868763736951717262329869E-1L), root_two_div_pi, 4 * 3);  //  sqrt(2/pi)

   // Euler
   CHECK_ULP_CLOSE(static_cast<long double>(2.71828182845904523536028747135266249775724709369995L), e, 2);

   //CHECK_ULP_CLOSE(static_cast<long double>(exp(-0.5L)), exp_minus_half, 2);  // See above.
   CHECK_ULP_CLOSE(pow(e, pi), e_pow_pi, 3); // See also above.
   CHECK_ULP_CLOSE(sqrt(e), root_e, 2);
   CHECK_ULP_CLOSE(log10(e), log10_e, 2);
   CHECK_ULP_CLOSE(static_cast<long double>(1)/log10(e), one_div_log10_e, 2);

   // Trigonometric
   CHECK_ULP_CLOSE(pi/180, degree, 2);
   CHECK_ULP_CLOSE(180 / pi, radian, 2);
   CHECK_ULP_CLOSE(sin(1.L), sin_one, 2);
   CHECK_ULP_CLOSE(cos(1.L), cos_one, 2);
   CHECK_ULP_CLOSE(sinh(1.L), sinh_one, 2);
   CHECK_ULP_CLOSE(cosh(1.L), cosh_one, 2);

   // Phi
   CHECK_ULP_CLOSE((1.L + sqrt(5.L)) /2, phi, 2);
   CHECK_ULP_CLOSE(log((1.L + sqrt(5.L)) /2), ln_phi, 2);
   CHECK_ULP_CLOSE(1.L / log((1.L + sqrt(5.L)) /2), one_div_ln_phi, 2);

   //Euler's Gamma
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992L, euler, 2); // (sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(1.L/ 0.57721566490153286060651209008240243104215933593992L, one_div_euler, 2); // (from sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992L * 0.57721566490153286060651209008240243104215933593992L, euler_sqr, 2); // (from sequence A001620 in OEIS).

   // Misc
   CHECK_ULP_CLOSE(1.644934066848226436472415166646025189218949901206L, zeta_two, 2); // A013661 as a constant (usually base 10) in OEIS.
   CHECK_ULP_CLOSE(1.20205690315959428539973816151144999076498629234049888179227L, zeta_three, 2); // (sequence A002117 in OEIS)
   CHECK_ULP_CLOSE(.91596559417721901505460351493238411077414937428167213L, catalan, 2); // A006752 as a constant in OEIS.
   CHECK_ULP_CLOSE(1.1395470994046486574927930193898461120875997958365518247216557100852480077060706857071875468869385150L, extreme_value_skewness, 2); //  Mathematica: N[12 Sqrt[6]  Zeta[3]/Pi^3, 1101]
   CHECK_ULP_CLOSE(0.6311106578189371381918993515442277798440422031347194976580945856929268196174737254599050270325373067L, rayleigh_skewness, 2); // Mathematica: N[2 Sqrt[Pi] (Pi - 3)/((4 - Pi)^(3/2)), 1100]
   CHECK_ULP_CLOSE(2.450893006876380628486604106197544154e-01L, rayleigh_kurtosis_excess, 2);
   CHECK_ULP_CLOSE(2.68545200106530644530971483548179569382038229399446295305115234555721885953715200280114117493184769799515L, khinchin, 4 ); // A002210 as a constant https://oeis.org/A002210/constant
   CHECK_ULP_CLOSE(1.2824271291006226368753425688697917277676889273250011L, glaisher, 4 ); // https://oeis.org/A074962/constant

} // template <typename RealType>void test_spots(RealType)

template <typename Policy>
void test_real_concept_policy(const Policy&)
{
   // Basic sanity checks for constants using real_concept.
   // Parameter Policy is used to control precision.

   using boost::math::concepts::real_concept;
   //typedef typename boost::math::policies::precision<boost::math::concepts::real_concept, boost::math::policies::policy<>>::type t1;
   // A precision of zero means we don't know what the precision of this type is until runtime.
   //std::cout << "Precision for type " << typeid(boost::math::concepts::real_concept).name()  << " is " << t1::value << "." << std::endl;

   using namespace boost::math::constants;
   BOOST_MATH_STD_USING

   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L, (pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sqrt(3.14159265358979323846264338327950288419716939937510L), (root_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sqrt(3.14159265358979323846264338327950288419716939937510L/2), (root_half_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sqrt(3.14159265358979323846264338327950288419716939937510L * 2), (root_two_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sqrt(log(4.0L)), (root_ln_four<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(2.71828182845904523536028747135266249775724709369995L, (e<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(0.5, (half<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104259335L, (euler<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sqrt(2.0L), (root_two<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(log(2.0L), (ln_two<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(log(log(2.0L)), (ln_ln_two<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(static_cast<long double>(1)/3, (third<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(static_cast<long double>(2)/3, (twothirds<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(0.14159265358979323846264338327950288419716939937510L, (pi_minus_three<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(4.L - 3.14159265358979323846264338327950288419716939937510L, (four_minus_pi<real_concept, Policy>)(), 2);
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   CHECK_ULP_CLOSE(pow((3.14159265358979323846264338327950288419716939937510L), 2.71828182845904523536028747135266249775724709369995L), (pi_pow_e<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(pow((3.14159265358979323846264338327950288419716939937510L), 0.33333333333333333333333333333333333333333333333333L), (cbrt_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(exp(-0.5L), (exp_minus_half<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(pow(2.71828182845904523536028747135266249775724709369995L, 3.14159265358979323846264338327950288419716939937510L), (e_pow_pi<real_concept, Policy>)(), 2);


#else // Only double, so no suffix L.
   CHECK_ULP_CLOSE(pow((3.14159265358979323846264338327950288419716939937510), 2.71828182845904523536028747135266249775724709369995), (pi_pow_e<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(pow((3.14159265358979323846264338327950288419716939937510), 0.33333333333333333333333333333333333333333333333333), (cbrt_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(exp(-0.5), (exp_minus_half<real_concept, Policy>)(), 2);
#endif
   // Rational fractions.
   CHECK_ULP_CLOSE(0.333333333333333333333333333333333333333L, (third<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(0.666666666666666666666666666666666666667L, (two_thirds<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(0.75L, (three_quarters<real_concept, Policy>)(), 2);
   // Two and related.
   CHECK_ULP_CLOSE(sqrt(2.L), (root_two<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sqrt(3.L), (root_three<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sqrt(2.L)/2, (half_root_two<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(log(2.L), (ln_two<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(log(log(2.0L)), (ln_ln_two<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sqrt(log(4.0L)), (root_ln_four<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(1/sqrt(2.0L), (one_div_root_two<real_concept, Policy>)(), 2);

   // pi.
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L, (pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L/2, (half_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L/4, (quarter_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L/3, (third_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L/6, (sixth_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(2 * 3.14159265358979323846264338327950288419716939937510L, (two_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(3 * 3.14159265358979323846264338327950288419716939937510L / 4, (three_quarters_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(4 * 3.14159265358979323846264338327950288419716939937510L / 3, (four_thirds_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(1 / (3.14159265358979323846264338327950288419716939937510L), (one_div_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(2 / (3.14159265358979323846264338327950288419716939937510L), (two_div_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(1 / (2 * 3.14159265358979323846264338327950288419716939937510L), (one_div_two_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sqrt(3.14159265358979323846264338327950288419716939937510L), (root_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sqrt(3.14159265358979323846264338327950288419716939937510L / 2), (root_half_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sqrt(2 * 3.14159265358979323846264338327950288419716939937510L), (root_two_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(1 / sqrt(3.14159265358979323846264338327950288419716939937510L), (one_div_root_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(2 / sqrt(3.14159265358979323846264338327950288419716939937510L), (two_div_root_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(1 / sqrt(2 * 3.14159265358979323846264338327950288419716939937510L), (one_div_root_two_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sqrt(1. / 3.14159265358979323846264338327950288419716939937510L), (root_one_div_pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L - 3.L, (pi_minus_three<real_concept, Policy>)(), 4 * 4 ); // 4 * 2 because of cancellation loss.
   CHECK_ULP_CLOSE(4.L - 3.14159265358979323846264338327950288419716939937510L, (four_minus_pi<real_concept, Policy>)(), 4 );
   //
   CHECK_ULP_CLOSE(pow((3.14159265358979323846264338327950288419716939937510L), 2.71828182845904523536028747135266249775724709369995L), (pi_pow_e<real_concept, Policy>)(), 2);  // See above.
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L, (pi_sqr<real_concept, Policy>)(), 2);  // See above.
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L/6, (pi_sqr_div_six<real_concept, Policy>)(), 2);  // See above.
   CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L, (pi_cubed<real_concept, Policy>)(), 2);  // See above.

   // CHECK_ULP_CLOSE(3.14159265358979323846264338327950288419716939937510L * 3.14159265358979323846264338327950288419716939937510L, (cbrt_pi<real_concept, Policy>)(), 2);  // See above.
   CHECK_ULP_CLOSE((cbrt_pi<real_concept, Policy>)() * (cbrt_pi<real_concept, Policy>)() * (cbrt_pi<real_concept, Policy>)(), (pi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE((1)/(cbrt_pi<real_concept, Policy>)(), (one_div_cbrt_pi<real_concept, Policy>)(), 2);

   // Euler
   CHECK_ULP_CLOSE(2.71828182845904523536028747135266249775724709369995L, (e<real_concept, Policy>)(), 2);

   //CHECK_ULP_CLOSE(exp(-0.5L), (exp_minus_half<real_concept, Policy>)(), 2);  // See above.
   CHECK_ULP_CLOSE(pow(e<real_concept, Policy>(), (pi<real_concept, Policy>)()), (e_pow_pi<real_concept, Policy>)(), 2); // See also above.
   CHECK_ULP_CLOSE(sqrt(e<real_concept, Policy>()), (root_e<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(log10(e<real_concept, Policy>()), (log10_e<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(1/log10(e<real_concept, Policy>()), (one_div_log10_e<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE((1/ln_two<real_concept, Policy>()), (log2_e<real_concept, Policy>)(), 2);

   // Trigonometric
   CHECK_ULP_CLOSE((pi<real_concept, Policy>)()/180, (degree<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(180 / (pi<real_concept, Policy>)(), (radian<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sin(1.L), (sin_one<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(cos(1.L), (cos_one<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(sinh(1.L), (sinh_one<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(cosh(1.L), (cosh_one<real_concept, Policy>)(), 2);

   // Phi
   CHECK_ULP_CLOSE((1.L + sqrt(5.L)) /2, (phi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(log((1.L + sqrt(5.L)) /2), (ln_phi<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(1.L / log((1.L + sqrt(5.L)) /2), (one_div_ln_phi<real_concept, Policy>)(), 2);

   //Euler's Gamma
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992L, (euler<real_concept, Policy>)(), 2); // (sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(1.L/ 0.57721566490153286060651209008240243104215933593992L, (one_div_euler<real_concept, Policy>)(), 2); // (from sequence A001620 in OEIS).
   CHECK_ULP_CLOSE(0.57721566490153286060651209008240243104215933593992L * 0.57721566490153286060651209008240243104215933593992L, (euler_sqr<real_concept, Policy>)(), 2); // (from sequence A001620 in OEIS).

   // Misc
   CHECK_ULP_CLOSE(1.644934066848226436472415166646025189218949901206L, (zeta_two<real_concept, Policy>)(), 2); // A013661 as a constant (usually base 10) in OEIS.
   CHECK_ULP_CLOSE(1.20205690315959428539973816151144999076498629234049888179227L, (zeta_three<real_concept, Policy>)(), 2); // (sequence A002117 in OEIS)
   CHECK_ULP_CLOSE(.91596559417721901505460351493238411077414937428167213L, (catalan<real_concept, Policy>)(), 2); // A006752 as a constant in OEIS.
   CHECK_ULP_CLOSE(1.1395470994046486574927930193898461120875997958365518247216557100852480077060706857071875468869385150L, (extreme_value_skewness<real_concept, Policy>)(), 2); //  Mathematica: N[12 Sqrt[6]  Zeta[3]/Pi^3, 1101]
   CHECK_ULP_CLOSE(0.6311106578189371381918993515442277798440422031347194976580945856929268196174737254599050270325373067L, (rayleigh_skewness<real_concept, Policy>)(), 2); // Mathematica: N[2 Sqrt[Pi] (Pi - 3)/((4 - Pi)^(3/2)), 1100]
   CHECK_ULP_CLOSE(2.450893006876380628486604106197544154e-01L, (rayleigh_kurtosis_excess<real_concept, Policy>)(), 2);
   CHECK_ULP_CLOSE(2.68545200106530644530971483548179569382038229399446295305115234555721885953715200280114117493184769799515L, (khinchin<real_concept, Policy>)(), 4 ); // A002210 as a constant https://oeis.org/A002210/constant
   CHECK_ULP_CLOSE(1.2824271291006226368753425688697917277676889273250011L, (glaisher<real_concept, Policy>)(), 4 ); // https://oeis.org/A074962/constant

   //
   // Last of all come the test cases that behave differently if we're calculating the constants on the fly:
   //
   if(boost::math::tools::digits<real_concept>() > boost::math::constants::max_string_digits)
   {
      // This suffers from cancellation error, so increased 4:
      CHECK_ULP_CLOSE((static_cast<real_concept>(4. - 3.14159265358979323846264338327950288419716939937510L)), (four_minus_pi<real_concept, Policy>)(), 4 * 3);
      CHECK_ULP_CLOSE((static_cast<real_concept>(0.14159265358979323846264338327950288419716939937510L)), (pi_minus_three<real_concept, Policy>)(), 4 * 3);
   }
   else
   {
      CHECK_ULP_CLOSE((static_cast<real_concept>(4. - 3.14159265358979323846264338327950288419716939937510L)), (four_minus_pi<real_concept, Policy>)(), 2);
      CHECK_ULP_CLOSE((static_cast<real_concept>(0.14159265358979323846264338327950288419716939937510L)), (pi_minus_three<real_concept, Policy>)(), 2);
   }

} // template <typename boost::math::concepts::real_concept>void test_spots(boost::math::concepts::real_concept)

#ifdef BOOST_MATH_HAS_FLOAT128
void test_float128()
{
   __float128 p = boost::math::constants::pi<__float128>();
   __float128 r = 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651Q;
   CHECK_ULP_CLOSE(boost::multiprecision::float128(p), boost::multiprecision::float128(r), 2);
}
#endif

void test_constexpr()
{
#ifndef BOOST_NO_CXX11_CONSTEXPR
   constexpr float f1 = boost::math::constants::pi<float>();
   constexpr double f2 = boost::math::constants::pi<double>();
   constexpr long double f3 = boost::math::constants::pi<long double>();
   constexpr float fval2 = boost::math::float_constants::pi;
   constexpr double dval2 = boost::math::double_constants::pi;
   constexpr long double ldval2 = boost::math::long_double_constants::pi;
   (void)f1;
   (void)f2;
   (void)f3;
   (void) fval2;
   (void) dval2;
   (void) ldval2;
#ifdef BOOST_MATH_USE_FLOAT128
   constexpr __float128 f4 = boost::math::constants::pi<__float128>();
   (void)f4;
#endif
#endif
}

#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900)
void test_feigenbaum()
{
   // This constant takes weeks to calculate.
   // So if the requested precision > precomputed precision, we need an error message.
   using boost::multiprecision::cpp_bin_float;
   using boost::multiprecision::number;
   auto f64 = boost::math::constants::first_feigenbaum<double>();
   using Real100 = number<cpp_bin_float<300>>;
   auto f = boost::math::constants::first_feigenbaum<Real100>();
   CHECK_ULP_CLOSE(static_cast<double>(f), f64, 0);
   Real100 g{"4.6692016091029906718532038204662016172581855774757686327456513430041343302113147371386897440239480138171659848551898151344086271420279325223124429888908908599449354632367134115324817142199474556443658237932020095610583305754586176522220703854106467494942849814533917262005687556659523398756038256372256480040951071283890611844702775854285419801113440175002428585382498335715522052236087250291678860362674527213399057131606875345083433934446103706309452019115876972432273589838903794946257251289097948986768334611626889116563123474460575179539122045562472807095202198199094558581946136877445617396074115614074243754435499204869180982648652368438702799649017397793425134723808737136211601860128186102056381818354097598477964173900328936171432159878240789776614391395764037760537119096932066998361984288981837003229412030210655743295550388845849737034727532121925706958414074661841981961006129640161487712944415901405467941800198133253378592493365883070459999938375411726563553016862529032210862320550634510679399023341675"};
   CHECK_ULP_CLOSE(f, g, 0);
}

template<typename Real>
void test_plastic()
{
    Real P = boost::math::constants::plastic<Real>();
    Real residual = P*P*P - P -1;
    using std::abs;
    CHECK_LE(abs(residual), 4*std::numeric_limits<Real>::epsilon());
}

template<typename Real>
void test_gauss()
{
    using boost::math::tools::agm;
    using std::sqrt;
    Real G_computed = boost::math::constants::gauss<Real>();
    Real G_expected = Real(1)/agm(sqrt(Real(2)), Real(1));
    CHECK_ULP_CLOSE(G_expected, G_computed, 1);
    CHECK_LE(G_computed, Real(0.8347));
    CHECK_LE(Real(0.8346), G_computed);
}

template<typename Real>
void test_dottie()
{
   using boost::math::constants::dottie;
   using std::cos;
   CHECK_ULP_CLOSE(dottie<Real>(), cos(dottie<Real>()), 1);
}

template<typename Real>
void test_reciprocal_fibonacci()
{
   using boost::math::constants::reciprocal_fibonacci;
   CHECK_LE(reciprocal_fibonacci<Real>(), Real(3.36));
   CHECK_LE(Real(3.35), reciprocal_fibonacci<Real>());
}

template<typename Real>
void test_laplace_limit()
{
   using std::exp;
   using std::sqrt;
   using boost::math::constants::laplace_limit;
   Real ll = laplace_limit<Real>();
   Real tmp = sqrt(1+ll*ll);
   CHECK_ULP_CLOSE(ll*exp(tmp), 1 + tmp, 2);
}

#endif

int main()
{
   // Basic sanity-check spot values.
   #ifdef __STDCPP_FLOAT32_T__
   test_f32_spots();
   #else
   test_float_spots(); // Test float_constants, like boost::math::float_constants::pi;
   #endif
   
   #ifdef __STDCPP_FLOAT64_T__
   test_f64_spots();
   #else
   test_double_spots(); // Test double_constants.
   #endif

   test_long_double_spots(); // Test long_double_constants.

#ifdef BOOST_MATH_HAS_FLOAT128
   test_float128();
#endif
   test_constexpr();

   // (Parameter value, arbitrarily zero, only communicates the floating-point type).
   test_spots(0.0F); // Test float.
   test_spots(0.0); // Test double.
   test_spots(0.0L); // Test long double.
   #ifdef __STDCPP_FLOAT32_T__
   test_spots(0.0F32);
   #endif
   #ifdef __STDCPP_FLOAT64_T__
   test_spots(0.0F64);
   #endif

#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900)
   test_feigenbaum();
   test_plastic<float>();
   test_plastic<double>();
   test_plastic<boost::multiprecision::number<boost::multiprecision::cpp_bin_float<400>>>();
   test_gauss<float>();
   test_gauss<double>();
   test_gauss<long double>();
   test_gauss<boost::multiprecision::number<boost::multiprecision::cpp_bin_float<400>>>();
   test_dottie<float>();
   test_dottie<double>();
   test_dottie<long double>();
   test_dottie<boost::multiprecision::number<boost::multiprecision::cpp_bin_float<400>>>();
   test_laplace_limit<float>();
   test_laplace_limit<double>();
   test_laplace_limit<long double>();
   test_laplace_limit<boost::multiprecision::number<boost::multiprecision::cpp_bin_float<400>>>();
#endif
   return boost::math::test::report_errors();
}
