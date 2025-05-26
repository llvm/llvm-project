//  Copyright John Maddock 2006.
//  Copyright Paul A. Bristow 2007, 2010.
//  Copyright Christopher Kormanyos 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_LIBS_MATH_TEST_INSTANTIATE_HPP
#define BOOST_LIBS_MATH_TEST_INSTANTIATE_HPP

#ifndef BOOST_MATH_ASSERT_UNDEFINED_POLICY
#  define BOOST_MATH_ASSERT_UNDEFINED_POLICY false
#endif

template <class RealType>
struct instantiate_runner_result { static bool value; };

template <class RealType> bool instantiate_runner_result<RealType>::value;

template <class RealType>
struct instantiate_mixed_runner_result { static bool value; };

template <class RealType> bool instantiate_mixed_runner_result<RealType>::value;

#include <boost/math/tools/config.hpp>

#include <boost/math/policies/policy.hpp>

#include <boost/math/special_functions.hpp>
#include <boost/concept_archetype.hpp>
#include <boost/concept_check.hpp>
#include <boost/math/distributions.hpp>

#if !defined(BOOST_MATH_NO_DISTRIBUTION_CONCEPT_TESTS)
#include <boost/math/concepts/distributions.hpp>
#endif

#include <memory>

#ifndef BOOST_MATH_INSTANTIATE_MINIMUM

typedef boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false> > test_policy;

namespace test{

BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS(test_policy)

}

namespace dist_test{

BOOST_MATH_DECLARE_DISTRIBUTIONS(double, test_policy)

}
#endif

#if !defined(TEST_GROUP_1) && !defined(TEST_GROUP_2) && !defined(TEST_GROUP_3) \
   && !defined(TEST_GROUP_4) && !defined(TEST_GROUP_5) && !defined(TEST_GROUP_6) && !defined(TEST_GROUP_7) && !defined(TEST_GROUP_8) && !defined(TEST_GROUP_9) \
   && !defined(TEST_GROUP_10) && !defined(TEST_GROUP_11) && !defined(TEST_GROUP_12) && !defined(TEST_GROUP_13) && !defined(TEST_GROUP_14) && !defined(TEST_GROUP_15)
#  define TEST_GROUP_1
#  define TEST_GROUP_2
#  define TEST_GROUP_3
#  define TEST_GROUP_4
#  define TEST_GROUP_5
#  define TEST_GROUP_6
#  define TEST_GROUP_7
#  define TEST_GROUP_8
#  define TEST_GROUP_9
#  define TEST_GROUP_10
#  define TEST_GROUP_11
#  define TEST_GROUP_12
#  define TEST_GROUP_13
#  define TEST_GROUP_14
#  define TEST_GROUP_15
#endif

template <class RealType>
void instantiate_for_fixed_precision_only(RealType, const std::true_type&)
{
   using namespace boost;
   using namespace boost::math;
   using namespace boost::math::concepts;

#ifdef TEST_GROUP_1
#if !defined(BOOST_MATH_NO_DISTRIBUTION_CONCEPT_TESTS)
   function_requires<DistributionConcept<landau_distribution<RealType> > >();
   function_requires<DistributionConcept<landau_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<dist_test::landau > >();
   function_requires<DistributionConcept<mapairy_distribution<RealType> > >();
   function_requires<DistributionConcept<mapairy_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<dist_test::mapairy > >();
   function_requires<DistributionConcept<holtsmark_distribution<RealType> > >();
   function_requires<DistributionConcept<holtsmark_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<dist_test::holtsmark> >();
   function_requires<DistributionConcept<saspoint5_distribution<RealType> > >();
   function_requires<DistributionConcept<saspoint5_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<dist_test::saspoint5> >();
#endif
#endif
}
template <class RealType>
void instantiate_for_fixed_precision_only(RealType, const std::false_type&){}

template <class RealType>
void instantiate(RealType)
{
   instantiate_runner_result<RealType>::value = false;

   using namespace boost;
   using namespace boost::math;
   using namespace boost::math::concepts;

#ifdef TEST_GROUP_1
   #if !defined(BOOST_MATH_NO_DISTRIBUTION_CONCEPT_TESTS)
   function_requires<DistributionConcept<arcsine_distribution<RealType> > >();
   function_requires<DistributionConcept<bernoulli_distribution<RealType> > >();
   function_requires<DistributionConcept<beta_distribution<RealType> > >();
   function_requires<DistributionConcept<binomial_distribution<RealType> > >();
   function_requires<DistributionConcept<cauchy_distribution<RealType> > >();
   function_requires<DistributionConcept<chi_squared_distribution<RealType> > >();
   function_requires<DistributionConcept<exponential_distribution<RealType> > >();
   function_requires<DistributionConcept<extreme_value_distribution<RealType> > >();
   function_requires<DistributionConcept<fisher_f_distribution<RealType> > >();
   function_requires<DistributionConcept<gamma_distribution<RealType> > >();
   function_requires<DistributionConcept<geometric_distribution<RealType> > >();
   function_requires<DistributionConcept<hypergeometric_distribution<RealType> > >();
   function_requires<DistributionConcept<hyperexponential_distribution<RealType> > >();
   function_requires<DistributionConcept<inverse_chi_squared_distribution<RealType> > >();
   function_requires<DistributionConcept<inverse_gamma_distribution<RealType> > >();
   function_requires<DistributionConcept<inverse_gaussian_distribution<RealType> > >();
   function_requires<DistributionConcept<kolmogorov_smirnov_distribution<RealType> > >();
   function_requires<DistributionConcept<laplace_distribution<RealType> > >();
   function_requires<DistributionConcept<logistic_distribution<RealType> > >();
   function_requires<DistributionConcept<lognormal_distribution<RealType> > >();
   function_requires<DistributionConcept<negative_binomial_distribution<RealType> > >();
   function_requires<DistributionConcept<non_central_chi_squared_distribution<RealType> > >();
   function_requires<DistributionConcept<non_central_beta_distribution<RealType> > >();
   function_requires<DistributionConcept<non_central_f_distribution<RealType> > >();
   function_requires<DistributionConcept<non_central_t_distribution<RealType> > >();
   function_requires<DistributionConcept<normal_distribution<RealType> > >();
   function_requires<DistributionConcept<pareto_distribution<RealType> > >();
   function_requires<DistributionConcept<poisson_distribution<RealType> > >();
   function_requires<DistributionConcept<rayleigh_distribution<RealType> > >();
   function_requires<DistributionConcept<students_t_distribution<RealType> > >();
   function_requires<DistributionConcept<skew_normal_distribution<RealType> > >();
   function_requires<DistributionConcept<triangular_distribution<RealType> > >();
   function_requires<DistributionConcept<uniform_distribution<RealType> > >();
   function_requires<DistributionConcept<weibull_distribution<RealType> > >();

   instantiate_for_fixed_precision_only(RealType(), std::integral_constant<bool, std::numeric_limits<RealType>::is_specialized && (std::numeric_limits<RealType>::digits <= 113) && (std::numeric_limits<RealType>::radix == 2)>());

   #endif // !defined(BOOST_MATH_NO_DISTRIBUTION_CONCEPT_TESTS)
#endif
#ifndef BOOST_MATH_INSTANTIATE_MINIMUM
#ifdef TEST_GROUP_2
   #if !defined(BOOST_MATH_NO_DISTRIBUTION_CONCEPT_TESTS)
   function_requires<DistributionConcept<arcsine_distribution<RealType> > >();
   function_requires<DistributionConcept<bernoulli_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<beta_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<binomial_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<cauchy_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<chi_squared_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<exponential_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<extreme_value_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<fisher_f_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<gamma_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<geometric_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<hypergeometric_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<inverse_chi_squared_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<inverse_gamma_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<inverse_gaussian_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<kolmogorov_smirnov_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<laplace_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<logistic_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<lognormal_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<negative_binomial_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<non_central_chi_squared_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<non_central_beta_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<non_central_f_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<non_central_t_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<normal_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<pareto_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<poisson_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<rayleigh_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<skew_normal_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<students_t_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<triangular_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<uniform_distribution<RealType, test_policy> > >();
   function_requires<DistributionConcept<weibull_distribution<RealType, test_policy> > >();
   #endif // !defined(BOOST_MATH_NO_DISTRIBUTION_CONCEPT_TESTS)
#endif
#ifdef TEST_GROUP_3
   #if !defined(BOOST_MATH_NO_DISTRIBUTION_CONCEPT_TESTS)
   function_requires<DistributionConcept<dist_test::arcsine > >();
   function_requires<DistributionConcept<dist_test::bernoulli > >();
   function_requires<DistributionConcept<dist_test::beta > >();
   function_requires<DistributionConcept<dist_test::binomial > >();
   function_requires<DistributionConcept<dist_test::cauchy > >();
   function_requires<DistributionConcept<dist_test::chi_squared > >();
   function_requires<DistributionConcept<dist_test::exponential > >();
   function_requires<DistributionConcept<dist_test::extreme_value > >();
   function_requires<DistributionConcept<dist_test::fisher_f > >();
   function_requires<DistributionConcept<dist_test::gamma > >();
   function_requires<DistributionConcept<dist_test::geometric > >();
   function_requires<DistributionConcept<dist_test::hypergeometric > >();
   function_requires<DistributionConcept<dist_test::inverse_chi_squared > >();
   function_requires<DistributionConcept<dist_test::inverse_gamma > >();
   function_requires<DistributionConcept<dist_test::inverse_gaussian > >();
   function_requires<DistributionConcept<dist_test::laplace > >();
   function_requires<DistributionConcept<dist_test::logistic > >();
   function_requires<DistributionConcept<dist_test::lognormal > >();
   function_requires<DistributionConcept<dist_test::negative_binomial > >();
   function_requires<DistributionConcept<dist_test::non_central_chi_squared > >();
   function_requires<DistributionConcept<dist_test::non_central_beta > >();
   function_requires<DistributionConcept<dist_test::non_central_f > >();
   function_requires<DistributionConcept<dist_test::non_central_t > >();
   function_requires<DistributionConcept<dist_test::normal > >();
   function_requires<DistributionConcept<dist_test::pareto > >();
   function_requires<DistributionConcept<dist_test::poisson > >();
   function_requires<DistributionConcept<dist_test::rayleigh > >();
   function_requires<DistributionConcept<dist_test::students_t > >();
   function_requires<DistributionConcept<dist_test::triangular > >();
   function_requires<DistributionConcept<dist_test::uniform > >();
   function_requires<DistributionConcept<dist_test::weibull > >();
   function_requires<DistributionConcept<dist_test::hypergeometric > >();
   #endif // !defined(BOOST_MATH_NO_DISTRIBUTION_CONCEPT_TESTS)
#endif
#endif
   int i { 1 };

   static_cast<void>(i);

   auto v1(static_cast<RealType>(0.51));
   auto v2(static_cast<RealType>(0.5));
   auto v3(static_cast<RealType>(0.5));
   boost::detail::dummy_constructor dc;
   boost::output_iterator_archetype<RealType> oi(dc);
#ifdef TEST_GROUP_4
   boost::math::tgamma(v1);
   boost::math::tgamma1pm1(v1);
   boost::math::lgamma(v1);
   {
      int i_other { i };

      boost::math::lgamma(v1, &i_other);
   }
   boost::math::digamma(v1);
   boost::math::trigamma(v1);
   boost::math::polygamma(i, v1);
   boost::math::tgamma_ratio(v1, v2);
   boost::math::tgamma_delta_ratio(v1, v2);
   boost::math::factorial<RealType>(i);
   boost::math::unchecked_factorial<RealType>(i);
   {
      const auto i_fact = boost::math::max_factorial<RealType>::value;

      boost::math::double_factorial<RealType>(i_fact);
      boost::math::rising_factorial(v1, i_fact);
      boost::math::falling_factorial(v1, i_fact);
   }
   i = 4;
   boost::math::tgamma(v1, v2);
   boost::math::tgamma_lower(v1, v2);
   boost::math::gamma_p(v1, v2);
   boost::math::gamma_q(v1, v2);
   boost::math::gamma_p_inv(v1, v2);
   boost::math::gamma_q_inv(v1, v2);
   boost::math::gamma_p_inva(v1, v2);
   boost::math::gamma_q_inva(v1, v2);
   boost::math::erf(v1);
   boost::math::erfc(v1);
   boost::math::erf_inv(v1);
   boost::math::erfc_inv(v1);
   boost::math::beta(v1, v2);
   boost::math::beta(v1, v2, v3);
   boost::math::betac(v1, v2, v3);
   boost::math::ibeta(v1, v2, v3);
   boost::math::ibetac(v1, v2, v3);
   boost::math::ibeta_inv(v1, v2, v3);
   boost::math::ibetac_inv(v1, v2, v3);
   boost::math::ibeta_inva(v1, v2, v3);
   boost::math::ibetac_inva(v1, v2, v3);
   boost::math::ibeta_invb(v1, v2, v3);
   boost::math::ibetac_invb(v1, v2, v3);
   boost::math::gamma_p_derivative(v2, v3);
   boost::math::ibeta_derivative(v1, v2, v3);
   boost::math::binomial_coefficient<RealType>(i, i);
   (boost::math::fpclassify)(v1);
   (boost::math::isfinite)(v1);
   (boost::math::isnormal)(v1);
   (boost::math::isnan)(v1);
   (boost::math::isinf)(v1);
   (boost::math::signbit)(v1);
   (boost::math::copysign)(v1, v2);
   (boost::math::changesign)(v1);
   (boost::math::sign)(v1);
   boost::math::log1p(v1);
   boost::math::expm1(v1);
   boost::math::cbrt(v1);
   boost::math::sqrt1pm1(v1);
   boost::math::powm1(v1, v2);
#endif
#ifdef TEST_GROUP_5
   boost::math::legendre_p(1, v1);
   boost::math::legendre_p(1, 0, v1);
   boost::math::legendre_q(1, v1);
   boost::math::legendre_p_prime(1, v1);
   boost::math::legendre_next(2, v1, v2, v3);
   boost::math::legendre_next(2, 2, v1, v2, v3);
   boost::math::laguerre(1, v1);
   boost::math::laguerre(2, 1, v1);
   boost::math::laguerre(2u, 1u, v1);
   boost::math::laguerre_next(2, v1, v2, v3);
   boost::math::laguerre_next(2, 1, v1, v2, v3);
   boost::math::hermite(1, v1);
   boost::math::hermite_next(2, v1, v2, v3);
   boost::math::chebyshev_next(v1, v2, v3);
   boost::math::chebyshev_t(1, v1);
   boost::math::chebyshev_u(1, v1);
   boost::math::chebyshev_t_prime(1, v1);
   {
      const RealType v1_other_const = v1;

      boost::math::chebyshev_clenshaw_recurrence(&v1_other_const, 0, v2);
   }
   boost::math::spherical_harmonic_r(2, 1, v1, v2);
   boost::math::spherical_harmonic_i(2, 1, v1, v2);
   boost::math::ellint_1(v1);
   boost::math::ellint_1(v1, v2);
   boost::math::ellint_2(v1);
   boost::math::ellint_2(v1, v2);
   boost::math::ellint_3(v1, v2);
   boost::math::ellint_3(v1, v2, v3);
   boost::math::ellint_d(v1);
   boost::math::ellint_d(v1, v2);
   boost::math::jacobi_zeta(v1, v2);
   boost::math::heuman_lambda(v1, v2);
   boost::math::ellint_rc(v1, v2);
   boost::math::ellint_rd(v1, v2, v3);
   boost::math::ellint_rf(v1, v2, v3);
   boost::math::ellint_rg(v1, v2, v3);
   boost::math::ellint_rj(v1, v2, v3, v1);
   {
      RealType v1_other { };
      boost::math::jacobi_elliptic(v1, v2, &v1_other, &v2);
   }
   boost::math::jacobi_cd(v1, v2);
   boost::math::jacobi_cn(v1, v2);
   boost::math::jacobi_cs(v1, v2);
   boost::math::jacobi_dc(v1, v2);
   boost::math::jacobi_dn(v1, v2);
   boost::math::jacobi_ds(v1, v2);
   boost::math::jacobi_nc(v1, v2);
   boost::math::jacobi_nd(v1, v2);
   boost::math::jacobi_ns(v1, v2);
   boost::math::jacobi_sc(v1, v2);
   boost::math::jacobi_sd(v1, v2);
   boost::math::jacobi_sn(v1, v2);
   boost::math::jacobi_theta1(v1, v2);
   boost::math::jacobi_theta1tau(v1, v2);
   boost::math::jacobi_theta2(v1, v2);
   boost::math::jacobi_theta2tau(v1, v2);
   boost::math::jacobi_theta3(v1, v2);
   boost::math::jacobi_theta3tau(v1, v2);
   boost::math::jacobi_theta3m1(v1, v2);
   boost::math::jacobi_theta3m1tau(v1, v2);
   boost::math::jacobi_theta4(v1, v2);
   boost::math::jacobi_theta4tau(v1, v2);
   boost::math::jacobi_theta4m1(v1, v2);
   boost::math::jacobi_theta4m1tau(v1, v2);
   boost::math::hypot(v1, v2);
   boost::math::sinc_pi(v1);
   boost::math::sinhc_pi(v1);
   boost::math::asinh(v1);
   {
      auto v_special = static_cast<RealType>(1.55L);

      boost::math::acosh(v_special);
   }
   boost::math::atanh(v1);
   boost::math::sin_pi(v1);
   boost::math::cos_pi(v1);
#endif
#ifdef TEST_GROUP_6
   boost::math::cyl_neumann(v1, v2);
   const int i_small = 3;
   i = ((i > i_small) ? i_small : i);
   boost::math::cyl_neumann(i, v2);
   boost::math::cyl_bessel_j(v1, v2);
   boost::math::cyl_bessel_j(i, v2);
   boost::math::cyl_bessel_i(v1, v2);
   boost::math::cyl_bessel_i(i, v2);
   boost::math::cyl_bessel_k(v1, v2);
   boost::math::cyl_bessel_k(i, v2);
   boost::math::sph_bessel(i, v2);
   boost::math::sph_bessel(i, 1);
   boost::math::sph_neumann(i, v2);
   boost::math::sph_neumann(i, i);
   boost::math::cyl_neumann_prime(v1, v2);
   boost::math::cyl_neumann_prime(i, v2);
   boost::math::cyl_bessel_j_prime(v1, v2);
   boost::math::cyl_bessel_j_prime(i, v2);
   boost::math::cyl_bessel_i_prime(v1, v2);
   boost::math::cyl_bessel_i_prime(i, v2);
   boost::math::cyl_bessel_k_prime(v1, v2);
   boost::math::cyl_bessel_k_prime(i, v2);
   boost::math::sph_bessel_prime(i, v2);
   boost::math::sph_bessel_prime(i, 1);
   boost::math::sph_neumann_prime(i, v2);
   boost::math::sph_neumann_prime(i, i);
   boost::math::cyl_bessel_j_zero(v1, i);
   boost::math::cyl_bessel_j_zero(v1, i, i, oi);
   boost::math::cyl_neumann_zero(v1, i);
   boost::math::cyl_neumann_zero(v1, i, i, oi);
   boost::math::lambert_w0(v1);
   {
      auto v_special = static_cast<RealType>(0.234L);

      boost::math::lambert_wm1(-v_special);
   }
   boost::math::lambert_w0_prime(v1);
#ifdef TEST_COMPLEX
   boost::math::cyl_hankel_1(v1, v2);
   boost::math::cyl_hankel_1(i, v2);
   boost::math::cyl_hankel_2(v1, v2);
   boost::math::cyl_hankel_2(i, v2);
   boost::math::sph_hankel_1(v1, v2);
   boost::math::sph_hankel_1(i, v2);
   boost::math::sph_hankel_2(v1, v2);
   boost::math::sph_hankel_2(i, v2);
#endif
#endif
#ifdef TEST_GROUP_7
   boost::math::airy_ai(v1);
   boost::math::airy_bi(v1);
   boost::math::airy_ai_prime(v1);
   boost::math::airy_bi_prime(v1);

   boost::math::airy_ai_zero<RealType>(i);
   boost::math::airy_bi_zero<RealType>(i);
   boost::math::airy_ai_zero<RealType>(i, i, oi);
   boost::math::airy_bi_zero<RealType>(i, i, oi);

   boost::math::hypergeometric_1F0(v1, v2);
   boost::math::hypergeometric_0F1(v1, v2);
   boost::math::hypergeometric_2F0(RealType(-3), v2, v3);
#if !defined(BOOST_NO_CXX11_AUTO_DECLARATIONS) && !defined(BOOST_NO_CXX11_LAMBDAS) && !defined(BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX) && !defined(BOOST_NO_CXX11_HDR_TUPLE)
   boost::math::hypergeometric_1F1(v1, v2, v3);
#ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
   boost::math::hypergeometric_pFq({ v1 }, { v2 }, v3);
#endif
#endif

   boost::math::expint(v1);
   boost::math::expint(i);
   boost::math::expint(i, v2);
   boost::math::expint(i, i);
   boost::math::zeta(v1);
   boost::math::zeta(i);
   boost::math::owens_t(v1, v2);
   boost::math::trunc(v1);
   boost::math::itrunc(v1);
   boost::math::ltrunc(v1);
   boost::math::round(v1);
   boost::math::iround(v1);
   boost::math::lround(v1);
   {
      RealType v1_other { };

      boost::math::modf(v1, &v1_other);
   }
   {
      int i_other { };

      boost::math::modf(v1, &i_other);
   }
   {
      long l_other { };

      boost::math::modf(v1, &l_other);
   }

   boost::math::lltrunc(v1);
   boost::math::llround(v1);
   {
      long long ll_other { };

      boost::math::modf(v1, &ll_other);
   }

   boost::math::pow<2>(v1);
   boost::math::nextafter(v1, v1);
   boost::math::float_next(v1);
   boost::math::float_prior(v1);
   boost::math::float_distance(v1, v1);
   boost::math::ulp(v1);
   boost::math::relative_difference(v1, v2);
   boost::math::epsilon_difference(v1, v2);

   boost::math::unchecked_bernoulli_b2n<RealType>(i);
   boost::math::bernoulli_b2n<RealType>(i);
   {
      auto v1_array = std::make_unique<RealType[]>(i);

      boost::math::bernoulli_b2n<RealType>(i, i, &v1_array[0U]);
   }
   boost::math::tangent_t2n<RealType>(i);
   {
      auto v1_array = std::make_unique<RealType[]>(i);

      boost::math::tangent_t2n<RealType>(i, i, &v1_array[0U]);
   }

#endif
#ifdef TEST_GROUP_12
   //
   // Over again, but arguments may be expression templates:
   //
   boost::math::tgamma(v1 + 0);
   boost::math::tgamma1pm1(v1 + 0);
   boost::math::lgamma(v1 * 1);
   {
      int i_other { i };

      boost::math::lgamma(v1 * 1, &i_other);
   }
   boost::math::digamma(v1 * 1);
   boost::math::trigamma(v1 * 1);
   boost::math::polygamma(i, v1 * 1);
   boost::math::tgamma_ratio(v1 * 1, v2 + 0);
   boost::math::tgamma_delta_ratio(v1 * 1, v2 + 0);
   boost::math::factorial<RealType>(i);
   boost::math::unchecked_factorial<RealType>(i);
   {
      const auto i_fact = boost::math::max_factorial<RealType>::value;
      boost::math::double_factorial<RealType>(i_fact);
      boost::math::rising_factorial(v1 * 1, i_fact);
      boost::math::falling_factorial(v1 * 1, i_fact);
   }
   boost::math::tgamma(v1 * 1, v2 + 0);
   boost::math::tgamma_lower(v1 * 1, v2 - 0);
   boost::math::gamma_p(v1 * 1, v2 + 0);
   boost::math::gamma_q(v1 * 1, v2 + 0);
   boost::math::gamma_p_inv(v1 * 1, v2 + 0);
   boost::math::gamma_q_inv(v1 * 1, v2 + 0);
   boost::math::gamma_p_inva(v1 * 1, v2 + 0);
   boost::math::gamma_q_inva(v1 * 1, v2 + 0);
   boost::math::erf(v1 * 1);
   boost::math::erfc(v1 * 1);
   boost::math::erf_inv(v1 * 1);
   boost::math::erfc_inv(v1 * 1);
   boost::math::beta(v1 * 1, v2 + 0);
   boost::math::beta(v1 * 1, v2 + 0, v3 / 1);
   boost::math::betac(v1 * 1, v2 + 0, v3 / 1);
   boost::math::ibeta(v1 * 1, v2 + 0, v3 / 1);
   boost::math::ibetac(v1 * 1, v2 + 0, v3 / 1);
   boost::math::ibeta_inv(v1 * 1, v2 + 0, v3 / 1);
   boost::math::ibetac_inv(v1 * 1, v2 + 0, v3 / 1);
   boost::math::ibeta_inva(v1 * 1, v2 + 0, v3 / 1);
   boost::math::ibetac_inva(v1 * 1, v2 + 0, v3 / 1);
   boost::math::ibeta_invb(v1 * 1, v2 + 0, v3 / 1);
   boost::math::ibetac_invb(v1 * 1, v2 + 0, v3 / 1);
   boost::math::gamma_p_derivative(v2 * 1, v3 + 0);
   boost::math::ibeta_derivative(v1 * 1, v2 + 0, v3 / 1);
   (boost::math::fpclassify)(v1 * 1);
   (boost::math::isfinite)(v1 * 1);
   (boost::math::isnormal)(v1 * 1);
   (boost::math::isnan)(v1 * 1);
   (boost::math::isinf)(v1 * 1);
   (boost::math::signbit)(v1 * 1);
   (boost::math::copysign)(v1 * 1, v2 + 0);
   (boost::math::changesign)(v1 * 1);
   (boost::math::sign)(v1 * 1);
   boost::math::log1p(v1 * 1);
   boost::math::expm1(v1 * 1);
   boost::math::cbrt(v1 * 1);
   boost::math::sqrt1pm1(v1 * 1);
   boost::math::powm1(v1 * 1, v2 + 0);
#endif
#ifdef TEST_GROUP_13
   boost::math::legendre_p(1, v1 * 1);
   boost::math::legendre_p(1, 0, v1 * 1);
   boost::math::legendre_p_prime(1, v1 * 1);
   boost::math::legendre_q(1, v1 * 1);
   boost::math::legendre_next(2, v1 * 1, v2 + 0, v3 / 1);
   boost::math::legendre_next(2, 2, v1 * 1, v2 + 0, v3 / 1);
   boost::math::laguerre(1, v1 * 1);
   boost::math::laguerre(2, 1, v1 * 1);
   boost::math::laguerre(2u, 1u, v1 * 1);
   boost::math::laguerre_next(2, v1 * 1, v2 + 0, v3 / 1);
   boost::math::laguerre_next(2, 1, v1 * 1, v2 + 0, v3 / 1);
   boost::math::hermite(1, v1 * 1);
   boost::math::hermite_next(2, v1 * 1, v2 + 0, v3 / 1);
   boost::math::chebyshev_next(2 * v1, 1 + v2, 3 * v3);
   boost::math::chebyshev_t(1, 2 * v1);
   boost::math::chebyshev_u(1, 2 * v1);
   boost::math::chebyshev_t_prime(1, 2 * v1);
   {
      const RealType v1_other_const = v1;

      boost::math::chebyshev_clenshaw_recurrence(&v1_other_const, 0, 2 * v2);
   }
   boost::math::spherical_harmonic_r(2, 1, v1 * 1, v2 + 0);
   boost::math::spherical_harmonic_i(2, 1, v1 * 1, v2 + 0);
   boost::math::ellint_1(v1 * 1);
   boost::math::ellint_1(v1 * 1, v2 + 0);
   boost::math::ellint_2(v1 * 1);
   boost::math::ellint_2(v1 * 1, v2 + 0);
   boost::math::ellint_3(v1 * 1, v2 + 0);
   boost::math::ellint_3(v1 * 1, v2 + 0, v3 / 1);
   boost::math::ellint_rc(v1 * 1, v2 + 0);
   boost::math::ellint_rd(v1 * 1, v2 + 0, v3 / 1);
   boost::math::ellint_rf(v1 * 1, v2 + 0, v3 / 1);
   boost::math::ellint_rg(v1 * 1, v2 + 0, v3 / 1);
   boost::math::ellint_rj(v1 * 1, v2 + 0, v3 / 1, v1 * 1);
   boost::math::ellint_d(v1 * 1);
   boost::math::ellint_d(v1 * 1, v2 + 0);
   boost::math::jacobi_zeta(v1 * 1, v2 + 0);
   boost::math::heuman_lambda(v1 * 1, v2 + 0);
   {
      RealType v1_to_get { v1 };

      boost::math::jacobi_elliptic(v1 * 1, v2 + 0, &v1_to_get, &v2);
   }
   boost::math::jacobi_cd(v1 * 1, v2 + 0);
   boost::math::jacobi_cn(v1 * 1, v2 + 0);
   boost::math::jacobi_cs(v1 * 1, v2 + 0);
   boost::math::jacobi_dc(v1 * 1, v2 + 0);
   boost::math::jacobi_dn(v1 * 1, v2 + 0);
   boost::math::jacobi_ds(v1 * 1, v2 + 0);
   boost::math::jacobi_nc(v1 * 1, v2 + 0);
   boost::math::jacobi_nd(v1 * 1, v2 + 0);
   boost::math::jacobi_ns(v1 * 1, v2 + 0);
   boost::math::jacobi_sc(v1 * 1, v2 + 0);
   boost::math::jacobi_sd(v1 * 1, v2 + 0);
   boost::math::jacobi_sn(v1 * 1, v2 + 0);
   boost::math::jacobi_theta1(v1 * 1, v2 + 0);
   boost::math::jacobi_theta1tau(v1 * 1, v2 + 0);
   boost::math::jacobi_theta2(v1 * 1, v2 + 0);
   boost::math::jacobi_theta2tau(v1 * 1, v2 + 0);
   boost::math::jacobi_theta3(v1 * 1, v2 + 0);
   boost::math::jacobi_theta3tau(v1 * 1, v2 + 0);
   boost::math::jacobi_theta3m1(v1 * 1, v2 + 0);
   boost::math::jacobi_theta3m1tau(v1 * 1, v2 + 0);
   boost::math::jacobi_theta4(v1 * 1, v2 + 0);
   boost::math::jacobi_theta4tau(v1 * 1, v2 + 0);
   boost::math::jacobi_theta4m1(v1 * 1, v2 + 0);
   boost::math::jacobi_theta4m1tau(v1 * 1, v2 + 0);
   boost::math::hypot(v1 * 1, v2 + 0);
   boost::math::sinc_pi(v1 * 1);
   boost::math::sinhc_pi(v1 * 1);
   boost::math::asinh(v1 * 1);
   {
      auto v_special = static_cast<RealType>(1.55L);

      boost::math::acosh(v_special * 1);
   }
   boost::math::atanh(v1 * 1);
   boost::math::sin_pi(v1 * 1);
   boost::math::cos_pi(v1 * 1);
#endif
#ifdef TEST_GROUP_14
   boost::math::cyl_neumann(v1 * 1, v2 + 0);
   boost::math::cyl_neumann(i, v2 * 1);
   boost::math::cyl_bessel_j(v1 * 1, v2 + 0);
   boost::math::cyl_bessel_j(i, v2 * 1);
   boost::math::cyl_bessel_i(v1 * 1, v2 + 0);
   boost::math::cyl_bessel_i(i, v2 * 1);
   boost::math::cyl_bessel_k(v1 * 1, v2 + 0);
   boost::math::cyl_bessel_k(i, v2 * 1);
   boost::math::sph_bessel(i, v2 * 1);
   boost::math::sph_bessel(i, 1);
   boost::math::sph_neumann(i, v2 * 1);
   boost::math::sph_neumann(i, i);
   boost::math::cyl_neumann_prime(v1 * 1, v2 + 0);
   boost::math::cyl_neumann_prime(i, v2 * 1);
   boost::math::cyl_bessel_j_prime(v1 * 1, v2 + 0);
   boost::math::cyl_bessel_j_prime(i, v2 * 1);
   boost::math::cyl_bessel_i_prime(v1 * 1, v2 + 0);
   boost::math::cyl_bessel_i_prime(i, v2 * 1);
   boost::math::cyl_bessel_k_prime(v1 * 1, v2 + 0);
   boost::math::cyl_bessel_k_prime(i, v2 * 1);
   boost::math::sph_bessel_prime(i, v2 * 1);
   boost::math::sph_bessel_prime(i, 1);
   boost::math::sph_neumann_prime(i, v2 * 1);
   boost::math::sph_neumann_prime(i, i);
   boost::math::cyl_bessel_j_zero(v1 * 1, i);
   boost::math::cyl_bessel_j_zero(v1 * 1, i, i, oi);
   boost::math::cyl_neumann_zero(v1 * 1, i);
   boost::math::cyl_neumann_zero(v1 * 1, i, i, oi);
   boost::math::lambert_w0(v1 * 1);
   {
      auto v_special = static_cast<RealType>(-0.234L);

      boost::math::lambert_wm1(v_special * 1);
   }
   boost::math::lambert_w0_prime(v1 * 1);
#ifdef TEST_COMPLEX
   boost::math::cyl_hankel_1(v1, v2);
   boost::math::cyl_hankel_1(i, v2);
   boost::math::cyl_hankel_2(v1, v2);
   boost::math::cyl_hankel_2(i, v2);
   boost::math::sph_hankel_1(v1, v2);
   boost::math::sph_hankel_1(i, v2);
   boost::math::sph_hankel_2(v1, v2);
   boost::math::sph_hankel_2(i, v2);
#endif
#endif
#ifdef TEST_GROUP_15
   boost::math::airy_ai(v1 * 1);
   boost::math::airy_bi(v1 * 1);
   boost::math::airy_ai_prime(v1 * 1);
   boost::math::airy_bi_prime(v1 * 1);
   boost::math::expint(v1 * 1);
   boost::math::expint(i);
   boost::math::expint(i, v2 * 1);
   boost::math::expint(i, i);
   boost::math::zeta(v1 * 1);
   boost::math::zeta(i);
   boost::math::owens_t(v1 * 1, v2 + 0);
   boost::math::trunc(v1 * 1);
   boost::math::itrunc(v1 * 1);
   boost::math::ltrunc(v1 * 1);
   boost::math::round(v1 * 1);
   boost::math::iround(v1 * 1);
   boost::math::lround(v1 * 1);
   {
      RealType v1_other { };
      RealType v1_1 { v1 * 1 };

      boost::math::modf(v1_1, &v1_other);
   }
   {
      int i_other { };
      RealType v1_1 { v1 * 1 };

      boost::math::modf(v1_1, &i_other);
   }
   {
      long l_other { };
      RealType v1_1 { v1 * 1 };

      boost::math::modf(v1_1, &l_other);
   }

   boost::math::lltrunc(v1 * 1);
   boost::math::llround(v1 * 1);
   {
      long long ll_other { };
      RealType v1_1 { v1 * 1 };

      boost::math::modf(v1_1, &ll_other);
   }

   boost::math::pow<2>(v1 * 1);
   boost::math::nextafter(v1 * 1, v1 + 0);
   boost::math::float_next(v1 * 1);
   boost::math::float_prior(v1 * 1);
   boost::math::float_distance(v1 * 1, v1 * 1);
   boost::math::ulp(v1 * 1);
   boost::math::relative_difference(v1 * 1, v2 * 1);
   boost::math::epsilon_difference(v1 * 1, v2 * 1);
#endif
#ifndef BOOST_MATH_INSTANTIATE_MINIMUM
#ifdef TEST_GROUP_8
   //
   // All over again, with a policy this time:
   //
   test_policy pol;
   boost::math::tgamma(v1, pol);
   boost::math::tgamma1pm1(v1, pol);
   boost::math::lgamma(v1, pol);
   {
      int i_other { i };

      boost::math::lgamma(v1, &i_other, pol);
   }
   boost::math::digamma(v1, pol);
   boost::math::trigamma(v1, pol);
   boost::math::polygamma(i, v1, pol);
   boost::math::tgamma_ratio(v1, v2, pol);
   boost::math::tgamma_delta_ratio(v1, v2, pol);
   boost::math::factorial<RealType>(i, pol);
   boost::math::unchecked_factorial<RealType>(i);
   {
      const auto i_fact = boost::math::max_factorial<RealType>::value;
      boost::math::double_factorial<RealType>(i_fact, pol);
      boost::math::rising_factorial(v1, i_fact, pol);
      boost::math::falling_factorial(v1, i_fact, pol);
   }
   boost::math::tgamma(v1, v2, pol);
   boost::math::tgamma_lower(v1, v2, pol);
   boost::math::gamma_p(v1, v2, pol);
   boost::math::gamma_q(v1, v2, pol);
   boost::math::gamma_p_inv(v1, v2, pol);
   boost::math::gamma_q_inv(v1, v2, pol);
   boost::math::gamma_p_inva(v1, v2, pol);
   boost::math::gamma_q_inva(v1, v2, pol);
   boost::math::erf(v1, pol);
   boost::math::erfc(v1, pol);
   boost::math::erf_inv(v1, pol);
   boost::math::erfc_inv(v1, pol);
   boost::math::beta(v1, v2, pol);
   boost::math::beta(v1, v2, v3, pol);
   boost::math::betac(v1, v2, v3, pol);
   boost::math::ibeta(v1, v2, v3, pol);
   boost::math::ibetac(v1, v2, v3, pol);
   boost::math::ibeta_inv(v1, v2, v3, pol);
   boost::math::ibetac_inv(v1, v2, v3, pol);
   boost::math::ibeta_inva(v1, v2, v3, pol);
   boost::math::ibetac_inva(v1, v2, v3, pol);
   boost::math::ibeta_invb(v1, v2, v3, pol);
   boost::math::ibetac_invb(v1, v2, v3, pol);
   boost::math::gamma_p_derivative(v2, v3, pol);
   boost::math::ibeta_derivative(v1, v2, v3, pol);
   boost::math::binomial_coefficient<RealType>(i, i, pol);
   boost::math::log1p(v1, pol);
   boost::math::expm1(v1, pol);
   boost::math::cbrt(v1, pol);
   boost::math::sqrt1pm1(v1, pol);
   boost::math::powm1(v1, v2, pol);
   boost::math::legendre_p(1, v1, pol);
   boost::math::legendre_p(1, 0, v1, pol);
   boost::math::legendre_p_prime(1, v1 * 1, pol);
   boost::math::legendre_q(1, v1, pol);
   boost::math::legendre_next(2, v1, v2, v3);
   boost::math::legendre_next(2, 2, v1, v2, v3);
   boost::math::laguerre(1, v1, pol);
   boost::math::laguerre(2, 1, v1, pol);
   boost::math::laguerre_next(2, v1, v2, v3);
   boost::math::laguerre_next(2, 1, v1, v2, v3);
   boost::math::hermite(1, v1, pol);
   boost::math::hermite_next(2, v1, v2, v3);
   boost::math::chebyshev_t(1, v1, pol);
   boost::math::chebyshev_u(1, v1, pol);
   boost::math::chebyshev_t_prime(1, v1, pol);
   boost::math::spherical_harmonic_r(2, 1, v1, v2, pol);
   boost::math::spherical_harmonic_i(2, 1, v1, v2, pol);
   boost::math::ellint_1(v1, pol);
   boost::math::ellint_1(v1, v2, pol);
   boost::math::ellint_2(v1, pol);
   boost::math::ellint_2(v1, v2, pol);
   boost::math::ellint_3(v1, v2, pol);
   boost::math::ellint_3(v1, v2, v3, pol);
   boost::math::ellint_d(v1, pol);
   boost::math::ellint_d(v1, v2, pol);
   boost::math::jacobi_zeta(v1, v2, pol);
   boost::math::heuman_lambda(v1, v2, pol);
   boost::math::ellint_rc(v1, v2, pol);
   boost::math::ellint_rd(v1, v2, v3, pol);
   boost::math::ellint_rf(v1, v2, v3, pol);
   boost::math::ellint_rg(v1, v2, v3, pol);
   boost::math::ellint_rj(v1, v2, v3, v1, pol);
   {
      RealType v1_to_get { v1 };

      boost::math::jacobi_elliptic(v1, v2, &v1_to_get, &v2, pol);
   }
   boost::math::jacobi_cd(v1, v2, pol);
   boost::math::jacobi_cn(v1, v2, pol);
   boost::math::jacobi_cs(v1, v2, pol);
   boost::math::jacobi_dc(v1, v2, pol);
   boost::math::jacobi_dn(v1, v2, pol);
   boost::math::jacobi_ds(v1, v2, pol);
   boost::math::jacobi_nc(v1, v2, pol);
   boost::math::jacobi_nd(v1, v2, pol);
   boost::math::jacobi_ns(v1, v2, pol);
   boost::math::jacobi_sc(v1, v2, pol);
   boost::math::jacobi_sd(v1, v2, pol);
   boost::math::jacobi_sn(v1, v2, pol);
   boost::math::jacobi_theta1(v1, v2, pol);
   boost::math::jacobi_theta1tau(v1, v2, pol);
   boost::math::jacobi_theta2(v1, v2, pol);
   boost::math::jacobi_theta2tau(v1, v2, pol);
   boost::math::jacobi_theta3(v1, v2, pol);
   boost::math::jacobi_theta3tau(v1, v2, pol);
   boost::math::jacobi_theta3m1(v1, v2, pol);
   boost::math::jacobi_theta3m1tau(v1, v2, pol);
   boost::math::jacobi_theta4(v1, v2, pol);
   boost::math::jacobi_theta4tau(v1, v2, pol);
   boost::math::jacobi_theta4m1(v1, v2, pol);
   boost::math::jacobi_theta4m1tau(v1, v2, pol);
   boost::math::hypot(v1, v2, pol);
   boost::math::sinc_pi(v1, pol);
   boost::math::sinhc_pi(v1, pol);
   boost::math::asinh(v1, pol);
   {
      auto v_special = static_cast<RealType>(1.55L);

      boost::math::acosh(v_special, pol);
   }
   boost::math::atanh(v1, pol);
   boost::math::sin_pi(v1, pol);
   boost::math::cos_pi(v1, pol);
   boost::math::cyl_neumann(v1, v2, pol);
   boost::math::cyl_neumann(i, v2, pol);
   boost::math::cyl_bessel_j(v1, v2, pol);
   boost::math::cyl_bessel_j(i, v2, pol);
   boost::math::cyl_bessel_i(v1, v2, pol);
   boost::math::cyl_bessel_i(i, v2, pol);
   boost::math::cyl_bessel_k(v1, v2, pol);
   boost::math::cyl_bessel_k(i, v2, pol);
   boost::math::sph_bessel(i, v2, pol);
   boost::math::sph_bessel(i, 1, pol);
   boost::math::sph_neumann(i, v2, pol);
   boost::math::sph_neumann(i, i, pol);
   boost::math::cyl_neumann_prime(v1, v2, pol);
   boost::math::cyl_neumann_prime(i, v2, pol);
   boost::math::cyl_bessel_j_prime(v1, v2, pol);
   boost::math::cyl_bessel_j_prime(i, v2, pol);
   boost::math::cyl_bessel_i_prime(v1, v2, pol);
   boost::math::cyl_bessel_i_prime(i, v2, pol);
   boost::math::cyl_bessel_k_prime(v1, v2, pol);
   boost::math::cyl_bessel_k_prime(i, v2, pol);
   boost::math::sph_bessel_prime(i, v2, pol);
   boost::math::sph_bessel_prime(i, 1, pol);
   boost::math::sph_neumann_prime(i, v2, pol);
   boost::math::sph_neumann_prime(i, i, pol);
   boost::math::cyl_bessel_j_zero(v1, i, pol);
   boost::math::cyl_bessel_j_zero(v1, i, i, oi, pol);
   boost::math::cyl_neumann_zero(v1, i, pol);
   boost::math::cyl_neumann_zero(v1, i, i, oi, pol);
   boost::math::lambert_w0(v1, pol);
   {
      auto v_special = static_cast<RealType>(-0.234L);

      boost::math::lambert_wm1(v_special, pol);
   }
   boost::math::lambert_w0_prime(v1, pol);
#ifdef TEST_COMPLEX
   boost::math::cyl_hankel_1(v1, v2, pol);
   boost::math::cyl_hankel_1(i, v2, pol);
   boost::math::cyl_hankel_2(v1, v2, pol);
   boost::math::cyl_hankel_2(i, v2, pol);
   boost::math::sph_hankel_1(v1, v2, pol);
   boost::math::sph_hankel_1(i, v2, pol);
   boost::math::sph_hankel_2(v1, v2, pol);
   boost::math::sph_hankel_2(i, v2, pol);
#endif
   boost::math::airy_ai(v1, pol);
   boost::math::airy_bi(v1, pol);
   boost::math::airy_ai_prime(v1, pol);
   boost::math::airy_bi_prime(v1, pol);

   boost::math::airy_ai_zero<RealType>(i, pol);
   boost::math::airy_bi_zero<RealType>(i, pol);
   boost::math::airy_ai_zero<RealType>(i, i, oi, pol);
   boost::math::airy_bi_zero<RealType>(i, i, oi, pol);

   boost::math::hypergeometric_1F0(i, v2, pol);
   {
      using std::floor;
      const auto v1_special = static_cast<RealType>(floor(v1) * 2);

      boost::math::hypergeometric_1F0(v1_special, i, pol);
   }
   boost::math::hypergeometric_1F0(i, i, pol);
   boost::math::hypergeometric_0F1(i, v2, pol);
   boost::math::hypergeometric_0F1(v1, i, pol);
   boost::math::hypergeometric_0F1(i, i, pol);
   {
      const auto i_negate = -i;

      boost::math::hypergeometric_2F0(i_negate, v2, v3, pol);
      boost::math::hypergeometric_2F0(v1, i_negate, v3, pol);
   }
   {
      using std::floor;
      const auto v1_special = static_cast<RealType>(floor(v1) * 2);

      boost::math::hypergeometric_2F0(v1_special, v2, i, pol);
   }
#if !defined(BOOST_NO_CXX11_AUTO_DECLARATIONS) && !defined(BOOST_NO_CXX11_LAMBDAS) && !defined(BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX) && !defined(BOOST_NO_CXX11_HDR_TUPLE)
   boost::math::hypergeometric_1F1(i, v2, v3, pol);
   boost::math::hypergeometric_1F1(v1, i, v3, pol);
   boost::math::hypergeometric_1F1(v1, v2, i, pol);
#endif

   boost::math::expint(v1, pol);
   boost::math::expint(i, pol);
   boost::math::expint(i, v2, pol);
   boost::math::expint(i, i, pol);
   boost::math::zeta(v1, pol);
   boost::math::zeta(i, pol);
   boost::math::owens_t(v1, v2, pol);
   //
   // These next functions are intended to be found via ADL:
   //
   BOOST_MATH_STD_USING
   trunc(v1, pol);
   itrunc(v1, pol);
   ltrunc(v1, pol);
   round(v1, pol);
   iround(v1, pol);
   lround(v1, pol);
   {
      RealType v1_other { };

      modf(v1, &v1_other, pol);
   }
   {
      int i_other { };

      modf(v1, &i_other, pol);
   }
   {
      long l_other { };

      modf(v1, &l_other, pol);
   }

   using boost::math::lltrunc;
   using boost::math::llround;
   lltrunc(v1, pol);
   llround(v1, pol);
   {
      long long ll_other { };

      modf(v1, &ll_other, pol);
   }

   boost::math::pow<2>(v1, pol);
   boost::math::nextafter(v1, v1, pol);
   boost::math::float_next(v1, pol);
   boost::math::float_prior(v1, pol);
   boost::math::float_distance(v1, v1, pol);
   boost::math::ulp(v1, pol);

   boost::math::bernoulli_b2n<RealType>(i, pol);
   {
      auto v1_array = std::make_unique<RealType[]>(i);

      boost::math::bernoulli_b2n<RealType>(i, i, &v1_array[0U], pol);
   }

   boost::math::tangent_t2n<RealType>(i, pol);
   {
      auto v1_array = std::make_unique<RealType[]>(i);

      boost::math::tangent_t2n<RealType>(i, i, &v1_array[0U], pol);
   }
#endif
#ifdef TEST_GROUP_9
   //
   // All over again with the versions in test::
   //
   test::tgamma(v1);
   test::tgamma1pm1(v1);
   test::lgamma(v1);
   {
      int i_other { i };

      test::lgamma(v1, &i_other);
   }
   test::digamma(v1);
   test::trigamma(v1);
   test::polygamma(i, v1);
   test::tgamma_ratio(v1, v2);
   test::tgamma_delta_ratio(v1, v2);
   test::factorial<RealType>(i);
   test::unchecked_factorial<RealType>(i);
   {
      const auto i_fact = test::max_factorial<RealType>::value;
      test::double_factorial<RealType>(i_fact);
      test::rising_factorial(v1, i_fact);
      test::falling_factorial(v1, i_fact);
   }
   test::tgamma(v1, v2);
   test::tgamma_lower(v1, v2);
   test::gamma_p(v1, v2);
   test::gamma_q(v1, v2);
   test::gamma_p_inv(v1, v2);
   test::gamma_q_inv(v1, v2);
   test::gamma_p_inva(v1, v2);
   test::gamma_q_inva(v1, v2);
   test::erf(v1);
   test::erfc(v1);
   test::erf_inv(v1);
   test::erfc_inv(v1);
   test::beta(v1, v2);
   test::beta(v1, v2, v3);
   test::betac(v1, v2, v3);
   test::ibeta(v1, v2, v3);
   test::ibetac(v1, v2, v3);
   test::ibeta_inv(v1, v2, v3);
   test::ibetac_inv(v1, v2, v3);
   test::ibeta_inva(v1, v2, v3);
   test::ibetac_inva(v1, v2, v3);
   test::ibeta_invb(v1, v2, v3);
   test::ibetac_invb(v1, v2, v3);
   test::gamma_p_derivative(v2, v3);
   test::ibeta_derivative(v1, v2, v3);
   test::binomial_coefficient<RealType>(i, i);
   (test::fpclassify)(v1);
   (test::isfinite)(v1);
   (test::isnormal)(v1);
   (test::isnan)(v1);
   (test::isinf)(v1);
   (test::signbit)(v1);
   (test::copysign)(v1, v2);
   (test::changesign)(v1);
   (test::sign)(v1);
   test::log1p(v1);
   test::expm1(v1);
   test::cbrt(v1);
   test::sqrt1pm1(v1);
   test::powm1(v1, v2);
   test::legendre_p(1, v1);
   test::legendre_p(1, 0, v1);
   test::legendre_p_prime(1, v1 * 1);
   test::legendre_q(1, v1);
   test::legendre_next(2, v1, v2, v3);
   test::legendre_next(2, 2, v1, v2, v3);
   test::laguerre(1, v1);
   test::laguerre(2, 1, v1);
   test::laguerre_next(2, v1, v2, v3);
   test::laguerre_next(2, 1, v1, v2, v3);
   test::hermite(1, v1);
   test::hermite_next(2, v1, v2, v3);
   test::chebyshev_next(v1, v2, v3);
   test::chebyshev_t(1, v1);
   test::chebyshev_u(1, v1);
   test::chebyshev_t_prime(1, v1);
   {
      const RealType v1_other_const = v1;

      test::chebyshev_clenshaw_recurrence(&v1_other_const, 0, v2);
   }
   test::spherical_harmonic_r(2, 1, v1, v2);
   test::spherical_harmonic_i(2, 1, v1, v2);
   test::ellint_1(v1);
   test::ellint_1(v1, v2);
   test::ellint_2(v1);
   test::ellint_2(v1, v2);
   test::ellint_3(v1, v2);
   test::ellint_3(v1, v2, v3);
   test::ellint_d(v1);
   test::ellint_d(v1, v2);
   test::jacobi_zeta(v1, v2);
   test::heuman_lambda(v1, v2);
   test::ellint_rc(v1, v2);
   test::ellint_rd(v1, v2, v3);
   test::ellint_rf(v1, v2, v3);
   test::ellint_rg(v1, v2, v3);
   test::ellint_rj(v1, v2, v3, v1);
   {
      RealType v1_to_get { v1 };

      test::jacobi_elliptic(v1, v2, &v1_to_get, &v2);
   }
   test::jacobi_cd(v1, v2);
   test::jacobi_cn(v1, v2);
   test::jacobi_cs(v1, v2);
   test::jacobi_dc(v1, v2);
   test::jacobi_dn(v1, v2);
   test::jacobi_ds(v1, v2);
   test::jacobi_nc(v1, v2);
   test::jacobi_nd(v1, v2);
   test::jacobi_ns(v1, v2);
   test::jacobi_sc(v1, v2);
   test::jacobi_sd(v1, v2);
   test::jacobi_sn(v1, v2);
   test::jacobi_theta1(v1, v2);
   test::jacobi_theta1tau(v1, v2);
   test::jacobi_theta2(v1, v2);
   test::jacobi_theta2tau(v1, v2);
   test::jacobi_theta3(v1, v2);
   test::jacobi_theta3tau(v1, v2);
   test::jacobi_theta3m1(v1, v2);
   test::jacobi_theta3m1tau(v1, v2);
   test::jacobi_theta4(v1, v2);
   test::jacobi_theta4tau(v1, v2);
   test::jacobi_theta4m1(v1, v2);
   test::jacobi_theta4m1tau(v1, v2);
   test::hypot(v1, v2);
   test::sinc_pi(v1);
   test::sinhc_pi(v1);
   test::asinh(v1);
   {
      auto v_special = static_cast<RealType>(1.55L);

      test::acosh(v_special);
   }
   test::atanh(v1);
   test::sin_pi(v1);
   test::cos_pi(v1);
   test::cyl_neumann(v1, v2);
   test::cyl_neumann(i, v2);
   test::cyl_bessel_j(v1, v2);
   test::cyl_bessel_j(i, v2);
   test::cyl_bessel_i(v1, v2);
   test::cyl_bessel_i(i, v2);
   test::cyl_bessel_k(v1, v2);
   test::cyl_bessel_k(i, v2);
   test::sph_bessel(i, v2);
   test::sph_bessel(i, 1);
   test::sph_neumann(i, v2);
   test::sph_neumann(i, i);
   test::cyl_neumann_prime(v1, v2);
   test::cyl_neumann_prime(i, v2);
   test::cyl_bessel_j_prime(v1, v2);
   test::cyl_bessel_j_prime(i, v2);
   test::cyl_bessel_i_prime(v1, v2);
   test::cyl_bessel_i_prime(i, v2);
   test::cyl_bessel_k_prime(v1, v2);
   test::cyl_bessel_k_prime(i, v2);
   test::sph_bessel_prime(i, v2);
   test::sph_bessel_prime(i, 1);
   test::sph_neumann_prime(i, v2);
   test::sph_neumann_prime(i, i);
   test::cyl_bessel_j_zero(v1, i);
   test::cyl_bessel_j_zero(v1, i, i, oi);
   test::cyl_neumann_zero(v1, i);
   test::cyl_neumann_zero(v1, i, i, oi);
   test::lambert_w0(v1);
   {
      auto v_special = static_cast<RealType>(-0.234L);

      test::lambert_wm1(v_special);
   }
   test::lambert_w0_prime(v1);
#ifdef TEST_COMPLEX
   test::cyl_hankel_1(v1, v2);
   test::cyl_hankel_1(i, v2);
   test::cyl_hankel_2(v1, v2);
   test::cyl_hankel_2(i, v2);
   test::sph_hankel_1(v1, v2);
   test::sph_hankel_1(i, v2);
   test::sph_hankel_2(v1, v2);
   test::sph_hankel_2(i, v2);
#endif
   test::airy_ai(i);
   test::airy_bi(i);
   test::airy_ai_prime(i);
   test::airy_bi_prime(i);

   test::airy_ai_zero<RealType>(i);
   test::airy_bi_zero<RealType>(i);
   test::airy_ai_zero<RealType>(i, i, oi);
   test::airy_bi_zero<RealType>(i, i, oi);

   test::hypergeometric_1F0(v1, v2);
   test::hypergeometric_0F1(v1, v2);
   {
      using std::floor;
      const auto v_special = static_cast<RealType>(floor(v1) * 2);

      test::hypergeometric_2F0(v1, v_special, v2);
   }
#if !defined(BOOST_NO_CXX11_AUTO_DECLARATIONS) && !defined(BOOST_NO_CXX11_LAMBDAS) && !defined(BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX) && !defined(BOOST_NO_CXX11_HDR_TUPLE)
   test::hypergeometric_1F1(v1, v2, v2);
#endif

   test::expint(v1);
   test::expint(i);
   test::expint(i, v2);
   test::expint(i, i);
   test::zeta(v1);
   test::zeta(i);
   test::owens_t(v1, v2);
   test::trunc(v1);
   test::itrunc(v1);
   test::ltrunc(v1);
   test::round(v1);
   test::iround(v1);
   test::lround(v1);
   {
      RealType v1_other { };

      test::modf(v1, &v1_other);
   }
   {
      int i_other { };

      test::modf(v1, &i_other);
   }
   {
      long l_other { };

      test::modf(v1, &l_other);
   }

   test::lltrunc(v1);
   test::llround(v1);
   {
      long long ll_other { };

      test::modf(v1, &ll_other);
   }

   test::pow<2>(v1);
   test::nextafter(v1, v1);
   test::float_next(v1);
   test::float_prior(v1);
   test::float_distance(v1, v1);
   test::ulp(v1);
#endif
#endif

   instantiate_runner_result<RealType>::value = true;
}

template <class RealType>
void instantiate_mixed(RealType)
{
   instantiate_mixed_runner_result<RealType>::value = false;

   using namespace boost;
   using namespace boost::math;
#ifndef BOOST_MATH_INSTANTIATE_MINIMUM
   int i { 1 };
   static_cast<void>(i);
   long l { 1 };
   static_cast<void>(l);
   short s { static_cast<short>(1) };
   static_cast<void>(s);
   float fr { 0.5F };
   static_cast<void>(fr);
   double dr = { 0.5 };
   static_cast<void>(dr);
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   long double lr { 0.5L };
   static_cast<void>(lr);
#else
   double lr { static_cast<double>(0.5L) };
   static_cast<void>(lr);
#endif
#ifdef TEST_GROUP_10
   boost::math::tgamma(i);
   boost::math::tgamma1pm1(i);
   boost::math::lgamma(i);
   {
      int i_other { i };

      boost::math::lgamma(i, &i_other);
   }
   boost::math::digamma(i);
   boost::math::trigamma(i);
   boost::math::polygamma(i, i);
   boost::math::tgamma_ratio(i, l);
   boost::math::tgamma_ratio(fr, lr);
   boost::math::tgamma_delta_ratio(i, s);
   boost::math::tgamma_delta_ratio(fr, lr);
   boost::math::rising_factorial(s, i);
   boost::math::falling_factorial(s, i);
   boost::math::tgamma(i, l);
   boost::math::tgamma(fr, lr);
   boost::math::tgamma_lower(i, s);
   boost::math::tgamma_lower(fr, lr);
   boost::math::gamma_p(i, s);
   boost::math::gamma_p(fr, lr);
   boost::math::gamma_q(i, s);
   boost::math::gamma_q(fr, lr);
   boost::math::gamma_p_inv(i, fr);
   boost::math::gamma_q_inv(s, fr);
   boost::math::gamma_p_inva(i, lr);
   boost::math::gamma_q_inva(i, lr);
   boost::math::erf(i);
   boost::math::erfc(i);
   {
      int i_zero { 0 };

      boost::math::erf_inv(i_zero);
   }
   boost::math::erfc_inv(i);
   boost::math::beta(i, s);
   boost::math::beta(fr, lr);
   boost::math::beta(i, s, l);
   boost::math::beta(fr, dr, lr);
   boost::math::betac(l, i, s);
   boost::math::betac(fr, dr, lr);
   boost::math::ibeta(l, i, s);
   boost::math::ibeta(fr, dr, lr);
   boost::math::ibetac(l, i, s);
   boost::math::ibetac(fr, dr, lr);
   boost::math::ibeta_inv(l, s, i);
   boost::math::ibeta_inv(fr, dr, lr);
   boost::math::ibetac_inv(l, i, s);
   boost::math::ibetac_inv(fr, dr, lr);
   boost::math::ibeta_inva(l, i, s);
   boost::math::ibeta_inva(fr, dr, lr);
   {
      int i_zero { 0 };
      short s_zero { 0 };

      boost::math::ibetac_inva(l, i_zero, s_zero);
   }
   boost::math::ibetac_inva(fr, dr, lr);
   {
      int i_zero { 0 };
      short s_zero { 0 };

      boost::math::ibeta_invb(l, i_zero, s_zero);
   }
   boost::math::ibeta_invb(fr, dr, lr);
   boost::math::ibetac_invb(l, i, s);
   boost::math::ibetac_invb(fr, dr, lr);
   boost::math::gamma_p_derivative(i, l);
   boost::math::gamma_p_derivative(fr, lr);
   boost::math::ibeta_derivative(l, i, s);
   boost::math::ibeta_derivative(fr, dr, lr);
   (boost::math::fpclassify)(i);
   (boost::math::isfinite)(s);
   (boost::math::isnormal)(l);
   (boost::math::isnan)(i);
   (boost::math::isinf)(l);
   boost::math::log1p(i);
   boost::math::expm1(s);
   boost::math::cbrt(l);
   boost::math::sqrt1pm1(s);
   boost::math::powm1(i, s);
   boost::math::powm1(fr, lr);
   //boost::math::legendre_p(1, i);
   boost::math::legendre_p(1, 0, s);
   {
      int i_zero { 0 };

      boost::math::legendre_q(1, i_zero);
   }
   boost::math::laguerre(1, i);
   boost::math::laguerre(2, 1, i);
   boost::math::laguerre(2u, 1u, s);
   boost::math::hermite(1, s);
   boost::math::chebyshev_t(1, i);
   boost::math::chebyshev_u(1, i);
   boost::math::chebyshev_t_prime(1, i);
   boost::math::spherical_harmonic_r(2, 1, s, i);
   boost::math::spherical_harmonic_i(2, 1, fr, lr);
   {
      int i_zero { 0 };

      boost::math::ellint_1(i_zero);
   }
   boost::math::ellint_1(i, s);
   boost::math::ellint_1(fr, lr);
   boost::math::ellint_2(i);
   boost::math::ellint_2(i, l);
   boost::math::ellint_2(fr, lr);
   {
      int i_zero { 0 };
      long l_zero { 0 };

      boost::math::ellint_3(i_zero, l_zero);
   }
   boost::math::ellint_3(fr, lr);
   {
      int i_zero { 0 };
      long l_zero { 0 };

      boost::math::ellint_3(s, l_zero, i_zero);
   }
   boost::math::ellint_3(fr, dr, lr);
   {
      int i_zero { 0 };

      boost::math::ellint_d(i_zero);
   }
   boost::math::ellint_d(i, l);
   boost::math::ellint_d(fr, lr);
   boost::math::jacobi_zeta(i, l);
   boost::math::jacobi_zeta(fr, lr);
   {
      int i_zero { 0 };

      boost::math::heuman_lambda(i_zero, l);
   }
   boost::math::heuman_lambda(fr, lr);
   boost::math::ellint_rc(i, s);
   boost::math::ellint_rc(fr, lr);
   boost::math::ellint_rd(s, i, l);
   boost::math::ellint_rd(fr, lr, dr);
   boost::math::ellint_rf(s, l, i);
   boost::math::ellint_rf(fr, dr, lr);
   boost::math::ellint_rg(s, l, i);
   boost::math::ellint_rg(fr, dr, lr);
   boost::math::ellint_rj(i, i, s, l);
   boost::math::ellint_rj(i, fr, dr, lr);
   boost::math::jacobi_cd(i, fr);
   boost::math::jacobi_cn(i, fr);
   boost::math::jacobi_cs(i, fr);
   boost::math::jacobi_dc(i, fr);
   boost::math::jacobi_dn(i, fr);
   boost::math::jacobi_ds(i, fr);
   boost::math::jacobi_nc(i, fr);
   boost::math::jacobi_nd(i, fr);
   boost::math::jacobi_ns(i, fr);
   boost::math::jacobi_sc(i, fr);
   boost::math::jacobi_sd(i, fr);
   boost::math::jacobi_sn(i, fr);
   boost::math::jacobi_theta1(i, fr);
   boost::math::jacobi_theta1tau(i, fr);
   boost::math::jacobi_theta2(i, fr);
   boost::math::jacobi_theta2tau(i, fr);
   boost::math::jacobi_theta3(i, fr);
   boost::math::jacobi_theta3tau(i, fr);
   boost::math::jacobi_theta3m1(i, fr);
   boost::math::jacobi_theta3m1tau(i, fr);
   boost::math::jacobi_theta4(i, fr);
   boost::math::jacobi_theta4tau(i, fr);
   boost::math::jacobi_theta4m1(i, fr);
   boost::math::jacobi_theta4m1tau(i, fr);
   boost::math::hypot(i, s);
   boost::math::hypot(fr, lr);
   boost::math::sinc_pi(i);
   boost::math::sinhc_pi(i);
   boost::math::asinh(s);
   boost::math::acosh(l);
   {
      long l_zero { 0 };

      boost::math::atanh(l_zero);
   }
   boost::math::sin_pi(s);
   boost::math::cos_pi(s);
   boost::math::cyl_neumann(fr, dr);
   boost::math::cyl_neumann(i, s);
   boost::math::cyl_bessel_j(fr, lr);
   boost::math::cyl_bessel_j(i, s);
   boost::math::cyl_bessel_i(fr, lr);
   boost::math::cyl_bessel_i(i, s);
   boost::math::cyl_bessel_k(fr, lr);
   boost::math::cyl_bessel_k(i, s);
   boost::math::sph_bessel(i, fr);
   boost::math::sph_bessel(i, 1);
   boost::math::sph_neumann(i, lr);
   boost::math::sph_neumann(i, i);
   boost::math::cyl_neumann_prime(fr, dr);
   boost::math::cyl_neumann_prime(i, s);
   boost::math::cyl_bessel_j_prime(fr, lr);
   boost::math::cyl_bessel_j_prime(i, s);
   boost::math::cyl_bessel_i_prime(fr, lr);
   boost::math::cyl_bessel_i_prime(i, s);
   boost::math::cyl_bessel_k_prime(fr, lr);
   boost::math::cyl_bessel_k_prime(i, s);
   boost::math::sph_bessel_prime(i, fr);
   boost::math::sph_bessel_prime(i, 1);
   boost::math::sph_neumann_prime(i, lr);
   boost::math::sph_neumann_prime(i, i);
   boost::math::owens_t(fr, dr);
   boost::math::owens_t(i, s);

   boost::math::policies::policy<> pol;

   boost::math::tgamma(i, pol);
   boost::math::tgamma1pm1(i, pol);
   boost::math::lgamma(i, pol);
   {
      int i_other { i };

      boost::math::lgamma(i, &i_other, pol);
   }
   boost::math::digamma(i, pol);
   boost::math::trigamma(i, pol);
   boost::math::polygamma(i, i, pol);
   boost::math::tgamma_ratio(i, l, pol);
   boost::math::tgamma_ratio(fr, lr, pol);
   boost::math::tgamma_delta_ratio(i, s, pol);
   boost::math::tgamma_delta_ratio(fr, lr, pol);
   boost::math::rising_factorial(s, i, pol);
   boost::math::falling_factorial(s, i, pol);
   boost::math::tgamma(i, l, pol);
   boost::math::tgamma(fr, lr, pol);
   boost::math::tgamma_lower(i, s, pol);
   boost::math::tgamma_lower(fr, lr, pol);
   boost::math::gamma_p(i, s, pol);
   boost::math::gamma_p(fr, lr, pol);
   boost::math::gamma_q(i, s, pol);
   boost::math::gamma_q(fr, lr, pol);
   boost::math::gamma_p_inv(i, fr, pol);
   boost::math::gamma_q_inv(s, fr, pol);
   boost::math::gamma_p_inva(i, lr, pol);
   boost::math::gamma_q_inva(i, lr, pol);
   boost::math::erf(i, pol);
   boost::math::erfc(i, pol);
   {
      int i_zero { 0 };

      boost::math::erf_inv(i_zero, pol);
   }
   boost::math::erfc_inv(i, pol);
   boost::math::beta(i, s, pol);
   boost::math::beta(fr, lr, pol);
   boost::math::beta(i, s, l, pol);
   boost::math::beta(fr, dr, lr, pol);
   boost::math::betac(l, i, s, pol);
   boost::math::betac(fr, dr, lr, pol);
   boost::math::ibeta(l, i, s, pol);
   boost::math::ibeta(fr, dr, lr, pol);
   boost::math::ibetac(l, i, s, pol);
   boost::math::ibetac(fr, dr, lr, pol);
   boost::math::ibeta_inv(l, s, i, pol);
   boost::math::ibeta_inv(fr, dr, lr, pol);
   boost::math::ibetac_inv(l, i, s, pol);
   boost::math::ibetac_inv(fr, dr, lr, pol);
   boost::math::ibeta_inva(l, i, s, pol);
   boost::math::ibeta_inva(fr, dr, lr, pol);
   {
      int i_zero { 0 };
      short s_zero { 0 };

      boost::math::ibetac_inva(l, i_zero, s_zero, pol);
   }
   boost::math::ibetac_inva(fr, dr, lr, pol);
   {
      int i_zero { 0 };
      short s_zero { 0 };

      boost::math::ibeta_invb(l, i_zero, s_zero, pol);
   }
   boost::math::ibeta_invb(fr, dr, lr, pol);
   boost::math::ibetac_invb(l, i, s, pol);
   boost::math::ibetac_invb(fr, dr, lr, pol);
   boost::math::gamma_p_derivative(i, l, pol);
   boost::math::gamma_p_derivative(fr, lr, pol);
   boost::math::ibeta_derivative(l, i, s, pol);
   boost::math::ibeta_derivative(fr, dr, lr, pol);
   boost::math::log1p(i, pol);
   boost::math::expm1(s, pol);
   boost::math::cbrt(l, pol);
   boost::math::sqrt1pm1(s, pol);
   boost::math::powm1(i, s, pol);
   boost::math::powm1(fr, lr, pol);
   //boost::math::legendre_p(1, i, pol);
   boost::math::legendre_p(1, 0, s, pol);
   {
      int i_zero { 0 };

      boost::math::legendre_q(1, i_zero, pol);
   }
   boost::math::laguerre(1, i, pol);
   boost::math::laguerre(2, 1, i, pol);
   boost::math::laguerre(2u, 1u, s, pol);
   boost::math::hermite(1, s, pol);
   boost::math::chebyshev_t(1, i, pol);
   boost::math::chebyshev_u(1, i, pol);
   boost::math::chebyshev_t_prime(1, i, pol);
   boost::math::spherical_harmonic_r(2, 1, s, i, pol);
   boost::math::spherical_harmonic_i(2, 1, fr, lr, pol);
   {
      int i_zero { 0 };

      boost::math::ellint_1(i_zero, pol);
   }
   boost::math::ellint_1(i, s, pol);
   boost::math::ellint_1(fr, lr, pol);
   boost::math::ellint_2(i, pol);
   boost::math::ellint_2(i, l, pol);
   boost::math::ellint_2(fr, lr, pol);
   {
      int i_zero { 0 };
      long l_zero { 0 };

      boost::math::ellint_3(i_zero, l_zero, pol);
   }
   boost::math::ellint_3(fr, lr, pol);
   {
      int i_zero { 0 };
      long l_zero { 0 };

      boost::math::ellint_3(s, l_zero, i_zero, pol);
   }
   boost::math::ellint_3(fr, dr, lr, pol);
   {
      int i_zero { 0 };

      boost::math::ellint_d(i_zero, pol);
   }
   boost::math::ellint_d(i, l, pol);
   boost::math::ellint_d(fr, lr, pol);
   boost::math::jacobi_zeta(i, l, pol);
   boost::math::jacobi_zeta(fr, lr, pol);
   {
      int i_zero { 0 };

      boost::math::heuman_lambda(i_zero, l, pol);
   }
   boost::math::heuman_lambda(fr, lr, pol);
   boost::math::ellint_rc(i, s, pol);
   boost::math::ellint_rc(fr, lr, pol);
   boost::math::ellint_rd(s, i, l, pol);
   boost::math::ellint_rd(fr, lr, dr, pol);
   boost::math::ellint_rf(s, l, i, pol);
   boost::math::ellint_rf(fr, dr, lr, pol);
   boost::math::ellint_rg(s, l, i, pol);
   boost::math::ellint_rg(fr, dr, lr, pol);
   boost::math::ellint_rj(i, i, s, l, pol);
   boost::math::ellint_rj(i, fr, dr, lr, pol);
   boost::math::jacobi_cd(i, fr, pol);
   boost::math::jacobi_cn(i, fr, pol);
   boost::math::jacobi_cs(i, fr, pol);
   boost::math::jacobi_dc(i, fr, pol);
   boost::math::jacobi_dn(i, fr, pol);
   boost::math::jacobi_ds(i, fr, pol);
   boost::math::jacobi_nc(i, fr, pol);
   boost::math::jacobi_nd(i, fr, pol);
   boost::math::jacobi_ns(i, fr, pol);
   boost::math::jacobi_sc(i, fr, pol);
   boost::math::jacobi_sd(i, fr, pol);
   boost::math::jacobi_sn(i, fr, pol);
   boost::math::jacobi_theta1(i, fr, pol);
   boost::math::jacobi_theta1tau(i, fr, pol);
   boost::math::jacobi_theta2(i, fr, pol);
   boost::math::jacobi_theta2tau(i, fr, pol);
   boost::math::jacobi_theta3(i, fr, pol);
   boost::math::jacobi_theta3tau(i, fr, pol);
   boost::math::jacobi_theta3m1(i, fr, pol);
   boost::math::jacobi_theta3m1tau(i, fr, pol);
   boost::math::jacobi_theta4(i, fr, pol);
   boost::math::jacobi_theta4tau(i, fr, pol);
   boost::math::jacobi_theta4m1(i, fr, pol);
   boost::math::jacobi_theta4m1tau(i, fr, pol);
   boost::math::hypot(i, s, pol);
   boost::math::hypot(fr, lr, pol);
   boost::math::sinc_pi(i, pol);
   boost::math::sinhc_pi(i, pol);
   boost::math::asinh(s, pol);
   boost::math::acosh(l, pol);
   {
      long l_zero { 0 };

      boost::math::atanh(l_zero, pol);
   }
   boost::math::sin_pi(s, pol);
   boost::math::cos_pi(s, pol);
   boost::math::cyl_neumann(fr, dr, pol);
   boost::math::cyl_neumann(i, s, pol);
   boost::math::cyl_bessel_j(fr, lr, pol);
   boost::math::cyl_bessel_j(i, s, pol);
   boost::math::cyl_bessel_i(fr, lr, pol);
   boost::math::cyl_bessel_i(i, s, pol);
   boost::math::cyl_bessel_k(fr, lr, pol);
   boost::math::cyl_bessel_k(i, s, pol);
   boost::math::sph_bessel(i, fr, pol);
   boost::math::sph_bessel(i, 1, pol);
   boost::math::sph_neumann(i, lr, pol);
   boost::math::sph_neumann(i, i, pol);
   boost::math::cyl_neumann_prime(fr, dr, pol);
   boost::math::cyl_neumann_prime(i, s, pol);
   boost::math::cyl_bessel_j_prime(fr, lr, pol);
   boost::math::cyl_bessel_j_prime(i, s, pol);
   boost::math::cyl_bessel_i_prime(fr, lr, pol);
   boost::math::cyl_bessel_i_prime(i, s, pol);
   boost::math::cyl_bessel_k_prime(fr, lr, pol);
   boost::math::cyl_bessel_k_prime(i, s, pol);
   boost::math::sph_bessel_prime(i, fr, pol);
   boost::math::sph_bessel_prime(i, 1, pol);
   boost::math::sph_neumann_prime(i, lr, pol);
   boost::math::sph_neumann_prime(i, i, pol);
   boost::math::owens_t(fr, dr, pol);
   boost::math::owens_t(i, s, pol);
   boost::math::lambert_w0(i, pol);
   {
      int i_zero { 0 };

      boost::math::lambert_wm1(i_zero, pol);
   }
   boost::math::lambert_w0_prime(i, pol);
#endif
#ifdef TEST_GROUP_11
   test::tgamma(i);
   test::tgamma1pm1(i);
   test::lgamma(i);
   {
      int i_other { i };

      test::lgamma(i, &i_other);
   }
   test::digamma(i);
   test::trigamma(i);
   test::polygamma(i, i);
   test::tgamma_ratio(i, l);
   test::tgamma_ratio(fr, lr);
   test::tgamma_delta_ratio(i, s);
   test::tgamma_delta_ratio(fr, lr);
   test::rising_factorial(s, i);
   test::falling_factorial(s, i);
   test::tgamma(i, l);
   test::tgamma(fr, lr);
   test::tgamma_lower(i, s);
   test::tgamma_lower(fr, lr);
   test::gamma_p(i, s);
   test::gamma_p(fr, lr);
   test::gamma_q(i, s);
   test::gamma_q(fr, lr);
   test::gamma_p_inv(i, fr);
   test::gamma_q_inv(s, fr);
   test::gamma_p_inva(i, lr);
   test::gamma_q_inva(i, lr);
   test::erf(i);
   test::erfc(i);
   {
      int i_zero { 0 };

      test::erf_inv(i_zero);
   }
   test::erfc_inv(i);
   test::beta(i, s);
   test::beta(fr, lr);
   test::beta(i, s, l);
   test::beta(fr, dr, lr);
   test::betac(l, i, s);
   test::betac(fr, dr, lr);
   test::ibeta(l, i, s);
   test::ibeta(fr, dr, lr);
   test::ibetac(l, i, s);
   test::ibetac(fr, dr, lr);
   test::ibeta_inv(l, s, i);
   test::ibeta_inv(fr, dr, lr);
   test::ibetac_inv(l, i, s);
   test::ibetac_inv(fr, dr, lr);
   test::ibeta_inva(l, i, s);
   test::ibeta_inva(fr, dr, lr);
   {
      int i_zero { 0 };
      short s_zero { 0 };

      test::ibetac_inva(l, i_zero, s_zero);
   }
   test::ibetac_inva(fr, dr, lr);
   {
      int i_zero { 0 };
      short s_zero { 0 };

      test::ibeta_invb(l, i_zero, s_zero);
   }
   test::ibeta_invb(fr, dr, lr);
   test::ibetac_invb(l, i, s);
   test::ibetac_invb(fr, dr, lr);
   test::gamma_p_derivative(i, l);
   test::gamma_p_derivative(fr, lr);
   test::ibeta_derivative(l, i, s);
   test::ibeta_derivative(fr, dr, lr);
   (test::fpclassify)(i);
   (test::isfinite)(s);
   (test::isnormal)(l);
   (test::isnan)(i);
   (test::isinf)(l);
   test::log1p(i);
   test::expm1(s);
   test::cbrt(l);
   test::sqrt1pm1(s);
   test::powm1(i, s);
   test::powm1(fr, lr);
   //test::legendre_p(1, i);
   test::legendre_p(1, 0, s);
   {
      int i_zero { 0 };

      test::legendre_q(1, i_zero);
   }
   test::laguerre(1, i);
   test::laguerre(2, 1, i);
   test::laguerre(2u, 1u, s);
   test::hermite(1, s);
   test::chebyshev_t(1, i);
   test::chebyshev_u(1, i);
   test::chebyshev_t_prime(1, s);
   test::spherical_harmonic_r(2, 1, s, i);
   test::spherical_harmonic_i(2, 1, fr, lr);
   {
      int i_zero { 0 };

      test::ellint_1(i_zero);
   }
   test::ellint_1(i, s);
   test::ellint_1(fr, lr);
   test::ellint_2(i);
   test::ellint_2(i, l);
   test::ellint_2(fr, lr);
   {
      int i_zero { 0 };
      long l_zero { 0 };

      test::ellint_3(i_zero, l_zero);
   }
   test::ellint_3(fr, lr);
   {
      int i_zero { 0 };
      long l_zero { 0 };

      test::ellint_3(s, l_zero, i_zero);
   }
   test::ellint_3(fr, dr, lr);
   {
      int i_zero { 0 };

      test::ellint_d(i_zero);
   }
   test::ellint_d(i, l);
   test::ellint_d(fr, lr);
   test::jacobi_zeta(i, l);
   test::jacobi_zeta(fr, lr);
   {
      int i_zero { 0 };

      test::heuman_lambda(i_zero, l);
   }
   test::heuman_lambda(fr, lr);
   test::ellint_rc(i, s);
   test::ellint_rc(fr, lr);
   test::ellint_rd(s, i, l);
   test::ellint_rd(fr, lr, dr);
   test::ellint_rf(s, l, i);
   test::ellint_rf(fr, dr, lr);
   test::ellint_rg(s, l, i);
   test::ellint_rg(fr, dr, lr);
   test::ellint_rj(i, i, s, l);
   test::ellint_rj(i, fr, dr, lr);
   test::hypot(i, s);
   test::hypot(fr, lr);
   test::sinc_pi(i);
   test::sinhc_pi(i);
   test::asinh(s);
   test::acosh(l);
   {
      long l_zero { 0 };

      test::atanh(l_zero);
   }
   test::sin_pi(s);
   test::cos_pi(s);
   test::cyl_neumann(fr, dr);
   test::cyl_neumann(i, s);
   test::cyl_bessel_j(fr, lr);
   test::cyl_bessel_j(i, s);
   test::cyl_bessel_i(fr, lr);
   test::cyl_bessel_i(i, s);
   test::cyl_bessel_k(fr, lr);
   test::cyl_bessel_k(i, s);
   test::sph_bessel(i, fr);
   test::sph_bessel(i, 1);
   test::sph_neumann(i, lr);
   test::sph_neumann(i, i);
   test::cyl_neumann_prime(fr, dr);
   test::cyl_neumann_prime(i, s);
   test::cyl_bessel_j_prime(fr, lr);
   test::cyl_bessel_j_prime(i, s);
   test::cyl_bessel_i_prime(fr, lr);
   test::cyl_bessel_i_prime(i, s);
   test::cyl_bessel_k_prime(fr, lr);
   test::cyl_bessel_k_prime(i, s);
   test::sph_bessel_prime(i, fr);
   test::sph_bessel_prime(i, 1);
   test::sph_neumann_prime(i, lr);
   test::sph_neumann_prime(i, i);
   test::airy_ai(i);
   test::airy_bi(i);
   test::airy_ai_prime(i);
   test::airy_bi_prime(i);
   test::owens_t(fr, dr);
   test::owens_t(i, s);
   boost::math::lambert_w0(i);
   {
      int i_zero { 0 };

      boost::math::lambert_wm1(i_zero);
   }
   boost::math::lambert_w0_prime(i);
#endif
#endif

   instantiate_mixed_runner_result<RealType>::value = true;
}


#endif // BOOST_LIBS_MATH_TEST_INSTANTIATE_HPP
