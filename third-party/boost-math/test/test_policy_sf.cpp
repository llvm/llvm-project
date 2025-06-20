//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions.hpp>

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file provides very basic sanity checks for the special functions
// declared with BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS, basically we
// just want to make sure that the inline forwarding functions do
// actually forward to the right function!!
//

namespace test{

   typedef boost::math::policies::policy<> policy;

   BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS(policy)

}

#define TEST_POLICY_SF(call)\
   BOOST_CHECK_EQUAL(boost::math::call , test::call)

//
// Prevent some macro conflicts just in case:
//
#undef fpclassify
#undef isnormal
#undef isinf
#undef isfinite
#undef isnan

BOOST_AUTO_TEST_CASE( test_main )
{
   int i;
   TEST_POLICY_SF(tgamma(3.0));
   TEST_POLICY_SF(tgamma1pm1(0.25));
   TEST_POLICY_SF(lgamma(50.0));
   TEST_POLICY_SF(lgamma(50.0, &i));
   TEST_POLICY_SF(digamma(12.0));
   TEST_POLICY_SF(tgamma_ratio(12.0, 13.5));
   TEST_POLICY_SF(tgamma_delta_ratio(100.0, 0.25));
   TEST_POLICY_SF(factorial<double>(8));
   TEST_POLICY_SF(unchecked_factorial<double>(3));
   TEST_POLICY_SF(double_factorial<double>(5));
   TEST_POLICY_SF(rising_factorial(20.5, 5));
   TEST_POLICY_SF(falling_factorial(10.2, 7));
   TEST_POLICY_SF(tgamma(12.0, 13.0));
   TEST_POLICY_SF(tgamma_lower(12.0, 13.0));
   TEST_POLICY_SF(gamma_p(12.0, 13.0));
   TEST_POLICY_SF(gamma_q(12.0, 15.0));
   TEST_POLICY_SF(gamma_p_inv(12.0, 0.25));
   TEST_POLICY_SF(gamma_q_inv(15.0, 0.25));
   TEST_POLICY_SF(gamma_p_inva(12.0, 0.25));
   TEST_POLICY_SF(gamma_q_inva(12.0, 0.25));
   TEST_POLICY_SF(erf(2.5));
   TEST_POLICY_SF(erfc(2.5));
   TEST_POLICY_SF(erf_inv(0.25));
   TEST_POLICY_SF(erfc_inv(0.25));
   TEST_POLICY_SF(beta(12.0, 15.0));
   TEST_POLICY_SF(beta(12.0, 15.0, 0.25));
   TEST_POLICY_SF(betac(12.0, 15.0, 0.25));
   TEST_POLICY_SF(ibeta(12.0, 15.0, 0.25));
   TEST_POLICY_SF(ibetac(12.0, 15.0, 0.25));
   TEST_POLICY_SF(ibeta_inv(12.0, 15.0, 0.25));
   TEST_POLICY_SF(ibetac_inv(12.0, 15.0, 0.25));
   TEST_POLICY_SF(ibeta_inva(12.0, 0.75, 0.25));
   TEST_POLICY_SF(ibetac_inva(12.0, 0.75, 0.25));
   TEST_POLICY_SF(ibeta_invb(12.0, 0.75, 0.25));
   TEST_POLICY_SF(ibetac_invb(12.0, 0.75, 0.25));
   TEST_POLICY_SF(gamma_p_derivative(12.0, 15.0));
   TEST_POLICY_SF(ibeta_derivative(12.0, 15.75, 0.25));
   TEST_POLICY_SF(fpclassify(12.0));
   TEST_POLICY_SF(isfinite(12.0));
   TEST_POLICY_SF(isnormal(12.0));
   TEST_POLICY_SF(isnan(12.0));
   TEST_POLICY_SF(isinf(12.0));
   TEST_POLICY_SF(log1p(0.0025));
   TEST_POLICY_SF(expm1(0.0025));
   TEST_POLICY_SF(cbrt(30.0));
   TEST_POLICY_SF(sqrt1pm1(0.0025));
   TEST_POLICY_SF(powm1(1.0025, 12.0));
   TEST_POLICY_SF(legendre_p(5, 0.75));
   TEST_POLICY_SF(legendre_p(7, 3, 0.75));
   TEST_POLICY_SF(legendre_q(5, 0.75));
   TEST_POLICY_SF(legendre_next(2, 0.25, 12.0, 5.0));
   TEST_POLICY_SF(legendre_next(2, 2, 0.25, 12.0, 5.0));
   TEST_POLICY_SF(laguerre(5, 12.2));
   TEST_POLICY_SF(laguerre(7, 3, 5.0));
   TEST_POLICY_SF(laguerre_next(2, 5.0, 12.0, 5.0));
   TEST_POLICY_SF(laguerre_next(5, 3, 5.0, 20.0, 10.0));
   TEST_POLICY_SF(hermite(1, 2.0));
   TEST_POLICY_SF(hermite_next(2, 2.0, 3.0, 2.0));
   TEST_POLICY_SF(spherical_harmonic_r(5, 4, 0.75, 0.25));
   TEST_POLICY_SF(spherical_harmonic_i(5, 4, 0.75, 0.25));
   TEST_POLICY_SF(ellint_1(0.25));
   TEST_POLICY_SF(ellint_1(0.25, 0.75));
   TEST_POLICY_SF(ellint_2(0.25));
   TEST_POLICY_SF(ellint_2(0.25, 0.75));
   TEST_POLICY_SF(ellint_3(0.25, 0.75));
   TEST_POLICY_SF(ellint_3(0.25, 0.125, 0.75));
   TEST_POLICY_SF(ellint_rc(3.0, 5.0));
   TEST_POLICY_SF(ellint_rd(2.0, 3.0, 4.0));
   TEST_POLICY_SF(ellint_rf(2.0, 3.0, 4.0));
   TEST_POLICY_SF(ellint_rj(2.0, 3.0, 5.0, 0.25));
   TEST_POLICY_SF(hypot(5.0, 3.0));
   TEST_POLICY_SF(sinc_pi(3.0));
   TEST_POLICY_SF(sinhc_pi(2.0));
   TEST_POLICY_SF(asinh(12.0));
   TEST_POLICY_SF(acosh(5.0));
   TEST_POLICY_SF(atanh(0.75));
   TEST_POLICY_SF(sin_pi(5.0));
   TEST_POLICY_SF(cos_pi(6.0));
   TEST_POLICY_SF(cyl_neumann(2.0, 5.0));
   TEST_POLICY_SF(cyl_neumann(2, 5.0));
   TEST_POLICY_SF(cyl_bessel_j(2.0, 5.0));
   TEST_POLICY_SF(cyl_bessel_j(2, 5.0));
   TEST_POLICY_SF(cyl_bessel_i(3.0, 5.0));
   TEST_POLICY_SF(cyl_bessel_i(3, 5.0));
   TEST_POLICY_SF(cyl_bessel_k(3.0, 5.0));
   TEST_POLICY_SF(cyl_bessel_k(3, 5.0));
   TEST_POLICY_SF(sph_bessel(3, 5.0));
   TEST_POLICY_SF(sph_bessel(3, 5));
   TEST_POLICY_SF(sph_neumann(3, 5.0));
   TEST_POLICY_SF(sph_neumann(3, 5));
   
}




