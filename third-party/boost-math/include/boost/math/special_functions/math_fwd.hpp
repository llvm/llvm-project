// math_fwd.hpp

// TODO revise completely for new distribution classes.

// Copyright Paul A. Bristow 2006.
// Copyright John Maddock 2006.
// Copyright Matt Borland 2024

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Omnibus list of forward declarations of math special functions.

// IT = Integer type.
// RT = Real type (built-in floating-point types, float, double, long double) & User Defined Types
// AT = Integer or Real type

#ifndef BOOST_MATH_SPECIAL_MATH_FWD_HPP
#define BOOST_MATH_SPECIAL_MATH_FWD_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/promotion.hpp> // for argument promotion.
#include <boost/math/tools/type_traits.hpp>
#include <boost/math/tools/complex.hpp>
#include <boost/math/policies/policy.hpp>

#ifdef BOOST_MATH_HAS_NVRTC

namespace boost {
namespace math {

template <class RT1, class RT2, class A>
BOOST_MATH_GPU_ENABLED inline typename tools::promote_args<RT1, RT2, A>::type
beta(RT1 a, RT2 b, A arg);

namespace detail{

   template <class T, class U, class V>
   struct ellint_3_result
   {
      using type = typename boost::math::conditional<
         policies::is_policy<V>::value,
         tools::promote_args_t<T, U>,
         tools::promote_args_t<T, U, V>
      >::type;
   };

   template <class T, class U>
   struct expint_result
   {
      using type = typename boost::math::conditional<
         policies::is_policy<U>::value,
         tools::promote_args_t<T>,
         typename tools::promote_args<U>::type
      >::type;
   };

   typedef boost::math::integral_constant<int, 0> bessel_no_int_tag;      // No integer optimisation possible.
   typedef boost::math::integral_constant<int, 1> bessel_maybe_int_tag;   // Maybe integer optimisation.
   typedef boost::math::integral_constant<int, 2> bessel_int_tag;         // Definite integer optimisation.

   template <class T1, class T2, class Policy>
   struct bessel_traits
   {
      using result_type = typename boost::math::conditional<
         boost::math::is_integral<T1>::value,
         typename tools::promote_args<T2>::type,
         tools::promote_args_t<T1, T2>
      >::type;

      typedef typename policies::precision<result_type, Policy>::type precision_type;

      using optimisation_tag = typename boost::math::conditional<
         (precision_type::value <= 0 || precision_type::value > 64),
         bessel_no_int_tag,
         typename boost::math::conditional<
            boost::math::is_integral<T1>::value,
            bessel_int_tag,
            bessel_maybe_int_tag
         >::type
      >::type;

      using optimisation_tag128 = typename boost::math::conditional<
         (precision_type::value <= 0 || precision_type::value > 113),
         bessel_no_int_tag,
         typename boost::math::conditional<
            boost::math::is_integral<T1>::value,
            bessel_int_tag,
            bessel_maybe_int_tag
         >::type
      >::type;
   };

} // namespace detail

} // namespace math
} // namespace boost

#else

#include <vector>
#include <complex>
#include <type_traits>
#include <boost/math/special_functions/detail/round_fwd.hpp>
#include <boost/math/tools/type_traits.hpp>
#include <boost/math/policies/policy.hpp>

#define BOOST_NO_MACRO_EXPAND /**/

namespace boost
{
   namespace math
   { // Math functions (in roughly alphabetic order).

   // Beta functions.
   template <class RT1, class RT2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2>
         beta(RT1 a, RT2 b); // Beta function (2 arguments).

   template <class RT1, class RT2, class A>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, A>
         beta(RT1 a, RT2 b, A x); // Beta function (3 arguments).

   template <class RT1, class RT2, class RT3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         beta(RT1 a, RT2 b, RT3 x, const Policy& pol); // Beta function (3 arguments).

   template <class RT1, class RT2, class RT3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         betac(RT1 a, RT2 b, RT3 x);

   template <class RT1, class RT2, class RT3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         betac(RT1 a, RT2 b, RT3 x, const Policy& pol);

   template <class RT1, class RT2, class RT3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibeta(RT1 a, RT2 b, RT3 x); // Incomplete beta function.

   template <class RT1, class RT2, class RT3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibeta(RT1 a, RT2 b, RT3 x, const Policy& pol); // Incomplete beta function.

   template <class RT1, class RT2, class RT3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibetac(RT1 a, RT2 b, RT3 x); // Incomplete beta complement function.

   template <class RT1, class RT2, class RT3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibetac(RT1 a, RT2 b, RT3 x, const Policy& pol); // Incomplete beta complement function.

   template <class T1, class T2, class T3, class T4>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3, T4>
         ibeta_inv(T1 a, T2 b, T3 p, T4* py);

   template <class T1, class T2, class T3, class T4, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3, T4>
         ibeta_inv(T1 a, T2 b, T3 p, T4* py, const Policy& pol);

   template <class RT1, class RT2, class RT3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibeta_inv(RT1 a, RT2 b, RT3 p); // Incomplete beta inverse function.

   template <class RT1, class RT2, class RT3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibeta_inv(RT1 a, RT2 b, RT3 p, const Policy&); // Incomplete beta inverse function.

   template <class RT1, class RT2, class RT3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibeta_inva(RT1 a, RT2 b, RT3 p); // Incomplete beta inverse function.

   template <class RT1, class RT2, class RT3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibeta_inva(RT1 a, RT2 b, RT3 p, const Policy&); // Incomplete beta inverse function.

   template <class RT1, class RT2, class RT3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibeta_invb(RT1 a, RT2 b, RT3 p); // Incomplete beta inverse function.

   template <class RT1, class RT2, class RT3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibeta_invb(RT1 a, RT2 b, RT3 p, const Policy&); // Incomplete beta inverse function.

   template <class T1, class T2, class T3, class T4>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3, T4>
         ibetac_inv(T1 a, T2 b, T3 q, T4* py);

   template <class T1, class T2, class T3, class T4, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3, T4>
         ibetac_inv(T1 a, T2 b, T3 q, T4* py, const Policy& pol);

   template <class RT1, class RT2, class RT3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibetac_inv(RT1 a, RT2 b, RT3 q); // Incomplete beta complement inverse function.

   template <class RT1, class RT2, class RT3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibetac_inv(RT1 a, RT2 b, RT3 q, const Policy&); // Incomplete beta complement inverse function.

   template <class RT1, class RT2, class RT3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibetac_inva(RT1 a, RT2 b, RT3 q); // Incomplete beta complement inverse function.

   template <class RT1, class RT2, class RT3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibetac_inva(RT1 a, RT2 b, RT3 q, const Policy&); // Incomplete beta complement inverse function.

   template <class RT1, class RT2, class RT3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibetac_invb(RT1 a, RT2 b, RT3 q); // Incomplete beta complement inverse function.

   template <class RT1, class RT2, class RT3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibetac_invb(RT1 a, RT2 b, RT3 q, const Policy&); // Incomplete beta complement inverse function.

   template <class RT1, class RT2, class RT3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibeta_derivative(RT1 a, RT2 b, RT3 x);  // derivative of incomplete beta

   template <class RT1, class RT2, class RT3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2, RT3>
         ibeta_derivative(RT1 a, RT2 b, RT3 x, const Policy& pol);  // derivative of incomplete beta

   // Binomial:
   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED T binomial_coefficient(unsigned n, unsigned k, const Policy& pol);
   template <class T>
   BOOST_MATH_GPU_ENABLED T binomial_coefficient(unsigned n, unsigned k);

   // erf & erfc error functions.
   template <class RT> // Error function.
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> erf(RT z);
   template <class RT, class Policy> // Error function.
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> erf(RT z, const Policy&);

   template <class RT>// Error function complement.
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> erfc(RT z);
   template <class RT, class Policy>// Error function complement.
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> erfc(RT z, const Policy&);

   template <class RT>// Error function inverse.
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> erf_inv(RT z);
   template <class RT, class Policy>// Error function inverse.
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> erf_inv(RT z, const Policy& pol);

   template <class RT>// Error function complement inverse.
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> erfc_inv(RT z);
   template <class RT, class Policy>// Error function complement inverse.
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> erfc_inv(RT z, const Policy& pol);

   // Polynomials:
   template <class T1, class T2, class T3>
   tools::promote_args_t<T1, T2, T3>
         legendre_next(unsigned l, T1 x, T2 Pl, T3 Plm1);

   template <class T>
   tools::promote_args_t<T>
         legendre_p(int l, T x);
   template <class T>
   tools::promote_args_t<T>
          legendre_p_prime(int l, T x);


   template <class T, class Policy>
   inline std::vector<T> legendre_p_zeros(int l, const Policy& pol);

   template <class T>
   inline std::vector<T> legendre_p_zeros(int l);

   template <class T, class Policy>
   typename std::enable_if<policies::is_policy<Policy>::value, tools::promote_args_t<T>>::type
         legendre_p(int l, T x, const Policy& pol);
   template <class T, class Policy>
   inline typename std::enable_if<policies::is_policy<Policy>::value, tools::promote_args_t<T>>::type
      legendre_p_prime(int l, T x, const Policy& pol);

   template <class T>
   tools::promote_args_t<T>
         legendre_q(unsigned l, T x);

   template <class T, class Policy>
   typename std::enable_if<policies::is_policy<Policy>::value, tools::promote_args_t<T>>::type
         legendre_q(unsigned l, T x, const Policy& pol);

   template <class T1, class T2, class T3>
   tools::promote_args_t<T1, T2, T3>
         legendre_next(unsigned l, unsigned m, T1 x, T2 Pl, T3 Plm1);

   template <class T>
   tools::promote_args_t<T>
         legendre_p(int l, int m, T x);

   template <class T, class Policy>
   tools::promote_args_t<T>
         legendre_p(int l, int m, T x, const Policy& pol);

   template <class T1, class T2, class T3>
   tools::promote_args_t<T1, T2, T3>
         laguerre_next(unsigned n, T1 x, T2 Ln, T3 Lnm1);

   template <class T1, class T2, class T3>
   tools::promote_args_t<T1, T2, T3>
      laguerre_next(unsigned n, unsigned l, T1 x, T2 Pl, T3 Plm1);

   template <class T>
   tools::promote_args_t<T>
      laguerre(unsigned n, T x);

   template <class T, class Policy>
   tools::promote_args_t<T>
      laguerre(unsigned n, unsigned m, T x, const Policy& pol);

   template <class T1, class T2>
   struct laguerre_result
   {
      using type = typename std::conditional<
         policies::is_policy<T2>::value,
         typename tools::promote_args<T1>::type,
         typename tools::promote_args<T2>::type
      >::type;
   };

   template <class T1, class T2>
   typename laguerre_result<T1, T2>::type
      laguerre(unsigned n, T1 m, T2 x);

   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T>
      hermite(unsigned n, T x);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T>
      hermite(unsigned n, T x, const Policy& pol);

   template <class T1, class T2, class T3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3>
      hermite_next(unsigned n, T1 x, T2 Hn, T3 Hnm1);

   template<class T1, class T2, class T3>
   tools::promote_args_t<T1, T2, T3> chebyshev_next(T1 const & x, T2 const & Tn, T3 const & Tn_1);

   template <class Real, class Policy>
   tools::promote_args_t<Real>
      chebyshev_t(unsigned n, Real const & x, const Policy&);
   template<class Real>
   tools::promote_args_t<Real> chebyshev_t(unsigned n, Real const & x);
   
   template <class Real, class Policy>
   tools::promote_args_t<Real>
      chebyshev_u(unsigned n, Real const & x, const Policy&);
   template<class Real>
   tools::promote_args_t<Real> chebyshev_u(unsigned n, Real const & x);

   template <class Real, class Policy>
   tools::promote_args_t<Real>
      chebyshev_t_prime(unsigned n, Real const & x, const Policy&);
   template<class Real>
   tools::promote_args_t<Real> chebyshev_t_prime(unsigned n, Real const & x);

   template<class Real, class T2>
   Real chebyshev_clenshaw_recurrence(const Real* const c, size_t length, const T2& x);

   template <class T1, class T2>
   std::complex<tools::promote_args_t<T1, T2>>
         spherical_harmonic(unsigned n, int m, T1 theta, T2 phi);

   template <class T1, class T2, class Policy>
   std::complex<tools::promote_args_t<T1, T2>>
      spherical_harmonic(unsigned n, int m, T1 theta, T2 phi, const Policy& pol);

   template <class T1, class T2>
   tools::promote_args_t<T1, T2>
         spherical_harmonic_r(unsigned n, int m, T1 theta, T2 phi);

   template <class T1, class T2, class Policy>
   tools::promote_args_t<T1, T2>
      spherical_harmonic_r(unsigned n, int m, T1 theta, T2 phi, const Policy& pol);

   template <class T1, class T2>
   tools::promote_args_t<T1, T2>
         spherical_harmonic_i(unsigned n, int m, T1 theta, T2 phi);

   template <class T1, class T2, class Policy>
   tools::promote_args_t<T1, T2>
      spherical_harmonic_i(unsigned n, int m, T1 theta, T2 phi, const Policy& pol);

   // Elliptic integrals:
   template <class T1, class T2, class T3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3>
         ellint_rf(T1 x, T2 y, T3 z);

   template <class T1, class T2, class T3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3>
         ellint_rf(T1 x, T2 y, T3 z, const Policy& pol);

   template <class T1, class T2, class T3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3>
         ellint_rd(T1 x, T2 y, T3 z);

   template <class T1, class T2, class T3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3>
         ellint_rd(T1 x, T2 y, T3 z, const Policy& pol);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2>
         ellint_rc(T1 x, T2 y);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2>
         ellint_rc(T1 x, T2 y, const Policy& pol);

   template <class T1, class T2, class T3, class T4>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3, T4>
         ellint_rj(T1 x, T2 y, T3 z, T4 p);

   template <class T1, class T2, class T3, class T4, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3, T4>
         ellint_rj(T1 x, T2 y, T3 z, T4 p, const Policy& pol);

   template <class T1, class T2, class T3>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3>
      ellint_rg(T1 x, T2 y, T3 z);

   template <class T1, class T2, class T3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3>
      ellint_rg(T1 x, T2 y, T3 z, const Policy& pol);

   template <typename T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> ellint_2(T k);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> ellint_2(T1 k, T2 phi);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> ellint_2(T1 k, T2 phi, const Policy& pol);

   template <typename T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> ellint_1(T k);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> ellint_1(T1 k, T2 phi);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> ellint_1(T1 k, T2 phi, const Policy& pol);

   template <typename T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> ellint_d(T k);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> ellint_d(T1 k, T2 phi);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> ellint_d(T1 k, T2 phi, const Policy& pol);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> jacobi_zeta(T1 k, T2 phi);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> jacobi_zeta(T1 k, T2 phi, const Policy& pol);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> heuman_lambda(T1 k, T2 phi);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> heuman_lambda(T1 k, T2 phi, const Policy& pol);

   namespace detail{

   template <class T, class U, class V>
   struct ellint_3_result
   {
      using type = typename boost::math::conditional<
         policies::is_policy<V>::value,
         tools::promote_args_t<T, U>,
         tools::promote_args_t<T, U, V>
      >::type;
   };

   } // namespace detail


   template <class T1, class T2, class T3>
   BOOST_MATH_GPU_ENABLED typename detail::ellint_3_result<T1, T2, T3>::type ellint_3(T1 k, T2 v, T3 phi);

   template <class T1, class T2, class T3, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2, T3> ellint_3(T1 k, T2 v, T3 phi, const Policy& pol);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> ellint_3(T1 k, T2 v);

   // Factorial functions.
   // Note: not for integral types, at present.
   template <class RT>
   struct max_factorial;
   template <class RT>
   BOOST_MATH_GPU_ENABLED RT factorial(unsigned int);
   template <class RT, class Policy>
   BOOST_MATH_GPU_ENABLED RT factorial(unsigned int, const Policy& pol);
   template <class RT>
   BOOST_MATH_GPU_ENABLED RT unchecked_factorial(unsigned int BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(RT));
   template <class RT>
   BOOST_MATH_GPU_ENABLED RT double_factorial(unsigned i);
   template <class RT, class Policy>
   BOOST_MATH_GPU_ENABLED RT double_factorial(unsigned i, const Policy& pol);

   template <class RT>
   tools::promote_args_t<RT> falling_factorial(RT x, unsigned n);

   template <class RT, class Policy>
   tools::promote_args_t<RT> falling_factorial(RT x, unsigned n, const Policy& pol);

   template <class RT>
   tools::promote_args_t<RT> rising_factorial(RT x, int n);

   template <class RT, class Policy>
   tools::promote_args_t<RT> rising_factorial(RT x, int n, const Policy& pol);

   // Gamma functions.
   template <class RT>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> tgamma(RT z);

   template <class RT>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> tgamma1pm1(RT z);

   template <class RT, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> tgamma1pm1(RT z, const Policy& pol);

   template <class RT1, class RT2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2> tgamma(RT1 a, RT2 z);

   template <class RT1, class RT2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2> tgamma(RT1 a, RT2 z, const Policy& pol);

   template <class RT>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> lgamma(RT z, int* sign);

   template <class RT, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> lgamma(RT z, int* sign, const Policy& pol);

   template <class RT>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> lgamma(RT x);

   template <class RT, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> lgamma(RT x, const Policy& pol);

   template <class RT1, class RT2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2> tgamma_lower(RT1 a, RT2 z);

   template <class RT1, class RT2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2> tgamma_lower(RT1 a, RT2 z, const Policy&);

   template <class RT1, class RT2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2> gamma_q(RT1 a, RT2 z);

   template <class RT1, class RT2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2> gamma_q(RT1 a, RT2 z, const Policy&);

   template <class RT1, class RT2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2> gamma_p(RT1 a, RT2 z);

   template <class RT1, class RT2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT1, RT2> gamma_p(RT1 a, RT2 z, const Policy&);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> tgamma_delta_ratio(T1 z, T2 delta);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> tgamma_delta_ratio(T1 z, T2 delta, const Policy&);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> tgamma_ratio(T1 a, T2 b);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> tgamma_ratio(T1 a, T2 b, const Policy&);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> gamma_p_derivative(T1 a, T2 x);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> gamma_p_derivative(T1 a, T2 x, const Policy&);

   // gamma inverse.
   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> gamma_p_inv(T1 a, T2 p);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> gamma_p_inva(T1 a, T2 p, const Policy&);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> gamma_p_inva(T1 a, T2 p);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> gamma_p_inv(T1 a, T2 p, const Policy&);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> gamma_q_inv(T1 a, T2 q);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> gamma_q_inv(T1 a, T2 q, const Policy&);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> gamma_q_inva(T1 a, T2 q);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2> gamma_q_inva(T1 a, T2 q, const Policy&);

   // digamma:
   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> digamma(T x);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> digamma(T x, const Policy&);

   // trigamma:
   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> trigamma(T x);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> trigamma(T x, const Policy&);

   // polygamma:
   template <class T>
   tools::promote_args_t<T> polygamma(int n, T x);

   template <class T, class Policy>
   tools::promote_args_t<T> polygamma(int n, T x, const Policy&);

   // Hypotenuse function sqrt(x ^ 2 + y ^ 2).
   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2>
         hypot(T1 x, T2 y);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2>
         hypot(T1 x, T2 y, const Policy&);

   // cbrt - cube root.
   template <class RT>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> cbrt(RT z);

   template <class RT, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<RT> cbrt(RT z, const Policy&);

   // log1p is log(x + 1)
   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> log1p(T);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> log1p(T, const Policy&);

   // log1pmx is log(x + 1) - x
   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> log1pmx(T);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> log1pmx(T, const Policy&);

   // Exp (x) minus 1 functions.
   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> expm1(T);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> expm1(T, const Policy&);

   // Power - 1
   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2>
         powm1(const T1 a, const T2 z);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T1, T2>
         powm1(const T1 a, const T2 z, const Policy&);

   // sqrt(1+x) - 1
   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> sqrt1pm1(const T& val);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> sqrt1pm1(const T& val, const Policy&);

   // sinus cardinals:
   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> sinc_pi(T x);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> sinc_pi(T x, const Policy&);

   template <class T>
   tools::promote_args_t<T> sinhc_pi(T x);

   template <class T, class Policy>
   tools::promote_args_t<T> sinhc_pi(T x, const Policy&);

   // inverse hyperbolics:
   template<typename T>
   tools::promote_args_t<T> asinh(T x);

   template<typename T, class Policy>
   tools::promote_args_t<T> asinh(T x, const Policy&);

   template<typename T>
   tools::promote_args_t<T> acosh(T x);

   template<typename T, class Policy>
   tools::promote_args_t<T> acosh(T x, const Policy&);

   template<typename T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> atanh(T x);

   template<typename T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> atanh(T x, const Policy&);

   namespace detail{

      typedef boost::math::integral_constant<int, 0> bessel_no_int_tag;      // No integer optimisation possible.
      typedef boost::math::integral_constant<int, 1> bessel_maybe_int_tag;   // Maybe integer optimisation.
      typedef boost::math::integral_constant<int, 2> bessel_int_tag;         // Definite integer optimisation.

      template <class T1, class T2, class Policy>
      struct bessel_traits
      {
         using result_type = typename boost::math::conditional<
            boost::math::is_integral<T1>::value,
            typename tools::promote_args<T2>::type,
            tools::promote_args_t<T1, T2>
         >::type;

         typedef typename policies::precision<result_type, Policy>::type precision_type;

         using optimisation_tag = typename boost::math::conditional<
            (precision_type::value <= 0 || precision_type::value > 64),
            bessel_no_int_tag,
            typename boost::math::conditional<
               boost::math::is_integral<T1>::value,
               bessel_int_tag,
               bessel_maybe_int_tag
            >::type
         >::type;

         using optimisation_tag128 = typename boost::math::conditional<
            (precision_type::value <= 0 || precision_type::value > 113),
            bessel_no_int_tag,
            typename boost::math::conditional<
               boost::math::is_integral<T1>::value,
               bessel_int_tag,
               bessel_maybe_int_tag
            >::type
         >::type;
      };
   } // detail

   // Bessel functions:
   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, Policy>::result_type cyl_bessel_j(T1 v, T2 x, const Policy& pol);
   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, Policy>::result_type cyl_bessel_j_prime(T1 v, T2 x, const Policy& pol);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, policies::policy<> >::result_type cyl_bessel_j(T1 v, T2 x);
   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, policies::policy<> >::result_type cyl_bessel_j_prime(T1 v, T2 x);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T, T, Policy>::result_type sph_bessel(unsigned v, T x, const Policy& pol);
   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T, T, Policy>::result_type sph_bessel_prime(unsigned v, T x, const Policy& pol);

   template <class T>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T, T, policies::policy<> >::result_type sph_bessel(unsigned v, T x);
   template <class T>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T, T, policies::policy<> >::result_type sph_bessel_prime(unsigned v, T x);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, Policy>::result_type cyl_bessel_i(T1 v, T2 x, const Policy& pol);
   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, Policy>::result_type cyl_bessel_i_prime(T1 v, T2 x, const Policy& pol);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, policies::policy<> >::result_type cyl_bessel_i(T1 v, T2 x);
   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, policies::policy<> >::result_type cyl_bessel_i_prime(T1 v, T2 x);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, Policy>::result_type cyl_bessel_k(T1 v, T2 x, const Policy& pol);
   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, Policy>::result_type cyl_bessel_k_prime(T1 v, T2 x, const Policy& pol);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, policies::policy<> >::result_type cyl_bessel_k(T1 v, T2 x);
   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, policies::policy<> >::result_type cyl_bessel_k_prime(T1 v, T2 x);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, Policy>::result_type cyl_neumann(T1 v, T2 x, const Policy& pol);
   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, Policy>::result_type cyl_neumann_prime(T1 v, T2 x, const Policy& pol);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, policies::policy<> >::result_type cyl_neumann(T1 v, T2 x);
   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T1, T2, policies::policy<> >::result_type cyl_neumann_prime(T1 v, T2 x);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T, T, Policy>::result_type sph_neumann(unsigned v, T x, const Policy& pol);
   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T, T, Policy>::result_type sph_neumann_prime(unsigned v, T x, const Policy& pol);

   template <class T>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T, T, policies::policy<> >::result_type sph_neumann(unsigned v, T x);
   template <class T>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T, T, policies::policy<> >::result_type sph_neumann_prime(unsigned v, T x);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T, T, Policy>::result_type cyl_bessel_j_zero(T v, int m, const Policy& pol);

   template <class T>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T, T, policies::policy<> >::result_type cyl_bessel_j_zero(T v, int m);

   template <class T, class OutputIterator>
   BOOST_MATH_GPU_ENABLED OutputIterator cyl_bessel_j_zero(T v,
                          int start_index,
                          unsigned number_of_zeros,
                          OutputIterator out_it);

   template <class T, class OutputIterator, class Policy>
   BOOST_MATH_GPU_ENABLED OutputIterator cyl_bessel_j_zero(T v,
                          int start_index,
                          unsigned number_of_zeros,
                          OutputIterator out_it,
                          const Policy&);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T, T, Policy>::result_type cyl_neumann_zero(T v, int m, const Policy& pol);

   template <class T>
   BOOST_MATH_GPU_ENABLED typename detail::bessel_traits<T, T, policies::policy<> >::result_type cyl_neumann_zero(T v, int m);

   template <class T, class OutputIterator>
   BOOST_MATH_GPU_ENABLED OutputIterator cyl_neumann_zero(T v,
                         int start_index,
                         unsigned number_of_zeros,
                         OutputIterator out_it);

   template <class T, class OutputIterator, class Policy>
   BOOST_MATH_GPU_ENABLED OutputIterator cyl_neumann_zero(T v,
                         int start_index,
                         unsigned number_of_zeros,
                         OutputIterator out_it,
                         const Policy&);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED boost::math::complex<typename detail::bessel_traits<T1, T2, policies::policy<> >::result_type> cyl_hankel_1(T1 v, T2 x);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED boost::math::complex<typename detail::bessel_traits<T1, T2, Policy>::result_type> cyl_hankel_1(T1 v, T2 x, const Policy& pol);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED boost::math::complex<typename detail::bessel_traits<T1, T2, Policy>::result_type> cyl_hankel_2(T1 v, T2 x, const Policy& pol);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED boost::math::complex<typename detail::bessel_traits<T1, T2, policies::policy<> >::result_type> cyl_hankel_2(T1 v, T2 x);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED boost::math::complex<typename detail::bessel_traits<T1, T2, Policy>::result_type> sph_hankel_1(T1 v, T2 x, const Policy& pol);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED boost::math::complex<typename detail::bessel_traits<T1, T2, policies::policy<> >::result_type> sph_hankel_1(T1 v, T2 x);

   template <class T1, class T2, class Policy>
   BOOST_MATH_GPU_ENABLED boost::math::complex<typename detail::bessel_traits<T1, T2, Policy>::result_type> sph_hankel_2(T1 v, T2 x, const Policy& pol);

   template <class T1, class T2>
   BOOST_MATH_GPU_ENABLED boost::math::complex<typename detail::bessel_traits<T1, T2, policies::policy<> >::result_type> sph_hankel_2(T1 v, T2 x);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> airy_ai(T x, const Policy&);

   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> airy_ai(T x);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> airy_bi(T x, const Policy&);

   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> airy_bi(T x);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> airy_ai_prime(T x, const Policy&);

   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> airy_ai_prime(T x);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> airy_bi_prime(T x, const Policy&);

   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> airy_bi_prime(T x);

   template <class T>
   BOOST_MATH_GPU_ENABLED T airy_ai_zero(int m);
   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED T airy_ai_zero(int m, const Policy&);

   template <class OutputIterator>
   BOOST_MATH_GPU_ENABLED OutputIterator airy_ai_zero(
                     int start_index,
                     unsigned number_of_zeros,
                     OutputIterator out_it);
   template <class OutputIterator, class Policy>
   BOOST_MATH_GPU_ENABLED OutputIterator airy_ai_zero(
                     int start_index,
                     unsigned number_of_zeros,
                     OutputIterator out_it,
                     const Policy&);

   template <class T>
   BOOST_MATH_GPU_ENABLED T airy_bi_zero(int m);
   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED T airy_bi_zero(int m, const Policy&);

   template <class OutputIterator>
   BOOST_MATH_GPU_ENABLED OutputIterator airy_bi_zero(
                     int start_index,
                     unsigned number_of_zeros,
                     OutputIterator out_it);
   template <class OutputIterator, class Policy>
   BOOST_MATH_GPU_ENABLED OutputIterator airy_bi_zero(
                     int start_index,
                     unsigned number_of_zeros,
                     OutputIterator out_it,
                     const Policy&);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> sin_pi(T x, const Policy&);

   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> sin_pi(T x);

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> cos_pi(T x, const Policy&);

   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> cos_pi(T x);

   template <class T>
   BOOST_MATH_GPU_ENABLED int fpclassify BOOST_NO_MACRO_EXPAND(T t);

   template <class T>
   BOOST_MATH_GPU_ENABLED bool isfinite BOOST_NO_MACRO_EXPAND(T z);

   template <class T>
   BOOST_MATH_GPU_ENABLED bool isinf BOOST_NO_MACRO_EXPAND(T t);

   template <class T>
   BOOST_MATH_GPU_ENABLED bool isnan BOOST_NO_MACRO_EXPAND(T t);

   template <class T>
   BOOST_MATH_GPU_ENABLED bool isnormal BOOST_NO_MACRO_EXPAND(T t);

   template<class T>
   BOOST_MATH_GPU_ENABLED int signbit BOOST_NO_MACRO_EXPAND(T x);

   template <class T>
   BOOST_MATH_GPU_ENABLED int sign BOOST_NO_MACRO_EXPAND(const T& z);

   template <class T, class U>
   BOOST_MATH_GPU_ENABLED typename tools::promote_args_permissive<T, U>::type 
   copysign BOOST_NO_MACRO_EXPAND(const T& x, const U& y);

   template <class T>
   BOOST_MATH_GPU_ENABLED typename tools::promote_args_permissive<T>::type 
   changesign BOOST_NO_MACRO_EXPAND(const T& z);

   // Exponential integrals:
   namespace detail{

   template <class T, class U>
   struct expint_result
   {
      typedef typename boost::math::conditional<
         policies::is_policy<U>::value,
         tools::promote_args_t<T>,
         typename tools::promote_args<U>::type
      >::type type;
   };

   } // namespace detail

   template <class T, class Policy>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> expint(unsigned n, T z, const Policy&);

   template <class T, class U>
   BOOST_MATH_GPU_ENABLED typename detail::expint_result<T, U>::type expint(T const z, U const u);

   template <class T>
   BOOST_MATH_GPU_ENABLED tools::promote_args_t<T> expint(T z);

   // Zeta:
   template <class T, class Policy>
   tools::promote_args_t<T> zeta(T s, const Policy&);

   // Owen's T function:
   template <class T1, class T2, class Policy>
   tools::promote_args_t<T1, T2> owens_t(T1 h, T2 a, const Policy& pol);

   template <class T1, class T2>
   tools::promote_args_t<T1, T2> owens_t(T1 h, T2 a);

   // Jacobi Functions:
   template <class T, class U, class V, class Policy>
   tools::promote_args_t<T, U, V> jacobi_elliptic(T k, U theta, V* pcn, V* pdn, const Policy&);

   template <class T, class U, class V>
   tools::promote_args_t<T, U, V> jacobi_elliptic(T k, U theta, V* pcn = 0, V* pdn = 0);

   template <class U, class T, class Policy>
   tools::promote_args_t<T, U> jacobi_sn(U k, T theta, const Policy& pol);

   template <class U, class T>
   tools::promote_args_t<T, U> jacobi_sn(U k, T theta);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_cn(T k, U theta, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_cn(T k, U theta);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_dn(T k, U theta, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_dn(T k, U theta);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_cd(T k, U theta, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_cd(T k, U theta);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_dc(T k, U theta, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_dc(T k, U theta);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_ns(T k, U theta, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_ns(T k, U theta);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_sd(T k, U theta, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_sd(T k, U theta);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_ds(T k, U theta, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_ds(T k, U theta);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_nc(T k, U theta, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_nc(T k, U theta);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_nd(T k, U theta, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_nd(T k, U theta);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_sc(T k, U theta, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_sc(T k, U theta);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_cs(T k, U theta, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_cs(T k, U theta);

   // Jacobi Theta Functions:
   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_theta1(T z, U q, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_theta1(T z, U q);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_theta2(T z, U q, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_theta2(T z, U q);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_theta3(T z, U q, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_theta3(T z, U q);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_theta4(T z, U q, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_theta4(T z, U q);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_theta1tau(T z, U tau, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_theta1tau(T z, U tau);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_theta2tau(T z, U tau, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_theta2tau(T z, U tau);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_theta3tau(T z, U tau, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_theta3tau(T z, U tau);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_theta4tau(T z, U tau, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_theta4tau(T z, U tau);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_theta3m1(T z, U q, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_theta3m1(T z, U q);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_theta4m1(T z, U q, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_theta4m1(T z, U q);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_theta3m1tau(T z, U tau, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_theta3m1tau(T z, U tau);

   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> jacobi_theta4m1tau(T z, U tau, const Policy& pol);

   template <class T, class U>
   tools::promote_args_t<T, U> jacobi_theta4m1tau(T z, U tau);


   template <class T>
   tools::promote_args_t<T> zeta(T s);

   // pow:
   template <int N, typename T, class Policy>
   BOOST_MATH_GPU_ENABLED BOOST_MATH_CXX14_CONSTEXPR tools::promote_args_t<T> pow(T base, const Policy& policy);

   template <int N, typename T>
   BOOST_MATH_GPU_ENABLED BOOST_MATH_CXX14_CONSTEXPR tools::promote_args_t<T> pow(T base);

   // next:
   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> nextafter(const T&, const U&, const Policy&);
   template <class T, class U>
   tools::promote_args_t<T, U> nextafter(const T&, const U&);
   template <class T, class Policy>
   tools::promote_args_t<T> float_next(const T&, const Policy&);
   template <class T>
   tools::promote_args_t<T> float_next(const T&);
   template <class T, class Policy>
   tools::promote_args_t<T> float_prior(const T&, const Policy&);
   template <class T>
   tools::promote_args_t<T> float_prior(const T&);
   template <class T, class U, class Policy>
   tools::promote_args_t<T, U> float_distance(const T&, const U&, const Policy&);
   template <class T, class U>
   tools::promote_args_t<T, U> float_distance(const T&, const U&);
   template <class T, class Policy>
   tools::promote_args_t<T> float_advance(T val, int distance, const Policy& pol);
   template <class T>
   tools::promote_args_t<T> float_advance(const T& val, int distance);

   template <class T, class Policy>
   tools::promote_args_t<T> ulp(const T& val, const Policy& pol);
   template <class T>
   tools::promote_args_t<T> ulp(const T& val);

   template <class T, class U>
   tools::promote_args_t<T, U> relative_difference(const T&, const U&);
   template <class T, class U>
   tools::promote_args_t<T, U> epsilon_difference(const T&, const U&);

   template<class T>
   BOOST_MATH_CONSTEXPR_TABLE_FUNCTION T unchecked_bernoulli_b2n(const std::size_t n);
   template <class T, class Policy>
   T bernoulli_b2n(const int i, const Policy &pol);
   template <class T>
   T bernoulli_b2n(const int i);
   template <class T, class OutputIterator, class Policy>
   OutputIterator bernoulli_b2n(const int start_index,
                                       const unsigned number_of_bernoullis_b2n,
                                       OutputIterator out_it,
                                       const Policy& pol);
   template <class T, class OutputIterator>
   OutputIterator bernoulli_b2n(const int start_index,
                                       const unsigned number_of_bernoullis_b2n,
                                       OutputIterator out_it);
   template <class T, class Policy>
   T tangent_t2n(const int i, const Policy &pol);
   template <class T>
   T tangent_t2n(const int i);
   template <class T, class OutputIterator, class Policy>
   OutputIterator tangent_t2n(const int start_index,
                                       const unsigned number_of_bernoullis_b2n,
                                       OutputIterator out_it,
                                       const Policy& pol);
   template <class T, class OutputIterator>
   OutputIterator tangent_t2n(const int start_index,
                                       const unsigned number_of_bernoullis_b2n,
                                       OutputIterator out_it);

   // Lambert W:
   template <class T, class Policy>
   boost::math::tools::promote_args_t<T> lambert_w0(T z, const Policy& pol);
   template <class T>
   boost::math::tools::promote_args_t<T> lambert_w0(T z);
   template <class T, class Policy>
   boost::math::tools::promote_args_t<T> lambert_wm1(T z, const Policy& pol);
   template <class T>
   boost::math::tools::promote_args_t<T> lambert_wm1(T z);
   template <class T, class Policy>
   boost::math::tools::promote_args_t<T> lambert_w0_prime(T z, const Policy& pol);
   template <class T>
   boost::math::tools::promote_args_t<T> lambert_w0_prime(T z);
   template <class T, class Policy>
   boost::math::tools::promote_args_t<T> lambert_wm1_prime(T z, const Policy& pol);
   template <class T>
   boost::math::tools::promote_args_t<T> lambert_wm1_prime(T z);

   // Hypergeometrics:
   template <class T1, class T2> tools::promote_args_t<T1, T2> hypergeometric_1F0(T1 a, T2 z);
   template <class T1, class T2, class Policy> tools::promote_args_t<T1, T2> hypergeometric_1F0(T1 a, T2 z, const Policy&);

   template <class T1, class T2> tools::promote_args_t<T1, T2> hypergeometric_0F1(T1 b, T2 z);
   template <class T1, class T2, class Policy> tools::promote_args_t<T1, T2> hypergeometric_0F1(T1 b, T2 z, const Policy&);

   template <class T1, class T2, class T3> tools::promote_args_t<T1, T2, T3> hypergeometric_2F0(T1 a1, T2 a2, T3 z);
   template <class T1, class T2, class T3, class Policy> tools::promote_args_t<T1, T2, T3> hypergeometric_2F0(T1 a1, T2 a2, T3 z, const Policy&);

   template <class T1, class T2, class T3> tools::promote_args_t<T1, T2, T3> hypergeometric_1F1(T1 a, T2 b, T3 z);
   template <class T1, class T2, class T3, class Policy> tools::promote_args_t<T1, T2, T3> hypergeometric_1F1(T1 a, T2 b, T3 z, const Policy&);


    } // namespace math
} // namespace boost

#define BOOST_MATH_DETAIL_LL_FUNC(Policy)\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline T modf(const T& v, long long* ipart){ using boost::math::modf; return modf(v, ipart, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline long long lltrunc(const T& v){ using boost::math::lltrunc; return lltrunc(v, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline long long llround(const T& v){ using boost::math::llround; return llround(v, Policy()); }\

#  define BOOST_MATH_DETAIL_11_FUNC(Policy)\
   template <class T, class U, class V>\
   inline boost::math::tools::promote_args_t<T, U> hypergeometric_1F1(const T& a, const U& b, const V& z)\
   { return boost::math::hypergeometric_1F1(a, b, z, Policy()); }\

#define BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS(Policy)\
   \
   BOOST_MATH_DETAIL_LL_FUNC(Policy)\
   BOOST_MATH_DETAIL_11_FUNC(Policy)\
   \
   template <class RT1, class RT2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2> \
   beta(RT1 a, RT2 b) { return ::boost::math::beta(a, b, Policy()); }\
\
   template <class RT1, class RT2, class A>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2, A> \
   beta(RT1 a, RT2 b, A x){ return ::boost::math::beta(a, b, x, Policy()); }\
\
   template <class RT1, class RT2, class RT3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2, RT3> \
   betac(RT1 a, RT2 b, RT3 x) { return ::boost::math::betac(a, b, x, Policy()); }\
\
   template <class RT1, class RT2, class RT3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2, RT3> \
   ibeta(RT1 a, RT2 b, RT3 x){ return ::boost::math::ibeta(a, b, x, Policy()); }\
\
   template <class RT1, class RT2, class RT3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2, RT3> \
   ibetac(RT1 a, RT2 b, RT3 x){ return ::boost::math::ibetac(a, b, x, Policy()); }\
\
   template <class T1, class T2, class T3, class T4>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2, T3, T4>  \
   ibeta_inv(T1 a, T2 b, T3 p, T4* py){ return ::boost::math::ibeta_inv(a, b, p, py, Policy()); }\
\
   template <class RT1, class RT2, class RT3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2, RT3> \
   ibeta_inv(RT1 a, RT2 b, RT3 p){ return ::boost::math::ibeta_inv(a, b, p, Policy()); }\
\
   template <class T1, class T2, class T3, class T4>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2, T3, T4> \
   ibetac_inv(T1 a, T2 b, T3 q, T4* py){ return ::boost::math::ibetac_inv(a, b, q, py, Policy()); }\
\
   template <class RT1, class RT2, class RT3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2, RT3> \
   ibeta_inva(RT1 a, RT2 b, RT3 p){ return ::boost::math::ibeta_inva(a, b, p, Policy()); }\
\
   template <class T1, class T2, class T3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2, T3> \
   ibetac_inva(T1 a, T2 b, T3 q){ return ::boost::math::ibetac_inva(a, b, q, Policy()); }\
\
   template <class RT1, class RT2, class RT3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2, RT3> \
   ibeta_invb(RT1 a, RT2 b, RT3 p){ return ::boost::math::ibeta_invb(a, b, p, Policy()); }\
\
   template <class T1, class T2, class T3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2, T3> \
   ibetac_invb(T1 a, T2 b, T3 q){ return ::boost::math::ibetac_invb(a, b, q, Policy()); }\
\
   template <class RT1, class RT2, class RT3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2, RT3> \
   ibetac_inv(RT1 a, RT2 b, RT3 q){ return ::boost::math::ibetac_inv(a, b, q, Policy()); }\
\
   template <class RT1, class RT2, class RT3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2, RT3> \
   ibeta_derivative(RT1 a, RT2 b, RT3 x){ return ::boost::math::ibeta_derivative(a, b, x, Policy()); }\
\
   template <class T> BOOST_MATH_GPU_ENABLED T binomial_coefficient(unsigned n, unsigned k){ return ::boost::math::binomial_coefficient<T, Policy>(n, k, Policy()); }\
\
   template <class RT>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT> erf(RT z) { return ::boost::math::erf(z, Policy()); }\
\
   template <class RT>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT> erfc(RT z){ return ::boost::math::erfc(z, Policy()); }\
\
   template <class RT>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT> erf_inv(RT z) { return ::boost::math::erf_inv(z, Policy()); }\
\
   template <class RT>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT> erfc_inv(RT z){ return ::boost::math::erfc_inv(z, Policy()); }\
\
   using boost::math::legendre_next;\
\
   template <class T>\
   inline boost::math::tools::promote_args_t<T> \
   legendre_p(int l, T x){ return ::boost::math::legendre_p(l, x, Policy()); }\
\
   template <class T>\
   inline boost::math::tools::promote_args_t<T> \
   legendre_p_prime(int l, T x){ return ::boost::math::legendre_p(l, x, Policy()); }\
\
   template <class T>\
   inline boost::math::tools::promote_args_t<T> \
   legendre_q(unsigned l, T x){ return ::boost::math::legendre_q(l, x, Policy()); }\
\
   using ::boost::math::legendre_next;\
\
   template <class T>\
   inline boost::math::tools::promote_args_t<T> \
   legendre_p(int l, int m, T x){ return ::boost::math::legendre_p(l, m, x, Policy()); }\
\
   using ::boost::math::laguerre_next;\
\
   template <class T>\
   inline boost::math::tools::promote_args_t<T> \
   laguerre(unsigned n, T x){ return ::boost::math::laguerre(n, x, Policy()); }\
\
   template <class T1, class T2>\
   inline typename boost::math::laguerre_result<T1, T2>::type \
   laguerre(unsigned n, T1 m, T2 x) { return ::boost::math::laguerre(n, m, x, Policy()); }\
\
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> \
   hermite(unsigned n, T x){ return ::boost::math::hermite(n, x, Policy()); }\
\
   using boost::math::hermite_next;\
\
   using boost::math::chebyshev_next;\
\
  template<class Real>\
  Real chebyshev_t(unsigned n, Real const & x){ return ::boost::math::chebyshev_t(n, x, Policy()); }\
\
  template<class Real>\
  Real chebyshev_u(unsigned n, Real const & x){ return ::boost::math::chebyshev_u(n, x, Policy()); }\
\
  template<class Real>\
  Real chebyshev_t_prime(unsigned n, Real const & x){ return ::boost::math::chebyshev_t_prime(n, x, Policy()); }\
\
  using ::boost::math::chebyshev_clenshaw_recurrence;\
\
   template <class T1, class T2>\
   inline std::complex<boost::math::tools::promote_args_t<T1, T2>> \
   spherical_harmonic(unsigned n, int m, T1 theta, T2 phi){ return boost::math::spherical_harmonic(n, m, theta, phi, Policy()); }\
\
   template <class T1, class T2>\
   inline boost::math::tools::promote_args_t<T1, T2> \
   spherical_harmonic_r(unsigned n, int m, T1 theta, T2 phi){ return ::boost::math::spherical_harmonic_r(n, m, theta, phi, Policy()); }\
\
   template <class T1, class T2>\
   inline boost::math::tools::promote_args_t<T1, T2> \
   spherical_harmonic_i(unsigned n, int m, T1 theta, T2 phi){ return boost::math::spherical_harmonic_i(n, m, theta, phi, Policy()); }\
\
   template <class T1, class T2, class Policy>\
   inline boost::math::tools::promote_args_t<T1, T2> \
      spherical_harmonic_i(unsigned n, int m, T1 theta, T2 phi, const Policy& pol);\
\
   template <class T1, class T2, class T3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2, T3> \
   ellint_rf(T1 x, T2 y, T3 z){ return ::boost::math::ellint_rf(x, y, z, Policy()); }\
\
   template <class T1, class T2, class T3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2, T3> \
   ellint_rd(T1 x, T2 y, T3 z){ return ::boost::math::ellint_rd(x, y, z, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> \
   ellint_rc(T1 x, T2 y){ return ::boost::math::ellint_rc(x, y, Policy()); }\
\
   template <class T1, class T2, class T3, class T4>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2, T3, T4> \
   ellint_rj(T1 x, T2 y, T3 z, T4 p){ return boost::math::ellint_rj(x, y, z, p, Policy()); }\
\
   template <class T1, class T2, class T3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2, T3> \
   ellint_rg(T1 x, T2 y, T3 z){ return ::boost::math::ellint_rg(x, y, z, Policy()); }\
   \
   template <typename T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> ellint_2(T k){ return boost::math::ellint_2(k, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> ellint_2(T1 k, T2 phi){ return boost::math::ellint_2(k, phi, Policy()); }\
\
   template <typename T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> ellint_d(T k){ return boost::math::ellint_d(k, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> ellint_d(T1 k, T2 phi){ return boost::math::ellint_d(k, phi, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> jacobi_zeta(T1 k, T2 phi){ return boost::math::jacobi_zeta(k, phi, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> heuman_lambda(T1 k, T2 phi){ return boost::math::heuman_lambda(k, phi, Policy()); }\
\
   template <typename T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> ellint_1(T k){ return boost::math::ellint_1(k, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> ellint_1(T1 k, T2 phi){ return boost::math::ellint_1(k, phi, Policy()); }\
\
   template <class T1, class T2, class T3>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2, T3> ellint_3(T1 k, T2 v, T3 phi){ return boost::math::ellint_3(k, v, phi, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> ellint_3(T1 k, T2 v){ return boost::math::ellint_3(k, v, Policy()); }\
\
   using boost::math::max_factorial;\
   template <class RT>\
   BOOST_MATH_GPU_ENABLED inline RT factorial(unsigned int i) { return boost::math::factorial<RT>(i, Policy()); }\
   using boost::math::unchecked_factorial;\
   template <class RT>\
   BOOST_MATH_GPU_ENABLED inline RT double_factorial(unsigned i){ return boost::math::double_factorial<RT>(i, Policy()); }\
   template <class RT>\
   inline boost::math::tools::promote_args_t<RT> falling_factorial(RT x, unsigned n){ return boost::math::falling_factorial(x, n, Policy()); }\
   template <class RT>\
   inline boost::math::tools::promote_args_t<RT> rising_factorial(RT x, unsigned n){ return boost::math::rising_factorial(x, n, Policy()); }\
\
   template <class RT>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT> tgamma(RT z){ return boost::math::tgamma(z, Policy()); }\
\
   template <class RT>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT> tgamma1pm1(RT z){ return boost::math::tgamma1pm1(z, Policy()); }\
\
   template <class RT1, class RT2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2> tgamma(RT1 a, RT2 z){ return boost::math::tgamma(a, z, Policy()); }\
\
   template <class RT>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT> lgamma(RT z, int* sign){ return boost::math::lgamma(z, sign, Policy()); }\
\
   template <class RT>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT> lgamma(RT x){ return boost::math::lgamma(x, Policy()); }\
\
   template <class RT1, class RT2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2> tgamma_lower(RT1 a, RT2 z){ return boost::math::tgamma_lower(a, z, Policy()); }\
\
   template <class RT1, class RT2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2> gamma_q(RT1 a, RT2 z){ return boost::math::gamma_q(a, z, Policy()); }\
\
   template <class RT1, class RT2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<RT1, RT2> gamma_p(RT1 a, RT2 z){ return boost::math::gamma_p(a, z, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> tgamma_delta_ratio(T1 z, T2 delta){ return boost::math::tgamma_delta_ratio(z, delta, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> tgamma_ratio(T1 a, T2 b) { return boost::math::tgamma_ratio(a, b, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> gamma_p_derivative(T1 a, T2 x){ return boost::math::gamma_p_derivative(a, x, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> gamma_p_inv(T1 a, T2 p){ return boost::math::gamma_p_inv(a, p, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> gamma_p_inva(T1 a, T2 p){ return boost::math::gamma_p_inva(a, p, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> gamma_q_inv(T1 a, T2 q){ return boost::math::gamma_q_inv(a, q, Policy()); }\
\
   template <class T1, class T2>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T1, T2> gamma_q_inva(T1 a, T2 q){ return boost::math::gamma_q_inva(a, q, Policy()); }\
\
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> digamma(T x){ return boost::math::digamma(x, Policy()); }\
\
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> trigamma(T x){ return boost::math::trigamma(x, Policy()); }\
\
   template <class T>\
   inline boost::math::tools::promote_args_t<T> polygamma(int n, T x){ return boost::math::polygamma(n, x, Policy()); }\
   \
   template <class T1, class T2>\
   inline boost::math::tools::promote_args_t<T1, T2> \
   BOOST_MATH_GPU_ENABLED hypot(T1 x, T2 y){ return boost::math::hypot(x, y, Policy()); }\
\
   template <class RT>\
   inline boost::math::tools::promote_args_t<RT> cbrt(RT z){ return boost::math::cbrt(z, Policy()); }\
\
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> log1p(T x){ return boost::math::log1p(x, Policy()); }\
\
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> log1pmx(T x){ return boost::math::log1pmx(x, Policy()); }\
\
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> expm1(T x){ return boost::math::expm1(x, Policy()); }\
\
   template <class T1, class T2>\
   inline boost::math::tools::promote_args_t<T1, T2> \
   BOOST_MATH_GPU_ENABLED powm1(const T1 a, const T2 z){ return boost::math::powm1(a, z, Policy()); }\
\
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> sqrt1pm1(const T& val){ return boost::math::sqrt1pm1(val, Policy()); }\
\
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> sinc_pi(T x){ return boost::math::sinc_pi(x, Policy()); }\
\
   template <class T>\
   inline boost::math::tools::promote_args_t<T> sinhc_pi(T x){ return boost::math::sinhc_pi(x, Policy()); }\
\
   template<typename T>\
   inline boost::math::tools::promote_args_t<T> asinh(const T x){ return boost::math::asinh(x, Policy()); }\
\
   template<typename T>\
   inline boost::math::tools::promote_args_t<T> acosh(const T x){ return boost::math::acosh(x, Policy()); }\
\
   template<typename T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> atanh(const T x){ return boost::math::atanh(x, Policy()); }\
\
   template <class T1, class T2>\
   inline typename boost::math::detail::bessel_traits<T1, T2, Policy >::result_type cyl_bessel_j(T1 v, T2 x)\
   { return boost::math::cyl_bessel_j(v, x, Policy()); }\
\
   template <class T1, class T2>\
   inline typename boost::math::detail::bessel_traits<T1, T2, Policy >::result_type cyl_bessel_j_prime(T1 v, T2 x)\
   { return boost::math::cyl_bessel_j_prime(v, x, Policy()); }\
\
   template <class T>\
   inline typename boost::math::detail::bessel_traits<T, T, Policy >::result_type sph_bessel(unsigned v, T x)\
   { return boost::math::sph_bessel(v, x, Policy()); }\
\
   template <class T>\
   inline typename boost::math::detail::bessel_traits<T, T, Policy >::result_type sph_bessel_prime(unsigned v, T x)\
   { return boost::math::sph_bessel_prime(v, x, Policy()); }\
\
   template <class T1, class T2>\
   inline typename boost::math::detail::bessel_traits<T1, T2, Policy >::result_type \
   cyl_bessel_i(T1 v, T2 x) { return boost::math::cyl_bessel_i(v, x, Policy()); }\
\
   template <class T1, class T2>\
   inline typename boost::math::detail::bessel_traits<T1, T2, Policy >::result_type \
   cyl_bessel_i_prime(T1 v, T2 x) { return boost::math::cyl_bessel_i_prime(v, x, Policy()); }\
\
   template <class T1, class T2>\
   inline typename boost::math::detail::bessel_traits<T1, T2, Policy >::result_type \
   cyl_bessel_k(T1 v, T2 x) { return boost::math::cyl_bessel_k(v, x, Policy()); }\
\
   template <class T1, class T2>\
   inline typename boost::math::detail::bessel_traits<T1, T2, Policy >::result_type \
   cyl_bessel_k_prime(T1 v, T2 x) { return boost::math::cyl_bessel_k_prime(v, x, Policy()); }\
\
   template <class T1, class T2>\
   inline typename boost::math::detail::bessel_traits<T1, T2, Policy >::result_type \
   cyl_neumann(T1 v, T2 x){ return boost::math::cyl_neumann(v, x, Policy()); }\
\
   template <class T1, class T2>\
   inline typename boost::math::detail::bessel_traits<T1, T2, Policy >::result_type \
   cyl_neumann_prime(T1 v, T2 x){ return boost::math::cyl_neumann_prime(v, x, Policy()); }\
\
   template <class T>\
   inline typename boost::math::detail::bessel_traits<T, T, Policy >::result_type \
   sph_neumann(unsigned v, T x){ return boost::math::sph_neumann(v, x, Policy()); }\
\
   template <class T>\
   inline typename boost::math::detail::bessel_traits<T, T, Policy >::result_type \
   sph_neumann_prime(unsigned v, T x){ return boost::math::sph_neumann_prime(v, x, Policy()); }\
\
   template <class T>\
   inline typename boost::math::detail::bessel_traits<T, T, Policy >::result_type cyl_bessel_j_zero(T v, int m)\
   { return boost::math::cyl_bessel_j_zero(v, m, Policy()); }\
\
template <class OutputIterator, class T>\
   inline void cyl_bessel_j_zero(T v,\
                                 int start_index,\
                                 unsigned number_of_zeros,\
                                 OutputIterator out_it)\
   { boost::math::cyl_bessel_j_zero(v, start_index, number_of_zeros, out_it, Policy()); }\
\
   template <class T>\
   inline typename boost::math::detail::bessel_traits<T, T, Policy >::result_type cyl_neumann_zero(T v, int m)\
   { return boost::math::cyl_neumann_zero(v, m, Policy()); }\
\
template <class OutputIterator, class T>\
   inline void cyl_neumann_zero(T v,\
                                int start_index,\
                                unsigned number_of_zeros,\
                                OutputIterator out_it)\
   { boost::math::cyl_neumann_zero(v, start_index, number_of_zeros, out_it, Policy()); }\
\
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> sin_pi(T x){ return boost::math::sin_pi(x, Policy()); }\
\
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> cos_pi(T x){ return boost::math::cos_pi(x, Policy()); }\
\
   using boost::math::fpclassify;\
   using boost::math::isfinite;\
   using boost::math::isinf;\
   using boost::math::isnan;\
   using boost::math::isnormal;\
   using boost::math::signbit;\
   using boost::math::sign;\
   using boost::math::copysign;\
   using boost::math::changesign;\
   \
   template <class T, class U>\
   BOOST_MATH_GPU_ENABLED inline typename boost::math::tools::promote_args_t<T,U> expint(T const& z, U const& u)\
   { return boost::math::expint(z, u, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> expint(T z){ return boost::math::expint(z, Policy()); }\
   \
   template <class T>\
   inline boost::math::tools::promote_args_t<T> zeta(T s){ return boost::math::zeta(s, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline T round(const T& v){ using boost::math::round; return round(v, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline int iround(const T& v){ using boost::math::iround; return iround(v, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline long lround(const T& v){ using boost::math::lround; return lround(v, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline T trunc(const T& v){ using boost::math::trunc; return trunc(v, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline int itrunc(const T& v){ using boost::math::itrunc; return itrunc(v, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline long ltrunc(const T& v){ using boost::math::ltrunc; return ltrunc(v, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline T modf(const T& v, T* ipart){ using boost::math::modf; return modf(v, ipart, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline T modf(const T& v, int* ipart){ using boost::math::modf; return modf(v, ipart, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline T modf(const T& v, long* ipart){ using boost::math::modf; return modf(v, ipart, Policy()); }\
   \
   template <int N, class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> pow(T v){ return boost::math::pow<N>(v, Policy()); }\
   \
   template <class T> T nextafter(const T& a, const T& b){ return static_cast<T>(boost::math::nextafter(a, b, Policy())); }\
   template <class T> T float_next(const T& a){ return static_cast<T>(boost::math::float_next(a, Policy())); }\
   template <class T> T float_prior(const T& a){ return static_cast<T>(boost::math::float_prior(a, Policy())); }\
   template <class T> T float_distance(const T& a, const T& b){ return static_cast<T>(boost::math::float_distance(a, b, Policy())); }\
   template <class T> T ulp(const T& a){ return static_cast<T>(boost::math::ulp(a, Policy())); }\
   \
   template <class RT1, class RT2>\
   inline boost::math::tools::promote_args_t<RT1, RT2> owens_t(RT1 a, RT2 z){ return boost::math::owens_t(a, z, Policy()); }\
   \
   template <class T1, class T2>\
   inline BOOST_MATH_GPU_ENABLED boost::math::complex<typename boost::math::detail::bessel_traits<T1, T2, Policy >::result_type> cyl_hankel_1(T1 v, T2 x)\
   {  return boost::math::cyl_hankel_1(v, x, Policy()); }\
   \
   template <class T1, class T2>\
   inline BOOST_MATH_GPU_ENABLED boost::math::complex<typename boost::math::detail::bessel_traits<T1, T2, Policy >::result_type> cyl_hankel_2(T1 v, T2 x)\
   { return boost::math::cyl_hankel_2(v, x, Policy()); }\
   \
   template <class T1, class T2>\
   inline BOOST_MATH_GPU_ENABLED boost::math::complex<typename boost::math::detail::bessel_traits<T1, T2, Policy >::result_type> sph_hankel_1(T1 v, T2 x)\
   { return boost::math::sph_hankel_1(v, x, Policy()); }\
   \
   template <class T1, class T2>\
   inline BOOST_MATH_GPU_ENABLED boost::math::complex<typename boost::math::detail::bessel_traits<T1, T2, Policy >::result_type> sph_hankel_2(T1 v, T2 x)\
   { return boost::math::sph_hankel_2(v, x, Policy()); }\
   \
   template <class T>\
   inline boost::math::tools::promote_args_t<T> jacobi_elliptic(T k, T theta, T* pcn, T* pdn)\
   { return static_cast<boost::math::tools::promote_args_t<T>>(boost::math::jacobi_elliptic(k, theta, pcn, pdn, Policy())); }\
   \
   template <class U, class T>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_sn(U k, T theta)\
   { return boost::math::jacobi_sn(k, theta, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_cn(T k, U theta)\
   { return boost::math::jacobi_cn(k, theta, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_dn(T k, U theta)\
   { return boost::math::jacobi_dn(k, theta, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_cd(T k, U theta)\
   { return boost::math::jacobi_cd(k, theta, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_dc(T k, U theta)\
   { return boost::math::jacobi_dc(k, theta, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_ns(T k, U theta)\
   { return boost::math::jacobi_ns(k, theta, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_sd(T k, U theta)\
   { return boost::math::jacobi_sd(k, theta, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_ds(T k, U theta)\
   { return boost::math::jacobi_ds(k, theta, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_nc(T k, U theta)\
   { return boost::math::jacobi_nc(k, theta, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_nd(T k, U theta)\
   { return boost::math::jacobi_nd(k, theta, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_sc(T k, U theta)\
   { return boost::math::jacobi_sc(k, theta, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_cs(T k, U theta)\
   { return boost::math::jacobi_cs(k, theta, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_theta1(T z, U q)\
   { return boost::math::jacobi_theta1(z, q, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_theta2(T z, U q)\
   { return boost::math::jacobi_theta2(z, q, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_theta3(T z, U q)\
   { return boost::math::jacobi_theta3(z, q, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_theta4(T z, U q)\
   { return boost::math::jacobi_theta4(z, q, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_theta1tau(T z, U q)\
   { return boost::math::jacobi_theta1tau(z, q, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_theta2tau(T z, U q)\
   { return boost::math::jacobi_theta2tau(z, q, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_theta3tau(T z, U q)\
   { return boost::math::jacobi_theta3tau(z, q, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_theta4tau(T z, U q)\
   { return boost::math::jacobi_theta4tau(z, q, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_theta3m1(T z, U q)\
   { return boost::math::jacobi_theta3m1(z, q, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_theta4m1(T z, U q)\
   { return boost::math::jacobi_theta4m1(z, q, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_theta3m1tau(T z, U q)\
   { return boost::math::jacobi_theta3m1tau(z, q, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> jacobi_theta4m1tau(T z, U q)\
   { return boost::math::jacobi_theta4m1tau(z, q, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> airy_ai(T x)\
   {  return boost::math::airy_ai(x, Policy());  }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> airy_bi(T x)\
   {  return boost::math::airy_bi(x, Policy());  }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> airy_ai_prime(T x)\
   {  return boost::math::airy_ai_prime(x, Policy());  }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline boost::math::tools::promote_args_t<T> airy_bi_prime(T x)\
   {  return boost::math::airy_bi_prime(x, Policy());  }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline T airy_ai_zero(int m)\
   { return boost::math::airy_ai_zero<T>(m, Policy()); }\
   template <class T, class OutputIterator>\
   BOOST_MATH_GPU_ENABLED OutputIterator airy_ai_zero(int start_index, unsigned number_of_zeros, OutputIterator out_it)\
   { return boost::math::airy_ai_zero<T>(start_index, number_of_zeros, out_it, Policy()); }\
   \
   template <class T>\
   BOOST_MATH_GPU_ENABLED inline T airy_bi_zero(int m)\
   { return boost::math::airy_bi_zero<T>(m, Policy()); }\
   template <class T, class OutputIterator>\
   BOOST_MATH_GPU_ENABLED OutputIterator airy_bi_zero(int start_index, unsigned number_of_zeros, OutputIterator out_it)\
   { return boost::math::airy_bi_zero<T>(start_index, number_of_zeros, out_it, Policy()); }\
   \
   template <class T>\
   T bernoulli_b2n(const int i)\
   { return boost::math::bernoulli_b2n<T>(i, Policy()); }\
   template <class T, class OutputIterator>\
   OutputIterator bernoulli_b2n(int start_index, unsigned number_of_bernoullis_b2n, OutputIterator out_it)\
   { return boost::math::bernoulli_b2n<T>(start_index, number_of_bernoullis_b2n, out_it, Policy()); }\
   \
   template <class T>\
   T tangent_t2n(const int i)\
   { return boost::math::tangent_t2n<T>(i, Policy()); }\
   template <class T, class OutputIterator>\
   OutputIterator tangent_t2n(int start_index, unsigned number_of_bernoullis_b2n, OutputIterator out_it)\
   { return boost::math::tangent_t2n<T>(start_index, number_of_bernoullis_b2n, out_it, Policy()); }\
   \
   template <class T> inline boost::math::tools::promote_args_t<T> lambert_w0(T z) { return boost::math::lambert_w0(z, Policy()); }\
   template <class T> inline boost::math::tools::promote_args_t<T> lambert_wm1(T z) { return boost::math::lambert_w0(z, Policy()); }\
   template <class T> inline boost::math::tools::promote_args_t<T> lambert_w0_prime(T z) { return boost::math::lambert_w0(z, Policy()); }\
   template <class T> inline boost::math::tools::promote_args_t<T> lambert_wm1_prime(T z) { return boost::math::lambert_w0(z, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> hypergeometric_1F0(const T& a, const U& z)\
   { return boost::math::hypergeometric_1F0(a, z, Policy()); }\
   \
   template <class T, class U>\
   inline boost::math::tools::promote_args_t<T, U> hypergeometric_0F1(const T& a, const U& z)\
   { return boost::math::hypergeometric_0F1(a, z, Policy()); }\
   \
   template <class T, class U, class V>\
   inline boost::math::tools::promote_args_t<T, U> hypergeometric_2F0(const T& a1, const U& a2, const V& z)\
   { return boost::math::hypergeometric_2F0(a1, a2, z, Policy()); }\
   \




#endif // BOOST_MATH_HAS_NVRTC

#endif // BOOST_MATH_SPECIAL_MATH_FWD_HPP
