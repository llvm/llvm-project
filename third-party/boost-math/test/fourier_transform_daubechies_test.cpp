// boost-no-inspect
/*
 * Copyright Nick Thompson, 2023
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <numeric>
#include <utility>
#include <iomanip>
#include <iostream>
#include <random>
#include <boost/math/tools/condition_numbers.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <boost/math/special_functions/daubechies_scaling.hpp>
#include <boost/math/special_functions/daubechies_wavelet.hpp>
#include <boost/math/special_functions/fourier_transform_daubechies.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

using boost::math::fourier_transform_daubechies_scaling;
using boost::math::fourier_transform_daubechies_wavelet;
using boost::math::tools::summation_condition_number;
using boost::math::constants::two_pi;
using boost::math::constants::pi;
using boost::math::constants::one_div_root_two_pi;
using boost::math::quadrature::trapezoidal;
// 𝓕[φ](-ω) = 𝓕[φ](ω)*
template<typename Real, int p>
void test_evaluation_symmetry() {
   std::cout << "Testing evaluation symmetry on the " << p << " vanishing moment scaling function.\n";
   auto phi = fourier_transform_daubechies_scaling<Real, p>(0.0);
   CHECK_ULP_CLOSE(one_div_root_two_pi<Real>(), phi.real(), 3);
   CHECK_ULP_CLOSE(static_cast<Real>(0), phi.imag(), 3);

   Real domega = Real(1)/128;
   for (Real omega = domega; omega < 10; omega += domega) {
      auto phi1 = fourier_transform_daubechies_scaling<Real, p>(-omega);
      auto phi2 = fourier_transform_daubechies_scaling<Real, p>(omega);
      CHECK_ULP_CLOSE(phi1.real(), phi2.real(), 3);
      CHECK_ULP_CLOSE(phi1.imag(), -phi2.imag(), 3);

      auto psi1 = fourier_transform_daubechies_wavelet<Real, p>(-omega);
      auto psi2 = fourier_transform_daubechies_wavelet<Real, p>(omega);
      CHECK_ULP_CLOSE(psi1.real(), psi2.real(), 3);
      CHECK_ULP_CLOSE(psi1.imag(), -psi2.imag(), 3);
   }

   for (Real omega = 10; omega < std::cbrt(std::numeric_limits<Real>::max()); omega *= 10) {
      auto phi1 = fourier_transform_daubechies_scaling<Real, p>(-omega);
      auto phi2 = fourier_transform_daubechies_scaling<Real, p>(omega);
      CHECK_ULP_CLOSE(phi1.real(), phi2.real(), 3);
      CHECK_ULP_CLOSE(phi1.imag(), -phi2.imag(), 3);
   }
   return;
}

template<typename Real, int p>
void test_roots() {
   std::cout << "Testing roots on the " << p << " vanishing moment scaling function.\n";
   for (long n = 1; n < 100; ++n) {
      // All arguments of the form 2πn are roots of the complex function:
      // See Daubechies, 10 Lectures on Wavelets, Section 6.2:
      Real omega = n*two_pi<Real>();
      Real residual = std::norm(fourier_transform_daubechies_scaling<Real, p>(omega));
      CHECK_LE(residual, std::numeric_limits<Real>::epsilon()*std::numeric_limits<Real>::epsilon());
   }
   std::cout << "Testing roots on the " << p << " vanishing moment wavelet.\n";
   // If ωₙ is a root of the 𝓕[𝜙], then 2ωₙ is a root of 𝓕[ψ].
   // In addition, m₀(π) = 0, and m₀ is 2π periodic.
   // Recalling 𝓕[ψ](ω) = exp(iω/2)m₀(ω/2 + π)^{*}𝓕[𝜙](ω/2)*phase,
   // ω=4nπ are also roots:
   for (long n = 0; n < 100; ++n) {
      Real omega = 4*n*pi<Real>();
      Real residual = std::norm(fourier_transform_daubechies_wavelet<Real, p>(omega));
      CHECK_LE(residual, std::numeric_limits<Real>::epsilon()*std::numeric_limits<Real>::epsilon());
   }
}

template<int p>
void test_scaling_quadrature() {
   std::cout << "Testing numerical quadrature of the scaling function with " << p << " vanishing moments matches numerical evaluation.\n";
   auto phi = boost::math::daubechies_scaling<double, p>();
   auto [tmin, tmax] = phi.support();
   double domega = 1/8.0;
   for (double omega = domega; omega < 10; omega += domega) {
      // I suspect the quadrature is less accurate than special function evaluation, so this is just a sanity check:
      auto f = [&](double t) {
        return phi(t)*std::exp(std::complex<double>(0, -omega*t))*one_div_root_two_pi<double>();
      };
      auto expected = trapezoidal(f, tmin, tmax, 2*std::numeric_limits<double>::epsilon());
      auto computed = fourier_transform_daubechies_scaling<float, p>(static_cast<float>(omega));
      CHECK_MOLLIFIED_CLOSE(static_cast<float>(expected.real()), computed.real(), 1e-4);
      CHECK_MOLLIFIED_CLOSE(static_cast<float>(expected.imag()), computed.imag(), 1e-4);
   }
}

template<int p>
void test_wavelet_quadrature() {
   std::cout << "Testing numerical quadrature of the wavelet with " << p << " vanishing moments matches numerical evaluation.\n";
   auto psi = boost::math::daubechies_wavelet<double, p>();
   auto [tmin, tmax] = psi.support();
   double domega = 1/8.0;
   // There is a root at at ω=0, so skip this one because we can't recover the phase of a root. 
   for (double omega = 2*domega; omega < 10; omega += domega) {
      // I suspect the quadrature is less accurate than special function evaluation, so this is just a sanity check:
      auto f = [&](double t) {
        return psi(t)*std::exp(std::complex<double>(0, -omega*t))*one_div_root_two_pi<double>();
      };
      auto expected = trapezoidal(f, tmin, tmax, std::numeric_limits<double>::epsilon());
      auto computed = fourier_transform_daubechies_wavelet<double, p>(omega);
      if(!CHECK_ABSOLUTE_ERROR(std::abs(expected), std::abs(computed), 1e-9)) {
         std::cerr << "  |𝓕[ψ](" << omega << ")| is incorrect.\n";
      }
      // Again, lots of evidence that the quadrature is less accurate than what we've implemented.
      // Graph of the quadrature phase is super janky; graph of the implementation phase is pretty good.
      if(!CHECK_ABSOLUTE_ERROR(std::arg(expected), std::arg(computed), 1e-2)) {
         std::cerr << "  arg(𝓕[ψ](" << omega << ")) is incorrect.\n";
      }
   }
}


// Tests Daubechies "Ten Lectures on Wavelets", equation 5.1.19:
template<typename Real, int p>
void test_ten_lectures_eq_5_1_19() {
   std::cout << "Testing Ten Lectures equation 5.1.19 on " << p << " vanishing moments.\n";
   Real domega = Real(1)/Real(16);
   for (Real omega = 0; omega < 1; omega += domega) {
       Real term = std::norm(fourier_transform_daubechies_scaling<Real, p>(omega));
       auto sum = summation_condition_number<Real>(term);
       int64_t l = 1;
       while (l < 10000 && term > 2*std::numeric_limits<Real>::epsilon()) {
           Real tpl = std::norm(fourier_transform_daubechies_scaling<Real, p>(omega + two_pi<Real>()*l));
           Real tml = std::norm(fourier_transform_daubechies_scaling<Real, p>(omega - two_pi<Real>()*l));

           sum += tpl;
           sum += tml;
           ++l;
       }
       // With arg promotion, I can get this to 13 ULPS:
       if (!CHECK_ULP_CLOSE(1/two_pi<Real>(), sum.sum(), 125)) {
            std::cerr << "  Failure with occurs on " << p << " vanishing moments.\n";
       }
   }
}

// Tests Daubechies "Ten Lectures on Wavelets", equation 5.1.38:
template<typename Real, int p>
void test_ten_lectures_eq_5_1_38() {
   std::cout << "Testing Ten Lectures equation 5.1.38 on " << p << " vanishing moments.\n";
   Real domega = Real(1)/Real(16);
   for (Real omega = 0; omega < 1; omega += domega) {
       Real phi_omega_sq = std::norm(fourier_transform_daubechies_scaling<Real, p>(omega));
       Real psi_omega_sq = std::norm(fourier_transform_daubechies_wavelet<Real, p>(omega));
       Real phi_half_omega_sq = std::norm(fourier_transform_daubechies_scaling<Real, p>(omega/2));

       if (!CHECK_ULP_CLOSE(phi_half_omega_sq, phi_omega_sq + psi_omega_sq, 125)) {
            std::cerr << "  Failure with occurs on " << p << " vanishing moments at omega = " << omega << "\n";
       }
   }
}


int main()
{
  test_roots<float, 1>();
  test_roots<float, 2>();
  test_roots<float, 3>();
  test_roots<float, 4>();
  test_roots<float, 5>();
  test_roots<float, 6>();
  test_roots<float, 7>();
  test_roots<float, 8>();
  test_roots<float, 9>();
  test_roots<float, 10>();

  test_evaluation_symmetry<float, 1>();
  test_evaluation_symmetry<float, 2>();
  test_evaluation_symmetry<float, 3>();
  test_evaluation_symmetry<float, 4>();
  test_evaluation_symmetry<float, 5>();
  test_evaluation_symmetry<float, 6>();
  test_evaluation_symmetry<float, 7>();
  test_evaluation_symmetry<float, 8>();
  test_evaluation_symmetry<float, 9>();
  test_evaluation_symmetry<float, 10>();

  // Slow tests:
  test_scaling_quadrature<9>();
  test_scaling_quadrature<10>();

  // This one converges really slowly:
  //test_ten_lectures_eq_5_1_19<float, 1>();
  test_ten_lectures_eq_5_1_19<float, 2>();
  test_ten_lectures_eq_5_1_19<float, 3>();
  test_ten_lectures_eq_5_1_19<float, 4>();
  test_ten_lectures_eq_5_1_19<float, 5>();
  test_ten_lectures_eq_5_1_19<float, 6>();
  test_ten_lectures_eq_5_1_19<float, 7>();
  test_ten_lectures_eq_5_1_19<float, 8>();
  test_ten_lectures_eq_5_1_19<float, 9>();
  test_ten_lectures_eq_5_1_19<float, 10>();

  test_ten_lectures_eq_5_1_38<float, 3>();
  test_ten_lectures_eq_5_1_38<float, 4>();
  test_ten_lectures_eq_5_1_38<float, 5>();
  test_ten_lectures_eq_5_1_38<float, 6>();

  test_wavelet_quadrature<9>();
 
  return boost::math::test::report_errors();
}
