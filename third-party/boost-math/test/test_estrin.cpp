//  (C) Copyright Nick Thompson, John Maddock 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include "math_unit_test.hpp"
#include <boost/math/tools/condition_numbers.hpp>
#include <boost/math/tools/estrin.hpp>
#include <numeric>
#include <random>
#if BOOST_MATH_TEST_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

using boost::math::tools::evaluate_polynomial_estrin;
using boost::math::tools::summation_condition_number;

template <typename Real, typename Complex> void test_symmetric_coefficients(size_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<Real> dis(0, 1);
  std::vector<Real> coeffs(0);
  for (int i = 0; i < 10; ++i) {
    auto p = evaluate_polynomial_estrin(coeffs, Complex(dis(gen)));
    if (!CHECK_LE(std::norm(p), Real(0))) {
      std::cerr << "  If there are zero coefficients, the polynomial should evaluate to zero; seed = " << seed << "\n";
    }
  }
  // Now a single coefficient:
  coeffs.resize(1);
  coeffs[0] = dis(gen);
  for (int i = 0; i < 10; ++i) {
    auto p = evaluate_polynomial_estrin(coeffs, Complex(dis(gen)));
    if (!CHECK_ULP_CLOSE(p.real(), coeffs[0], 0)) {
      std::cerr << "  For a polynomial with 1 coefficient, the polynomial should evaluate to c0 for all z; seed = " << seed << "\n";
    }
  }
}

template <typename Real, typename Complex> void test_polynomial_properties(size_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<Real> dis(0, 1);
  // We're already testing the n=0 case above:
  size_t n = (seed % 128) + 1;
  std::vector<Real> coeffs(n);
  for (auto &c : coeffs) {
    c = dis(gen);
  }
  // Test evaluation at zero:
  auto p0 = evaluate_polynomial_estrin(coeffs, Complex(0));
  if (!CHECK_ULP_CLOSE(p0.real(), coeffs[0], 0)) {
    std::cerr << "  p(0) != c0 with seed " << seed << "\n";
  }
  // Just given the structure of the calculation, I believe the complex part should be identically zero:
  if (!CHECK_ULP_CLOSE(p0.imag(), Real(0), 0)) {
    std::cerr << "  p(0) != c0 with seed " << seed << "\n";
  }

  auto p1_computed = evaluate_polynomial_estrin(coeffs, Complex(1));
  // The recursive nature of Estrin's method makes it more accurate than a naive sum.
  // Use Kahan summation to make sure the expected value is accurate:
  auto s = summation_condition_number<Real>(0);
  for (auto c : coeffs) {
    s += c;
  }
  if (!CHECK_ULP_CLOSE(s.sum(), p1_computed.real(), coeffs.size())) {
    std::cerr << " p(1) != sum(coeffs) with seed " << seed << "\n";
  }
  if (!CHECK_ULP_CLOSE(Real(0), p1_computed.imag(), coeffs.size())) {
    std::cerr << " p(1) != sum(coeffs) with seed " << seed << "\n";
  }
}

template <typename Real> void test_std_array_overload(size_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<Real> dis(0, 1);
  std::array<Real, 12> coeffs;
  for (auto &c : coeffs) {
    c = dis(gen);
  }
  // Test evaluation at zero:
  auto p0 = evaluate_polynomial_estrin(coeffs, Real(0));
  if (!CHECK_ULP_CLOSE(p0, coeffs[0], 0)) {
    std::cerr << "  p(0) != c0 with seed " << seed << "\n";
  }
  auto p0_real = evaluate_polynomial_estrin(coeffs, Real(0));
  if (!CHECK_ULP_CLOSE(p0_real, coeffs[0], 0)) {
    std::cerr << "  p(0) != c0 with seed " << seed << "\n";
  }

  auto p1_computed = evaluate_polynomial_estrin(coeffs, Real(1));
  auto s = summation_condition_number<Real>(0);
  for (auto c : coeffs) {
    s += c;
  }
  if (!CHECK_ULP_CLOSE(s.sum(), p1_computed, coeffs.size())) {
    std::cerr << " p(1) != sum(coeffs) with seed " << seed << "\n";
  }
}

int main() {
  std::random_device rd;
  auto seed = rd();
  test_symmetric_coefficients<float, std::complex<float>>(seed);
  test_symmetric_coefficients<double, std::complex<double>>(seed);
  test_polynomial_properties<float, std::complex<float>>(seed);
  test_polynomial_properties<double, std::complex<double>>(seed);
  test_std_array_overload<float>(seed);
  test_std_array_overload<double>(seed);
#if BOOST_MATH_TEST_FLOAT128
  test_std_array_overload<boost::multiprecision::float128>(seed);
#endif
  return boost::math::test::report_errors();
}
