/*
 * Copyright Nick Thompson, 2024
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include "test_functions_for_optimization.hpp"
#include <boost/math/optimization/jso.hpp>
#include <random>
#include <limits>

using boost::math::optimization::jso;
using boost::math::optimization::jso_parameters;
using boost::math::optimization::detail::weighted_lehmer_mean;

void test_weighted_lehmer_mean() {
  size_t n = 50;
  std::vector<double> weights(n, 1.0);
  std::vector<double> values(n, 2.5);
  // Technically, this is not a fully general weighted Lehmer mean,
  // but just a weighted contraharmonic mean.
  // So we have a few more invariants available to us:
  CHECK_ULP_CLOSE(2.5, weighted_lehmer_mean(values, weights), n);
  std::mt19937_64 gen(12345);
  std::uniform_real_distribution<double> unif(std::numeric_limits<double>::epsilon(),1);
  for (size_t i = 0; i < n; ++i) {
    weights[i] = unif(gen);
    values[i] = unif(gen);
  }
  auto mean = weighted_lehmer_mean(values, weights);
  CHECK_LE(mean, 1.0);
  CHECK_LE(std::numeric_limits<double>::epsilon(), mean);
}

template <class Real> void test_ackley() {
  std::cout << "Testing jSO on Ackley function . . .\n";
  using ArgType = std::array<Real, 2>;
  auto jso_params = jso_parameters<ArgType>();
  jso_params.lower_bounds = {-5, -5};
  jso_params.upper_bounds = {5, 5};

  std::mt19937_64 gen(12345);
  auto local_minima = jso(ackley<Real>, jso_params, gen);
  CHECK_LE(std::abs(local_minima[0]), 10 * std::numeric_limits<Real>::epsilon());
  CHECK_LE(std::abs(local_minima[1]), 10 * std::numeric_limits<Real>::epsilon());

  // Does it work with a lambda?
  auto ack = [](std::array<Real, 2> const &x) { return ackley<Real>(x); };
  local_minima = jso(ack, jso_params, gen);
  CHECK_LE(std::abs(local_minima[0]), 10 * std::numeric_limits<Real>::epsilon());
  CHECK_LE(std::abs(local_minima[1]), 10 * std::numeric_limits<Real>::epsilon());

  // Test that if an intial guess is the exact solution, the returned solution is the exact solution:
  std::array<Real, 2> initial_guess{0, 0};
  jso_params.initial_guess = &initial_guess;
  local_minima = jso(ack, jso_params, gen);
  CHECK_EQUAL(local_minima[0], Real(0));
  CHECK_EQUAL(local_minima[1], Real(0));
}

template <class Real> void test_rosenbrock_saddle() {
  std::cout << "Testing jSO on Rosenbrock saddle . . .\n";
  using ArgType = std::array<Real, 2>;
  auto jso_params = jso_parameters<ArgType>();
  jso_params.lower_bounds = {0.5, 0.5};
  jso_params.upper_bounds = {2.048, 2.048};
  std::mt19937_64 gen(234568);
  auto local_minima = jso(rosenbrock_saddle<Real>, jso_params, gen);

  CHECK_ABSOLUTE_ERROR(Real(1), local_minima[0], 10 * std::numeric_limits<Real>::epsilon());
  CHECK_ABSOLUTE_ERROR(Real(1), local_minima[1], 10 * std::numeric_limits<Real>::epsilon());

  // Does cancellation work?
  std::atomic<bool> cancel = true;
  gen.seed(12345);
  local_minima =
      jso(rosenbrock_saddle<Real>, jso_params, gen, std::numeric_limits<Real>::quiet_NaN(), &cancel);
  CHECK_GE(std::abs(local_minima[0] - Real(1)), std::sqrt(std::numeric_limits<Real>::epsilon()));
}



template <class Real> void test_rastrigin() {
  std::cout << "Testing jSO on Rastrigin function (global minimum = (0,0,...,0))\n";
  using ArgType = std::vector<Real>;
  auto jso_params = jso_parameters<ArgType>();
  jso_params.lower_bounds.resize(3, static_cast<Real>(-5.12));
  jso_params.upper_bounds.resize(3, static_cast<Real>(5.12));
  jso_params.initial_population_size = 5000;
  jso_params.max_function_evaluations = 1000000;
  std::mt19937_64 gen(34567);

  // By definition, the value of the function which a target value is provided must be <= target_value.
  Real target_value = 1e-3;
  auto local_minima = jso(rastrigin<Real>, jso_params, gen, target_value);
  CHECK_LE(rastrigin(local_minima), target_value);
}

// Tests NaN return types and return type != input type:
void test_sphere() {
  std::cout << "Testing jSO on sphere . . .\n";
  using ArgType = std::vector<float>;
  auto jso_params = jso_parameters<ArgType>();
  jso_params.lower_bounds.resize(8, -1);
  jso_params.upper_bounds.resize(8, 1);
  std::mt19937_64 gen(56789);
  auto local_minima = jso(sphere, jso_params, gen);
  for (auto x : local_minima) {
    CHECK_ABSOLUTE_ERROR(0.0f, x, 2e-4f);
  }
}

template<typename Real>
void test_three_hump_camel() {
  std::cout << "Testing jSO on three hump camel . . .\n";
  using ArgType = std::array<Real, 2>;
  auto jso_params = jso_parameters<ArgType>();
  jso_params.lower_bounds[0] = -5.0;
  jso_params.lower_bounds[1] = -5.0;
  jso_params.upper_bounds[0] = 5.0;
  jso_params.upper_bounds[1] = 5.0;
  std::mt19937_64 gen(56789);
  auto local_minima = jso(three_hump_camel<Real>, jso_params, gen);
  for (auto x : local_minima) {
    CHECK_ABSOLUTE_ERROR(0.0f, x, 2e-4f);
  }
}

template<typename Real>
void test_beale() {
  std::cout << "Testing jSO on the Beale function . . .\n";
  using ArgType = std::array<Real, 2>;
  auto jso_params = jso_parameters<ArgType>();
  jso_params.lower_bounds[0] = -5.0;
  jso_params.lower_bounds[1] = -5.0;
  jso_params.upper_bounds[0]= 5.0;
  jso_params.upper_bounds[1]= 5.0;
  std::mt19937_64 gen(56789);
  auto local_minima = jso(beale<Real>, jso_params, gen);
  CHECK_ABSOLUTE_ERROR(Real(3), local_minima[0], Real(2e-4));
  CHECK_ABSOLUTE_ERROR(Real(1)/Real(2), local_minima[1], Real(2e-4));
}

#if BOOST_MATH_TEST_UNITS_COMPATIBILITY
void test_dimensioned_sphere() {
  std::cout << "Testing jso on dimensioned sphere . . .\n";
  using ArgType = std::vector<quantity<length>>;
  auto params = jso_parameters<ArgType>();
  params.lower_bounds.resize(4, -1.0*meter);
  params.upper_bounds.resize(4, 1*meter);
  params.threads = 2;
  std::mt19937_64 gen(56789);
  auto local_minima = jso(dimensioned_sphere, params, gen);
}
#endif

int main() {
#if defined(__clang__) || defined(_MSC_VER)
  test_ackley<float>();
  test_ackley<double>();
  test_rosenbrock_saddle<double>();
  test_rastrigin<double>();
  test_three_hump_camel<float>();
  test_beale<double>();
#endif
#if BOOST_MATH_TEST_UNITS_COMPATIBILITY
  test_dimensioned_sphere();
#endif
  test_sphere();
  test_weighted_lehmer_mean();
  return boost::math::test::report_errors();
}
