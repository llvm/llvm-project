/*
 * Copyright Nick Thompson, 2024
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include "test_functions_for_optimization.hpp"
#include <boost/math/optimization/cma_es.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>

using std::abs;
using boost::math::optimization::cma_es;
using boost::math::optimization::cma_es_parameters;

template <class Real> void test_ackley() {
  std::cout << "Testing CMA-ES on Ackley function . . .\n";
  using ArgType = std::array<Real, 2>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds = {-5, -5};
  params.upper_bounds = {5, 5};

  std::mt19937_64 gen(12345);
  auto local_minima = cma_es(ackley<Real>, params, gen);
  CHECK_LE(std::abs(local_minima[0]), Real(0.1));
  CHECK_LE(std::abs(local_minima[1]), Real(0.1));

  // Does it work with a lambda?
  auto ack = [](std::array<Real, 2> const &x) { return ackley<Real>(x); };
  local_minima = cma_es(ack, params, gen);
  CHECK_LE(std::abs(local_minima[0]), Real(0.1));
  CHECK_LE(std::abs(local_minima[1]), Real(0.1));

  // Test that if an intial guess is the exact solution, the returned solution is the exact solution:
  std::array<Real, 2> initial_guess{0, 0};
  params.initial_guess = &initial_guess;
  local_minima = cma_es(ack, params, gen);
  CHECK_EQUAL(local_minima[0], Real(0));
  CHECK_EQUAL(local_minima[1], Real(0));

  std::atomic<bool> cancel = false;
  Real target_value = 0.0;
  std::atomic<Real> current_minimum_cost = std::numeric_limits<Real>::quiet_NaN();
  // Test query storage:
  std::vector<std::pair<ArgType, Real>> queries;
  local_minima = cma_es(ack, params, gen, target_value, &cancel, &current_minimum_cost, &queries);
  CHECK_EQUAL(local_minima[0], Real(0));
  CHECK_EQUAL(local_minima[1], Real(0));
  CHECK_LE(size_t(1), queries.size());
  for (auto const & q : queries) {
    auto expected = ackley<Real>(q.first);
    CHECK_EQUAL(expected, q.second);
  }
}


template <class Real> void test_rosenbrock_saddle() {
  std::cout << "Testing CMA-ES on Rosenbrock saddle . . .\n";
  using ArgType = std::array<Real, 2>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds = {0.5, 0.5};
  params.upper_bounds = {2.048, 2.048};
  params.max_generations = 2000;
  std::mt19937_64 gen(234568);
  auto local_minima = cma_es(rosenbrock_saddle<Real>, params, gen);

  CHECK_ABSOLUTE_ERROR(Real(1), local_minima[0], Real(0.05));
  CHECK_ABSOLUTE_ERROR(Real(1), local_minima[1], Real(0.05));

  // Does cancellation work?
  std::atomic<bool> cancel = true;
  gen.seed(12345);
  local_minima =
      cma_es(rosenbrock_saddle<Real>, params, gen, std::numeric_limits<Real>::quiet_NaN(), &cancel);
  CHECK_GE(std::abs(local_minima[0] - Real(1)), std::sqrt(std::numeric_limits<Real>::epsilon()));
}


template <class Real> void test_rastrigin() {
  std::cout << "Testing CMA-ES on Rastrigin function (global minimum = (0,0,...,0))\n";
  using ArgType = std::vector<Real>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds.resize(3, static_cast<Real>(-5.12));
  params.upper_bounds.resize(3, static_cast<Real>(5.12));
  params.max_generations = 1000000;
  params.population_size = 100;
  std::mt19937_64 gen(34567);

  // By definition, the value of the function which a target value is provided must be <= target_value.
  Real target_value = 2.0;
  auto local_minima = cma_es(rastrigin<Real>, params, gen, target_value);
  CHECK_LE(rastrigin(local_minima), target_value);
}


// Tests NaN return types and return type != input type:
void test_sphere() {
  std::cout << "Testing CMA-ES on sphere . . .\n";
  using ArgType = std::vector<float>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds.resize(4, -1);
  params.upper_bounds.resize(4, 1);
  params.max_generations = 100000;
  std::mt19937_64 gen(56789);
  auto local_minima = cma_es(sphere, params, gen, 1e-6f);
  for (auto x : local_minima) {
    CHECK_ABSOLUTE_ERROR(0.0f, x, 0.5f);
  }
}


template<typename Real>
void test_three_hump_camel() {
  std::cout << "Testing CMA-ES on three hump camel . . .\n";
  using ArgType = std::array<Real, 2>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds[0] = -5.0;
  params.lower_bounds[1] = -5.0;
  params.upper_bounds[0] = 5.0;
  params.upper_bounds[1] = 5.0;
  std::mt19937_64 gen(56789);
  auto local_minima = cma_es(three_hump_camel<Real>, params, gen);
  for (auto x : local_minima) {
    CHECK_ABSOLUTE_ERROR(0.0f, x, 0.2f);
  }
}


template<typename Real>
void test_beale() {
  std::cout << "Testing CMA-ES on the Beale function . . .\n";
  using ArgType = std::array<Real, 2>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds[0] = -5.0;
  params.lower_bounds[1] = -5.0;
  params.upper_bounds[0]= 5.0;
  params.upper_bounds[1]= 5.0;
  std::mt19937_64 gen(56789);
  auto local_minima = cma_es(beale<Real>, params, gen);
  CHECK_ABSOLUTE_ERROR(Real(3), local_minima[0], Real(0.1));
  CHECK_ABSOLUTE_ERROR(Real(1)/Real(2), local_minima[1], Real(0.1));
}

#if BOOST_MATH_TEST_UNITS_COMPATIBILITY
void test_dimensioned_sphere() {
  std::cout << "Testing CMA-ES on dimensioned sphere . . .\n";
  using ArgType = std::vector<quantity<length>>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds.resize(4, -1.0*meter);
  params.upper_bounds.resize(4, 1*meter);
  std::mt19937_64 gen(56789);
  auto local_minima = cma_es(dimensioned_sphere, params, gen);
}
#endif

int main() {
#if (defined(__clang__) || defined(_MSC_VER))
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
  return boost::math::test::report_errors();
}
