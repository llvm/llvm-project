/*
 * Copyright Nick Thompson, 2023
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include "test_functions_for_optimization.hpp"
#include <boost/math/optimization/differential_evolution.hpp>
#include <random>

using boost::math::optimization::differential_evolution;
using boost::math::optimization::differential_evolution_parameters;
using boost::math::optimization::validate_differential_evolution_parameters;
using boost::math::optimization::detail::best_indices;
using boost::math::optimization::detail::random_initial_population;
using boost::math::optimization::detail::validate_initial_guess;

void test_random_initial_population() {
  std::array<double, 2> lower_bounds = {-5, -5};
  std::array<double, 2> upper_bounds = {5, 5};
  size_t n = 500;
  std::mt19937_64 gen(12345);
  auto population = random_initial_population(lower_bounds, upper_bounds, n, gen);
  CHECK_EQUAL(population.size(), n);
  for (auto const & individual : population) {
    validate_initial_guess(individual, lower_bounds, upper_bounds);
  }
  // Reproducibility:
  gen.seed(12345);
  auto population2 = random_initial_population(lower_bounds, upper_bounds, n, gen);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      CHECK_EQUAL(population[i][j], population2[i][j]);
    }
  }
}
void test_nan_sorting() {
  auto nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v{-1.2, nan, -3.5, 2.3, nan, 8.7, -4.2};
  auto indices = best_indices(v);
  CHECK_EQUAL(indices[0], size_t(6));
  CHECK_EQUAL(indices[1], size_t(2));
  CHECK_EQUAL(indices[2], size_t(0));
  CHECK_EQUAL(indices[3], size_t(3));
  CHECK_EQUAL(indices[4], size_t(5));
  CHECK_NAN(v[indices[5]]);
  CHECK_NAN(v[indices[6]]);
}

void test_parameter_checks() {
  using ArgType = std::array<double, 2>;
  auto de_params = differential_evolution_parameters<ArgType>();
  de_params.threads = 0;
  bool caught = false;
  try {
    validate_differential_evolution_parameters(de_params);
  } catch(std::exception const &) {
    caught = true;
  }
  CHECK_TRUE(caught);
  caught = false;
  de_params = differential_evolution_parameters<ArgType>();
  de_params.NP = 1;
  try {
    validate_differential_evolution_parameters(de_params);
  } catch(std::exception const &) {
    caught = true;
  }
  CHECK_TRUE(caught);
}
template <class Real> void test_ackley() {
  std::cout << "Testing differential evolution on the Ackley function . . .\n";
  using ArgType = std::array<Real, 2>;
  auto de_params = differential_evolution_parameters<ArgType>();
  de_params.lower_bounds = {-5, -5};
  de_params.upper_bounds = {5, 5};

  std::mt19937_64 gen(12345);
  auto local_minima = differential_evolution(ackley<Real>, de_params, gen);
  CHECK_LE(std::abs(local_minima[0]), 10 * std::numeric_limits<Real>::epsilon());
  CHECK_LE(std::abs(local_minima[1]), 10 * std::numeric_limits<Real>::epsilon());

  // Does it work with a lambda?
  auto ack = [](std::array<Real, 2> const &x) { return ackley<Real>(x); };
  local_minima = differential_evolution(ack, de_params, gen);
  CHECK_LE(std::abs(local_minima[0]), 10 * std::numeric_limits<Real>::epsilon());
  CHECK_LE(std::abs(local_minima[1]), 10 * std::numeric_limits<Real>::epsilon());

  // Test that if an intial guess is the exact solution, the returned solution is the exact solution:
  std::array<Real, 2> initial_guess{0, 0};
  de_params.initial_guess = &initial_guess;
  local_minima = differential_evolution(ack, de_params, gen);
  CHECK_EQUAL(local_minima[0], Real(0));
  CHECK_EQUAL(local_minima[1], Real(0));
}

template <class Real> void test_rosenbrock_saddle() {
  std::cout << "Testing differential evolution on the Rosenbrock saddle . . .\n";
  using ArgType = std::array<Real, 2>;
  auto de_params = differential_evolution_parameters<ArgType>();
  de_params.lower_bounds = {0.5, 0.5};
  de_params.upper_bounds = {2.048, 2.048};
  std::mt19937_64 gen(234568);
  auto local_minima = differential_evolution(rosenbrock_saddle<Real>, de_params, gen);

  CHECK_ABSOLUTE_ERROR(Real(1), local_minima[0], 10 * std::numeric_limits<Real>::epsilon());
  CHECK_ABSOLUTE_ERROR(Real(1), local_minima[1], 10 * std::numeric_limits<Real>::epsilon());

  // Does cancellation work?
  std::atomic<bool> cancel = true;
  gen.seed(12345);
  local_minima =
      differential_evolution(rosenbrock_saddle<Real>, de_params, gen, std::numeric_limits<Real>::quiet_NaN(), &cancel);
  CHECK_GE(std::abs(local_minima[0] - Real(1)), std::sqrt(std::numeric_limits<Real>::epsilon()));
}

template <class Real> void test_rastrigin() {
  std::cout << "Testing differential evolution on the Rastrigin function . . .\n";
  using ArgType = std::vector<Real>;
  auto de_params = differential_evolution_parameters<ArgType>();
  de_params.lower_bounds.resize(8, static_cast<Real>(-5.12));
  de_params.upper_bounds.resize(8, static_cast<Real>(5.12));
  std::mt19937_64 gen(34567);
  auto local_minima = differential_evolution(rastrigin<Real>, de_params, gen);
  for (auto x : local_minima) {
    CHECK_ABSOLUTE_ERROR(x, Real(0), Real(2e-4));
  }

  // By definition, the value of the function which a target value is provided must be <= target_value.
  auto target_value = static_cast<Real>(1e-3);
  local_minima = differential_evolution(rastrigin<Real>, de_params, gen, target_value);
  CHECK_LE(rastrigin(local_minima), target_value);
}

// Tests NaN return types and return type != input type:
void test_sphere() {
  std::cout << "Testing differential evolution on the sphere function . . .\n";
  using ArgType = std::vector<float>;
  auto de_params = differential_evolution_parameters<ArgType>();
  de_params.lower_bounds.resize(3, -1);
  de_params.upper_bounds.resize(3, 1);
  de_params.NP *= 10;
  de_params.max_generations *= 10;
  de_params.crossover_probability = 0.9;
  double target_value = 1e-8;
  de_params.threads = 1;
  std::mt19937_64 gen(56789);
  auto local_minima = differential_evolution(sphere, de_params, gen, target_value);
  CHECK_LE(sphere(local_minima), target_value);
  // Check computational reproducibility:
  gen.seed(56789);
  auto local_minima_2 = differential_evolution(sphere, de_params, gen, target_value);
  for (size_t i = 0; i < local_minima.size(); ++i) {
    CHECK_EQUAL(local_minima[i], local_minima_2[i]);
  }
}

template<typename Real>
void test_three_hump_camel() {
  std::cout << "Testing differential evolution on the three hump camel . . .\n";
  using ArgType = std::array<Real, 2>;
  auto de_params = differential_evolution_parameters<ArgType>();
  de_params.lower_bounds[0] = -5.0;
  de_params.lower_bounds[1] = -5.0;
  de_params.upper_bounds[0] = 5.0;
  de_params.upper_bounds[1] = 5.0;
  std::mt19937_64 gen(56789);
  auto local_minima = differential_evolution(three_hump_camel<Real>, de_params, gen);
  for (auto x : local_minima) {
    CHECK_ABSOLUTE_ERROR(0.0f, x, 2e-4f);
  }
}

template<typename Real>
void test_beale() {
  std::cout << "Testing differential evolution on the Beale function . . .\n";
  using ArgType = std::array<Real, 2>;
  auto de_params = differential_evolution_parameters<ArgType>();
  de_params.lower_bounds[0] = -5.0;
  de_params.lower_bounds[1] = -5.0;
  de_params.upper_bounds[0]= 5.0;
  de_params.upper_bounds[1]= 5.0;
  std::mt19937_64 gen(56789);
  auto local_minima = differential_evolution(beale<Real>, de_params, gen);
  CHECK_ABSOLUTE_ERROR(Real(3), local_minima[0], Real(2e-4));
  CHECK_ABSOLUTE_ERROR(Real(1)/Real(2), local_minima[1], Real(2e-4));
}

#if BOOST_MATH_TEST_UNITS_COMPATIBILITY
void test_dimensioned_sphere() {
  std::cout << "Testing differential evolution on dimensioned sphere . . .\n";
  using ArgType = std::vector<quantity<length>>;
  auto params = differential_evolution_parameters<ArgType>();
  params.lower_bounds.resize(4, -1.0*meter);
  params.upper_bounds.resize(4, 1*meter);
  params.threads = 2;
  std::mt19937_64 gen(56789);
  auto local_minima = differential_evolution(dimensioned_sphere, params, gen);
}
#endif

int main() {

#if defined(__clang__) || defined(_MSC_VER)
  test_ackley<float>();
  test_ackley<double>();
  test_rosenbrock_saddle<double>();
  test_rastrigin<float>();
  test_three_hump_camel<float>();
  test_beale<double>();
#endif
#if BOOST_MATH_TEST_UNITS_COMPATIBILITY
  test_dimensioned_sphere();
#endif
  test_sphere();
  test_parameter_checks();
  return boost::math::test::report_errors();
}
