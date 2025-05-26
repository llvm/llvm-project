/*
 * Copyright Nick Thompson, 2024
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include "test_functions_for_optimization.hpp"
#include <boost/math/optimization/random_search.hpp>
#include <random>
#include <limits>
using boost::math::optimization::random_search;
using boost::math::optimization::random_search_parameters;


template <class Real> void test_ackley() {
  std::cout << "Testing random search on Ackley function . . .\n";
  using ArgType = std::array<Real, 2>;
  auto rs_params = random_search_parameters<ArgType>();
  rs_params.lower_bounds = {-5, -5};
  rs_params.upper_bounds = {5, 5};
  // This makes the CI a bit more robust;
  // the computation is only deterministic with a deterministic number of threads:
  rs_params.threads = 2;
  rs_params.max_function_calls *= 10;
  std::mt19937_64 gen(12345);
  auto local_minima = random_search(ackley<Real>, rs_params, gen);
  CHECK_LE(std::abs(local_minima[0]), Real(0.2));
  CHECK_LE(std::abs(local_minima[1]), Real(0.2));

  // Does it work with a lambda?
  auto ack = [](std::array<Real, 2> const &x) { return ackley<Real>(x); };
  local_minima = random_search(ack, rs_params, gen);
  CHECK_LE(std::abs(local_minima[0]), Real(0.2));
  CHECK_LE(std::abs(local_minima[1]), Real(0.2));

  // Test that if an intial guess is the exact solution, the returned solution is the exact solution:
  std::array<Real, 2> initial_guess{0, 0};
  rs_params.initial_guess = &initial_guess;
  local_minima = random_search(ack, rs_params, gen);
  CHECK_EQUAL(local_minima[0], Real(0));
  CHECK_EQUAL(local_minima[1], Real(0));

  std::atomic<bool> cancel = false;
  Real target_value = 0.0;
  std::atomic<Real> current_minimum_cost = std::numeric_limits<Real>::quiet_NaN();
  // Test query storage:
  std::vector<std::pair<ArgType, Real>> queries;
  local_minima = random_search(ack, rs_params, gen, target_value, &cancel, &current_minimum_cost, &queries);
  CHECK_EQUAL(local_minima[0], Real(0));
  CHECK_EQUAL(local_minima[1], Real(0));
  CHECK_LE(size_t(1), queries.size());
  for (auto const & q : queries) {
    auto expected = ackley<Real>(q.first);
    CHECK_EQUAL(expected, q.second);
  }
}


template <class Real> void test_rosenbrock_saddle() {
  std::cout << "Testing random search on Rosenbrock saddle . . .\n";
  using ArgType = std::array<Real, 2>;
  auto rs_params = random_search_parameters<ArgType>();
  rs_params.lower_bounds = {0.5, 0.5};
  rs_params.upper_bounds = {2.048, 2.048};
  rs_params.max_function_calls = 20000;
  rs_params.threads = 2;
  std::mt19937_64 gen(234568);
  auto local_minima = random_search(rosenbrock_saddle<Real>, rs_params, gen);

  CHECK_ABSOLUTE_ERROR(Real(1), local_minima[0], Real(0.05));
  CHECK_ABSOLUTE_ERROR(Real(1), local_minima[1], Real(0.05));

  // Does cancellation work?
  std::atomic<bool> cancel = true;
  gen.seed(12345);
  local_minima =
      random_search(rosenbrock_saddle<Real>, rs_params, gen, std::numeric_limits<Real>::quiet_NaN(), &cancel);
  CHECK_GE(std::abs(local_minima[0] - Real(1)), std::sqrt(std::numeric_limits<Real>::epsilon()));
}


template <class Real> void test_rastrigin() {
  std::cout << "Testing random search on Rastrigin function (global minimum = (0,0,...,0))\n";
  using ArgType = std::vector<Real>;
  auto rs_params = random_search_parameters<ArgType>();
  rs_params.lower_bounds.resize(3, static_cast<Real>(-5.12));
  rs_params.upper_bounds.resize(3, static_cast<Real>(5.12));
  rs_params.max_function_calls = 1000000;
  rs_params.threads = 2;
  std::mt19937_64 gen(34567);

  // By definition, the value of the function which a target value is provided must be <= target_value.
  Real target_value = 2.0;
  auto local_minima = random_search(rastrigin<Real>, rs_params, gen, target_value);
  CHECK_LE(rastrigin(local_minima), target_value);
}


// Tests NaN return types and return type != input type:
void test_sphere() {
  std::cout << "Testing random search on sphere . . .\n";
  using ArgType = std::vector<float>;
  auto rs_params = random_search_parameters<ArgType>();
  rs_params.lower_bounds.resize(4, -1);
  rs_params.upper_bounds.resize(4, 1);
  rs_params.max_function_calls = 100000;
  rs_params.threads = 2;
  std::mt19937_64 gen(56789);
  auto local_minima = random_search(sphere, rs_params, gen);
  for (auto x : local_minima) {
    CHECK_ABSOLUTE_ERROR(0.0f, x, 0.5f);
  }
}


template<typename Real>
void test_three_hump_camel() {
  std::cout << "Testing random search on three hump camel . . .\n";
  using ArgType = std::array<Real, 2>;
  auto rs_params = random_search_parameters<ArgType>();
  rs_params.lower_bounds[0] = -5.0;
  rs_params.lower_bounds[1] = -5.0;
  rs_params.upper_bounds[0] = 5.0;
  rs_params.upper_bounds[1] = 5.0;
  rs_params.threads = 2;
  rs_params.max_function_calls *= 10;
  std::mt19937_64 gen(56789);
  auto local_minima = random_search(three_hump_camel<Real>, rs_params, gen);
  for (auto x : local_minima) {
    CHECK_ABSOLUTE_ERROR(0.0f, x, 0.2f);
  }
}


template<typename Real>
void test_beale() {
  std::cout << "Testing random search on the Beale function . . .\n";
  using ArgType = std::array<Real, 2>;
  auto rs_params = random_search_parameters<ArgType>();
  rs_params.lower_bounds[0] = -5.0;
  rs_params.lower_bounds[1] = -5.0;
  rs_params.upper_bounds[0]= 5.0;
  rs_params.upper_bounds[1]= 5.0;
  rs_params.threads = 2;
  rs_params.max_function_calls *= 10;
  std::mt19937_64 gen(56789);
  auto local_minima = random_search(beale<Real>, rs_params, gen);
  CHECK_ABSOLUTE_ERROR(Real(3), local_minima[0], Real(0.1));
  CHECK_ABSOLUTE_ERROR(Real(1)/Real(2), local_minima[1], Real(0.1));
}

#if BOOST_MATH_TEST_UNITS_COMPATIBILITY
void test_dimensioned_sphere() {
  std::cout << "Testing random search on dimensioned sphere . . .\n";
  using ArgType = std::vector<quantity<length>>;
  auto rs_params = random_search_parameters<ArgType>();
  rs_params.lower_bounds.resize(4, -1.0*meter);
  rs_params.upper_bounds.resize(4, 1*meter);
  rs_params.max_function_calls = 100000;
  rs_params.threads = 2;
  std::mt19937_64 gen(56789);
  auto local_minima = random_search(dimensioned_sphere, rs_params, gen);
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
  return boost::math::test::report_errors();
}
