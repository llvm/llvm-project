/*
 * Copyright Nick Thompson, 2024
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#if __APPLE__ || __linux__
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <future>
#include <chrono>
#include <boost/math/constants/constants.hpp>
#include <boost/math/optimization/jso.hpp>

using boost::math::optimization::jso_parameters;
using boost::math::optimization::jso;
using namespace std::chrono_literals;

template <class Real> Real rastrigin(std::vector<Real> const &v) {
  using std::cos;
  using boost::math::constants::two_pi;
  Real A = 10;
  Real y = 10 * v.size();
  for (auto x : v) {
    y += x * x - A * cos(two_pi<Real>() * x);
  }
  return y;
}

std::atomic<bool> cancel = false;

void ctrl_c_handler(int){
    cancel = true;
    std::cout << "Cancellation requested-this could take a second . . ." << std::endl;
}

int main() {
  std::cout << "Running jSO on Rastrigin function (global minimum = (0,0,...,0))\n";
  signal(SIGINT, ctrl_c_handler);
  using ArgType = std::vector<double>;
  auto jso_params = jso_parameters<ArgType>();
  jso_params.lower_bounds.resize(500, -5.12);
  jso_params.upper_bounds.resize(500, 5.12);
  jso_params.initial_population_size = 5000;
  jso_params.max_function_evaluations = 10000000;
  std::mt19937_64 gen(34567);

  // By definition, the value of the function which a target value is provided must be <= target_value.
  double target_value = 1e-3;
  std::atomic<double> current_minimum_cost;
  std::cout << "Hit ctrl-C to gracefully terminate the optimization." << std::endl;
  auto f = [&]() {
    return jso(rastrigin<double>, jso_params, gen, target_value, &cancel, &current_minimum_cost);
  };
  auto future = std::async(std::launch::async, f);
  std::future_status status = future.wait_for(3ms);
  while (!cancel && (status != std::future_status::ready)) {
    status = future.wait_for(3ms);
    std::cout << "Current cost is " << current_minimum_cost << "\r";
  }

  auto local_minima = future.get();
  std::cout << "Local minimum is {";
  for (size_t i = 0; i < local_minima.size() - 1; ++i) {
    std::cout << local_minima[i] << ", ";
  }
  std::cout << local_minima.back() << "}.\n";
  std::cout << "Final cost: " << current_minimum_cost << "\n";
}
#else
#warning "Signal handling for the jSO example only works on Linux and Mac."
int main() {
    return 0;
}
#endif
