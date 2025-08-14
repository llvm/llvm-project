/*
 * Copyright Nick Thompson, 2023
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#include <iostream>
#include <boost/math/optimization/differential_evolution.hpp>

using boost::math::optimization::differential_evolution_parameters;
using boost::math::optimization::differential_evolution;

double rosenbrock(std::vector<double> const & x) {
   double result = 0;
   for (size_t i = 0; i < x.size() - 1; ++i) {
       double tmp = x[i+1] - x[i]*x[i];
       result += 100*tmp*tmp + (1-x[i])*(1-x[i]); 
   }
   return result;
}

int main() {
    auto de_params = differential_evolution_parameters<std::vector<double>>();
    constexpr const size_t dimension = 10;
    // Search on [0, 2]^dimension:
    de_params.lower_bounds.resize(dimension, 0);
    de_params.upper_bounds.resize(dimension, 2);
    // This is a challenging function, increase the max generations 10x from default so we don't terminate prematurely:
    de_params.max_generations *= 10;
    std::random_device rd;
    std::mt19937_64 rng(rd());
    // The global minima is exactly zero-but some leeway is required:
    double value_to_reach = 1e-5;
    auto local_minima = differential_evolution(rosenbrock, de_params, rng, value_to_reach);
    std::cout << "Minima: {";
    for (auto l : local_minima) {
        std::cout << l << ", ";
    }
    std::cout << "}\n";
    std::cout << "Value of cost function at minima: " << rosenbrock(local_minima) << "\n";
}
