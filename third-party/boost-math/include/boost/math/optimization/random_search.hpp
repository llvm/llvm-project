/*
 * Copyright Nick Thompson, 2024
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#ifndef BOOST_MATH_OPTIMIZATION_RANDOM_SEARCH_HPP
#define BOOST_MATH_OPTIMIZATION_RANDOM_SEARCH_HPP
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <boost/math/optimization/detail/common.hpp>

namespace boost::math::optimization {

template <typename ArgumentContainer> struct random_search_parameters {
  using Real = typename ArgumentContainer::value_type;
  ArgumentContainer lower_bounds;
  ArgumentContainer upper_bounds;
  size_t max_function_calls = 10000*std::thread::hardware_concurrency();
  ArgumentContainer const *initial_guess = nullptr;
  unsigned threads = std::thread::hardware_concurrency();
};

template <typename ArgumentContainer>
void validate_random_search_parameters(random_search_parameters<ArgumentContainer> const &params) {
  using std::isfinite;
  using std::isnan;
  std::ostringstream oss;
  detail::validate_bounds(params.lower_bounds, params.upper_bounds);
  if (params.initial_guess) {
    detail::validate_initial_guess(*params.initial_guess, params.lower_bounds, params.upper_bounds);
  }
  if (params.threads == 0) {
    oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
    oss << ": There must be at least one thread.";
    throw std::invalid_argument(oss.str());
  }
}

template <typename ArgumentContainer, class Func, class URBG>
ArgumentContainer random_search(
    const Func cost_function,
    random_search_parameters<ArgumentContainer> const &params,
    URBG &gen,
    std::invoke_result_t<Func, ArgumentContainer> target_value = std::numeric_limits<std::invoke_result_t<Func, ArgumentContainer>>::quiet_NaN(),
    std::atomic<bool> *cancellation = nullptr,
    std::atomic<std::invoke_result_t<Func, ArgumentContainer>> *current_minimum_cost = nullptr,
    std::vector<std::pair<ArgumentContainer, std::invoke_result_t<Func, ArgumentContainer>>> *queries = nullptr)
 {
  using Real = typename ArgumentContainer::value_type;
  using DimensionlessReal = decltype(Real()/Real());
  using ResultType = std::invoke_result_t<Func, ArgumentContainer>;
  using std::isnan;
  using std::uniform_real_distribution;
  validate_random_search_parameters(params);
  const size_t dimension = params.lower_bounds.size();
  std::atomic<bool> target_attained = false;
  // Unfortunately, the "minimum_cost" variable can either be passed
  // (for observability) or not (if the user doesn't care).
  // That makes this a bit awkward . . .
  std::atomic<ResultType> lowest_cost = std::numeric_limits<ResultType>::infinity();

  ArgumentContainer best_vector;
  if constexpr (detail::has_resize_v<ArgumentContainer>) {
    best_vector.resize(dimension, std::numeric_limits<Real>::quiet_NaN());
  }
  if (params.initial_guess) {
    auto initial_cost = cost_function(*params.initial_guess);
    if (!isnan(initial_cost)) {
      lowest_cost = initial_cost;
      best_vector = *params.initial_guess;
      if (current_minimum_cost) {
        *current_minimum_cost = initial_cost;
      }
    }
  }
  std::mutex mt;
  std::vector<std::thread> thread_pool;
  std::atomic<size_t> function_calls = 0;
  for (unsigned j = 0; j < params.threads; ++j) {
    auto seed = gen();
    thread_pool.emplace_back([&, seed]() {
      URBG g(seed);
      ArgumentContainer trial_vector;
      // This vector is empty unless the user requests the queries be stored:
      std::vector<std::pair<ArgumentContainer, std::invoke_result_t<Func, ArgumentContainer>>> local_queries;
      if constexpr (detail::has_resize_v<ArgumentContainer>) {
          trial_vector.resize(dimension, std::numeric_limits<Real>::quiet_NaN());
      }
      while (function_calls < params.max_function_calls) {
        if (cancellation && *cancellation) {
            break;
        }
        if (target_attained) {
            break;
        }
        // Fill trial vector: 
        uniform_real_distribution<DimensionlessReal> unif01(DimensionlessReal(0), DimensionlessReal(1));
        for (size_t i = 0; i < dimension; ++i) {
            trial_vector[i] = params.lower_bounds[i] + (params.upper_bounds[i] - params.lower_bounds[i])*unif01(g);
        }
        ResultType trial_cost = cost_function(trial_vector);
        ++function_calls;
        if (isnan(trial_cost)) {
          continue;
        }
        if (trial_cost < lowest_cost) {
          lowest_cost = trial_cost;
          if (current_minimum_cost) {
            *current_minimum_cost = trial_cost;
          }
          // We expect to need to acquire this lock with decreasing frequency
          // as the computation proceeds:
          std::scoped_lock lock(mt);
          best_vector = trial_vector;
        }
        if (queries) {
          local_queries.push_back(std::make_pair(trial_vector, trial_cost));
        }
        if (!isnan(target_value) && trial_cost <= target_value) {
          target_attained = true;
        }
      }
      if (queries) {
        std::scoped_lock lock(mt);
        queries->insert(queries->begin(), local_queries.begin(), local_queries.end());
      }
    });
  }
  for (auto &thread : thread_pool) {
    thread.join();
  }
  return best_vector;
}

} // namespace boost::math::optimization
#endif
