/*
 * Copyright Nick Thompson, 2024
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#ifndef BOOST_MATH_OPTIMIZATION_JSO_HPP
#define BOOST_MATH_OPTIMIZATION_JSO_HPP
#include <atomic>
#include <boost/math/optimization/detail/common.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace boost::math::optimization {

#ifndef BOOST_MATH_DEBUG_JSO
#define BOOST_MATH_DEBUG_JSO 0
#endif
// Follows: Brest, Janez, Mirjam Sepesy Maucec, and Borko Boskovic. "Single
// objective real-parameter optimization: Algorithm jSO." 2017 IEEE congress on
// evolutionary computation (CEC). IEEE, 2017. In the sequel, this wil be
// referred to as "the reference". Note that the reference is rather difficult
// to understand without also reading: Zhang, J., & Sanderson, A. C. (2009).
// JADE: adaptive differential evolution with optional external archive.
// IEEE Transactions on evolutionary computation, 13(5), 945-958."
template <typename ArgumentContainer> struct jso_parameters {
  using Real = typename ArgumentContainer::value_type;
  using DimensionlessReal = decltype(Real()/Real());
  ArgumentContainer lower_bounds;
  ArgumentContainer upper_bounds;
  // Population in the first generation.
  // This defaults to 0, which indicates "use the default specified in the
  // referenced paper". To wit, initial population size
  // =ceil(25log(D+1)sqrt(D)), where D is the dimension of the problem.
  size_t initial_population_size = 0;
  // Maximum number of function evaluations.
  // The default of 0 indicates "use max_function_evaluations = 10,000D", where
  // D is the dimension of the problem.
  size_t max_function_evaluations = 0;
  size_t threads = std::thread::hardware_concurrency();
  ArgumentContainer const *initial_guess = nullptr;
};

template <typename ArgumentContainer>
void validate_jso_parameters(jso_parameters<ArgumentContainer> &jso_params) {
  using std::isfinite;
  using std::isnan;
  std::ostringstream oss;
  if (jso_params.threads == 0) {
    oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
    oss << ": Requested zero threads to perform the calculation, but at least "
           "1 is required.";
    throw std::invalid_argument(oss.str());
  }
  detail::validate_bounds(jso_params.lower_bounds, jso_params.upper_bounds);
  auto const dimension = jso_params.lower_bounds.size();
  if (jso_params.initial_population_size == 0) {
    // Ever so slightly different than the reference-the dimension can be 1,
    // but if we followed the reference, the population size would then be zero.
    jso_params.initial_population_size = static_cast<size_t>(
        std::ceil(25 * std::log(dimension + 1.0) * sqrt(dimension)));
  }
  if (jso_params.max_function_evaluations == 0) {
    // Recommended value from the reference:
    jso_params.max_function_evaluations = 10000 * dimension;
  }
  if (jso_params.initial_population_size < 4) {
    oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
    oss << ": The population size must be at least 4, but requested population "
           "size of "
        << jso_params.initial_population_size << ".";
    throw std::invalid_argument(oss.str());
  }
  if (jso_params.threads > jso_params.initial_population_size) {
    oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
    oss << ": There must be more individuals in the population than threads.";
    throw std::invalid_argument(oss.str());
  }
  if (jso_params.initial_guess) {
    detail::validate_initial_guess(*jso_params.initial_guess,
                                   jso_params.lower_bounds,
                                   jso_params.upper_bounds);
  }
}

template <typename ArgumentContainer, class Func, class URBG>
ArgumentContainer
jso(const Func cost_function, jso_parameters<ArgumentContainer> &jso_params,
    URBG &gen,
    std::invoke_result_t<Func, ArgumentContainer> target_value =
        std::numeric_limits<
            std::invoke_result_t<Func, ArgumentContainer>>::quiet_NaN(),
    std::atomic<bool> *cancellation = nullptr,
    std::atomic<std::invoke_result_t<Func, ArgumentContainer>>
        *current_minimum_cost = nullptr,
    std::vector<std::pair<ArgumentContainer,
                          std::invoke_result_t<Func, ArgumentContainer>>>
        *queries = nullptr) {
  using Real = typename ArgumentContainer::value_type;
  using DimensionlessReal = decltype(Real()/Real());
  validate_jso_parameters(jso_params);

  using ResultType = std::invoke_result_t<Func, ArgumentContainer>;
  using std::abs;
  using std::cauchy_distribution;
  using std::clamp;
  using std::isnan;
  using std::max;
  using std::round;
  using std::isfinite;
  using std::uniform_real_distribution;

  // Refer to the referenced paper, pseudocode on page 1313:
  // Algorithm 1, jSO, Line 1:
  std::vector<ArgumentContainer> archive;

  // Algorithm 1, jSO, Line 2
  // "Initialize population P_g = (x_0,g, ..., x_{np-1},g) randomly.
  auto population = detail::random_initial_population(
      jso_params.lower_bounds, jso_params.upper_bounds,
      jso_params.initial_population_size, gen);
  if (jso_params.initial_guess) {
    population[0] = *jso_params.initial_guess;
  }
  size_t dimension = jso_params.lower_bounds.size();
  // Don't force the user to initialize to sane value:
  if (current_minimum_cost) {
    *current_minimum_cost = std::numeric_limits<ResultType>::infinity();
  }
  std::atomic<bool> target_attained = false;
  std::vector<ResultType> cost(jso_params.initial_population_size,
                               std::numeric_limits<ResultType>::quiet_NaN());
  for (size_t i = 0; i < cost.size(); ++i) {
    cost[i] = cost_function(population[i]);
    if (!isnan(target_value) && cost[i] <= target_value) {
      target_attained = true;
    }
    if (current_minimum_cost && cost[i] < *current_minimum_cost) {
      *current_minimum_cost = cost[i];
    }
    if (queries) {
      queries->push_back(std::make_pair(population[i], cost[i]));
    }
  }
  std::vector<size_t> indices = detail::best_indices(cost);
  std::atomic<size_t> function_evaluations = population.size();
#if BOOST_MATH_DEBUG_JSO
  {
    auto const &min_cost = cost[indices[0]];
    auto const &location = population[indices[0]];
    std::cout << __FILE__ << ":" << __LINE__ << ":" << __func__;
    std::cout << "\n\tMinimum cost after evaluation of initial random "
                 "population of size "
              << population.size() << " is " << min_cost << "."
              << "\n\tLocation {";
    for (size_t i = 0; i < location.size() - 1; ++i) {
      std::cout << location[i] << ", ";
    }
    std::cout << location.back() << "}.\n";
  }
#endif
  // Algorithm 1: jSO algorithm, Line 3:
  // "Set all values in M_F to 0.5":
  // I believe this is a typo: Compare with "Section B. Experimental Results",
  // last bullet, which claims this should be set to 0.3. The reference
  // implementation also does 0.3:
  size_t H = 5;
  std::vector<DimensionlessReal> M_F(H, static_cast<DimensionlessReal>(0.3));
  // Algorithm 1: jSO algorithm, Line 4:
  // "Set all values in M_CR to 0.8":
  std::vector<DimensionlessReal> M_CR(H, static_cast<DimensionlessReal>(0.8));

  std::uniform_real_distribution<DimensionlessReal> unif01(DimensionlessReal(0), DimensionlessReal(1));
  bool keep_going = !target_attained;
  if (cancellation && *cancellation) {
    keep_going = false;
  }
  // k from:
  // http://metahack.org/CEC2014-Tanabe-Fukunaga.pdf, Algorithm 1:
  // Determines where Lehmer means are stored in the memory:
  size_t k = 0;
  size_t minimum_population_size = (max)(size_t(4), size_t(jso_params.threads));
  while (keep_going) {
    // Algorithm 1, jSO, Line 7:
    std::vector<DimensionlessReal> S_CR;
    std::vector<DimensionlessReal> S_F;
    // Equation 9 of L-SHADE:
    std::vector<ResultType> delta_f;
    for (size_t i = 0; i < population.size(); ++i) {
      // Algorithm 1, jSO, Line 9:
      auto ri = gen() % H;
      // Algorithm 1, jSO, Line 10-13:
      // Again, what is written in the pseudocode is not quite right.
      // What they mean is mu_F = 0.9-the historical memory is not used.
      // I confess I find it weird to store the historical memory if we're just
      // gonna ignore it, but that's what the paper and the reference
      // implementation says!
      DimensionlessReal mu_F = static_cast<DimensionlessReal>(0.9);
      DimensionlessReal mu_CR = static_cast<DimensionlessReal>(0.9);
      if (ri != H - 1) {
        mu_F = M_F[ri];
        mu_CR = M_CR[ri];
      }
      // Algorithm 1, jSO, Line 14-18:
      DimensionlessReal crossover_probability = static_cast<DimensionlessReal>(0);
      if (mu_CR >= 0) {
        using std::normal_distribution;
        normal_distribution<DimensionlessReal> normal(mu_CR, static_cast<DimensionlessReal>(0.1));
        crossover_probability = normal(gen);
        // Clamp comes from L-SHADE description:
        crossover_probability = clamp(crossover_probability, DimensionlessReal(0), DimensionlessReal(1));
      }
      // Algorithm 1, jSO, Line 19-23:
      // Note that the pseudocode uses a "max_generations parameter",
      // but the reference implementation does not.
      // Since we already require specification of max_function_evaluations,
      // the pseudocode adds an unnecessary parameter.
      if (4 * function_evaluations < jso_params.max_function_evaluations) {
        crossover_probability = (max)(crossover_probability, DimensionlessReal(0.7));
      } else if (2 * function_evaluations <
                 jso_params.max_function_evaluations) {
        crossover_probability = (max)(crossover_probability, DimensionlessReal(0.6));
      }

      // Algorithm 1, jSO, Line 24-27:
      // Note the adjustments to the pseudocode given in the reference
      // implementation.
      cauchy_distribution<DimensionlessReal> cauchy(mu_F, static_cast<DimensionlessReal>(0.1));
      DimensionlessReal F;
      do {
        F = cauchy(gen);
        if (F > 1) {
          F = 1;
        }
      } while (F <= 0);
      DimensionlessReal threshold = static_cast<DimensionlessReal>(7) / static_cast<DimensionlessReal>(10);
      if ((10 * function_evaluations <
           6 * jso_params.max_function_evaluations) &&
          (F > threshold)) {
        F = threshold;
      }
      // > p value for mutation strategy linearly decreases from pmax to pmin
      // during the evolutionary process, > where pmax = 0.25 in jSO and pmin =
      // pmax/2.
      DimensionlessReal p = DimensionlessReal(0.25) * (1 - static_cast<DimensionlessReal>(function_evaluations) /
                                     (2 * jso_params.max_function_evaluations));
      // Equation (4) of the reference:
      DimensionlessReal Fw = static_cast<DimensionlessReal>(1.2) * F;
      if (10 * function_evaluations < 4 * jso_params.max_function_evaluations) {
        if (10 * function_evaluations <
            2 * jso_params.max_function_evaluations) {
          Fw = static_cast<DimensionlessReal>(0.7) * F;
        } else {
          Fw = static_cast<DimensionlessReal>(0.8) * F;
        }
      }
      // Algorithm 1, jSO, Line 28:
      // "ui,g := current-to-pBest-w/1/bin using Eq. (3)"
      // This is not explained in the reference, but "current-to-pBest"
      // strategy means "select randomly among the best values available."
      // See:
      // Zhang, J., & Sanderson, A. C. (2009).
      // JADE: adaptive differential evolution with optional external archive.
      // IEEE Transactions on evolutionary computation, 13(5), 945-958.
      // > As a generalization of DE/current-to- best,
      // > DE/current-to-pbest utilizes not only the best solution information
      // > but also the information of other good solutions.
      // > To be specific, any of the top 100p%, p in (0, 1],
      // > solutions can be randomly chosen in DE/current-to-pbest to play the
      // role > designed exclusively for the single best solution in
      // DE/current-to-best."
      size_t max_p_index = static_cast<size_t>(std::ceil(p * indices.size()));
      size_t p_best_idx = gen() % max_p_index;
      // We now need r1, r2 so that r1 != r2 != i:
      size_t r1;
      do {
        r1 = gen() % population.size();
      } while (r1 == i);
      size_t r2;
      do {
        r2 = gen() % (population.size() + archive.size());
      } while (r2 == r1 || r2 == i);

      ArgumentContainer trial_vector;
      if constexpr (detail::has_resize_v<ArgumentContainer>) {
        trial_vector.resize(dimension);
      }
      auto const &xi = population[i];
      auto const &xr1 = population[r1];
      ArgumentContainer xr2;
      if (r2 < population.size()) {
        xr2 = population[r2];
      } else {
        xr2 = archive[r2 - population.size()];
      }
      auto const &x_p_best = population[p_best_idx];
      for (size_t j = 0; j < dimension; ++j) {
        auto guaranteed_changed_idx = gen() % dimension;
        if (unif01(gen) < crossover_probability ||
            j == guaranteed_changed_idx) {
          auto tmp = xi[j] + Fw * (x_p_best[j] - xi[j]) + F * (xr1[j] - xr2[j]);
          auto const &lb = jso_params.lower_bounds[j];
          auto const &ub = jso_params.upper_bounds[j];
          // Some others recommend regenerating the indices rather than
          // clamping; I dunno seems like it could get stuck regenerating . . .
          // Another suggestion is provided in:
          // "JADE: Adaptive Differential Evolution with Optional External
          // Archive" page 947. Perhaps we should implement it!
          trial_vector[j] = clamp(tmp, lb, ub);
        } else {
          trial_vector[j] = population[i][j];
        }
      }
      auto trial_cost = cost_function(trial_vector);
      function_evaluations++;
      if (isnan(trial_cost)) {
        continue;
      }
      if (queries) {
        queries->push_back(std::make_pair(trial_vector, trial_cost));
      }

      // Successful trial:
      if (trial_cost < cost[i] || isnan(cost[i])) {
        if (!isnan(target_value) && trial_cost <= target_value) {
          target_attained = true;
        }
        if (current_minimum_cost && trial_cost < *current_minimum_cost) {
          *current_minimum_cost = trial_cost;
        }
        // Can't decide on improvement if the previous evaluation was a NaN:
        if (!isnan(cost[i])) {
          if (crossover_probability > 1 || crossover_probability < 0) {
            throw std::domain_error("Crossover probability is weird.");
          }
          if (F > 1 || F < 0) {
            throw std::domain_error("Scale factor (F) is weird.");
          }
          S_CR.push_back(crossover_probability);
          S_F.push_back(F);
          delta_f.push_back(abs(cost[i] - trial_cost));
        }
        // Build the historical archive:
        if (archive.size() < cost.size()) {
          archive.push_back(trial_vector);
        } else {
          // If it's already built, then put the successful trial in a random index:
          archive.resize(cost.size());
          auto idx = gen() % archive.size();
          archive[idx] = trial_vector;
        }
        cost[i] = trial_cost;
        population[i] = trial_vector;
      }
    }

    indices = detail::best_indices(cost);

    // If there are no successful updates this generation, we do not update the
    // historical memory:
    if (S_CR.size() > 0) {
      std::vector<DimensionlessReal> weights(S_CR.size(),
                                std::numeric_limits<DimensionlessReal>::quiet_NaN());
      ResultType delta_sum = static_cast<ResultType>(0);
      for (auto const &delta : delta_f) {
        delta_sum += delta;
      }
      if (delta_sum <= 0 || !isfinite(delta_sum)) {
        std::ostringstream oss;
        oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
        oss << "\n\tYou've hit a bug: The sum of improvements must be strictly "
               "positive, but got "
            << delta_sum << " improvement from " << S_CR.size()
            << " successful updates.\n";
        oss << "\tImprovements: {" << std::hexfloat;
        for (size_t l = 0; l < delta_f.size() -1; ++l) {
          oss << delta_f[l] << ", ";
        }
        oss << delta_f.back() << "}.\n";
        throw std::logic_error(oss.str());
      }
      for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = delta_f[i] / delta_sum;
      }

      M_CR[k] = detail::weighted_lehmer_mean(S_CR, weights);
      M_F[k] = detail::weighted_lehmer_mean(S_F, weights);
    }
    k++;
    if (k == M_F.size()) {
      k = 0;
    }
    if (target_attained) {
      break;
    }
    if (cancellation && *cancellation) {
      break;
    }
    if (function_evaluations >= jso_params.max_function_evaluations) {
      break;
    }
    // Linear population size reduction:
    size_t new_population_size = static_cast<size_t>(round(
        -double(jso_params.initial_population_size - minimum_population_size) *
            function_evaluations /
            static_cast<double>(jso_params.max_function_evaluations) +
        jso_params.initial_population_size));
    size_t num_erased = population.size() - new_population_size;
    std::vector<size_t> indices_to_erase(num_erased);
    size_t j = 0;
    for (size_t i = population.size() - 1; i >= new_population_size; --i) {
      indices_to_erase[j++] = indices[i];
    }
    std::sort(indices_to_erase.rbegin(), indices_to_erase.rend());
    for (auto const &idx : indices_to_erase) {
      population.erase(population.begin() + idx);
      cost.erase(cost.begin() + idx);
    }
    indices.resize(new_population_size);
  }

#if BOOST_MATH_DEBUG_JSO
  {
    auto const &min_cost = cost[indices[0]];
    auto const &location = population[indices[0]];
    std::cout << __FILE__ << ":" << __LINE__ << ":" << __func__;
    std::cout << "\n\tMinimum cost after completion is " << min_cost
              << ".\n\tLocation: {";
    for (size_t i = 0; i < location.size() - 1; ++i) {
      std::cout << location[i] << ", ";
    }
    std::cout << location.back() << "}.\n";
    std::cout << "\tRequired " << function_evaluations
              << " function calls out of "
              << jso_params.max_function_evaluations << " allowed.\n";
    if (target_attained) {
      std::cout << "\tReason for return: Target value attained.\n";
    }
    std::cout << "\tHistorical crossover probabilities (M_CR): {";
    for (size_t i = 0; i < M_CR.size() - 1; ++i) {
      std::cout << M_CR[i] << ", ";
    }
    std::cout << M_CR.back() << "}.\n";
    std::cout << "\tHistorical scale factors (M_F): {";
    for (size_t i = 0; i < M_F.size() - 1; ++i) {
      std::cout << M_F[i] << ", ";
    }
    std::cout << M_F.back() << "}.\n";
    std::cout << "\tFinal population size: " << population.size() << ".\n";
  }
#endif
  return population[indices[0]];
}

} // namespace boost::math::optimization
#endif
