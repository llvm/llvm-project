/*
 * Copyright Nick Thompson, 2024
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#ifndef BOOST_MATH_OPTIMIZATION_DETAIL_COMMON_HPP
#define BOOST_MATH_OPTIMIZATION_DETAIL_COMMON_HPP
#include <algorithm> // for std::sort
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <random>
#include <type_traits>  // for std::false_type

namespace boost::math::optimization::detail {

template <typename T, typename = void> struct has_resize : std::false_type {};

template <typename T>
struct has_resize<T, std::void_t<decltype(std::declval<T>().resize(size_t{}))>> : std::true_type {};

template <typename T> constexpr bool has_resize_v = has_resize<T>::value;

template <typename ArgumentContainer>
void validate_bounds(ArgumentContainer const &lower_bounds, ArgumentContainer const &upper_bounds) {
  using std::isfinite;
  std::ostringstream oss;
  if (lower_bounds.size() == 0) {
    oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
    oss << ": The dimension of the problem cannot be zero.";
    throw std::domain_error(oss.str());
  }
  if (upper_bounds.size() != lower_bounds.size()) {
    oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
    oss << ": There must be the same number of lower bounds as upper bounds, but given ";
    oss << upper_bounds.size() << " upper bounds, and " << lower_bounds.size() << " lower bounds.";
    throw std::domain_error(oss.str());
  }
  for (size_t i = 0; i < lower_bounds.size(); ++i) {
    auto lb = lower_bounds[i];
    auto ub = upper_bounds[i];
    if (lb > ub) {
      oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
      oss << ": The upper bound must be greater than or equal to the lower bound, but the upper bound is " << ub
          << " and the lower is " << lb << ".";
      throw std::domain_error(oss.str());
    }
    if (!isfinite(lb)) {
      oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
      oss << ": The lower bound must be finite, but got " << lb << ".";
      oss << " For infinite bounds, emulate with std::numeric_limits<Real>::lower() or use a standard infinite->finite "
             "transform.";
      throw std::domain_error(oss.str());
    }
    if (!isfinite(ub)) {
      oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
      oss << ": The upper bound must be finite, but got " << ub << ".";
      oss << " For infinite bounds, emulate with std::numeric_limits<Real>::max() or use a standard infinite->finite "
             "transform.";
      throw std::domain_error(oss.str());
    }
  }
}

template <typename ArgumentContainer, class URBG>
std::vector<ArgumentContainer> random_initial_population(ArgumentContainer const &lower_bounds,
                                                         ArgumentContainer const &upper_bounds,
                                                         size_t initial_population_size, URBG &&gen) {
  using Real = typename ArgumentContainer::value_type;
  using DimensionlessReal = decltype(Real()/Real());
  constexpr bool has_resize = detail::has_resize_v<ArgumentContainer>;
  std::vector<ArgumentContainer> population(initial_population_size);
  auto const dimension = lower_bounds.size();
  for (size_t i = 0; i < population.size(); ++i) {
    if constexpr (has_resize) {
      population[i].resize(dimension);
    } else {
      // Argument type must be known at compile-time; like std::array:
      if (population[i].size() != dimension) {
        std::ostringstream oss;
        oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
        oss << ": For containers which do not have resize, the default size must be the same as the dimension, ";
        oss << "but the default container size is " << population[i].size() << " and the dimension of the problem is "
            << dimension << ".";
        oss << " The function argument container type is " << typeid(ArgumentContainer).name() << ".\n";
        throw std::runtime_error(oss.str());
      }
    }
  }

  // Why don't we provide an option to initialize with (say) a Gaussian distribution?
  // > If the optimum's location is fairly well known,
  // > a Gaussian distribution may prove somewhat faster, although it
  // > may also increase the probability that the population will converge prematurely.
  // > In general, uniform distributions are preferred, since they best reflect
  // > the lack of knowledge about the optimum's location.
  //  - Differential Evolution: A Practical Approach to Global Optimization
  // That said, scipy uses Latin Hypercube sampling and says self-avoiding sequences are preferable.
  // So this is something that could be investigated and potentially improved.
  using std::uniform_real_distribution;
  uniform_real_distribution<DimensionlessReal> dis(DimensionlessReal(0), DimensionlessReal(1));
  for (size_t i = 0; i < population.size(); ++i) {
    for (size_t j = 0; j < dimension; ++j) {
      auto const &lb = lower_bounds[j];
      auto const &ub = upper_bounds[j];
      population[i][j] = lb + dis(gen) * (ub - lb);
    }
  }

  return population;
}

template <typename ArgumentContainer>
void validate_initial_guess(ArgumentContainer const &initial_guess, ArgumentContainer const &lower_bounds,
                            ArgumentContainer const &upper_bounds) {
  using std::isfinite;
  std::ostringstream oss;
  auto const dimension = lower_bounds.size();
  if (initial_guess.size() != dimension) {
    oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
    oss << ": The initial guess must have the same dimensions as the problem,";
    oss << ", but the problem size is " << dimension << " and the initial guess has " << initial_guess.size()
        << " elements.";
    throw std::domain_error(oss.str());
  }
  for (size_t i = 0; i < dimension; ++i) {
    auto lb = lower_bounds[i];
    auto ub = upper_bounds[i];
    if (!isfinite(initial_guess[i])) {
      oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
      oss << ": At index " << i << ", the initial guess is " << initial_guess[i]
          << ", make sure all elements of the initial guess are finite.";
      throw std::domain_error(oss.str());
    }
    if (initial_guess[i] < lb || initial_guess[i] > ub) {
      oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
      oss << ": At index " << i << " the initial guess " << initial_guess[i] << " is not in the bounds [" << lb << ", "
          << ub << "].";
      throw std::domain_error(oss.str());
    }
  }
}

// Return indices corresponding to the minimum function values.
template <typename Real> std::vector<size_t> best_indices(std::vector<Real> const &function_values) {
  using std::isnan;
  const size_t n = function_values.size();
  std::vector<size_t> indices(n);
  for (size_t i = 0; i < n; ++i) {
    indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    if (isnan(function_values[a])) {
      return false;
    }
    if (isnan(function_values[b])) {
      return true;
    }
    return function_values[a] < function_values[b];
  });
  return indices;
}

template<typename RandomAccessContainer>
auto weighted_lehmer_mean(RandomAccessContainer const & values, RandomAccessContainer const & weights) {
  using std::isfinite;
  if (values.size() != weights.size()) {
    std::ostringstream oss;
    oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
    oss << ": There must be the same number of weights as values, but got " << values.size() << " values and " << weights.size() << " weights.";
    throw std::logic_error(oss.str());
  }
  if (values.size() == 0) {
    std::ostringstream oss;
    oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
    oss << ": There must at least one value provided.";
    throw std::logic_error(oss.str());
  }
  using Real = typename RandomAccessContainer::value_type;
  Real numerator = 0;
  Real denominator = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    if (weights[i] < 0 || !isfinite(weights[i])) {
      std::ostringstream oss;
      oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
      oss << ": All weights must be positive and finite, but got received weight " << weights[i] << " at index " << i << " of " << weights.size() << ".";
      throw std::domain_error(oss.str());
    }
    Real tmp = weights[i]*values[i];
    numerator += tmp*values[i];
    denominator += tmp;
  }
  return numerator/denominator;
}

} // namespace boost::math::optimization::detail
#endif
