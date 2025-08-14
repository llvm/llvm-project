//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include "test_autodiff.hpp"

BOOST_AUTO_TEST_SUITE(test_autodiff_7)

BOOST_AUTO_TEST_CASE_TEMPLATE(expm1_hpp, T, all_float_types) {
  using boost::math::differentiation::detail::log;
  using boost::multiprecision::log;
  using std::log;
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-log(T(2000)), log(T(2000))};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    BOOST_CHECK_CLOSE(boost::math::expm1(make_fvar<T, m>(x)).derivative(0u),
                      boost::math::expm1(x),
                      50 * test_constants::pct_epsilon());
  }
}

BOOST_AUTO_TEST_SUITE_END()
