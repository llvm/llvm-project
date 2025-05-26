//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include "test_autodiff.hpp"
#include <boost/math/tools/test_value.hpp>

BOOST_AUTO_TEST_SUITE(test_autodiff_4)

BOOST_AUTO_TEST_CASE_TEMPLATE(lround_llround_lltrunc_truncl, T,
                              all_float_types) {
  using boost::math::llround;
  using boost::math::lltrunc;
  using boost::math::lround;
  using boost::multiprecision::llround;
  using boost::multiprecision::lltrunc;
  using boost::multiprecision::lround;
  using detail::llround;
  using detail::lltrunc;
  using detail::lround;
  using detail::truncl;
  using std::truncl;

  constexpr std::size_t m = 3;
  const auto &cx = static_cast<T>(3.25);
  auto x = make_fvar<T, m>(cx);
  auto yl = lround(x);
  BOOST_CHECK_EQUAL(yl, lround(cx));
  auto yll = llround(x);
  BOOST_CHECK_EQUAL(yll, llround(cx));
  BOOST_CHECK_EQUAL(lltrunc(cx), lltrunc(x));

#ifndef BOOST_NO_CXX17_IF_CONSTEXPR
  if constexpr (!bmp::is_number<T>::value &&
                !bmp::is_number_expression<T>::value) {
    BOOST_CHECK_EQUAL(truncl(x), truncl(cx));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(equality, T, all_float_types) {
  BOOST_MATH_STD_USING
  using boost::math::epsilon_difference;
  using boost::math::fpclassify;
  using boost::math::ulp;
  using std::fpclassify;

  constexpr std::size_t m = 3;
  // check zeros
  {
    auto x = make_fvar<T, m>(T(0));
    auto y = T(-0.0);
    BOOST_CHECK_EQUAL(x.derivative(0u), y);
  }
}

#if defined(BOOST_AUTODIFF_TESTING_INCLUDE_MULTIPRECISION)
BOOST_AUTO_TEST_CASE_TEMPLATE(multiprecision, T, multiprecision_float_types) {
  using boost::multiprecision::fabs;
  using detail::fabs;
  using std::fabs;

  const T eps = 3000 * std::numeric_limits<T>::epsilon();
  constexpr std::size_t Nw = 3;
  constexpr std::size_t Nx = 2;
  constexpr std::size_t Ny = 4;
  constexpr std::size_t Nz = 3;
  const auto w = make_fvar<T, Nw>(11);
  const auto x = make_fvar<T, 0, Nx>(12);
  const auto y = make_fvar<T, 0, 0, Ny>(13);
  const auto z = make_fvar<T, 0, 0, 0, Nz>(14);
  const auto v =
      mixed_partials_f(w, x, y, z); // auto = autodiff_fvar<T,Nw,Nx,Ny,Nz>
  // Calculated from Mathematica symbolic differentiation.
  const T answer = BOOST_MATH_TEST_VALUE(T, 1976.31960074779771777988187529041872090812118921875499076582535951111845769110560421820940516423255314);
  // BOOST_CHECK_CLOSE(v.derivative(Nw,Nx,Ny,Nz), answer, eps); // Doesn't work
  // for cpp_dec_float
  const T relative_error =
      static_cast<T>(fabs(v.derivative(Nw, Nx, Ny, Nz) / answer - 1));
  BOOST_CHECK_LT(relative_error, eps);
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(acosh_hpp, T, all_float_types) {
  using boost::math::acosh;
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{1, 100};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto autodiff_v = acosh(make_fvar<T, m>(x));
    auto anchor_v = acosh(x);
    BOOST_CHECK_CLOSE(autodiff_v.derivative(0u), anchor_v,
                      1e3 * test_constants::pct_epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(asinh_hpp, T, all_float_types) {
  using boost::math::asinh;
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-100, 100};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();

    auto autodiff_v = asinh(make_fvar<T, m>(x));
    auto anchor_v = asinh(x);
    BOOST_CHECK_CLOSE(autodiff_v.derivative(0u), anchor_v,
                      1e3 * test_constants::pct_epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atanh_hpp, T, all_float_types) {
  using boost::math::nextafter;
  using std::nextafter;

  using boost::math::atanh;
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-1, 1};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = nextafter(x_sampler.next(), T(0));

    auto autodiff_v = atanh(make_fvar<T, m>(x));
    auto anchor_v = atanh(x);
    BOOST_CHECK_CLOSE(autodiff_v.derivative(0u), anchor_v,
                      1e3 * test_constants::pct_epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atan_hpp, T, all_float_types) {
  using boost::math::float_prior;
  using boost::math::fpclassify;
  using boost::math::signbit;
  using boost::math::differentiation::detail::atan;
  using boost::multiprecision::atan;
  using boost::multiprecision::fabs;
  using boost::multiprecision::fpclassify;
  using boost::multiprecision::signbit;
  using detail::fabs;
  using std::atan;
  using std::fabs;

  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-1, 1};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = T(1);
    while (fpclassify(T(fabs(x) - 1)) == FP_ZERO) {
      x = x_sampler.next();
    }

    auto autodiff_v = atan(make_fvar<T, m>(x));
    auto anchor_v = atan(x);
    BOOST_CHECK_CLOSE(autodiff_v.derivative(0u), anchor_v,
                      T(1e3) * test_constants::pct_epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(bernoulli_hpp, T, all_float_types) {

  using boost::multiprecision::min;
  using std::min;
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  for (auto i : boost::irange(test_constants::n_samples)) {
    {
      auto autodiff_v = boost::math::bernoulli_b2n<autodiff_fvar<T, m>>(i);
      auto anchor_v = boost::math::bernoulli_b2n<T>(i);
      BOOST_CHECK_CLOSE(autodiff_v.derivative(0u), anchor_v,
                        50 * test_constants::pct_epsilon());
    }
    {
      auto i_ = (min)(19, i);
      auto autodiff_v = boost::math::tangent_t2n<autodiff_fvar<T, m>>(i_);
      auto anchor_v = boost::math::tangent_t2n<T>(i_);
      BOOST_CHECK_CLOSE(autodiff_v.derivative(0u), anchor_v,
                        50 * test_constants::pct_epsilon());
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
