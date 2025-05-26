//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include "test_autodiff.hpp"

BOOST_AUTO_TEST_SUITE(test_autodiff_2)

BOOST_AUTO_TEST_CASE_TEMPLATE(one_over_one_plus_x_squared, T, all_float_types) {
  constexpr std::size_t m = 4;
  const T cx(1);
  auto f = make_fvar<T, m>(cx);
  // f = 1 / ((f *= f) += 1);
  f *= f;
  f += T(1);
  f = f.inverse();
  BOOST_CHECK_EQUAL(f.derivative(0u), T(0.5));
  BOOST_CHECK_EQUAL(f.derivative(1u), T(-0.5));
  BOOST_CHECK_EQUAL(f.derivative(2u), T(0.5));
  BOOST_CHECK_EQUAL(f.derivative(3u), 0);
  BOOST_CHECK_EQUAL(f.derivative(4u), -3);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(exp_test, T, all_float_types) {
  using std::exp;
  constexpr std::size_t m = 4;
  const T cx = 2;
  const auto x = make_fvar<T, m>(cx);
  auto y = exp(x);
  for (auto i : boost::irange(m + 1)) {
    // std::cout.precision(100);
    // std::cout << "y.derivative("<<i<<") = " << y.derivative(i) << ",
    // std::exp(cx) = " << std::exp(cx) << std::endl;
    BOOST_CHECK_CLOSE_FRACTION(static_cast<T>(y.derivative(i)), static_cast<T>(exp(cx)),
                               std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pow, T, bin_float_types) {
  const T eps = 201 * std::numeric_limits<T>::epsilon(); // percent
  using std::log;
  using std::pow;
  constexpr std::size_t m = 5;
  constexpr std::size_t n = 4;
  const T cx = 2;
  const T cy = 3;
  const auto x = make_fvar<T, m>(cx);
  const auto y = make_fvar<T, m, n>(cy);
  auto z0 = pow(x, cy);
  BOOST_CHECK_EQUAL(z0.derivative(0u), pow(cx, cy));
  BOOST_CHECK_EQUAL(z0.derivative(1u), cy * pow(cx, cy - 1));
  BOOST_CHECK_EQUAL(z0.derivative(2u), cy * (cy - 1) * pow(cx, cy - 2));
  BOOST_CHECK_EQUAL(z0.derivative(3u),
                    cy * (cy - 1) * (cy - 2) * pow(cx, cy - 3));
  BOOST_CHECK_EQUAL(z0.derivative(4u), 0u);
  BOOST_CHECK_EQUAL(z0.derivative(5u), 0u);
  auto z1 = pow(cx, y);
  BOOST_CHECK_CLOSE(z1.derivative(0u, 0u), pow(cx, cy), eps);
  for (auto j : boost::irange(std::size_t(1), n + 1)) {
    BOOST_CHECK_CLOSE(z1.derivative(0u, j), pow(log(cx), T(j)) * pow(cx, cy), eps);
  }

  for (auto i : boost::irange(std::size_t(1), m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      BOOST_CHECK_EQUAL(z1.derivative(i, j), 0);
    }
  }

  const auto z2 = pow(x, y);

  for (auto j : boost::irange(n + 1)) {
    BOOST_CHECK_CLOSE(z2.derivative(0u, j), pow(cx, cy) * pow(log(cx), T(j)), eps);
  }
  for (auto j : boost::irange(n + 1)) {
    BOOST_CHECK_CLOSE(z2.derivative(1u, j),
                      pow(cx, cy - 1) * pow(log(cx), T(static_cast<int>(j) - 1)) *
                          (cy * log(cx) + j),
                      eps);
  }
  BOOST_CHECK_CLOSE(z2.derivative(2u, 0u), pow(cx, cy - 2) * cy * (cy - 1),
                    eps);
  BOOST_CHECK_CLOSE(z2.derivative(2u, 1u),
                    pow(cx, cy - 2) * (cy * (cy - 1) * log(cx) + 2 * cy - 1),
                    eps);
  for (auto j : boost::irange(std::size_t(2u), n + 1)) {
    BOOST_CHECK_CLOSE(z2.derivative(2u, j),
                      pow(cx, cy - 2) * pow(log(cx), T(j - 2)) *
                          (j * (2 * cy - 1) * log(cx) + (j - 1) * j +
                           (cy - 1) * cy * pow(log(cx), T(2))),
                      eps);
  }
  BOOST_CHECK_CLOSE(z2.derivative(2u, 4u),
                    pow(cx, cy - 2) * pow(log(cx), T(2)) *
                        (4 * (2 * cy - 1) * log(cx) + (4 - 1) * 4 +
                         (cy - 1) * cy * pow(log(cx), T(2))),
                    eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pow0, T, bin_float_types) {
  using std::pow;
  constexpr std::size_t m = 5;
  const T cx = 0;
  {
    const T cy = 3;
    const auto x = make_fvar<T, m>(cx);
    auto z0 = pow(x, cy);
    BOOST_CHECK_EQUAL(z0.derivative(0u), pow(cx, cy));
    BOOST_CHECK_EQUAL(z0.derivative(1u), cy * pow(cx, cy - 1));
    BOOST_CHECK_EQUAL(z0.derivative(2u), cy * (cy - 1) * pow(cx, cy - 2));
    BOOST_CHECK_EQUAL(z0.derivative(3u),
                      cy * (cy - 1) * (cy - 2) * pow(cx, cy - 3));
    BOOST_CHECK_EQUAL(z0.derivative(4u), 0u);
    BOOST_CHECK_EQUAL(z0.derivative(5u), 0u);
  }
  {
    const T cy = T(3.5);
    const auto x = make_fvar<T, m>(cx);
    auto z0 = pow(x, cy);
    BOOST_CHECK_EQUAL(z0.derivative(0u), pow(cx, cy));
    BOOST_CHECK_EQUAL(z0.derivative(1u), cy * pow(cx, cy - 1));
    BOOST_CHECK_EQUAL(z0.derivative(2u), cy * (cy - 1) * pow(cx, cy - 2));
    BOOST_CHECK_EQUAL(z0.derivative(3u),
                      cy * (cy - 1) * (cy - 2) * pow(cx, cy - 3));
    BOOST_CHECK_EQUAL(z0.derivative(4u),
                      cy * (cy - 1) * (cy - 2) * (cy - 3) * pow(cx, cy - 4));
    BOOST_CHECK_EQUAL(z0.derivative(5u),
                      cy * (cy - 1) * (cy - 2) * (cy - 3) * (cy - 4) * pow(cx, cy - 5));
  }
}

// TODO Tests around y=0: pow(x,y)
BOOST_AUTO_TEST_CASE_TEMPLATE(pow2, T, bin_float_types) {
  const T eps = 4000 * std::numeric_limits<T>::epsilon(); // percent
  using std::pow;
  constexpr std::size_t m = 5;
  constexpr std::size_t n = 5;
  const T cx = 2;
  const T cy = 5 / T(2);
  const auto x = make_fvar<T, m>(cx);
  const auto y = make_fvar<T, 0, n>(cy);
  const auto z = pow(x, y);
  using namespace boost::math::constants;
  // Mathematica: Export["pow.csv", Flatten@Table[ Simplify@D[x^y,{x,i},{y,j}]
  // /. {x->2, y->5/2},
  //                    { i, 0, 5 }, { j, 0, 5 } ] ]
  // sed -rf pow.sed < pow.csv
  // with pow.sed script:
  // s/Log\[2\]\^([0-9]+)/pow(ln_two<T>(),\1)/g
  // s/Log\[2\]/ln_two<T>()/g
  // s/Sqrt\[2\]/root_two<T>()/g
  // s/[0-9]\/[0-9]+/\0.0/g
  // s/^"/static_cast<T>(/
  // s/"$/),/
  const T mathematica[]{
      static_cast<T>(4 * root_two<T>()),
      static_cast<T>(4 * root_two<T>() * ln_two<T>()),
      static_cast<T>(4 * root_two<T>() * pow(ln_two<T>(), (T)2)),
      static_cast<T>(4 * root_two<T>() * pow(ln_two<T>(), (T)3)),
      static_cast<T>(4 * root_two<T>() * pow(ln_two<T>(), (T)4)),
      static_cast<T>(4 * root_two<T>() * pow(ln_two<T>(), (T)5)),
      static_cast<T>(5 * root_two<T>()),
      static_cast<T>(2 * root_two<T>() * (1 + (5 * ln_two<T>()) / 2)),
      static_cast<T>(2 * root_two<T>() * ln_two<T>() *
                     (2 + (5 * ln_two<T>()) / 2)),
      static_cast<T>(2 * root_two<T>() * pow(ln_two<T>(), (T)2) *
                     (3 + (5 * ln_two<T>()) / 2)),
      static_cast<T>(2 * root_two<T>() * pow(ln_two<T>(), (T)3) *
                     (4 + (5 * ln_two<T>()) / 2)),
      static_cast<T>(2 * root_two<T>() * pow(ln_two<T>(), (T)4) *
                     (5 + (5 * ln_two<T>()) / 2)),
      static_cast<T>(15 / (2 * root_two<T>())),
      static_cast<T>(root_two<T>() * (4 + (15 * ln_two<T>()) / 4)),
      static_cast<T>(root_two<T>() *
                     (2 + 8 * ln_two<T>() + (15 * pow(ln_two<T>(), (T)2)) / 4)),
      static_cast<T>(root_two<T>() * ln_two<T>() *
                     (6 + 12 * ln_two<T>() + (15 * pow(ln_two<T>(), (T)2)) / 4)),
      static_cast<T>(root_two<T>() * pow(ln_two<T>(), (T)2) *
                     (12 + 16 * ln_two<T>() + (15 * pow(ln_two<T>(), (T)2)) / 4)),
      static_cast<T>(root_two<T>() * pow(ln_two<T>(), (T)3) *
                     (20 + 20 * ln_two<T>() + (15 * pow(ln_two<T>(), (T)2)) / 4)),
      static_cast<T>(15 / (8 * root_two<T>())),
      static_cast<T>((23 / 4.0 + (15 * ln_two<T>()) / 8) / root_two<T>()),
      static_cast<T>(
          (9 + (23 * ln_two<T>()) / 2 + (15 * pow(ln_two<T>(), (T)2)) / 8) /
          root_two<T>()),
      static_cast<T>((6 + 27 * ln_two<T>() + (69 * pow(ln_two<T>(), (T)2)) / 4 +
                      (15 * pow(ln_two<T>(), (T)3)) / 8) /
                     root_two<T>()),
      static_cast<T>(
          (ln_two<T>() * (24 + 54 * ln_two<T>() + 23 * pow(ln_two<T>(), (T)2) +
                          (15 * pow(ln_two<T>(), (T)3)) / 8)) /
          root_two<T>()),
      static_cast<T>((pow(ln_two<T>(), (T)2) *
                      (60 + 90 * ln_two<T>() + (115 * pow(ln_two<T>(), (T)2)) / 4 +
                       (15 * pow(ln_two<T>(), (T)3)) / 8)) /
                     root_two<T>()),
      static_cast<T>(-15 / (32 * root_two<T>())),
      static_cast<T>((-1 - (15 * ln_two<T>()) / 16) / (2 * root_two<T>())),
      static_cast<T>((7 - 2 * ln_two<T>() - (15 * pow(ln_two<T>(), (T)2)) / 16) /
                     (2 * root_two<T>())),
      static_cast<T>((24 + 21 * ln_two<T>() - 3 * pow(ln_two<T>(), (T)2) -
                      (15 * pow(ln_two<T>(), (T)3)) / 16) /
                     (2 * root_two<T>())),
      static_cast<T>((24 + 96 * ln_two<T>() + 42 * pow(ln_two<T>(), (T)2) -
                      4 * pow(ln_two<T>(), (T)3) -
                      (15 * pow(ln_two<T>(), (T)4)) / 16) /
                     (2 * root_two<T>())),
      static_cast<T>(
          (ln_two<T>() *
           (120 + 240 * ln_two<T>() + 70 * pow(ln_two<T>(), (T)2) -
            5 * pow(ln_two<T>(), (T)3) - (15 * pow(ln_two<T>(), (T)4)) / 16)) /
          (2 * root_two<T>())),
      static_cast<T>(45 / (128 * root_two<T>())),
      static_cast<T>((9 / 16.0 + (45 * ln_two<T>()) / 32) /
                     (4 * root_two<T>())),
      static_cast<T>((-25 / 2.0 + (9 * ln_two<T>()) / 8 +
                      (45 * pow(ln_two<T>(), (T)2)) / 32) /
                     (4 * root_two<T>())),
      static_cast<T>((-15 - (75 * ln_two<T>()) / 2 +
                      (27 * pow(ln_two<T>(), (T)2)) / 16 +
                      (45 * pow(ln_two<T>(), (T)3)) / 32) /
                     (4 * root_two<T>())),
      static_cast<T>((60 - 60 * ln_two<T>() - 75 * pow(ln_two<T>(), (T)2) +
                      (9 * pow(ln_two<T>(), (T)3)) / 4 +
                      (45 * pow(ln_two<T>(), (T)4)) / 32) /
                     (4 * root_two<T>())),
      static_cast<T>((120 + 300 * ln_two<T>() - 150 * pow(ln_two<T>(), (T)2) -
                      125 * pow(ln_two<T>(), (T)3) +
                      (45 * pow(ln_two<T>(), (T)4)) / 16 +
                      (45 * pow(ln_two<T>(), (T)5)) / 32) /
                     (4 * root_two<T>()))};
  std::size_t k = 0;
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      BOOST_CHECK_CLOSE(z.derivative(i, j), mathematica[k++], eps);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sqrt_test, T, all_float_types) {
  using std::pow;
  using std::sqrt;
  constexpr std::size_t m = 5;
  const T cx = 4;
  auto x = make_fvar<T, m>(cx);
  auto y = sqrt(x);
  BOOST_CHECK_CLOSE_FRACTION(y.derivative(0u), static_cast<T>(sqrt(cx)),
                             std::numeric_limits<T>::epsilon());
  BOOST_CHECK_CLOSE_FRACTION(y.derivative(1u), static_cast<T>(0.5 * pow(cx, T(-0.5))),
                             std::numeric_limits<T>::epsilon());
  BOOST_CHECK_CLOSE_FRACTION(y.derivative(2u), static_cast<T>(-0.5 * 0.5 * pow(cx, T(-1.5))),
                             std::numeric_limits<T>::epsilon());
  BOOST_CHECK_CLOSE_FRACTION(y.derivative(3u), static_cast<T>(0.5 * 0.5 * 1.5 * pow(cx, T(-2.5))),
                             std::numeric_limits<T>::epsilon());
  BOOST_CHECK_CLOSE_FRACTION(y.derivative(4u),
                             static_cast<T>(-0.5 * 0.5 * 1.5 * 2.5 * pow(cx, T(-3.5))),
                             std::numeric_limits<T>::epsilon());
  BOOST_CHECK_CLOSE_FRACTION(y.derivative(5u),
                             static_cast<T>(0.5 * 0.5 * 1.5 * 2.5 * 3.5 * pow(cx, T(-4.5))),
                             std::numeric_limits<T>::epsilon());
  x = make_fvar<T, m>(0);
  y = sqrt(x);
  // std::cout << "sqrt(0) = " << y << std::endl; // (0,inf,-inf,inf,-inf,inf)
  BOOST_CHECK_EQUAL(y.derivative(0u), 0);
  for (auto i : boost::irange(std::size_t(1), m + 1)) {
    BOOST_CHECK_EQUAL(y.derivative(i), (i % 2 == 1 ? 1 : -1) *
                                           std::numeric_limits<T>::infinity());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(log_test, T, all_float_types) {
  using std::log;
  using std::pow;
  constexpr std::size_t m = 5;
  const T cx = 2;
  auto x = make_fvar<T, m>(cx);
  auto y = log(x);
  BOOST_CHECK_CLOSE_FRACTION(y.derivative(0u), log(cx),
                             std::numeric_limits<T>::epsilon());
  BOOST_CHECK_CLOSE_FRACTION(y.derivative(1u), 1 / cx,
                             std::numeric_limits<T>::epsilon());
  BOOST_CHECK_CLOSE_FRACTION(y.derivative(2u), -1 / pow(cx, T(2)),
                             std::numeric_limits<T>::epsilon());
  BOOST_CHECK_CLOSE_FRACTION(y.derivative(3u), 2 / pow(cx, T(3)),
                             std::numeric_limits<T>::epsilon());
  BOOST_CHECK_CLOSE_FRACTION(y.derivative(4u), -6 / pow(cx, T(4)),
                             std::numeric_limits<T>::epsilon());
  BOOST_CHECK_CLOSE_FRACTION(y.derivative(5u), 24 / pow(cx, T(5)),
                             std::numeric_limits<T>::epsilon());
  x = make_fvar<T, m>(0);
  y = log(x);
  // std::cout << "log(0) = " << y << std::endl; // log(0) =
  // depth(1)(-inf,inf,-inf,inf,-inf,inf)
  for (auto i : boost::irange(m + 1)) {
    BOOST_CHECK_EQUAL(y.derivative(i), (i % 2 == 1 ? 1 : -1) *
                                           std::numeric_limits<T>::infinity());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ylogx, T, all_float_types) {
  using std::log;
  using std::pow;
  const T eps = (std::numeric_limits<T>::digits > 100 ? 300 : 100) * std::numeric_limits<T>::epsilon(); // percent
  constexpr std::size_t m = 5;
  constexpr std::size_t n = 4;
  const T cx = 2;
  const T cy = 3;
  const auto x = make_fvar<T, m>(cx);
  const auto y = make_fvar<T, m, n>(cy);
  auto z = y * log(x);
  BOOST_CHECK_EQUAL(z.derivative(0u, 0u), cy * log(cx));
  BOOST_CHECK_EQUAL(z.derivative(0u, 1u), log(cx));
  BOOST_CHECK_EQUAL(z.derivative(0u, 2u), 0);
  BOOST_CHECK_EQUAL(z.derivative(0u, 3u), 0);
  BOOST_CHECK_EQUAL(z.derivative(0u, 4u), 0);
  for (auto i : boost::irange(1u, static_cast<unsigned>(m + 1))) {
    BOOST_CHECK_CLOSE(z.derivative(i, 0u),
                      pow(-1, i - 1) * boost::math::factorial<T>(i - 1) * cy /
                          pow(cx, T(i)),
                      eps);
    BOOST_CHECK_CLOSE(
        z.derivative(i, 1u),
        pow(T(-1), T(i - 1)) * boost::math::factorial<T>(i - 1) / pow(cx, T(i)), eps);
    for (auto j : boost::irange(std::size_t(2), n + 1)) {
      BOOST_CHECK_EQUAL(z.derivative(i, j), 0u);
    }
  }
  auto z1 = exp(z);
  // RHS is confirmed by
  // https://www.wolframalpha.com/input/?i=D%5Bx%5Ey,%7Bx,2%7D,%7By,4%7D%5D+%2F.+%7Bx-%3E2.0,+y-%3E3.0%7D
  BOOST_CHECK_CLOSE(z1.derivative(2u, 4u),
                    pow(cx, cy - 2) * pow(log(cx), T(2)) *
                        (4 * (2 * cy - 1) * log(cx) + (4 - 1) * 4 +
                         (cy - 1) * cy * pow(log(cx), T(2))),
                    eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(frexp_test, T, all_float_types) {
  using std::exp2;
  using std::frexp;
  constexpr std::size_t m = 3;
  const T cx = T(3.5);
  const auto x = make_fvar<T, m>(cx);
  int exp, testexp;
  auto y = frexp(x, &exp);
  BOOST_CHECK_EQUAL(y.derivative(0u), frexp(cx, &testexp));
  BOOST_CHECK_EQUAL(exp, testexp);
  BOOST_CHECK_EQUAL(y.derivative(1u), exp2(-exp));
  BOOST_CHECK_EQUAL(y.derivative(2u), 0);
  BOOST_CHECK_EQUAL(y.derivative(3u), 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ldexp_test, T, all_float_types) {
  BOOST_MATH_STD_USING
  using boost::multiprecision::ldexp;
  constexpr auto m = 3u;
  const T cx = T(3.5);
  const auto x = make_fvar<T, m>(cx);
  constexpr auto exponent = 3;
  auto y = ldexp(x, exponent);
  BOOST_CHECK_EQUAL(y.derivative(0u), ldexp(cx, exponent));
  BOOST_CHECK_EQUAL(y.derivative(1u), exp2(exponent));
  BOOST_CHECK_EQUAL(y.derivative(2u), 0);
  BOOST_CHECK_EQUAL(y.derivative(3u), 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(cos_and_sin, T, bin_float_types) {
  using std::cos;
  using std::sin;
  const T eps = 200 * std::numeric_limits<T>::epsilon(); // percent
  constexpr std::size_t m = 5;
  const T cx = boost::math::constants::third_pi<T>();
  const auto x = make_fvar<T, m>(cx);
  auto cos5 = cos(x);
  BOOST_CHECK_CLOSE(cos5.derivative(0u), cos(cx), eps);
  BOOST_CHECK_CLOSE(cos5.derivative(1u), -sin(cx), eps);
  BOOST_CHECK_CLOSE(cos5.derivative(2u), -cos(cx), eps);
  BOOST_CHECK_CLOSE(cos5.derivative(3u), sin(cx), eps);
  BOOST_CHECK_CLOSE(cos5.derivative(4u), cos(cx), eps);
  BOOST_CHECK_CLOSE(cos5.derivative(5u), -sin(cx), eps);
  auto sin5 = sin(x);
  BOOST_CHECK_CLOSE(sin5.derivative(0u), sin(cx), eps);
  BOOST_CHECK_CLOSE(sin5.derivative(1u), cos(cx), eps);
  BOOST_CHECK_CLOSE(sin5.derivative(2u), -sin(cx), eps);
  BOOST_CHECK_CLOSE(sin5.derivative(3u), -cos(cx), eps);
  BOOST_CHECK_CLOSE(sin5.derivative(4u), sin(cx), eps);
  BOOST_CHECK_CLOSE(sin5.derivative(5u), cos(cx), eps);
  // Test Order = 0 for codecov
  auto cos0 = cos(make_fvar<T, 0>(cx));
  BOOST_CHECK_CLOSE(cos0.derivative(0u), cos(cx), eps);
  auto sin0 = sin(make_fvar<T, 0>(cx));
  BOOST_CHECK_CLOSE(sin0.derivative(0u), sin(cx), eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(acos_test, T, bin_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon(); // percent
  using std::acos;
  using std::pow;
  using std::sqrt;
  constexpr std::size_t m = 5;
  const T cx = T(0.5);
  auto x = make_fvar<T, m>(cx);
  auto y = acos(x);
  BOOST_CHECK_CLOSE(y.derivative(0u), acos(cx), eps);
  BOOST_CHECK_CLOSE(y.derivative(1u), -1 / sqrt(1 - cx * cx), eps);
  BOOST_CHECK_CLOSE(y.derivative(2u), static_cast<T>(-cx / pow(1 - cx * cx, T(1.5))), eps);
  BOOST_CHECK_CLOSE(y.derivative(3u),
                    static_cast<T>(-(2 * cx * cx + 1) / pow(1 - cx * cx, T(2.5))), eps);
  BOOST_CHECK_CLOSE(y.derivative(4u),
                    static_cast<T>(-3 * cx * (2 * cx * cx + 3) / pow(1 - cx * cx, T(3.5))), eps);
  BOOST_CHECK_CLOSE(y.derivative(5u),
                    static_cast<T>(-(24 * (cx * cx + 3) * cx * cx + 9) / pow(1 - cx * cx, T(4.5))),
                    eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(acosh_test, T, bin_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon(); // percent
  using boost::math::acosh;
  constexpr std::size_t m = 5;
  const T cx = 2;
  auto x = make_fvar<T, m>(cx);
  auto y = acosh(x);
  // BOOST_CHECK_EQUAL(y.derivative(0) == acosh(cx)); // FAILS! acosh(2) is
  // overloaded for integral types
  BOOST_CHECK_CLOSE(y.derivative(0u), acosh(static_cast<T>(x)), eps);
  BOOST_CHECK_CLOSE(y.derivative(1u),
                    1 / boost::math::constants::root_three<T>(), eps);
  BOOST_CHECK_CLOSE(y.derivative(2u),
                    -2 / (3 * boost::math::constants::root_three<T>()), eps);
  BOOST_CHECK_CLOSE(y.derivative(3u),
                    1 / boost::math::constants::root_three<T>(), eps);
  BOOST_CHECK_CLOSE(y.derivative(4u),
                    -22 / (9 * boost::math::constants::root_three<T>()), eps);
  BOOST_CHECK_CLOSE(y.derivative(5u),
                    227 / (27 * boost::math::constants::root_three<T>()),
                    2 * eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(asin_test, T, bin_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon(); // percent
  using std::asin;
  using std::pow;
  using std::sqrt;
  constexpr std::size_t m = 5;
  const T cx = T(0.5);
  auto x = make_fvar<T, m>(cx);
  auto y = asin(x);
  BOOST_CHECK_CLOSE(y.derivative(0u), asin(static_cast<T>(x)), eps);
  BOOST_CHECK_CLOSE(y.derivative(1u), 1 / sqrt(1 - cx * cx), eps);
  BOOST_CHECK_CLOSE(y.derivative(2u), cx / pow(1 - cx * cx, T(1.5)), eps);
  BOOST_CHECK_CLOSE(y.derivative(3u), (2 * cx * cx + 1) / pow(1 - cx * cx, T(2.5)),
                    eps);
  BOOST_CHECK_CLOSE(y.derivative(4u),
                    3 * cx * (2 * cx * cx + 3) / pow(1 - cx * cx, T(3.5)), eps);
  BOOST_CHECK_CLOSE(y.derivative(5u),
                    (24 * (cx * cx + 3) * cx * cx + 9) / pow(1 - cx * cx, T(4.5)),
                    eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(asin_infinity, T, all_float_types) {
  const T eps = 100 * std::numeric_limits<T>::epsilon(); // percent
  constexpr std::size_t m = 5;
  auto x = make_fvar<T, m>(1);
  auto y = asin(x);
  // std::cout << "asin(1) = " << y << std::endl; //
  // depth(1)(1.5707963267949,inf,inf,-nan,-nan,-nan)
  BOOST_CHECK_CLOSE(y.derivative(0u), boost::math::constants::half_pi<T>(),
                    eps); // MacOS is not exact
  BOOST_CHECK_EQUAL(y.derivative(1u), std::numeric_limits<T>::infinity());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(asin_derivative, T, bin_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon(); // percent
  using std::pow;
  using std::sqrt;
  constexpr std::size_t m = 4;
  const T cx = T(0.5);
  auto x = make_fvar<T, m>(cx);
  auto y = T(1) - x * x;
  BOOST_CHECK_EQUAL(y.derivative(0u), 1 - cx * cx);
  BOOST_CHECK_EQUAL(y.derivative(1u), -2 * cx);
  BOOST_CHECK_EQUAL(y.derivative(2u), -2);
  BOOST_CHECK_EQUAL(y.derivative(3u), 0);
  BOOST_CHECK_EQUAL(y.derivative(4u), 0);
  y = sqrt(y);
  BOOST_CHECK_EQUAL(y.derivative(0u), sqrt(1 - cx * cx));
  BOOST_CHECK_CLOSE(y.derivative(1u), -cx / sqrt(1 - cx * cx), eps);
  BOOST_CHECK_CLOSE(y.derivative(2u), -1 / pow(1 - cx * cx, T(1.5)), eps);
  BOOST_CHECK_CLOSE(y.derivative(3u), -3 * cx / pow(1 - cx * cx, T(2.5)), eps);
  BOOST_CHECK_CLOSE(y.derivative(4u),
                    -(12 * cx * cx + 3) / pow(1 - cx * cx, T(3.5)), eps);
  y = y.inverse(); // asin'(x) = 1 / sqrt(1-x*x).
  BOOST_CHECK_CLOSE(y.derivative(0u), 1 / sqrt(1 - cx * cx), eps);
  BOOST_CHECK_CLOSE(y.derivative(1u), cx / pow(1 - cx * cx, T(1.5)), eps);
  BOOST_CHECK_CLOSE(y.derivative(2u), (2 * cx * cx + 1) / pow(1 - cx * cx, T(2.5)),
                    eps);
  BOOST_CHECK_CLOSE(y.derivative(3u),
                    3 * cx * (2 * cx * cx + 3) / pow(1 - cx * cx, T(3.5)), eps);
  BOOST_CHECK_CLOSE(y.derivative(4u),
                    (24 * (cx * cx + 3) * cx * cx + 9) / pow(1 - cx * cx, T(4.5)),
                    eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(asinh_test, T, bin_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon(); // percent
  using boost::math::asinh;
  constexpr std::size_t m = 5;
  const T cx = 1;
  auto x = make_fvar<T, m>(cx);
  auto y = asinh(x);
  BOOST_CHECK_CLOSE(y.derivative(0u), asinh(static_cast<T>(x)), eps);
  BOOST_CHECK_CLOSE(y.derivative(1u), 1 / boost::math::constants::root_two<T>(),
                    eps);
  BOOST_CHECK_CLOSE(y.derivative(2u),
                    -1 / (2 * boost::math::constants::root_two<T>()), eps);
  BOOST_CHECK_CLOSE(y.derivative(3u),
                    1 / (4 * boost::math::constants::root_two<T>()), eps);
  BOOST_CHECK_CLOSE(y.derivative(4u),
                    3 / (8 * boost::math::constants::root_two<T>()), eps);
  BOOST_CHECK_CLOSE(y.derivative(5u),
                    -39 / (16 * boost::math::constants::root_two<T>()), eps);
}

template<typename T>
static T atan2_wrap(T x, T y)
{
    return atan2(x, y);
}

static long double atan2_wrap(long double x, long double y)
{
    return std::atan2(x, y);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atan2_function, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  using std::atan2;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  test_detail::RandomSample<T> y_sampler{-2000, 2000};

  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto y = y_sampler.next();

    auto autodiff_v = atan2(make_fvar<T, m>(x), make_fvar<T, m>(y));
    auto anchor_v = atan2_wrap(x, y);
    BOOST_CHECK_CLOSE(autodiff_v, anchor_v,
                      5000 * test_constants::pct_epsilon());
  }
}

BOOST_AUTO_TEST_SUITE_END()
