//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include "test_autodiff.hpp"

BOOST_AUTO_TEST_SUITE(test_autodiff_1)

BOOST_AUTO_TEST_CASE_TEMPLATE(constructors, T, all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  // Verify value-initialized instance has all 0 entries.
  const autodiff_fvar<T, m> empty1 = autodiff_fvar<T, m>();
  for (auto i : boost::irange(m + 1)) {
    BOOST_CHECK_EQUAL(empty1.derivative(i), 0);
  }
  const auto empty2 = autodiff_fvar<T, m, n>();
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      BOOST_CHECK_EQUAL(empty2.derivative(i, j), 0);
    }
  }
  // Single variable
  const T cx = 10;
  const auto x = make_fvar<T, m>(cx);
  for (auto i : boost::irange(m + 1)) {
    if (i == 0u) {
      BOOST_CHECK_EQUAL(x.derivative(i), cx);
    } else if (i == 1) {
      BOOST_CHECK_EQUAL(x.derivative(i), 1);
    } else {
      BOOST_CHECK_EQUAL(x.derivative(i), 0);
    }
  }
  const autodiff_fvar<T, n> xn = x;
  for (auto i : boost::irange(n + 1)) {
    if (i == 0) {
      BOOST_CHECK_EQUAL(xn.derivative(i), cx);
    } else if (i == 1) {
      BOOST_CHECK_EQUAL(xn.derivative(i), 1);
    } else {
      BOOST_CHECK_EQUAL(xn.derivative(i), 0);
    }
  }
  // Second independent variable
  const T cy = 100;
  const auto y = make_fvar<T, m, n>(cy);
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(y.derivative(i, j), cy);
      } else if (i == 0 && j == 1) {
        BOOST_CHECK_EQUAL(y.derivative(i, j), static_cast<T>(1.0));
      } else {
        BOOST_CHECK_EQUAL(y.derivative(i, j), static_cast<T>(0.0));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(implicit_constructors, T, all_float_types) {
  constexpr std::size_t m = 3;
  const autodiff_fvar<T, m> x = 3;
  const autodiff_fvar<T, m> one = uncast_return(x);
  const autodiff_fvar<T, m> two_and_a_half = 2.5;
  BOOST_CHECK_EQUAL(static_cast<T>(x), static_cast<T>(3.0));
  BOOST_CHECK_EQUAL(static_cast<T>(one), static_cast<T>(1.0));
  BOOST_CHECK_EQUAL(static_cast<T>(two_and_a_half), static_cast<T>(2.5));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(assignment, T, all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 10;
  const T cy = 10;
  autodiff_fvar<T, m, n>
      empty; // Uninitialized variable<> may have non-zero values.
  // Single variable
  auto x = make_fvar<T, m>(cx);
  empty = static_cast<decltype(empty)>(
      x); // Test static_cast of single-variable to double-variable type.
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(empty.derivative(i, j), cx);
      } else if (i == 1 && j == 0) {
        BOOST_CHECK_EQUAL(empty.derivative(i, j), 1.0);
      } else {
        BOOST_CHECK_EQUAL(empty.derivative(i, j), 0.0);
      }
    }
  }
  auto y = make_fvar<T, m, n>(cy);
  empty = y; // default assignment operator
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(empty.derivative(i, j), cy);
      } else if (i == 0 && j == 1) {
        BOOST_CHECK_EQUAL(empty.derivative(i, j), 1.0);
      } else {
        BOOST_CHECK_EQUAL(empty.derivative(i, j), 0.0);
      }
    }
  }
  empty = cx; // set a constant
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(empty.derivative(i, j), cx);
      } else {
        BOOST_CHECK_EQUAL(empty.derivative(i, j), 0.0);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ostream, T, all_float_types) {
  constexpr std::size_t m = 3;
  const T cx = 10;
  const auto x = make_fvar<T, m>(cx);
  std::ostringstream ss;
  ss << "x = " << x;
  BOOST_CHECK_EQUAL(ss.str(), "x = depth(1)(10,1,0,0)");
  ss.str(std::string());
  const auto scalar = make_fvar<T,0>(cx);
  ss << "scalar = " << scalar;
  BOOST_CHECK_EQUAL(ss.str(), "scalar = depth(1)(10)");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(addition_assignment, T, all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 10;
  auto sum = autodiff_fvar<T, m, n>(); // zero-initialized
  // Single variable
  const auto x = make_fvar<T, m>(cx);
  sum += x;
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(sum.derivative(i, j), cx);
      } else if (i == 1 && j == 0) {
        BOOST_CHECK_EQUAL(sum.derivative(i, j), T(1));
      } else {
        BOOST_CHECK_EQUAL(sum.derivative(i, j), T(0));
      }
    }
  }
  // Arithmetic constant
  const T cy = 11;
  sum = 0;
  sum += cy;
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(sum.derivative(i, j), cy);
      } else {
        BOOST_CHECK_EQUAL(sum.derivative(i, j), T(0));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(subtraction_assignment, T, all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 10;
  auto sum = autodiff_fvar<T, m, n>(); // zero-initialized
  // Single variable
  const auto x = make_fvar<T, m>(cx);
  sum -= x;
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(sum.derivative(i, j), -cx);
      } else if (i == 1 && j == 0) {
        BOOST_CHECK_EQUAL(sum.derivative(i, j), T(-1));
      } else {
        BOOST_CHECK_EQUAL(sum.derivative(i, j), T(0));
      }
    }
  }
  // Arithmetic constant
  const T cy = 11;
  sum = 0;
  sum -= cy;
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(sum.derivative(i, j), -cy);
      } else {
        BOOST_CHECK_EQUAL(sum.derivative(i, j), T(0));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(multiplication_assignment, T, all_float_types) {
  // Try explicit bracing based on feedback. Doesn't add very much except 26
  // extra lines.
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 10;
  auto product = autodiff_fvar<T, m, n>(1); // unit constant
  // Single variable
  auto x = make_fvar<T, m>(cx);
  product *= x;
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(product.derivative(i, j), cx);
      } else if (i == 1 && j == 0) {
        BOOST_CHECK_EQUAL(product.derivative(i, j), T(1));
      } else {
        BOOST_CHECK_EQUAL(product.derivative(i, j), T(0));
      }
    }
  }
  // Arithmetic constant
  const T cy = 11;
  product = 1;
  product *= cy;
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(product.derivative(i, j), cy);
      } else {
        BOOST_CHECK_EQUAL(product.derivative(i, j), T(0));
      }
    }
  }
  // 0 * inf = nan
  x = make_fvar<T, m>(T(0.0));
  x *= std::numeric_limits<T>::infinity();
  // std::cout << "x = " << x << std::endl;
  for (auto i : boost::irange(m + 1)) {
    if (i == 0) {
      BOOST_CHECK(boost::math::isnan(static_cast<T>(x))); // Correct
      // BOOST_CHECK_EQUAL(x.derivative(i) == 0.0); // Wrong. See
      // multiply_assign_by_root_type().
    } else if (i == 1) {
      BOOST_CHECK(boost::math::isinf(x.derivative(i)));
    } else {
      BOOST_CHECK_EQUAL(x.derivative(i), 0.0);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(division_assignment, T, all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 16;
  auto quotient = autodiff_fvar<T, m, n>(1); // unit constant
  // Single variable
  const auto x = make_fvar<T, m>(cx);
  quotient /= x;
  BOOST_CHECK_EQUAL(quotient.derivative(0, 0), 1 / cx);
  BOOST_CHECK_EQUAL(quotient.derivative(1, 0), -1 / pow(cx, 2));
  BOOST_CHECK_EQUAL(quotient.derivative(2, 0), 2 / pow(cx, 3));
  BOOST_CHECK_EQUAL(quotient.derivative(3, 0), -6 / pow(cx, 4));
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(std::size_t(1), n + 1)) {
      BOOST_CHECK_EQUAL(quotient.derivative(i, j), T(0));
    }
  }
  // Arithmetic constant
  const T cy = 32;
  quotient = 1;
  quotient /= cy;
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(quotient.derivative(i, j), 1 / cy);
      } else {
        BOOST_CHECK_EQUAL(quotient.derivative(i, j), T(0));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(unary_signs, T, all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 16;
  autodiff_fvar<T, m, n> lhs;
  // Single variable
  const auto x = make_fvar<T, m>(cx);
  lhs = static_cast<decltype(lhs)>(-x);
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(lhs.derivative(i, j), -cx);
      } else if (i == 1 && j == 0) {
        BOOST_CHECK_EQUAL(lhs.derivative(i, j), T(-1));
      } else {
        BOOST_CHECK_EQUAL(lhs.derivative(i, j), T(0));
      }
    }
  }
  lhs = static_cast<decltype(lhs)>(+x);
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      if (i == 0 && j == 0) {
        BOOST_CHECK_EQUAL(lhs.derivative(i, j), cx);
      } else if (i == 1 && j == 0) {
        BOOST_CHECK_EQUAL(lhs.derivative(i, j), T(1));
      } else {
        BOOST_CHECK_EQUAL(lhs.derivative(i, j), T(0));
      }
    }
  }
}

// TODO 3 tests for 3 operator+() definitions.
BOOST_AUTO_TEST_CASE_TEMPLATE(cast_double, T, all_float_types) {
  const T ca(13);
  const T i(12);
  constexpr std::size_t m = 3;
  const auto x = make_fvar<T, m>(ca);
  BOOST_CHECK_LT(i, x);
  BOOST_CHECK_EQUAL(i * x, i * ca);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(int_double_casting, T, all_float_types) {
  const T ca = 3;
  const auto x0 = make_fvar<T, 0>(ca);
  BOOST_CHECK_EQUAL(static_cast<T>(x0), ca);
  const auto x1 = make_fvar<T, 1>(ca);
  BOOST_CHECK_EQUAL(static_cast<T>(x1), ca);
  const auto x2 = make_fvar<T, 2>(ca);
  BOOST_CHECK_EQUAL(static_cast<T>(x2), ca);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scalar_addition, T, all_float_types) {
  const T ca = 3;
  const T cb = 4;
  const auto sum0 = autodiff_fvar<T, 0>(ca) + autodiff_fvar<T, 0>(cb);
  BOOST_CHECK_EQUAL(ca + cb, static_cast<T>(sum0));
  const auto sum1 = autodiff_fvar<T, 0>(ca) + cb;
  BOOST_CHECK_EQUAL(ca + cb, static_cast<T>(sum1));
  const auto sum2 = ca + autodiff_fvar<T, 0>(cb);
  BOOST_CHECK_EQUAL(ca + cb, static_cast<T>(sum2));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(power8, T, all_float_types) {
  constexpr std::size_t n = 8u;
  const T ca = 3;
  auto x = make_fvar<T, n>(ca);
  // Test operator*=()
  x *= x;
  x *= x;
  x *= x;
  const T power_factorial = boost::math::factorial<T>(n);
  for (auto i : boost::irange(n + 1)) {
    BOOST_CHECK_CLOSE(
        static_cast<T>(x.derivative(i)),
        static_cast<T>(power_factorial /
                       boost::math::factorial<T>(static_cast<unsigned>(n - i)) *
                       pow(ca, n - i)),
        std::numeric_limits<T>::epsilon());
  }
  x = make_fvar<T, n>(ca);
  // Test operator*()
  x = x * x * x * x * x * x * x * x;
  for (auto i : boost::irange(n + 1)) {
    BOOST_CHECK_CLOSE(
        static_cast<T>(x.derivative(i)),
        static_cast<T>(power_factorial /
            boost::math::factorial<T>(static_cast<unsigned>(n - i)) *
            pow(ca, n - i)),
        std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dim1_multiplication, T, all_float_types) {
  constexpr std::size_t m = 2;
  constexpr std::size_t n = 3;
  const T cy = 4;
  auto y0 = make_fvar<T, m>(cy);
  auto y = make_fvar<T, n>(cy);
  y *= y0;
  BOOST_CHECK_EQUAL(y.derivative(0), cy * cy);
  BOOST_CHECK_EQUAL(y.derivative(1), 2 * cy);
  BOOST_CHECK_EQUAL(y.derivative(2), T(2));
  BOOST_CHECK_EQUAL(y.derivative(3), T(0));
  y = y * cy;
  BOOST_CHECK_EQUAL(y.derivative(0), cy * cy * cy);
  BOOST_CHECK_EQUAL(y.derivative(1), 2 * cy * cy);
  BOOST_CHECK_EQUAL(y.derivative(2), 2.0 * cy);
  BOOST_CHECK_EQUAL(y.derivative(3), 0.0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dim1and2_multiplication, T, all_float_types) {
  constexpr std::size_t m = 2;
  constexpr std::size_t n = 3;
  const T cx = 3;
  const T cy = 4;
  auto x = make_fvar<T, m>(cx);
  auto y = make_fvar<T, m, n>(cy);
  y *= x;
  BOOST_CHECK_EQUAL(y.derivative(0, 0), cx * cy);
  BOOST_CHECK_EQUAL(y.derivative(0, 1), cx);
  BOOST_CHECK_EQUAL(y.derivative(1, 0), cy);
  BOOST_CHECK_EQUAL(y.derivative(1, 1), T(1));
  for (auto i : boost::irange(std::size_t(1), m)) {
    for (auto j : boost::irange(std::size_t(1), n)) {
      if (i == 1 && j == 1) {
        BOOST_CHECK_EQUAL(y.derivative(i, j), T(1));
      } else {
        BOOST_CHECK_EQUAL(y.derivative(i, j), T(0));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dim2_addition, T, all_float_types) {
  constexpr std::size_t m = 2;
  constexpr std::size_t n = 3;
  const T cx = 3;
  const auto x = make_fvar<T, m>(cx);
  BOOST_CHECK_EQUAL(x.derivative(0), cx);
  BOOST_CHECK_EQUAL(x.derivative(1), T(1));
  BOOST_CHECK_EQUAL(x.derivative(2), 0.0);
  const T cy = 4;
  const auto y = make_fvar<T, m, n>(cy);
  BOOST_CHECK_EQUAL(static_cast<T>(y.derivative(0)), cy);
  BOOST_CHECK_EQUAL(static_cast<T>(y.derivative(1)),
                    0.0); // partial of y w.r.t. x.

  BOOST_CHECK_EQUAL(y.derivative(0, 0), cy);
  BOOST_CHECK_EQUAL(y.derivative(0, 1), T(1));
  BOOST_CHECK_EQUAL(y.derivative(1, 0), 0.0);
  BOOST_CHECK_EQUAL(y.derivative(1, 1), 0.0);
  const auto z = x + y;
  BOOST_CHECK_EQUAL(z.derivative(0, 0), cx + cy);
  BOOST_CHECK_EQUAL(z.derivative(0, 1), T(1));
  BOOST_CHECK_EQUAL(z.derivative(1, 0), T(1));
  BOOST_CHECK_EQUAL(z.derivative(1, 1), 0.0);
  // The following 4 are unnecessarily more expensive than the previous 4.
  BOOST_CHECK_EQUAL(z.derivative(0).derivative(0), cx + cy);
  BOOST_CHECK_EQUAL(z.derivative(0).derivative(1), T(1));
  BOOST_CHECK_EQUAL(z.derivative(1).derivative(0), T(1));
  BOOST_CHECK_EQUAL(z.derivative(1).derivative(1), 0.0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dim2_multiplication, T, all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 6;
  const auto x = make_fvar<T, m>(cx);
  const T cy = 5;
  const auto y = make_fvar<T, 0, n>(cy);
  const auto z = x * x * y * y * y;
  BOOST_CHECK_EQUAL(z.derivative(0, 0), cx * cx * cy * cy * cy); // x^2 * y^3
  BOOST_CHECK_EQUAL(z.derivative(0, 1), cx * cx * 3 * cy * cy);  // x^2 * 3y^2
  BOOST_CHECK_EQUAL(z.derivative(0, 2), cx * cx * 6 * cy);       // x^2 * 6y
  BOOST_CHECK_EQUAL(z.derivative(0, 3), cx * cx * 6);            // x^2 * 6
  BOOST_CHECK_EQUAL(z.derivative(0, 4), T(0));                    // x^2 * 0
  BOOST_CHECK_EQUAL(z.derivative(1, 0), 2 * cx * cy * cy * cy);  // 2x * y^3
  BOOST_CHECK_EQUAL(z.derivative(1, 1), 2 * cx * 3 * cy * cy);   // 2x * 3y^2
  BOOST_CHECK_EQUAL(z.derivative(1, 2), 2 * cx * 6 * cy);        // 2x * 6y
  BOOST_CHECK_EQUAL(z.derivative(1, 3), 2 * cx * 6);             // 2x * 6
  BOOST_CHECK_EQUAL(z.derivative(1, 4), T(0));                    // 2x * 0
  BOOST_CHECK_EQUAL(z.derivative(2, 0), 2 * cy * cy * cy);       // 2 * y^3
  BOOST_CHECK_EQUAL(z.derivative(2, 1), 2 * 3 * cy * cy);        // 2 * 3y^2
  BOOST_CHECK_EQUAL(z.derivative(2, 2), 2 * 6 * cy);             // 2 * 6y
  BOOST_CHECK_EQUAL(z.derivative(2, 3), 2 * 6);                  // 2 * 6
  BOOST_CHECK_EQUAL(z.derivative(2, 4), T(0));                    // 2 * 0
  BOOST_CHECK_EQUAL(z.derivative(3, 0), T(0));                    // 0 * y^3
  BOOST_CHECK_EQUAL(z.derivative(3, 1), T(0));                    // 0 * 3y^2
  BOOST_CHECK_EQUAL(z.derivative(3, 2), T(0));                    // 0 * 6y
  BOOST_CHECK_EQUAL(z.derivative(3, 3), T(0));                    // 0 * 6
  BOOST_CHECK_EQUAL(z.derivative(3, 4), T(0));                    // 0 * 0
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dim2_multiplication_and_subtraction, T,
                              all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 6;
  const auto x = make_fvar<T, m>(cx);
  const T cy = 5;
  const auto y = make_fvar<T, 0, n>(cy);
  const auto z = x * x - y * y;
  BOOST_CHECK_EQUAL(z.derivative(0, 0), cx * cx - cy * cy);
  BOOST_CHECK_EQUAL(z.derivative(0, 1), -2 * cy);
  BOOST_CHECK_EQUAL(z.derivative(0, 2), T(-2));
  BOOST_CHECK_EQUAL(z.derivative(0, 3), T(0));
  BOOST_CHECK_EQUAL(z.derivative(0, 4), T(0));
  BOOST_CHECK_EQUAL(z.derivative(1, 0), 2 * cx);
  BOOST_CHECK_EQUAL(z.derivative(2, 0), T(2));
  for (auto i : boost::irange(std::size_t(1), m + 1)) {
    for (auto j : boost::irange(std::size_t(1), n + 1)) {
      BOOST_CHECK_EQUAL(z.derivative(i, j), T(0));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(inverse, T, all_float_types) {
  constexpr std::size_t m = 3;
  const T cx = 4;
  const auto x = make_fvar<T, m>(cx);
  const auto xinv = x.inverse();
  BOOST_CHECK_EQUAL(xinv.derivative(0), 1 / cx);
  BOOST_CHECK_EQUAL(xinv.derivative(1), -1 / pow(cx, 2));
  BOOST_CHECK_EQUAL(xinv.derivative(2), 2 / pow(cx, 3));
  BOOST_CHECK_EQUAL(xinv.derivative(3), -6 / pow(cx, 4));
  const auto zero = make_fvar<T, m>(0);
  const auto inf = zero.inverse();
  for (auto i : boost::irange(m + 1)) {
    BOOST_CHECK_EQUAL(inf.derivative(i),
                      (i % 2 == 1 ? -1 : 1) *
                          std::numeric_limits<T>::infinity());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(division, T, all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 16;
  auto x = make_fvar<T, m>(cx);
  const T cy = 4;
  auto y = make_fvar<T, 1, n>(cy);
  auto z = x * x / (y * y);
  BOOST_CHECK_EQUAL(z.derivative(0, 0), cx * cx / (cy * cy)); // x^2 * y^-2
  BOOST_CHECK_EQUAL(z.derivative(0, 1), cx * cx * (-2) * pow(cy, -3));
  BOOST_CHECK_EQUAL(z.derivative(0, 2), cx * cx * (6) * pow(cy, -4));
  BOOST_CHECK_EQUAL(z.derivative(0, 3), cx * cx * (-24) * pow(cy, -5));
  BOOST_CHECK_EQUAL(z.derivative(0, 4), cx * cx * (120) * pow(cy, -6));
  BOOST_CHECK_EQUAL(z.derivative(1, 0), 2 * cx / (cy * cy));
  BOOST_CHECK_EQUAL(z.derivative(1, 1), 2 * cx * (-2) * pow(cy, -3));
  BOOST_CHECK_EQUAL(z.derivative(1, 2), 2 * cx * (6) * pow(cy, -4));
  BOOST_CHECK_EQUAL(z.derivative(1, 3), 2 * cx * (-24) * pow(cy, -5));
  BOOST_CHECK_EQUAL(z.derivative(1, 4), 2 * cx * (120) * pow(cy, -6));
  BOOST_CHECK_EQUAL(z.derivative(2, 0), 2 / (cy * cy));
  BOOST_CHECK_EQUAL(z.derivative(2, 1), 2 * (-2) * pow(cy, -3));
  BOOST_CHECK_EQUAL(z.derivative(2, 2), 2 * (6) * pow(cy, -4));
  BOOST_CHECK_EQUAL(z.derivative(2, 3), 2 * (-24) * pow(cy, -5));
  BOOST_CHECK_EQUAL(z.derivative(2, 4), 2 * (120) * pow(cy, -6));
  for (auto j : boost::irange(n + 1)) {
    BOOST_CHECK_EQUAL(z.derivative(3, j), 0.0);
  }

  auto x1 = make_fvar<T, m>(cx);
  auto z1 = x1 / cy;
  BOOST_CHECK_EQUAL(z1.derivative(0), cx / cy);
  BOOST_CHECK_EQUAL(z1.derivative(1), 1 / cy);
  BOOST_CHECK_EQUAL(z1.derivative(2), 0.0);
  BOOST_CHECK_EQUAL(z1.derivative(3), 0.0);
  auto y2 = make_fvar<T, m, n>(cy);
  auto z2 = cx / y2;
  BOOST_CHECK_EQUAL(z2.derivative(0, 0), cx / cy);
  BOOST_CHECK_EQUAL(z2.derivative(0, 1), -cx / pow(cy, 2));
  BOOST_CHECK_EQUAL(z2.derivative(0, 2), 2 * cx / pow(cy, 3));
  BOOST_CHECK_EQUAL(z2.derivative(0, 3), -6 * cx / pow(cy, 4));
  BOOST_CHECK_EQUAL(z2.derivative(0, 4), 24 * cx / pow(cy, 5));
  for (auto i : boost::irange(std::size_t(1), m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      BOOST_CHECK_EQUAL(z2.derivative(i, j), 0.0);
    }
  }

  const auto z3 = y / x;
  BOOST_CHECK_EQUAL(z3.derivative(0, 0), cy / cx);
  BOOST_CHECK_EQUAL(z3.derivative(0, 1), 1 / cx);
  BOOST_CHECK_EQUAL(z3.derivative(1, 0), -cy / pow(cx, 2));
  BOOST_CHECK_EQUAL(z3.derivative(1, 1), -1 / pow(cx, 2));
  BOOST_CHECK_EQUAL(z3.derivative(2, 0), 2 * cy / pow(cx, 3));
  BOOST_CHECK_EQUAL(z3.derivative(2, 1), 2 / pow(cx, 3));
  BOOST_CHECK_EQUAL(z3.derivative(3, 0), -6 * cy / pow(cx, 4));
  BOOST_CHECK_EQUAL(z3.derivative(3, 1), -6 / pow(cx, 4));
  for (auto i : boost::irange(m + 1)) {
    for (auto j : boost::irange(std::size_t(2), n + 1)) {
      BOOST_CHECK_EQUAL(z3.derivative(i, j), 0.0);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(equality, T, all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 10;
  const T cy = 10;
  const auto x = make_fvar<T, m>(cx);
  const auto y = make_fvar<T, 0, n>(cy);
  BOOST_CHECK_EQUAL(x, y);
  BOOST_CHECK_EQUAL(x, cy);
  BOOST_CHECK_EQUAL(cx, y);
  BOOST_CHECK_EQUAL(cy, x);
  BOOST_CHECK_EQUAL(y, cx);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(inequality, T, all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 10;
  const T cy = 11;
  const auto x = make_fvar<T, m>(cx);
  const auto y = make_fvar<T, 0, n>(cy);
  BOOST_CHECK_NE(x, y);
  BOOST_CHECK_NE(x, cy);
  BOOST_CHECK_NE(cx, y);
  BOOST_CHECK_NE(cy, x);
  BOOST_CHECK_NE(y, cx);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(less_than_or_equal_to, T, all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 10;
  const T cy = 11;
  const auto x = make_fvar<T, m>(cx);
  const auto y = make_fvar<T, 0, n>(cy);
  BOOST_CHECK_LE(x, y);
  BOOST_CHECK_LE(x, y - 1);
  BOOST_CHECK_LT(x, y);
  BOOST_CHECK_LE(x, cy);
  BOOST_CHECK_LE(x, cy - 1);
  BOOST_CHECK_LT(x, cy);
  BOOST_CHECK_LE(cx, y);
  BOOST_CHECK_LE(cx, y - 1);
  BOOST_CHECK_LT(cx, y);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(greater_than_or_equal_to, T, all_float_types) {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  const T cx = 11;
  const T cy = 10;
  const auto x = make_fvar<T, m>(cx);
  const auto y = make_fvar<T, 0, n>(cy);
  BOOST_CHECK_GE(x, y);
  BOOST_CHECK_GE(x, y + 1);
  BOOST_CHECK_GT(x, y);
  BOOST_CHECK_GE(x, cy);
  BOOST_CHECK_GE(x, cy + 1);
  BOOST_CHECK_GT(x, cy);
  BOOST_CHECK_GE(cx, y);
  BOOST_CHECK_GE(cx, y + 1);
  BOOST_CHECK_GT(cx, y);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fabs_test, T, all_float_types) {
  using bmp::fabs;
  using detail::fabs;
  using std::fabs;
  constexpr std::size_t m = 3;
  const T cx = 11;
  const auto x = make_fvar<T, m>(cx);
  auto a = fabs(x);
  BOOST_CHECK_EQUAL(a.derivative(0), fabs(cx));
  BOOST_CHECK_EQUAL(a.derivative(1), T(1));
  BOOST_CHECK_EQUAL(a.derivative(2), T(0));
  BOOST_CHECK_EQUAL(a.derivative(3), T(0));
  a = fabs(-x);
  BOOST_CHECK_EQUAL(a.derivative(0), fabs(cx));
  BOOST_CHECK_EQUAL(a.derivative(1), T(1)); // fabs(-x) = fabs(x)
  BOOST_CHECK_EQUAL(a.derivative(2), T(0));
  BOOST_CHECK_EQUAL(a.derivative(3), T(0));
  const auto xneg = make_fvar<T, m>(-cx);
  a = fabs(xneg);
  BOOST_CHECK_EQUAL(a.derivative(0), fabs(cx));
  BOOST_CHECK_EQUAL(a.derivative(1), -T(1));
  BOOST_CHECK_EQUAL(a.derivative(2), T(0));
  BOOST_CHECK_EQUAL(a.derivative(3), T(0));
  const auto zero = make_fvar<T, m>(0);
  a = fabs(zero);
  for (auto i : boost::irange(m + 1)) {
    BOOST_CHECK_EQUAL(a.derivative(i), T(0));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ceil_and_floor, T, all_float_types) {
  using bmp::ceil;
  using bmp::floor;
  using std::ceil;
  using std::floor;
  constexpr std::size_t m = 3;
  T tests[]{T(-1.5), T(0.0), T(1.5)};
  for (T &test : tests) {
    const auto x = make_fvar<T, m>(test);
    auto c = ceil(x);
    auto f = floor(x);
    BOOST_CHECK_EQUAL(c.derivative(0), ceil(test));
    BOOST_CHECK_EQUAL(f.derivative(0), floor(test));
    for (auto i : boost::irange(std::size_t(1), m + 1)) {
      BOOST_CHECK_EQUAL(c.derivative(i), T(0));
      BOOST_CHECK_EQUAL(f.derivative(i), T(0));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
