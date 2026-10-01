//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the mlir::NonFloatComplex type and
/// mlir::Complex type alias. The interface is intended to match the
/// std::complex type, and the mlir::Complex alias defers to std::complex for
/// builtin floating point types.
///
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_COMPLEX_H
#define MLIR_SUPPORT_COMPLEX_H

#include <complex>
#include <type_traits>

namespace mlir {

// The copy constructors should only be implicit iff the underlying constructors
// are explicit and the conversion would not narrow. This is the case if the
// underlying destination type is copy-list-initializeable from the source type,
// so define a helper to determine if that is the case.
namespace detail {
// NOLINTBEGIN
template <typename From, typename To>
auto test_copy_list_initializable(int)
    -> decltype(void(std::declval<To &>() = {std::declval<From &>()}),
                std::true_type{});

template <typename, typename>
auto test_copy_list_initializable(...) -> std::false_type;

template <typename From, typename To>
struct is_copy_list_initializable
    : std::bool_constant<
          decltype(detail::test_copy_list_initializable<From, To>(0))::value> {
};

template <typename From, typename To>
constexpr bool is_copy_list_initializable_v =
    is_copy_list_initializable<From, To>::value;
// NOLINTEND
} // namespace detail

template <typename T>
class NonFloatComplex {
public:
  using value_type = T;

private:
  T re;
  T im;

public:
  constexpr NonFloatComplex(const T &re = T{}, const T &im = T{})
      : re(re), im(im) {}

  constexpr NonFloatComplex(const NonFloatComplex &other) = default;

  template <typename U,
            std::enable_if_t<detail::is_copy_list_initializable_v<U, T>>...>
  constexpr NonFloatComplex(const NonFloatComplex<U> &other)
      : re{other.re}, im{other.im} {}

  template <typename U,
            std::enable_if_t<!detail::is_copy_list_initializable_v<U, T>>...>
  constexpr explicit NonFloatComplex(const NonFloatComplex<U> &other)
      : re(other.re), im(other.im) {}

  template <typename U,
            std::enable_if_t<detail::is_copy_list_initializable_v<U, T>>...>
  constexpr NonFloatComplex(const std::complex<U> &other)
      : re{other.real()}, im{other.imag()} {}

  template <typename U,
            std::enable_if_t<!detail::is_copy_list_initializable_v<U, T>>...>
  constexpr explicit NonFloatComplex(const std::complex<U> &other)
      : re(other.real()), im(other.imag()) {}

  [[nodiscard]] constexpr T real() const { return re; }
  constexpr void real(T value) { re = value; }
  [[nodiscard]] constexpr T imag() const { return im; }
  constexpr void imag(T value) { im = value; }

  constexpr NonFloatComplex &operator=(const NonFloatComplex &other) = default;

  constexpr NonFloatComplex &operator=(const T &real) {
    re = real;
    im = T{};
    return *this;
  }

  constexpr NonFloatComplex &operator+=(const T &real) {
    re += real;
    return *this;
  }

  constexpr NonFloatComplex &operator-=(const T &real) {
    re -= real;
    return *this;
  }

  constexpr NonFloatComplex &operator*=(const T &real) {
    re *= real;
    im *= real;
    return *this;
  }

  constexpr NonFloatComplex &operator/=(const T &real) {
    re /= real;
    im /= real;
    return *this;
  }

  constexpr NonFloatComplex &operator+=(const NonFloatComplex &other) {
    re += other.re;
    im += other.im;
    return *this;
  }

  constexpr NonFloatComplex &operator-=(const NonFloatComplex &other) {
    re -= other.re;
    im -= other.im;
    return *this;
  }

  constexpr NonFloatComplex &operator*=(const NonFloatComplex &other) {
    *this = *this * NonFloatComplex{other.re, other.im};
    return *this;
  }

  constexpr NonFloatComplex &operator/=(const NonFloatComplex &other) {
    *this = *this / NonFloatComplex{other.re, other.im};
    return *this;
  }

  template <typename U>
  constexpr NonFloatComplex &operator=(const std::complex<U> &other) {
    re = other.real();
    im = other.imag();
    return *this;
  }
};

template <typename T, typename U>
[[nodiscard]] constexpr NonFloatComplex<T>
operator+(const NonFloatComplex<T> &x, const U &y) {
  NonFloatComplex<T> t{x};
  t += y;
  return t;
}

template <typename T, typename U>
[[nodiscard]] constexpr NonFloatComplex<T>
operator-(const NonFloatComplex<T> &x, const U &y) {
  NonFloatComplex<T> t{x};
  t -= y;
  return t;
}

template <typename T>
[[nodiscard]] constexpr NonFloatComplex<T>
operator*(const NonFloatComplex<T> &x, const NonFloatComplex<T> &y) {
  T a = x.real();
  T b = x.imag();
  T c = y.real();
  T d = y.imag();

  return {(a * c) - (b * d), (a * d) + (b * c)};
}

template <typename T, typename U>
[[nodiscard]] constexpr NonFloatComplex<T>
operator*(const NonFloatComplex<T> &x, const U &y) {
  NonFloatComplex<T> t{x};
  t *= y;
  return t;
}

template <typename T>
[[nodiscard]] constexpr NonFloatComplex<T>
operator/(const NonFloatComplex<T> &x, const NonFloatComplex<T> &y) {
  T a = x.real();
  T b = x.imag();
  T c = y.real();
  T d = y.imag();

  T denom = c * c + d * d;
  return {(a * c + b * d) / denom, (b * c - a * d) / denom};
}

template <typename T, typename U>
[[nodiscard]] constexpr NonFloatComplex<T>
operator/(const NonFloatComplex<T> &x, const U &y) {
  NonFloatComplex<T> t{x};
  t /= y;
  return t;
}

template <typename T>
[[nodiscard]] constexpr NonFloatComplex<T>
operator+(const NonFloatComplex<T> &x) {
  return x;
}

template <typename T>
[[nodiscard]] constexpr NonFloatComplex<T>
operator-(const NonFloatComplex<T> &x) {
  return {-x.real(), -x.imag()};
}

template <typename T>
[[nodiscard]] constexpr bool operator==(const NonFloatComplex<T> &x,
                                        const NonFloatComplex<T> &y) {
  return x.real() == y.real() && x.imag() == y.imag();
}

template <typename T, typename U>
[[nodiscard]] constexpr bool operator==(const NonFloatComplex<T> &x,
                                        const U &y) {
  return x == NonFloatComplex<T>{y};
}

template <typename T, typename U>
[[nodiscard]] constexpr bool operator==(const T &x,
                                        const NonFloatComplex<U> &y) {
  return NonFloatComplex<U>{x} == y;
}

template <typename T>
[[nodiscard]] constexpr bool operator!=(const NonFloatComplex<T> &x,
                                        const NonFloatComplex<T> &y) {
  return !(x == y);
}

template <typename T, typename U>
[[nodiscard]] constexpr bool operator!=(const NonFloatComplex<T> &x,
                                        const U &y) {
  return !(x == y);
}

template <typename T, typename U>
[[nodiscard]] constexpr bool operator!=(const U &x,
                                        const NonFloatComplex<T> &y) {
  return !(y == x);
}

template <typename T>
[[nodiscard]] constexpr T real(const NonFloatComplex<T> &x) {
  return x.real();
}

template <typename T>
[[nodiscard]] constexpr T imag(const NonFloatComplex<T> &x) {
  return x.imag();
}

template <typename T>
using Complex = std::conditional_t<std::is_floating_point_v<T>, std::complex<T>,
                                   NonFloatComplex<T>>;
} // namespace mlir

#endif
