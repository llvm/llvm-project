//===- STLForwardCompat.h - Library features from future STLs ------C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains library features backported from future STL versions.
///
/// These should be replaced with their STL counterparts as the C++ version LLVM
/// is compiled with is updated.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STLFORWARDCOMPAT_H
#define LLVM_ADT_STLFORWARDCOMPAT_H

#include <array>
#include <optional>
#include <type_traits>

namespace llvm {

//===----------------------------------------------------------------------===//
//     Features from C++20
//===----------------------------------------------------------------------===//

template <typename T>
struct remove_cvref // NOLINT(readability-identifier-naming)
{
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <typename T>
using remove_cvref_t // NOLINT(readability-identifier-naming)
    = typename llvm::remove_cvref<T>::type;

template <typename T>
struct type_identity // NOLINT(readability-identifier-naming)
{
  using type = T;
};

//===----------------------------------------------------------------------===//
//     Features from C++17
//===----------------------------------------------------------------------===//

template <typename T>
struct negation // NOLINT(readability-identifier-naming)
    : std::integral_constant<bool, !bool(T::value)> {};

template <typename...>
struct conjunction // NOLINT(readability-identifier-naming)
    : std::true_type {};
template <typename B1> struct conjunction<B1> : B1 {};
template <typename B1, typename... Bn>
struct conjunction<B1, Bn...>
    : std::conditional<bool(B1::value), conjunction<Bn...>, B1>::type {};

template <typename...>
struct disjunction // NOLINT(readability-identifier-naming)
    : std::false_type {};
template <typename B1> struct disjunction<B1> : B1 {};
template <typename B1, typename... Bn>
struct disjunction<B1, Bn...>
    : std::conditional<bool(B1::value), B1, disjunction<Bn...>>::type {};

struct in_place_t // NOLINT(readability-identifier-naming)
{
  explicit in_place_t() = default;
};
/// \warning This must not be odr-used, as it cannot be made \c inline in C++14.
constexpr in_place_t in_place; // NOLINT(readability-identifier-naming)

template <typename T>
struct in_place_type_t // NOLINT(readability-identifier-naming)
{
  explicit in_place_type_t() = default;
};
/// \warning This must not be odr-used, as it cannot be made \c inline in C++14.
template <typename T>
constexpr in_place_type_t<T>
    in_place_type; // NOLINT(readability-identifier-naming)

template <std::size_t I>
struct in_place_index_t // NOLINT(readability-identifier-naming)
{
  explicit in_place_index_t() = default;
};
/// \warning This must not be odr-used, as it cannot be made \c inline in C++14.
template <std::size_t I>
constexpr in_place_index_t<I>
    in_place_index; // NOLINT(readability-identifier-naming)

/// Implementation of std::experimental::make_array. Should be deleted in favor
/// of the C++17 deduction guide for std::array.
template <typename ExplicitT = void, typename... ArgTs>
static constexpr auto
make_array(ArgTs &&...Args) // NOLINT(readability-identifier-naming)
    -> std::array<typename std::conditional_t<std::is_same<ExplicitT, void>::value,
                                              std::common_type<ArgTs...>,
                                              type_identity<ExplicitT>>::type,
                  sizeof...(ArgTs)> {
  return {std::forward<ArgTs>(Args)...};
}

//===----------------------------------------------------------------------===//
//     Features from C++23
//===----------------------------------------------------------------------===//

// TODO: Remove this in favor of std::optional<T>::transform once we switch to
// C++23.
template <typename T, typename Function>
auto transformOptional(const std::optional<T> &O, const Function &F)
    -> std::optional<decltype(F(*O))> {
  if (O)
    return F(*O);
  return std::nullopt;
}

// TODO: Remove this in favor of std::optional<T>::transform once we switch to
// C++23.
template <typename T, typename Function>
auto transformOptional(std::optional<T> &&O, const Function &F)
    -> std::optional<decltype(F(*std::move(O)))> {
  if (O)
    return F(*std::move(O));
  return std::nullopt;
}

/// Returns underlying integer value of an enum. Backport of C++23
/// std::to_underlying.
template <typename Enum>
[[nodiscard]] constexpr std::underlying_type_t<Enum> to_underlying(Enum E) {
  return static_cast<std::underlying_type_t<Enum>>(E);
}

} // namespace llvm

#endif // LLVM_ADT_STLFORWARDCOMPAT_H
