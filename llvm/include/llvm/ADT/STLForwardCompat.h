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

// TODO: Remove this in favor of std::type_identity<T> once we switch to C++20.
template <typename T>
struct type_identity // NOLINT(readability-identifier-naming)
{
  using type = T;
};

// TODO: Remove this in favor of std::type_identity_t<T> once we switch to
// C++20.
template <typename T>
using type_identity_t // NOLINT(readability-identifier-naming)
    = typename llvm::type_identity<T>::type;

namespace detail {
template <class, template <class...> class Op, class... Args> struct detector {
  using value_t = std::false_type;
};
template <template <class...> class Op, class... Args>
struct detector<std::void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
};
} // end namespace detail

/// Detects if a given trait holds for some set of arguments 'Args'.
/// For example, the given trait could be used to detect if a given type
/// has a copy assignment operator:
///   template<class T>
///   using has_copy_assign_t = decltype(std::declval<T&>()
///                                                 = std::declval<const T&>());
///   bool fooHasCopyAssign = is_detected<has_copy_assign_t, FooClass>::value;
template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<void, Op, Args...>::value_t;

//===----------------------------------------------------------------------===//
//     Features from C++23
//===----------------------------------------------------------------------===//

// TODO: Remove this in favor of std::optional<T>::transform once we switch to
// C++23.
template <typename Optional, typename Function,
          typename Value = typename llvm::remove_cvref_t<Optional>::value_type>
std::optional<std::invoke_result_t<Function, Value>>
transformOptional(Optional &&O, Function &&F) {
  if (O) {
    return F(*std::forward<Optional>(O));
  }
  return std::nullopt;
}

/// Returns underlying integer value of an enum. Backport of C++23
/// std::to_underlying.
template <typename Enum>
[[nodiscard]] constexpr std::underlying_type_t<Enum> to_underlying(Enum E) {
  return static_cast<std::underlying_type_t<Enum>>(E);
}

// A tag for constructors accepting ranges.
struct from_range_t {
  explicit from_range_t() = default;
};
inline constexpr from_range_t from_range{};
} // namespace llvm

#endif // LLVM_ADT_STLFORWARDCOMPAT_H
