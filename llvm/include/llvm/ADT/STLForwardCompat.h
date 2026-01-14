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

#include "llvm/Support/Compiler.h"
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace llvm {

//===----------------------------------------------------------------------===//
//     Features from C++20
//===----------------------------------------------------------------------===//

namespace numbers {
// clang-format off
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T e_v          = T(0x1.5bf0a8b145769P+1); // (2.7182818284590452354) https://oeis.org/A001113
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T egamma_v     = T(0x1.2788cfc6fb619P-1); // (.57721566490153286061) https://oeis.org/A001620
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T ln2_v        = T(0x1.62e42fefa39efP-1); // (.69314718055994530942) https://oeis.org/A002162
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T ln10_v       = T(0x1.26bb1bbb55516P+1); // (2.3025850929940456840) https://oeis.org/A002392
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T log2e_v      = T(0x1.71547652b82feP+0); // (1.4426950408889634074)
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T log10e_v     = T(0x1.bcb7b1526e50eP-2); // (.43429448190325182765)
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T pi_v         = T(0x1.921fb54442d18P+1); // (3.1415926535897932385) https://oeis.org/A000796
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T inv_pi_v     = T(0x1.45f306dc9c883P-2); // (.31830988618379067154) https://oeis.org/A049541
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T inv_sqrtpi_v = T(0x1.20dd750429b6dP-1); // (.56418958354775628695) https://oeis.org/A087197
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T sqrt2_v      = T(0x1.6a09e667f3bcdP+0); // (1.4142135623730950488) https://oeis.org/A00219
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T inv_sqrt2_v  = T(0x1.6a09e667f3bcdP-1); // (.70710678118654752440)
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T sqrt3_v      = T(0x1.bb67ae8584caaP+0); // (1.7320508075688772935) https://oeis.org/A002194
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T inv_sqrt3_v  = T(0x1.279a74590331cP-1); // (.57735026918962576451)
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr T phi_v        = T(0x1.9e3779b97f4a8P+0); // (1.6180339887498948482) https://oeis.org/A001622

inline constexpr double e          = e_v<double>;
inline constexpr double egamma     = egamma_v<double>;
inline constexpr double ln2        = ln2_v<double>;
inline constexpr double ln10       = ln10_v<double>;
inline constexpr double log2e      = log2e_v<double>;
inline constexpr double log10e     = log10e_v<double>;
inline constexpr double pi         = pi_v<double>;
inline constexpr double inv_pi     = inv_pi_v<double>;
inline constexpr double inv_sqrtpi = inv_sqrtpi_v<double>;
inline constexpr double sqrt2      = sqrt2_v<double>;
inline constexpr double inv_sqrt2  = inv_sqrt2_v<double>;
inline constexpr double sqrt3      = sqrt3_v<double>;
inline constexpr double inv_sqrt3  = inv_sqrt3_v<double>;
inline constexpr double phi        = phi_v<double>;
// clang-format on
} // namespace numbers

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
///
/// NOTE: The C++20 standard has adopted concepts and requires clauses as a
/// superior alternative to std::is_detected.
///
/// This utility is placed in STLForwardCompat.h as a reminder
/// to migrate usages of llvm::is_detected to concepts and 'requires'
/// clauses when the codebase adopts C++20.
template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<void, Op, Args...>::value_t;

struct identity // NOLINT(readability-identifier-naming)
{
  using is_transparent = void;

  template <typename T> constexpr T &&operator()(T &&self) const noexcept {
    return std::forward<T>(self);
  }
};

/// Returns a raw pointer that represents the same address as the argument.
///
/// This implementation can be removed once we move to C++20 where it's defined
/// as std::to_address().
///
/// The std::pointer_traits<>::to_address(p) variations of these overloads has
/// not been implemented.
template <class Ptr> auto to_address(const Ptr &P) { return P.operator->(); }
template <class T> constexpr T *to_address(T *P) {
  static_assert(!std::is_function_v<T>);
  return P;
}

/// C++20 constexpr invoke. This uses `std::apply` (constexpr in C++17) to
/// achieve constexpr invocation.
template <typename FnT, typename... ArgsT>
constexpr std::invoke_result_t<FnT, ArgsT...>
invoke(FnT &&Fn, ArgsT &&...Args) { // NOLINT(readability-identifier-naming)
  return std::apply(std::forward<FnT>(Fn),
                    std::forward_as_tuple(std::forward<ArgsT>(Args)...));
}

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

//===----------------------------------------------------------------------===//
//     Bind functions from C++20 / C++23 / C++26
//===----------------------------------------------------------------------===//

namespace detail {
// Tag for constructing with a runtime callable.
struct RuntimeFnTag {};
// Tag for constructing with a compile-time constant callable.
struct ConstantFnTag {};

/// Stores a callable as a data member.
template <typename FnT> struct FnHolder {
  FnT Fn;

  template <typename FnArgT>
  constexpr explicit FnHolder(FnArgT &&F) : Fn(std::forward<FnArgT>(F)) {}

  constexpr FnT &get() { return Fn; }
  constexpr const FnT &get() const { return Fn; }
};

/// Holds a compile-time constant callable (empty storage).
template <auto ConstFn> struct FnConstant {
  constexpr decltype(auto) get() const { return ConstFn; }
};

// Storage class for bind_front/bind_back that properly handles const/non-const
// qualification of the wrapper when invoking the stored callable.
// If BindFront is true, bound args are prepended; otherwise appended.
// FnStorageT is either FnHolder<FnT> (runtime) or FnConstant<ConstFn>.
template <bool BindFront, typename BoundArgsTupleT, typename FnStorageT,
          typename IndicesT>
class BindStorage;

template <bool BindFront, typename BoundArgsTupleT, typename FnStorageT,
          size_t... Indices>
class BindStorage<BindFront, BoundArgsTupleT, FnStorageT,
                  std::index_sequence<Indices...>> {
  BoundArgsTupleT BoundArgs;
  // This may be empty for const functions, hence the `no_unique_address`.
  LLVM_NO_UNIQUE_ADDRESS FnStorageT FnStorage;

public:
  // Constructor for FnHolder (runtime callable).
  template <typename FnArgT, typename... BoundArgsArgT>
  constexpr BindStorage(RuntimeFnTag, FnArgT &&F, BoundArgsArgT &&...Args)
      : BoundArgs(std::forward<BoundArgsArgT>(Args)...),
        FnStorage(std::forward<FnArgT>(F)) {}

  // Constructor for FnConstant (compile-time callable).
  template <typename... BoundArgsArgT>
  constexpr BindStorage(ConstantFnTag, BoundArgsArgT &&...Args)
      : BoundArgs(std::forward<BoundArgsArgT>(Args)...), FnStorage() {}

  template <typename... CallArgsT>
  constexpr decltype(auto) operator()(CallArgsT &&...CallArgs) {
    if constexpr (BindFront)
      return llvm::invoke(FnStorage.get(), std::get<Indices>(BoundArgs)...,
                          std::forward<CallArgsT>(CallArgs)...);
    else
      return llvm::invoke(FnStorage.get(), std::forward<CallArgsT>(CallArgs)...,
                          std::get<Indices>(BoundArgs)...);
  }

  template <typename... CallArgsT>
  constexpr decltype(auto) operator()(CallArgsT &&...CallArgs) const {
    if constexpr (BindFront)
      return llvm::invoke(FnStorage.get(), std::get<Indices>(BoundArgs)...,
                          std::forward<CallArgsT>(CallArgs)...);
    else
      return llvm::invoke(FnStorage.get(), std::forward<CallArgsT>(CallArgs)...,
                          std::get<Indices>(BoundArgs)...);
  }
};
} // end namespace detail

/// C++20 bind_front. Prepends bound arguments to the callable. All bind
/// arguments and the callable are forwarded and *stored* by value. If you would
/// like to pass by reference, use `std::ref` or `std::cref`.
template <typename FnT, typename... BindArgsT>
constexpr auto bind_front(FnT &&Fn, // NOLINT(readability-identifier-naming)
                          BindArgsT &&...BindArgs) {
  return detail::BindStorage</*BindFront=*/true,
                             std::tuple<std::decay_t<BindArgsT>...>,
                             detail::FnHolder<std::decay_t<FnT>>,
                             std::index_sequence_for<BindArgsT...>>(
      detail::RuntimeFnTag{}, std::forward<FnT>(Fn),
      std::forward<BindArgsT>(BindArgs)...);
}

/// C++26 bind_front with compile-time callable. Prepends bound arguments.
/// Bound arguments are forwarded and *stored* by value.
template <auto ConstFn, typename... BindArgsT>
constexpr auto
bind_front(BindArgsT &&...BindArgs) { // NOLINT(readability-identifier-naming)
  if constexpr (std::is_pointer_v<decltype(ConstFn)> ||
                std::is_member_pointer_v<decltype(ConstFn)>)
    static_assert(ConstFn != nullptr);

  return detail::BindStorage<
      /*BindFront=*/true, std::tuple<std::decay_t<BindArgsT>...>,
      detail::FnConstant<ConstFn>, std::index_sequence_for<BindArgsT...>>(
      detail::ConstantFnTag{}, std::forward<BindArgsT>(BindArgs)...);
}

/// C++23 bind_back. Appends bound arguments to the callable. All bind
/// arguments and the callable are forwarded and *stored* by value. If you would
/// like to pass by reference, use `std::ref` or `std::cref`.
template <typename FnT, typename... BindArgsT>
constexpr auto bind_back(FnT &&Fn, // NOLINT(readability-identifier-naming)
                         BindArgsT &&...BindArgs) {
  return detail::BindStorage</*BindFront=*/false,
                             std::tuple<std::decay_t<BindArgsT>...>,
                             detail::FnHolder<std::decay_t<FnT>>,
                             std::index_sequence_for<BindArgsT...>>(
      detail::RuntimeFnTag{}, std::forward<FnT>(Fn),
      std::forward<BindArgsT>(BindArgs)...);
}

/// C++26 bind_back with compile-time callable. Appends bound arguments.
/// Bound arguments are forwarded and *stored* by value.
template <auto ConstFn, typename... BindArgsT>
constexpr auto
bind_back(BindArgsT &&...BindArgs) { // NOLINT(readability-identifier-naming)
  if constexpr (std::is_pointer_v<decltype(ConstFn)> ||
                std::is_member_pointer_v<decltype(ConstFn)>)
    static_assert(ConstFn != nullptr);

  return detail::BindStorage<
      /*BindFront=*/false, std::tuple<std::decay_t<BindArgsT>...>,
      detail::FnConstant<ConstFn>, std::index_sequence_for<BindArgsT...>>(
      detail::ConstantFnTag{}, std::forward<BindArgsT>(BindArgs)...);
}
} // namespace llvm

#endif // LLVM_ADT_STLFORWARDCOMPAT_H
