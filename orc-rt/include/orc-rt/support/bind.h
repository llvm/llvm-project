//===---- bind.h - Substitute for future STL bind_front APIs ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Substitute for STL bind* APIs that aren't available to the ORC runtime yet.
//
// TODO: Replace all uses once the respective APIs are available.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_BIND_H
#define ORC_RT_SUPPORT_BIND_H

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace orc_rt {
namespace detail {

/// Call wrapper returned by bind_front. Matches the behavior of the C++20
/// std::bind_front perfect forwarding call wrapper: the wrapper's cv and
/// value category are forwarded to the stored callable and bound arguments.
template <typename Fn, typename... BoundArgTs> class BoundFn {
private:
  template <typename Self, size_t... Is, typename... ArgTs>
  static decltype(auto) callExpandingBound(Self &&S, std::index_sequence<Is...>,
                                           ArgTs &&...Args) {
    return std::invoke(std::forward<Self>(S).F,
                       std::get<Is>(std::forward<Self>(S).BoundArgs)...,
                       std::forward<ArgTs>(Args)...);
  }

public:
  template <typename FnInit, typename... BoundArgInitTs,
            typename = std::enable_if_t<
                !std::is_same_v<std::decay_t<FnInit>, BoundFn>>>
  explicit BoundFn(FnInit &&F, BoundArgInitTs &&...BoundArgs) noexcept(
      std::is_nothrow_constructible_v<Fn, FnInit &&> &&
      (std::is_nothrow_constructible_v<BoundArgTs, BoundArgInitTs &&> && ...))
      : F(std::forward<FnInit>(F)),
        BoundArgs(std::forward<BoundArgInitTs>(BoundArgs)...) {}

  // Each call operator below is paired with a deleted overload that is
  // selected only when the call is ill-formed. This prevents fallback to a
  // differently-qualified overload (e.g. a non-const call silently using the
  // const overload), matching std::bind_front.

  template <typename... ArgTs>
  std::invoke_result_t<Fn &, BoundArgTs &..., ArgTs...>
  operator()(ArgTs &&...Args) & noexcept(
      std::is_nothrow_invocable_v<Fn &, BoundArgTs &..., ArgTs...>) {
    return callExpandingBound(*this, std::index_sequence_for<BoundArgTs...>(),
                              std::forward<ArgTs>(Args)...);
  }

  template <typename... ArgTs, typename = std::enable_if_t<!std::is_invocable_v<
                                   Fn &, BoundArgTs &..., ArgTs...>>>
  void operator()(ArgTs &&...) & = delete;

  template <typename... ArgTs>
  std::invoke_result_t<const Fn &, const BoundArgTs &..., ArgTs...>
  operator()(ArgTs &&...Args) const & noexcept(
      std::is_nothrow_invocable_v<const Fn &, const BoundArgTs &...,
                                  ArgTs...>) {
    return callExpandingBound(*this, std::index_sequence_for<BoundArgTs...>(),
                              std::forward<ArgTs>(Args)...);
  }

  template <typename... ArgTs,
            typename = std::enable_if_t<!std::is_invocable_v<
                const Fn &, const BoundArgTs &..., ArgTs...>>>
  void operator()(ArgTs &&...) const & = delete;

  template <typename... ArgTs>
  std::invoke_result_t<Fn &&, BoundArgTs &&..., ArgTs...>
  operator()(ArgTs &&...Args) && noexcept(
      std::is_nothrow_invocable_v<Fn &&, BoundArgTs &&..., ArgTs...>) {
    return callExpandingBound(std::move(*this),
                              std::index_sequence_for<BoundArgTs...>(),
                              std::forward<ArgTs>(Args)...);
  }

  template <typename... ArgTs, typename = std::enable_if_t<!std::is_invocable_v<
                                   Fn &&, BoundArgTs &&..., ArgTs...>>>
  void operator()(ArgTs &&...) && = delete;

  template <typename... ArgTs>
  std::invoke_result_t<const Fn &&, const BoundArgTs &&..., ArgTs...>
  operator()(ArgTs &&...Args) const && noexcept(
      std::is_nothrow_invocable_v<const Fn &&, const BoundArgTs &&...,
                                  ArgTs...>) {
    return callExpandingBound(std::move(*this),
                              std::index_sequence_for<BoundArgTs...>(),
                              std::forward<ArgTs>(Args)...);
  }

  template <typename... ArgTs,
            typename = std::enable_if_t<!std::is_invocable_v<
                const Fn &&, const BoundArgTs &&..., ArgTs...>>>
  void operator()(ArgTs &&...) const && = delete;

private:
  Fn F;
  std::tuple<BoundArgTs...> BoundArgs;
};

} // namespace detail

template <typename Fn, typename... BoundArgTs>
detail::BoundFn<std::decay_t<Fn>, std::decay_t<BoundArgTs>...>
bind_front(Fn &&F, BoundArgTs &&...BoundArgs) {
  static_assert(std::is_constructible_v<std::decay_t<Fn>, Fn> &&
                    std::is_move_constructible_v<std::decay_t<Fn>>,
                "bind_front requires a move-constructible callable");
  static_assert(
      ((std::is_constructible_v<std::decay_t<BoundArgTs>, BoundArgTs> &&
        std::is_move_constructible_v<std::decay_t<BoundArgTs>>) &&
       ...),
      "bind_front requires move-constructible bound arguments");
  return detail::BoundFn<std::decay_t<Fn>, std::decay_t<BoundArgTs>...>(
      std::forward<Fn>(F), std::forward<BoundArgTs>(BoundArgs)...);
}

} // namespace orc_rt

#endif // ORC_RT_SUPPORT_BIND_H
