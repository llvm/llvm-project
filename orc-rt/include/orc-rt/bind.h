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

#ifndef ORC_RT_BIND_H
#define ORC_RT_BIND_H

#include <tuple>
#include <type_traits>

namespace orc_rt {
namespace detail {

template <typename Fn, typename... BoundArgTs> class BoundFn {
private:
  template <size_t... Is, typename... ArgTs>
  auto callExpandingBound(std::index_sequence<Is...>, ArgTs &&...Args) {
    return F(std::get<Is>(BoundArgs)..., std::forward<ArgTs>(Args)...);
  }

public:
  BoundFn(Fn &&F, BoundArgTs &&...BoundArgs)
      : F(std::move(F)), BoundArgs(std::forward<BoundArgTs>(BoundArgs)...) {}

  template <typename... ArgTs> auto operator()(ArgTs &&...Args) {
    return callExpandingBound(std::index_sequence_for<BoundArgTs...>(),
                              std::forward<ArgTs>(Args)...);
  }

private:
  std::decay_t<Fn> F;
  std::tuple<std::decay_t<BoundArgTs>...> BoundArgs;
};

} // namespace detail

template <typename Fn, typename... BoundArgTs>
detail::BoundFn<Fn, BoundArgTs...> bind_front(Fn &&F,
                                              BoundArgTs &&...BoundArgs) {
  return detail::BoundFn<Fn, BoundArgTs...>(
      std::forward<Fn>(F), std::forward<BoundArgTs>(BoundArgs)...);
}

} // namespace orc_rt

#endif // ORC_RT_BIND_H
