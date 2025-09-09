//===---------- ScopeExit.h - Execute code at scope exit --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// make_scope_exit and related APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SCOPEEXIT_H
#define ORC_RT_SCOPEEXIT_H

#include <type_traits>
#include <utility>

namespace orc_rt {
namespace detail {

template <typename Fn> class ScopeExitRunner {
public:
  ScopeExitRunner(Fn &&F) : F(F) {}
  ScopeExitRunner(const ScopeExitRunner &) = delete;
  ScopeExitRunner &operator=(const ScopeExitRunner &) = delete;
  ScopeExitRunner(ScopeExitRunner &&) = delete;
  ScopeExitRunner &operator=(ScopeExitRunner &&) = delete;
  ~ScopeExitRunner() {
    if (Engaged)
      F();
  }
  void release() { Engaged = false; }

private:
  Fn F;
  bool Engaged = true;
};

} // namespace detail

/// Creates an object that runs the given function object upon destruction.
/// Calling the object's release method prior to destruction will prevent the
/// function object from running.
template <typename Fn>
[[nodiscard]] detail::ScopeExitRunner<std::decay_t<Fn>>
make_scope_exit(Fn &&F) {
  return detail::ScopeExitRunner<std::decay_t<Fn>>(std::forward<Fn>(F));
}

} // namespace orc_rt

#endif // ORC_RT_SCOPEEXIT_H
