//===---------- scope_exit.h - Execute code at scope exit -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// scope_exit and related APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SCOPE_EXIT_H
#define ORC_RT_SCOPE_EXIT_H

#include <utility>

namespace orc_rt {

template <typename Fn> class [[nodiscard]] scope_exit {
public:
  template <typename FnInit>
  scope_exit(FnInit &&F) : F(std::forward<FnInit>(F)) {}
  scope_exit(const scope_exit &) = delete;
  scope_exit &operator=(const scope_exit &) = delete;
  scope_exit(scope_exit &&) = delete;
  scope_exit &operator=(scope_exit &&) = delete;
  ~scope_exit() {
    if (Engaged)
      F();
  }
  void release() { Engaged = false; }

private:
  Fn F;
  bool Engaged = true;
};

template <typename Fn> scope_exit(Fn) -> scope_exit<Fn>;

} // namespace orc_rt

#endif // ORC_RT_SCOPE_EXIT_H
