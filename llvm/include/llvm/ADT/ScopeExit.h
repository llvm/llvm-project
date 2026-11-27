//===- llvm/ADT/ScopeExit.h - Execute code at scope exit --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the scope_exit class, which executes user-defined
/// cleanup logic at scope exit.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SCOPEEXIT_H
#define LLVM_ADT_SCOPEEXIT_H

#include <utility>

namespace llvm {

template <typename Callable> class [[nodiscard]] scope_exit {
  Callable ExitFunction;
  bool Engaged = true; // False once moved-from or release()d.

public:
  template <typename Fp>
  explicit scope_exit(Fp &&F) : ExitFunction(std::forward<Fp>(F)) {}

  scope_exit(scope_exit &&Rhs)
      : ExitFunction(std::move(Rhs.ExitFunction)), Engaged(Rhs.Engaged) {
    Rhs.release();
  }
  scope_exit(const scope_exit &) = delete;
  scope_exit &operator=(scope_exit &&) = delete;
  scope_exit &operator=(const scope_exit &) = delete;

  void release() { Engaged = false; }

  ~scope_exit() {
    if (Engaged)
      ExitFunction();
  }
};

template <typename Callable> scope_exit(Callable) -> scope_exit<Callable>;

} // end namespace llvm

#endif
