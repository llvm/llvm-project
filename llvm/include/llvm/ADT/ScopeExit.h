//===- llvm/ADT/ScopeExit.h - Execute code at scope exit --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the make_scope_exit function, which executes user-defined
// cleanup logic at scope exit.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SCOPE_EXIT_H
#define LLVM_ADT_SCOPE_EXIT_H

#include "llvm/Support/Compiler.h"

#include <type_traits>
#include <utility>

namespace llvm {
namespace detail {

template <typename Callable> class scope_exit {
  Callable ExitFunction;

public:
  template <typename Fp>
  explicit scope_exit(Fp &&F) : ExitFunction(std::forward<Fp>(F)) {}

  scope_exit(scope_exit &&Rhs) : ExitFunction(std::move(Rhs.ExitFunction)) {}

  ~scope_exit() { ExitFunction(); }
};

} // end namespace detail

// Keeps the callable object that is passed in, and execute it at the
// destruction of the returned object (usually at the scope exit where the
// returned object is kept).
//
// Interface is specified by p0052r2.
template <typename Callable>
detail::scope_exit<typename std::decay<Callable>::type>
    LLVM_ATTRIBUTE_UNUSED_RESULT make_scope_exit(Callable &&F) {
  return detail::scope_exit<typename std::decay<Callable>::type>(
      std::forward<Callable>(F));
}

} // end namespace llvm

#endif
