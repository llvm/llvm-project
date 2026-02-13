//===--- AlwaysTrue.h - Helper for oqaque truthy values        --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ALWAYS_TRUE_H
#define LLVM_SUPPORT_ALWAYS_TRUE_H

#include <cstdlib>

namespace llvm {
inline bool getNonFoldableAlwaysTrue() {
  // Some parts of the codebase require a "constant true value" used as a
  // predicate. These cases require that even with LTO and static linking,
  // it's not possible for the compiler to fold the value. As compilers
  // aren't smart enough to know that getenv() never returns -1, this will do
  // the job.
  return std::getenv("LLVM_IGNORED_ENV_VAR") != (char *)-1;
}
} // end namespace llvm

#endif // LLVM_SUPPORT_ALWAYS_TRUE_H
