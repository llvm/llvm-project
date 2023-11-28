//===- Utils.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_UTILS_H
#define LLVM_LIB_DWARFLINKERPARALLEL_UTILS_H

#include "llvm/Support/Error.h"

namespace llvm {
namespace dwarflinker_parallel {

/// This function calls \p Iteration() until it returns false.
/// If number of iterations exceeds \p MaxCounter then an Error is returned.
/// This function should be used for loops which assumed to have number of
/// iterations significantly smaller than \p MaxCounter to avoid infinite
/// looping in error cases.
inline Error finiteLoop(function_ref<Expected<bool>()> Iteration,
                        size_t MaxCounter = 100000) {
  size_t iterationsCounter = 0;
  while (iterationsCounter++ < MaxCounter) {
    Expected<bool> IterationResultOrError = Iteration();
    if (!IterationResultOrError)
      return IterationResultOrError.takeError();

    if (!IterationResultOrError.get())
      return Error::success();
  }

  return createStringError(std::errc::invalid_argument, "Infinite recursion");
}

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_UTILS_H
