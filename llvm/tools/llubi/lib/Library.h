//===--- Library.h - Library Function Simulator for llubi -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLUBI_LIBRARY_H
#define LLVM_TOOLS_LLUBI_LIBRARY_H

#include "Context.h"
#include "Interpreter.h"
#include "Value.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/InstrTypes.h"
#include <string>

namespace llvm::ubi {

class LibraryEnvironment {
  Context &Ctx;
  EventHandler &Handler;
  const DataLayout &DL;
  InstExecutor &Executor;

  std::optional<std::string> readStringFromMemory(const Pointer &Ptr);

  std::optional<AnyValue> executeMalloc(CallBase &CB);
  std::optional<AnyValue> executeCalloc(CallBase &CB);
  std::optional<AnyValue> executeFree(CallBase &CB);
  std::optional<AnyValue> executePuts(CallBase &CB);
  std::optional<AnyValue> executePrintf(CallBase &CB);
  std::optional<AnyValue> executeExit(CallBase &CB);
  std::optional<AnyValue> executeAbort(CallBase &CB);
  std::optional<AnyValue> executeTerminate(CallBase &CB);

public:
  LibraryEnvironment(Context &C, EventHandler &H, const DataLayout &DL,
                     InstExecutor &Executor)
      : Ctx(C), Handler(H), DL(DL), Executor(Executor) {}

  /// Simulates a standard library call. Returns std::nullopt to indicate that
  /// execution should halt (either due to an exit/abort call or an immediate
  /// UB trigger).
  std::optional<AnyValue> call(LibFunc LF, CallBase &CB);
};

} // namespace llvm::ubi

#endif