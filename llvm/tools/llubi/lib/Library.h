//===--- Library.h - Library calls for llubi ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common libcalls for llubi.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLUBI_LIBRARY_H
#define LLVM_TOOLS_LLUBI_LIBRARY_H

#include "Context.h"
#include "ExecutorBase.h"
#include "Value.h"
#include <optional>
#include <string>

namespace llvm::ubi {

class Library {
  Context &Ctx;
  EventHandler &Handler;
  const DataLayout &DL;
  ExecutorBase &Executor;

  std::optional<std::string> readStringFromMemory(const Pointer &Ptr);

  AnyValue executeMalloc(StringRef Name, Type *Type, ArrayRef<AnyValue> Args,
                         MemAllocKind AllocKind);
  AnyValue executeCalloc(StringRef Name, Type *Type, ArrayRef<AnyValue> Args,
                         MemAllocKind AllocKind);
  AnyValue executeFree(ArrayRef<AnyValue> Args);
  AnyValue executePuts(ArrayRef<AnyValue> Args);
  AnyValue executePrintf(ArrayRef<AnyValue> Args);
  AnyValue executeExit(ArrayRef<AnyValue> Args);
  AnyValue executeAbort();
  AnyValue executeTerminate();

public:
  Library(Context &Ctx, EventHandler &Handler, const DataLayout &DL,
          ExecutorBase &Executor);

  /// Simulates a libcall. Returns std::nullopt if an unsupported LibFunc is
  /// passed. Note that the caller is responsible for ensuring the types and
  /// number of the arguments are correct.
  std::optional<AnyValue> executeLibcall(LibFunc LF, StringRef Name, Type *Type,
                                         ArrayRef<AnyValue> Args);
};

} // namespace llvm::ubi

#endif // LLVM_TOOLS_LLUBI_LIBRARY_H
