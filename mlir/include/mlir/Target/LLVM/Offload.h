//===- Offload.h - LLVM Target Offload --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares LLVM target offload utility classes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_OFFLOAD_H
#define MLIR_TARGET_LLVM_OFFLOAD_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Constant;
class GlobalVariable;
class Module;
} // namespace llvm

namespace mlir {
namespace LLVM {
/// `OffloadHandler` is a utility class for creating LLVM offload entries. LLVM
/// offload entries hold information on offload symbols; for example, for a GPU
/// kernel, this includes its host address to identify the kernel and the kernel
/// identifier in the binary. Arrays of offload entries can be used to register
/// functions within the CUDA/HIP runtime. Libomptarget also uses these entries
/// to register OMP target offload kernels and variables.
class OffloadHandler {
public:
  using OffloadEntryArray =
      std::pair<llvm::GlobalVariable *, llvm::GlobalVariable *>;
  OffloadHandler(llvm::Module &module) : module(module) {}

  /// Returns the begin symbol name used in the entry array.
  static std::string getBeginSymbol(StringRef suffix);

  /// Returns the end symbol name used in the entry array.
  static std::string getEndSymbol(StringRef suffix);

  /// Returns the entry array if it exists or a pair of null pointers.
  OffloadEntryArray getEntryArray(StringRef suffix);

  /// Emits an empty array of offloading entries.
  OffloadEntryArray emitEmptyEntryArray(StringRef suffix);

  /// Inserts an offloading entry into an existing entry array. This method
  /// returns failure if the entry array hasn't been declared.
  LogicalResult insertOffloadEntry(StringRef suffix, llvm::Constant *entry);

protected:
  llvm::Module &module;
};
} // namespace LLVM
} // namespace mlir

#endif // MLIR_TARGET_LLVM_OFFLOAD_H
