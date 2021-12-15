//===- DebugTranslation.h - MLIR to LLVM Debug conversion -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the translation between an MLIR debug information and
// the corresponding LLVMIR representation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_LLVMIR_DEBUGTRANSLATION_H_
#define MLIR_LIB_TARGET_LLVMIR_DEBUGTRANSLATION_H_

#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/DIBuilder.h"

namespace mlir {
class Operation;

namespace LLVM {
class LLVMFuncOp;

namespace detail {
class DebugTranslation {
public:
  DebugTranslation(Operation *module, llvm::Module &llvmModule);

  /// Finalize the translation of debug information.
  void finalize();

  /// Translate the given location to an llvm debug location.
  const llvm::DILocation *translateLoc(Location loc, llvm::DILocalScope *scope);

  /// Translate the debug information for the given function.
  void translate(LLVMFuncOp func, llvm::Function &llvmFunc);

private:
  /// Translate the given location to an llvm debug location with the given
  /// scope and inlinedAt parameters.
  const llvm::DILocation *translateLoc(Location loc, llvm::DILocalScope *scope,
                                       const llvm::DILocation *inlinedAt);

  /// Create an llvm debug file for the given file path.
  llvm::DIFile *translateFile(StringRef fileName);

  /// A mapping between mlir location+scope and the corresponding llvm debug
  /// metadata.
  DenseMap<std::pair<Location, llvm::DILocalScope *>, const llvm::DILocation *>
      locationToLoc;

  /// A mapping between filename and llvm debug file.
  /// TODO: Change this to DenseMap<Identifier, ...> when we can
  /// access the Identifier filename in FileLineColLoc.
  llvm::StringMap<llvm::DIFile *> fileMap;

  /// A string containing the current working directory of the compiler.
  SmallString<256> currentWorkingDir;

  /// Debug information fields.
  llvm::DIBuilder builder;
  llvm::LLVMContext &llvmCtx;
  llvm::DICompileUnit *compileUnit;
};

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // MLIR_LIB_TARGET_LLVMIR_DEBUGTRANSLATION_H_
