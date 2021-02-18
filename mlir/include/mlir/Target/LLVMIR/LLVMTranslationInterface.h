//===- LLVMTranslationInterface.h - Translation to LLVM iface ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines dialect interfaces for translation to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_LLVMTRANSLATIONINTERFACE_H
#define MLIR_TARGET_LLVMIR_LLVMTRANSLATIONINTERFACE_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class IRBuilderBase;
}

namespace mlir {
namespace LLVM {
class ModuleTranslation;
} // namespace LLVM

/// Base class for dialect interfaces providing translation to LLVM IR.
/// Dialects that can be translated should provide an implementation of this
/// interface for the supported operations. The interface may be implemented in
/// a separate library to avoid the "main" dialect library depending on LLVM IR.
/// The interface can be attached using the delayed registration mechanism
/// available in DialectRegistry.
class LLVMTranslationDialectInterface
    : public DialectInterface::Base<LLVMTranslationDialectInterface> {
public:
  LLVMTranslationDialectInterface(Dialect *dialect) : Base(dialect) {}

  /// Hook for derived dialect interface to provide translation of the
  /// operations to LLVM IR.
  virtual LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const {
    return failure();
  }
};

/// Interface collection for translation to LLVM IR, dispatches to a concrete
/// interface implementation based on the dialect to which the given op belongs.
class LLVMTranslationInterface
    : public DialectInterfaceCollection<LLVMTranslationDialectInterface> {
public:
  using Base::Base;

  /// Translates the given operation to LLVM IR using the interface implemented
  /// by the op's dialect.
  virtual LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const {
    if (const LLVMTranslationDialectInterface *iface = getInterfaceFor(op))
      return iface->convertOperation(op, builder, moduleTranslation);
    return failure();
  }
};

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_LLVMTRANSLATIONINTERFACE_H
