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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Identifier.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class IRBuilderBase;
} // namespace llvm

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

  /// Hook for derived dialect interface to act on an operation that has dialect
  /// attributes from the derived dialect (the operation itself may be from a
  /// different dialect). This gets called after the operation has been
  /// translated. The hook is expected to use moduleTranslation to look up the
  /// translation results and amend the corresponding IR constructs. Does
  /// nothing and succeeds by default.
  virtual LogicalResult
  amendOperation(Operation *op, NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const {
    return success();
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

  /// Acts on the given operation using the interface implemented by the dialect
  /// of one of the operation's dialect attributes.
  virtual LogicalResult
  amendOperation(Operation *op, NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const {
    if (const LLVMTranslationDialectInterface *iface =
            getInterfaceFor(attribute.first.getDialect())) {
      return iface->amendOperation(op, attribute, moduleTranslation);
    }
    return success();
  }
};

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_LLVMTRANSLATIONINTERFACE_H
