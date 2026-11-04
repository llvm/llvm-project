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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectInterface.h"

namespace llvm {
class Instruction;
class IRBuilderBase;
} // namespace llvm

namespace mlir {
namespace LLVM {
class ModuleTranslation;
class LLVMFuncOp;
} // namespace LLVM
} // namespace mlir

#include "mlir/Target/LLVMIR/LLVMTranslationDialectInterface.h.inc"

namespace mlir {

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
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const {
    if (const LLVMTranslationDialectInterface *iface =
            getInterfaceFor(attribute.getNameDialect())) {
      return iface->amendOperation(op, instructions, attribute,
                                   moduleTranslation);
    }
    return success();
  }

  /// Acts on the given function operation using the interface implemented by
  /// the dialect of one of the function parameter attributes.
  virtual LogicalResult
  convertParameterAttr(LLVM::LLVMFuncOp function, int argIdx,
                       NamedAttribute attribute,
                       LLVM::ModuleTranslation &moduleTranslation) const {
    if (const LLVMTranslationDialectInterface *iface =
            getInterfaceFor(attribute.getNameDialect())) {
      return iface->convertParameterAttr(function, argIdx, attribute,
                                         moduleTranslation);
    }
    function.emitWarning("Unhandled parameter attribute '" +
                         attribute.getName().str() + "'");
    return success();
  }
};

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_LLVMTRANSLATIONINTERFACE_H
