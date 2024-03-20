//===- PDLExtensionOps.h - PDL extension for Transform dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_PDLEXTENSION_PDLEXTENSIONOPS_H
#define MLIR_DIALECT_TRANSFORM_PDLEXTENSION_PDLEXTENSIONOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.h.inc"

namespace mlir {
namespace transform {
/// PDL constraint callbacks that can be used by the PDL extension of the
/// Transform dialect. These are owned by the Transform dialect and can be
/// populated by extensions.
class PDLMatchHooks : public TransformDialectData<PDLMatchHooks> {
public:
  PDLMatchHooks(MLIRContext *ctx) : TransformDialectData(ctx) {}

  /// Takes ownership of the named PDL constraint function from the given
  /// map and makes them available for use by the operations in the dialect.
  void
  mergeInPDLMatchHooks(llvm::StringMap<PDLConstraintFunction> &&constraintFns);

  /// Returns the named PDL constraint functions available in the dialect
  /// as a map from their name to the function.
  const llvm::StringMap<::mlir::PDLConstraintFunction> &
  getPDLConstraintHooks() const;

private:
  /// A container for PDL constraint function that can be used by
  /// operations in this dialect.
  PDLPatternModule pdlMatchHooks;
};
} // namespace transform
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::transform::PDLMatchHooks)

#endif // MLIR_DIALECT_TRANSFORM_PDLEXTENSION_PDLEXTENSIONOPS_H
