//===- SMTExtension.h - SMT extension for Transform dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_SMTEXTENSION_SMTEXTENSION_H
#define MLIR_DIALECT_TRANSFORM_SMTEXTENSION_SMTEXTENSION_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
class DialectRegistry;

namespace transform {
/// Registers the SMT extension of the Transform dialect in the given registry.
void registerSMTExtension(DialectRegistry &dialectRegistry);
} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORM_SMTEXTENSION_SMTEXTENSION_H
