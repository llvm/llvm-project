//===- TestTransformDialectExtension.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an extension of the MLIR Transform dialect for testing
// purposes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTTRANSFORMDIALECTEXTENSION_H
#define MLIR_TESTTRANSFORMDIALECTEXTENSION_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

#define GET_OP_CLASSES
#include "TestTransformDialectExtension.h.inc"

namespace test {
/// Registers the test extension to the Transform dialect.
void registerTestTransformDialectExtension(::mlir::DialectRegistry &registry);
} // namespace test

#endif // MLIR_TESTTRANSFORMDIALECTEXTENSION_H
