//===- InferIntDivisibilityOpInterfaceImpl.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_IR_INFERINTDIVISIBILITYOPINTERFACEIMPL_H
#define MLIR_DIALECT_AFFINE_IR_INFERINTDIVISIBILITYOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace affine {
void registerInferIntDivisibilityOpInterfaceExternalModels(
    DialectRegistry &registry);
} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_IR_INFERINTDIVISIBILITYOPINTERFACEIMPL_H
