//===- Transforms.h - Mesh Transforms ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_MESH_TRANSFORMS_TRANSFORMS_H

namespace mlir {
class RewritePatternSet;
class SymbolTableCollection;
class DialectRegistry;
namespace mesh {

void processMultiIndexOpLoweringPopulatePatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection);

void processMultiIndexOpLoweringRegisterDialects(DialectRegistry &registry);

} // namespace mesh
} // namespace mlir

#endif // MLIR_DIALECT_MESH_TRANSFORMS_TRANSFORMS_H
