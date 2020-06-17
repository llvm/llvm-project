//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the
// shape transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES_H_

#include <memory>

namespace mlir {

class MLIRContext;
class OwningRewritePatternList;
class Pass;

/// Creates an instance of the ShapeToShapeLowering pass that legalizes Shape
/// dialect to be convertible to Standard. For example, `shape.num_elements` get
/// transformed to `shape.reduce`, which can be lowered to SCF and Standard.
std::unique_ptr<Pass> createShapeToShapeLowering();

/// Collects a set of patterns to rewrite ops within the Shape dialect.
void populateShapeRewritePatterns(MLIRContext *context,
                                  OwningRewritePatternList &patterns);
} // end namespace mlir

#endif // MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
