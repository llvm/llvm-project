//===- Transforms.h - XeGPU Dialect transformations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_XEGPU_TRANSFORMS_TRANSFORMS_H

namespace mlir {
class RewritePatternSet;

namespace xegpu {

/// Appends patterns for folding aliasing ops into XeGPU ops into `patterns`.
void populateXeGPUFoldAliasOpsPatterns(RewritePatternSet &patterns);

} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_TRANSFORMS_TRANSFORMS_H
