//===- ComposeSubView.h - Combining composed subview ops --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Patterns for combining composed subview ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STANDARDOPS_TRANSFORMS_COMPOSESUBVIEW_H_
#define MLIR_DIALECT_STANDARDOPS_TRANSFORMS_COMPOSESUBVIEW_H_

namespace mlir {

// Forward declarations.
class MLIRContext;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

void populateComposeSubViewPatterns(OwningRewritePatternList &patterns,
                                    MLIRContext *context);

} // namespace mlir

#endif // MLIR_DIALECT_STANDARDOPS_TRANSFORMS_COMPOSESUBVIEW_H_
