
//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the loop
// transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"

namespace mlir {

class OwningRewritePatternList;

/// Creates an instance of the ExpandAtomic pass.
std::unique_ptr<Pass> createExpandAtomicPass();

void populateExpandMemRefReshapePattern(OwningRewritePatternList &patterns,
                                        MLIRContext *ctx);

void populateExpandTanhPattern(OwningRewritePatternList &patterns,
                               MLIRContext *ctx);

void populateStdBufferizePatterns(MLIRContext *context,
                                  BufferizeTypeConverter &typeConverter,
                                  OwningRewritePatternList &patterns);

/// Creates an instance of std bufferization pass.
std::unique_ptr<Pass> createStdBufferizePass();

/// Creates an instance of func bufferization pass.
std::unique_ptr<Pass> createFuncBufferizePass();

/// Creates an instance of the StdExpandDivs pass that legalizes Std
/// dialect Divs to be convertible to StaLLVMndard. For example,
/// `std.ceildivi_signed` get transformed to a number of std operations,
/// which can be lowered to LLVM.
std::unique_ptr<Pass> createStdExpandDivsPass();

/// Collects a set of patterns to rewrite ops within the Std dialect.
void populateStdExpandDivsRewritePatterns(MLIRContext *context,
                                        OwningRewritePatternList &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/StandardOps/Transforms/Passes.h.inc"

} // end namespace mlir

#endif // MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES_H_
