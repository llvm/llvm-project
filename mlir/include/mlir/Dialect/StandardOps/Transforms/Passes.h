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

namespace mlir {
namespace bufferization {
class BufferizeTypeConverter;
} // namespace bufferization

class GlobalCreator;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

void populateStdBufferizePatterns(
    bufferization::BufferizeTypeConverter &typeConverter,
    RewritePatternSet &patterns);

/// Creates an instance of std bufferization pass.
std::unique_ptr<Pass> createStdBufferizePass();

/// Creates an instance of func bufferization pass.
std::unique_ptr<Pass> createFuncBufferizePass();

/// Add patterns to bufferize tensor constants into global memrefs to the given
/// pattern list.
void populateTensorConstantBufferizePatterns(
    GlobalCreator &globalCreator,
    bufferization::BufferizeTypeConverter &typeConverter,
    RewritePatternSet &patterns);

/// Creates an instance of tensor constant bufferization pass.
std::unique_ptr<Pass> createTensorConstantBufferizePass(unsigned alignment = 0);

/// Creates an instance of the StdExpand pass that legalizes Std
/// dialect ops to be convertible to LLVM. For example,
/// `std.arith.ceildivsi` gets transformed to a number of std operations,
/// which can be lowered to LLVM; `memref.reshape` gets converted to
/// `memref_reinterpret_cast`.
std::unique_ptr<Pass> createStdExpandOpsPass();

/// Collects a set of patterns to rewrite ops within the Std dialect.
void populateStdExpandOpsPatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/StandardOps/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES_H_
