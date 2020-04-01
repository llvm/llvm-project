//===- ConvertVectorToLoops.h - Utils to convert from the vector dialect --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLOOPS_H_
#define MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLOOPS_H_

namespace mlir {
class MLIRContext;
class OwningRewritePatternList;

/// Collect a set of patterns to convert from the Vector dialect to loops + std.
void populateVectorToAffineLoopsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLOOPS_H_
