//===- VectorToArmSME.h - Convert vector to ArmSME dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_VECTORTOARMSME_VECTORTOARMSME_H_
#define MLIR_CONVERSION_VECTORTOARMSME_VECTORTOARMSME_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTVECTORTOARMSME
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to lower Vector ops to ArmSME ops that map to LLVM
/// intrinsics.
void populateVectorToArmSMEPatterns(RewritePatternSet &patterns,
                                    MLIRContext &ctx);

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOARMSME_VECTORTOARMSME_H_
