//===-- Passes.h - AGX transformation passes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AGX_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_AGX_TRANSFORMS_PASSES_H_

#include "mlir/Dialect/AGX/Utils/ChipInfo.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace agx {

#define GEN_PASS_DECL_AGXEXPANDOPSPASS
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/AGX/Transforms/Passes.h.inc"

/// Patterns that rewrite the agx operations `chip` does not run natively into
/// their portable forms.
void populateAGXExpandOpsPatterns(RewritePatternSet &patterns,
                                  const ChipInfo &chip);

/// The chip of the `#agxir.target` of the gpu.module enclosing `op`
/// ("agx-g14g" when there is none).
StringRef getTargetChip(Operation *op);

} // namespace agx
} // namespace mlir

#endif // MLIR_DIALECT_AGX_TRANSFORMS_PASSES_H_
