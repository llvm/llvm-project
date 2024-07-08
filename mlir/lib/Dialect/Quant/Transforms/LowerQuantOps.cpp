//===- LowerQuantOps.cpp - Lower 'quant' dialect ops ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms `quant.dcast` and `quant.qcast` into lower-level ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace quant {

#define GEN_PASS_DEF_LOWERQUANTOPS
#include "mlir/Dialect/Quant/Transforms/Passes.h.inc"

namespace {

struct LowerQuantOps : public impl::LowerQuantOpsBase<LowerQuantOps> {
  void runOnOperation() override {
    Operation *parentOp = getOperation();
  }
};

} // namespace

} // namespace quant
} // namespace mlir
