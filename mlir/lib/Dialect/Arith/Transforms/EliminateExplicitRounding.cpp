//===- EliminateExplicitRounding.cpp - Remove intermediate extf/truncf pairs
//-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements removing intermediate extf/truncf pairs inserted from
// type conversion.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace arith {
#define GEN_PASS_DEF_ARITHELIMINATEEXPLICITROUNDING
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace arith
} // namespace mlir

using namespace mlir;

namespace {

struct EliminateExplicitRounding final
    : arith::impl::ArithEliminateExplicitRoundingBase<
          EliminateExplicitRounding> {
  using ArithEliminateExplicitRoundingBase::ArithEliminateExplicitRoundingBase;
  void runOnOperation() override {
    getOperation()->walk([&](arith::ExtFOp extFOp) {
      // Check whether match `truncF->extF` pair.
      auto truncFOp = extFOp.getOperand().getDefiningOp<arith::TruncFOp>();
      if (!truncFOp)
        return;
      // Check whether the rounding pair's input and output data type are the
      // same. Currently only consider to eliminate rounding pairs for (bf16 /
      // f16 <-> f32).
      Value input = truncFOp.getOperand();
      Type inTy = getElementTypeOrSelf(input.getType());
      Type outTy = getElementTypeOrSelf(extFOp.getType());
      Type shortTy = getElementTypeOrSelf(truncFOp.getType());
      if (isa<Float32Type>(inTy) && isa<Float32Type>(outTy) &&
          (isa<Float16Type, BFloat16Type>(shortTy))) {
        extFOp.replaceAllUsesWith(input);
        extFOp.erase();
        if (truncFOp.getResult().getUses().empty())
          truncFOp.erase();
      }
    });
  }
};

} // namespace
