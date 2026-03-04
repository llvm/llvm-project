//===- InferExactFromDLTI.cpp - Infer exact flags from DLTI ------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir {
namespace arith {
#define GEN_PASS_DEF_ARITHINFEREXACTFROMDLTI
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace arith
} // namespace mlir

using namespace mlir;
using namespace mlir::arith;

namespace {
struct ArithInferExactFromDLTIPass
    : public arith::impl::ArithInferExactFromDLTIBase<
          ArithInferExactFromDLTIPass> {

  void runOnOperation() override {
    // TODO: Query DataLayout for index bitwidth and apply patterns to add
    // exact on index casts where the source type is narrower than the dest.
  }
};
} // end anonymous namespace

void mlir::arith::populateInferExactFromDLTIPatterns(
    RewritePatternSet &patterns, unsigned indexBitwidth) {
  // TODO: Add patterns for IndexCastOp and IndexCastUIOp that set exact
  // when the source type is narrower than the destination type.
}
