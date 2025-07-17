//===- ConvertToReplicatedConstantCompositePass.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert a splat composite spirv.Constant and
// spirv.SpecConstantComposite to spirv.EXT.ConstantCompositeReplicate and
// spirv.EXT.SpecConstantCompositeReplicate respectively.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::spirv {
#define GEN_PASS_DEF_SPIRVREPLICATEDCONSTANTCOMPOSITEPASS
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"

namespace {

static std::pair<Attribute, uint32_t>
getSplatAttrAndNumElements(Attribute valueAttr) {
  Attribute attr;
  uint32_t splatCount = 0;
  if (auto splatAttr = dyn_cast<SplatElementsAttr>(valueAttr)) {
    return {splatAttr.getSplatValue<Attribute>(), splatAttr.size()};
  }
  if (auto arrayAttr = dyn_cast<ArrayAttr>(valueAttr)) {
    if (llvm::all_equal(arrayAttr)) {
      attr = arrayAttr[0];
      splatCount = arrayAttr.size();
    }

    if (attr) {
      // Find the inner-most splat value for array of composites
      std::pair<Attribute, uint32_t> newSplatAttrAndCount =
          getSplatAttrAndNumElements(attr);
      if (newSplatAttrAndCount.first) {
        return newSplatAttrAndCount;
      }
    }
  }

  return {attr, splatCount};
}

struct ConstantOpConversion final : OpRewritePattern<spirv::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto compositeType = dyn_cast_or_null<spirv::CompositeType>(op.getType());
    if (!compositeType)
      return rewriter.notifyMatchFailure(op, "not a composite constant");

    auto [splatAttr, splatCount] = getSplatAttrAndNumElements(op.getValue());
    if (!splatAttr)
      return rewriter.notifyMatchFailure(op, "composite is not splat");

    if (splatCount == 1)
      return rewriter.notifyMatchFailure(op,
                                         "composite has only one constituent");

    rewriter.replaceOpWithNewOp<spirv::EXTConstantCompositeReplicateOp>(
        op, op.getType(), splatAttr);
    return success();
  }
};

struct SpecConstantCompositeOpConversion final
    : OpRewritePattern<spirv::SpecConstantCompositeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::SpecConstantCompositeOp op,
                                PatternRewriter &rewriter) const override {
    auto compositeType = dyn_cast_or_null<spirv::CompositeType>(op.getType());
    if (!compositeType)
      return rewriter.notifyMatchFailure(op, "not a composite constant");

    ArrayAttr constituents = op.getConstituents();
    if (constituents.size() == 1)
      return rewriter.notifyMatchFailure(op,
                                         "composite has only one consituent");

    if (!(llvm::all_equal(constituents)))
      return rewriter.notifyMatchFailure(op, "composite is not splat");

    auto splatConstituent = dyn_cast<FlatSymbolRefAttr>(constituents[0]);
    if (!splatConstituent)
      return rewriter.notifyMatchFailure(
          op, "expected flat symbol reference for splat constituent");

    rewriter.replaceOpWithNewOp<spirv::EXTSpecConstantCompositeReplicateOp>(
        op, TypeAttr::get(op.getType()), op.getSymNameAttr(), splatConstituent);

    return success();
  }
};

struct ConvertToReplicatedConstantCompositePass final
    : spirv::impl::SPIRVReplicatedConstantCompositePassBase<
          ConvertToReplicatedConstantCompositePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConstantOpConversion, SpecConstantCompositeOpConversion>(
        context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
} // namespace mlir::spirv
