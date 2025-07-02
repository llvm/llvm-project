//===- ConversionToReplicatedConstantCompositePass.cpp
//---------------------------===//
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace spirv {
#define GEN_PASS_DEF_SPIRVREPLICATEDCONSTANTCOMPOSITEPASS
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"
} // namespace spirv
} // namespace mlir

using namespace mlir;

namespace {

Attribute getSplatAttribute(Attribute valueAttr, uint32_t splatCount) {
  Attribute attr;
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
    if (denseAttr.isSplat()) {
      attr = denseAttr.getSplatValue<Attribute>();
      splatCount = denseAttr.size();
    }
  } else if (auto arrayAttr = dyn_cast<ArrayAttr>(valueAttr)) {
    if (std::adjacent_find(arrayAttr.begin(), arrayAttr.end(),
                           std::not_equal_to<>()) == arrayAttr.end()) {
      attr = arrayAttr[0];
      splatCount = arrayAttr.size();
    }
  }

  if (attr) {
    if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
      if (isa<spirv::CompositeType>(typedAttr.getType()))
        if (Attribute newAttr = getSplatAttribute(attr, splatCount))
          attr = newAttr;
    } else if (isa<ArrayAttr>(attr)) {
      if (Attribute newAttr = getSplatAttribute(attr, splatCount))
        attr = newAttr;
    }
  }

  return attr;
}

} // namespace

namespace {
class ConversionToReplicatedConstantCompositePass
    : public spirv::impl::SPIRVReplicatedConstantCompositePassBase<
          ConversionToReplicatedConstantCompositePass> {
public:
  void runOnOperation() override;
};

class ConstantOpConversion : public OpRewritePattern<spirv::ConstantOp> {
  using OpRewritePattern<spirv::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto compositeType = dyn_cast_or_null<spirv::CompositeType>(op.getType());
    if (!compositeType)
      return rewriter.notifyMatchFailure(op, "not a composite constant");

    uint32_t splatCount = 0;
    Attribute splatAttr = getSplatAttribute(op.getValue(), splatCount);
    if (!splatAttr)
      return rewriter.notifyMatchFailure(op, "composite is not splat");

    if (splatCount == 1)
      return rewriter.notifyMatchFailure(op,
                                         "composite has only one consituent");

    rewriter.replaceOpWithNewOp<spirv::EXTConstantCompositeReplicateOp>(
        op, op.getType(), splatAttr);

    return success();
  }
};

class SpecConstantCompositeOpConversion
    : public OpRewritePattern<spirv::SpecConstantCompositeOp> {
  using OpRewritePattern<spirv::SpecConstantCompositeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::SpecConstantCompositeOp op,
                                PatternRewriter &rewriter) const override {
    auto compositeType = dyn_cast_or_null<spirv::CompositeType>(op.getType());
    if (!compositeType)
      return rewriter.notifyMatchFailure(op, "not a composite constant");

    auto constituents = op.getConstituents();
    if (constituents.size() == 1)
      return rewriter.notifyMatchFailure(op,
                                         "composite has only one consituent");

    if (!(std::adjacent_find(constituents.begin(), constituents.end(),
                             std::not_equal_to<>()) == constituents.end()))
      return rewriter.notifyMatchFailure(op, "composite is not splat");

    auto splatConstituent =
        dyn_cast<FlatSymbolRefAttr>(op.getConstituents()[0]);

    rewriter.replaceOpWithNewOp<spirv::EXTSpecConstantCompositeReplicateOp>(
        op, TypeAttr::get(op.getType()), op.getSymNameAttr(), splatConstituent);

    return success();
  }
};

void ConversionToReplicatedConstantCompositePass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<ConstantOpConversion>(context);
  patterns.add<SpecConstantCompositeOpConversion>(context);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace