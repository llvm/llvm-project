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

static Type getArrayElemType(Attribute attr) {
  if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
    return typedAttr.getType();
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    return ArrayType::get(getArrayElemType(arrayAttr[0]), arrayAttr.size());
  }

  return nullptr;
}

static std::pair<Attribute, uint32_t>
getSplatAttrAndNumElements(Attribute valueAttr, Type valueType) {
  auto compositeType = dyn_cast_or_null<spirv::CompositeType>(valueType);
  if (!compositeType)
    return {nullptr, 1};

  if (auto splatAttr = dyn_cast<SplatElementsAttr>(valueAttr)) {
    return {splatAttr.getSplatValue<Attribute>(), splatAttr.size()};
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(valueAttr)) {
    if (llvm::all_equal(arrayAttr)) {
      Attribute attr = arrayAttr[0];
      uint32_t numElements = arrayAttr.size();

      // Find the inner-most splat value for array of composites
      auto [newAttr, newNumElements] =
          getSplatAttrAndNumElements(attr, getArrayElemType(attr));
      if (newAttr) {
        attr = newAttr;
        numElements *= newNumElements;
      }
      return {attr, numElements};
    }
  }

  return {nullptr, 1};
}

struct ConstantOpConversion final : OpRewritePattern<spirv::ConstantOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(spirv::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto [attr, numElements] =
        getSplatAttrAndNumElements(op.getValue(), op.getType());
    if (!attr)
      return rewriter.notifyMatchFailure(op, "composite is not splat");

    if (numElements == 1)
      return rewriter.notifyMatchFailure(op,
                                         "composite has only one constituent");

    rewriter.replaceOpWithNewOp<spirv::EXTConstantCompositeReplicateOp>(
        op, op.getType(), attr);
    return success();
  }
};

struct SpecConstantCompositeOpConversion final
    : OpRewritePattern<spirv::SpecConstantCompositeOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(spirv::SpecConstantCompositeOp op,
                                PatternRewriter &rewriter) const override {
    auto compositeType = dyn_cast_or_null<spirv::CompositeType>(op.getType());
    if (!compositeType)
      return rewriter.notifyMatchFailure(op, "not a composite constant");

    ArrayAttr constituents = op.getConstituents();
    if (constituents.size() == 1)
      return rewriter.notifyMatchFailure(op,
                                         "composite has only one consituent");

    if (!llvm::all_equal(constituents))
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
