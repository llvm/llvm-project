//===- StructuralTypeConversions.cpp - scf structural type conversions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::scf;

namespace {

// Unpacks the single unrealized_conversion_cast using the list of inputs
// e.g., return [%b, %c, %d] for %a = unrealized_conversion_cast(%b, %c, %d)
static void unpackUnrealizedConversionCast(Value v,
                                           SmallVectorImpl<Value> &unpacked) {
  if (auto cast =
          dyn_cast_or_null<UnrealizedConversionCastOp>(v.getDefiningOp())) {
    if (cast.getInputs().size() != 1) {
      // 1 : N type conversion.
      unpacked.append(cast.getInputs().begin(), cast.getInputs().end());
      return;
    }
  }
  // 1 : 1 type conversion.
  unpacked.push_back(v);
}

// CRTP
// A base class that takes care of 1:N type conversion, which maps the converted
// op results (computed by the derived class) and materializes 1:N conversion.
template <typename SourceOp, typename ConcretePattern>
class Structural1ToNConversionPattern : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::typeConverter;
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<SourceOp>::OpAdaptor;

  //
  // Derived classes should provide the following method which performs the
  // actual conversion. It should return llvm::None upon conversion failure and
  // return the converted operation upon success.
  //
  // Optional<SourceOp> convertSourceOp(SourceOp op, OpAdaptor adaptor,
  //                                    ConversionPatternRewriter &rewriter,
  //                                    TypeRange dstTypes) const;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> dstTypes;
    SmallVector<unsigned> offsets;
    offsets.push_back(0);
    // Do the type conversion and record the offsets.
    for (Type type : op.getResultTypes()) {
      if (failed(typeConverter->convertTypes(type, dstTypes)))
        return rewriter.notifyMatchFailure(op, "could not convert result type");
      offsets.push_back(dstTypes.size());
    }

    // Calls the actual converter implementation to convert the operation.
    Optional<SourceOp> newOp =
        static_cast<const ConcretePattern *>(this)->convertSourceOp(
            op, adaptor, rewriter, dstTypes);

    if (!newOp)
      return rewriter.notifyMatchFailure(op, "could not convert operation");

    // Packs the return value.
    SmallVector<Value> packedRets;
    for (unsigned i = 1, e = offsets.size(); i < e; i++) {
      unsigned start = offsets[i - 1], end = offsets[i];
      unsigned len = end - start;
      ValueRange mappedValue = newOp->getResults().slice(start, len);
      if (len != 1) {
        // 1 : N type conversion.
        Type origType = op.getResultTypes()[i - 1];
        Value mat = typeConverter->materializeSourceConversion(
            rewriter, op.getLoc(), origType, mappedValue);
        if (!mat) {
          return rewriter.notifyMatchFailure(
              op, "Failed to materialize 1:N type conversion");
        }
        packedRets.push_back(mat);
      } else {
        // 1 : 1 type conversion.
        packedRets.push_back(mappedValue.front());
      }
    }

    rewriter.replaceOp(op, packedRets);
    return success();
  }
};

class ConvertForOpTypes
    : public Structural1ToNConversionPattern<ForOp, ConvertForOpTypes> {
public:
  using Structural1ToNConversionPattern::Structural1ToNConversionPattern;

  // The callback required by CRTP.
  Optional<ForOp> convertSourceOp(ForOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  TypeRange dstTypes) const {
    // Create a empty new op and inline the regions from the old op.
    //
    // This is a little bit tricky. We have two concerns here:
    //
    // 1. We cannot update the op in place because the dialect conversion
    // framework does not track type changes for ops updated in place, so it
    // won't insert appropriate materializations on the changed result types.
    // PR47938 tracks this issue, but it seems hard to fix. Instead, we need
    // to clone the op.
    //
    // 2. We need to resue the original region instead of cloning it, otherwise
    // the dialect conversion framework thinks that we just inserted all the
    // cloned child ops. But what we want is to "take" the child regions and let
    // the dialect conversion framework continue recursively into ops inside
    // those regions (which are already in its worklist; inlining them into the
    // new op's regions doesn't remove the child ops from the worklist).

    // convertRegionTypes already takes care of 1:N conversion.
    if (failed(rewriter.convertRegionTypes(&op.getLoopBody(), *typeConverter)))
      return llvm::None;

    // Unpacked the iteration arguments.
    SmallVector<Value> flatArgs;
    for (Value arg : adaptor.getInitArgs())
      unpackUnrealizedConversionCast(arg, flatArgs);

    // We can not do clone as the number of result types after conversion
    // might be different.
    ForOp newOp = rewriter.create<ForOp>(op.getLoc(), adaptor.getLowerBound(),
                                         adaptor.getUpperBound(),
                                         adaptor.getStep(), flatArgs);

    // Reserve whatever attributes in the original op.
    newOp->setAttrs(op->getAttrs());

    // We do not need the empty block created by rewriter.
    rewriter.eraseBlock(newOp.getBody(0));
    // Inline the type converted region from the original operation.
    rewriter.inlineRegionBefore(op.getLoopBody(), newOp.getLoopBody(),
                                newOp.getLoopBody().end());

    return newOp;
  }
};
} // namespace

namespace {
class ConvertIfOpTypes
    : public Structural1ToNConversionPattern<IfOp, ConvertIfOpTypes> {
public:
  using Structural1ToNConversionPattern::Structural1ToNConversionPattern;

  Optional<IfOp> convertSourceOp(IfOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 TypeRange dstTypes) const {

    IfOp newOp = rewriter.create<IfOp>(op.getLoc(), dstTypes,
                                       adaptor.getCondition(), true);
    newOp->setAttrs(op->getAttrs());

    // We do not need the empty blocks created by rewriter.
    rewriter.eraseBlock(newOp.elseBlock());
    rewriter.eraseBlock(newOp.thenBlock());

    // Inlines block from the original operation.
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    return newOp;
  }
};
} // namespace

namespace {
class ConvertWhileOpTypes
    : public Structural1ToNConversionPattern<WhileOp, ConvertWhileOpTypes> {
public:
  using Structural1ToNConversionPattern::Structural1ToNConversionPattern;

  Optional<WhileOp> convertSourceOp(WhileOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    TypeRange dstTypes) const {
    // Unpacked the iteration arguments.
    SmallVector<Value> flatArgs;
    for (Value arg : adaptor.getOperands())
      unpackUnrealizedConversionCast(arg, flatArgs);

    auto newOp = rewriter.create<WhileOp>(op.getLoc(), dstTypes, flatArgs);

    for (auto i : {0u, 1u}) {
      if (failed(rewriter.convertRegionTypes(&op.getRegion(i), *typeConverter)))
        return llvm::None;
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
    }
    return newOp;
  }
};
} // namespace

namespace {
// When the result types of a ForOp/IfOp get changed, the operand types of the
// corresponding yield op need to be changed. In order to trigger the
// appropriate type conversions / materializations, we need a dummy pattern.
class ConvertYieldOpTypes : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> unpackedYield;
    for (Value operand : adaptor.getOperands())
      unpackUnrealizedConversionCast(operand, unpackedYield);

    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, unpackedYield);
    return success();
  }
};
} // namespace

namespace {
class ConvertConditionOpTypes : public OpConversionPattern<ConditionOp> {
public:
  using OpConversionPattern<ConditionOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> unpackedYield;
    for (Value operand : adaptor.getOperands())
      unpackUnrealizedConversionCast(operand, unpackedYield);

    rewriter.updateRootInPlace(op, [&]() { op->setOperands(unpackedYield); });
    return success();
  }
};
} // namespace

void mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  patterns.add<ConvertForOpTypes, ConvertIfOpTypes, ConvertYieldOpTypes,
               ConvertWhileOpTypes, ConvertConditionOpTypes>(
      typeConverter, patterns.getContext());
  target.addDynamicallyLegalOp<ForOp, IfOp>([&](Operation *op) {
    return typeConverter.isLegal(op->getResultTypes());
  });
  target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
    // We only have conversions for a subset of ops that use scf.yield
    // terminators.
    if (!isa<ForOp, IfOp, WhileOp>(op->getParentOp()))
      return true;
    return typeConverter.isLegal(op.getOperandTypes());
  });
  target.addDynamicallyLegalOp<WhileOp, ConditionOp>(
      [&](Operation *op) { return typeConverter.isLegal(op); });
}
