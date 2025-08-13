//===- StructuralTypeConversions.cpp - scf structural type conversions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Transforms/DialectConversion.h"
#include <optional>

using namespace mlir;
using namespace mlir::scf;

namespace {

/// Flatten the given value ranges into a single vector of values.
static SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (const auto &vals : values)
    llvm::append_range(result, vals);
  return result;
}

// CRTP
// A base class that takes care of 1:N type conversion, which maps the converted
// op results (computed by the derived class) and materializes 1:N conversion.
template <typename SourceOp, typename ConcretePattern>
class Structural1ToNConversionPattern : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::typeConverter;
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OneToNOpAdaptor =
      typename OpConversionPattern<SourceOp>::OneToNOpAdaptor;

  //
  // Derived classes should provide the following method which performs the
  // actual conversion. It should return std::nullopt upon conversion failure
  // and return the converted operation upon success.
  //
  // std::optional<SourceOp> convertSourceOp(
  //     SourceOp op, OneToNOpAdaptor adaptor,
  //     ConversionPatternRewriter &rewriter,
  //     TypeRange dstTypes) const;

  LogicalResult
  matchAndRewrite(SourceOp op, OneToNOpAdaptor adaptor,
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
    std::optional<SourceOp> newOp =
        static_cast<const ConcretePattern *>(this)->convertSourceOp(
            op, adaptor, rewriter, dstTypes);

    if (!newOp)
      return rewriter.notifyMatchFailure(op, "could not convert operation");

    // Packs the return value.
    SmallVector<ValueRange> packedRets;
    for (unsigned i = 1, e = offsets.size(); i < e; i++) {
      unsigned start = offsets[i - 1], end = offsets[i];
      unsigned len = end - start;
      ValueRange mappedValue = newOp->getResults().slice(start, len);
      packedRets.push_back(mappedValue);
    }

    rewriter.replaceOpWithMultiple(op, packedRets);
    return success();
  }
};

class ConvertForOpTypes
    : public Structural1ToNConversionPattern<ForOp, ConvertForOpTypes> {
public:
  using Structural1ToNConversionPattern::Structural1ToNConversionPattern;

  // The callback required by CRTP.
  std::optional<ForOp> convertSourceOp(ForOp op, OneToNOpAdaptor adaptor,
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
    // 2. We need to reuse the original region instead of cloning it, otherwise
    // the dialect conversion framework thinks that we just inserted all the
    // cloned child ops. But what we want is to "take" the child regions and let
    // the dialect conversion framework continue recursively into ops inside
    // those regions (which are already in its worklist; inlining them into the
    // new op's regions doesn't remove the child ops from the worklist).

    // convertRegionTypes already takes care of 1:N conversion.
    if (failed(rewriter.convertRegionTypes(&op.getRegion(), *typeConverter)))
      return std::nullopt;

    // We can not do clone as the number of result types after conversion
    // might be different.
    ForOp newOp = ForOp::create(rewriter, op.getLoc(),
                                llvm::getSingleElement(adaptor.getLowerBound()),
                                llvm::getSingleElement(adaptor.getUpperBound()),
                                llvm::getSingleElement(adaptor.getStep()),
                                flattenValues(adaptor.getInitArgs()),
                                /*bodyBuilder=*/nullptr, op.getUnsignedCmp());

    // Reserve whatever attributes in the original op.
    newOp->setAttrs(op->getAttrs());

    // We do not need the empty block created by rewriter.
    rewriter.eraseBlock(newOp.getBody(0));
    // Inline the type converted region from the original operation.
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    return newOp;
  }
};
} // namespace

namespace {
class ConvertIfOpTypes
    : public Structural1ToNConversionPattern<IfOp, ConvertIfOpTypes> {
public:
  using Structural1ToNConversionPattern::Structural1ToNConversionPattern;

  std::optional<IfOp> convertSourceOp(IfOp op, OneToNOpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter,
                                      TypeRange dstTypes) const {

    IfOp newOp =
        IfOp::create(rewriter, op.getLoc(), dstTypes,
                     llvm::getSingleElement(adaptor.getCondition()), true);
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

  std::optional<WhileOp> convertSourceOp(WhileOp op, OneToNOpAdaptor adaptor,
                                         ConversionPatternRewriter &rewriter,
                                         TypeRange dstTypes) const {
    auto newOp = WhileOp::create(rewriter, op.getLoc(), dstTypes,
                                 flattenValues(adaptor.getOperands()));

    for (auto i : {0u, 1u}) {
      if (failed(rewriter.convertRegionTypes(&op.getRegion(i), *typeConverter)))
        return std::nullopt;
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
  matchAndRewrite(scf::YieldOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(
        op, flattenValues(adaptor.getOperands()));
    return success();
  }
};
} // namespace

namespace {
class ConvertConditionOpTypes : public OpConversionPattern<ConditionOp> {
public:
  using OpConversionPattern<ConditionOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConditionOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(
        op, [&]() { op->setOperands(flattenValues(adaptor.getOperands())); });
    return success();
  }
};
} // namespace

void mlir::scf::populateSCFStructuralTypeConversions(
    const TypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ConvertForOpTypes, ConvertIfOpTypes, ConvertYieldOpTypes,
               ConvertWhileOpTypes, ConvertConditionOpTypes>(
      typeConverter, patterns.getContext());
}

void mlir::scf::populateSCFStructuralTypeConversionTarget(
    const TypeConverter &typeConverter, ConversionTarget &target) {
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

void mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
    const TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  populateSCFStructuralTypeConversions(typeConverter, patterns);
  populateSCFStructuralTypeConversionTarget(typeConverter, target);
}
