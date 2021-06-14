//===- LinalgToLLVM.cpp - conversion from Linalg to LLVM dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::linalg;

template <typename T>
static Type getPtrToElementType(T containerType, LLVMTypeConverter &lowering) {
  return LLVMPointerType::get(
      lowering.convertType(containerType.getElementType()));
}

/// Convert the given range descriptor type to the LLVMIR dialect.
/// Range descriptor contains the range bounds and the step as 64-bit integers.
///
/// struct {
///   int64_t min;
///   int64_t max;
///   int64_t step;
/// };
static Type convertRangeType(RangeType t, LLVMTypeConverter &converter) {
  auto *context = t.getContext();
  auto int64Ty = converter.convertType(IntegerType::get(context, 64));
  return LLVMStructType::getLiteral(context, {int64Ty, int64Ty, int64Ty});
}

namespace {
// RangeOp creates a new range descriptor.
class RangeOpConversion : public ConvertOpToLLVMPattern<RangeOp> {
public:
  using ConvertOpToLLVMPattern<RangeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RangeOp rangeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto rangeDescriptorTy = convertRangeType(
        rangeOp.getType().cast<RangeType>(), *getTypeConverter());

    ImplicitLocOpBuilder b(rangeOp->getLoc(), rewriter);

    // Fill in an aggregate value of the descriptor.
    RangeOpAdaptor adaptor(operands);
    Value desc = b.create<LLVM::UndefOp>(rangeDescriptorTy);
    desc = b.create<LLVM::InsertValueOp>(desc, adaptor.min(),
                                         rewriter.getI64ArrayAttr(0));
    desc = b.create<LLVM::InsertValueOp>(desc, adaptor.max(),
                                         rewriter.getI64ArrayAttr(1));
    desc = b.create<LLVM::InsertValueOp>(desc, adaptor.step(),
                                         rewriter.getI64ArrayAttr(2));
    rewriter.replaceOp(rangeOp, desc);
    return success();
  }
};

// ReshapeOp creates a new view descriptor of the proper rank.
// For now, the only conversion supported is for target MemRef with static sizes
// and strides.
template <typename ReshapeOp>
class ReshapeOpConversion : public ConvertOpToLLVMPattern<ReshapeOp> {
public:
  using ConvertOpToLLVMPattern<ReshapeOp>::ConvertOpToLLVMPattern;
  using ReshapeOpAdaptor = typename ReshapeOp::Adaptor;

  LogicalResult
  matchAndRewrite(ReshapeOp reshapeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType dstType = reshapeOp.getResultType();

    if (!dstType.hasStaticShape())
      return failure();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto res = getStridesAndOffset(dstType, strides, offset);
    if (failed(res) || llvm::any_of(strides, [](int64_t val) {
          return ShapedType::isDynamicStrideOrOffset(val);
        }))
      return failure();

    ReshapeOpAdaptor adaptor(operands);
    MemRefDescriptor baseDesc(adaptor.src());
    Location loc = reshapeOp->getLoc();
    auto desc =
        MemRefDescriptor::undef(rewriter, reshapeOp->getLoc(),
                                this->typeConverter->convertType(dstType));
    desc.setAllocatedPtr(rewriter, loc, baseDesc.allocatedPtr(rewriter, loc));
    desc.setAlignedPtr(rewriter, loc, baseDesc.alignedPtr(rewriter, loc));
    desc.setOffset(rewriter, loc, baseDesc.offset(rewriter, loc));
    for (auto en : llvm::enumerate(dstType.getShape()))
      desc.setConstantSize(rewriter, loc, en.index(), en.value());
    for (auto en : llvm::enumerate(strides))
      desc.setConstantStride(rewriter, loc, en.index(), en.value());
    rewriter.replaceOp(reshapeOp, {desc});
    return success();
  }
};

// YieldOp produces and LLVM::ReturnOp.
class YieldOpConversion : public ConvertOpToLLVMPattern<linalg::YieldOp> {
public:
  using ConvertOpToLLVMPattern<linalg::YieldOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(linalg::YieldOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
    return success();
  }
};
} // namespace

/// Populate the given list with patterns that convert from Linalg to LLVM.
void mlir::populateLinalgToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns) {
  patterns.add<RangeOpConversion, ReshapeOpConversion<ExpandShapeOp>,
               ReshapeOpConversion<CollapseShapeOp>, YieldOpConversion>(
      converter);

  // Populate the type conversions for the linalg types.
  converter.addConversion(
      [&](RangeType type) { return convertRangeType(type, converter); });
}

namespace {
struct ConvertLinalgToLLVMPass
    : public ConvertLinalgToLLVMBase<ConvertLinalgToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertLinalgToLLVMPass::runOnOperation() {
  auto module = getOperation();

  // Convert to the LLVM IR dialect using the converter defined above.
  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext());
  populateLinalgToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  target.addIllegalOp<RangeOp>();
  target.addLegalOp<ModuleOp, LLVM::DialectCastOp>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertLinalgToLLVMPass() {
  return std::make_unique<ConvertLinalgToLLVMPass>();
}
