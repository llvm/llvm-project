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
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
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
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::LLVM;
using namespace mlir::linalg;

using llvm_add = ValueBuilder<LLVM::AddOp>;
using llvm_bitcast = ValueBuilder<LLVM::BitcastOp>;
using llvm_constant = ValueBuilder<LLVM::ConstantOp>;
using llvm_extractvalue = ValueBuilder<LLVM::ExtractValueOp>;
using llvm_gep = ValueBuilder<LLVM::GEPOp>;
using llvm_insertvalue = ValueBuilder<LLVM::InsertValueOp>;
using llvm_call = OperationBuilder<LLVM::CallOp>;
using llvm_icmp = ValueBuilder<LLVM::ICmpOp>;
using llvm_load = ValueBuilder<LLVM::LoadOp>;
using llvm_store = OperationBuilder<LLVM::StoreOp>;
using llvm_select = ValueBuilder<LLVM::SelectOp>;
using llvm_mul = ValueBuilder<LLVM::MulOp>;
using llvm_ptrtoint = ValueBuilder<LLVM::PtrToIntOp>;
using llvm_sub = ValueBuilder<LLVM::SubOp>;
using llvm_undef = ValueBuilder<LLVM::UndefOp>;
using llvm_urem = ValueBuilder<LLVM::URemOp>;
using llvm_alloca = ValueBuilder<LLVM::AllocaOp>;
using llvm_return = OperationBuilder<LLVM::ReturnOp>;

template <typename T>
static LLVMType getPtrToElementType(T containerType,
                                    LLVMTypeConverter &lowering) {
  return lowering.convertType(containerType.getElementType())
      .template cast<LLVMType>()
      .getPointerTo();
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
  auto int64Ty = converter.convertType(IntegerType::get(64, context))
                     .cast<LLVM::LLVMType>();
  return LLVMType::getStructTy(int64Ty, int64Ty, int64Ty);
}

namespace {
/// EDSC-compatible wrapper for MemRefDescriptor.
class BaseViewConversionHelper {
public:
  BaseViewConversionHelper(Type type)
      : d(MemRefDescriptor::undef(rewriter(), loc(), type)) {}

  BaseViewConversionHelper(Value v) : d(v) {}

  /// Wrappers around MemRefDescriptor that use EDSC builder and location.
  Value allocatedPtr() { return d.allocatedPtr(rewriter(), loc()); }
  void setAllocatedPtr(Value v) { d.setAllocatedPtr(rewriter(), loc(), v); }
  Value alignedPtr() { return d.alignedPtr(rewriter(), loc()); }
  void setAlignedPtr(Value v) { d.setAlignedPtr(rewriter(), loc(), v); }
  Value offset() { return d.offset(rewriter(), loc()); }
  void setOffset(Value v) { d.setOffset(rewriter(), loc(), v); }
  Value size(unsigned i) { return d.size(rewriter(), loc(), i); }
  void setSize(unsigned i, Value v) { d.setSize(rewriter(), loc(), i, v); }
  void setConstantSize(unsigned i, int64_t v) {
    d.setConstantSize(rewriter(), loc(), i, v);
  }
  Value stride(unsigned i) { return d.stride(rewriter(), loc(), i); }
  void setStride(unsigned i, Value v) { d.setStride(rewriter(), loc(), i, v); }
  void setConstantStride(unsigned i, int64_t v) {
    d.setConstantStride(rewriter(), loc(), i, v);
  }

  operator Value() { return d; }

private:
  OpBuilder &rewriter() { return ScopedContext::getBuilderRef(); }
  Location loc() { return ScopedContext::getLocation(); }

  MemRefDescriptor d;
};

// RangeOp creates a new range descriptor.
class RangeOpConversion : public ConvertToLLVMPattern {
public:
  explicit RangeOpConversion(MLIRContext *context, LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(RangeOp::getOperationName(), context, lowering_) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto rangeOp = cast<RangeOp>(op);
    auto rangeDescriptorTy =
        convertRangeType(rangeOp.getType().cast<RangeType>(), typeConverter);

    edsc::ScopedContext context(rewriter, op->getLoc());

    // Fill in an aggregate value of the descriptor.
    RangeOpAdaptor adaptor(operands);
    Value desc = llvm_undef(rangeDescriptorTy);
    desc = llvm_insertvalue(desc, adaptor.min(), rewriter.getI64ArrayAttr(0));
    desc = llvm_insertvalue(desc, adaptor.max(), rewriter.getI64ArrayAttr(1));
    desc = llvm_insertvalue(desc, adaptor.step(), rewriter.getI64ArrayAttr(2));
    rewriter.replaceOp(op, desc);
    return success();
  }
};

// ReshapeOp creates a new view descriptor of the proper rank.
// For now, the only conversion supported is for target MemRef with static sizes
// and strides.
class ReshapeOpConversion : public ConvertToLLVMPattern {
public:
  explicit ReshapeOpConversion(MLIRContext *context,
                               LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(ReshapeOp::getOperationName(), context,
                             lowering_) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto reshapeOp = cast<ReshapeOp>(op);
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

    edsc::ScopedContext context(rewriter, op->getLoc());
    ReshapeOpAdaptor adaptor(operands);
    BaseViewConversionHelper baseDesc(adaptor.src());
    BaseViewConversionHelper desc(typeConverter.convertType(dstType));
    desc.setAllocatedPtr(baseDesc.allocatedPtr());
    desc.setAlignedPtr(baseDesc.alignedPtr());
    desc.setOffset(baseDesc.offset());
    for (auto en : llvm::enumerate(dstType.getShape()))
      desc.setConstantSize(en.index(), en.value());
    for (auto en : llvm::enumerate(strides))
      desc.setConstantStride(en.index(), en.value());
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

/// Conversion pattern that transforms a linalg.slice op into:
///   1. An "undef" value for the ViewDescriptor.
///   2. Updates to the ViewDescriptor to introduce the data ptr, offset, size
///      and stride corresponding to the region of memory within the bounds of
///      the parent view.
/// The linalg.slice op is replaced by the alloca'ed pointer.
class SliceOpConversion : public ConvertToLLVMPattern {
public:
  explicit SliceOpConversion(MLIRContext *context, LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(SliceOp::getOperationName(), context, lowering_) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op->getLoc());
    SliceOpAdaptor adaptor(operands);
    BaseViewConversionHelper baseDesc(adaptor.view());

    auto sliceOp = cast<SliceOp>(op);
    auto memRefType = sliceOp.getBaseViewType();
    auto int64Ty = typeConverter.convertType(rewriter.getIntegerType(64))
                       .cast<LLVM::LLVMType>();

    BaseViewConversionHelper desc(
        typeConverter.convertType(sliceOp.getShapedType()));

    // TODO: extract sizes and emit asserts.
    SmallVector<Value, 4> strides(memRefType.getRank());
    for (int i = 0, e = memRefType.getRank(); i < e; ++i)
      strides[i] = baseDesc.stride(i);

    auto pos = [&rewriter](ArrayRef<int64_t> values) {
      return rewriter.getI64ArrayAttr(values);
    };

    // Compute base offset.
    Value baseOffset = baseDesc.offset();
    for (int i = 0, e = memRefType.getRank(); i < e; ++i) {
      Value indexing = adaptor.indexings()[i];
      Value min = indexing;
      if (sliceOp.indexing(i).getType().isa<RangeType>())
        min = llvm_extractvalue(int64Ty, indexing, pos(0));
      baseOffset = llvm_add(baseOffset, llvm_mul(min, strides[i]));
    }

    // Insert the base and aligned pointers.
    desc.setAllocatedPtr(baseDesc.allocatedPtr());
    desc.setAlignedPtr(baseDesc.alignedPtr());

    // Insert base offset.
    desc.setOffset(baseOffset);

    // Corner case, no sizes or strides: early return the descriptor.
    if (sliceOp.getShapedType().getRank() == 0)
      return rewriter.replaceOp(op, {desc}), success();

    Value zero = llvm_constant(
        int64Ty, rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    // Compute and insert view sizes (max - min along the range) and strides.
    // Skip the non-range operands as they will be projected away from the view.
    int numNewDims = 0;
    for (auto en : llvm::enumerate(sliceOp.indexings())) {
      Value indexing = en.value();
      if (indexing.getType().isa<RangeType>()) {
        int rank = en.index();
        Value rangeDescriptor = adaptor.indexings()[rank];
        Value min = llvm_extractvalue(int64Ty, rangeDescriptor, pos(0));
        Value max = llvm_extractvalue(int64Ty, rangeDescriptor, pos(1));
        Value step = llvm_extractvalue(int64Ty, rangeDescriptor, pos(2));
        Value baseSize = baseDesc.size(rank);

        // Bound upper by base view upper bound.
        max = llvm_select(llvm_icmp(ICmpPredicate::slt, max, baseSize), max,
                          baseSize);
        Value size = llvm_sub(max, min);
        // Bound lower by zero.
        size =
            llvm_select(llvm_icmp(ICmpPredicate::slt, size, zero), zero, size);
        Value stride = llvm_mul(strides[rank], step);
        desc.setSize(numNewDims, size);
        desc.setStride(numNewDims, stride);
        ++numNewDims;
      }
    }

    rewriter.replaceOp(op, {desc});
    return success();
  }
};

// YieldOp produces and LLVM::ReturnOp.
class YieldOpConversion : public ConvertToLLVMPattern {
public:
  explicit YieldOpConversion(MLIRContext *context, LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(linalg::YieldOp::getOperationName(), context,
                             lowering_) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
    return success();
  }
};
} // namespace

/// Populate the given list with patterns that convert from Linalg to LLVM.
void mlir::populateLinalgToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    MLIRContext *ctx) {
  patterns.insert<RangeOpConversion, ReshapeOpConversion, SliceOpConversion,
                  YieldOpConversion>(ctx, converter);

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
  OwningRewritePatternList patterns;
  LLVMTypeConverter converter(&getContext());
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateVectorToSCFConversionPatterns(patterns, &getContext());
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateLinalgToLLVMConversionPatterns(converter, patterns, &getContext());

  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertLinalgToLLVMPass() {
  return std::make_unique<ConvertLinalgToLLVMPass>();
}
