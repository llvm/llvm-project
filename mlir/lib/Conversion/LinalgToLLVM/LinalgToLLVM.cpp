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
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
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
  OpBuilder &rewriter() { return ScopedContext::getBuilder(); }
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
    RangeOpOperandAdaptor adaptor(operands);
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
    ReshapeOpOperandAdaptor adaptor(operands);
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
    SliceOpOperandAdaptor adaptor(operands);
    BaseViewConversionHelper baseDesc(adaptor.view());

    auto sliceOp = cast<SliceOp>(op);
    auto memRefType = sliceOp.getBaseViewType();
    auto int64Ty = typeConverter.convertType(rewriter.getIntegerType(64))
                       .cast<LLVM::LLVMType>();

    BaseViewConversionHelper desc(
        typeConverter.convertType(sliceOp.getShapedType()));

    // TODO(ntv): extract sizes and emit asserts.
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

/// Conversion pattern that transforms a linalg.transpose op into:
///   1. A function entry `alloca` operation to allocate a ViewDescriptor.
///   2. A load of the ViewDescriptor from the pointer allocated in 1.
///   3. Updates to the ViewDescriptor to introduce the data ptr, offset, size
///      and stride. Size and stride are permutations of the original values.
///   4. A store of the resulting ViewDescriptor to the alloca'ed pointer.
/// The linalg.transpose op is replaced by the alloca'ed pointer.
class TransposeOpConversion : public ConvertToLLVMPattern {
public:
  explicit TransposeOpConversion(MLIRContext *context,
                                 LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(TransposeOp::getOperationName(), context,
                             lowering_) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Initialize the common boilerplate and alloca at the top of the FuncOp.
    edsc::ScopedContext context(rewriter, op->getLoc());
    TransposeOpOperandAdaptor adaptor(operands);
    BaseViewConversionHelper baseDesc(adaptor.view());

    auto transposeOp = cast<TransposeOp>(op);
    // No permutation, early exit.
    if (transposeOp.permutation().isIdentity())
      return rewriter.replaceOp(op, {baseDesc}), success();

    BaseViewConversionHelper desc(
        typeConverter.convertType(transposeOp.getShapedType()));

    // Copy the base and aligned pointers from the old descriptor to the new
    // one.
    desc.setAllocatedPtr(baseDesc.allocatedPtr());
    desc.setAlignedPtr(baseDesc.alignedPtr());

    // Copy the offset pointer from the old descriptor to the new one.
    desc.setOffset(baseDesc.offset());

    // Iterate over the dimensions and apply size/stride permutation.
    for (auto en : llvm::enumerate(transposeOp.permutation().getResults())) {
      int sourcePos = en.index();
      int targetPos = en.value().cast<AffineDimExpr>().getPosition();
      desc.setSize(targetPos, baseDesc.size(sourcePos));
      desc.setStride(targetPos, baseDesc.stride(sourcePos));
    }

    rewriter.replaceOp(op, {desc});
    return success();
  }
};

// YieldOp produces and LLVM::ReturnOp.
class YieldOpConversion : public ConvertToLLVMPattern {
public:
  explicit YieldOpConversion(MLIRContext *context, LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(YieldOp::getOperationName(), context, lowering_) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
    return success();
  }
};
} // namespace

template <typename LinalgOp>
static SmallVector<Type, 4> ExtractOperandTypes(Operation *op) {
  return SmallVector<Type, 4>{op->getOperandTypes()};
}

template <>
SmallVector<Type, 4> ExtractOperandTypes<IndexedGenericOp>(Operation *op) {
  auto ctx = op->getContext();
  auto indexedGenericOp = cast<IndexedGenericOp>(op);
  auto numLoops = indexedGenericOp.getNumLoops();

  SmallVector<Type, 4> result;
  result.reserve(numLoops + op->getNumOperands());
  for (unsigned i = 0; i < numLoops; ++i) {
    result.push_back(IndexType::get(ctx));
  }
  for (auto type : op->getOperandTypes()) {
    result.push_back(type);
  }
  return result;
}

// Get a SymbolRefAttr containing the library function name for the LinalgOp.
// If the library function does not exist, insert a declaration.
template <typename LinalgOp>
static FlatSymbolRefAttr getLibraryCallSymbolRef(Operation *op,
                                                 PatternRewriter &rewriter) {
  auto linalgOp = cast<LinalgOp>(op);
  auto fnName = linalgOp.getLibraryCallName();
  if (fnName.empty()) {
    op->emitWarning("No library call defined for: ") << *op;
    return {};
  }

  // fnName is a dynamic std::String, unique it via a SymbolRefAttr.
  FlatSymbolRefAttr fnNameAttr = rewriter.getSymbolRefAttr(fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnName)) {
    return fnNameAttr;
  }

  SmallVector<Type, 4> inputTypes(ExtractOperandTypes<LinalgOp>(op));
  assert(op->getNumResults() == 0 &&
         "Library call for linalg operation can be generated only for ops that "
         "have void return types");
  auto libFnType = FunctionType::get(inputTypes, {}, rewriter.getContext());

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  FuncOp funcOp =
      rewriter.create<FuncOp>(op->getLoc(), fnNameAttr.getValue(), libFnType,
                              ArrayRef<NamedAttribute>{});
  // Insert a function attribute that will trigger the emission of the
  // corresponding `_mlir_ciface_xxx` interface so that external libraries see
  // a normalized ABI. This interface is added during std to llvm conversion.
  funcOp.setAttr("llvm.emit_c_interface", UnitAttr::get(op->getContext()));
  return fnNameAttr;
}

namespace {

// LinalgOpConversion<LinalgOp> creates a new call to the
// `LinalgOp::getLibraryCallName()` function.
// The implementation of the function can be either in the same module or in an
// externally linked library.
template <typename LinalgOp>
class LinalgOpConversion : public OpRewritePattern<LinalgOp> {
public:
  using OpRewritePattern<LinalgOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    auto libraryCallName = getLibraryCallSymbolRef<LinalgOp>(op, rewriter);
    if (!libraryCallName)
      return failure();

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, libraryCallName.getValue(), ArrayRef<Type>{}, op.getOperands());
    return success();
  }
};

/// Conversion pattern specialization for CopyOp. This kicks in when both input
/// and output permutations are left unspecified or are the identity.
template <> class LinalgOpConversion<CopyOp> : public OpRewritePattern<CopyOp> {
public:
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto inputPerm = op.inputPermutation();
    if (inputPerm.hasValue() && !inputPerm->isIdentity())
      return failure();
    auto outputPerm = op.outputPermutation();
    if (outputPerm.hasValue() && !outputPerm->isIdentity())
      return failure();

    auto libraryCallName = getLibraryCallSymbolRef<CopyOp>(op, rewriter);
    if (!libraryCallName)
      return failure();

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, libraryCallName.getValue(), ArrayRef<Type>{}, op.getOperands());
    return success();
  }
};

/// Conversion pattern specialization for IndexedGenericOp.
template <>
class LinalgOpConversion<IndexedGenericOp>
    : public OpRewritePattern<IndexedGenericOp> {
public:
  using OpRewritePattern<IndexedGenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexedGenericOp op,
                                PatternRewriter &rewriter) const override {
    auto libraryCallName =
        getLibraryCallSymbolRef<IndexedGenericOp>(op, rewriter);
    if (!libraryCallName)
      return failure();

    // TODO(pifon, ntv): Use induction variables values instead of zeros, when
    // IndexedGenericOp is tiled.
    auto zero = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    auto indexedGenericOp = cast<IndexedGenericOp>(op);
    auto numLoops = indexedGenericOp.getNumLoops();
    SmallVector<Value, 4> operands;
    operands.reserve(numLoops + op.getNumOperands());
    for (unsigned i = 0; i < numLoops; ++i) {
      operands.push_back(zero);
    }
    for (auto operand : op.getOperands()) {
      operands.push_back(operand);
    }
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, libraryCallName.getValue(),
                                              ArrayRef<Type>{}, operands);
    return success();
  }
};

/// A non-conversion rewrite pattern kicks in to convert CopyOp with
/// permutations into a sequence of TransposeOp and permutation-free CopyOp.
/// This interplays together with TransposeOpConversion and
/// LinalgConversion<CopyOp> to create a path to the LLVM dialect.
class CopyTransposeConversion : public OpRewritePattern<CopyOp> {
public:
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op.input(), out = op.output();

    // If either inputPerm or outputPerm are non-identities, insert transposes.
    auto inputPerm = op.inputPermutation();
    if (inputPerm.hasValue() && !inputPerm->isIdentity())
      in = rewriter.create<linalg::TransposeOp>(op.getLoc(), in,
                                                AffineMapAttr::get(*inputPerm));
    auto outputPerm = op.outputPermutation();
    if (outputPerm.hasValue() && !outputPerm->isIdentity())
      out = rewriter.create<linalg::TransposeOp>(
          op.getLoc(), out, AffineMapAttr::get(*outputPerm));

    // If nothing was transposed, fail and let the conversion kick in.
    if (in == op.input() && out == op.output())
      return failure();

    rewriter.replaceOpWithNewOp<CopyOp>(op, in, out);
    return success();
  }
};

/// Populate the given list with patterns that convert from Linalg to Standard.
static void
populateLinalgToStandardConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx) {
  // TODO(ntv) ConvOp conversion needs to export a descriptor with relevant
  // attribute values such as kernel striding and dilation.
  // clang-format off
  patterns.insert<
      CopyTransposeConversion,
      LinalgOpConversion<ConvOp>,
      LinalgOpConversion<PoolingMaxOp>,
      LinalgOpConversion<PoolingMinOp>,
      LinalgOpConversion<PoolingSumOp>,
      LinalgOpConversion<CopyOp>,
      LinalgOpConversion<DotOp>,
      LinalgOpConversion<FillOp>,
      LinalgOpConversion<GenericOp>,
      LinalgOpConversion<IndexedGenericOp>,
      LinalgOpConversion<MatmulOp>,
      LinalgOpConversion<MatvecOp>>(ctx);
  // clang-format on
}

} // namespace

/// Populate the given list with patterns that convert from Linalg to LLVM.
void mlir::populateLinalgToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    MLIRContext *ctx) {
  patterns.insert<RangeOpConversion, ReshapeOpConversion, SliceOpConversion,
                  TransposeOpConversion, YieldOpConversion>(ctx, converter);

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
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateLinalgToStandardConversionPatterns(patterns, &getContext());
  populateLinalgToLLVMConversionPatterns(converter, patterns, &getContext());

  LLVMConversionTarget target(getContext());
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  if (failed(applyFullConversion(module, target, patterns, &converter)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertLinalgToLLVMPass() {
  return std::make_unique<ConvertLinalgToLLVMPass>();
}
