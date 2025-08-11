//===- VectorToLLVM.cpp - Conversion from Vector to the LLVM dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/PrintCallHelper.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Interfaces/MaskableOpInterface.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Casting.h"

#include <optional>

using namespace mlir;
using namespace mlir::vector;

// Helper that picks the proper sequence for inserting.
static Value insertOne(ConversionPatternRewriter &rewriter,
                       const LLVMTypeConverter &typeConverter, Location loc,
                       Value val1, Value val2, Type llvmType, int64_t rank,
                       int64_t pos) {
  assert(rank > 0 && "0-D vector corner case should have been handled already");
  if (rank == 1) {
    auto idxType = rewriter.getIndexType();
    auto constant = LLVM::ConstantOp::create(
        rewriter, loc, typeConverter.convertType(idxType),
        rewriter.getIntegerAttr(idxType, pos));
    return LLVM::InsertElementOp::create(rewriter, loc, llvmType, val1, val2,
                                         constant);
  }
  return LLVM::InsertValueOp::create(rewriter, loc, val1, val2, pos);
}

// Helper that picks the proper sequence for extracting.
static Value extractOne(ConversionPatternRewriter &rewriter,
                        const LLVMTypeConverter &typeConverter, Location loc,
                        Value val, Type llvmType, int64_t rank, int64_t pos) {
  if (rank <= 1) {
    auto idxType = rewriter.getIndexType();
    auto constant = LLVM::ConstantOp::create(
        rewriter, loc, typeConverter.convertType(idxType),
        rewriter.getIntegerAttr(idxType, pos));
    return LLVM::ExtractElementOp::create(rewriter, loc, llvmType, val,
                                          constant);
  }
  return LLVM::ExtractValueOp::create(rewriter, loc, val, pos);
}

// Helper that returns data layout alignment of a vector.
LogicalResult getVectorAlignment(const LLVMTypeConverter &typeConverter,
                                 VectorType vectorType, unsigned &align) {
  Type convertedVectorTy = typeConverter.convertType(vectorType);
  if (!convertedVectorTy)
    return failure();

  llvm::LLVMContext llvmContext;
  align = LLVM::TypeToLLVMIRTranslator(llvmContext)
              .getPreferredAlignment(convertedVectorTy,
                                     typeConverter.getDataLayout());

  return success();
}

// Helper that returns data layout alignment of a memref.
LogicalResult getMemRefAlignment(const LLVMTypeConverter &typeConverter,
                                 MemRefType memrefType, unsigned &align) {
  Type elementTy = typeConverter.convertType(memrefType.getElementType());
  if (!elementTy)
    return failure();

  // TODO: this should use the MLIR data layout when it becomes available and
  // stop depending on translation.
  llvm::LLVMContext llvmContext;
  align = LLVM::TypeToLLVMIRTranslator(llvmContext)
              .getPreferredAlignment(elementTy, typeConverter.getDataLayout());
  return success();
}

// Helper to resolve the alignment for vector load/store, gather and scatter
// ops. If useVectorAlignment is true, get the preferred alignment for the
// vector type in the operation. This option is used for hardware backends with
// vectorization. Otherwise, use the preferred alignment of the element type of
// the memref. Note that if you choose to use vector alignment, the shape of the
// vector type must be resolved before the ConvertVectorToLLVM pass is run.
LogicalResult getVectorToLLVMAlignment(const LLVMTypeConverter &typeConverter,
                                       VectorType vectorType,
                                       MemRefType memrefType, unsigned &align,
                                       bool useVectorAlignment) {
  if (useVectorAlignment) {
    if (failed(getVectorAlignment(typeConverter, vectorType, align))) {
      return failure();
    }
  } else {
    if (failed(getMemRefAlignment(typeConverter, memrefType, align))) {
      return failure();
    }
  }
  return success();
}

// Check if the last stride is non-unit and has a valid memory space.
static LogicalResult isMemRefTypeSupported(MemRefType memRefType,
                                           const LLVMTypeConverter &converter) {
  if (!memRefType.isLastDimUnitStride())
    return failure();
  if (failed(converter.getMemRefAddressSpace(memRefType)))
    return failure();
  return success();
}

// Add an index vector component to a base pointer.
static Value getIndexedPtrs(ConversionPatternRewriter &rewriter, Location loc,
                            const LLVMTypeConverter &typeConverter,
                            MemRefType memRefType, Value llvmMemref, Value base,
                            Value index, VectorType vectorType) {
  assert(succeeded(isMemRefTypeSupported(memRefType, typeConverter)) &&
         "unsupported memref type");
  assert(vectorType.getRank() == 1 && "expected a 1-d vector type");
  auto pType = MemRefDescriptor(llvmMemref).getElementPtrType();
  auto ptrsType =
      LLVM::getVectorType(pType, vectorType.getDimSize(0),
                          /*isScalable=*/vectorType.getScalableDims()[0]);
  return LLVM::GEPOp::create(
      rewriter, loc, ptrsType,
      typeConverter.convertType(memRefType.getElementType()), base, index);
}

/// Convert `foldResult` into a Value. Integer attribute is converted to
/// an LLVM constant op.
static Value getAsLLVMValue(OpBuilder &builder, Location loc,
                            OpFoldResult foldResult) {
  if (auto attr = dyn_cast<Attribute>(foldResult)) {
    auto intAttr = cast<IntegerAttr>(attr);
    return LLVM::ConstantOp::create(builder, loc, intAttr).getResult();
  }

  return cast<Value>(foldResult);
}

namespace {

/// Trivial Vector to LLVM conversions
using VectorScaleOpConversion =
    OneToOneConvertToLLVMPattern<vector::VectorScaleOp, LLVM::vscale>;

/// Conversion pattern for a vector.bitcast.
class VectorBitCastOpConversion
    : public ConvertOpToLLVMPattern<vector::BitCastOp> {
public:
  using ConvertOpToLLVMPattern<vector::BitCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::BitCastOp bitCastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only 0-D and 1-D vectors can be lowered to LLVM.
    VectorType resultTy = bitCastOp.getResultVectorType();
    if (resultTy.getRank() > 1)
      return failure();
    Type newResultTy = typeConverter->convertType(resultTy);
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(bitCastOp, newResultTy,
                                                 adaptor.getOperands()[0]);
    return success();
  }
};

/// Overloaded utility that replaces a vector.load, vector.store,
/// vector.maskedload and vector.maskedstore with their respective LLVM
/// couterparts.
static void replaceLoadOrStoreOp(vector::LoadOp loadOp,
                                 vector::LoadOpAdaptor adaptor,
                                 VectorType vectorTy, Value ptr, unsigned align,
                                 ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(loadOp, vectorTy, ptr, align,
                                            /*volatile_=*/false,
                                            loadOp.getNontemporal());
}

static void replaceLoadOrStoreOp(vector::MaskedLoadOp loadOp,
                                 vector::MaskedLoadOpAdaptor adaptor,
                                 VectorType vectorTy, Value ptr, unsigned align,
                                 ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<LLVM::MaskedLoadOp>(
      loadOp, vectorTy, ptr, adaptor.getMask(), adaptor.getPassThru(), align);
}

static void replaceLoadOrStoreOp(vector::StoreOp storeOp,
                                 vector::StoreOpAdaptor adaptor,
                                 VectorType vectorTy, Value ptr, unsigned align,
                                 ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, adaptor.getValueToStore(),
                                             ptr, align, /*volatile_=*/false,
                                             storeOp.getNontemporal());
}

static void replaceLoadOrStoreOp(vector::MaskedStoreOp storeOp,
                                 vector::MaskedStoreOpAdaptor adaptor,
                                 VectorType vectorTy, Value ptr, unsigned align,
                                 ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<LLVM::MaskedStoreOp>(
      storeOp, adaptor.getValueToStore(), ptr, adaptor.getMask(), align);
}

/// Conversion pattern for a vector.load, vector.store, vector.maskedload, and
/// vector.maskedstore.
template <class LoadOrStoreOp>
class VectorLoadStoreConversion : public ConvertOpToLLVMPattern<LoadOrStoreOp> {
public:
  explicit VectorLoadStoreConversion(const LLVMTypeConverter &typeConv,
                                     bool useVectorAlign)
      : ConvertOpToLLVMPattern<LoadOrStoreOp>(typeConv),
        useVectorAlignment(useVectorAlign) {}
  using ConvertOpToLLVMPattern<LoadOrStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LoadOrStoreOp loadOrStoreOp,
                  typename LoadOrStoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only 1-D vectors can be lowered to LLVM.
    VectorType vectorTy = loadOrStoreOp.getVectorType();
    if (vectorTy.getRank() > 1)
      return failure();

    auto loc = loadOrStoreOp->getLoc();
    MemRefType memRefTy = loadOrStoreOp.getMemRefType();

    // Resolve alignment.
    unsigned align = loadOrStoreOp.getAlignment().value_or(0);
    if (!align &&
        failed(getVectorToLLVMAlignment(*this->getTypeConverter(), vectorTy,
                                        memRefTy, align, useVectorAlignment)))
      return rewriter.notifyMatchFailure(loadOrStoreOp,
                                         "could not resolve alignment");

    // Resolve address.
    auto vtype = cast<VectorType>(
        this->typeConverter->convertType(loadOrStoreOp.getVectorType()));
    Value dataPtr = this->getStridedElementPtr(
        rewriter, loc, memRefTy, adaptor.getBase(), adaptor.getIndices());
    replaceLoadOrStoreOp(loadOrStoreOp, adaptor, vtype, dataPtr, align,
                         rewriter);
    return success();
  }

private:
  // If true, use the preferred alignment of the vector type.
  // If false, use the preferred alignment of the element type
  // of the memref. This flag is intended for use with hardware
  // backends that require alignment of vector operations.
  const bool useVectorAlignment;
};

/// Conversion pattern for a vector.gather.
class VectorGatherOpConversion
    : public ConvertOpToLLVMPattern<vector::GatherOp> {
public:
  explicit VectorGatherOpConversion(const LLVMTypeConverter &typeConv,
                                    bool useVectorAlign)
      : ConvertOpToLLVMPattern<vector::GatherOp>(typeConv),
        useVectorAlignment(useVectorAlign) {}
  using ConvertOpToLLVMPattern<vector::GatherOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::GatherOp gather, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = gather->getLoc();
    MemRefType memRefType = dyn_cast<MemRefType>(gather.getBaseType());
    assert(memRefType && "The base should be bufferized");

    if (failed(isMemRefTypeSupported(memRefType, *this->getTypeConverter())))
      return rewriter.notifyMatchFailure(gather, "memref type not supported");

    VectorType vType = gather.getVectorType();
    if (vType.getRank() > 1) {
      return rewriter.notifyMatchFailure(
          gather, "only 1-D vectors can be lowered to LLVM");
    }

    // Resolve alignment.
    unsigned align;
    if (failed(getVectorToLLVMAlignment(*this->getTypeConverter(), vType,
                                        memRefType, align, useVectorAlignment)))
      return rewriter.notifyMatchFailure(gather, "could not resolve alignment");

    // Resolve address.
    Value ptr = getStridedElementPtr(rewriter, loc, memRefType,
                                     adaptor.getBase(), adaptor.getIndices());
    Value base = adaptor.getBase();
    Value ptrs =
        getIndexedPtrs(rewriter, loc, *this->getTypeConverter(), memRefType,
                       base, ptr, adaptor.getIndexVec(), vType);

    // Replace with the gather intrinsic.
    rewriter.replaceOpWithNewOp<LLVM::masked_gather>(
        gather, typeConverter->convertType(vType), ptrs, adaptor.getMask(),
        adaptor.getPassThru(), rewriter.getI32IntegerAttr(align));
    return success();
  }

private:
  // If true, use the preferred alignment of the vector type.
  // If false, use the preferred alignment of the element type
  // of the memref. This flag is intended for use with hardware
  // backends that require alignment of vector operations.
  const bool useVectorAlignment;
};

/// Conversion pattern for a vector.scatter.
class VectorScatterOpConversion
    : public ConvertOpToLLVMPattern<vector::ScatterOp> {
public:
  explicit VectorScatterOpConversion(const LLVMTypeConverter &typeConv,
                                     bool useVectorAlign)
      : ConvertOpToLLVMPattern<vector::ScatterOp>(typeConv),
        useVectorAlignment(useVectorAlign) {}

  using ConvertOpToLLVMPattern<vector::ScatterOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::ScatterOp scatter, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = scatter->getLoc();
    MemRefType memRefType = scatter.getMemRefType();

    if (failed(isMemRefTypeSupported(memRefType, *this->getTypeConverter())))
      return rewriter.notifyMatchFailure(scatter, "memref type not supported");

    VectorType vType = scatter.getVectorType();
    if (vType.getRank() > 1) {
      return rewriter.notifyMatchFailure(
          scatter, "only 1-D vectors can be lowered to LLVM");
    }

    // Resolve alignment.
    unsigned align;
    if (failed(getVectorToLLVMAlignment(*this->getTypeConverter(), vType,
                                        memRefType, align, useVectorAlignment)))
      return rewriter.notifyMatchFailure(scatter,
                                         "could not resolve alignment");

    // Resolve address.
    Value ptr = getStridedElementPtr(rewriter, loc, memRefType,
                                     adaptor.getBase(), adaptor.getIndices());
    Value ptrs =
        getIndexedPtrs(rewriter, loc, *this->getTypeConverter(), memRefType,
                       adaptor.getBase(), ptr, adaptor.getIndexVec(), vType);

    // Replace with the scatter intrinsic.
    rewriter.replaceOpWithNewOp<LLVM::masked_scatter>(
        scatter, adaptor.getValueToStore(), ptrs, adaptor.getMask(),
        rewriter.getI32IntegerAttr(align));
    return success();
  }

private:
  // If true, use the preferred alignment of the vector type.
  // If false, use the preferred alignment of the element type
  // of the memref. This flag is intended for use with hardware
  // backends that require alignment of vector operations.
  const bool useVectorAlignment;
};

/// Conversion pattern for a vector.expandload.
class VectorExpandLoadOpConversion
    : public ConvertOpToLLVMPattern<vector::ExpandLoadOp> {
public:
  using ConvertOpToLLVMPattern<vector::ExpandLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::ExpandLoadOp expand, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = expand->getLoc();
    MemRefType memRefType = expand.getMemRefType();

    // Resolve address.
    auto vtype = typeConverter->convertType(expand.getVectorType());
    Value ptr = getStridedElementPtr(rewriter, loc, memRefType,
                                     adaptor.getBase(), adaptor.getIndices());

    rewriter.replaceOpWithNewOp<LLVM::masked_expandload>(
        expand, vtype, ptr, adaptor.getMask(), adaptor.getPassThru());
    return success();
  }
};

/// Conversion pattern for a vector.compressstore.
class VectorCompressStoreOpConversion
    : public ConvertOpToLLVMPattern<vector::CompressStoreOp> {
public:
  using ConvertOpToLLVMPattern<vector::CompressStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::CompressStoreOp compress, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = compress->getLoc();
    MemRefType memRefType = compress.getMemRefType();

    // Resolve address.
    Value ptr = getStridedElementPtr(rewriter, loc, memRefType,
                                     adaptor.getBase(), adaptor.getIndices());

    rewriter.replaceOpWithNewOp<LLVM::masked_compressstore>(
        compress, adaptor.getValueToStore(), ptr, adaptor.getMask());
    return success();
  }
};

/// Reduction neutral classes for overloading.
class ReductionNeutralZero {};
class ReductionNeutralIntOne {};
class ReductionNeutralFPOne {};
class ReductionNeutralAllOnes {};
class ReductionNeutralSIntMin {};
class ReductionNeutralUIntMin {};
class ReductionNeutralSIntMax {};
class ReductionNeutralUIntMax {};
class ReductionNeutralFPMin {};
class ReductionNeutralFPMax {};

/// Create the reduction neutral zero value.
static Value createReductionNeutralValue(ReductionNeutralZero neutral,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmType) {
  return LLVM::ConstantOp::create(rewriter, loc, llvmType,
                                  rewriter.getZeroAttr(llvmType));
}

/// Create the reduction neutral integer one value.
static Value createReductionNeutralValue(ReductionNeutralIntOne neutral,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmType) {
  return LLVM::ConstantOp::create(rewriter, loc, llvmType,
                                  rewriter.getIntegerAttr(llvmType, 1));
}

/// Create the reduction neutral fp one value.
static Value createReductionNeutralValue(ReductionNeutralFPOne neutral,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmType) {
  return LLVM::ConstantOp::create(rewriter, loc, llvmType,
                                  rewriter.getFloatAttr(llvmType, 1.0));
}

/// Create the reduction neutral all-ones value.
static Value createReductionNeutralValue(ReductionNeutralAllOnes neutral,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmType) {
  return LLVM::ConstantOp::create(
      rewriter, loc, llvmType,
      rewriter.getIntegerAttr(
          llvmType, llvm::APInt::getAllOnes(llvmType.getIntOrFloatBitWidth())));
}

/// Create the reduction neutral signed int minimum value.
static Value createReductionNeutralValue(ReductionNeutralSIntMin neutral,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmType) {
  return LLVM::ConstantOp::create(
      rewriter, loc, llvmType,
      rewriter.getIntegerAttr(llvmType, llvm::APInt::getSignedMinValue(
                                            llvmType.getIntOrFloatBitWidth())));
}

/// Create the reduction neutral unsigned int minimum value.
static Value createReductionNeutralValue(ReductionNeutralUIntMin neutral,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmType) {
  return LLVM::ConstantOp::create(
      rewriter, loc, llvmType,
      rewriter.getIntegerAttr(llvmType, llvm::APInt::getMinValue(
                                            llvmType.getIntOrFloatBitWidth())));
}

/// Create the reduction neutral signed int maximum value.
static Value createReductionNeutralValue(ReductionNeutralSIntMax neutral,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmType) {
  return LLVM::ConstantOp::create(
      rewriter, loc, llvmType,
      rewriter.getIntegerAttr(llvmType, llvm::APInt::getSignedMaxValue(
                                            llvmType.getIntOrFloatBitWidth())));
}

/// Create the reduction neutral unsigned int maximum value.
static Value createReductionNeutralValue(ReductionNeutralUIntMax neutral,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmType) {
  return LLVM::ConstantOp::create(
      rewriter, loc, llvmType,
      rewriter.getIntegerAttr(llvmType, llvm::APInt::getMaxValue(
                                            llvmType.getIntOrFloatBitWidth())));
}

/// Create the reduction neutral fp minimum value.
static Value createReductionNeutralValue(ReductionNeutralFPMin neutral,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmType) {
  auto floatType = cast<FloatType>(llvmType);
  return LLVM::ConstantOp::create(
      rewriter, loc, llvmType,
      rewriter.getFloatAttr(
          llvmType, llvm::APFloat::getQNaN(floatType.getFloatSemantics(),
                                           /*Negative=*/false)));
}

/// Create the reduction neutral fp maximum value.
static Value createReductionNeutralValue(ReductionNeutralFPMax neutral,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmType) {
  auto floatType = cast<FloatType>(llvmType);
  return LLVM::ConstantOp::create(
      rewriter, loc, llvmType,
      rewriter.getFloatAttr(
          llvmType, llvm::APFloat::getQNaN(floatType.getFloatSemantics(),
                                           /*Negative=*/true)));
}

/// Returns `accumulator` if it has a valid value. Otherwise, creates and
/// returns a new accumulator value using `ReductionNeutral`.
template <class ReductionNeutral>
static Value getOrCreateAccumulator(ConversionPatternRewriter &rewriter,
                                    Location loc, Type llvmType,
                                    Value accumulator) {
  if (accumulator)
    return accumulator;

  return createReductionNeutralValue(ReductionNeutral(), rewriter, loc,
                                     llvmType);
}

/// Creates a value with the 1-D vector shape provided in `llvmType`.
/// This is used as effective vector length by some intrinsics supporting
/// dynamic vector lengths at runtime.
static Value createVectorLengthValue(ConversionPatternRewriter &rewriter,
                                     Location loc, Type llvmType) {
  VectorType vType = cast<VectorType>(llvmType);
  auto vShape = vType.getShape();
  assert(vShape.size() == 1 && "Unexpected multi-dim vector type");

  Value baseVecLength = LLVM::ConstantOp::create(
      rewriter, loc, rewriter.getI32Type(),
      rewriter.getIntegerAttr(rewriter.getI32Type(), vShape[0]));

  if (!vType.getScalableDims()[0])
    return baseVecLength;

  // For a scalable vector type, create and return `vScale * baseVecLength`.
  Value vScale = vector::VectorScaleOp::create(rewriter, loc);
  vScale =
      arith::IndexCastOp::create(rewriter, loc, rewriter.getI32Type(), vScale);
  Value scalableVecLength =
      arith::MulIOp::create(rewriter, loc, baseVecLength, vScale);
  return scalableVecLength;
}

/// Helper method to lower a `vector.reduction` op that performs an arithmetic
/// operation like add,mul, etc.. `VectorOp` is the LLVM vector intrinsic to use
/// and `ScalarOp` is the scalar operation used to add the accumulation value if
/// non-null.
template <class LLVMRedIntrinOp, class ScalarOp>
static Value createIntegerReductionArithmeticOpLowering(
    ConversionPatternRewriter &rewriter, Location loc, Type llvmType,
    Value vectorOperand, Value accumulator) {

  Value result =
      LLVMRedIntrinOp::create(rewriter, loc, llvmType, vectorOperand);

  if (accumulator)
    result = ScalarOp::create(rewriter, loc, accumulator, result);
  return result;
}

/// Helper method to lower a `vector.reduction` operation that performs
/// a comparison operation like `min`/`max`. `VectorOp` is the LLVM vector
/// intrinsic to use and `predicate` is the predicate to use to compare+combine
/// the accumulator value if non-null.
template <class LLVMRedIntrinOp>
static Value createIntegerReductionComparisonOpLowering(
    ConversionPatternRewriter &rewriter, Location loc, Type llvmType,
    Value vectorOperand, Value accumulator, LLVM::ICmpPredicate predicate) {
  Value result =
      LLVMRedIntrinOp::create(rewriter, loc, llvmType, vectorOperand);
  if (accumulator) {
    Value cmp =
        LLVM::ICmpOp::create(rewriter, loc, predicate, accumulator, result);
    result = LLVM::SelectOp::create(rewriter, loc, cmp, accumulator, result);
  }
  return result;
}

namespace {
template <typename Source>
struct VectorToScalarMapper;
template <>
struct VectorToScalarMapper<LLVM::vector_reduce_fmaximum> {
  using Type = LLVM::MaximumOp;
};
template <>
struct VectorToScalarMapper<LLVM::vector_reduce_fminimum> {
  using Type = LLVM::MinimumOp;
};
template <>
struct VectorToScalarMapper<LLVM::vector_reduce_fmax> {
  using Type = LLVM::MaxNumOp;
};
template <>
struct VectorToScalarMapper<LLVM::vector_reduce_fmin> {
  using Type = LLVM::MinNumOp;
};
} // namespace

template <class LLVMRedIntrinOp>
static Value createFPReductionComparisonOpLowering(
    ConversionPatternRewriter &rewriter, Location loc, Type llvmType,
    Value vectorOperand, Value accumulator, LLVM::FastmathFlagsAttr fmf) {
  Value result =
      LLVMRedIntrinOp::create(rewriter, loc, llvmType, vectorOperand, fmf);

  if (accumulator) {
    result = VectorToScalarMapper<LLVMRedIntrinOp>::Type::create(
        rewriter, loc, result, accumulator);
  }

  return result;
}

/// Reduction neutral classes for overloading
class MaskNeutralFMaximum {};
class MaskNeutralFMinimum {};

/// Get the mask neutral floating point maximum value
static llvm::APFloat
getMaskNeutralValue(MaskNeutralFMaximum,
                    const llvm::fltSemantics &floatSemantics) {
  return llvm::APFloat::getSmallest(floatSemantics, /*Negative=*/true);
}
/// Get the mask neutral floating point minimum value
static llvm::APFloat
getMaskNeutralValue(MaskNeutralFMinimum,
                    const llvm::fltSemantics &floatSemantics) {
  return llvm::APFloat::getLargest(floatSemantics, /*Negative=*/false);
}

/// Create the mask neutral floating point MLIR vector constant
template <typename MaskNeutral>
static Value createMaskNeutralValue(ConversionPatternRewriter &rewriter,
                                    Location loc, Type llvmType,
                                    Type vectorType) {
  const auto &floatSemantics = cast<FloatType>(llvmType).getFloatSemantics();
  auto value = getMaskNeutralValue(MaskNeutral{}, floatSemantics);
  auto denseValue = DenseElementsAttr::get(cast<ShapedType>(vectorType), value);
  return LLVM::ConstantOp::create(rewriter, loc, vectorType, denseValue);
}

/// Lowers masked `fmaximum` and `fminimum` reductions using the non-masked
/// intrinsics. It is a workaround to overcome the lack of masked intrinsics for
/// `fmaximum`/`fminimum`.
/// More information: https://github.com/llvm/llvm-project/issues/64940
template <class LLVMRedIntrinOp, class MaskNeutral>
static Value
lowerMaskedReductionWithRegular(ConversionPatternRewriter &rewriter,
                                Location loc, Type llvmType,
                                Value vectorOperand, Value accumulator,
                                Value mask, LLVM::FastmathFlagsAttr fmf) {
  const Value vectorMaskNeutral = createMaskNeutralValue<MaskNeutral>(
      rewriter, loc, llvmType, vectorOperand.getType());
  const Value selectedVectorByMask = LLVM::SelectOp::create(
      rewriter, loc, mask, vectorOperand, vectorMaskNeutral);
  return createFPReductionComparisonOpLowering<LLVMRedIntrinOp>(
      rewriter, loc, llvmType, selectedVectorByMask, accumulator, fmf);
}

template <class LLVMRedIntrinOp, class ReductionNeutral>
static Value
lowerReductionWithStartValue(ConversionPatternRewriter &rewriter, Location loc,
                             Type llvmType, Value vectorOperand,
                             Value accumulator, LLVM::FastmathFlagsAttr fmf) {
  accumulator = getOrCreateAccumulator<ReductionNeutral>(rewriter, loc,
                                                         llvmType, accumulator);
  return LLVMRedIntrinOp::create(rewriter, loc, llvmType,
                                 /*startValue=*/accumulator, vectorOperand,
                                 fmf);
}

/// Overloaded methods to lower a *predicated* reduction to an llvm intrinsic
/// that requires a start value. This start value format spans across fp
/// reductions without mask and all the masked reduction intrinsics.
template <class LLVMVPRedIntrinOp, class ReductionNeutral>
static Value
lowerPredicatedReductionWithStartValue(ConversionPatternRewriter &rewriter,
                                       Location loc, Type llvmType,
                                       Value vectorOperand, Value accumulator) {
  accumulator = getOrCreateAccumulator<ReductionNeutral>(rewriter, loc,
                                                         llvmType, accumulator);
  return LLVMVPRedIntrinOp::create(rewriter, loc, llvmType,
                                   /*startValue=*/accumulator, vectorOperand);
}

template <class LLVMVPRedIntrinOp, class ReductionNeutral>
static Value lowerPredicatedReductionWithStartValue(
    ConversionPatternRewriter &rewriter, Location loc, Type llvmType,
    Value vectorOperand, Value accumulator, Value mask) {
  accumulator = getOrCreateAccumulator<ReductionNeutral>(rewriter, loc,
                                                         llvmType, accumulator);
  Value vectorLength =
      createVectorLengthValue(rewriter, loc, vectorOperand.getType());
  return LLVMVPRedIntrinOp::create(rewriter, loc, llvmType,
                                   /*startValue=*/accumulator, vectorOperand,
                                   mask, vectorLength);
}

template <class LLVMIntVPRedIntrinOp, class IntReductionNeutral,
          class LLVMFPVPRedIntrinOp, class FPReductionNeutral>
static Value lowerPredicatedReductionWithStartValue(
    ConversionPatternRewriter &rewriter, Location loc, Type llvmType,
    Value vectorOperand, Value accumulator, Value mask) {
  if (llvmType.isIntOrIndex())
    return lowerPredicatedReductionWithStartValue<LLVMIntVPRedIntrinOp,
                                                  IntReductionNeutral>(
        rewriter, loc, llvmType, vectorOperand, accumulator, mask);

  // FP dispatch.
  return lowerPredicatedReductionWithStartValue<LLVMFPVPRedIntrinOp,
                                                FPReductionNeutral>(
      rewriter, loc, llvmType, vectorOperand, accumulator, mask);
}

/// Conversion pattern for all vector reductions.
class VectorReductionOpConversion
    : public ConvertOpToLLVMPattern<vector::ReductionOp> {
public:
  explicit VectorReductionOpConversion(const LLVMTypeConverter &typeConv,
                                       bool reassociateFPRed)
      : ConvertOpToLLVMPattern<vector::ReductionOp>(typeConv),
        reassociateFPReductions(reassociateFPRed) {}

  LogicalResult
  matchAndRewrite(vector::ReductionOp reductionOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kind = reductionOp.getKind();
    Type eltType = reductionOp.getDest().getType();
    Type llvmType = typeConverter->convertType(eltType);
    Value operand = adaptor.getVector();
    Value acc = adaptor.getAcc();
    Location loc = reductionOp.getLoc();

    if (eltType.isIntOrIndex()) {
      // Integer reductions: add/mul/min/max/and/or/xor.
      Value result;
      switch (kind) {
      case vector::CombiningKind::ADD:
        result =
            createIntegerReductionArithmeticOpLowering<LLVM::vector_reduce_add,
                                                       LLVM::AddOp>(
                rewriter, loc, llvmType, operand, acc);
        break;
      case vector::CombiningKind::MUL:
        result =
            createIntegerReductionArithmeticOpLowering<LLVM::vector_reduce_mul,
                                                       LLVM::MulOp>(
                rewriter, loc, llvmType, operand, acc);
        break;
      case vector::CombiningKind::MINUI:
        result = createIntegerReductionComparisonOpLowering<
            LLVM::vector_reduce_umin>(rewriter, loc, llvmType, operand, acc,
                                      LLVM::ICmpPredicate::ule);
        break;
      case vector::CombiningKind::MINSI:
        result = createIntegerReductionComparisonOpLowering<
            LLVM::vector_reduce_smin>(rewriter, loc, llvmType, operand, acc,
                                      LLVM::ICmpPredicate::sle);
        break;
      case vector::CombiningKind::MAXUI:
        result = createIntegerReductionComparisonOpLowering<
            LLVM::vector_reduce_umax>(rewriter, loc, llvmType, operand, acc,
                                      LLVM::ICmpPredicate::uge);
        break;
      case vector::CombiningKind::MAXSI:
        result = createIntegerReductionComparisonOpLowering<
            LLVM::vector_reduce_smax>(rewriter, loc, llvmType, operand, acc,
                                      LLVM::ICmpPredicate::sge);
        break;
      case vector::CombiningKind::AND:
        result =
            createIntegerReductionArithmeticOpLowering<LLVM::vector_reduce_and,
                                                       LLVM::AndOp>(
                rewriter, loc, llvmType, operand, acc);
        break;
      case vector::CombiningKind::OR:
        result =
            createIntegerReductionArithmeticOpLowering<LLVM::vector_reduce_or,
                                                       LLVM::OrOp>(
                rewriter, loc, llvmType, operand, acc);
        break;
      case vector::CombiningKind::XOR:
        result =
            createIntegerReductionArithmeticOpLowering<LLVM::vector_reduce_xor,
                                                       LLVM::XOrOp>(
                rewriter, loc, llvmType, operand, acc);
        break;
      default:
        return failure();
      }
      rewriter.replaceOp(reductionOp, result);

      return success();
    }

    if (!isa<FloatType>(eltType))
      return failure();

    arith::FastMathFlagsAttr fMFAttr = reductionOp.getFastMathFlagsAttr();
    LLVM::FastmathFlagsAttr fmf = LLVM::FastmathFlagsAttr::get(
        reductionOp.getContext(),
        convertArithFastMathFlagsToLLVM(fMFAttr.getValue()));
    fmf = LLVM::FastmathFlagsAttr::get(
        reductionOp.getContext(),
        fmf.getValue() | (reassociateFPReductions ? LLVM::FastmathFlags::reassoc
                                                  : LLVM::FastmathFlags::none));

    // Floating-point reductions: add/mul/min/max
    Value result;
    if (kind == vector::CombiningKind::ADD) {
      result = lowerReductionWithStartValue<LLVM::vector_reduce_fadd,
                                            ReductionNeutralZero>(
          rewriter, loc, llvmType, operand, acc, fmf);
    } else if (kind == vector::CombiningKind::MUL) {
      result = lowerReductionWithStartValue<LLVM::vector_reduce_fmul,
                                            ReductionNeutralFPOne>(
          rewriter, loc, llvmType, operand, acc, fmf);
    } else if (kind == vector::CombiningKind::MINIMUMF) {
      result =
          createFPReductionComparisonOpLowering<LLVM::vector_reduce_fminimum>(
              rewriter, loc, llvmType, operand, acc, fmf);
    } else if (kind == vector::CombiningKind::MAXIMUMF) {
      result =
          createFPReductionComparisonOpLowering<LLVM::vector_reduce_fmaximum>(
              rewriter, loc, llvmType, operand, acc, fmf);
    } else if (kind == vector::CombiningKind::MINNUMF) {
      result = createFPReductionComparisonOpLowering<LLVM::vector_reduce_fmin>(
          rewriter, loc, llvmType, operand, acc, fmf);
    } else if (kind == vector::CombiningKind::MAXNUMF) {
      result = createFPReductionComparisonOpLowering<LLVM::vector_reduce_fmax>(
          rewriter, loc, llvmType, operand, acc, fmf);
    } else {
      return failure();
    }

    rewriter.replaceOp(reductionOp, result);
    return success();
  }

private:
  const bool reassociateFPReductions;
};

/// Base class to convert a `vector.mask` operation while matching traits
/// of the maskable operation nested inside. A `VectorMaskOpConversionBase`
/// instance matches against a `vector.mask` operation. The `matchAndRewrite`
/// method performs a second match against the maskable operation `MaskedOp`.
/// Finally, it invokes the virtual method `matchAndRewriteMaskableOp` to be
/// implemented by the concrete conversion classes. This method can match
/// against specific traits of the `vector.mask` and the maskable operation. It
/// must replace the `vector.mask` operation.
template <class MaskedOp>
class VectorMaskOpConversionBase
    : public ConvertOpToLLVMPattern<vector::MaskOp> {
public:
  using ConvertOpToLLVMPattern<vector::MaskOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::MaskOp maskOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Match against the maskable operation kind.
    auto maskedOp = llvm::dyn_cast_or_null<MaskedOp>(maskOp.getMaskableOp());
    if (!maskedOp)
      return failure();
    return matchAndRewriteMaskableOp(maskOp, maskedOp, rewriter);
  }

protected:
  virtual LogicalResult
  matchAndRewriteMaskableOp(vector::MaskOp maskOp,
                            vector::MaskableOpInterface maskableOp,
                            ConversionPatternRewriter &rewriter) const = 0;
};

class MaskedReductionOpConversion
    : public VectorMaskOpConversionBase<vector::ReductionOp> {

public:
  using VectorMaskOpConversionBase<
      vector::ReductionOp>::VectorMaskOpConversionBase;

  LogicalResult matchAndRewriteMaskableOp(
      vector::MaskOp maskOp, MaskableOpInterface maskableOp,
      ConversionPatternRewriter &rewriter) const override {
    auto reductionOp = cast<ReductionOp>(maskableOp.getOperation());
    auto kind = reductionOp.getKind();
    Type eltType = reductionOp.getDest().getType();
    Type llvmType = typeConverter->convertType(eltType);
    Value operand = reductionOp.getVector();
    Value acc = reductionOp.getAcc();
    Location loc = reductionOp.getLoc();

    arith::FastMathFlagsAttr fMFAttr = reductionOp.getFastMathFlagsAttr();
    LLVM::FastmathFlagsAttr fmf = LLVM::FastmathFlagsAttr::get(
        reductionOp.getContext(),
        convertArithFastMathFlagsToLLVM(fMFAttr.getValue()));

    Value result;
    switch (kind) {
    case vector::CombiningKind::ADD:
      result = lowerPredicatedReductionWithStartValue<
          LLVM::VPReduceAddOp, ReductionNeutralZero, LLVM::VPReduceFAddOp,
          ReductionNeutralZero>(rewriter, loc, llvmType, operand, acc,
                                maskOp.getMask());
      break;
    case vector::CombiningKind::MUL:
      result = lowerPredicatedReductionWithStartValue<
          LLVM::VPReduceMulOp, ReductionNeutralIntOne, LLVM::VPReduceFMulOp,
          ReductionNeutralFPOne>(rewriter, loc, llvmType, operand, acc,
                                 maskOp.getMask());
      break;
    case vector::CombiningKind::MINUI:
      result = lowerPredicatedReductionWithStartValue<LLVM::VPReduceUMinOp,
                                                      ReductionNeutralUIntMax>(
          rewriter, loc, llvmType, operand, acc, maskOp.getMask());
      break;
    case vector::CombiningKind::MINSI:
      result = lowerPredicatedReductionWithStartValue<LLVM::VPReduceSMinOp,
                                                      ReductionNeutralSIntMax>(
          rewriter, loc, llvmType, operand, acc, maskOp.getMask());
      break;
    case vector::CombiningKind::MAXUI:
      result = lowerPredicatedReductionWithStartValue<LLVM::VPReduceUMaxOp,
                                                      ReductionNeutralUIntMin>(
          rewriter, loc, llvmType, operand, acc, maskOp.getMask());
      break;
    case vector::CombiningKind::MAXSI:
      result = lowerPredicatedReductionWithStartValue<LLVM::VPReduceSMaxOp,
                                                      ReductionNeutralSIntMin>(
          rewriter, loc, llvmType, operand, acc, maskOp.getMask());
      break;
    case vector::CombiningKind::AND:
      result = lowerPredicatedReductionWithStartValue<LLVM::VPReduceAndOp,
                                                      ReductionNeutralAllOnes>(
          rewriter, loc, llvmType, operand, acc, maskOp.getMask());
      break;
    case vector::CombiningKind::OR:
      result = lowerPredicatedReductionWithStartValue<LLVM::VPReduceOrOp,
                                                      ReductionNeutralZero>(
          rewriter, loc, llvmType, operand, acc, maskOp.getMask());
      break;
    case vector::CombiningKind::XOR:
      result = lowerPredicatedReductionWithStartValue<LLVM::VPReduceXorOp,
                                                      ReductionNeutralZero>(
          rewriter, loc, llvmType, operand, acc, maskOp.getMask());
      break;
    case vector::CombiningKind::MINNUMF:
      result = lowerPredicatedReductionWithStartValue<LLVM::VPReduceFMinOp,
                                                      ReductionNeutralFPMax>(
          rewriter, loc, llvmType, operand, acc, maskOp.getMask());
      break;
    case vector::CombiningKind::MAXNUMF:
      result = lowerPredicatedReductionWithStartValue<LLVM::VPReduceFMaxOp,
                                                      ReductionNeutralFPMin>(
          rewriter, loc, llvmType, operand, acc, maskOp.getMask());
      break;
    case CombiningKind::MAXIMUMF:
      result = lowerMaskedReductionWithRegular<LLVM::vector_reduce_fmaximum,
                                               MaskNeutralFMaximum>(
          rewriter, loc, llvmType, operand, acc, maskOp.getMask(), fmf);
      break;
    case CombiningKind::MINIMUMF:
      result = lowerMaskedReductionWithRegular<LLVM::vector_reduce_fminimum,
                                               MaskNeutralFMinimum>(
          rewriter, loc, llvmType, operand, acc, maskOp.getMask(), fmf);
      break;
    }

    // Replace `vector.mask` operation altogether.
    rewriter.replaceOp(maskOp, result);
    return success();
  }
};

class VectorShuffleOpConversion
    : public ConvertOpToLLVMPattern<vector::ShuffleOp> {
public:
  using ConvertOpToLLVMPattern<vector::ShuffleOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::ShuffleOp shuffleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = shuffleOp->getLoc();
    auto v1Type = shuffleOp.getV1VectorType();
    auto v2Type = shuffleOp.getV2VectorType();
    auto vectorType = shuffleOp.getResultVectorType();
    Type llvmType = typeConverter->convertType(vectorType);
    ArrayRef<int64_t> mask = shuffleOp.getMask();

    // Bail if result type cannot be lowered.
    if (!llvmType)
      return failure();

    // Get rank and dimension sizes.
    int64_t rank = vectorType.getRank();
#ifndef NDEBUG
    bool wellFormed0DCase =
        v1Type.getRank() == 0 && v2Type.getRank() == 0 && rank == 1;
    bool wellFormedNDCase =
        v1Type.getRank() == rank && v2Type.getRank() == rank;
    assert((wellFormed0DCase || wellFormedNDCase) && "op is not well-formed");
#endif

    // For rank 0 and 1, where both operands have *exactly* the same vector
    // type, there is direct shuffle support in LLVM. Use it!
    if (rank <= 1 && v1Type == v2Type) {
      Value llvmShuffleOp = LLVM::ShuffleVectorOp::create(
          rewriter, loc, adaptor.getV1(), adaptor.getV2(),
          llvm::to_vector_of<int32_t>(mask));
      rewriter.replaceOp(shuffleOp, llvmShuffleOp);
      return success();
    }

    // For all other cases, insert the individual values individually.
    int64_t v1Dim = v1Type.getDimSize(0);
    Type eltType;
    if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(llvmType))
      eltType = arrayType.getElementType();
    else
      eltType = cast<VectorType>(llvmType).getElementType();
    Value insert = LLVM::PoisonOp::create(rewriter, loc, llvmType);
    int64_t insPos = 0;
    for (int64_t extPos : mask) {
      Value value = adaptor.getV1();
      if (extPos >= v1Dim) {
        extPos -= v1Dim;
        value = adaptor.getV2();
      }
      Value extract = extractOne(rewriter, *getTypeConverter(), loc, value,
                                 eltType, rank, extPos);
      insert = insertOne(rewriter, *getTypeConverter(), loc, insert, extract,
                         llvmType, rank, insPos++);
    }
    rewriter.replaceOp(shuffleOp, insert);
    return success();
  }
};

class VectorExtractOpConversion
    : public ConvertOpToLLVMPattern<vector::ExtractOp> {
public:
  using ConvertOpToLLVMPattern<vector::ExtractOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = extractOp->getLoc();
    auto resultType = extractOp.getResult().getType();
    auto llvmResultType = typeConverter->convertType(resultType);
    // Bail if result type cannot be lowered.
    if (!llvmResultType)
      return failure();

    SmallVector<OpFoldResult> positionVec = getMixedValues(
        adaptor.getStaticPosition(), adaptor.getDynamicPosition(), rewriter);

    // The Vector -> LLVM lowering models N-D vectors as nested aggregates of
    // 1-d vectors. This nesting is modeled using arrays. We do this conversion
    // from a N-d vector extract to a nested aggregate vector extract in two
    // steps:
    //  - Extract a member from the nested aggregate. The result can be
    //    a lower rank nested aggregate or a vector (1-D). This is done using
    //    `llvm.extractvalue`.
    //  - Extract a scalar out of the vector if needed. This is done using
    //   `llvm.extractelement`.

    // Determine if we need to extract a member out of the aggregate. We
    // always need to extract a member if the input rank >= 2.
    bool extractsAggregate = extractOp.getSourceVectorType().getRank() >= 2;
    // Determine if we need to extract a scalar as the result. We extract
    // a scalar if the extract is full rank, i.e., the number of indices is
    // equal to source vector rank.
    bool extractsScalar = static_cast<int64_t>(positionVec.size()) ==
                          extractOp.getSourceVectorType().getRank();

    // Since the LLVM type converter converts 0-d vectors to 1-d vectors, we
    // need to add a position for this change.
    if (extractOp.getSourceVectorType().getRank() == 0) {
      Type idxType = typeConverter->convertType(rewriter.getIndexType());
      positionVec.push_back(rewriter.getZeroAttr(idxType));
    }

    Value extracted = adaptor.getVector();
    if (extractsAggregate) {
      ArrayRef<OpFoldResult> position(positionVec);
      if (extractsScalar) {
        // If we are extracting a scalar from the extracted member, we drop
        // the last index, which will be used to extract the scalar out of the
        // vector.
        position = position.drop_back();
      }
      // llvm.extractvalue does not support dynamic dimensions.
      if (!llvm::all_of(position, llvm::IsaPred<Attribute>)) {
        return failure();
      }
      extracted = LLVM::ExtractValueOp::create(rewriter, loc, extracted,
                                               getAsIntegers(position));
    }

    if (extractsScalar) {
      extracted = LLVM::ExtractElementOp::create(
          rewriter, loc, extracted,
          getAsLLVMValue(rewriter, loc, positionVec.back()));
    }

    rewriter.replaceOp(extractOp, extracted);
    return success();
  }
};

/// Conversion pattern that turns a vector.fma on a 1-D vector
/// into an llvm.intr.fmuladd. This is a trivial 1-1 conversion.
/// This does not match vectors of n >= 2 rank.
///
/// Example:
/// ```
///  vector.fma %a, %a, %a : vector<8xf32>
/// ```
/// is converted to:
/// ```
///  llvm.intr.fmuladd %va, %va, %va:
///    (!llvm."<8 x f32>">, !llvm<"<8 x f32>">, !llvm<"<8 x f32>">)
///    -> !llvm."<8 x f32>">
/// ```
class VectorFMAOp1DConversion : public ConvertOpToLLVMPattern<vector::FMAOp> {
public:
  using ConvertOpToLLVMPattern<vector::FMAOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType vType = fmaOp.getVectorType();
    if (vType.getRank() > 1)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::FMulAddOp>(
        fmaOp, adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc());
    return success();
  }
};

class VectorInsertOpConversion
    : public ConvertOpToLLVMPattern<vector::InsertOp> {
public:
  using ConvertOpToLLVMPattern<vector::InsertOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::InsertOp insertOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = insertOp->getLoc();
    auto destVectorType = insertOp.getDestVectorType();
    auto llvmResultType = typeConverter->convertType(destVectorType);
    // Bail if result type cannot be lowered.
    if (!llvmResultType)
      return failure();

    SmallVector<OpFoldResult> positionVec = getMixedValues(
        adaptor.getStaticPosition(), adaptor.getDynamicPosition(), rewriter);

    // The logic in this pattern mirrors VectorExtractOpConversion. Refer to
    // its explanatory comment about how N-D vectors are converted as nested
    // aggregates (llvm.array's) of 1D vectors.
    //
    // The innermost dimension of the destination vector, when converted to a
    // nested aggregate form, will always be a 1D vector.
    //
    // * If the insertion is happening into the innermost dimension of the
    //   destination vector:
    //   - If the destination is a nested aggregate, extract a 1D vector out of
    //     the aggregate. This can be done using llvm.extractvalue. The
    //     destination is now guaranteed to be a 1D vector, to which we are
    //     inserting.
    //   - Do the insertion into the 1D destination vector, and make the result
    //     the new source nested aggregate. This can be done using
    //     llvm.insertelement.
    // * Insert the source nested aggregate into the destination nested
    //   aggregate.

    // Determine if we need to extract/insert a 1D vector out of the aggregate.
    bool isNestedAggregate = isa<LLVM::LLVMArrayType>(llvmResultType);
    // Determine if we need to insert a scalar into the 1D vector.
    bool insertIntoInnermostDim =
        static_cast<int64_t>(positionVec.size()) == destVectorType.getRank();

    ArrayRef<OpFoldResult> positionOf1DVectorWithinAggregate(
        positionVec.begin(),
        insertIntoInnermostDim ? positionVec.size() - 1 : positionVec.size());
    OpFoldResult positionOfScalarWithin1DVector;
    if (destVectorType.getRank() == 0) {
      // Since the LLVM type converter converts 0D vectors to 1D vectors, we
      // need to create a 0 here as the position into the 1D vector.
      Type idxType = typeConverter->convertType(rewriter.getIndexType());
      positionOfScalarWithin1DVector = rewriter.getZeroAttr(idxType);
    } else if (insertIntoInnermostDim) {
      positionOfScalarWithin1DVector = positionVec.back();
    }

    // We are going to mutate this 1D vector until it is either the final
    // result (in the non-aggregate case) or the value that needs to be
    // inserted into the aggregate result.
    Value sourceAggregate = adaptor.getValueToStore();
    if (insertIntoInnermostDim) {
      // Scalar-into-1D-vector case, so we know we will have to create a
      // InsertElementOp. The question is into what destination.
      if (isNestedAggregate) {
        // Aggregate case: the destination for the InsertElementOp needs to be
        // extracted from the aggregate.
        if (!llvm::all_of(positionOf1DVectorWithinAggregate,
                          llvm::IsaPred<Attribute>)) {
          // llvm.extractvalue does not support dynamic dimensions.
          return failure();
        }
        sourceAggregate = LLVM::ExtractValueOp::create(
            rewriter, loc, adaptor.getDest(),
            getAsIntegers(positionOf1DVectorWithinAggregate));
      } else {
        // No-aggregate case. The destination for the InsertElementOp is just
        // the insertOp's destination.
        sourceAggregate = adaptor.getDest();
      }
      // Insert the scalar into the 1D vector.
      sourceAggregate = LLVM::InsertElementOp::create(
          rewriter, loc, sourceAggregate.getType(), sourceAggregate,
          adaptor.getValueToStore(),
          getAsLLVMValue(rewriter, loc, positionOfScalarWithin1DVector));
    }

    Value result = sourceAggregate;
    if (isNestedAggregate) {
      result = LLVM::InsertValueOp::create(
          rewriter, loc, adaptor.getDest(), sourceAggregate,
          getAsIntegers(positionOf1DVectorWithinAggregate));
    }

    rewriter.replaceOp(insertOp, result);
    return success();
  }
};

/// Lower vector.scalable.insert ops to LLVM vector.insert
struct VectorScalableInsertOpLowering
    : public ConvertOpToLLVMPattern<vector::ScalableInsertOp> {
  using ConvertOpToLLVMPattern<
      vector::ScalableInsertOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::ScalableInsertOp insOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::vector_insert>(
        insOp, adaptor.getDest(), adaptor.getValueToStore(), adaptor.getPos());
    return success();
  }
};

/// Lower vector.scalable.extract ops to LLVM vector.extract
struct VectorScalableExtractOpLowering
    : public ConvertOpToLLVMPattern<vector::ScalableExtractOp> {
  using ConvertOpToLLVMPattern<
      vector::ScalableExtractOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::ScalableExtractOp extOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::vector_extract>(
        extOp, typeConverter->convertType(extOp.getResultVectorType()),
        adaptor.getSource(), adaptor.getPos());
    return success();
  }
};

/// Rank reducing rewrite for n-D FMA into (n-1)-D FMA where n > 1.
///
/// Example:
/// ```
///   %d = vector.fma %a, %b, %c : vector<2x4xf32>
/// ```
/// is rewritten into:
/// ```
///  %r = vector.broadcast %f0 : f32 to vector<2x4xf32>
///  %va = vector.extractvalue %a[0] : vector<2x4xf32>
///  %vb = vector.extractvalue %b[0] : vector<2x4xf32>
///  %vc = vector.extractvalue %c[0] : vector<2x4xf32>
///  %vd = vector.fma %va, %vb, %vc : vector<4xf32>
///  %r2 = vector.insertvalue %vd, %r[0] : vector<4xf32> into vector<2x4xf32>
///  %va2 = vector.extractvalue %a2[1] : vector<2x4xf32>
///  %vb2 = vector.extractvalue %b2[1] : vector<2x4xf32>
///  %vc2 = vector.extractvalue %c2[1] : vector<2x4xf32>
///  %vd2 = vector.fma %va2, %vb2, %vc2 : vector<4xf32>
///  %r3 = vector.insertvalue %vd2, %r2[1] : vector<4xf32> into vector<2x4xf32>
///  // %r3 holds the final value.
/// ```
class VectorFMAOpNDRewritePattern : public OpRewritePattern<FMAOp> {
public:
  using OpRewritePattern<FMAOp>::OpRewritePattern;

  void initialize() {
    // This pattern recursively unpacks one dimension at a time. The recursion
    // bounded as the rank is strictly decreasing.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(FMAOp op,
                                PatternRewriter &rewriter) const override {
    auto vType = op.getVectorType();
    if (vType.getRank() < 2)
      return failure();

    auto loc = op.getLoc();
    auto elemType = vType.getElementType();
    Value zero = arith::ConstantOp::create(rewriter, loc, elemType,
                                           rewriter.getZeroAttr(elemType));
    Value desc = vector::BroadcastOp::create(rewriter, loc, vType, zero);
    for (int64_t i = 0, e = vType.getShape().front(); i != e; ++i) {
      Value extrLHS = ExtractOp::create(rewriter, loc, op.getLhs(), i);
      Value extrRHS = ExtractOp::create(rewriter, loc, op.getRhs(), i);
      Value extrACC = ExtractOp::create(rewriter, loc, op.getAcc(), i);
      Value fma = FMAOp::create(rewriter, loc, extrLHS, extrRHS, extrACC);
      desc = InsertOp::create(rewriter, loc, fma, desc, i);
    }
    rewriter.replaceOp(op, desc);
    return success();
  }
};

/// Returns the strides if the memory underlying `memRefType` has a contiguous
/// static layout.
static std::optional<SmallVector<int64_t, 4>>
computeContiguousStrides(MemRefType memRefType) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(memRefType.getStridesAndOffset(strides, offset)))
    return std::nullopt;
  if (!strides.empty() && strides.back() != 1)
    return std::nullopt;
  // If no layout or identity layout, this is contiguous by definition.
  if (memRefType.getLayout().isIdentity())
    return strides;

  // Otherwise, we must determine contiguity form shapes. This can only ever
  // work in static cases because MemRefType is underspecified to represent
  // contiguous dynamic shapes in other ways than with just empty/identity
  // layout.
  auto sizes = memRefType.getShape();
  for (int index = 0, e = strides.size() - 1; index < e; ++index) {
    if (ShapedType::isDynamic(sizes[index + 1]) ||
        ShapedType::isDynamic(strides[index]) ||
        ShapedType::isDynamic(strides[index + 1]))
      return std::nullopt;
    if (strides[index] != strides[index + 1] * sizes[index + 1])
      return std::nullopt;
  }
  return strides;
}

class VectorTypeCastOpConversion
    : public ConvertOpToLLVMPattern<vector::TypeCastOp> {
public:
  using ConvertOpToLLVMPattern<vector::TypeCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::TypeCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = castOp->getLoc();
    MemRefType sourceMemRefType =
        cast<MemRefType>(castOp.getOperand().getType());
    MemRefType targetMemRefType = castOp.getType();

    // Only static shape casts supported atm.
    if (!sourceMemRefType.hasStaticShape() ||
        !targetMemRefType.hasStaticShape())
      return failure();

    auto llvmSourceDescriptorTy =
        dyn_cast<LLVM::LLVMStructType>(adaptor.getOperands()[0].getType());
    if (!llvmSourceDescriptorTy)
      return failure();
    MemRefDescriptor sourceMemRef(adaptor.getOperands()[0]);

    auto llvmTargetDescriptorTy = dyn_cast_or_null<LLVM::LLVMStructType>(
        typeConverter->convertType(targetMemRefType));
    if (!llvmTargetDescriptorTy)
      return failure();

    // Only contiguous source buffers supported atm.
    auto sourceStrides = computeContiguousStrides(sourceMemRefType);
    if (!sourceStrides)
      return failure();
    auto targetStrides = computeContiguousStrides(targetMemRefType);
    if (!targetStrides)
      return failure();
    // Only support static strides for now, regardless of contiguity.
    if (llvm::any_of(*targetStrides, ShapedType::isDynamic))
      return failure();

    auto int64Ty = IntegerType::get(rewriter.getContext(), 64);

    // Create descriptor.
    auto desc = MemRefDescriptor::poison(rewriter, loc, llvmTargetDescriptorTy);
    // Set allocated ptr.
    Value allocated = sourceMemRef.allocatedPtr(rewriter, loc);
    desc.setAllocatedPtr(rewriter, loc, allocated);

    // Set aligned ptr.
    Value ptr = sourceMemRef.alignedPtr(rewriter, loc);
    desc.setAlignedPtr(rewriter, loc, ptr);
    // Fill offset 0.
    auto attr = rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
    auto zero = LLVM::ConstantOp::create(rewriter, loc, int64Ty, attr);
    desc.setOffset(rewriter, loc, zero);

    // Fill size and stride descriptors in memref.
    for (const auto &indexedSize :
         llvm::enumerate(targetMemRefType.getShape())) {
      int64_t index = indexedSize.index();
      auto sizeAttr =
          rewriter.getIntegerAttr(rewriter.getIndexType(), indexedSize.value());
      auto size = LLVM::ConstantOp::create(rewriter, loc, int64Ty, sizeAttr);
      desc.setSize(rewriter, loc, index, size);
      auto strideAttr = rewriter.getIntegerAttr(rewriter.getIndexType(),
                                                (*targetStrides)[index]);
      auto stride =
          LLVM::ConstantOp::create(rewriter, loc, int64Ty, strideAttr);
      desc.setStride(rewriter, loc, index, stride);
    }

    rewriter.replaceOp(castOp, {desc});
    return success();
  }
};

/// Conversion pattern for a `vector.create_mask` (1-D scalable vectors only).
/// Non-scalable versions of this operation are handled in Vector Transforms.
class VectorCreateMaskOpConversion
    : public OpConversionPattern<vector::CreateMaskOp> {
public:
  explicit VectorCreateMaskOpConversion(MLIRContext *context,
                                        bool enableIndexOpt)
      : OpConversionPattern<vector::CreateMaskOp>(context),
        force32BitVectorIndices(enableIndexOpt) {}

  LogicalResult
  matchAndRewrite(vector::CreateMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = op.getType();
    if (dstType.getRank() != 1 || !cast<VectorType>(dstType).isScalable())
      return failure();
    IntegerType idxType =
        force32BitVectorIndices ? rewriter.getI32Type() : rewriter.getI64Type();
    auto loc = op->getLoc();
    Value indices = LLVM::StepVectorOp::create(
        rewriter, loc,
        LLVM::getVectorType(idxType, dstType.getShape()[0],
                            /*isScalable=*/true));
    auto bound = getValueOrCreateCastToIndexLike(rewriter, loc, idxType,
                                                 adaptor.getOperands()[0]);
    Value bounds = BroadcastOp::create(rewriter, loc, indices.getType(), bound);
    Value comp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                       indices, bounds);
    rewriter.replaceOp(op, comp);
    return success();
  }

private:
  const bool force32BitVectorIndices;
};

class VectorPrintOpConversion : public ConvertOpToLLVMPattern<vector::PrintOp> {
  SymbolTableCollection *symbolTables = nullptr;

public:
  explicit VectorPrintOpConversion(
      const LLVMTypeConverter &typeConverter,
      SymbolTableCollection *symbolTables = nullptr)
      : ConvertOpToLLVMPattern<vector::PrintOp>(typeConverter),
        symbolTables(symbolTables) {}

  // Lowering implementation that relies on a small runtime support library,
  // which only needs to provide a few printing methods (single value for all
  // data types, opening/closing bracket, comma, newline). The lowering splits
  // the vector into elementary printing operations. The advantage of this
  // approach is that the library can remain unaware of all low-level
  // implementation details of vectors while still supporting output of any
  // shaped and dimensioned vector.
  //
  // Note: This lowering only handles scalars, n-D vectors are broken into
  // printing scalars in loops in VectorToSCF.
  //
  // TODO: rely solely on libc in future? something else?
  //
  LogicalResult
  matchAndRewrite(vector::PrintOp printOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parent = printOp->getParentOfType<ModuleOp>();
    if (!parent)
      return failure();

    auto loc = printOp->getLoc();

    if (auto value = adaptor.getSource()) {
      Type printType = printOp.getPrintType();
      if (isa<VectorType>(printType)) {
        // Vectors should be broken into elementary print ops in VectorToSCF.
        return failure();
      }
      if (failed(emitScalarPrint(rewriter, parent, loc, printType, value)))
        return failure();
    }

    auto punct = printOp.getPunctuation();
    if (auto stringLiteral = printOp.getStringLiteral()) {
      auto createResult =
          LLVM::createPrintStrCall(rewriter, loc, parent, "vector_print_str",
                                   *stringLiteral, *getTypeConverter(),
                                   /*addNewline=*/false);
      if (createResult.failed())
        return failure();

    } else if (punct != PrintPunctuation::NoPunctuation) {
      FailureOr<LLVM::LLVMFuncOp> op = [&]() {
        switch (punct) {
        case PrintPunctuation::Close:
          return LLVM::lookupOrCreatePrintCloseFn(rewriter, parent,
                                                  symbolTables);
        case PrintPunctuation::Open:
          return LLVM::lookupOrCreatePrintOpenFn(rewriter, parent,
                                                 symbolTables);
        case PrintPunctuation::Comma:
          return LLVM::lookupOrCreatePrintCommaFn(rewriter, parent,
                                                  symbolTables);
        case PrintPunctuation::NewLine:
          return LLVM::lookupOrCreatePrintNewlineFn(rewriter, parent,
                                                    symbolTables);
        default:
          llvm_unreachable("unexpected punctuation");
        }
      }();
      if (failed(op))
        return failure();
      emitCall(rewriter, printOp->getLoc(), op.value());
    }

    rewriter.eraseOp(printOp);
    return success();
  }

private:
  enum class PrintConversion {
    // clang-format off
    None,
    ZeroExt64,
    SignExt64,
    Bitcast16
    // clang-format on
  };

  LogicalResult emitScalarPrint(ConversionPatternRewriter &rewriter,
                                ModuleOp parent, Location loc, Type printType,
                                Value value) const {
    if (typeConverter->convertType(printType) == nullptr)
      return failure();

    // Make sure element type has runtime support.
    PrintConversion conversion = PrintConversion::None;
    FailureOr<Operation *> printer;
    if (printType.isF32()) {
      printer = LLVM::lookupOrCreatePrintF32Fn(rewriter, parent, symbolTables);
    } else if (printType.isF64()) {
      printer = LLVM::lookupOrCreatePrintF64Fn(rewriter, parent, symbolTables);
    } else if (printType.isF16()) {
      conversion = PrintConversion::Bitcast16; // bits!
      printer = LLVM::lookupOrCreatePrintF16Fn(rewriter, parent, symbolTables);
    } else if (printType.isBF16()) {
      conversion = PrintConversion::Bitcast16; // bits!
      printer = LLVM::lookupOrCreatePrintBF16Fn(rewriter, parent, symbolTables);
    } else if (printType.isIndex()) {
      printer = LLVM::lookupOrCreatePrintU64Fn(rewriter, parent, symbolTables);
    } else if (auto intTy = dyn_cast<IntegerType>(printType)) {
      // Integers need a zero or sign extension on the operand
      // (depending on the source type) as well as a signed or
      // unsigned print method. Up to 64-bit is supported.
      unsigned width = intTy.getWidth();
      if (intTy.isUnsigned()) {
        if (width <= 64) {
          if (width < 64)
            conversion = PrintConversion::ZeroExt64;
          printer =
              LLVM::lookupOrCreatePrintU64Fn(rewriter, parent, symbolTables);
        } else {
          return failure();
        }
      } else {
        assert(intTy.isSignless() || intTy.isSigned());
        if (width <= 64) {
          // Note that we *always* zero extend booleans (1-bit integers),
          // so that true/false is printed as 1/0 rather than -1/0.
          if (width == 1)
            conversion = PrintConversion::ZeroExt64;
          else if (width < 64)
            conversion = PrintConversion::SignExt64;
          printer =
              LLVM::lookupOrCreatePrintI64Fn(rewriter, parent, symbolTables);
        } else {
          return failure();
        }
      }
    } else {
      return failure();
    }
    if (failed(printer))
      return failure();

    switch (conversion) {
    case PrintConversion::ZeroExt64:
      value = arith::ExtUIOp::create(
          rewriter, loc, IntegerType::get(rewriter.getContext(), 64), value);
      break;
    case PrintConversion::SignExt64:
      value = arith::ExtSIOp::create(
          rewriter, loc, IntegerType::get(rewriter.getContext(), 64), value);
      break;
    case PrintConversion::Bitcast16:
      value = LLVM::BitcastOp::create(
          rewriter, loc, IntegerType::get(rewriter.getContext(), 16), value);
      break;
    case PrintConversion::None:
      break;
    }
    emitCall(rewriter, loc, printer.value(), value);
    return success();
  }

  // Helper to emit a call.
  static void emitCall(ConversionPatternRewriter &rewriter, Location loc,
                       Operation *ref, ValueRange params = ValueRange()) {
    LLVM::CallOp::create(rewriter, loc, TypeRange(), SymbolRefAttr::get(ref),
                         params);
  }
};

/// A broadcast of a scalar is lowered to an insertelement + a shufflevector
/// operation. Only broadcasts to 0-d and 1-d vectors are lowered by this
/// pattern, the higher rank cases are handled by another pattern.
struct VectorBroadcastScalarToLowRankLowering
    : public ConvertOpToLLVMPattern<vector::BroadcastOp> {
  using ConvertOpToLLVMPattern<vector::BroadcastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp broadcast, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<VectorType>(broadcast.getSourceType()))
      return rewriter.notifyMatchFailure(
          broadcast, "broadcast from vector type not handled");

    VectorType resultType = broadcast.getType();
    if (resultType.getRank() > 1)
      return rewriter.notifyMatchFailure(broadcast,
                                         "broadcast to 2+-d handled elsewhere");

    // First insert it into a poison vector so we can shuffle it.
    auto vectorType = typeConverter->convertType(broadcast.getType());
    Value poison =
        LLVM::PoisonOp::create(rewriter, broadcast.getLoc(), vectorType);
    auto zero = LLVM::ConstantOp::create(
        rewriter, broadcast.getLoc(),
        typeConverter->convertType(rewriter.getIntegerType(32)),
        rewriter.getZeroAttr(rewriter.getIntegerType(32)));

    // For 0-d vector, we simply do `insertelement`.
    if (resultType.getRank() == 0) {
      rewriter.replaceOpWithNewOp<LLVM::InsertElementOp>(
          broadcast, vectorType, poison, adaptor.getSource(), zero);
      return success();
    }

    // For 1-d vector, we additionally do a `vectorshuffle`.
    auto v =
        LLVM::InsertElementOp::create(rewriter, broadcast.getLoc(), vectorType,
                                      poison, adaptor.getSource(), zero);

    int64_t width = cast<VectorType>(broadcast.getType()).getDimSize(0);
    SmallVector<int32_t> zeroValues(width, 0);

    // Shuffle the value across the desired number of elements.
    rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(broadcast, v, poison,
                                                       zeroValues);
    return success();
  }
};

/// The broadcast of a scalar is lowered to an insertelement + a shufflevector
/// operation. Only broadcasts to 2+-d vector result types are lowered by this
/// pattern, the 1-d case is handled by another pattern. Broadcasts from vectors
/// are not converted to LLVM, only broadcasts from scalars are.
struct VectorBroadcastScalarToNdLowering
    : public ConvertOpToLLVMPattern<BroadcastOp> {
  using ConvertOpToLLVMPattern<BroadcastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BroadcastOp broadcast, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<VectorType>(broadcast.getSourceType()))
      return rewriter.notifyMatchFailure(
          broadcast, "broadcast from vector type not handled");

    VectorType resultType = broadcast.getType();
    if (resultType.getRank() <= 1)
      return rewriter.notifyMatchFailure(
          broadcast, "broadcast to 1-d or 0-d handled elsewhere");

    // First insert it into an undef vector so we can shuffle it.
    auto loc = broadcast.getLoc();
    auto vectorTypeInfo =
        LLVM::detail::extractNDVectorTypeInfo(resultType, *getTypeConverter());
    auto llvmNDVectorTy = vectorTypeInfo.llvmNDVectorTy;
    auto llvm1DVectorTy = vectorTypeInfo.llvm1DVectorTy;
    if (!llvmNDVectorTy || !llvm1DVectorTy)
      return failure();

    // Construct returned value.
    Value desc = LLVM::PoisonOp::create(rewriter, loc, llvmNDVectorTy);

    // Construct a 1-D vector with the broadcasted value that we insert in all
    // the places within the returned descriptor.
    Value vdesc = LLVM::PoisonOp::create(rewriter, loc, llvm1DVectorTy);
    auto zero = LLVM::ConstantOp::create(
        rewriter, loc, typeConverter->convertType(rewriter.getIntegerType(32)),
        rewriter.getZeroAttr(rewriter.getIntegerType(32)));
    Value v = LLVM::InsertElementOp::create(rewriter, loc, llvm1DVectorTy,
                                            vdesc, adaptor.getSource(), zero);

    // Shuffle the value across the desired number of elements.
    int64_t width = resultType.getDimSize(resultType.getRank() - 1);
    SmallVector<int32_t> zeroValues(width, 0);
    v = LLVM::ShuffleVectorOp::create(rewriter, loc, v, v, zeroValues);

    // Iterate of linear index, convert to coords space and insert broadcasted
    // 1-D vector in each position.
    nDVectorIterate(vectorTypeInfo, rewriter, [&](ArrayRef<int64_t> position) {
      desc = LLVM::InsertValueOp::create(rewriter, loc, desc, v, position);
    });
    rewriter.replaceOp(broadcast, desc);
    return success();
  }
};

/// Conversion pattern for a `vector.interleave`.
/// This supports fixed-sized vectors and scalable vectors.
struct VectorInterleaveOpLowering
    : public ConvertOpToLLVMPattern<vector::InterleaveOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::InterleaveOp interleaveOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = interleaveOp.getResultVectorType();
    // n-D interleaves should have been lowered already.
    if (resultType.getRank() != 1)
      return rewriter.notifyMatchFailure(interleaveOp,
                                         "InterleaveOp not rank 1");
    // If the result is rank 1, then this directly maps to LLVM.
    if (resultType.isScalable()) {
      rewriter.replaceOpWithNewOp<LLVM::vector_interleave2>(
          interleaveOp, typeConverter->convertType(resultType),
          adaptor.getLhs(), adaptor.getRhs());
      return success();
    }
    // Lower fixed-size interleaves to a shufflevector. While the
    // vector.interleave2 intrinsic supports fixed and scalable vectors, the
    // langref still recommends fixed-vectors use shufflevector, see:
    // https://llvm.org/docs/LangRef.html#id876.
    int64_t resultVectorSize = resultType.getNumElements();
    SmallVector<int32_t> interleaveShuffleMask;
    interleaveShuffleMask.reserve(resultVectorSize);
    for (int i = 0, end = resultVectorSize / 2; i < end; ++i) {
      interleaveShuffleMask.push_back(i);
      interleaveShuffleMask.push_back((resultVectorSize / 2) + i);
    }
    rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(
        interleaveOp, adaptor.getLhs(), adaptor.getRhs(),
        interleaveShuffleMask);
    return success();
  }
};

/// Conversion pattern for a `vector.deinterleave`.
/// This supports fixed-sized vectors and scalable vectors.
struct VectorDeinterleaveOpLowering
    : public ConvertOpToLLVMPattern<vector::DeinterleaveOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::DeinterleaveOp deinterleaveOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = deinterleaveOp.getResultVectorType();
    VectorType sourceType = deinterleaveOp.getSourceVectorType();
    auto loc = deinterleaveOp.getLoc();

    // Note: n-D deinterleave operations should be lowered to the 1-D before
    // converting to LLVM.
    if (resultType.getRank() != 1)
      return rewriter.notifyMatchFailure(deinterleaveOp,
                                         "DeinterleaveOp not rank 1");

    if (resultType.isScalable()) {
      auto llvmTypeConverter = this->getTypeConverter();
      auto deinterleaveResults = deinterleaveOp.getResultTypes();
      auto packedOpResults =
          llvmTypeConverter->packOperationResults(deinterleaveResults);
      auto intrinsic = LLVM::vector_deinterleave2::create(
          rewriter, loc, packedOpResults, adaptor.getSource());

      auto evenResult = LLVM::ExtractValueOp::create(
          rewriter, loc, intrinsic->getResult(0), 0);
      auto oddResult = LLVM::ExtractValueOp::create(rewriter, loc,
                                                    intrinsic->getResult(0), 1);

      rewriter.replaceOp(deinterleaveOp, ValueRange{evenResult, oddResult});
      return success();
    }
    // Lower fixed-size deinterleave to two shufflevectors. While the
    // vector.deinterleave2 intrinsic supports fixed and scalable vectors, the
    // langref still recommends fixed-vectors use shufflevector, see:
    // https://llvm.org/docs/LangRef.html#id889.
    int64_t resultVectorSize = resultType.getNumElements();
    SmallVector<int32_t> evenShuffleMask;
    SmallVector<int32_t> oddShuffleMask;

    evenShuffleMask.reserve(resultVectorSize);
    oddShuffleMask.reserve(resultVectorSize);

    for (int i = 0; i < sourceType.getNumElements(); ++i) {
      if (i % 2 == 0)
        evenShuffleMask.push_back(i);
      else
        oddShuffleMask.push_back(i);
    }

    auto poison = LLVM::PoisonOp::create(rewriter, loc, sourceType);
    auto evenShuffle = LLVM::ShuffleVectorOp::create(
        rewriter, loc, adaptor.getSource(), poison, evenShuffleMask);
    auto oddShuffle = LLVM::ShuffleVectorOp::create(
        rewriter, loc, adaptor.getSource(), poison, oddShuffleMask);

    rewriter.replaceOp(deinterleaveOp, ValueRange{evenShuffle, oddShuffle});
    return success();
  }
};

/// Conversion pattern for a `vector.from_elements`.
struct VectorFromElementsLowering
    : public ConvertOpToLLVMPattern<vector::FromElementsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::FromElementsOp fromElementsOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = fromElementsOp.getLoc();
    VectorType vectorType = fromElementsOp.getType();
    // TODO: Multi-dimensional vectors lower to !llvm.array<... x vector<>>.
    // Such ops should be handled in the same way as vector.insert.
    if (vectorType.getRank() > 1)
      return rewriter.notifyMatchFailure(fromElementsOp,
                                         "rank > 1 vectors are not supported");
    Type llvmType = typeConverter->convertType(vectorType);
    Value result = LLVM::PoisonOp::create(rewriter, loc, llvmType);
    for (auto [idx, val] : llvm::enumerate(adaptor.getElements()))
      result = vector::InsertOp::create(rewriter, loc, val, result, idx);
    rewriter.replaceOp(fromElementsOp, result);
    return success();
  }
};

/// Conversion pattern for a `vector.to_elements`.
struct VectorToElementsLowering
    : public ConvertOpToLLVMPattern<vector::ToElementsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::ToElementsOp toElementsOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = toElementsOp.getLoc();
    auto idxType = typeConverter->convertType(rewriter.getIndexType());
    Value source = adaptor.getSource();

    SmallVector<Value> results(toElementsOp->getNumResults());
    for (auto [idx, element] : llvm::enumerate(toElementsOp.getElements())) {
      // Create an extractelement operation only for results that are not dead.
      if (element.use_empty())
        continue;

      auto constIdx = LLVM::ConstantOp::create(
          rewriter, loc, idxType, rewriter.getIntegerAttr(idxType, idx));
      auto llvmType = typeConverter->convertType(element.getType());

      Value result = LLVM::ExtractElementOp::create(rewriter, loc, llvmType,
                                                    source, constIdx);
      results[idx] = result;
    }

    rewriter.replaceOp(toElementsOp, results);
    return success();
  }
};

/// Conversion pattern for vector.step.
struct VectorScalableStepOpLowering
    : public ConvertOpToLLVMPattern<vector::StepOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::StepOp stepOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<VectorType>(stepOp.getType());
    if (!resultType.isScalable()) {
      return failure();
    }
    Type llvmType = typeConverter->convertType(stepOp.getType());
    rewriter.replaceOpWithNewOp<LLVM::StepVectorOp>(stepOp, llvmType);
    return success();
  }
};

/// Progressive lowering of a `vector.contract %a, %b, %c` with row-major matmul
/// semantics to:
/// ```
///    %flattened_a = vector.shape_cast %a
///    %flattened_b = vector.shape_cast %b
///    %flattened_d = vector.matrix_multiply %flattened_a, %flattened_b
///    %d = vector.shape_cast %%flattened_d
///    %e = add %c, %d
/// ```
/// `vector.matrix_multiply` later lowers to `llvm.matrix.multiply`.
//
/// This only kicks in when vectorContractLowering is set to Matmul and
/// the vector.contract op is a row-major matrix multiply.
class ContractionOpToMatmulOpLowering
    : public vector::MaskableOpRewritePattern<vector::ContractionOp> {
public:
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  ContractionOpToMatmulOpLowering(
      vector::VectorContractLowering vectorContractLowering,
      MLIRContext *context, PatternBenefit benefit = 100)
      : MaskableOpRewritePattern<vector::ContractionOp>(context, benefit) {}

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::ContractionOp op, MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override;
};

/// Progressively lower a `vector.contract %a, %b, %c` with row-major matmul
/// semantics to:
/// ```
///    %mta = maybe_transpose
///    %mtb = maybe_transpose
///    %flattened_a = vector.shape_cast %mta
///    %flattened_b = vector.shape_cast %mtb
///    %flattened_d = llvm.intr.matrix.multiply %flattened_a, %flattened_b
///    %mtd = vector.shape_cast %flattened_d
///    %d = maybe_untranspose %mtd
///    %e = add %c, %d
/// ```
//
/// This only kicks in when vectorContractLowering is set to `Matmul`.
/// vector.transpose operations are inserted if the vector.contract op is not a
/// row-major matrix multiply.
///
/// Scalable vectors are not supported.
FailureOr<Value> ContractionOpToMatmulOpLowering::matchAndRewriteMaskableOp(
    vector::ContractionOp op, MaskingOpInterface maskOp,
    PatternRewriter &rew) const {
  // TODO: Support vector.mask.
  if (maskOp)
    return failure();

  auto iteratorTypes = op.getIteratorTypes().getValue();
  if (!isParallelIterator(iteratorTypes[0]) ||
      !isParallelIterator(iteratorTypes[1]) ||
      !isReductionIterator(iteratorTypes[2]))
    return failure();

  Type opResType = op.getType();
  VectorType vecType = dyn_cast<VectorType>(opResType);
  if (vecType && vecType.isScalable()) {
    // Note - this is sufficient to reject all cases with scalable vectors.
    return failure();
  }

  Type elementType = op.getLhsType().getElementType();
  if (!elementType.isIntOrFloat())
    return failure();

  Type dstElementType = vecType ? vecType.getElementType() : opResType;
  if (elementType != dstElementType)
    return failure();

  // Perform lhs + rhs transpositions to conform to matmul row-major semantics.
  // Bail out if the contraction cannot be put in this form.
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  AffineExpr m, n, k;
  bindDims(rew.getContext(), m, n, k);
  // LHS must be A(m, k) or A(k, m).
  Value lhs = op.getLhs();
  auto lhsMap = op.getIndexingMapsArray()[0];
  if (lhsMap == AffineMap::get(3, 0, {k, m}, ctx))
    lhs = vector::TransposeOp::create(rew, loc, lhs, ArrayRef<int64_t>{1, 0});
  else if (lhsMap != AffineMap::get(3, 0, {m, k}, ctx))
    return failure();

  // RHS must be B(k, n) or B(n, k).
  Value rhs = op.getRhs();
  auto rhsMap = op.getIndexingMapsArray()[1];
  if (rhsMap == AffineMap::get(3, 0, {n, k}, ctx))
    rhs = vector::TransposeOp::create(rew, loc, rhs, ArrayRef<int64_t>{1, 0});
  else if (rhsMap != AffineMap::get(3, 0, {k, n}, ctx))
    return failure();

  // At this point lhs and rhs are in row-major.
  VectorType lhsType = cast<VectorType>(lhs.getType());
  VectorType rhsType = cast<VectorType>(rhs.getType());
  int64_t lhsRows = lhsType.getDimSize(0);
  int64_t lhsColumns = lhsType.getDimSize(1);
  int64_t rhsColumns = rhsType.getDimSize(1);

  Type flattenedLHSType =
      VectorType::get(lhsType.getNumElements(), lhsType.getElementType());
  lhs = vector::ShapeCastOp::create(rew, loc, flattenedLHSType, lhs);

  Type flattenedRHSType =
      VectorType::get(rhsType.getNumElements(), rhsType.getElementType());
  rhs = vector::ShapeCastOp::create(rew, loc, flattenedRHSType, rhs);

  Value mul = LLVM::MatrixMultiplyOp::create(
      rew, loc,
      VectorType::get(lhsRows * rhsColumns,
                      cast<VectorType>(lhs.getType()).getElementType()),
      lhs, rhs, lhsRows, lhsColumns, rhsColumns);

  mul = vector::ShapeCastOp::create(
      rew, loc,
      VectorType::get({lhsRows, rhsColumns},
                      getElementTypeOrSelf(op.getAcc().getType())),
      mul);

  // ACC must be C(m, n) or C(n, m).
  auto accMap = op.getIndexingMapsArray()[2];
  if (accMap == AffineMap::get(3, 0, {n, m}, ctx))
    mul = vector::TransposeOp::create(rew, loc, mul, ArrayRef<int64_t>{1, 0});
  else if (accMap != AffineMap::get(3, 0, {m, n}, ctx))
    llvm_unreachable("invalid contraction semantics");

  Value res = isa<IntegerType>(elementType)
                  ? static_cast<Value>(
                        arith::AddIOp::create(rew, loc, op.getAcc(), mul))
                  : static_cast<Value>(
                        arith::AddFOp::create(rew, loc, op.getAcc(), mul));

  return res;
}

/// Lowers vector.transpose to llvm.intr.matrix.transpose
class TransposeOpToMatrixTransposeOpLowering
    : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value input = op.getVector();
    VectorType inputType = op.getSourceVectorType();
    VectorType resType = op.getResultVectorType();

    if (inputType.isScalable())
      return rewriter.notifyMatchFailure(
          op, "This lowering does not support scalable vectors");

    // Set up convenience transposition table.
    ArrayRef<int64_t> transp = op.getPermutation();

    if (resType.getRank() != 2 || transp[0] != 1 || transp[1] != 0) {
      return failure();
    }

    Type flattenedType =
        VectorType::get(resType.getNumElements(), resType.getElementType());
    auto matrix =
        vector::ShapeCastOp::create(rewriter, loc, flattenedType, input);
    auto rows = rewriter.getI32IntegerAttr(resType.getShape()[0]);
    auto columns = rewriter.getI32IntegerAttr(resType.getShape()[1]);
    Value trans = LLVM::MatrixTransposeOp::create(rewriter, loc, flattenedType,
                                                  matrix, rows, columns);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resType, trans);
    return success();
  }
};

/// Convert `vector.splat` to `vector.broadcast`. There is a path to LLVM from
/// `vector.broadcast` through other patterns.
struct VectorSplatToBroadcast : public ConvertOpToLLVMPattern<vector::SplatOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(vector::SplatOp splat, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(splat, splat.getType(),
                                                     adaptor.getInput());
    return success();
  }
};

} // namespace

void mlir::vector::populateVectorRankReducingFMAPattern(
    RewritePatternSet &patterns) {
  patterns.add<VectorFMAOpNDRewritePattern>(patterns.getContext());
}

void mlir::vector::populateVectorContractToMatrixMultiply(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ContractionOpToMatmulOpLowering>(patterns.getContext(), benefit);
}

void mlir::vector::populateVectorTransposeToFlatTranspose(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<TransposeOpToMatrixTransposeOpLowering>(patterns.getContext(),
                                                       benefit);
}

/// Populate the given list with patterns that convert from Vector to LLVM.
void mlir::populateVectorToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    bool reassociateFPReductions, bool force32BitVectorIndices,
    bool useVectorAlignment) {
  // This function populates only ConversionPatterns, not RewritePatterns.
  MLIRContext *ctx = converter.getDialect()->getContext();
  patterns.add<VectorReductionOpConversion>(converter, reassociateFPReductions);
  patterns.add<VectorCreateMaskOpConversion>(ctx, force32BitVectorIndices);
  patterns.add<VectorLoadStoreConversion<vector::LoadOp>,
               VectorLoadStoreConversion<vector::MaskedLoadOp>,
               VectorLoadStoreConversion<vector::StoreOp>,
               VectorLoadStoreConversion<vector::MaskedStoreOp>,
               VectorGatherOpConversion, VectorScatterOpConversion>(
      converter, useVectorAlignment);
  patterns.add<VectorBitCastOpConversion, VectorShuffleOpConversion,
               VectorExtractOpConversion, VectorFMAOp1DConversion,
               VectorInsertOpConversion, VectorPrintOpConversion,
               VectorTypeCastOpConversion, VectorScaleOpConversion,
               VectorExpandLoadOpConversion, VectorCompressStoreOpConversion,
               VectorSplatToBroadcast, VectorBroadcastScalarToLowRankLowering,
               VectorBroadcastScalarToNdLowering,
               VectorScalableInsertOpLowering, VectorScalableExtractOpLowering,
               MaskedReductionOpConversion, VectorInterleaveOpLowering,
               VectorDeinterleaveOpLowering, VectorFromElementsLowering,
               VectorToElementsLowering, VectorScalableStepOpLowering>(
      converter);
}

namespace {
struct VectorToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  }
};
} // namespace

void mlir::vector::registerConvertVectorToLLVMInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, vector::VectorDialect *dialect) {
    dialect->addInterfaces<VectorToLLVMDialectInterface>();
  });
}
