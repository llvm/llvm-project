//===- VectorToLLVM.cpp - Conversion from Vector to the LLVM dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::vector;

// Helper to reduce vector type by one rank at front.
static VectorType reducedVectorTypeFront(VectorType tp) {
  assert((tp.getRank() > 1) && "unlowerable vector type");
  return VectorType::get(tp.getShape().drop_front(), tp.getElementType());
}

// Helper to reduce vector type by *all* but one rank at back.
static VectorType reducedVectorTypeBack(VectorType tp) {
  assert((tp.getRank() > 1) && "unlowerable vector type");
  return VectorType::get(tp.getShape().take_back(), tp.getElementType());
}

// Helper that picks the proper sequence for inserting.
static Value insertOne(ConversionPatternRewriter &rewriter,
                       LLVMTypeConverter &typeConverter, Location loc,
                       Value val1, Value val2, Type llvmType, int64_t rank,
                       int64_t pos) {
  if (rank == 1) {
    auto idxType = rewriter.getIndexType();
    auto constant = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter.convertType(idxType),
        rewriter.getIntegerAttr(idxType, pos));
    return rewriter.create<LLVM::InsertElementOp>(loc, llvmType, val1, val2,
                                                  constant);
  }
  return rewriter.create<LLVM::InsertValueOp>(loc, llvmType, val1, val2,
                                              rewriter.getI64ArrayAttr(pos));
}

// Helper that picks the proper sequence for extracting.
static Value extractOne(ConversionPatternRewriter &rewriter,
                        LLVMTypeConverter &typeConverter, Location loc,
                        Value val, Type llvmType, int64_t rank, int64_t pos) {
  if (rank == 1) {
    auto idxType = rewriter.getIndexType();
    auto constant = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter.convertType(idxType),
        rewriter.getIntegerAttr(idxType, pos));
    return rewriter.create<LLVM::ExtractElementOp>(loc, llvmType, val,
                                                   constant);
  }
  return rewriter.create<LLVM::ExtractValueOp>(loc, llvmType, val,
                                               rewriter.getI64ArrayAttr(pos));
}

// Helper that returns data layout alignment of a memref.
LogicalResult getMemRefAlignment(LLVMTypeConverter &typeConverter,
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

// Return the minimal alignment value that satisfies all the AssumeAlignment
// uses of `value`. If no such uses exist, return 1.
static unsigned getAssumedAlignment(Value value) {
  unsigned align = 1;
  for (auto &u : value.getUses()) {
    Operation *owner = u.getOwner();
    if (auto op = dyn_cast<memref::AssumeAlignmentOp>(owner))
      align = mlir::lcm(align, op.alignment());
  }
  return align;
}

// Helper that returns data layout alignment of a memref associated with a
// load, store, scatter, or gather op, including additional information from
// assume_alignment calls on the source of the transfer
template <class OpAdaptor>
LogicalResult getMemRefOpAlignment(LLVMTypeConverter &typeConverter,
                                   OpAdaptor op, unsigned &align) {
  if (failed(getMemRefAlignment(typeConverter, op.getMemRefType(), align)))
    return failure();
  align = std::max(align, getAssumedAlignment(op.base()));
  return success();
}

// Add an index vector component to a base pointer. This almost always succeeds
// unless the last stride is non-unit or the memory space is not zero.
static LogicalResult getIndexedPtrs(ConversionPatternRewriter &rewriter,
                                    Location loc, Value memref, Value base,
                                    Value index, MemRefType memRefType,
                                    VectorType vType, Value &ptrs) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto successStrides = getStridesAndOffset(memRefType, strides, offset);
  if (failed(successStrides) || strides.back() != 1 ||
      memRefType.getMemorySpaceAsInt() != 0)
    return failure();
  auto pType = MemRefDescriptor(memref).getElementPtrType();
  auto ptrsType = LLVM::getFixedVectorType(pType, vType.getDimSize(0));
  ptrs = rewriter.create<LLVM::GEPOp>(loc, ptrsType, base, index);
  return success();
}

// Casts a strided element pointer to a vector pointer.  The vector pointer
// will be in the same address space as the incoming memref type.
static Value castDataPtr(ConversionPatternRewriter &rewriter, Location loc,
                         Value ptr, MemRefType memRefType, Type vt) {
  auto pType = LLVM::LLVMPointerType::get(vt, memRefType.getMemorySpaceAsInt());
  return rewriter.create<LLVM::BitcastOp>(loc, pType, ptr);
}

namespace {

/// Conversion pattern for a vector.bitcast.
class VectorBitCastOpConversion
    : public ConvertOpToLLVMPattern<vector::BitCastOp> {
public:
  using ConvertOpToLLVMPattern<vector::BitCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::BitCastOp bitCastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only 1-D vectors can be lowered to LLVM.
    VectorType resultTy = bitCastOp.getType();
    if (resultTy.getRank() != 1)
      return failure();
    Type newResultTy = typeConverter->convertType(resultTy);
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(bitCastOp, newResultTy,
                                                 adaptor.getOperands()[0]);
    return success();
  }
};

/// Conversion pattern for a vector.matrix_multiply.
/// This is lowered directly to the proper llvm.intr.matrix.multiply.
class VectorMatmulOpConversion
    : public ConvertOpToLLVMPattern<vector::MatmulOp> {
public:
  using ConvertOpToLLVMPattern<vector::MatmulOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::MatmulOp matmulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::MatrixMultiplyOp>(
        matmulOp, typeConverter->convertType(matmulOp.res().getType()),
        adaptor.lhs(), adaptor.rhs(), matmulOp.lhs_rows(),
        matmulOp.lhs_columns(), matmulOp.rhs_columns());
    return success();
  }
};

/// Conversion pattern for a vector.flat_transpose.
/// This is lowered directly to the proper llvm.intr.matrix.transpose.
class VectorFlatTransposeOpConversion
    : public ConvertOpToLLVMPattern<vector::FlatTransposeOp> {
public:
  using ConvertOpToLLVMPattern<vector::FlatTransposeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::FlatTransposeOp transOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::MatrixTransposeOp>(
        transOp, typeConverter->convertType(transOp.res().getType()),
        adaptor.matrix(), transOp.rows(), transOp.columns());
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
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(loadOp, ptr, align);
}

static void replaceLoadOrStoreOp(vector::MaskedLoadOp loadOp,
                                 vector::MaskedLoadOpAdaptor adaptor,
                                 VectorType vectorTy, Value ptr, unsigned align,
                                 ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<LLVM::MaskedLoadOp>(
      loadOp, vectorTy, ptr, adaptor.mask(), adaptor.pass_thru(), align);
}

static void replaceLoadOrStoreOp(vector::StoreOp storeOp,
                                 vector::StoreOpAdaptor adaptor,
                                 VectorType vectorTy, Value ptr, unsigned align,
                                 ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, adaptor.valueToStore(),
                                             ptr, align);
}

static void replaceLoadOrStoreOp(vector::MaskedStoreOp storeOp,
                                 vector::MaskedStoreOpAdaptor adaptor,
                                 VectorType vectorTy, Value ptr, unsigned align,
                                 ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<LLVM::MaskedStoreOp>(
      storeOp, adaptor.valueToStore(), ptr, adaptor.mask(), align);
}

/// Conversion pattern for a vector.load, vector.store, vector.maskedload, and
/// vector.maskedstore.
template <class LoadOrStoreOp, class LoadOrStoreOpAdaptor>
class VectorLoadStoreConversion : public ConvertOpToLLVMPattern<LoadOrStoreOp> {
public:
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
    unsigned align;
    if (failed(getMemRefOpAlignment(*this->getTypeConverter(), loadOrStoreOp,
                                    align)))
      return failure();

    // Resolve address.
    auto vtype = this->typeConverter->convertType(loadOrStoreOp.getVectorType())
                     .template cast<VectorType>();
    Value dataPtr = this->getStridedElementPtr(loc, memRefTy, adaptor.base(),
                                               adaptor.indices(), rewriter);
    Value ptr = castDataPtr(rewriter, loc, dataPtr, memRefTy, vtype);

    replaceLoadOrStoreOp(loadOrStoreOp, adaptor, vtype, ptr, align, rewriter);
    return success();
  }
};

/// Conversion pattern for a vector.gather.
class VectorGatherOpConversion
    : public ConvertOpToLLVMPattern<vector::GatherOp> {
public:
  using ConvertOpToLLVMPattern<vector::GatherOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::GatherOp gather, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = gather->getLoc();
    MemRefType memRefType = gather.getMemRefType();

    // Resolve alignment.
    unsigned align;
    if (failed(getMemRefOpAlignment(*getTypeConverter(), gather, align)))
      return failure();

    // Resolve address.
    Value ptrs;
    VectorType vType = gather.getVectorType();
    Value ptr = getStridedElementPtr(loc, memRefType, adaptor.base(),
                                     adaptor.indices(), rewriter);
    if (failed(getIndexedPtrs(rewriter, loc, adaptor.base(), ptr,
                              adaptor.index_vec(), memRefType, vType, ptrs)))
      return failure();

    // Replace with the gather intrinsic.
    rewriter.replaceOpWithNewOp<LLVM::masked_gather>(
        gather, typeConverter->convertType(vType), ptrs, adaptor.mask(),
        adaptor.pass_thru(), rewriter.getI32IntegerAttr(align));
    return success();
  }
};

/// Conversion pattern for a vector.scatter.
class VectorScatterOpConversion
    : public ConvertOpToLLVMPattern<vector::ScatterOp> {
public:
  using ConvertOpToLLVMPattern<vector::ScatterOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::ScatterOp scatter, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = scatter->getLoc();
    MemRefType memRefType = scatter.getMemRefType();

    // Resolve alignment.
    unsigned align;
    if (failed(getMemRefOpAlignment(*getTypeConverter(), scatter, align)))
      return failure();

    // Resolve address.
    Value ptrs;
    VectorType vType = scatter.getVectorType();
    Value ptr = getStridedElementPtr(loc, memRefType, adaptor.base(),
                                     adaptor.indices(), rewriter);
    if (failed(getIndexedPtrs(rewriter, loc, adaptor.base(), ptr,
                              adaptor.index_vec(), memRefType, vType, ptrs)))
      return failure();

    // Replace with the scatter intrinsic.
    rewriter.replaceOpWithNewOp<LLVM::masked_scatter>(
        scatter, adaptor.valueToStore(), ptrs, adaptor.mask(),
        rewriter.getI32IntegerAttr(align));
    return success();
  }
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
    Value ptr = getStridedElementPtr(loc, memRefType, adaptor.base(),
                                     adaptor.indices(), rewriter);

    rewriter.replaceOpWithNewOp<LLVM::masked_expandload>(
        expand, vtype, ptr, adaptor.mask(), adaptor.pass_thru());
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
    Value ptr = getStridedElementPtr(loc, memRefType, adaptor.base(),
                                     adaptor.indices(), rewriter);

    rewriter.replaceOpWithNewOp<LLVM::masked_compressstore>(
        compress, adaptor.valueToStore(), ptr, adaptor.mask());
    return success();
  }
};

/// Conversion pattern for all vector reductions.
class VectorReductionOpConversion
    : public ConvertOpToLLVMPattern<vector::ReductionOp> {
public:
  explicit VectorReductionOpConversion(LLVMTypeConverter &typeConv,
                                       bool reassociateFPRed)
      : ConvertOpToLLVMPattern<vector::ReductionOp>(typeConv),
        reassociateFPReductions(reassociateFPRed) {}

  LogicalResult
  matchAndRewrite(vector::ReductionOp reductionOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kind = reductionOp.kind();
    Type eltType = reductionOp.dest().getType();
    Type llvmType = typeConverter->convertType(eltType);
    Value operand = adaptor.getOperands()[0];
    if (eltType.isIntOrIndex()) {
      // Integer reductions: add/mul/min/max/and/or/xor.
      if (kind == "add")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_add>(reductionOp,
                                                             llvmType, operand);
      else if (kind == "mul")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_mul>(reductionOp,
                                                             llvmType, operand);
      else if (kind == "minui")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_umin>(
            reductionOp, llvmType, operand);
      else if (kind == "minsi")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_smin>(
            reductionOp, llvmType, operand);
      else if (kind == "maxui")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_umax>(
            reductionOp, llvmType, operand);
      else if (kind == "maxsi")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_smax>(
            reductionOp, llvmType, operand);
      else if (kind == "and")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_and>(reductionOp,
                                                             llvmType, operand);
      else if (kind == "or")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_or>(reductionOp,
                                                            llvmType, operand);
      else if (kind == "xor")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_xor>(reductionOp,
                                                             llvmType, operand);
      else
        return failure();
      return success();
    }

    if (!eltType.isa<FloatType>())
      return failure();

    // Floating-point reductions: add/mul/min/max
    if (kind == "add") {
      // Optional accumulator (or zero).
      Value acc = adaptor.getOperands().size() > 1
                      ? adaptor.getOperands()[1]
                      : rewriter.create<LLVM::ConstantOp>(
                            reductionOp->getLoc(), llvmType,
                            rewriter.getZeroAttr(eltType));
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_fadd>(
          reductionOp, llvmType, acc, operand,
          rewriter.getBoolAttr(reassociateFPReductions));
    } else if (kind == "mul") {
      // Optional accumulator (or one).
      Value acc = adaptor.getOperands().size() > 1
                      ? adaptor.getOperands()[1]
                      : rewriter.create<LLVM::ConstantOp>(
                            reductionOp->getLoc(), llvmType,
                            rewriter.getFloatAttr(eltType, 1.0));
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_fmul>(
          reductionOp, llvmType, acc, operand,
          rewriter.getBoolAttr(reassociateFPReductions));
    } else if (kind == "minf")
      // FIXME: MLIR's 'minf' and LLVM's 'vector_reduce_fmin' do not handle
      // NaNs/-0.0/+0.0 in the same way.
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_fmin>(reductionOp,
                                                            llvmType, operand);
    else if (kind == "maxf")
      // FIXME: MLIR's 'maxf' and LLVM's 'vector_reduce_fmax' do not handle
      // NaNs/-0.0/+0.0 in the same way.
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_fmax>(reductionOp,
                                                            llvmType, operand);
    else
      return failure();
    return success();
  }

private:
  const bool reassociateFPReductions;
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
    auto vectorType = shuffleOp.getVectorType();
    Type llvmType = typeConverter->convertType(vectorType);
    auto maskArrayAttr = shuffleOp.mask();

    // Bail if result type cannot be lowered.
    if (!llvmType)
      return failure();

    // Get rank and dimension sizes.
    int64_t rank = vectorType.getRank();
    assert(v1Type.getRank() == rank);
    assert(v2Type.getRank() == rank);
    int64_t v1Dim = v1Type.getDimSize(0);

    // For rank 1, where both operands have *exactly* the same vector type,
    // there is direct shuffle support in LLVM. Use it!
    if (rank == 1 && v1Type == v2Type) {
      Value llvmShuffleOp = rewriter.create<LLVM::ShuffleVectorOp>(
          loc, adaptor.v1(), adaptor.v2(), maskArrayAttr);
      rewriter.replaceOp(shuffleOp, llvmShuffleOp);
      return success();
    }

    // For all other cases, insert the individual values individually.
    Type eltType;
    llvm::errs() << llvmType << "\n";
    if (auto arrayType = llvmType.dyn_cast<LLVM::LLVMArrayType>())
      eltType = arrayType.getElementType();
    else
      eltType = llvmType.cast<VectorType>().getElementType();
    Value insert = rewriter.create<LLVM::UndefOp>(loc, llvmType);
    int64_t insPos = 0;
    for (auto en : llvm::enumerate(maskArrayAttr)) {
      int64_t extPos = en.value().cast<IntegerAttr>().getInt();
      Value value = adaptor.v1();
      if (extPos >= v1Dim) {
        extPos -= v1Dim;
        value = adaptor.v2();
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

class VectorExtractElementOpConversion
    : public ConvertOpToLLVMPattern<vector::ExtractElementOp> {
public:
  using ConvertOpToLLVMPattern<
      vector::ExtractElementOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractElementOp extractEltOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vectorType = extractEltOp.getVectorType();
    auto llvmType = typeConverter->convertType(vectorType.getElementType());

    // Bail if result type cannot be lowered.
    if (!llvmType)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::ExtractElementOp>(
        extractEltOp, llvmType, adaptor.vector(), adaptor.position());
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
    auto vectorType = extractOp.getVectorType();
    auto resultType = extractOp.getResult().getType();
    auto llvmResultType = typeConverter->convertType(resultType);
    auto positionArrayAttr = extractOp.position();

    // Bail if result type cannot be lowered.
    if (!llvmResultType)
      return failure();

    // Extract entire vector. Should be handled by folder, but just to be safe.
    if (positionArrayAttr.empty()) {
      rewriter.replaceOp(extractOp, adaptor.vector());
      return success();
    }

    // One-shot extraction of vector from array (only requires extractvalue).
    if (resultType.isa<VectorType>()) {
      Value extracted = rewriter.create<LLVM::ExtractValueOp>(
          loc, llvmResultType, adaptor.vector(), positionArrayAttr);
      rewriter.replaceOp(extractOp, extracted);
      return success();
    }

    // Potential extraction of 1-D vector from array.
    auto *context = extractOp->getContext();
    Value extracted = adaptor.vector();
    auto positionAttrs = positionArrayAttr.getValue();
    if (positionAttrs.size() > 1) {
      auto oneDVectorType = reducedVectorTypeBack(vectorType);
      auto nMinusOnePositionAttrs =
          ArrayAttr::get(context, positionAttrs.drop_back());
      extracted = rewriter.create<LLVM::ExtractValueOp>(
          loc, typeConverter->convertType(oneDVectorType), extracted,
          nMinusOnePositionAttrs);
    }

    // Remaining extraction of element from 1-D LLVM vector
    auto position = positionAttrs.back().cast<IntegerAttr>();
    auto i64Type = IntegerType::get(rewriter.getContext(), 64);
    auto constant = rewriter.create<LLVM::ConstantOp>(loc, i64Type, position);
    extracted =
        rewriter.create<LLVM::ExtractElementOp>(loc, extracted, constant);
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
    if (vType.getRank() != 1)
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::FMulAddOp>(fmaOp, adaptor.lhs(),
                                                 adaptor.rhs(), adaptor.acc());
    return success();
  }
};

class VectorInsertElementOpConversion
    : public ConvertOpToLLVMPattern<vector::InsertElementOp> {
public:
  using ConvertOpToLLVMPattern<vector::InsertElementOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::InsertElementOp insertEltOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vectorType = insertEltOp.getDestVectorType();
    auto llvmType = typeConverter->convertType(vectorType);

    // Bail if result type cannot be lowered.
    if (!llvmType)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::InsertElementOp>(
        insertEltOp, llvmType, adaptor.dest(), adaptor.source(),
        adaptor.position());
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
    auto sourceType = insertOp.getSourceType();
    auto destVectorType = insertOp.getDestVectorType();
    auto llvmResultType = typeConverter->convertType(destVectorType);
    auto positionArrayAttr = insertOp.position();

    // Bail if result type cannot be lowered.
    if (!llvmResultType)
      return failure();

    // Overwrite entire vector with value. Should be handled by folder, but
    // just to be safe.
    if (positionArrayAttr.empty()) {
      rewriter.replaceOp(insertOp, adaptor.source());
      return success();
    }

    // One-shot insertion of a vector into an array (only requires insertvalue).
    if (sourceType.isa<VectorType>()) {
      Value inserted = rewriter.create<LLVM::InsertValueOp>(
          loc, llvmResultType, adaptor.dest(), adaptor.source(),
          positionArrayAttr);
      rewriter.replaceOp(insertOp, inserted);
      return success();
    }

    // Potential extraction of 1-D vector from array.
    auto *context = insertOp->getContext();
    Value extracted = adaptor.dest();
    auto positionAttrs = positionArrayAttr.getValue();
    auto position = positionAttrs.back().cast<IntegerAttr>();
    auto oneDVectorType = destVectorType;
    if (positionAttrs.size() > 1) {
      oneDVectorType = reducedVectorTypeBack(destVectorType);
      auto nMinusOnePositionAttrs =
          ArrayAttr::get(context, positionAttrs.drop_back());
      extracted = rewriter.create<LLVM::ExtractValueOp>(
          loc, typeConverter->convertType(oneDVectorType), extracted,
          nMinusOnePositionAttrs);
    }

    // Insertion of an element into a 1-D LLVM vector.
    auto i64Type = IntegerType::get(rewriter.getContext(), 64);
    auto constant = rewriter.create<LLVM::ConstantOp>(loc, i64Type, position);
    Value inserted = rewriter.create<LLVM::InsertElementOp>(
        loc, typeConverter->convertType(oneDVectorType), extracted,
        adaptor.source(), constant);

    // Potential insertion of resulting 1-D vector into array.
    if (positionAttrs.size() > 1) {
      auto nMinusOnePositionAttrs =
          ArrayAttr::get(context, positionAttrs.drop_back());
      inserted = rewriter.create<LLVM::InsertValueOp>(loc, llvmResultType,
                                                      adaptor.dest(), inserted,
                                                      nMinusOnePositionAttrs);
    }

    rewriter.replaceOp(insertOp, inserted);
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
///  %r = splat %f0: vector<2x4xf32>
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

  LogicalResult matchAndRewrite(FMAOp op,
                                PatternRewriter &rewriter) const override {
    auto vType = op.getVectorType();
    if (vType.getRank() < 2)
      return failure();

    auto loc = op.getLoc();
    auto elemType = vType.getElementType();
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getZeroAttr(elemType));
    Value desc = rewriter.create<SplatOp>(loc, vType, zero);
    for (int64_t i = 0, e = vType.getShape().front(); i != e; ++i) {
      Value extrLHS = rewriter.create<ExtractOp>(loc, op.lhs(), i);
      Value extrRHS = rewriter.create<ExtractOp>(loc, op.rhs(), i);
      Value extrACC = rewriter.create<ExtractOp>(loc, op.acc(), i);
      Value fma = rewriter.create<FMAOp>(loc, extrLHS, extrRHS, extrACC);
      desc = rewriter.create<InsertOp>(loc, fma, desc, i);
    }
    rewriter.replaceOp(op, desc);
    return success();
  }
};

/// Returns the strides if the memory underlying `memRefType` has a contiguous
/// static layout.
static llvm::Optional<SmallVector<int64_t, 4>>
computeContiguousStrides(MemRefType memRefType) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(memRefType, strides, offset)))
    return None;
  if (!strides.empty() && strides.back() != 1)
    return None;
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
        ShapedType::isDynamicStrideOrOffset(strides[index]) ||
        ShapedType::isDynamicStrideOrOffset(strides[index + 1]))
      return None;
    if (strides[index] != strides[index + 1] * sizes[index + 1])
      return None;
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
        castOp.getOperand().getType().cast<MemRefType>();
    MemRefType targetMemRefType = castOp.getType();

    // Only static shape casts supported atm.
    if (!sourceMemRefType.hasStaticShape() ||
        !targetMemRefType.hasStaticShape())
      return failure();

    auto llvmSourceDescriptorTy =
        adaptor.getOperands()[0].getType().dyn_cast<LLVM::LLVMStructType>();
    if (!llvmSourceDescriptorTy)
      return failure();
    MemRefDescriptor sourceMemRef(adaptor.getOperands()[0]);

    auto llvmTargetDescriptorTy = typeConverter->convertType(targetMemRefType)
                                      .dyn_cast_or_null<LLVM::LLVMStructType>();
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
    if (llvm::any_of(*targetStrides, [](int64_t stride) {
          return ShapedType::isDynamicStrideOrOffset(stride);
        }))
      return failure();

    auto int64Ty = IntegerType::get(rewriter.getContext(), 64);

    // Create descriptor.
    auto desc = MemRefDescriptor::undef(rewriter, loc, llvmTargetDescriptorTy);
    Type llvmTargetElementTy = desc.getElementPtrType();
    // Set allocated ptr.
    Value allocated = sourceMemRef.allocatedPtr(rewriter, loc);
    allocated =
        rewriter.create<LLVM::BitcastOp>(loc, llvmTargetElementTy, allocated);
    desc.setAllocatedPtr(rewriter, loc, allocated);
    // Set aligned ptr.
    Value ptr = sourceMemRef.alignedPtr(rewriter, loc);
    ptr = rewriter.create<LLVM::BitcastOp>(loc, llvmTargetElementTy, ptr);
    desc.setAlignedPtr(rewriter, loc, ptr);
    // Fill offset 0.
    auto attr = rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
    auto zero = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, attr);
    desc.setOffset(rewriter, loc, zero);

    // Fill size and stride descriptors in memref.
    for (auto indexedSize : llvm::enumerate(targetMemRefType.getShape())) {
      int64_t index = indexedSize.index();
      auto sizeAttr =
          rewriter.getIntegerAttr(rewriter.getIndexType(), indexedSize.value());
      auto size = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, sizeAttr);
      desc.setSize(rewriter, loc, index, size);
      auto strideAttr = rewriter.getIntegerAttr(rewriter.getIndexType(),
                                                (*targetStrides)[index]);
      auto stride = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, strideAttr);
      desc.setStride(rewriter, loc, index, stride);
    }

    rewriter.replaceOp(castOp, {desc});
    return success();
  }
};

class VectorPrintOpConversion : public ConvertOpToLLVMPattern<vector::PrintOp> {
public:
  using ConvertOpToLLVMPattern<vector::PrintOp>::ConvertOpToLLVMPattern;

  // Proof-of-concept lowering implementation that relies on a small
  // runtime support library, which only needs to provide a few
  // printing methods (single value for all data types, opening/closing
  // bracket, comma, newline). The lowering fully unrolls a vector
  // in terms of these elementary printing operations. The advantage
  // of this approach is that the library can remain unaware of all
  // low-level implementation details of vectors while still supporting
  // output of any shaped and dimensioned vector. Due to full unrolling,
  // this approach is less suited for very large vectors though.
  //
  // TODO: rely solely on libc in future? something else?
  //
  LogicalResult
  matchAndRewrite(vector::PrintOp printOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type printType = printOp.getPrintType();

    if (typeConverter->convertType(printType) == nullptr)
      return failure();

    // Make sure element type has runtime support.
    PrintConversion conversion = PrintConversion::None;
    VectorType vectorType = printType.dyn_cast<VectorType>();
    Type eltType = vectorType ? vectorType.getElementType() : printType;
    Operation *printer;
    if (eltType.isF32()) {
      printer =
          LLVM::lookupOrCreatePrintF32Fn(printOp->getParentOfType<ModuleOp>());
    } else if (eltType.isF64()) {
      printer =
          LLVM::lookupOrCreatePrintF64Fn(printOp->getParentOfType<ModuleOp>());
    } else if (eltType.isIndex()) {
      printer =
          LLVM::lookupOrCreatePrintU64Fn(printOp->getParentOfType<ModuleOp>());
    } else if (auto intTy = eltType.dyn_cast<IntegerType>()) {
      // Integers need a zero or sign extension on the operand
      // (depending on the source type) as well as a signed or
      // unsigned print method. Up to 64-bit is supported.
      unsigned width = intTy.getWidth();
      if (intTy.isUnsigned()) {
        if (width <= 64) {
          if (width < 64)
            conversion = PrintConversion::ZeroExt64;
          printer = LLVM::lookupOrCreatePrintU64Fn(
              printOp->getParentOfType<ModuleOp>());
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
          printer = LLVM::lookupOrCreatePrintI64Fn(
              printOp->getParentOfType<ModuleOp>());
        } else {
          return failure();
        }
      }
    } else {
      return failure();
    }

    // Unroll vector into elementary print calls.
    int64_t rank = vectorType ? vectorType.getRank() : 0;
    emitRanks(rewriter, printOp, adaptor.source(), vectorType, printer, rank,
              conversion);
    emitCall(rewriter, printOp->getLoc(),
             LLVM::lookupOrCreatePrintNewlineFn(
                 printOp->getParentOfType<ModuleOp>()));
    rewriter.eraseOp(printOp);
    return success();
  }

private:
  enum class PrintConversion {
    // clang-format off
    None,
    ZeroExt64,
    SignExt64
    // clang-format on
  };

  void emitRanks(ConversionPatternRewriter &rewriter, Operation *op,
                 Value value, VectorType vectorType, Operation *printer,
                 int64_t rank, PrintConversion conversion) const {
    Location loc = op->getLoc();
    if (rank == 0) {
      switch (conversion) {
      case PrintConversion::ZeroExt64:
        value = rewriter.create<arith::ExtUIOp>(
            loc, value, IntegerType::get(rewriter.getContext(), 64));
        break;
      case PrintConversion::SignExt64:
        value = rewriter.create<arith::ExtSIOp>(
            loc, value, IntegerType::get(rewriter.getContext(), 64));
        break;
      case PrintConversion::None:
        break;
      }
      emitCall(rewriter, loc, printer, value);
      return;
    }

    emitCall(rewriter, loc,
             LLVM::lookupOrCreatePrintOpenFn(op->getParentOfType<ModuleOp>()));
    Operation *printComma =
        LLVM::lookupOrCreatePrintCommaFn(op->getParentOfType<ModuleOp>());
    int64_t dim = vectorType.getDimSize(0);
    for (int64_t d = 0; d < dim; ++d) {
      auto reducedType =
          rank > 1 ? reducedVectorTypeFront(vectorType) : nullptr;
      auto llvmType = typeConverter->convertType(
          rank > 1 ? reducedType : vectorType.getElementType());
      Value nestedVal = extractOne(rewriter, *getTypeConverter(), loc, value,
                                   llvmType, rank, d);
      emitRanks(rewriter, op, nestedVal, reducedType, printer, rank - 1,
                conversion);
      if (d != dim - 1)
        emitCall(rewriter, loc, printComma);
    }
    emitCall(rewriter, loc,
             LLVM::lookupOrCreatePrintCloseFn(op->getParentOfType<ModuleOp>()));
  }

  // Helper to emit a call.
  static void emitCall(ConversionPatternRewriter &rewriter, Location loc,
                       Operation *ref, ValueRange params = ValueRange()) {
    rewriter.create<LLVM::CallOp>(loc, TypeRange(), SymbolRefAttr::get(ref),
                                  params);
  }
};

} // namespace

/// Populate the given list with patterns that convert from Vector to LLVM.
void mlir::populateVectorToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    bool reassociateFPReductions) {
  MLIRContext *ctx = converter.getDialect()->getContext();
  patterns.add<VectorFMAOpNDRewritePattern>(ctx);
  populateVectorInsertExtractStridedSliceTransforms(patterns);
  patterns.add<VectorReductionOpConversion>(converter, reassociateFPReductions);
  patterns
      .add<VectorBitCastOpConversion, VectorShuffleOpConversion,
           VectorExtractElementOpConversion, VectorExtractOpConversion,
           VectorFMAOp1DConversion, VectorInsertElementOpConversion,
           VectorInsertOpConversion, VectorPrintOpConversion,
           VectorTypeCastOpConversion,
           VectorLoadStoreConversion<vector::LoadOp, vector::LoadOpAdaptor>,
           VectorLoadStoreConversion<vector::MaskedLoadOp,
                                     vector::MaskedLoadOpAdaptor>,
           VectorLoadStoreConversion<vector::StoreOp, vector::StoreOpAdaptor>,
           VectorLoadStoreConversion<vector::MaskedStoreOp,
                                     vector::MaskedStoreOpAdaptor>,
           VectorGatherOpConversion, VectorScatterOpConversion,
           VectorExpandLoadOpConversion, VectorCompressStoreOpConversion>(
          converter);
  // Transfer ops with rank > 1 are handled by VectorToSCF.
  populateVectorTransferLoweringPatterns(patterns, /*maxTransferRank=*/1);
}

void mlir::populateVectorToLLVMMatrixConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<VectorMatmulOpConversion>(converter);
  patterns.add<VectorFlatTransposeOpConversion>(converter);
}
