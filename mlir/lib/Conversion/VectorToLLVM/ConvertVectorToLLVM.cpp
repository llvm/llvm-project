//===- VectorToLLVM.cpp - Conversion from Vector to the LLVM dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Target/LLVMIR/TypeTranslation.h"
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

// Helper that picks the proper sequence for inserting.
static Value insertOne(PatternRewriter &rewriter, Location loc, Value from,
                       Value into, int64_t offset) {
  auto vectorType = into.getType().cast<VectorType>();
  if (vectorType.getRank() > 1)
    return rewriter.create<InsertOp>(loc, from, into, offset);
  return rewriter.create<vector::InsertElementOp>(
      loc, vectorType, from, into,
      rewriter.create<ConstantIndexOp>(loc, offset));
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

// Helper that picks the proper sequence for extracting.
static Value extractOne(PatternRewriter &rewriter, Location loc, Value vector,
                        int64_t offset) {
  auto vectorType = vector.getType().cast<VectorType>();
  if (vectorType.getRank() > 1)
    return rewriter.create<ExtractOp>(loc, vector, offset);
  return rewriter.create<vector::ExtractElementOp>(
      loc, vectorType.getElementType(), vector,
      rewriter.create<ConstantIndexOp>(loc, offset));
}

// Helper that returns a subset of `arrayAttr` as a vector of int64_t.
// TODO: Better support for attribute subtype forwarding + slicing.
static SmallVector<int64_t, 4> getI64SubArray(ArrayAttr arrayAttr,
                                              unsigned dropFront = 0,
                                              unsigned dropBack = 0) {
  assert(arrayAttr.size() > dropFront + dropBack && "Out of bounds");
  auto range = arrayAttr.getAsRange<IntegerAttr>();
  SmallVector<int64_t, 4> res;
  res.reserve(arrayAttr.size() - dropFront - dropBack);
  for (auto it = range.begin() + dropFront, eit = range.end() - dropBack;
       it != eit; ++it)
    res.push_back((*it).getValue().getSExtValue());
  return res;
}

// Helper that returns a vector comparison that constructs a mask:
//     mask = [0,1,..,n-1] + [o,o,..,o] < [b,b,..,b]
//
// NOTE: The LLVM::GetActiveLaneMaskOp intrinsic would provide an alternative,
//       much more compact, IR for this operation, but LLVM eventually
//       generates more elaborate instructions for this intrinsic since it
//       is very conservative on the boundary conditions.
static Value buildVectorComparison(ConversionPatternRewriter &rewriter,
                                   Operation *op, bool enableIndexOptimizations,
                                   int64_t dim, Value b, Value *off = nullptr) {
  auto loc = op->getLoc();
  // If we can assume all indices fit in 32-bit, we perform the vector
  // comparison in 32-bit to get a higher degree of SIMD parallelism.
  // Otherwise we perform the vector comparison using 64-bit indices.
  Value indices;
  Type idxType;
  if (enableIndexOptimizations) {
    indices = rewriter.create<ConstantOp>(
        loc, rewriter.getI32VectorAttr(
                 llvm::to_vector<4>(llvm::seq<int32_t>(0, dim))));
    idxType = rewriter.getI32Type();
  } else {
    indices = rewriter.create<ConstantOp>(
        loc, rewriter.getI64VectorAttr(
                 llvm::to_vector<4>(llvm::seq<int64_t>(0, dim))));
    idxType = rewriter.getI64Type();
  }
  // Add in an offset if requested.
  if (off) {
    Value o = rewriter.create<IndexCastOp>(loc, idxType, *off);
    Value ov = rewriter.create<SplatOp>(loc, indices.getType(), o);
    indices = rewriter.create<AddIOp>(loc, ov, indices);
  }
  // Construct the vector comparison.
  Value bound = rewriter.create<IndexCastOp>(loc, idxType, b);
  Value bounds = rewriter.create<SplatOp>(loc, indices.getType(), bound);
  return rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, indices, bounds);
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

// Helper that returns the base address of a memref.
static LogicalResult getBase(ConversionPatternRewriter &rewriter, Location loc,
                             Value memref, MemRefType memRefType, Value &base) {
  // Inspect stride and offset structure.
  //
  // TODO: flat memory only for now, generalize
  //
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto successStrides = getStridesAndOffset(memRefType, strides, offset);
  if (failed(successStrides) || strides.size() != 1 || strides[0] != 1 ||
      offset != 0 || memRefType.getMemorySpace() != 0)
    return failure();
  base = MemRefDescriptor(memref).alignedPtr(rewriter, loc);
  return success();
}

// Helper that returns vector of pointers given a memref base with index vector.
static LogicalResult getIndexedPtrs(ConversionPatternRewriter &rewriter,
                                    Location loc, Value memref, Value indices,
                                    MemRefType memRefType, VectorType vType,
                                    Type iType, Value &ptrs) {
  Value base;
  if (failed(getBase(rewriter, loc, memref, memRefType, base)))
    return failure();
  auto pType = MemRefDescriptor(memref).getElementPtrType();
  auto ptrsType = LLVM::getFixedVectorType(pType, vType.getDimSize(0));
  ptrs = rewriter.create<LLVM::GEPOp>(loc, ptrsType, base, indices);
  return success();
}

// Casts a strided element pointer to a vector pointer. The vector pointer
// would always be on address space 0, therefore addrspacecast shall be
// used when source/dst memrefs are not on address space 0.
static Value castDataPtr(ConversionPatternRewriter &rewriter, Location loc,
                         Value ptr, MemRefType memRefType, Type vt) {
  auto pType = LLVM::LLVMPointerType::get(vt);
  if (memRefType.getMemorySpace() == 0)
    return rewriter.create<LLVM::BitcastOp>(loc, pType, ptr);
  return rewriter.create<LLVM::AddrSpaceCastOp>(loc, pType, ptr);
}

static LogicalResult
replaceTransferOpWithLoadOrStore(ConversionPatternRewriter &rewriter,
                                 LLVMTypeConverter &typeConverter, Location loc,
                                 TransferReadOp xferOp,
                                 ArrayRef<Value> operands, Value dataPtr) {
  unsigned align;
  if (failed(getMemRefAlignment(
          typeConverter, xferOp.getShapedType().cast<MemRefType>(), align)))
    return failure();
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(xferOp, dataPtr, align);
  return success();
}

static LogicalResult
replaceTransferOpWithMasked(ConversionPatternRewriter &rewriter,
                            LLVMTypeConverter &typeConverter, Location loc,
                            TransferReadOp xferOp, ArrayRef<Value> operands,
                            Value dataPtr, Value mask) {
  auto toLLVMTy = [&](Type t) { return typeConverter.convertType(t); };
  VectorType fillType = xferOp.getVectorType();
  Value fill = rewriter.create<SplatOp>(loc, fillType, xferOp.padding());
  fill = rewriter.create<LLVM::DialectCastOp>(loc, toLLVMTy(fillType), fill);

  Type vecTy = typeConverter.convertType(xferOp.getVectorType());
  if (!vecTy)
    return failure();

  unsigned align;
  if (failed(getMemRefAlignment(
          typeConverter, xferOp.getShapedType().cast<MemRefType>(), align)))
    return failure();

  rewriter.replaceOpWithNewOp<LLVM::MaskedLoadOp>(
      xferOp, vecTy, dataPtr, mask, ValueRange{fill},
      rewriter.getI32IntegerAttr(align));
  return success();
}

static LogicalResult
replaceTransferOpWithLoadOrStore(ConversionPatternRewriter &rewriter,
                                 LLVMTypeConverter &typeConverter, Location loc,
                                 TransferWriteOp xferOp,
                                 ArrayRef<Value> operands, Value dataPtr) {
  unsigned align;
  if (failed(getMemRefAlignment(
          typeConverter, xferOp.getShapedType().cast<MemRefType>(), align)))
    return failure();
  auto adaptor = TransferWriteOpAdaptor(operands);
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(xferOp, adaptor.vector(), dataPtr,
                                             align);
  return success();
}

static LogicalResult
replaceTransferOpWithMasked(ConversionPatternRewriter &rewriter,
                            LLVMTypeConverter &typeConverter, Location loc,
                            TransferWriteOp xferOp, ArrayRef<Value> operands,
                            Value dataPtr, Value mask) {
  unsigned align;
  if (failed(getMemRefAlignment(
          typeConverter, xferOp.getShapedType().cast<MemRefType>(), align)))
    return failure();

  auto adaptor = TransferWriteOpAdaptor(operands);
  rewriter.replaceOpWithNewOp<LLVM::MaskedStoreOp>(
      xferOp, adaptor.vector(), dataPtr, mask,
      rewriter.getI32IntegerAttr(align));
  return success();
}

static TransferReadOpAdaptor getTransferOpAdapter(TransferReadOp xferOp,
                                                  ArrayRef<Value> operands) {
  return TransferReadOpAdaptor(operands);
}

static TransferWriteOpAdaptor getTransferOpAdapter(TransferWriteOp xferOp,
                                                   ArrayRef<Value> operands) {
  return TransferWriteOpAdaptor(operands);
}

namespace {

/// Conversion pattern for a vector.matrix_multiply.
/// This is lowered directly to the proper llvm.intr.matrix.multiply.
class VectorMatmulOpConversion
    : public ConvertOpToLLVMPattern<vector::MatmulOp> {
public:
  using ConvertOpToLLVMPattern<vector::MatmulOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::MatmulOp matmulOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto adaptor = vector::MatmulOpAdaptor(operands);
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
  matchAndRewrite(vector::FlatTransposeOp transOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto adaptor = vector::FlatTransposeOpAdaptor(operands);
    rewriter.replaceOpWithNewOp<LLVM::MatrixTransposeOp>(
        transOp, typeConverter->convertType(transOp.res().getType()),
        adaptor.matrix(), transOp.rows(), transOp.columns());
    return success();
  }
};

/// Conversion pattern for a vector.maskedload.
class VectorMaskedLoadOpConversion
    : public ConvertOpToLLVMPattern<vector::MaskedLoadOp> {
public:
  using ConvertOpToLLVMPattern<vector::MaskedLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedLoadOp load, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = load->getLoc();
    auto adaptor = vector::MaskedLoadOpAdaptor(operands);
    MemRefType memRefType = load.getMemRefType();

    // Resolve alignment.
    unsigned align;
    if (failed(getMemRefAlignment(*getTypeConverter(), memRefType, align)))
      return failure();

    // Resolve address.
    auto vtype = typeConverter->convertType(load.getResultVectorType());
    Value dataPtr = this->getStridedElementPtr(loc, memRefType, adaptor.base(),
                                               adaptor.indices(), rewriter);
    Value ptr = castDataPtr(rewriter, loc, dataPtr, memRefType, vtype);

    rewriter.replaceOpWithNewOp<LLVM::MaskedLoadOp>(
        load, vtype, ptr, adaptor.mask(), adaptor.pass_thru(),
        rewriter.getI32IntegerAttr(align));
    return success();
  }
};

/// Conversion pattern for a vector.maskedstore.
class VectorMaskedStoreOpConversion
    : public ConvertOpToLLVMPattern<vector::MaskedStoreOp> {
public:
  using ConvertOpToLLVMPattern<vector::MaskedStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedStoreOp store, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = store->getLoc();
    auto adaptor = vector::MaskedStoreOpAdaptor(operands);
    MemRefType memRefType = store.getMemRefType();

    // Resolve alignment.
    unsigned align;
    if (failed(getMemRefAlignment(*getTypeConverter(), memRefType, align)))
      return failure();

    // Resolve address.
    auto vtype = typeConverter->convertType(store.getValueVectorType());
    Value dataPtr = this->getStridedElementPtr(loc, memRefType, adaptor.base(),
                                               adaptor.indices(), rewriter);
    Value ptr = castDataPtr(rewriter, loc, dataPtr, memRefType, vtype);

    rewriter.replaceOpWithNewOp<LLVM::MaskedStoreOp>(
        store, adaptor.value(), ptr, adaptor.mask(),
        rewriter.getI32IntegerAttr(align));
    return success();
  }
};

/// Conversion pattern for a vector.gather.
class VectorGatherOpConversion
    : public ConvertOpToLLVMPattern<vector::GatherOp> {
public:
  using ConvertOpToLLVMPattern<vector::GatherOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::GatherOp gather, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = gather->getLoc();
    auto adaptor = vector::GatherOpAdaptor(operands);

    // Resolve alignment.
    unsigned align;
    if (failed(getMemRefAlignment(*getTypeConverter(), gather.getMemRefType(),
                                  align)))
      return failure();

    // Get index ptrs.
    VectorType vType = gather.getResultVectorType();
    Type iType = gather.getIndicesVectorType().getElementType();
    Value ptrs;
    if (failed(getIndexedPtrs(rewriter, loc, adaptor.base(), adaptor.indices(),
                              gather.getMemRefType(), vType, iType, ptrs)))
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
  matchAndRewrite(vector::ScatterOp scatter, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = scatter->getLoc();
    auto adaptor = vector::ScatterOpAdaptor(operands);

    // Resolve alignment.
    unsigned align;
    if (failed(getMemRefAlignment(*getTypeConverter(), scatter.getMemRefType(),
                                  align)))
      return failure();

    // Get index ptrs.
    VectorType vType = scatter.getValueVectorType();
    Type iType = scatter.getIndicesVectorType().getElementType();
    Value ptrs;
    if (failed(getIndexedPtrs(rewriter, loc, adaptor.base(), adaptor.indices(),
                              scatter.getMemRefType(), vType, iType, ptrs)))
      return failure();

    // Replace with the scatter intrinsic.
    rewriter.replaceOpWithNewOp<LLVM::masked_scatter>(
        scatter, adaptor.value(), ptrs, adaptor.mask(),
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
  matchAndRewrite(vector::ExpandLoadOp expand, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = expand->getLoc();
    auto adaptor = vector::ExpandLoadOpAdaptor(operands);
    MemRefType memRefType = expand.getMemRefType();

    // Resolve address.
    auto vtype = typeConverter->convertType(expand.getResultVectorType());
    Value ptr = this->getStridedElementPtr(loc, memRefType, adaptor.base(),
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
  matchAndRewrite(vector::CompressStoreOp compress, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = compress->getLoc();
    auto adaptor = vector::CompressStoreOpAdaptor(operands);
    MemRefType memRefType = compress.getMemRefType();

    // Resolve address.
    Value ptr = this->getStridedElementPtr(loc, memRefType, adaptor.base(),
                                           adaptor.indices(), rewriter);

    rewriter.replaceOpWithNewOp<LLVM::masked_compressstore>(
        compress, adaptor.value(), ptr, adaptor.mask());
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
  matchAndRewrite(vector::ReductionOp reductionOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto kind = reductionOp.kind();
    Type eltType = reductionOp.dest().getType();
    Type llvmType = typeConverter->convertType(eltType);
    if (eltType.isIntOrIndex()) {
      // Integer reductions: add/mul/min/max/and/or/xor.
      if (kind == "add")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_add>(
            reductionOp, llvmType, operands[0]);
      else if (kind == "mul")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_mul>(
            reductionOp, llvmType, operands[0]);
      else if (kind == "min" &&
               (eltType.isIndex() || eltType.isUnsignedInteger()))
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_umin>(
            reductionOp, llvmType, operands[0]);
      else if (kind == "min")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_smin>(
            reductionOp, llvmType, operands[0]);
      else if (kind == "max" &&
               (eltType.isIndex() || eltType.isUnsignedInteger()))
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_umax>(
            reductionOp, llvmType, operands[0]);
      else if (kind == "max")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_smax>(
            reductionOp, llvmType, operands[0]);
      else if (kind == "and")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_and>(
            reductionOp, llvmType, operands[0]);
      else if (kind == "or")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_or>(
            reductionOp, llvmType, operands[0]);
      else if (kind == "xor")
        rewriter.replaceOpWithNewOp<LLVM::vector_reduce_xor>(
            reductionOp, llvmType, operands[0]);
      else
        return failure();
      return success();
    }

    if (!eltType.isa<FloatType>())
      return failure();

    // Floating-point reductions: add/mul/min/max
    if (kind == "add") {
      // Optional accumulator (or zero).
      Value acc = operands.size() > 1 ? operands[1]
                                      : rewriter.create<LLVM::ConstantOp>(
                                            reductionOp->getLoc(), llvmType,
                                            rewriter.getZeroAttr(eltType));
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_fadd>(
          reductionOp, llvmType, acc, operands[0],
          rewriter.getBoolAttr(reassociateFPReductions));
    } else if (kind == "mul") {
      // Optional accumulator (or one).
      Value acc = operands.size() > 1
                      ? operands[1]
                      : rewriter.create<LLVM::ConstantOp>(
                            reductionOp->getLoc(), llvmType,
                            rewriter.getFloatAttr(eltType, 1.0));
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_fmul>(
          reductionOp, llvmType, acc, operands[0],
          rewriter.getBoolAttr(reassociateFPReductions));
    } else if (kind == "min")
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_fmin>(
          reductionOp, llvmType, operands[0]);
    else if (kind == "max")
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_fmax>(
          reductionOp, llvmType, operands[0]);
    else
      return failure();
    return success();
  }

private:
  const bool reassociateFPReductions;
};

/// Conversion pattern for a vector.create_mask (1-D only).
class VectorCreateMaskOpConversion
    : public ConvertOpToLLVMPattern<vector::CreateMaskOp> {
public:
  explicit VectorCreateMaskOpConversion(LLVMTypeConverter &typeConv,
                                        bool enableIndexOpt)
      : ConvertOpToLLVMPattern<vector::CreateMaskOp>(typeConv),
        enableIndexOptimizations(enableIndexOpt) {}

  LogicalResult
  matchAndRewrite(vector::CreateMaskOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = op.getType();
    int64_t rank = dstType.getRank();
    if (rank == 1) {
      rewriter.replaceOp(
          op, buildVectorComparison(rewriter, op, enableIndexOptimizations,
                                    dstType.getDimSize(0), operands[0]));
      return success();
    }
    return failure();
  }

private:
  const bool enableIndexOptimizations;
};

class VectorShuffleOpConversion
    : public ConvertOpToLLVMPattern<vector::ShuffleOp> {
public:
  using ConvertOpToLLVMPattern<vector::ShuffleOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::ShuffleOp shuffleOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = shuffleOp->getLoc();
    auto adaptor = vector::ShuffleOpAdaptor(operands);
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
                                 llvmType, rank, extPos);
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
  matchAndRewrite(vector::ExtractElementOp extractEltOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto adaptor = vector::ExtractElementOpAdaptor(operands);
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
  matchAndRewrite(vector::ExtractOp extractOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = extractOp->getLoc();
    auto adaptor = vector::ExtractOpAdaptor(operands);
    auto vectorType = extractOp.getVectorType();
    auto resultType = extractOp.getResult().getType();
    auto llvmResultType = typeConverter->convertType(resultType);
    auto positionArrayAttr = extractOp.position();

    // Bail if result type cannot be lowered.
    if (!llvmResultType)
      return failure();

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
          ArrayAttr::get(positionAttrs.drop_back(), context);
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
  matchAndRewrite(vector::FMAOp fmaOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto adaptor = vector::FMAOpAdaptor(operands);
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
  matchAndRewrite(vector::InsertElementOp insertEltOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto adaptor = vector::InsertElementOpAdaptor(operands);
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
  matchAndRewrite(vector::InsertOp insertOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = insertOp->getLoc();
    auto adaptor = vector::InsertOpAdaptor(operands);
    auto sourceType = insertOp.getSourceType();
    auto destVectorType = insertOp.getDestVectorType();
    auto llvmResultType = typeConverter->convertType(destVectorType);
    auto positionArrayAttr = insertOp.position();

    // Bail if result type cannot be lowered.
    if (!llvmResultType)
      return failure();

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
          ArrayAttr::get(positionAttrs.drop_back(), context);
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
          ArrayAttr::get(positionAttrs.drop_back(), context);
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
    Value zero = rewriter.create<ConstantOp>(loc, elemType,
                                             rewriter.getZeroAttr(elemType));
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

// When ranks are different, InsertStridedSlice needs to extract a properly
// ranked vector from the destination vector into which to insert. This pattern
// only takes care of this part and forwards the rest of the conversion to
// another pattern that converts InsertStridedSlice for operands of the same
// rank.
//
// RewritePattern for InsertStridedSliceOp where source and destination vectors
// have different ranks. In this case:
//   1. the proper subvector is extracted from the destination vector
//   2. a new InsertStridedSlice op is created to insert the source in the
//   destination subvector
//   3. the destination subvector is inserted back in the proper place
//   4. the op is replaced by the result of step 3.
// The new InsertStridedSlice from step 2. will be picked up by a
// `VectorInsertStridedSliceOpSameRankRewritePattern`.
class VectorInsertStridedSliceOpDifferentRankRewritePattern
    : public OpRewritePattern<InsertStridedSliceOp> {
public:
  using OpRewritePattern<InsertStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = op.getSourceVectorType();
    auto dstType = op.getDestVectorType();

    if (op.offsets().getValue().empty())
      return failure();

    auto loc = op.getLoc();
    int64_t rankDiff = dstType.getRank() - srcType.getRank();
    assert(rankDiff >= 0);
    if (rankDiff == 0)
      return failure();

    int64_t rankRest = dstType.getRank() - rankDiff;
    // Extract / insert the subvector of matching rank and InsertStridedSlice
    // on it.
    Value extracted =
        rewriter.create<ExtractOp>(loc, op.dest(),
                                   getI64SubArray(op.offsets(), /*dropFront=*/0,
                                                  /*dropBack=*/rankRest));
    // A different pattern will kick in for InsertStridedSlice with matching
    // ranks.
    auto stridedSliceInnerOp = rewriter.create<InsertStridedSliceOp>(
        loc, op.source(), extracted,
        getI64SubArray(op.offsets(), /*dropFront=*/rankDiff),
        getI64SubArray(op.strides(), /*dropFront=*/0));
    rewriter.replaceOpWithNewOp<InsertOp>(
        op, stridedSliceInnerOp.getResult(), op.dest(),
        getI64SubArray(op.offsets(), /*dropFront=*/0,
                       /*dropBack=*/rankRest));
    return success();
  }
};

// RewritePattern for InsertStridedSliceOp where source and destination vectors
// have the same rank. In this case, we reduce
//   1. the proper subvector is extracted from the destination vector
//   2. a new InsertStridedSlice op is created to insert the source in the
//   destination subvector
//   3. the destination subvector is inserted back in the proper place
//   4. the op is replaced by the result of step 3.
// The new InsertStridedSlice from step 2. will be picked up by a
// `VectorInsertStridedSliceOpSameRankRewritePattern`.
class VectorInsertStridedSliceOpSameRankRewritePattern
    : public OpRewritePattern<InsertStridedSliceOp> {
public:
  VectorInsertStridedSliceOpSameRankRewritePattern(MLIRContext *ctx)
      : OpRewritePattern<InsertStridedSliceOp>(ctx) {
    // This pattern creates recursive InsertStridedSliceOp, but the recursion is
    // bounded as the rank is strictly decreasing.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(InsertStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = op.getSourceVectorType();
    auto dstType = op.getDestVectorType();

    if (op.offsets().getValue().empty())
      return failure();

    int64_t rankDiff = dstType.getRank() - srcType.getRank();
    assert(rankDiff >= 0);
    if (rankDiff != 0)
      return failure();

    if (srcType == dstType) {
      rewriter.replaceOp(op, op.source());
      return success();
    }

    int64_t offset =
        op.offsets().getValue().front().cast<IntegerAttr>().getInt();
    int64_t size = srcType.getShape().front();
    int64_t stride =
        op.strides().getValue().front().cast<IntegerAttr>().getInt();

    auto loc = op.getLoc();
    Value res = op.dest();
    // For each slice of the source vector along the most major dimension.
    for (int64_t off = offset, e = offset + size * stride, idx = 0; off < e;
         off += stride, ++idx) {
      // 1. extract the proper subvector (or element) from source
      Value extractedSource = extractOne(rewriter, loc, op.source(), idx);
      if (extractedSource.getType().isa<VectorType>()) {
        // 2. If we have a vector, extract the proper subvector from destination
        // Otherwise we are at the element level and no need to recurse.
        Value extractedDest = extractOne(rewriter, loc, op.dest(), off);
        // 3. Reduce the problem to lowering a new InsertStridedSlice op with
        // smaller rank.
        extractedSource = rewriter.create<InsertStridedSliceOp>(
            loc, extractedSource, extractedDest,
            getI64SubArray(op.offsets(), /* dropFront=*/1),
            getI64SubArray(op.strides(), /* dropFront=*/1));
      }
      // 4. Insert the extractedSource into the res vector.
      res = insertOne(rewriter, loc, extractedSource, res, off);
    }

    rewriter.replaceOp(op, res);
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
  if (memRefType.getAffineMaps().empty() ||
      memRefType.getAffineMaps().front().isIdentity())
    return strides;

  // Otherwise, we must determine contiguity form shapes. This can only ever
  // work in static cases because MemRefType is underspecified to represent
  // contiguous dynamic shapes in other ways than with just empty/identity
  // layout.
  auto sizes = memRefType.getShape();
  for (int index = 0, e = strides.size() - 2; index < e; ++index) {
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
  matchAndRewrite(vector::TypeCastOp castOp, ArrayRef<Value> operands,
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
        operands[0].getType().dyn_cast<LLVM::LLVMStructType>();
    if (!llvmSourceDescriptorTy)
      return failure();
    MemRefDescriptor sourceMemRef(operands[0]);

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

/// Conversion pattern that converts a 1-D vector transfer read/write op in a
/// sequence of:
/// 1. Get the source/dst address as an LLVM vector pointer.
/// 2. Create a vector with linear indices [ 0 .. vector_length - 1 ].
/// 3. Create an offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
/// 4. Create a mask where offsetVector is compared against memref upper bound.
/// 5. Rewrite op as a masked read or write.
template <typename ConcreteOp>
class VectorTransferConversion : public ConvertOpToLLVMPattern<ConcreteOp> {
public:
  explicit VectorTransferConversion(LLVMTypeConverter &typeConv,
                                    bool enableIndexOpt)
      : ConvertOpToLLVMPattern<ConcreteOp>(typeConv),
        enableIndexOptimizations(enableIndexOpt) {}

  LogicalResult
  matchAndRewrite(ConcreteOp xferOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto adaptor = getTransferOpAdapter(xferOp, operands);

    if (xferOp.getVectorType().getRank() > 1 ||
        llvm::size(xferOp.indices()) == 0)
      return failure();
    if (xferOp.permutation_map() !=
        AffineMap::getMinorIdentityMap(xferOp.permutation_map().getNumInputs(),
                                       xferOp.getVectorType().getRank(),
                                       xferOp->getContext()))
      return failure();
    auto memRefType = xferOp.getShapedType().template dyn_cast<MemRefType>();
    if (!memRefType)
      return failure();
    // Only contiguous source tensors supported atm.
    auto strides = computeContiguousStrides(memRefType);
    if (!strides)
      return failure();

    auto toLLVMTy = [&](Type t) {
      return this->getTypeConverter()->convertType(t);
    };

    Location loc = xferOp->getLoc();

    if (auto memrefVectorElementType =
            memRefType.getElementType().template dyn_cast<VectorType>()) {
      // Memref has vector element type.
      if (memrefVectorElementType.getElementType() !=
          xferOp.getVectorType().getElementType())
        return failure();
#ifndef NDEBUG
      // Check that memref vector type is a suffix of 'vectorType.
      unsigned memrefVecEltRank = memrefVectorElementType.getRank();
      unsigned resultVecRank = xferOp.getVectorType().getRank();
      assert(memrefVecEltRank <= resultVecRank);
      // TODO: Move this to isSuffix in Vector/Utils.h.
      unsigned rankOffset = resultVecRank - memrefVecEltRank;
      auto memrefVecEltShape = memrefVectorElementType.getShape();
      auto resultVecShape = xferOp.getVectorType().getShape();
      for (unsigned i = 0; i < memrefVecEltRank; ++i)
        assert(memrefVecEltShape[i] != resultVecShape[rankOffset + i] &&
               "memref vector element shape should match suffix of vector "
               "result shape.");
#endif // ifndef NDEBUG
    }

    // 1. Get the source/dst address as an LLVM vector pointer.
    VectorType vtp = xferOp.getVectorType();
    Value dataPtr = this->getStridedElementPtr(
        loc, memRefType, adaptor.source(), adaptor.indices(), rewriter);
    Value vectorDataPtr =
        castDataPtr(rewriter, loc, dataPtr, memRefType, toLLVMTy(vtp));

    if (!xferOp.isMaskedDim(0))
      return replaceTransferOpWithLoadOrStore(rewriter,
                                              *this->getTypeConverter(), loc,
                                              xferOp, operands, vectorDataPtr);

    // 2. Create a vector with linear indices [ 0 .. vector_length - 1 ].
    // 3. Create offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
    // 4. Let dim the memref dimension, compute the vector comparison mask:
    //   [ offset + 0 .. offset + vector_length - 1 ] < [ dim .. dim ]
    //
    // TODO: when the leaf transfer rank is k > 1, we need the last `k`
    //       dimensions here.
    unsigned vecWidth = LLVM::getVectorNumElements(vtp).getFixedValue();
    unsigned lastIndex = llvm::size(xferOp.indices()) - 1;
    Value off = xferOp.indices()[lastIndex];
    Value dim = rewriter.create<DimOp>(loc, xferOp.source(), lastIndex);
    Value mask = buildVectorComparison(
        rewriter, xferOp, enableIndexOptimizations, vecWidth, dim, &off);

    // 5. Rewrite as a masked read / write.
    return replaceTransferOpWithMasked(rewriter, *this->getTypeConverter(), loc,
                                       xferOp, operands, vectorDataPtr, mask);
  }

private:
  const bool enableIndexOptimizations;
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
  matchAndRewrite(vector::PrintOp printOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto adaptor = vector::PrintOpAdaptor(operands);
    Type printType = printOp.getPrintType();

    if (typeConverter->convertType(printType) == nullptr)
      return failure();

    // Make sure element type has runtime support.
    PrintConversion conversion = PrintConversion::None;
    VectorType vectorType = printType.dyn_cast<VectorType>();
    Type eltType = vectorType ? vectorType.getElementType() : printType;
    Operation *printer;
    if (eltType.isF32()) {
      printer = getPrintFloat(printOp);
    } else if (eltType.isF64()) {
      printer = getPrintDouble(printOp);
    } else if (eltType.isIndex()) {
      printer = getPrintU64(printOp);
    } else if (auto intTy = eltType.dyn_cast<IntegerType>()) {
      // Integers need a zero or sign extension on the operand
      // (depending on the source type) as well as a signed or
      // unsigned print method. Up to 64-bit is supported.
      unsigned width = intTy.getWidth();
      if (intTy.isUnsigned()) {
        if (width <= 64) {
          if (width < 64)
            conversion = PrintConversion::ZeroExt64;
          printer = getPrintU64(printOp);
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
          printer = getPrintI64(printOp);
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
    emitCall(rewriter, printOp->getLoc(), getPrintNewline(printOp));
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
        value = rewriter.create<ZeroExtendIOp>(
            loc, value, IntegerType::get(rewriter.getContext(), 64));
        break;
      case PrintConversion::SignExt64:
        value = rewriter.create<SignExtendIOp>(
            loc, value, IntegerType::get(rewriter.getContext(), 64));
        break;
      case PrintConversion::None:
        break;
      }
      emitCall(rewriter, loc, printer, value);
      return;
    }

    emitCall(rewriter, loc, getPrintOpen(op));
    Operation *printComma = getPrintComma(op);
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
    emitCall(rewriter, loc, getPrintClose(op));
  }

  // Helper to emit a call.
  static void emitCall(ConversionPatternRewriter &rewriter, Location loc,
                       Operation *ref, ValueRange params = ValueRange()) {
    rewriter.create<LLVM::CallOp>(loc, TypeRange(),
                                  rewriter.getSymbolRefAttr(ref), params);
  }

  // Helper for printer method declaration (first hit) and lookup.
  static Operation *getPrint(Operation *op, StringRef name,
                             ArrayRef<Type> params) {
    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (func)
      return func;
    OpBuilder moduleBuilder(module.getBodyRegion());
    return moduleBuilder.create<LLVM::LLVMFuncOp>(
        op->getLoc(), name,
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(op->getContext()),
                                    params));
  }

  // Helpers for method names.
  Operation *getPrintI64(Operation *op) const {
    return getPrint(op, "printI64", IntegerType::get(op->getContext(), 64));
  }
  Operation *getPrintU64(Operation *op) const {
    return getPrint(op, "printU64", IntegerType::get(op->getContext(), 64));
  }
  Operation *getPrintFloat(Operation *op) const {
    return getPrint(op, "printF32", Float32Type::get(op->getContext()));
  }
  Operation *getPrintDouble(Operation *op) const {
    return getPrint(op, "printF64", Float64Type::get(op->getContext()));
  }
  Operation *getPrintOpen(Operation *op) const {
    return getPrint(op, "printOpen", {});
  }
  Operation *getPrintClose(Operation *op) const {
    return getPrint(op, "printClose", {});
  }
  Operation *getPrintComma(Operation *op) const {
    return getPrint(op, "printComma", {});
  }
  Operation *getPrintNewline(Operation *op) const {
    return getPrint(op, "printNewline", {});
  }
};

/// Progressive lowering of ExtractStridedSliceOp to either:
///   1. express single offset extract as a direct shuffle.
///   2. extract + lower rank strided_slice + insert for the n-D case.
class VectorExtractStridedSliceOpConversion
    : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  VectorExtractStridedSliceOpConversion(MLIRContext *ctx)
      : OpRewritePattern<ExtractStridedSliceOp>(ctx) {
    // This pattern creates recursive ExtractStridedSliceOp, but the recursion
    // is bounded as the rank is strictly decreasing.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto dstType = op.getType();

    assert(!op.offsets().getValue().empty() && "Unexpected empty offsets");

    int64_t offset =
        op.offsets().getValue().front().cast<IntegerAttr>().getInt();
    int64_t size = op.sizes().getValue().front().cast<IntegerAttr>().getInt();
    int64_t stride =
        op.strides().getValue().front().cast<IntegerAttr>().getInt();

    auto loc = op.getLoc();
    auto elemType = dstType.getElementType();
    assert(elemType.isSignlessIntOrIndexOrFloat());

    // Single offset can be more efficiently shuffled.
    if (op.offsets().getValue().size() == 1) {
      SmallVector<int64_t, 4> offsets;
      offsets.reserve(size);
      for (int64_t off = offset, e = offset + size * stride; off < e;
           off += stride)
        offsets.push_back(off);
      rewriter.replaceOpWithNewOp<ShuffleOp>(op, dstType, op.vector(),
                                             op.vector(),
                                             rewriter.getI64ArrayAttr(offsets));
      return success();
    }

    // Extract/insert on a lower ranked extract strided slice op.
    Value zero = rewriter.create<ConstantOp>(loc, elemType,
                                             rewriter.getZeroAttr(elemType));
    Value res = rewriter.create<SplatOp>(loc, dstType, zero);
    for (int64_t off = offset, e = offset + size * stride, idx = 0; off < e;
         off += stride, ++idx) {
      Value one = extractOne(rewriter, loc, op.vector(), off);
      Value extracted = rewriter.create<ExtractStridedSliceOp>(
          loc, one, getI64SubArray(op.offsets(), /* dropFront=*/1),
          getI64SubArray(op.sizes(), /* dropFront=*/1),
          getI64SubArray(op.strides(), /* dropFront=*/1));
      res = insertOne(rewriter, loc, extracted, res, idx);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

} // namespace

/// Populate the given list with patterns that convert from Vector to LLVM.
void mlir::populateVectorToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    bool reassociateFPReductions, bool enableIndexOptimizations) {
  MLIRContext *ctx = converter.getDialect()->getContext();
  // clang-format off
  patterns.insert<VectorFMAOpNDRewritePattern,
                  VectorInsertStridedSliceOpDifferentRankRewritePattern,
                  VectorInsertStridedSliceOpSameRankRewritePattern,
                  VectorExtractStridedSliceOpConversion>(ctx);
  patterns.insert<VectorReductionOpConversion>(
      converter, reassociateFPReductions);
  patterns.insert<VectorCreateMaskOpConversion,
                  VectorTransferConversion<TransferReadOp>,
                  VectorTransferConversion<TransferWriteOp>>(
      converter, enableIndexOptimizations);
  patterns
      .insert<VectorShuffleOpConversion,
              VectorExtractElementOpConversion,
              VectorExtractOpConversion,
              VectorFMAOp1DConversion,
              VectorInsertElementOpConversion,
              VectorInsertOpConversion,
              VectorPrintOpConversion,
              VectorTypeCastOpConversion,
              VectorMaskedLoadOpConversion,
              VectorMaskedStoreOpConversion,
              VectorGatherOpConversion,
              VectorScatterOpConversion,
              VectorExpandLoadOpConversion,
              VectorCompressStoreOpConversion>(converter);
  // clang-format on
}

void mlir::populateVectorToLLVMMatrixConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  patterns.insert<VectorMatmulOpConversion>(converter);
  patterns.insert<VectorFlatTransposeOpConversion>(converter);
}
