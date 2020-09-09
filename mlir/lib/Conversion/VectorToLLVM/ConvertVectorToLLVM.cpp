//===- VectorToLLVM.cpp - Conversion from Vector to the LLVM dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Target/LLVMIR/TypeTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

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

// Helper that returns data layout alignment of an operation with memref.
template <typename T>
LogicalResult getMemRefAlignment(LLVMTypeConverter &typeConverter, T op,
                                 unsigned &align) {
  Type elementTy =
      typeConverter.convertType(op.getMemRefType().getElementType());
  if (!elementTy)
    return failure();

  // TODO: this should use the MLIR data layout when it becomes available and
  // stop depending on translation.
  llvm::LLVMContext llvmContext;
  align = LLVM::TypeToLLVMIRTranslator(llvmContext)
              .getPreferredAlignment(elementTy.cast<LLVM::LLVMType>(),
                                     typeConverter.getDataLayout());
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

// Helper that returns a pointer given a memref base.
static LogicalResult getBasePtr(ConversionPatternRewriter &rewriter,
                                Location loc, Value memref,
                                MemRefType memRefType, Value &ptr) {
  Value base;
  if (failed(getBase(rewriter, loc, memref, memRefType, base)))
    return failure();
  auto pType = MemRefDescriptor(memref).getElementPtrType();
  ptr = rewriter.create<LLVM::GEPOp>(loc, pType, base);
  return success();
}

// Helper that returns a bit-casted pointer given a memref base.
static LogicalResult getBasePtr(ConversionPatternRewriter &rewriter,
                                Location loc, Value memref,
                                MemRefType memRefType, Type type, Value &ptr) {
  Value base;
  if (failed(getBase(rewriter, loc, memref, memRefType, base)))
    return failure();
  auto pType = type.template cast<LLVM::LLVMType>().getPointerTo();
  base = rewriter.create<LLVM::BitcastOp>(loc, pType, base);
  ptr = rewriter.create<LLVM::GEPOp>(loc, pType, base);
  return success();
}

// Helper that returns vector of pointers given a memref base and an index
// vector.
static LogicalResult getIndexedPtrs(ConversionPatternRewriter &rewriter,
                                    Location loc, Value memref, Value indices,
                                    MemRefType memRefType, VectorType vType,
                                    Type iType, Value &ptrs) {
  Value base;
  if (failed(getBase(rewriter, loc, memref, memRefType, base)))
    return failure();
  auto pType = MemRefDescriptor(memref).getElementPtrType();
  auto ptrsType = LLVM::LLVMType::getVectorTy(pType, vType.getDimSize(0));
  ptrs = rewriter.create<LLVM::GEPOp>(loc, ptrsType, base, indices);
  return success();
}

static LogicalResult
replaceTransferOpWithLoadOrStore(ConversionPatternRewriter &rewriter,
                                 LLVMTypeConverter &typeConverter, Location loc,
                                 TransferReadOp xferOp,
                                 ArrayRef<Value> operands, Value dataPtr) {
  unsigned align;
  if (failed(getMemRefAlignment(typeConverter, xferOp, align)))
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
  if (failed(getMemRefAlignment(typeConverter, xferOp, align)))
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
  if (failed(getMemRefAlignment(typeConverter, xferOp, align)))
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
  if (failed(getMemRefAlignment(typeConverter, xferOp, align)))
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
class VectorMatmulOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorMatmulOpConversion(MLIRContext *context,
                                    LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::MatmulOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto matmulOp = cast<vector::MatmulOp>(op);
    auto adaptor = vector::MatmulOpAdaptor(operands);
    rewriter.replaceOpWithNewOp<LLVM::MatrixMultiplyOp>(
        op, typeConverter.convertType(matmulOp.res().getType()), adaptor.lhs(),
        adaptor.rhs(), matmulOp.lhs_rows(), matmulOp.lhs_columns(),
        matmulOp.rhs_columns());
    return success();
  }
};

/// Conversion pattern for a vector.flat_transpose.
/// This is lowered directly to the proper llvm.intr.matrix.transpose.
class VectorFlatTransposeOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorFlatTransposeOpConversion(MLIRContext *context,
                                           LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::FlatTransposeOp::getOperationName(),
                             context, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto transOp = cast<vector::FlatTransposeOp>(op);
    auto adaptor = vector::FlatTransposeOpAdaptor(operands);
    rewriter.replaceOpWithNewOp<LLVM::MatrixTransposeOp>(
        transOp, typeConverter.convertType(transOp.res().getType()),
        adaptor.matrix(), transOp.rows(), transOp.columns());
    return success();
  }
};

/// Conversion pattern for a vector.maskedload.
class VectorMaskedLoadOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorMaskedLoadOpConversion(MLIRContext *context,
                                        LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::MaskedLoadOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto load = cast<vector::MaskedLoadOp>(op);
    auto adaptor = vector::MaskedLoadOpAdaptor(operands);

    // Resolve alignment.
    unsigned align;
    if (failed(getMemRefAlignment(typeConverter, load, align)))
      return failure();

    auto vtype = typeConverter.convertType(load.getResultVectorType());
    Value ptr;
    if (failed(getBasePtr(rewriter, loc, adaptor.base(), load.getMemRefType(),
                          vtype, ptr)))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::MaskedLoadOp>(
        load, vtype, ptr, adaptor.mask(), adaptor.pass_thru(),
        rewriter.getI32IntegerAttr(align));
    return success();
  }
};

/// Conversion pattern for a vector.maskedstore.
class VectorMaskedStoreOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorMaskedStoreOpConversion(MLIRContext *context,
                                         LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::MaskedStoreOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto store = cast<vector::MaskedStoreOp>(op);
    auto adaptor = vector::MaskedStoreOpAdaptor(operands);

    // Resolve alignment.
    unsigned align;
    if (failed(getMemRefAlignment(typeConverter, store, align)))
      return failure();

    auto vtype = typeConverter.convertType(store.getValueVectorType());
    Value ptr;
    if (failed(getBasePtr(rewriter, loc, adaptor.base(), store.getMemRefType(),
                          vtype, ptr)))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::MaskedStoreOp>(
        store, adaptor.value(), ptr, adaptor.mask(),
        rewriter.getI32IntegerAttr(align));
    return success();
  }
};

/// Conversion pattern for a vector.gather.
class VectorGatherOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorGatherOpConversion(MLIRContext *context,
                                    LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::GatherOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto gather = cast<vector::GatherOp>(op);
    auto adaptor = vector::GatherOpAdaptor(operands);

    // Resolve alignment.
    unsigned align;
    if (failed(getMemRefAlignment(typeConverter, gather, align)))
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
        gather, typeConverter.convertType(vType), ptrs, adaptor.mask(),
        adaptor.pass_thru(), rewriter.getI32IntegerAttr(align));
    return success();
  }
};

/// Conversion pattern for a vector.scatter.
class VectorScatterOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorScatterOpConversion(MLIRContext *context,
                                     LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::ScatterOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto scatter = cast<vector::ScatterOp>(op);
    auto adaptor = vector::ScatterOpAdaptor(operands);

    // Resolve alignment.
    unsigned align;
    if (failed(getMemRefAlignment(typeConverter, scatter, align)))
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
class VectorExpandLoadOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorExpandLoadOpConversion(MLIRContext *context,
                                        LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::ExpandLoadOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto expand = cast<vector::ExpandLoadOp>(op);
    auto adaptor = vector::ExpandLoadOpAdaptor(operands);

    Value ptr;
    if (failed(getBasePtr(rewriter, loc, adaptor.base(), expand.getMemRefType(),
                          ptr)))
      return failure();

    auto vType = expand.getResultVectorType();
    rewriter.replaceOpWithNewOp<LLVM::masked_expandload>(
        op, typeConverter.convertType(vType), ptr, adaptor.mask(),
        adaptor.pass_thru());
    return success();
  }
};

/// Conversion pattern for a vector.compressstore.
class VectorCompressStoreOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorCompressStoreOpConversion(MLIRContext *context,
                                           LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::CompressStoreOp::getOperationName(),
                             context, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto compress = cast<vector::CompressStoreOp>(op);
    auto adaptor = vector::CompressStoreOpAdaptor(operands);

    Value ptr;
    if (failed(getBasePtr(rewriter, loc, adaptor.base(),
                          compress.getMemRefType(), ptr)))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::masked_compressstore>(
        op, adaptor.value(), ptr, adaptor.mask());
    return success();
  }
};

/// Conversion pattern for all vector reductions.
class VectorReductionOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorReductionOpConversion(MLIRContext *context,
                                       LLVMTypeConverter &typeConverter,
                                       bool reassociateFPRed)
      : ConvertToLLVMPattern(vector::ReductionOp::getOperationName(), context,
                             typeConverter),
        reassociateFPReductions(reassociateFPRed) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto reductionOp = cast<vector::ReductionOp>(op);
    auto kind = reductionOp.kind();
    Type eltType = reductionOp.dest().getType();
    Type llvmType = typeConverter.convertType(eltType);
    if (eltType.isSignlessInteger(32) || eltType.isSignlessInteger(64)) {
      // Integer reductions: add/mul/min/max/and/or/xor.
      if (kind == "add")
        rewriter.replaceOpWithNewOp<LLVM::experimental_vector_reduce_add>(
            op, llvmType, operands[0]);
      else if (kind == "mul")
        rewriter.replaceOpWithNewOp<LLVM::experimental_vector_reduce_mul>(
            op, llvmType, operands[0]);
      else if (kind == "min")
        rewriter.replaceOpWithNewOp<LLVM::experimental_vector_reduce_smin>(
            op, llvmType, operands[0]);
      else if (kind == "max")
        rewriter.replaceOpWithNewOp<LLVM::experimental_vector_reduce_smax>(
            op, llvmType, operands[0]);
      else if (kind == "and")
        rewriter.replaceOpWithNewOp<LLVM::experimental_vector_reduce_and>(
            op, llvmType, operands[0]);
      else if (kind == "or")
        rewriter.replaceOpWithNewOp<LLVM::experimental_vector_reduce_or>(
            op, llvmType, operands[0]);
      else if (kind == "xor")
        rewriter.replaceOpWithNewOp<LLVM::experimental_vector_reduce_xor>(
            op, llvmType, operands[0]);
      else
        return failure();
      return success();

    } else if (eltType.isF32() || eltType.isF64()) {
      // Floating-point reductions: add/mul/min/max
      if (kind == "add") {
        // Optional accumulator (or zero).
        Value acc = operands.size() > 1 ? operands[1]
                                        : rewriter.create<LLVM::ConstantOp>(
                                              op->getLoc(), llvmType,
                                              rewriter.getZeroAttr(eltType));
        rewriter.replaceOpWithNewOp<LLVM::experimental_vector_reduce_v2_fadd>(
            op, llvmType, acc, operands[0],
            rewriter.getBoolAttr(reassociateFPReductions));
      } else if (kind == "mul") {
        // Optional accumulator (or one).
        Value acc = operands.size() > 1
                        ? operands[1]
                        : rewriter.create<LLVM::ConstantOp>(
                              op->getLoc(), llvmType,
                              rewriter.getFloatAttr(eltType, 1.0));
        rewriter.replaceOpWithNewOp<LLVM::experimental_vector_reduce_v2_fmul>(
            op, llvmType, acc, operands[0],
            rewriter.getBoolAttr(reassociateFPReductions));
      } else if (kind == "min")
        rewriter.replaceOpWithNewOp<LLVM::experimental_vector_reduce_fmin>(
            op, llvmType, operands[0]);
      else if (kind == "max")
        rewriter.replaceOpWithNewOp<LLVM::experimental_vector_reduce_fmax>(
            op, llvmType, operands[0]);
      else
        return failure();
      return success();
    }
    return failure();
  }

private:
  const bool reassociateFPReductions;
};

/// Conversion pattern for a vector.create_mask (1-D only).
class VectorCreateMaskOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorCreateMaskOpConversion(MLIRContext *context,
                                        LLVMTypeConverter &typeConverter,
                                        bool enableIndexOpt)
      : ConvertToLLVMPattern(vector::CreateMaskOp::getOperationName(), context,
                             typeConverter),
        enableIndexOptimizations(enableIndexOpt) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = op->getResult(0).getType().cast<VectorType>();
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

class VectorShuffleOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorShuffleOpConversion(MLIRContext *context,
                                     LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::ShuffleOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto adaptor = vector::ShuffleOpAdaptor(operands);
    auto shuffleOp = cast<vector::ShuffleOp>(op);
    auto v1Type = shuffleOp.getV1VectorType();
    auto v2Type = shuffleOp.getV2VectorType();
    auto vectorType = shuffleOp.getVectorType();
    Type llvmType = typeConverter.convertType(vectorType);
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
      Value shuffle = rewriter.create<LLVM::ShuffleVectorOp>(
          loc, adaptor.v1(), adaptor.v2(), maskArrayAttr);
      rewriter.replaceOp(op, shuffle);
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
      Value extract = extractOne(rewriter, typeConverter, loc, value, llvmType,
                                 rank, extPos);
      insert = insertOne(rewriter, typeConverter, loc, insert, extract,
                         llvmType, rank, insPos++);
    }
    rewriter.replaceOp(op, insert);
    return success();
  }
};

class VectorExtractElementOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorExtractElementOpConversion(MLIRContext *context,
                                            LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::ExtractElementOp::getOperationName(),
                             context, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto adaptor = vector::ExtractElementOpAdaptor(operands);
    auto extractEltOp = cast<vector::ExtractElementOp>(op);
    auto vectorType = extractEltOp.getVectorType();
    auto llvmType = typeConverter.convertType(vectorType.getElementType());

    // Bail if result type cannot be lowered.
    if (!llvmType)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::ExtractElementOp>(
        op, llvmType, adaptor.vector(), adaptor.position());
    return success();
  }
};

class VectorExtractOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorExtractOpConversion(MLIRContext *context,
                                     LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::ExtractOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto adaptor = vector::ExtractOpAdaptor(operands);
    auto extractOp = cast<vector::ExtractOp>(op);
    auto vectorType = extractOp.getVectorType();
    auto resultType = extractOp.getResult().getType();
    auto llvmResultType = typeConverter.convertType(resultType);
    auto positionArrayAttr = extractOp.position();

    // Bail if result type cannot be lowered.
    if (!llvmResultType)
      return failure();

    // One-shot extraction of vector from array (only requires extractvalue).
    if (resultType.isa<VectorType>()) {
      Value extracted = rewriter.create<LLVM::ExtractValueOp>(
          loc, llvmResultType, adaptor.vector(), positionArrayAttr);
      rewriter.replaceOp(op, extracted);
      return success();
    }

    // Potential extraction of 1-D vector from array.
    auto *context = op->getContext();
    Value extracted = adaptor.vector();
    auto positionAttrs = positionArrayAttr.getValue();
    if (positionAttrs.size() > 1) {
      auto oneDVectorType = reducedVectorTypeBack(vectorType);
      auto nMinusOnePositionAttrs =
          ArrayAttr::get(positionAttrs.drop_back(), context);
      extracted = rewriter.create<LLVM::ExtractValueOp>(
          loc, typeConverter.convertType(oneDVectorType), extracted,
          nMinusOnePositionAttrs);
    }

    // Remaining extraction of element from 1-D LLVM vector
    auto position = positionAttrs.back().cast<IntegerAttr>();
    auto i64Type = LLVM::LLVMType::getInt64Ty(rewriter.getContext());
    auto constant = rewriter.create<LLVM::ConstantOp>(loc, i64Type, position);
    extracted =
        rewriter.create<LLVM::ExtractElementOp>(loc, extracted, constant);
    rewriter.replaceOp(op, extracted);

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
///    (!llvm<"<8 x float>">, !llvm<"<8 x float>">, !llvm<"<8 x float>">)
///    -> !llvm<"<8 x float>">
/// ```
class VectorFMAOp1DConversion : public ConvertToLLVMPattern {
public:
  explicit VectorFMAOp1DConversion(MLIRContext *context,
                                   LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::FMAOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto adaptor = vector::FMAOpAdaptor(operands);
    vector::FMAOp fmaOp = cast<vector::FMAOp>(op);
    VectorType vType = fmaOp.getVectorType();
    if (vType.getRank() != 1)
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::FMulAddOp>(op, adaptor.lhs(),
                                                 adaptor.rhs(), adaptor.acc());
    return success();
  }
};

class VectorInsertElementOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorInsertElementOpConversion(MLIRContext *context,
                                           LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::InsertElementOp::getOperationName(),
                             context, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto adaptor = vector::InsertElementOpAdaptor(operands);
    auto insertEltOp = cast<vector::InsertElementOp>(op);
    auto vectorType = insertEltOp.getDestVectorType();
    auto llvmType = typeConverter.convertType(vectorType);

    // Bail if result type cannot be lowered.
    if (!llvmType)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::InsertElementOp>(
        op, llvmType, adaptor.dest(), adaptor.source(), adaptor.position());
    return success();
  }
};

class VectorInsertOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorInsertOpConversion(MLIRContext *context,
                                    LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::InsertOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto adaptor = vector::InsertOpAdaptor(operands);
    auto insertOp = cast<vector::InsertOp>(op);
    auto sourceType = insertOp.getSourceType();
    auto destVectorType = insertOp.getDestVectorType();
    auto llvmResultType = typeConverter.convertType(destVectorType);
    auto positionArrayAttr = insertOp.position();

    // Bail if result type cannot be lowered.
    if (!llvmResultType)
      return failure();

    // One-shot insertion of a vector into an array (only requires insertvalue).
    if (sourceType.isa<VectorType>()) {
      Value inserted = rewriter.create<LLVM::InsertValueOp>(
          loc, llvmResultType, adaptor.dest(), adaptor.source(),
          positionArrayAttr);
      rewriter.replaceOp(op, inserted);
      return success();
    }

    // Potential extraction of 1-D vector from array.
    auto *context = op->getContext();
    Value extracted = adaptor.dest();
    auto positionAttrs = positionArrayAttr.getValue();
    auto position = positionAttrs.back().cast<IntegerAttr>();
    auto oneDVectorType = destVectorType;
    if (positionAttrs.size() > 1) {
      oneDVectorType = reducedVectorTypeBack(destVectorType);
      auto nMinusOnePositionAttrs =
          ArrayAttr::get(positionAttrs.drop_back(), context);
      extracted = rewriter.create<LLVM::ExtractValueOp>(
          loc, typeConverter.convertType(oneDVectorType), extracted,
          nMinusOnePositionAttrs);
    }

    // Insertion of an element into a 1-D LLVM vector.
    auto i64Type = LLVM::LLVMType::getInt64Ty(rewriter.getContext());
    auto constant = rewriter.create<LLVM::ConstantOp>(loc, i64Type, position);
    Value inserted = rewriter.create<LLVM::InsertElementOp>(
        loc, typeConverter.convertType(oneDVectorType), extracted,
        adaptor.source(), constant);

    // Potential insertion of resulting 1-D vector into array.
    if (positionAttrs.size() > 1) {
      auto nMinusOnePositionAttrs =
          ArrayAttr::get(positionAttrs.drop_back(), context);
      inserted = rewriter.create<LLVM::InsertValueOp>(loc, llvmResultType,
                                                      adaptor.dest(), inserted,
                                                      nMinusOnePositionAttrs);
    }

    rewriter.replaceOp(op, inserted);
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
                                                  /*dropFront=*/rankRest));
    // A different pattern will kick in for InsertStridedSlice with matching
    // ranks.
    auto stridedSliceInnerOp = rewriter.create<InsertStridedSliceOp>(
        loc, op.source(), extracted,
        getI64SubArray(op.offsets(), /*dropFront=*/rankDiff),
        getI64SubArray(op.strides(), /*dropFront=*/0));
    rewriter.replaceOpWithNewOp<InsertOp>(
        op, stridedSliceInnerOp.getResult(), op.dest(),
        getI64SubArray(op.offsets(), /*dropFront=*/0,
                       /*dropFront=*/rankRest));
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
  using OpRewritePattern<InsertStridedSliceOp>::OpRewritePattern;

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
  /// This pattern creates recursive InsertStridedSliceOp, but the recursion is
  /// bounded as the rank is strictly decreasing.
  bool hasBoundedRewriteRecursion() const final { return true; }
};

/// Returns true if the memory underlying `memRefType` has a contiguous layout.
/// Strides are written to `strides`.
static bool isContiguous(MemRefType memRefType,
                         SmallVectorImpl<int64_t> &strides) {
  int64_t offset;
  auto successStrides = getStridesAndOffset(memRefType, strides, offset);
  bool isContiguous = strides.empty() || strides.back() == 1;
  if (isContiguous) {
    auto sizes = memRefType.getShape();
    for (int index = 0, e = strides.size() - 2; index < e; ++index) {
      if (strides[index] != strides[index + 1] * sizes[index + 1]) {
        isContiguous = false;
        break;
      }
    }
  }
  return succeeded(successStrides) && isContiguous;
}

class VectorTypeCastOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorTypeCastOpConversion(MLIRContext *context,
                                      LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::TypeCastOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    vector::TypeCastOp castOp = cast<vector::TypeCastOp>(op);
    MemRefType sourceMemRefType =
        castOp.getOperand().getType().cast<MemRefType>();
    MemRefType targetMemRefType =
        castOp.getResult().getType().cast<MemRefType>();

    // Only static shape casts supported atm.
    if (!sourceMemRefType.hasStaticShape() ||
        !targetMemRefType.hasStaticShape())
      return failure();

    auto llvmSourceDescriptorTy =
        operands[0].getType().dyn_cast<LLVM::LLVMType>();
    if (!llvmSourceDescriptorTy || !llvmSourceDescriptorTy.isStructTy())
      return failure();
    MemRefDescriptor sourceMemRef(operands[0]);

    auto llvmTargetDescriptorTy = typeConverter.convertType(targetMemRefType)
                                      .dyn_cast_or_null<LLVM::LLVMType>();
    if (!llvmTargetDescriptorTy || !llvmTargetDescriptorTy.isStructTy())
      return failure();

    // Only contiguous source tensors supported atm.
    SmallVector<int64_t, 4> strides;
    if (!isContiguous(sourceMemRefType, strides))
      return failure();

    auto int64Ty = LLVM::LLVMType::getInt64Ty(rewriter.getContext());

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
      auto strideAttr =
          rewriter.getIntegerAttr(rewriter.getIndexType(), strides[index]);
      auto stride = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, strideAttr);
      desc.setStride(rewriter, loc, index, stride);
    }

    rewriter.replaceOp(op, {desc});
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
class VectorTransferConversion : public ConvertToLLVMPattern {
public:
  explicit VectorTransferConversion(MLIRContext *context,
                                    LLVMTypeConverter &typeConv,
                                    bool enableIndexOpt)
      : ConvertToLLVMPattern(ConcreteOp::getOperationName(), context, typeConv),
        enableIndexOptimizations(enableIndexOpt) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto xferOp = cast<ConcreteOp>(op);
    auto adaptor = getTransferOpAdapter(xferOp, operands);

    if (xferOp.getVectorType().getRank() > 1 ||
        llvm::size(xferOp.indices()) == 0)
      return failure();
    if (xferOp.permutation_map() !=
        AffineMap::getMinorIdentityMap(xferOp.permutation_map().getNumInputs(),
                                       xferOp.getVectorType().getRank(),
                                       op->getContext()))
      return failure();
    // Only contiguous source tensors supported atm.
    SmallVector<int64_t, 4> strides;
    if (!isContiguous(xferOp.getMemRefType(), strides))
      return failure();

    auto toLLVMTy = [&](Type t) { return typeConverter.convertType(t); };

    Location loc = op->getLoc();
    MemRefType memRefType = xferOp.getMemRefType();

    if (auto memrefVectorElementType =
            memRefType.getElementType().dyn_cast<VectorType>()) {
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
    //    The vector pointer would always be on address space 0, therefore
    //    addrspacecast shall be used when source/dst memrefs are not on
    //    address space 0.
    // TODO: support alignment when possible.
    Value dataPtr = getDataPtr(loc, memRefType, adaptor.memref(),
                               adaptor.indices(), rewriter);
    auto vecTy =
        toLLVMTy(xferOp.getVectorType()).template cast<LLVM::LLVMType>();
    Value vectorDataPtr;
    if (memRefType.getMemorySpace() == 0)
      vectorDataPtr =
          rewriter.create<LLVM::BitcastOp>(loc, vecTy.getPointerTo(), dataPtr);
    else
      vectorDataPtr = rewriter.create<LLVM::AddrSpaceCastOp>(
          loc, vecTy.getPointerTo(), dataPtr);

    if (!xferOp.isMaskedDim(0))
      return replaceTransferOpWithLoadOrStore(rewriter, typeConverter, loc,
                                              xferOp, operands, vectorDataPtr);

    // 2. Create a vector with linear indices [ 0 .. vector_length - 1 ].
    // 3. Create offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
    // 4. Let dim the memref dimension, compute the vector comparison mask:
    //   [ offset + 0 .. offset + vector_length - 1 ] < [ dim .. dim ]
    //
    // TODO: when the leaf transfer rank is k > 1, we need the last `k`
    //       dimensions here.
    unsigned vecWidth = vecTy.getVectorNumElements();
    unsigned lastIndex = llvm::size(xferOp.indices()) - 1;
    Value off = xferOp.indices()[lastIndex];
    Value dim = rewriter.create<DimOp>(loc, xferOp.memref(), lastIndex);
    Value mask = buildVectorComparison(rewriter, op, enableIndexOptimizations,
                                       vecWidth, dim, &off);

    // 5. Rewrite as a masked read / write.
    return replaceTransferOpWithMasked(rewriter, typeConverter, loc, xferOp,
                                       operands, vectorDataPtr, mask);
  }

private:
  const bool enableIndexOptimizations;
};

class VectorPrintOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorPrintOpConversion(MLIRContext *context,
                                   LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::PrintOp::getOperationName(), context,
                             typeConverter) {}

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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto printOp = cast<vector::PrintOp>(op);
    auto adaptor = vector::PrintOpAdaptor(operands);
    Type printType = printOp.getPrintType();

    if (typeConverter.convertType(printType) == nullptr)
      return failure();

    // Make sure element type has runtime support (currently just Float/Double).
    VectorType vectorType = printType.dyn_cast<VectorType>();
    Type eltType = vectorType ? vectorType.getElementType() : printType;
    int64_t rank = vectorType ? vectorType.getRank() : 0;
    Operation *printer;
    if (eltType.isSignlessInteger(1) || eltType.isSignlessInteger(32))
      printer = getPrintI32(op);
    else if (eltType.isSignlessInteger(64))
      printer = getPrintI64(op);
    else if (eltType.isF32())
      printer = getPrintFloat(op);
    else if (eltType.isF64())
      printer = getPrintDouble(op);
    else
      return failure();

    // Unroll vector into elementary print calls.
    emitRanks(rewriter, op, adaptor.source(), vectorType, printer, rank);
    emitCall(rewriter, op->getLoc(), getPrintNewline(op));
    rewriter.eraseOp(op);
    return success();
  }

private:
  void emitRanks(ConversionPatternRewriter &rewriter, Operation *op,
                 Value value, VectorType vectorType, Operation *printer,
                 int64_t rank) const {
    Location loc = op->getLoc();
    if (rank == 0) {
      if (value.getType() == LLVM::LLVMType::getInt1Ty(rewriter.getContext())) {
        // Convert i1 (bool) to i32 so we can use the print_i32 method.
        // This avoids the need for a print_i1 method with an unclear ABI.
        auto i32Type = LLVM::LLVMType::getInt32Ty(rewriter.getContext());
        auto trueVal = rewriter.create<ConstantOp>(
            loc, i32Type, rewriter.getI32IntegerAttr(1));
        auto falseVal = rewriter.create<ConstantOp>(
            loc, i32Type, rewriter.getI32IntegerAttr(0));
        value = rewriter.create<SelectOp>(loc, value, trueVal, falseVal);
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
      auto llvmType = typeConverter.convertType(
          rank > 1 ? reducedType : vectorType.getElementType());
      Value nestedVal =
          extractOne(rewriter, typeConverter, loc, value, llvmType, rank, d);
      emitRanks(rewriter, op, nestedVal, reducedType, printer, rank - 1);
      if (d != dim - 1)
        emitCall(rewriter, loc, printComma);
    }
    emitCall(rewriter, loc, getPrintClose(op));
  }

  // Helper to emit a call.
  static void emitCall(ConversionPatternRewriter &rewriter, Location loc,
                       Operation *ref, ValueRange params = ValueRange()) {
    rewriter.create<LLVM::CallOp>(loc, ArrayRef<Type>{},
                                  rewriter.getSymbolRefAttr(ref), params);
  }

  // Helper for printer method declaration (first hit) and lookup.
  static Operation *getPrint(Operation *op, StringRef name,
                             ArrayRef<LLVM::LLVMType> params) {
    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (func)
      return func;
    OpBuilder moduleBuilder(module.getBodyRegion());
    return moduleBuilder.create<LLVM::LLVMFuncOp>(
        op->getLoc(), name,
        LLVM::LLVMType::getFunctionTy(
            LLVM::LLVMType::getVoidTy(op->getContext()), params,
            /*isVarArg=*/false));
  }

  // Helpers for method names.
  Operation *getPrintI32(Operation *op) const {
    return getPrint(op, "print_i32",
                    LLVM::LLVMType::getInt32Ty(op->getContext()));
  }
  Operation *getPrintI64(Operation *op) const {
    return getPrint(op, "print_i64",
                    LLVM::LLVMType::getInt64Ty(op->getContext()));
  }
  Operation *getPrintFloat(Operation *op) const {
    return getPrint(op, "print_f32",
                    LLVM::LLVMType::getFloatTy(op->getContext()));
  }
  Operation *getPrintDouble(Operation *op) const {
    return getPrint(op, "print_f64",
                    LLVM::LLVMType::getDoubleTy(op->getContext()));
  }
  Operation *getPrintOpen(Operation *op) const {
    return getPrint(op, "print_open", {});
  }
  Operation *getPrintClose(Operation *op) const {
    return getPrint(op, "print_close", {});
  }
  Operation *getPrintComma(Operation *op) const {
    return getPrint(op, "print_comma", {});
  }
  Operation *getPrintNewline(Operation *op) const {
    return getPrint(op, "print_newline", {});
  }
};

/// Progressive lowering of ExtractStridedSliceOp to either:
///   1. express single offset extract as a direct shuffle.
///   2. extract + lower rank strided_slice + insert for the n-D case.
class VectorExtractStridedSliceOpConversion
    : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  using OpRewritePattern<ExtractStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto dstType = op.getResult().getType().cast<VectorType>();

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
  /// This pattern creates recursive ExtractStridedSliceOp, but the recursion is
  /// bounded as the rank is strictly decreasing.
  bool hasBoundedRewriteRecursion() const final { return true; }
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
      ctx, converter, reassociateFPReductions);
  patterns.insert<VectorCreateMaskOpConversion,
                  VectorTransferConversion<TransferReadOp>,
                  VectorTransferConversion<TransferWriteOp>>(
      ctx, converter, enableIndexOptimizations);
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
              VectorCompressStoreOpConversion>(ctx, converter);
  // clang-format on
}

void mlir::populateVectorToLLVMMatrixConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();
  patterns.insert<VectorMatmulOpConversion>(ctx, converter);
  patterns.insert<VectorFlatTransposeOpConversion>(ctx, converter);
}

namespace {
struct LowerVectorToLLVMPass
    : public ConvertVectorToLLVMBase<LowerVectorToLLVMPass> {
  LowerVectorToLLVMPass(const LowerVectorToLLVMOptions &options) {
    this->reassociateFPReductions = options.reassociateFPReductions;
    this->enableIndexOptimizations = options.enableIndexOptimizations;
  }
  void runOnOperation() override;
};
} // namespace

void LowerVectorToLLVMPass::runOnOperation() {
  // Perform progressive lowering of operations on slices and
  // all contraction operations. Also applies folding and DCE.
  {
    OwningRewritePatternList patterns;
    populateVectorToVectorCanonicalizationPatterns(patterns, &getContext());
    populateVectorSlicesLoweringPatterns(patterns, &getContext());
    populateVectorContractLoweringPatterns(patterns, &getContext());
    applyPatternsAndFoldGreedily(getOperation(), patterns);
  }

  // Convert to the LLVM IR dialect.
  LLVMTypeConverter converter(&getContext());
  OwningRewritePatternList patterns;
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(
      converter, patterns, reassociateFPReductions, enableIndexOptimizations);
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateStdToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  if (failed(applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertVectorToLLVMPass(const LowerVectorToLLVMOptions &options) {
  return std::make_unique<LowerVectorToLLVMPass>(options);
}
