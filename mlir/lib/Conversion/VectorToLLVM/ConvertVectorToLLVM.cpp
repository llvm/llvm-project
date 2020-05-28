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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::vector;

template <typename T>
static LLVM::LLVMType getPtrToElementType(T containerType,
                                          LLVMTypeConverter &typeConverter) {
  return typeConverter.convertType(containerType.getElementType())
      .template cast<LLVM::LLVMType>()
      .getPointerTo();
}

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
// TODO(rriddle): Better support for attribute subtype forwarding + slicing.
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

template <typename TransferOp>
LogicalResult getVectorTransferAlignment(LLVMTypeConverter &typeConverter,
                                         TransferOp xferOp, unsigned &align) {
  Type elementTy =
      typeConverter.convertType(xferOp.getMemRefType().getElementType());
  if (!elementTy)
    return failure();

  auto dataLayout = typeConverter.getDialect()->getLLVMModule().getDataLayout();
  align = dataLayout.getPrefTypeAlignment(
      elementTy.cast<LLVM::LLVMType>().getUnderlyingType());
  return success();
}

static LogicalResult
replaceTransferOpWithLoadOrStore(ConversionPatternRewriter &rewriter,
                                 LLVMTypeConverter &typeConverter, Location loc,
                                 TransferReadOp xferOp,
                                 ArrayRef<Value> operands, Value dataPtr) {
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(xferOp, dataPtr);
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
  if (failed(getVectorTransferAlignment(typeConverter, xferOp, align)))
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
  auto adaptor = TransferWriteOpOperandAdaptor(operands);
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(xferOp, adaptor.vector(), dataPtr);
  return success();
}

static LogicalResult
replaceTransferOpWithMasked(ConversionPatternRewriter &rewriter,
                            LLVMTypeConverter &typeConverter, Location loc,
                            TransferWriteOp xferOp, ArrayRef<Value> operands,
                            Value dataPtr, Value mask) {
  unsigned align;
  if (failed(getVectorTransferAlignment(typeConverter, xferOp, align)))
    return failure();

  auto adaptor = TransferWriteOpOperandAdaptor(operands);
  rewriter.replaceOpWithNewOp<LLVM::MaskedStoreOp>(
      xferOp, adaptor.vector(), dataPtr, mask,
      rewriter.getI32IntegerAttr(align));
  return success();
}

static TransferReadOpOperandAdaptor
getTransferOpAdapter(TransferReadOp xferOp, ArrayRef<Value> operands) {
  return TransferReadOpOperandAdaptor(operands);
}

static TransferWriteOpOperandAdaptor
getTransferOpAdapter(TransferWriteOp xferOp, ArrayRef<Value> operands) {
  return TransferWriteOpOperandAdaptor(operands);
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
    auto adaptor = vector::MatmulOpOperandAdaptor(operands);
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
    auto adaptor = vector::FlatTransposeOpOperandAdaptor(operands);
    rewriter.replaceOpWithNewOp<LLVM::MatrixTransposeOp>(
        transOp, typeConverter.convertType(transOp.res().getType()),
        adaptor.matrix(), transOp.rows(), transOp.columns());
    return success();
  }
};

class VectorReductionOpConversion : public ConvertToLLVMPattern {
public:
  explicit VectorReductionOpConversion(MLIRContext *context,
                                       LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(vector::ReductionOp::getOperationName(), context,
                             typeConverter) {}

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
            op, llvmType, acc, operands[0]);
      } else if (kind == "mul") {
        // Optional accumulator (or one).
        Value acc = operands.size() > 1
                        ? operands[1]
                        : rewriter.create<LLVM::ConstantOp>(
                              op->getLoc(), llvmType,
                              rewriter.getFloatAttr(eltType, 1.0));
        rewriter.replaceOpWithNewOp<LLVM::experimental_vector_reduce_v2_fmul>(
            op, llvmType, acc, operands[0]);
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
    auto adaptor = vector::ShuffleOpOperandAdaptor(operands);
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
    auto adaptor = vector::ExtractElementOpOperandAdaptor(operands);
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
    auto adaptor = vector::ExtractOpOperandAdaptor(operands);
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
    auto i64Type = LLVM::LLVMType::getInt64Ty(typeConverter.getDialect());
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
///  llvm.intr.fma %va, %va, %va:
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
    auto adaptor = vector::FMAOpOperandAdaptor(operands);
    vector::FMAOp fmaOp = cast<vector::FMAOp>(op);
    VectorType vType = fmaOp.getVectorType();
    if (vType.getRank() != 1)
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::FMAOp>(op, adaptor.lhs(), adaptor.rhs(),
                                             adaptor.acc());
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
    auto adaptor = vector::InsertElementOpOperandAdaptor(operands);
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
    auto adaptor = vector::InsertOpOperandAdaptor(operands);
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
    auto i64Type = LLVM::LLVMType::getInt64Ty(typeConverter.getDialect());
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

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides =
        getStridesAndOffset(sourceMemRefType, strides, offset);
    bool isContiguous = (strides.back() == 1);
    if (isContiguous) {
      auto sizes = sourceMemRefType.getShape();
      for (int index = 0, e = strides.size() - 2; index < e; ++index) {
        if (strides[index] != strides[index + 1] * sizes[index + 1]) {
          isContiguous = false;
          break;
        }
      }
    }
    // Only contiguous source tensors supported atm.
    if (failed(successStrides) || !isContiguous)
      return failure();

    auto int64Ty = LLVM::LLVMType::getInt64Ty(typeConverter.getDialect());

    // Create descriptor.
    auto desc = MemRefDescriptor::undef(rewriter, loc, llvmTargetDescriptorTy);
    Type llvmTargetElementTy = desc.getElementType();
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
/// 1. Bitcast or addrspacecast to vector form.
/// 2. Create an offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
/// 3. Create a mask where offsetVector is compared against memref upper bound.
/// 4. Rewrite op as a masked read or write.
template <typename ConcreteOp>
class VectorTransferConversion : public ConvertToLLVMPattern {
public:
  explicit VectorTransferConversion(MLIRContext *context,
                                    LLVMTypeConverter &typeConv)
      : ConvertToLLVMPattern(ConcreteOp::getOperationName(), context,
                             typeConv) {}

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

    auto toLLVMTy = [&](Type t) { return typeConverter.convertType(t); };

    Location loc = op->getLoc();
    Type i64Type = rewriter.getIntegerType(64);
    MemRefType memRefType = xferOp.getMemRefType();

    // 1. Get the source/dst address as an LLVM vector pointer.
    //    The vector pointer would always be on address space 0, therefore
    //    addrspacecast shall be used when source/dst memrefs are not on
    //    address space 0.
    // TODO: support alignment when possible.
    Value dataPtr = getDataPtr(loc, memRefType, adaptor.memref(),
                               adaptor.indices(), rewriter, getModule());
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
    unsigned vecWidth = vecTy.getVectorNumElements();
    VectorType vectorCmpType = VectorType::get(vecWidth, i64Type);
    SmallVector<int64_t, 8> indices;
    indices.reserve(vecWidth);
    for (unsigned i = 0; i < vecWidth; ++i)
      indices.push_back(i);
    Value linearIndices = rewriter.create<ConstantOp>(
        loc, vectorCmpType,
        DenseElementsAttr::get(vectorCmpType, ArrayRef<int64_t>(indices)));
    linearIndices = rewriter.create<LLVM::DialectCastOp>(
        loc, toLLVMTy(vectorCmpType), linearIndices);

    // 3. Create offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
    // TODO(ntv, ajcbik): when the leaf transfer rank is k > 1 we need the last
    // `k` dimensions here.
    unsigned lastIndex = llvm::size(xferOp.indices()) - 1;
    Value offsetIndex = *(xferOp.indices().begin() + lastIndex);
    offsetIndex = rewriter.create<IndexCastOp>(loc, i64Type, offsetIndex);
    Value base = rewriter.create<SplatOp>(loc, vectorCmpType, offsetIndex);
    Value offsetVector = rewriter.create<AddIOp>(loc, base, linearIndices);

    // 4. Let dim the memref dimension, compute the vector comparison mask:
    //   [ offset + 0 .. offset + vector_length - 1 ] < [ dim .. dim ]
    Value dim = rewriter.create<DimOp>(loc, xferOp.memref(), lastIndex);
    dim = rewriter.create<IndexCastOp>(loc, i64Type, dim);
    dim = rewriter.create<SplatOp>(loc, vectorCmpType, dim);
    Value mask =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, offsetVector, dim);
    mask = rewriter.create<LLVM::DialectCastOp>(loc, toLLVMTy(mask.getType()),
                                                mask);

    // 5. Rewrite as a masked read / write.
    return replaceTransferOpWithMasked(rewriter, typeConverter, loc, xferOp,
                                       operands, vectorDataPtr, mask);
  }
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
  // TODO(ajcbik): rely solely on libc in future? something else?
  //
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto printOp = cast<vector::PrintOp>(op);
    auto adaptor = vector::PrintOpOperandAdaptor(operands);
    Type printType = printOp.getPrintType();

    if (typeConverter.convertType(printType) == nullptr)
      return failure();

    // Make sure element type has runtime support (currently just Float/Double).
    VectorType vectorType = printType.dyn_cast<VectorType>();
    Type eltType = vectorType ? vectorType.getElementType() : printType;
    int64_t rank = vectorType ? vectorType.getRank() : 0;
    Operation *printer;
    if (eltType.isSignlessInteger(1))
      printer = getPrintI1(op);
    else if (eltType.isSignlessInteger(32))
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
  static Operation *getPrint(Operation *op, LLVM::LLVMDialect *dialect,
                             StringRef name, ArrayRef<LLVM::LLVMType> params) {
    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (func)
      return func;
    OpBuilder moduleBuilder(module.getBodyRegion());
    return moduleBuilder.create<LLVM::LLVMFuncOp>(
        op->getLoc(), name,
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getVoidTy(dialect),
                                      params, /*isVarArg=*/false));
  }

  // Helpers for method names.
  Operation *getPrintI1(Operation *op) const {
    LLVM::LLVMDialect *dialect = typeConverter.getDialect();
    return getPrint(op, dialect, "print_i1",
                    LLVM::LLVMType::getInt1Ty(dialect));
  }
  Operation *getPrintI32(Operation *op) const {
    LLVM::LLVMDialect *dialect = typeConverter.getDialect();
    return getPrint(op, dialect, "print_i32",
                    LLVM::LLVMType::getInt32Ty(dialect));
  }
  Operation *getPrintI64(Operation *op) const {
    LLVM::LLVMDialect *dialect = typeConverter.getDialect();
    return getPrint(op, dialect, "print_i64",
                    LLVM::LLVMType::getInt64Ty(dialect));
  }
  Operation *getPrintFloat(Operation *op) const {
    LLVM::LLVMDialect *dialect = typeConverter.getDialect();
    return getPrint(op, dialect, "print_f32",
                    LLVM::LLVMType::getFloatTy(dialect));
  }
  Operation *getPrintDouble(Operation *op) const {
    LLVM::LLVMDialect *dialect = typeConverter.getDialect();
    return getPrint(op, dialect, "print_f64",
                    LLVM::LLVMType::getDoubleTy(dialect));
  }
  Operation *getPrintOpen(Operation *op) const {
    return getPrint(op, typeConverter.getDialect(), "print_open", {});
  }
  Operation *getPrintClose(Operation *op) const {
    return getPrint(op, typeConverter.getDialect(), "print_close", {});
  }
  Operation *getPrintComma(Operation *op) const {
    return getPrint(op, typeConverter.getDialect(), "print_comma", {});
  }
  Operation *getPrintNewline(Operation *op) const {
    return getPrint(op, typeConverter.getDialect(), "print_newline", {});
  }
};

/// Progressive lowering of ExtractStridedSliceOp to either:
///   1. extractelement + insertelement for the 1-D case
///   2. extract + optional strided_slice + insert for the n-D case.
class VectorStridedSliceOpConversion
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
    Value zero = rewriter.create<ConstantOp>(loc, elemType,
                                             rewriter.getZeroAttr(elemType));
    Value res = rewriter.create<SplatOp>(loc, dstType, zero);
    for (int64_t off = offset, e = offset + size * stride, idx = 0; off < e;
         off += stride, ++idx) {
      Value extracted = extractOne(rewriter, loc, op.vector(), off);
      if (op.offsets().getValue().size() > 1) {
        extracted = rewriter.create<ExtractStridedSliceOp>(
            loc, extracted, getI64SubArray(op.offsets(), /* dropFront=*/1),
            getI64SubArray(op.sizes(), /* dropFront=*/1),
            getI64SubArray(op.strides(), /* dropFront=*/1));
      }
      res = insertOne(rewriter, loc, extracted, res, idx);
    }
    rewriter.replaceOp(op, {res});
    return success();
  }
  /// This pattern creates recursive ExtractStridedSliceOp, but the recursion is
  /// bounded as the rank is strictly decreasing.
  bool hasBoundedRewriteRecursion() const final { return true; }
};

} // namespace

/// Populate the given list with patterns that convert from Vector to LLVM.
void mlir::populateVectorToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();
  // clang-format off
  patterns.insert<VectorFMAOpNDRewritePattern,
                  VectorInsertStridedSliceOpDifferentRankRewritePattern,
                  VectorInsertStridedSliceOpSameRankRewritePattern,
                  VectorStridedSliceOpConversion>(ctx);
  patterns
      .insert<VectorReductionOpConversion,
              VectorShuffleOpConversion,
              VectorExtractElementOpConversion,
              VectorExtractOpConversion,
              VectorFMAOp1DConversion,
              VectorInsertElementOpConversion,
              VectorInsertOpConversion,
              VectorPrintOpConversion,
              VectorTransferConversion<TransferReadOp>,
              VectorTransferConversion<TransferWriteOp>,
              VectorTypeCastOpConversion>(ctx, converter);
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
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateStdToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
  if (failed(applyPartialConversion(getOperation(), target, patterns,
                                    &converter))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertVectorToLLVMPass() {
  return std::make_unique<LowerVectorToLLVMPass>();
}
