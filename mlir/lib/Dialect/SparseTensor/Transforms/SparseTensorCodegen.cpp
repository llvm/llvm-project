//===- SparseTensorCodegen.cpp - Sparse tensor primitives conversion ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that converts sparse tensor types and primitives to actual compiler
// visible buffers and actual compiler IR that implements these primitives on
// the selected sparse tensor storage schemes. This pass provides an alternative
// to the SparseTensorConversion pass, eliminating the dependence on a runtime
// support library, and providing much more opportunities for subsequent
// compiler optimization of the generated code.
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// Reorders original dimension to stored dimension.
static unsigned toStored(const SparseTensorEncodingAttr &enc, unsigned i) {
  auto order = enc.getDimOrdering();
  if (order) {
    assert(order.isPermutation());
    return order.getPermutedPosition(i);
  }
  return i;
}

/// Maps a sparse tensor type to the appropriate compounded buffers.
static Optional<Type> convertSparseTensorType(Type type) {
  auto enc = getSparseTensorEncoding(type);
  if (!enc)
    return llvm::None;
  // Construct the basic types.
  auto context = type.getContext();
  unsigned idxWidth = enc.getIndexBitWidth();
  unsigned ptrWidth = enc.getPointerBitWidth();
  RankedTensorType rType = type.cast<RankedTensorType>();
  Type indexType = IndexType::get(context);
  Type idxType = idxWidth ? IntegerType::get(context, idxWidth) : indexType;
  Type ptrType = ptrWidth ? IntegerType::get(context, ptrWidth) : indexType;
  Type eltType = rType.getElementType();
  //
  // Sparse tensor storage for rank-dimensional tensor is organized as a
  // single compound type with the following fields:
  //
  // struct {
  //   memref<rank x index> dimSizes     ; size in each dimension
  //   ; per-dimension d:
  //   ;  if dense:
  //        <nothing>
  //   ;  if compresed:
  //        memref<? x ptr>  pointers-d  ; pointers for sparse dim d
  //        memref<? x idx>  indices-d   ; indices for sparse dim d
  //   ;  if singleton:
  //        memref<? x idx>  indices-d   ; indices for singleton dim d
  //   memref<? x eltType> values        ; values
  // };
  //
  unsigned rank = rType.getShape().size();
  SmallVector<Type, 8> fields;
  // The dimSizes array.
  fields.push_back(MemRefType::get({rank}, indexType));
  // Per-dimension storage.
  for (unsigned r = 0; r < rank; r++) {
    // Dimension level types apply in order to the reordered dimension.
    // As a result, the compound type can be constructed directly in the given
    // order. Clients of this type know what field is what from the sparse
    // tensor type.
    switch (enc.getDimLevelType()[r]) {
    case SparseTensorEncodingAttr::DimLevelType::Dense:
      break;
    case SparseTensorEncodingAttr::DimLevelType::Compressed:
    case SparseTensorEncodingAttr::DimLevelType::CompressedNu:
    case SparseTensorEncodingAttr::DimLevelType::CompressedNo:
    case SparseTensorEncodingAttr::DimLevelType::CompressedNuNo:
      fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, ptrType));
      fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, idxType));
      break;
    case SparseTensorEncodingAttr::DimLevelType::Singleton:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNu:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNo:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNuNo:
      fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, idxType));
      break;
    }
  }
  // The values array.
  fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, eltType));
  // Sparse tensor storage (temporarily) lives in a tuple. This allows a
  // simple 1:1 type conversion during codegen. A subsequent pass uses
  // a 1:N type conversion to expand the tuple into its fields.
  return TupleType::get(context, fields);
}

// Returns field index for pointers (d), indices (d) for set field.
static unsigned getFieldIndex(Type type, unsigned ptrDim, unsigned idxDim) {
  auto enc = getSparseTensorEncoding(type);
  assert(enc);
  RankedTensorType rType = type.cast<RankedTensorType>();
  unsigned field = 1; // start at DimSizes;
  unsigned ptr = 0;
  unsigned idx = 0;
  for (unsigned r = 0, rank = rType.getShape().size(); r < rank; r++) {
    switch (enc.getDimLevelType()[r]) {
    case SparseTensorEncodingAttr::DimLevelType::Dense:
      break; // no fields
    case SparseTensorEncodingAttr::DimLevelType::Compressed:
    case SparseTensorEncodingAttr::DimLevelType::CompressedNu:
    case SparseTensorEncodingAttr::DimLevelType::CompressedNo:
    case SparseTensorEncodingAttr::DimLevelType::CompressedNuNo:
      if (ptr++ == ptrDim)
        return field;
      field++;
      if (idx++ == idxDim)
        return field;
      field++;
      break;
    case SparseTensorEncodingAttr::DimLevelType::Singleton:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNu:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNo:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNuNo:
      if (idx++ == idxDim)
        return field;
      field++;
      break;
    }
  }
  llvm_unreachable("failed to find ptr/idx field index");
  return -1;
}

/// Returns field type in tuple at given index.
static Type getFieldType(Value tuple, unsigned field) {
  return tuple.getType().cast<TupleType>().getType(field);
}

/// Creates tuple get operation at given index.
static Value createTupleGet(OpBuilder &builder, Location loc, Value tuple,
                            unsigned field) {
  Type indexType = builder.getIndexType();
  return builder.create<StorageGetOp>(loc, getFieldType(tuple, field), tuple,
                                      builder.getIntegerAttr(indexType, field));
}

/// Returns integral constant, if defined.
static Optional<int64_t> getConstantInt(Value val) {
  if (auto constantOp = val.getDefiningOp<arith::ConstantOp>())
    return constantOp.getValue().cast<IntegerAttr>().getInt();
  return {};
}

//===----------------------------------------------------------------------===//
// Codegen rules.
//===----------------------------------------------------------------------===//

/// Sparse codegen rule for returns.
class SparseReturnConverter : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

/// Sparse codegen rule for dimension accesses.
class SparseDimOpConverter : public OpConversionPattern<tensor::DimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only rewrite annotated DimOp with constant index.
    auto enc = getSparseTensorEncoding(op.getSource().getType());
    if (!enc)
      return failure();
    Optional<int64_t> index = getConstantInt(adaptor.getIndex());
    if (!index)
      return failure();
    // Access into static dimension can query original type directly.
    // Note that this is typically already done by DimOp's folding.
    Location loc = op->getLoc();
    auto shape = op.getSource().getType().cast<RankedTensorType>().getShape();
    if (!ShapedType::isDynamic(shape[*index])) {
      rewriter.replaceOp(op, constantIndex(rewriter, loc, shape[*index]));
      return success();
    }
    // Any other query can consult the dimSizes array at field 0 using,
    // accounting for the reordering applied to the sparse storage.
    Value tuple = adaptor.getSource();
    Value dimSizes = createTupleGet(rewriter, loc, tuple, 0);
    rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, dimSizes, constantIndex(rewriter, loc, toStored(enc, *index)));
    return success();
  }
};

/// Sparse codegen rule for trivial tensor casts.
class SparseCastConverter : public OpConversionPattern<tensor::CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only rewrite identically annotated source/dest.
    auto encDst = getSparseTensorEncoding(op.getType());
    auto encSrc = getSparseTensorEncoding(op.getSource().getType());
    if (!encDst || encDst != encSrc)
      return failure();
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Sparse codegen rule for the dealloc operator.
class SparseTensorDeallocConverter
    : public OpConversionPattern<bufferization::DeallocTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(bufferization::DeallocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto enc = getSparseTensorEncoding(op.getTensor().getType());
    if (!enc)
      return failure();
    // Replace the tuple deallocation with field deallocations.
    Location loc = op->getLoc();
    Value tuple = adaptor.getTensor();
    for (unsigned i = 0, sz = tuple.getType().cast<TupleType>().size(); i < sz;
         i++) {
      Value mem = createTupleGet(rewriter, loc, tuple, i);
      rewriter.create<memref::DeallocOp>(loc, mem);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Sparse codegen rule for pointer accesses.
class SparseToPointersConverter : public OpConversionPattern<ToPointersOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToPointersOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Optional<int64_t> index = getConstantInt(adaptor.getOperands()[1]);
    if (!index)
      return failure();
    // Replace the requested pointer access with corresponding field.
    Location loc = op->getLoc();
    Value tuple = adaptor.getTensor();
    unsigned i = getFieldIndex(op.getTensor().getType(), /*ptrDim=*/*index, -1);
    rewriter.replaceOp(op, createTupleGet(rewriter, loc, tuple, i));
    return success();
  }
};

/// Sparse codegen rule for index accesses.
class SparseToIndicesConverter : public OpConversionPattern<ToIndicesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToIndicesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Optional<int64_t> index = getConstantInt(adaptor.getOperands()[1]);
    if (!index)
      return failure();
    // Replace the requested indices access with corresponding field.
    Location loc = op->getLoc();
    Value tuple = adaptor.getTensor();
    unsigned i = getFieldIndex(op.getTensor().getType(), -1, /*idxDim=*/*index);
    rewriter.replaceOp(op, createTupleGet(rewriter, loc, tuple, i));
    return success();
  }
};

/// Sparse codegen rule for value accesses.
class SparseToValuesConverter : public OpConversionPattern<ToValuesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToValuesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the requested values access with corresponding field.
    Location loc = op->getLoc();
    Value tuple = adaptor.getTensor();
    unsigned i = tuple.getType().cast<TupleType>().size() - 1; // last
    rewriter.replaceOp(op, createTupleGet(rewriter, loc, tuple, i));
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Sparse tensor type conversion into an actual buffer.
//===----------------------------------------------------------------------===//

mlir::SparseTensorTypeToBufferConverter::SparseTensorTypeToBufferConverter() {
  addConversion([](Type type) { return type; });
  addConversion(convertSparseTensorType);
}

//===----------------------------------------------------------------------===//
// Public method for populating conversion rules.
//===----------------------------------------------------------------------===//

/// Populates the given patterns list with conversion rules required for
/// the sparsification of linear algebra operations.
void mlir::populateSparseTensorCodegenPatterns(TypeConverter &typeConverter,
                                               RewritePatternSet &patterns) {
  patterns.add<SparseReturnConverter, SparseDimOpConverter, SparseCastConverter,
               SparseTensorDeallocConverter, SparseToPointersConverter,
               SparseToIndicesConverter, SparseToValuesConverter>(
      typeConverter, patterns.getContext());
}
