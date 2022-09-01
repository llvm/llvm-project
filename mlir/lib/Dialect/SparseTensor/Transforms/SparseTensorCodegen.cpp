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

/// Reorders stored dimension to logical dimension.
static unsigned reorder(const SparseTensorEncodingAttr &enc, unsigned d) {
  auto order = enc.getDimOrdering();
  if (order) {
    assert(order.isPermutation());
    return order.getDimPosition(d);
  }
  return d;
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
  ArrayRef<int64_t> shape = rType.getShape();
  //
  // Sparse tensor storage for rank-dimensional tensor is organized as a
  // single compound type with the following fields:
  //
  // struct {
  //   ; if dynamic shape:
  //     memref<rank x index> dimSize    ; size in each dimension
  //   ; per-dimension d:
  //   ;  if dense:
  //        <nothing>
  //   ;  if compresed:
  //        memref<? x idx>  indices-d   ; indices for sparse dim d
  //        memref<? x ptr>  pointers-d  ; pointers for sparse dim d
  //   ;  if singleton:
  //        memref<? x idx>  indices-d   ; indices for singleton dim d
  //   memref<? x eltType> values        ; values
  // };
  //
  int64_t linear = 1;
  bool allDense = true;
  unsigned rank = rType.getShape().size();
  SmallVector<Type, 8> fields;
  // The dimSizes array.
  if (!rType.hasStaticShape())
    fields.push_back(MemRefType::get({rank}, indexType));
  // Per-dimension storage.
  for (unsigned r = 0; r < rank; r++) {
    // Get the original dimension (ro) for the current stored dimension (r).
    unsigned ro = reorder(enc, r);
    // Dimension level types apply in order to the reordered dimension.
    // As a result, the compound type can be constructed directly in the given
    // order. Clients of this type know what field is what from the sparse
    // tensor type.
    switch (enc.getDimLevelType()[r]) {
    case SparseTensorEncodingAttr::DimLevelType::Dense:
      // Linearize the size of consecutive dense dimensions.
      if (ShapedType::isDynamic(shape[ro]) || ShapedType::isDynamic(linear))
        linear = ShapedType::kDynamicSize;
      else
        linear *= shape[ro];
      break;
    case SparseTensorEncodingAttr::DimLevelType::Compressed:
    case SparseTensorEncodingAttr::DimLevelType::CompressedNu:
    case SparseTensorEncodingAttr::DimLevelType::CompressedNo:
    case SparseTensorEncodingAttr::DimLevelType::CompressedNuNo:
      fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, idxType));
      fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, ptrType));
      allDense = false;
      linear = 1;
      break;
    case SparseTensorEncodingAttr::DimLevelType::Singleton:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNu:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNo:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNuNo:
      fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, idxType));
      allDense = false;
      linear = 1;
      break;
    }
  }
  // The values array.
  int64_t nnz =
      (rType.hasStaticShape() && allDense) ? linear : ShapedType::kDynamicSize;
  fields.push_back(MemRefType::get({nnz}, eltType));
  // Sparse tensor storage (temporarily) lives in a tuple. This allows a
  // simple 1:1 type conversion during codegen. A subsequent pass uses
  // a 1:N type conversion to expand the tuple into its fields.
  return TupleType::get(context, fields);
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
    Location loc = op->getLoc();
    Type type = op.getSource().getType();
    // Only rewrite annotated DimOp with constant index.
    auto enc = getSparseTensorEncoding(type);
    if (!enc)
      return failure();
    Optional<int64_t> index = op.getConstantIndex();
    if (!index)
      return failure();
    // Access into static shape can query original type directly.
    // Note that this is typically already done by DimOp's folding.
    RankedTensorType rType = type.cast<RankedTensorType>();
    if (rType.hasStaticShape()) {
      rewriter.replaceOp(
          op, constantIndex(rewriter, loc, rType.getShape()[*index]));
      return success();
    }
    // Any other query can consult the dimSize array.
    // TODO: this needs tuple access
    return failure();
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
  patterns.add<SparseReturnConverter, SparseDimOpConverter>(
      typeConverter, patterns.getContext());
}
