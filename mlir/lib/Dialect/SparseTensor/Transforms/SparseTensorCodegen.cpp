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
  //   memref<rank x index> dimSize  ; size in each dimension
  //   ; per-dimension d:
  //   ;  if dense:
  //        <nothing>
  //   ;  if compresed:
  //        memref<? x idx>  indices-d   ; indices for sparse dim d
  //        memref<? x ptr>  pointers-d  ; pointers for sparse dim d
  //   ;  if singleton:
  //        memref<? x idx>  indices-d   ; indices for singleton dim d
  //   memref<? x eltType> values    ; values
  // };
  //
  // TODO: fill in the ? when statically known
  //
  // TODO: emit dimSizes when not needed (e.g. all-dense)
  //
  unsigned rank = rType.getShape().size();
  SmallVector<Type, 8> fields;
  fields.push_back(MemRefType::get({rank}, indexType));
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
      fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, idxType));
      fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, ptrType));
      break;
    case SparseTensorEncodingAttr::DimLevelType::Singleton:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNu:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNo:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNuNo:
      fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, idxType));
      break;
    }
  }
  fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, eltType));
  // Sparse tensor storage (temporarily) lives in a tuple. This allows a
  // simple 1:1 type conversion during codegen. A subsequent pass uses
  // a 1:N type conversion to expand the tuple into its fields.
  return TupleType::get(context, fields);
}

//===----------------------------------------------------------------------===//
// Conversion rules.
//===----------------------------------------------------------------------===//

/// Sparse conversion rule for returns.
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
  patterns.add<SparseReturnConverter>(typeConverter, patterns.getContext());
}
