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
#include "mlir/Dialect/Linalg/Utils/Utils.h"
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

/// Reorders stored dimension to original dimension.
static unsigned toOrig(const SparseTensorEncodingAttr &enc, unsigned i) {
  auto order = enc.getDimOrdering();
  if (order) {
    assert(order.isPermutation());
    return order.getDimPosition(i);
  }
  return i;
}

/// Reorders original dimension to stored dimension.
static unsigned toStored(const SparseTensorEncodingAttr &enc, unsigned i) {
  auto order = enc.getDimOrdering();
  if (order) {
    assert(order.isPermutation());
    return order.getPermutedPosition(i);
  }
  return i;
}

/// Flatten a list of operands that may contain sparse tensors.
static void flattenOperands(ValueRange operands,
                            SmallVectorImpl<Value> &flattened) {
  // In case of
  // sparse_tensor, c, sparse_tensor
  // ==>
  // memref ..., c, memref ...
  for (auto operand : operands) {
    if (auto cast =
            dyn_cast<UnrealizedConversionCastOp>(operand.getDefiningOp());
        cast && getSparseTensorEncoding(cast->getResultTypes()[0]))
      // An unrealized_conversion_cast will be inserted by type converter to
      // inter-mix the gap between 1:N conversion between sparse tensors and
      // fields. In this case, take the operands in the cast and replace the
      // sparse tensor output with the flattened type array.
      flattened.append(cast.getOperands().begin(), cast.getOperands().end());
    else
      flattened.push_back(operand);
  }
}

/// Gets the dimension size for the given sparse tensor at the given dim.
/// Returns None if no sparse encoding is attached to the tensor type.
static Optional<Value> sizeFromTensorAtDim(OpBuilder &rewriter, Location loc,
                                           ShapedType tensorTp,
                                           Value adaptedValue, unsigned dim) {
  auto enc = getSparseTensorEncoding(tensorTp);
  if (!enc)
    return llvm::None;

  // Access into static dimension can query original type directly.
  // Note that this is typically already done by DimOp's folding.
  auto shape = tensorTp.getShape();
  if (!ShapedType::isDynamic(shape[dim]))
    return constantIndex(rewriter, loc, shape[dim]);

  // Any other query can consult the dimSizes array at field 0 using,
  // accounting for the reordering applied to the sparse storage.
  auto tuple =
      llvm::cast<UnrealizedConversionCastOp>(adaptedValue.getDefiningOp());
  return rewriter
      .create<memref::LoadOp>(loc, tuple.getInputs().front(),
                              constantIndex(rewriter, loc, toStored(enc, dim)))
      .getResult();
}

/// Returns field index of sparse tensor type for pointers/indices, when set.
static unsigned getFieldIndex(Type type, unsigned ptrDim, unsigned idxDim) {
  auto enc = getSparseTensorEncoding(type);
  assert(enc);
  RankedTensorType rType = type.cast<RankedTensorType>();
  unsigned field = 2; // start past sizes
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
  return field + 1; // return values field index
}

/// Maps a sparse tensor type to the appropriate compounded buffers.
static Optional<LogicalResult>
convertSparseTensorType(Type type, SmallVectorImpl<Type> &fields) {
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
  // single compound type with the following fields. Note that every
  // memref with ? size actually behaves as a "vector", i.e. the stored
  // size is the capacity and the used size resides in the memSizes array.
  //
  // struct {
  //   memref<rank x index> dimSizes     ; size in each dimension
  //   memref<n x index> memSizes        ; sizes of ptrs/inds/values
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
  // The dimSizes array.
  fields.push_back(MemRefType::get({rank}, indexType));
  // The memSizes array.
  unsigned lastField = getFieldIndex(type, -1, -1);
  fields.push_back(MemRefType::get({lastField - 2}, indexType));
  // Per-dimension storage.
  for (unsigned r = 0; r < rank; r++) {
    // Dimension level types apply in order to the reordered dimension.
    // As a result, the compound type can be constructed directly in the given
    // order. Clients of this type know what field is what from the sparse
    // tensor type.
    switch (enc.getDimLevelType()[r]) {
    case SparseTensorEncodingAttr::DimLevelType::Dense:
      break; // no fields
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
  assert(fields.size() == lastField);
  return success();
}

/// Create allocation operation.
static Value createAllocation(OpBuilder &builder, Location loc, Type type,
                              Value sz) {
  auto memType = MemRefType::get({ShapedType::kDynamicSize}, type);
  return builder.create<memref::AllocOp>(loc, memType, sz);
}

/// Creates allocation for each field in sparse tensor type. Note that
/// for all dynamic memrefs, the memory size is really the capacity of
/// the "vector", while the actual size resides in the sizes array.
///
/// TODO: for efficiency, we will need heuristis to make educated guesses
///       on the required capacities
///
static void createAllocFields(OpBuilder &builder, Location loc, Type type,
                              ValueRange dynSizes,
                              SmallVectorImpl<Value> &fields) {
  auto enc = getSparseTensorEncoding(type);
  assert(enc);
  // Construct the basic types.
  unsigned idxWidth = enc.getIndexBitWidth();
  unsigned ptrWidth = enc.getPointerBitWidth();
  RankedTensorType rType = type.cast<RankedTensorType>();
  Type indexType = builder.getIndexType();
  Type idxType = idxWidth ? builder.getIntegerType(idxWidth) : indexType;
  Type ptrType = ptrWidth ? builder.getIntegerType(ptrWidth) : indexType;
  Type eltType = rType.getElementType();
  auto shape = rType.getShape();
  unsigned rank = shape.size();
  bool allDense = true;
  Value one = constantIndex(builder, loc, 1);
  Value linear = one;
  Value heuristic = one; // FIX, see TODO above
  // Build original sizes.
  SmallVector<Value, 8> sizes;
  for (unsigned r = 0, o = 0; r < rank; r++) {
    if (ShapedType::isDynamic(shape[r]))
      sizes.push_back(dynSizes[o++]);
    else
      sizes.push_back(constantIndex(builder, loc, shape[r]));
  }
  // The dimSizes array.
  Value dimSizes =
      builder.create<memref::AllocOp>(loc, MemRefType::get({rank}, indexType));
  fields.push_back(dimSizes);
  // The sizes array.
  unsigned lastField = getFieldIndex(type, -1, -1);
  Value memSizes = builder.create<memref::AllocOp>(
      loc, MemRefType::get({lastField - 2}, indexType));
  fields.push_back(memSizes);
  // Per-dimension storage.
  for (unsigned r = 0; r < rank; r++) {
    // Get the original dimension (ro) for the current stored dimension.
    unsigned ro = toOrig(enc, r);
    builder.create<memref::StoreOp>(loc, sizes[ro], dimSizes,
                                    constantIndex(builder, loc, r));
    linear = builder.create<arith::MulIOp>(loc, linear, sizes[ro]);
    // Allocate fiels.
    switch (enc.getDimLevelType()[r]) {
    case SparseTensorEncodingAttr::DimLevelType::Dense:
      break; // no fields
    case SparseTensorEncodingAttr::DimLevelType::Compressed:
    case SparseTensorEncodingAttr::DimLevelType::CompressedNu:
    case SparseTensorEncodingAttr::DimLevelType::CompressedNo:
    case SparseTensorEncodingAttr::DimLevelType::CompressedNuNo:
      fields.push_back(createAllocation(builder, loc, ptrType, heuristic));
      fields.push_back(createAllocation(builder, loc, idxType, heuristic));
      allDense = false;
      break;
    case SparseTensorEncodingAttr::DimLevelType::Singleton:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNu:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNo:
    case SparseTensorEncodingAttr::DimLevelType::SingletonNuNo:
      fields.push_back(createAllocation(builder, loc, idxType, heuristic));
      allDense = false;
      break;
    }
  }
  // The values array. For all-dense, the full length is required.
  // In all other case, we resort to the heuristical initial value.
  Value valuesSz = allDense ? linear : heuristic;
  fields.push_back(createAllocation(builder, loc, eltType, valuesSz));
  // Set memSizes.
  if (allDense)
    builder.create<memref::StoreOp>(
        loc, valuesSz, memSizes,
        constantIndex(builder, loc, 0)); // TODO: avoid memSizes in this case?
  else
    builder.create<linalg::FillOp>(
        loc, ValueRange{constantZero(builder, loc, indexType)},
        ValueRange{memSizes});
  assert(fields.size() == lastField);
}

/// Creates a straightforward counting for-loop.
static scf::ForOp createFor(OpBuilder &builder, Location loc, Value count) {
  Type indexType = builder.getIndexType();
  Value zero = constantZero(builder, loc, indexType);
  Value one = constantOne(builder, loc, indexType);
  scf::ForOp forOp = builder.create<scf::ForOp>(loc, zero, count, one);
  builder.setInsertionPointToStart(forOp.getBody());
  return forOp;
}

//===----------------------------------------------------------------------===//
// Codegen rules.
//===----------------------------------------------------------------------===//

/// Sparse tensor storage conversion rule for returns.
class SparseReturnConverter : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> flattened;
    flattenOperands(adaptor.getOperands(), flattened);
    // Create a return with the flattened value extracted from sparse tensors.
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, flattened);
    return success();
  }
};

/// Sparse tensor storage conversion rule for calls.
class SparseCallConverter : public OpConversionPattern<func::CallOp> {
public:
  // The default CallOp converter can not handle 1:N type conversion.
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // In case of:
    //  sparse_tensor, f, sparse_tensor = call @foo(...)
    // ==>
    //  memref..., f, memref = call @foo(...) replace with
    //  cast(memref...)->sparse_tensor, f, cast(memref...)->sparse_tensor
    SmallVector<Type, 8> finalRetTy;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), finalRetTy)))
      return failure();

    // (1) Genereates new call with flattened return value.
    SmallVector<Value, 8> flattened;
    flattenOperands(adaptor.getOperands(), flattened);
    auto newCall = rewriter.create<func::CallOp>(loc, op.getCallee(),
                                                 finalRetTy, flattened);
    // (2) Create cast operation for sparse tensor returns.
    SmallVector<Value, 4> castedRet;
    // Tracks the offset of current return value (of the orignal call)
    // relative to the new call (after sparse tensor flattening);
    unsigned retOffset = 0;
    // Temporal buffer to hold the flattened list of type for
    // a sparse tensor.
    SmallVector<Type, 8> sparseFlat;
    for (auto ret : op.getResults()) {
      assert(retOffset < newCall.getNumResults());
      auto retType = ret.getType();
      if (failed(typeConverter->convertType(retType, sparseFlat)))
        // This should never happen.
        llvm_unreachable("Failed to convert type in sparse tensor codegen");

      // Converted types can not be empty when the type conversion succeed.
      assert(!sparseFlat.empty());
      if (sparseFlat.size() > 1) {
        auto flatSize = sparseFlat.size();
        ValueRange sparseElem(iterator_range<ResultRange::iterator>(
            newCall.result_begin() + retOffset,
            newCall.result_begin() + retOffset + flatSize));
        auto castOp = rewriter.create<UnrealizedConversionCastOp>(
            loc, TypeRange({retType}), sparseElem);
        castedRet.push_back(castOp.getResult(0));
        retOffset += flatSize;
      } else {
        // If this is an 1:1 conversion, no need for casting.
        castedRet.push_back(newCall.getResult(retOffset));
        retOffset++;
      }
      sparseFlat.clear();
    }

    assert(castedRet.size() == op.getNumResults());
    rewriter.replaceOp(op, castedRet);
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
    Optional<int64_t> index = op.getConstantIndex();
    if (!index)
      return failure();
    auto sz =
        sizeFromTensorAtDim(rewriter, op.getLoc(),
                            op.getSource().getType().cast<RankedTensorType>(),
                            adaptor.getSource(), *index);
    if (!sz)
      return failure();

    rewriter.replaceOp(op, *sz);
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

/// Sparse codgen rule for the alloc operator.
class SparseTensorAllocConverter
    : public OpConversionPattern<bufferization::AllocTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(bufferization::AllocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resType = op.getType();
    auto enc = getSparseTensorEncoding(resType);
    if (!enc)
      return failure();
    if (op.getCopy())
      return rewriter.notifyMatchFailure(op, "tensor copy not implemented");

    // Construct allocation for each field.
    Location loc = op.getLoc();
    SmallVector<Value, 8> fields;
    createAllocFields(rewriter, loc, resType, adaptor.getOperands(), fields);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, TypeRange{resType}, fields);
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

    // Replace the sparse tensor deallocation with field deallocations.
    Location loc = op.getLoc();
    auto tuple = llvm::cast<UnrealizedConversionCastOp>(
        adaptor.getTensor().getDefiningOp());
    for (auto input : tuple.getInputs())
      // Deallocate every buffer used to store the sparse tensor handler.
      rewriter.create<memref::DeallocOp>(loc, input);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Sparse codegen rule for tensor rematerialization.
class SparseTensorLoadConverter : public OpConversionPattern<LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getHasInserts()) {
      // Finalize any pending insertions.
      // TODO: implement
    }
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Sparse codegen rule for the expand op.
class SparseExpandConverter : public OpConversionPattern<ExpandOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ExpandOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    ShapedType srcType = op.getTensor().getType().cast<ShapedType>();
    Type eltType = srcType.getElementType();
    Type boolType = rewriter.getIntegerType(1);
    Type idxType = rewriter.getIndexType();
    // All initialization should be done on entry of the loop nest.
    rewriter.setInsertionPointAfter(op.getTensor().getDefiningOp());
    // Determine the size for access expansion (always the innermost stored
    // dimension size, translated back to original dimension). Note that we
    // recursively rewrite the new DimOp on the **original** tensor.
    auto enc = getSparseTensorEncoding(srcType);
    unsigned innerDim = srcType.getRank() - 1;
    if (AffineMap p = enc.getDimOrdering())
      innerDim = p.getDimPosition(innerDim);
    auto sz = sizeFromTensorAtDim(rewriter, loc, srcType, adaptor.getTensor(),
                                  innerDim);
    assert(sz); // This for sure is a sparse tensor
    // Generate a memref for `sz` elements of type `t`.
    auto genAlloc = [&](Type t) {
      auto memTp = MemRefType::get({ShapedType::kDynamicSize}, t);
      return rewriter.create<memref::AllocOp>(loc, memTp, ValueRange{*sz});
    };
    // Allocate temporary buffers for values/filled-switch and added.
    // We do not use stack buffers for this, since the expanded size may
    // be rather large (as it envelops a single expanded dense dimension).
    Value values = genAlloc(eltType);
    Value filled = genAlloc(boolType);
    Value added = genAlloc(idxType);
    Value zero = constantZero(rewriter, loc, idxType);
    // Reset the values/filled-switch to all-zero/false. Note that this
    // introduces an O(N) operation into the computation, but this reset
    // operation is amortized over the innermost loops for the access
    // pattern expansion. As noted in the operation doc, we would like
    // to amortize this setup cost even between kernels.
    rewriter.create<linalg::FillOp>(
        loc, ValueRange{constantZero(rewriter, loc, eltType)},
        ValueRange{values});
    rewriter.create<linalg::FillOp>(
        loc, ValueRange{constantZero(rewriter, loc, boolType)},
        ValueRange{filled});
    // Replace expansion op with these buffers and initial index.
    assert(op.getNumResults() == 4);
    rewriter.replaceOp(op, {values, filled, added, zero});
    return success();
  }
};

/// Sparse codegen rule for the compress operator.
class SparseCompressConverter : public OpConversionPattern<CompressOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    ShapedType srcType = op.getTensor().getType().cast<ShapedType>();
    Type eltType = srcType.getElementType();
    Value values = adaptor.getValues();
    Value filled = adaptor.getFilled();
    Value added = adaptor.getAdded();
    Value count = adaptor.getCount();

    //
    // TODO: need to implement "std::sort(added, added + count);" for ordered
    //

    // While performing the insertions, we also need to reset the elements
    // of the values/filled-switch by only iterating over the set elements,
    // to ensure that the runtime complexity remains proportional to the
    // sparsity of the expanded access pattern.
    //
    // Generate
    //    for (i = 0; i < count; i++) {
    //      index = added[i];
    //      value = values[index];
    //
    //      TODO: insert prev_indices, index, value
    //
    //      values[index] = 0;
    //      filled[index] = false;
    //    }
    Value i = createFor(rewriter, loc, count).getInductionVar();
    Value index = rewriter.create<memref::LoadOp>(loc, added, i);
    rewriter.create<memref::LoadOp>(loc, values, index);
    // TODO: insert
    rewriter.create<memref::StoreOp>(loc, constantZero(rewriter, loc, eltType),
                                     values, index);
    rewriter.create<memref::StoreOp>(loc, constantI1(rewriter, loc, false),
                                     filled, index);

    // Deallocate the buffers on exit of the full loop nest.
    Operation *parent = op;
    for (; isa<scf::ForOp>(parent->getParentOp()) ||
           isa<scf::WhileOp>(parent->getParentOp()) ||
           isa<scf::ParallelOp>(parent->getParentOp()) ||
           isa<scf::IfOp>(parent->getParentOp());
         parent = parent->getParentOp())
      ;
    rewriter.setInsertionPointAfter(parent);
    rewriter.create<memref::DeallocOp>(loc, values);
    rewriter.create<memref::DeallocOp>(loc, filled);
    rewriter.create<memref::DeallocOp>(loc, added);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Sparse codegen rule for the push_back operator.
class SparsePushBackConverter : public OpConversionPattern<PushBackOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PushBackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lower push_back(buffer, value) to:
    // if (size(buffer) >= capacity(buffer))
    //    new_capacity = capacity(buffer)*2
    //    new_buffer = realloc(buffer, new_capacity)
    // buffer = new_buffer
    // store(buffer, value)
    // size(buffer)++
    Location loc = op->getLoc();
    Value c0 = constantIndex(rewriter, loc, 0);
    Value buffer = adaptor.getInBuffer();
    Value capacity = rewriter.create<memref::DimOp>(loc, buffer, c0);
    Value idx = constantIndex(rewriter, loc, op.getIdx().getZExtValue());
    Value bufferSizes = adaptor.getBufferSizes();
    Value size = rewriter.create<memref::LoadOp>(loc, bufferSizes, idx);
    Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                                size, capacity);
    Value value = adaptor.getValue();
    auto bufferType =
        MemRefType::get({ShapedType::kDynamicSize}, value.getType());
    scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, bufferType, cond,
                                                /*else=*/true);
    // True branch.
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    Value c2 = constantIndex(rewriter, loc, 2);
    capacity = rewriter.create<arith::MulIOp>(loc, capacity, c2);
    Value newBuffer =
        rewriter.create<memref::ReallocOp>(loc, bufferType, buffer, capacity);
    rewriter.create<scf::YieldOp>(loc, newBuffer);

    // False branch.
    rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
    rewriter.create<scf::YieldOp>(loc, buffer);

    // Add the value to the end of the buffer.
    rewriter.setInsertionPointAfter(ifOp);
    buffer = ifOp.getResult(0);
    rewriter.create<memref::StoreOp>(loc, value, buffer, size);

    // Increment the size of the buffer by 1.
    Value c1 = constantIndex(rewriter, loc, 1);
    size = rewriter.create<arith::AddIOp>(loc, size, c1);
    rewriter.create<memref::StoreOp>(loc, size, bufferSizes, idx);

    rewriter.replaceOp(op, buffer);
    return success();
  }
};

/// Base class for getter-like operations, e.g., to_indices, to_pointers.
template <typename SourceOp, typename Base>
class SparseGetterOpConverter : public OpConversionPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the requested pointer access with corresponding field.
    // The cast_op is inserted by type converter to intermix 1:N type
    // conversion.
    auto tuple = llvm::cast<UnrealizedConversionCastOp>(
        adaptor.getTensor().getDefiningOp());
    unsigned idx = Base::getIndexForOp(tuple, op);
    auto fields = tuple.getInputs();
    assert(idx < fields.size());
    rewriter.replaceOp(op, fields[idx]);
    return success();
  }
};

/// Sparse codegen rule for pointer accesses.
class SparseToPointersConverter
    : public SparseGetterOpConverter<ToPointersOp, SparseToPointersConverter> {
public:
  using SparseGetterOpConverter::SparseGetterOpConverter;
  // Callback for SparseGetterOpConverter.
  static unsigned getIndexForOp(UnrealizedConversionCastOp /*tuple*/,
                                ToPointersOp op) {
    uint64_t dim = op.getDimension().getZExtValue();
    return getFieldIndex(op.getTensor().getType(), /*ptrDim=*/dim, -1);
  }
};

/// Sparse codegen rule for index accesses.
class SparseToIndicesConverter
    : public SparseGetterOpConverter<ToIndicesOp, SparseToIndicesConverter> {
public:
  using SparseGetterOpConverter::SparseGetterOpConverter;
  // Callback for SparseGetterOpConverter.
  static unsigned getIndexForOp(UnrealizedConversionCastOp /*tuple*/,
                                ToIndicesOp op) {
    uint64_t dim = op.getDimension().getZExtValue();
    return getFieldIndex(op.getTensor().getType(), -1, /*idxDim=*/dim);
  }
};

/// Sparse codegen rule for value accesses.
class SparseToValuesConverter
    : public SparseGetterOpConverter<ToValuesOp, SparseToValuesConverter> {
public:
  using SparseGetterOpConverter::SparseGetterOpConverter;
  // Callback for SparseGetterOpConverter.
  static unsigned getIndexForOp(UnrealizedConversionCastOp tuple,
                                ToValuesOp /*op*/) {
    // The last field holds the value buffer.
    return tuple.getInputs().size() - 1;
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
  patterns.add<SparseReturnConverter, SparseCallConverter, SparseDimOpConverter,
               SparseCastConverter, SparseTensorAllocConverter,
               SparseTensorDeallocConverter, SparseTensorLoadConverter,
               SparseExpandConverter, SparseCompressConverter,
               SparsePushBackConverter, SparseToPointersConverter,
               SparseToIndicesConverter, SparseToValuesConverter>(
      typeConverter, patterns.getContext());
}
