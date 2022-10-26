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

static constexpr uint64_t DimSizesIdx = 0;
static constexpr uint64_t DimCursorIdx = 1;
static constexpr uint64_t MemSizesIdx = 2;
static constexpr uint64_t FieldsIdx = 3;

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// Returns the "tuple" value of the adapted tensor.
static UnrealizedConversionCastOp getTuple(Value tensor) {
  return llvm::cast<UnrealizedConversionCastOp>(tensor.getDefiningOp());
}

/// Packs the given values as a "tuple" value.
static Value genTuple(OpBuilder &rewriter, Location loc, Type tp,
                      ValueRange values) {
  return rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange(tp), values)
      .getResult(0);
}

/// Flatten a list of operands that may contain sparse tensors.
static void flattenOperands(ValueRange operands,
                            SmallVectorImpl<Value> &flattened) {
  // In case of
  // sparse_tensor, c, sparse_tensor
  // ==>
  // memref ..., c, memref ...
  for (auto operand : operands) {
    if (auto tuple = getTuple(operand);
        tuple && getSparseTensorEncoding(tuple->getResultTypes()[0]))
      // An unrealized_conversion_cast will be inserted by type converter to
      // inter-mix the gap between 1:N conversion between sparse tensors and
      // fields. In this case, take the operands in the cast and replace the
      // sparse tensor output with the flattened type array.
      flattened.append(tuple.getOperands().begin(), tuple.getOperands().end());
    else
      flattened.push_back(operand);
  }
}

/// Gets the dimension size for the given sparse tensor at the given dim.
/// Returns None if no sparse encoding is attached to the tensor type.
static Optional<Value> sizeFromTensorAtDim(OpBuilder &rewriter, Location loc,
                                           RankedTensorType tensorTp,
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
  auto tuple = getTuple(adaptedValue);
  Value idx = constantIndex(rewriter, loc, toStoredDim(tensorTp, dim));
  return rewriter.create<memref::LoadOp>(loc, tuple.getInputs().front(), idx)
      .getResult();
}

/// Translates field index to memSizes index.
static unsigned getMemSizesIndex(unsigned field) {
  assert(FieldsIdx <= field);
  return field - FieldsIdx;
}

/// Returns field index of sparse tensor type for pointers/indices, when set.
static unsigned getFieldIndex(Type type, unsigned ptrDim, unsigned idxDim) {
  assert(getSparseTensorEncoding(type));
  RankedTensorType rType = type.cast<RankedTensorType>();
  unsigned field = FieldsIdx; // start past header
  unsigned ptr = 0;
  unsigned idx = 0;
  for (unsigned r = 0, rank = rType.getShape().size(); r < rank; r++) {
    if (isCompressedDim(rType, r)) {
      if (ptr++ == ptrDim)
        return field;
      field++;
      if (idx++ == idxDim)
        return field;
      field++;
    } else if (isSingletonDim(rType, r)) {
      if (idx++ == idxDim)
        return field;
      field++;
    } else {
      assert(isDenseDim(rType, r)); // no fields
    }
  }
  assert(ptrDim == -1u && idxDim == -1u);
  return field + 1; // return values field index
}

/// Maps a sparse tensor type to the appropriate compounded buffers.
static Optional<LogicalResult>
convertSparseTensorType(Type type, SmallVectorImpl<Type> &fields) {
  auto enc = getSparseTensorEncoding(type);
  if (!enc)
    return llvm::None;
  // Construct the basic types.
  auto *context = type.getContext();
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
  //   memref<rank x index> dimCursor    ; cursor in each dimension
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
  unsigned lastField = getFieldIndex(type, -1u, -1u);
  // The dimSizes array, dimCursor array, and memSizes array.
  fields.push_back(MemRefType::get({rank}, indexType));
  fields.push_back(MemRefType::get({rank}, indexType));
  fields.push_back(MemRefType::get({getMemSizesIndex(lastField)}, indexType));
  // Per-dimension storage.
  for (unsigned r = 0; r < rank; r++) {
    // Dimension level types apply in order to the reordered dimension.
    // As a result, the compound type can be constructed directly in the given
    // order. Clients of this type know what field is what from the sparse
    // tensor type.
    if (isCompressedDim(rType, r)) {
      fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, ptrType));
      fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, idxType));
    } else if (isSingletonDim(rType, r)) {
      fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, idxType));
    } else {
      assert(isDenseDim(rType, r)); // no fields
    }
  }
  // The values array.
  fields.push_back(MemRefType::get({ShapedType::kDynamicSize}, eltType));
  assert(fields.size() == lastField);
  return success();
}

/// Creates allocation operation.
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
  // The dimSizes array, dimCursor array, and memSizes array.
  unsigned lastField = getFieldIndex(type, -1u, -1u);
  Value dimSizes =
      builder.create<memref::AllocOp>(loc, MemRefType::get({rank}, indexType));
  Value dimCursor =
      builder.create<memref::AllocOp>(loc, MemRefType::get({rank}, indexType));
  Value memSizes = builder.create<memref::AllocOp>(
      loc, MemRefType::get({getMemSizesIndex(lastField)}, indexType));
  fields.push_back(dimSizes);
  fields.push_back(dimCursor);
  fields.push_back(memSizes);
  // Per-dimension storage.
  for (unsigned r = 0; r < rank; r++) {
    // Get the original dimension (ro) for the current stored dimension.
    unsigned ro = toOrigDim(rType, r);
    builder.create<memref::StoreOp>(loc, sizes[ro], dimSizes,
                                    constantIndex(builder, loc, r));
    linear = builder.create<arith::MulIOp>(loc, linear, sizes[ro]);
    // Allocate fields.
    if (isCompressedDim(rType, r)) {
      fields.push_back(createAllocation(builder, loc, ptrType, heuristic));
      fields.push_back(createAllocation(builder, loc, idxType, heuristic));
      allDense = false;
    } else if (isSingletonDim(rType, r)) {
      fields.push_back(createAllocation(builder, loc, idxType, heuristic));
      allDense = false;
    } else {
      assert(isDenseDim(rType, r)); // no fields
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
static scf::ForOp createFor(OpBuilder &builder, Location loc, Value count,
                            SmallVectorImpl<Value> &fields) {
  Type indexType = builder.getIndexType();
  Value zero = constantZero(builder, loc, indexType);
  Value one = constantOne(builder, loc, indexType);
  scf::ForOp forOp = builder.create<scf::ForOp>(loc, zero, count, one, fields);
  for (unsigned i = 0, e = fields.size(); i < e; i++)
    fields[i] = forOp.getRegionIterArg(i);
  builder.setInsertionPointToStart(forOp.getBody());
  return forOp;
}

/// Creates a pushback op for given field and updates the fields array
/// accordingly.
static void createPushback(OpBuilder &builder, Location loc,
                           SmallVectorImpl<Value> &fields, unsigned field,
                           Value value) {
  assert(FieldsIdx <= field && field < fields.size());
  Type etp = fields[field].getType().cast<ShapedType>().getElementType();
  if (value.getType() != etp)
    value = builder.create<arith::IndexCastOp>(loc, etp, value);
  fields[field] = builder.create<PushBackOp>(
      loc, fields[field].getType(), fields[MemSizesIdx], fields[field], value,
      APInt(64, getMemSizesIndex(field)));
}

/// Generates insertion code.
//
// TODO: generalize this for any rank and format currently it is just sparse
//       vectors as a proof of concept that we have everything in place!
//
static void genInsert(OpBuilder &builder, Location loc, RankedTensorType rtp,
                      SmallVectorImpl<Value> &fields,
                      SmallVectorImpl<Value> &indices, Value value) {
  unsigned rank = indices.size();
  assert(rtp.getShape().size() == rank);
  if (rank != 1 || !isCompressedDim(rtp, 0) || !isUniqueDim(rtp, 0) ||
      !isOrderedDim(rtp, 0))
    return; // TODO: add codegen
  // push_back memSizes indices-0 index
  // push_back memSizes values    value
  createPushback(builder, loc, fields, FieldsIdx + 1, indices[0]);
  createPushback(builder, loc, fields, FieldsIdx + 2, value);
}

/// Generations insertion finalization code.
//
// TODO: this too only works for the very simple case
//
static void genEndInsert(OpBuilder &builder, Location loc, RankedTensorType rtp,
                         SmallVectorImpl<Value> &fields) {
  if (rtp.getShape().size() != 1 || !isCompressedDim(rtp, 0) ||
      !isUniqueDim(rtp, 0) || !isOrderedDim(rtp, 0))
    return; // TODO: add codegen
  // push_back memSizes pointers-0 0
  // push_back memSizes pointers-0 memSizes[2]
  Value zero = constantIndex(builder, loc, 0);
  Value two = constantIndex(builder, loc, 2);
  Value size = builder.create<memref::LoadOp>(loc, fields[MemSizesIdx], two);
  createPushback(builder, loc, fields, FieldsIdx, zero);
  createPushback(builder, loc, fields, FieldsIdx, size);
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
        ValueRange fields(iterator_range<ResultRange::iterator>(
            newCall.result_begin() + retOffset,
            newCall.result_begin() + retOffset + flatSize));
        castedRet.push_back(genTuple(rewriter, loc, retType, fields));
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
    // Replace operation with resulting memrefs.
    rewriter.replaceOp(op, genTuple(rewriter, loc, resType, fields));
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
    auto tuple = getTuple(adaptor.getTensor());
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
    RankedTensorType srcType =
        op.getTensor().getType().cast<RankedTensorType>();
    auto tuple = getTuple(adaptor.getTensor());
    // Prepare fields.
    SmallVector<Value, 8> fields(tuple.getInputs());
    // Generate optional insertion finalization code.
    if (op.getHasInserts())
      genEndInsert(rewriter, op.getLoc(), srcType, fields);
    // Replace operation with resulting memrefs.
    rewriter.replaceOp(op, genTuple(rewriter, op.getLoc(), srcType, fields));
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
    RankedTensorType srcType =
        op.getTensor().getType().cast<RankedTensorType>();
    Type eltType = srcType.getElementType();
    Type boolType = rewriter.getIntegerType(1);
    Type idxType = rewriter.getIndexType();
    // All initialization should be done on entry of the loop nest.
    rewriter.setInsertionPointAfter(op.getTensor().getDefiningOp());
    // Determine the size for access expansion (always the innermost stored
    // dimension size, translated back to original dimension). Note that we
    // recursively rewrite the new DimOp on the **original** tensor.
    unsigned innerDim = toOrigDim(srcType, srcType.getRank() - 1);
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
    RankedTensorType dstType =
        op.getTensor().getType().cast<RankedTensorType>();
    Type eltType = dstType.getElementType();
    auto tuple = getTuple(adaptor.getTensor());
    Value values = adaptor.getValues();
    Value filled = adaptor.getFilled();
    Value added = adaptor.getAdded();
    Value count = adaptor.getCount();
    // Prepare fields and indices.
    SmallVector<Value, 8> fields(tuple.getInputs());
    SmallVector<Value, 8> indices(adaptor.getIndices());
    // If the innermost dimension is ordered, we need to sort the indices
    // in the "added" array prior to applying the compression.
    unsigned rank = dstType.getShape().size();
    if (isOrderedDim(dstType, rank - 1))
      rewriter.create<SortOp>(loc, count, ValueRange{added}, ValueRange{});
    // While performing the insertions, we also need to reset the elements
    // of the values/filled-switch by only iterating over the set elements,
    // to ensure that the runtime complexity remains proportional to the
    // sparsity of the expanded access pattern.
    //
    // Generate
    //    out_memrefs = for (i = 0; i < count; i++)(in_memrefs) {
    //      index = added[i];
    //      value = values[index];
    //      insert({prev_indices, index}, value);
    //      new_memrefs = insert(in_memrefs, {prev_indices, index}, value);
    //      values[index] = 0;
    //      filled[index] = false;
    //      yield new_memrefs
    //    }
    scf::ForOp loop = createFor(rewriter, loc, count, fields);
    Value i = loop.getInductionVar();
    Value index = rewriter.create<memref::LoadOp>(loc, added, i);
    Value value = rewriter.create<memref::LoadOp>(loc, values, index);
    indices.push_back(index);
    genInsert(rewriter, loc, dstType, fields, indices, value);
    rewriter.create<memref::StoreOp>(loc, constantZero(rewriter, loc, eltType),
                                     values, index);
    rewriter.create<memref::StoreOp>(loc, constantI1(rewriter, loc, false),
                                     filled, index);
    rewriter.create<scf::YieldOp>(loc, fields);
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
    // Replace operation with resulting memrefs.
    rewriter.replaceOp(op,
                       genTuple(rewriter, loc, dstType, loop->getResults()));
    return success();
  }
};

/// Sparse codegen rule for the insert operator.
class SparseInsertConverter : public OpConversionPattern<InsertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType dstType =
        op.getTensor().getType().cast<RankedTensorType>();
    auto tuple = getTuple(adaptor.getTensor());
    // Prepare fields and indices.
    SmallVector<Value, 8> fields(tuple.getInputs());
    SmallVector<Value, 8> indices(adaptor.getIndices());
    // Generate insertion.
    Value value = adaptor.getValue();
    genInsert(rewriter, op->getLoc(), dstType, fields, indices, value);
    // Replace operation with resulting memrefs.
    rewriter.replaceOp(op, genTuple(rewriter, op.getLoc(), dstType, fields));
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
    auto tuple = getTuple(adaptor.getTensor());
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
    return getFieldIndex(op.getTensor().getType(), /*ptrDim=*/dim, -1u);
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
    return getFieldIndex(op.getTensor().getType(), -1u, /*idxDim=*/dim);
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

/// Sparse codegen rule for the convert operator.
class SparseConvertConverter : public OpConversionPattern<ConvertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SparseTensorEncodingAttr encDst = getSparseTensorEncoding(op.getType());
    SparseTensorEncodingAttr encSrc =
        getSparseTensorEncoding(op.getSource().getType());
    if (encDst != encSrc) {
      // This should be handled by rewriting before codegen.
      return failure();
    }
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

/// Sparse codegen rule for number of entries operator.
class SparseNumberOfEntriesConverter
    : public OpConversionPattern<NumberOfEntriesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NumberOfEntriesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Query memSizes for the actually stored values size.
    auto tuple = getTuple(adaptor.getTensor());
    auto fields = tuple.getInputs();
    unsigned lastField = fields.size() - 1;
    Value field =
        constantIndex(rewriter, op.getLoc(), getMemSizesIndex(lastField));
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, fields[MemSizesIdx], field);
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

  // Required by scf.for 1:N type conversion.
  addSourceMaterialization([](OpBuilder &builder, RankedTensorType tp,
                              ValueRange inputs,
                              Location loc) -> Optional<Value> {
    if (!getSparseTensorEncoding(tp))
      // Not a sparse tensor.
      return llvm::None;
    // Sparse compiler knows how to cancel out these casts.
    return genTuple(builder, loc, tp, inputs);
  });
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
               SparseInsertConverter, SparseToPointersConverter,
               SparseToIndicesConverter, SparseToValuesConverter,
               SparseConvertConverter, SparseNumberOfEntriesConverter>(
      typeConverter, patterns.getContext());
}
