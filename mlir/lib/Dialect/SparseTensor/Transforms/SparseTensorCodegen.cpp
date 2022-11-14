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

static constexpr uint64_t dimSizesIdx = 0;
static constexpr uint64_t memSizesIdx = 1;
static constexpr uint64_t fieldsIdx = 2;

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// Returns the "tuple" value of the adapted tensor.
static UnrealizedConversionCastOp getTuple(Value tensor) {
  return llvm::cast<UnrealizedConversionCastOp>(tensor.getDefiningOp());
}

/// Packs the given values as a "tuple" value.
static Value genTuple(OpBuilder &builder, Location loc, Type tp,
                      ValueRange values) {
  return builder.create<UnrealizedConversionCastOp>(loc, TypeRange(tp), values)
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
    if (getSparseTensorEncoding(operand.getType())) {
      auto tuple = getTuple(operand);
      // An unrealized_conversion_cast will be inserted by type converter to
      // inter-mix the gap between 1:N conversion between sparse tensors and
      // fields. In this case, take the operands in the cast and replace the
      // sparse tensor output with the flattened type array.
      flattened.append(tuple.getOperands().begin(), tuple.getOperands().end());
    } else {
      flattened.push_back(operand);
    }
  }
}

/// Adds index conversions where needed.
static Value toType(OpBuilder &builder, Location loc, Value value, Type tp) {
  if (value.getType() != tp)
    return builder.create<arith::IndexCastOp>(loc, tp, value);
  return value;
}

/// Generates a load with proper index typing.
static Value genLoad(OpBuilder &builder, Location loc, Value mem, Value idx) {
  idx = toType(builder, loc, idx, builder.getIndexType());
  return builder.create<memref::LoadOp>(loc, mem, idx);
}

/// Generates a store with proper index typing and (for indices) proper value.
static void genStore(OpBuilder &builder, Location loc, Value val, Value mem,
                     Value idx) {
  idx = toType(builder, loc, idx, builder.getIndexType());
  val = toType(builder, loc, val,
               mem.getType().cast<ShapedType>().getElementType());
  builder.create<memref::StoreOp>(loc, val, mem, idx);
}

/// Creates a straightforward counting for-loop.
static scf::ForOp createFor(OpBuilder &builder, Location loc, Value upper,
                            SmallVectorImpl<Value> &fields,
                            Value lower = Value()) {
  Type indexType = builder.getIndexType();
  if (!lower)
    lower = constantZero(builder, loc, indexType);
  Value one = constantOne(builder, loc, indexType);
  scf::ForOp forOp = builder.create<scf::ForOp>(loc, lower, upper, one, fields);
  for (unsigned i = 0, e = fields.size(); i < e; i++)
    fields[i] = forOp.getRegionIterArg(i);
  builder.setInsertionPointToStart(forOp.getBody());
  return forOp;
}

/// Gets the dimension size for the given sparse tensor at the given
/// original dimension 'dim'. Returns None if no sparse encoding is
/// attached to the given tensor type.
static Optional<Value> sizeFromTensorAtDim(OpBuilder &builder, Location loc,
                                           RankedTensorType tensorTp,
                                           Value adaptedValue, unsigned dim) {
  auto enc = getSparseTensorEncoding(tensorTp);
  if (!enc)
    return llvm::None;

  // Access into static dimension can query original type directly.
  // Note that this is typically already done by DimOp's folding.
  auto shape = tensorTp.getShape();
  if (!ShapedType::isDynamic(shape[dim]))
    return constantIndex(builder, loc, shape[dim]);

  // Any other query can consult the dimSizes array at field DimSizesIdx,
  // accounting for the reordering applied to the sparse storage.
  auto tuple = getTuple(adaptedValue);
  Value idx = constantIndex(builder, loc, toStoredDim(tensorTp, dim));
  return builder
      .create<memref::LoadOp>(loc, tuple.getInputs()[dimSizesIdx], idx)
      .getResult();
}

// Gets the dimension size at the given stored dimension 'd', either as a
// constant for a static size, or otherwise dynamically through memSizes.
Value sizeAtStoredDim(OpBuilder &builder, Location loc, RankedTensorType rtp,
                      SmallVectorImpl<Value> &fields, unsigned d) {
  unsigned dim = toOrigDim(rtp, d);
  auto shape = rtp.getShape();
  if (!ShapedType::isDynamic(shape[dim]))
    return constantIndex(builder, loc, shape[dim]);
  return genLoad(builder, loc, fields[dimSizesIdx],
                 constantIndex(builder, loc, d));
}

/// Translates field index to memSizes index.
static unsigned getMemSizesIndex(unsigned field) {
  assert(fieldsIdx <= field);
  return field - fieldsIdx;
}

/// Creates a pushback op for given field and updates the fields array
/// accordingly. This operation also updates the memSizes contents.
static void createPushback(OpBuilder &builder, Location loc,
                           SmallVectorImpl<Value> &fields, unsigned field,
                           Value value, Value repeat = Value()) {
  assert(fieldsIdx <= field && field < fields.size());
  Type etp = fields[field].getType().cast<ShapedType>().getElementType();
  fields[field] = builder.create<PushBackOp>(
      loc, fields[field].getType(), fields[memSizesIdx], fields[field],
      toType(builder, loc, value, etp), APInt(64, getMemSizesIndex(field)),
      repeat);
}

/// Returns field index of sparse tensor type for pointers/indices, when set.
static unsigned getFieldIndex(Type type, unsigned ptrDim, unsigned idxDim) {
  assert(getSparseTensorEncoding(type));
  RankedTensorType rType = type.cast<RankedTensorType>();
  unsigned field = fieldsIdx; // start past header
  for (unsigned r = 0, rank = rType.getShape().size(); r < rank; r++) {
    if (isCompressedDim(rType, r)) {
      if (r == ptrDim)
        return field;
      field++;
      if (r == idxDim)
        return field;
      field++;
    } else if (isSingletonDim(rType, r)) {
      if (r == idxDim)
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
  // Sparse tensor storage scheme for rank-dimensional tensor is organized
  // as a single compound type with the following fields. Note that every
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
  unsigned lastField = getFieldIndex(type, -1u, -1u);
  // The dimSizes array and memSizes array.
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

/// Generates code that allocates a sparse storage scheme for given rank.
static void allocSchemeForRank(OpBuilder &builder, Location loc,
                               RankedTensorType rtp,
                               SmallVectorImpl<Value> &fields, unsigned field,
                               unsigned r0) {
  unsigned rank = rtp.getShape().size();
  Value linear = constantIndex(builder, loc, 1);
  for (unsigned r = r0; r < rank; r++) {
    if (isCompressedDim(rtp, r)) {
      // Append linear x pointers, initialized to zero. Since each compressed
      // dimension initially already has a single zero entry, this maintains
      // the desired "linear + 1" length property at all times.
      unsigned ptrWidth = getSparseTensorEncoding(rtp).getPointerBitWidth();
      Type indexType = builder.getIndexType();
      Type ptrType = ptrWidth ? builder.getIntegerType(ptrWidth) : indexType;
      Value ptrZero = constantZero(builder, loc, ptrType);
      createPushback(builder, loc, fields, field, ptrZero, linear);
      return;
    }
    if (isSingletonDim(rtp, r)) {
      return; // nothing to do
    } else {
      // Keep compounding the size, but nothing needs to be initialized
      // at this level. We will eventually reach a compressed level or
      // otherwise the values array for the from-here "all-dense" case.
      assert(isDenseDim(rtp, r));
      Value size = sizeAtStoredDim(builder, loc, rtp, fields, r);
      linear = builder.create<arith::MulIOp>(loc, linear, size);
    }
  }
  // Reached values array so prepare for an insertion.
  Value valZero = constantZero(builder, loc, rtp.getElementType());
  createPushback(builder, loc, fields, field, valZero, linear);
  assert(fields.size() == ++field);
}

/// Creates allocation operation.
static Value createAllocation(OpBuilder &builder, Location loc, Type type,
                              Value sz, bool enableInit) {
  auto memType = MemRefType::get({ShapedType::kDynamicSize}, type);
  Value buffer = builder.create<memref::AllocOp>(loc, memType, sz);
  if (enableInit) {
    Value fillValue =
        builder.create<arith::ConstantOp>(loc, type, builder.getZeroAttr(type));
    builder.create<linalg::FillOp>(loc, fillValue, buffer);
  }
  return buffer;
}

/// Creates allocation for each field in sparse tensor type. Note that
/// for all dynamic memrefs, the memory size is really the capacity of
/// the "vector", while the actual size resides in the sizes array.
///
/// TODO: for efficiency, we will need heuristis to make educated guesses
///       on the required capacities (see heuristic variable).
///
static void createAllocFields(OpBuilder &builder, Location loc, Type type,
                              ValueRange dynSizes, bool enableInit,
                              SmallVectorImpl<Value> &fields) {
  auto enc = getSparseTensorEncoding(type);
  assert(enc);
  // Construct the basic types.
  unsigned idxWidth = enc.getIndexBitWidth();
  unsigned ptrWidth = enc.getPointerBitWidth();
  RankedTensorType rtp = type.cast<RankedTensorType>();
  Type indexType = builder.getIndexType();
  Type idxType = idxWidth ? builder.getIntegerType(idxWidth) : indexType;
  Type ptrType = ptrWidth ? builder.getIntegerType(ptrWidth) : indexType;
  Type eltType = rtp.getElementType();
  auto shape = rtp.getShape();
  unsigned rank = shape.size();
  Value heuristic = constantIndex(builder, loc, 16);
  // Build original sizes.
  SmallVector<Value, 8> sizes;
  for (unsigned r = 0, o = 0; r < rank; r++) {
    if (ShapedType::isDynamic(shape[r]))
      sizes.push_back(dynSizes[o++]);
    else
      sizes.push_back(constantIndex(builder, loc, shape[r]));
  }
  // The dimSizes array and memSizes array.
  unsigned lastField = getFieldIndex(type, -1u, -1u);
  Value dimSizes =
      builder.create<memref::AllocOp>(loc, MemRefType::get({rank}, indexType));
  Value memSizes = builder.create<memref::AllocOp>(
      loc, MemRefType::get({getMemSizesIndex(lastField)}, indexType));
  fields.push_back(dimSizes);
  fields.push_back(memSizes);
  // Per-dimension storage.
  for (unsigned r = 0; r < rank; r++) {
    if (isCompressedDim(rtp, r)) {
      fields.push_back(
          createAllocation(builder, loc, ptrType, heuristic, enableInit));
      fields.push_back(
          createAllocation(builder, loc, idxType, heuristic, enableInit));
    } else if (isSingletonDim(rtp, r)) {
      fields.push_back(
          createAllocation(builder, loc, idxType, heuristic, enableInit));
    } else {
      assert(isDenseDim(rtp, r)); // no fields
    }
  }
  // The values array.
  fields.push_back(
      createAllocation(builder, loc, eltType, heuristic, enableInit));
  assert(fields.size() == lastField);
  // Initialize the storage scheme to an empty tensor. Initialized memSizes
  // to all zeros, sets the dimSizes to known values and gives all pointer
  // fields an initial zero entry, so that it is easier to maintain the
  // "linear + 1" length property.
  builder.create<linalg::FillOp>(
      loc, ValueRange{constantZero(builder, loc, indexType)},
      ValueRange{memSizes}); // zero memSizes
  Value ptrZero = constantZero(builder, loc, ptrType);
  for (unsigned r = 0, field = fieldsIdx; r < rank; r++) {
    unsigned ro = toOrigDim(rtp, r);
    genStore(builder, loc, sizes[ro], dimSizes, constantIndex(builder, loc, r));
    if (isCompressedDim(rtp, r)) {
      createPushback(builder, loc, fields, field, ptrZero);
      field += 2;
    } else if (isSingletonDim(rtp, r)) {
      field += 1;
    }
  }
  allocSchemeForRank(builder, loc, rtp, fields, fieldsIdx, /*rank=*/0);
}

/// Helper method that generates block specific to compressed case:
///
///  plo = pointers[d][pos[d-1]]
///  phi = pointers[d][pos[d-1]+1]
///  msz = indices[d].size()
///  if (plo < phi) {
///    present = indices[d][phi-1] == i[d]
///  } else { // first insertion
///    present = false
///    pointers[d][pos[d-1]] = msz
///  }
///  if (present) { // index already present
///    next = phi-1
///  } else {
///    indices[d].push_back(i[d])
///    pointers[d][pos[d-1]+1] = msz+1
///    next = msz
///    <prepare dimension d + 1>
///  }
///  pos[d] = next
static Value genCompressed(OpBuilder &builder, Location loc,
                           RankedTensorType rtp, SmallVectorImpl<Value> &fields,
                           SmallVectorImpl<Value> &indices, Value value,
                           Value pos, unsigned field, unsigned d) {
  unsigned rank = rtp.getShape().size();
  SmallVector<Type, 4> types;
  Type indexType = builder.getIndexType();
  Type boolType = builder.getIntegerType(1);
  Value one = constantIndex(builder, loc, 1);
  Value pp1 = builder.create<arith::AddIOp>(loc, pos, one);
  Value plo = genLoad(builder, loc, fields[field], pos);
  Value phi = genLoad(builder, loc, fields[field], pp1);
  Value psz = constantIndex(builder, loc, getMemSizesIndex(field + 1));
  Value msz = genLoad(builder, loc, fields[memSizesIdx], psz);
  Value phim1 = builder.create<arith::SubIOp>(
      loc, toType(builder, loc, phi, indexType), one);
  // Conditional expression.
  Value lt =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, plo, phi);
  types.push_back(boolType);
  scf::IfOp ifOp1 = builder.create<scf::IfOp>(loc, types, lt, /*else*/ true);
  types.pop_back();
  builder.setInsertionPointToStart(&ifOp1.getThenRegion().front());
  Value crd = genLoad(builder, loc, fields[field + 1], phim1);
  Value eq = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           toType(builder, loc, crd, indexType),
                                           indices[d]);
  builder.create<scf::YieldOp>(loc, eq);
  builder.setInsertionPointToStart(&ifOp1.getElseRegion().front());
  if (d > 0)
    genStore(builder, loc, msz, fields[field], pos);
  builder.create<scf::YieldOp>(loc, constantI1(builder, loc, false));
  builder.setInsertionPointAfter(ifOp1);
  Value p = ifOp1.getResult(0);
  // If present construct. Note that for a non-unique dimension level, we simply
  // set the condition to false and rely on CSE/DCE to clean up the IR.
  //
  // TODO: generate less temporary IR?
  //
  for (unsigned i = 0, e = fields.size(); i < e; i++)
    types.push_back(fields[i].getType());
  types.push_back(indexType);
  if (!isUniqueDim(rtp, d))
    p = constantI1(builder, loc, false);
  scf::IfOp ifOp2 = builder.create<scf::IfOp>(loc, types, p, /*else*/ true);
  // If present (fields unaffected, update next to phim1).
  builder.setInsertionPointToStart(&ifOp2.getThenRegion().front());
  fields.push_back(phim1);
  builder.create<scf::YieldOp>(loc, fields);
  fields.pop_back();
  // If !present (changes fields, update next).
  builder.setInsertionPointToStart(&ifOp2.getElseRegion().front());
  Value mszp1 = builder.create<arith::AddIOp>(loc, msz, one);
  genStore(builder, loc, mszp1, fields[field], pp1);
  createPushback(builder, loc, fields, field + 1, indices[d]);
  // Prepare the next dimension "as needed".
  if ((d + 1) < rank)
    allocSchemeForRank(builder, loc, rtp, fields, field + 2, d + 1);
  fields.push_back(msz);
  builder.create<scf::YieldOp>(loc, fields);
  fields.pop_back();
  // Update fields and return next pos.
  builder.setInsertionPointAfter(ifOp2);
  unsigned o = 0;
  for (unsigned i = 0, e = fields.size(); i < e; i++)
    fields[i] = ifOp2.getResult(o++);
  return ifOp2.getResult(o);
}

/// Generates code along an insertion path without the need for a "cursor".
/// This current insertion strategy comes at the expense of some testing
/// overhead for each insertion. The strategy will be optimized later for
/// common insertion patterns. The current insertion strategy also assumes
/// insertions occur in "a reasonable order" that enables building the
/// storage scheme in an appending/inserting kind of fashion (i.e. no
/// in-between insertions that need data movement). The implementation
/// relies on CSE/DCE to clean up all bookkeeping that is not needed.
///
/// TODO: better unord/not-unique; also generalize, optimize, specialize!
///
static void genInsert(OpBuilder &builder, Location loc, RankedTensorType rtp,
                      SmallVectorImpl<Value> &fields,
                      SmallVectorImpl<Value> &indices, Value value) {
  unsigned rank = rtp.getShape().size();
  assert(rank == indices.size());
  unsigned field = fieldsIdx; // start past header
  Value pos = constantZero(builder, loc, builder.getIndexType());
  // Generate code for every dimension.
  for (unsigned d = 0; d < rank; d++) {
    if (isCompressedDim(rtp, d)) {
      // Create:
      //   if (!present) {
      //     indices[d].push_back(i[d])
      //     <update pointers and prepare dimension d + 1>
      //   }
      //   pos[d] = indices.size() - 1
      //   <insert @ pos[d] at next dimension d + 1>
      pos = genCompressed(builder, loc, rtp, fields, indices, value, pos, field,
                          d);
      field += 2;
    } else if (isSingletonDim(rtp, d)) {
      // Create:
      //   indices[d].push_back(i[d])
      //   pos[d] = pos[d-1]
      //   <insert @ pos[d] at next dimension d + 1>
      createPushback(builder, loc, fields, field, indices[d]);
      field += 1;
    } else {
      assert(isDenseDim(rtp, d));
      // Construct the new position as:
      //   pos[d] = size * pos[d-1] + i[d]
      //   <insert @ pos[d] at next dimension d + 1>
      Value size = sizeAtStoredDim(builder, loc, rtp, fields, d);
      Value mult = builder.create<arith::MulIOp>(loc, size, pos);
      pos = builder.create<arith::AddIOp>(loc, mult, indices[d]);
    }
  }
  // Reached the actual value append/insert.
  if (!isDenseDim(rtp, rank - 1))
    createPushback(builder, loc, fields, field++, value);
  else
    genStore(builder, loc, value, fields[field++], pos);
  assert(fields.size() == field);
}

/// Generations insertion finalization code.
static void genEndInsert(OpBuilder &builder, Location loc, RankedTensorType rtp,
                         SmallVectorImpl<Value> &fields) {
  unsigned rank = rtp.getShape().size();
  unsigned field = fieldsIdx; // start past header
  for (unsigned d = 0; d < rank; d++) {
    if (isCompressedDim(rtp, d)) {
      // Compressed dimensions need a pointer cleanup for all entries
      // that were not visited during the insertion pass.
      //
      // TODO: avoid cleanup and keep compressed scheme consistent at all times?
      //
      if (d > 0) {
        unsigned ptrWidth = getSparseTensorEncoding(rtp).getPointerBitWidth();
        Type indexType = builder.getIndexType();
        Type ptrType = ptrWidth ? builder.getIntegerType(ptrWidth) : indexType;
        Value mz = constantIndex(builder, loc, getMemSizesIndex(field));
        Value hi = genLoad(builder, loc, fields[memSizesIdx], mz);
        Value zero = constantIndex(builder, loc, 0);
        Value one = constantIndex(builder, loc, 1);
        SmallVector<Value, 1> inits;
        inits.push_back(genLoad(builder, loc, fields[field], zero));
        scf::ForOp loop = createFor(builder, loc, hi, inits, one);
        Value i = loop.getInductionVar();
        Value oldv = loop.getRegionIterArg(0);
        Value newv = genLoad(builder, loc, fields[field], i);
        Value ptrZero = constantZero(builder, loc, ptrType);
        Value cond = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, newv, ptrZero);
        scf::IfOp ifOp = builder.create<scf::IfOp>(loc, TypeRange(ptrType),
                                                   cond, /*else*/ true);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        genStore(builder, loc, oldv, fields[field], i);
        builder.create<scf::YieldOp>(loc, oldv);
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        builder.create<scf::YieldOp>(loc, newv);
        builder.setInsertionPointAfter(ifOp);
        builder.create<scf::YieldOp>(loc, ifOp.getResult(0));
        builder.setInsertionPointAfter(loop);
      }
      field += 2;
    } else if (isSingletonDim(rtp, d)) {
      field++;
    } else {
      assert(isDenseDim(rtp, d));
    }
  }
  assert(fields.size() == ++field);
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
  SparseTensorAllocConverter(TypeConverter &typeConverter, MLIRContext *context,
                             bool enableInit)
      : OpConversionPattern(typeConverter, context),
        enableBufferInitialization(enableInit) {}
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
    createAllocFields(rewriter, loc, resType, adaptor.getOperands(),
                      enableBufferInitialization, fields);
    // Replace operation with resulting memrefs.
    rewriter.replaceOp(op, genTuple(rewriter, loc, resType, fields));
    return success();
  }

private:
  bool enableBufferInitialization;
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
    Value index = genLoad(rewriter, loc, added, i);
    Value value = genLoad(rewriter, loc, values, index);
    indices.push_back(index);
    // TODO: faster for subsequent insertions?
    genInsert(rewriter, loc, dstType, fields, indices, value);
    genStore(rewriter, loc, constantZero(rewriter, loc, eltType), values,
             index);
    genStore(rewriter, loc, constantI1(rewriter, loc, false), filled, index);
    rewriter.create<scf::YieldOp>(loc, fields);
    rewriter.setInsertionPointAfter(loop);
    Value result = genTuple(rewriter, loc, dstType, loop->getResults());
    // Deallocate the buffers on exit of the full loop nest.
    Operation *parent = getTop(op);
    rewriter.setInsertionPointAfter(parent);
    rewriter.create<memref::DeallocOp>(loc, values);
    rewriter.create<memref::DeallocOp>(loc, filled);
    rewriter.create<memref::DeallocOp>(loc, added);
    // Replace operation with resulting memrefs.
    rewriter.replaceOp(op, result);
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
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, fields[memSizesIdx], field);
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
void mlir::populateSparseTensorCodegenPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    bool enableBufferInitialization) {
  patterns.add<SparseReturnConverter, SparseCallConverter, SparseDimOpConverter,
               SparseCastConverter, SparseTensorAllocConverter,
               SparseTensorDeallocConverter, SparseTensorLoadConverter,
               SparseExpandConverter, SparseCompressConverter,
               SparseInsertConverter, SparseToPointersConverter,
               SparseToIndicesConverter, SparseToValuesConverter,
               SparseConvertConverter, SparseNumberOfEntriesConverter>(
      typeConverter, patterns.getContext());
  patterns.add<SparseTensorAllocConverter>(typeConverter, patterns.getContext(),
                                           enableBufferInitialization);
}
