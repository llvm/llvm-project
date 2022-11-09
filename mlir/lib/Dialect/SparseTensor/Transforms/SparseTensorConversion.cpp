//===- SparseTensorConversion.cpp - Sparse tensor primitives conversion ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that converts sparse tensor primitives into calls into a runtime
// support library. Sparse tensor types are converted into opaque pointers
// to the underlying sparse storage schemes. The use of opaque pointers
// together with runtime support library keeps the conversion relatively
// simple, but at the expense of IR opacity, which obscures opportunities
// for subsequent optimization of the IR. An alternative is provided by
// the SparseTensorCodegen pass.
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
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

/// Maps each sparse tensor type to an opaque pointer.
static Optional<Type> convertSparseTensorTypes(Type type) {
  if (getSparseTensorEncoding(type) != nullptr)
    return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
  return llvm::None;
}

/// Replaces the `op` with  a `CallOp` to the function reference returned
/// by `getFunc()`.
static func::CallOp replaceOpWithFuncCall(RewriterBase &rewriter, Operation *op,
                                          StringRef name, TypeRange resultType,
                                          ValueRange operands,
                                          EmitCInterface emitCInterface) {
  auto fn = getFunc(op->getParentOfType<ModuleOp>(), name, resultType, operands,
                    emitCInterface);
  return rewriter.replaceOpWithNewOp<func::CallOp>(op, resultType, fn,
                                                   operands);
}

/// Generates dimension size call.
static Value genDimSizeCall(OpBuilder &builder, Location loc,
                            SparseTensorEncodingAttr &enc, Value src,
                            uint64_t idx) {
  // Generate the call.
  StringRef name = "sparseDimSize";
  SmallVector<Value, 2> params{
      src, constantIndex(builder, loc, toStoredDim(enc, idx))};
  Type iTp = builder.getIndexType();
  return createFuncCall(builder, loc, name, iTp, params, EmitCInterface::Off)
      .getResult(0);
}

/// Compute the size from type (for static sizes) or from an already-converted
/// opaque pointer source (for dynamic sizes) at the given dimension.
static Value sizeFromPtrAtDim(OpBuilder &builder, Location loc,
                              SparseTensorEncodingAttr &enc, ShapedType stp,
                              Value src, unsigned dim) {
  auto shape = stp.getShape();
  if (shape[dim] == ShapedType::kDynamicSize)
    return genDimSizeCall(builder, loc, enc, src, dim);
  return constantIndex(builder, loc, shape[dim]);
}

/// Populates given sizes array from type (for static sizes) and from
/// an already-converted opaque pointer source (for dynamic sizes).
static void sizesFromPtr(OpBuilder &builder, SmallVector<Value, 4> &sizes,
                         Location loc, SparseTensorEncodingAttr &enc,
                         ShapedType stp, Value src) {
  for (unsigned i = 0, rank = stp.getRank(); i < rank; i++)
    sizes.push_back(sizeFromPtrAtDim(builder, loc, enc, stp, src, i));
}

/// Populates given sizes array from type.
static void sizesFromType(OpBuilder &builder, SmallVector<Value, 4> &sizes,
                          Location loc, ShapedType stp) {
  auto shape = stp.getShape();
  for (unsigned i = 0, rank = stp.getRank(); i < rank; i++) {
    uint64_t s = shape[i] == ShapedType::kDynamicSize ? 0 : shape[i];
    sizes.push_back(constantIndex(builder, loc, s));
  }
}

/// Populates the given sizes array for concatenation from type (for static
/// sizes) and from an already-converted opaque pointer source (for dynamic
/// sizes).
static void concatSizesFromInputs(OpBuilder &builder,
                                  SmallVector<Value, 4> &sizes, Location loc,
                                  ShapedType dstTp, ValueRange srcs,
                                  unsigned dim) {
  auto dstShape = dstTp.getShape();

  auto srcTp = srcs[0].getType().cast<ShapedType>();
  auto srcEnc = getSparseTensorEncoding(srcTp);
  // We first fills the sizes from an input tensor, and then
  // compute the size of the concatenation dimension if necessary.
  if (srcEnc)
    // Reuses sizes from an arbitrary input tensor is fine.
    sizesFromPtr(builder, sizes, loc, srcEnc, srcTp, srcs[0]);
  else
    sizesFromSrc(builder, sizes, loc, srcs[0]);

  // Sum up on the `dim` if the dimension is dynamic.
  if (dstShape[dim] != ShapedType::kDynamicSize) {
    // Faithfully take the static size.
    sizes[dim] = constantIndex(builder, loc, dstShape[dim]);
  } else {
    // Else, compute the shape dynamically.
    for (size_t i = 1, sz = srcs.size(); i < sz; i++) {
      auto srcTp = srcs[i].getType().cast<ShapedType>();
      auto encSrc = getSparseTensorEncoding(srcTp);
      Value srcSz =
          encSrc ? sizeFromPtrAtDim(builder, loc, encSrc, srcTp, srcs[i], dim)
                 : linalg::createOrFoldDimOp(builder, loc, srcs[i], dim);
      // Sum up all the sizes.
      sizes[dim] = builder.create<arith::AddIOp>(loc, sizes[dim], srcSz);
    }
  }
}

/// Generates an uninitialized buffer of the given size and type,
/// but returns it as type `memref<? x $tp>` (rather than as type
/// `memref<$sz x $tp>`). Unlike temporary buffers on the stack,
/// this buffer must be explicitly deallocated by client.
static Value genAlloc(RewriterBase &rewriter, Location loc, Value sz, Type tp) {
  auto memTp = MemRefType::get({ShapedType::kDynamicSize}, tp);
  return rewriter.create<memref::AllocOp>(loc, memTp, ValueRange{sz});
}

/// Generates a temporary buffer of the given type and given contents.
static Value genBuffer(OpBuilder &builder, Location loc, ValueRange values) {
  unsigned sz = values.size();
  assert(sz >= 1);
  Value buffer = genAlloca(builder, loc, sz, values[0].getType());
  for (unsigned i = 0; i < sz; i++) {
    Value idx = constantIndex(builder, loc, i);
    builder.create<memref::StoreOp>(loc, values[i], buffer, idx);
  }
  return buffer;
}

/// This class abstracts over the API of `_mlir_ciface_newSparseTensor`:
/// the "swiss army knife" method of the sparse runtime support library
/// for materializing sparse tensors into the computation.  This abstraction
/// reduces the need to make modifications to client code whenever that
/// API changes.
class NewCallParams final {
public:
  /// Allocates the `ValueRange` for the `func::CallOp` parameters,
  /// but does not initialize them.
  NewCallParams(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc), pTp(getOpaquePointerType(builder)) {}

  /// Initializes all static parameters (i.e., those which indicate
  /// type-level information such as the encoding and sizes), generating
  /// MLIR buffers as needed, and returning `this` for method chaining.
  /// This method does not set the action and pointer arguments, since
  /// those are handled by `genNewCall` instead.
  NewCallParams &genBuffers(SparseTensorEncodingAttr enc, ValueRange sizes,
                            ShapedType stp);

  /// (Re)sets the C++ template type parameters, and returns `this`
  /// for method chaining.  This is already done as part of `genBuffers`,
  /// but is factored out so that it can also be called independently
  /// whenever subsequent `genNewCall` calls want to reuse the same
  /// buffers but different type parameters.
  //
  // TODO: This is only ever used by sparse2sparse-viaCOO `ConvertOp`;
  // is there a better way to handle that than this one-off setter method?
  NewCallParams &setTemplateTypes(SparseTensorEncodingAttr enc,
                                  ShapedType stp) {
    params[kParamPtrTp] = constantPointerTypeEncoding(builder, loc, enc);
    params[kParamIndTp] = constantIndexTypeEncoding(builder, loc, enc);
    params[kParamValTp] =
        constantPrimaryTypeEncoding(builder, loc, stp.getElementType());
    return *this;
  }

  /// Checks whether all the static parameters have been initialized.
  bool isInitialized() const {
    for (unsigned i = 0; i < kNumStaticParams; ++i)
      if (!params[i])
        return false;
    return true;
  }

  /// Gets the dimension-to-level mapping.
  //
  // TODO: This is only ever used for passing into `genAddEltCall`;
  // is there a better way to encapsulate that pattern (both to avoid
  // this one-off getter, and to avoid potential mixups)?
  Value getDim2LvlMap() const {
    assert(isInitialized() && "Must initialize before getDim2LvlMap");
    return params[kParamDim2Lvl];
  }

  /// Generates a function call, with the current static parameters
  /// and the given dynamic arguments.
  Value genNewCall(Action action, Value ptr = Value()) {
    assert(isInitialized() && "Must initialize before genNewCall");
    StringRef name = "newSparseTensor";
    params[kParamAction] = constantAction(builder, loc, action);
    params[kParamPtr] = ptr ? ptr : builder.create<LLVM::NullOp>(loc, pTp);
    return createFuncCall(builder, loc, name, pTp, params, EmitCInterface::On)
        .getResult(0);
  }

private:
  static constexpr unsigned kNumStaticParams = 6;
  static constexpr unsigned kNumDynamicParams = 2;
  static constexpr unsigned kNumParams = kNumStaticParams + kNumDynamicParams;
  static constexpr unsigned kParamLvlTypes = 0;
  static constexpr unsigned kParamDimSizes = 1;
  static constexpr unsigned kParamDim2Lvl = 2;
  static constexpr unsigned kParamPtrTp = 3;
  static constexpr unsigned kParamIndTp = 4;
  static constexpr unsigned kParamValTp = 5;
  static constexpr unsigned kParamAction = 6;
  static constexpr unsigned kParamPtr = 7;

  OpBuilder &builder;
  Location loc;
  Type pTp;
  Value params[kNumParams];
};

// TODO: see the note at `_mlir_ciface_newSparseTensor` about how
// the meaning of the various arguments (e.g., "sizes" vs "shapes")
// is inconsistent between the different actions.
NewCallParams &NewCallParams::genBuffers(SparseTensorEncodingAttr enc,
                                         ValueRange dimSizes, ShapedType stp) {
  const unsigned lvlRank = enc.getDimLevelType().size();
  const unsigned dimRank = stp.getRank();
  // Sparsity annotations.
  SmallVector<Value, 4> lvlTypes;
  for (auto dlt : enc.getDimLevelType())
    lvlTypes.push_back(constantDimLevelTypeEncoding(builder, loc, dlt));
  assert(lvlTypes.size() == lvlRank && "Level-rank mismatch");
  params[kParamLvlTypes] = genBuffer(builder, loc, lvlTypes);
  // Dimension-sizes array of the enveloping tensor.  Useful for either
  // verification of external data, or for construction of internal data.
  assert(dimSizes.size() == dimRank && "Dimension-rank mismatch");
  params[kParamDimSizes] = genBuffer(builder, loc, dimSizes);
  // The dimension-to-level mapping.  We must preinitialize `dim2lvl`
  // so that the true branch below can perform random-access `operator[]`
  // assignment.
  SmallVector<Value, 4> dim2lvl(dimRank);
  auto dimOrder = enc.getDimOrdering();
  if (dimOrder) {
    assert(dimOrder.isPermutation());
    for (unsigned l = 0; l < lvlRank; l++) {
      // The `d`th source variable occurs in the `l`th result position.
      uint64_t d = dimOrder.getDimPosition(l);
      dim2lvl[d] = constantIndex(builder, loc, l);
    }
  } else {
    assert(dimRank == lvlRank && "Rank mismatch");
    for (unsigned i = 0; i < lvlRank; i++)
      dim2lvl[i] = constantIndex(builder, loc, i);
  }
  params[kParamDim2Lvl] = genBuffer(builder, loc, dim2lvl);
  // Secondary and primary types encoding.
  setTemplateTypes(enc, stp);
  // Finally, make note that initialization is complete.
  assert(isInitialized() && "Initialization failed");
  // And return `this` for method chaining.
  return *this;
}

/// Generates a call to obtain the values array.
static Value genValuesCall(OpBuilder &builder, Location loc, ShapedType tp,
                           ValueRange ptr) {
  SmallString<15> name{"sparseValues",
                       primaryTypeFunctionSuffix(tp.getElementType())};
  return createFuncCall(builder, loc, name, tp, ptr, EmitCInterface::On)
      .getResult(0);
}

/// Generates a call to release/delete a `SparseTensorCOO`.
static void genDelCOOCall(OpBuilder &builder, Location loc, Type elemTp,
                          Value coo) {
  SmallString<21> name{"delSparseTensorCOO", primaryTypeFunctionSuffix(elemTp)};
  createFuncCall(builder, loc, name, {}, coo, EmitCInterface::Off);
}

/// Generates a call to release/delete a `SparseTensorIterator`.
static void genDelIteratorCall(OpBuilder &builder, Location loc, Type elemTp,
                               Value iter) {
  SmallString<26> name{"delSparseTensorIterator",
                       primaryTypeFunctionSuffix(elemTp)};
  createFuncCall(builder, loc, name, {}, iter, EmitCInterface::Off);
}

/// Generates a call that adds one element to a coordinate scheme.
/// In particular, this generates code like the following:
///   val = a[i1,..,ik];
///   if val != 0
///     t->add(&val, [i1,..,ik], [p1,..,pk]);
static void genAddEltCall(OpBuilder &builder, Location loc, Type eltType,
                          Value ptr, Value valPtr, Value ind, Value perm) {
  SmallString<9> name{"addElt", primaryTypeFunctionSuffix(eltType)};
  SmallVector<Value, 4> params{ptr, valPtr, ind, perm};
  Type pTp = getOpaquePointerType(builder);
  createFuncCall(builder, loc, name, pTp, params, EmitCInterface::On);
}

/// Generates a call to `iter->getNext()`.  If there is a next element,
/// then it is copied into the out-parameters `ind` and `elemPtr`,
/// and the return value is true.  If there isn't a next element, then
/// the return value is false.
static Value genGetNextCall(OpBuilder &builder, Location loc, Value iter,
                            Value ind, Value elemPtr) {
  Type elemTp = elemPtr.getType().cast<ShapedType>().getElementType();
  SmallString<10> name{"getNext", primaryTypeFunctionSuffix(elemTp)};
  SmallVector<Value, 3> params{iter, ind, elemPtr};
  Type i1 = builder.getI1Type();
  return createFuncCall(builder, loc, name, i1, params, EmitCInterface::On)
      .getResult(0);
}

/// Generates code to deallocate a dense buffer.
static void deallocDenseTensor(OpBuilder &builder, Location loc, Value buffer) {
  builder.create<memref::DeallocOp>(loc, buffer);
}

/// Converts a pointer to COO (from calls to iter->next()) into a vector of
/// indices, apply (optional) `offset` on `offsetDim`.
static SmallVector<Value, 4> loadIndices(OpBuilder &builder, Location loc,
                                         unsigned rank, Value ind,
                                         unsigned offsetDim = 0,
                                         Value offset = Value()) {
  SmallVector<Value, 4> ivs;
  ivs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) {
    Value idx = constantIndex(builder, loc, i);
    idx = builder.create<memref::LoadOp>(loc, ind, idx);
    if (offsetDim == i && offset)
      idx = builder.create<arith::AddIOp>(loc, idx, offset);
    ivs.push_back(idx);
  }
  return ivs;
}

/// Converts the vector indices and store it into the memory pointed by
/// `ind`, apply (optional) `offset` on `offsetDim`.
static void storeIndices(OpBuilder &builder, Location loc, unsigned rank,
                         Value ind, ValueRange ivs, unsigned offsetDim = 0,
                         Value offset = Value()) {
  for (unsigned i = 0; i < rank; i++) {
    Value idx = ivs[i];
    if (offsetDim == i && offset)
      idx = builder.create<arith::AddIOp>(loc, idx, offset);
    builder.create<memref::StoreOp>(loc, idx, ind,
                                    constantIndex(builder, loc, i));
  }
}

/// Inserts a value stored in `elemPtr` into a dense tensor created by
/// allocDenseTensor().
static void insertScalarIntoDenseTensor(OpBuilder &builder, Location loc,
                                        Value elemPtr, Value tensor,
                                        ValueRange ivs) {
  Value elemV = builder.create<memref::LoadOp>(loc, elemPtr);
  builder.create<memref::StoreOp>(loc, elemV, tensor, ivs);
}

/// Determine if the runtime library supports direct conversion to the
/// given target `dimTypes`.
static bool canUseDirectConversion(ArrayRef<DimLevelType> dimTypes) {
  bool alreadyCompressed = false;
  for (uint64_t rank = dimTypes.size(), r = 0; r < rank; r++) {
    const DimLevelType dlt = dimTypes[r];
    if (isCompressedDLT(dlt)) {
      if (alreadyCompressed)
        return false; // Multiple compressed dimensions not yet supported.
      alreadyCompressed = true;
    } else if (isDenseDLT(dlt)) {
      if (alreadyCompressed)
        return false; // Dense after Compressed not yet supported.
    } else if (isSingletonDLT(dlt)) {
      // Direct conversion doesn't have any particular problems with
      // singleton after compressed.
    } else { // TODO: investigate
      return false;
    }
  }
  return true;
}

/// Helper method to translate indices during a reshaping operation.
/// TODO: provide as general utility to MLIR at large?
static void translateIndices(Location loc, ConversionPatternRewriter &rewriter,
                             ArrayRef<ReassociationIndices> reassociation,
                             TensorType dstTp, TensorType srcTp, Value dstIdx,
                             Value srcIdx, ArrayRef<Value> dstShape,
                             ArrayRef<Value> srcShape) {
  unsigned dstRank = dstTp.getRank();
  unsigned srcRank = srcTp.getRank();

  SmallVector<Value, 4> srcIndices;
  for (unsigned i = 0; i < srcRank; i++) {
    Value idx = rewriter.create<memref::LoadOp>(
        loc, srcIdx, constantIndex(rewriter, loc, i));
    srcIndices.push_back(idx);
  }

  SmallVector<Value, 4> dstIndices;
  translateIndicesArray(rewriter, loc, reassociation, srcIndices, srcShape,
                        dstShape, dstIndices);

  for (unsigned i = 0; i < dstRank; i++)
    rewriter.create<memref::StoreOp>(loc, dstIndices[i], dstIdx,
                                     constantIndex(rewriter, loc, i));
}

/// Generate code for a general sparse to sparse reshaping operation.
/// Note that unlike dense reshaping (which can be done with a "cheap"
/// change of view), sparse reshaping is currently done with actual
/// data shuffling.
///
/// TODO: proportional to nnz, but still a lot of data movement
///       https://github.com/llvm/llvm-project/issues/56477
///
///   iter = src->toCOO();
///   coo = newSparseCOO()
///   while (elem = iter->getNext()) {
///     coo->add(reshape(elem.indices), elem.value)
///   }
///   s = newSparseTensor(coo)
template <typename ReshapeOp>
static LogicalResult
genSparse2SparseReshape(ReshapeOp op, typename ReshapeOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto srcTp = op.getSrc().getType().template cast<RankedTensorType>();
  auto dstTp = op.getResult().getType().template cast<RankedTensorType>();
  auto encSrc = getSparseTensorEncoding(srcTp);
  auto encDst = getSparseTensorEncoding(dstTp);
  if (!encDst || !encSrc)
    return failure();

  unsigned srcRank = srcTp.getRank();
  unsigned dstRank = dstTp.getRank();
  Type elemTp = srcTp.getElementType();
  assert(elemTp == dstTp.getElementType() &&
         "reshape should not change element type");
  // Start an iterator over the source tensor (in original index order).
  auto noPerm = SparseTensorEncodingAttr::get(
      op->getContext(), encSrc.getDimLevelType(), AffineMap(), AffineMap(),
      encSrc.getPointerBitWidth(), encSrc.getIndexBitWidth());
  SmallVector<Value, 4> srcSizes;
  sizesFromPtr(rewriter, srcSizes, loc, encSrc, srcTp, adaptor.getSrc());
  NewCallParams params(rewriter, loc);
  Value iter = params.genBuffers(noPerm, srcSizes, srcTp)
                   .genNewCall(Action::kToIterator, adaptor.getSrc());
  // Start a new COO for the destination tensor.
  SmallVector<Value, 4> dstSizes;
  if (dstTp.hasStaticShape()) {
    sizesFromType(rewriter, dstSizes, loc, dstTp);
  } else {
    ArrayRef<int64_t> dstShape = dstTp.getShape();
    genReshapeDstShape(loc, rewriter, dstSizes, srcSizes, dstShape,
                       op.getReassociationIndices());
  }
  Value coo =
      params.genBuffers(encDst, dstSizes, dstTp).genNewCall(Action::kEmptyCOO);
  Value dstPerm = params.getDim2LvlMap();
  // Construct a while loop over the iterator.
  Value srcIdx = genAlloca(rewriter, loc, srcRank, rewriter.getIndexType());
  Value dstIdx = genAlloca(rewriter, loc, dstRank, rewriter.getIndexType());
  Value elemPtr = genAllocaScalar(rewriter, loc, elemTp);
  SmallVector<Value> noArgs;
  SmallVector<Type> noTypes;
  auto whileOp = rewriter.create<scf::WhileOp>(loc, noTypes, noArgs);
  Block *before = rewriter.createBlock(&whileOp.getBefore(), {}, noTypes);
  rewriter.setInsertionPointToEnd(before);
  Value cond = genGetNextCall(rewriter, loc, iter, srcIdx, elemPtr);
  rewriter.create<scf::ConditionOp>(loc, cond, before->getArguments());
  // Translate indices from source to target and insert. Note that we do
  // not need to store the value in elemPtr, as the value is still there.
  Block *after = rewriter.createBlock(&whileOp.getAfter(), {}, noTypes);
  rewriter.setInsertionPointToStart(after);
  translateIndices(loc, rewriter, op.getReassociationIndices(), dstTp, srcTp,
                   dstIdx, srcIdx, dstSizes, srcSizes);
  genAddEltCall(rewriter, loc, elemTp, coo, elemPtr, dstIdx, dstPerm);
  rewriter.create<scf::YieldOp>(loc);
  // Final call to construct sparse tensor storage and free temporary resources.
  rewriter.setInsertionPointAfter(whileOp);
  Value dst = params.genNewCall(Action::kFromCOO, coo);
  genDelCOOCall(rewriter, loc, elemTp, coo);
  genDelIteratorCall(rewriter, loc, elemTp, iter);
  rewriter.replaceOp(op, dst);
  return success();
}

// Generates a while loop that iterates over the COO list extracted
// from `t`, using `bodyBuilder` to build the loop body.
//   while (elem = coo->getNext()) {
//     bodyBuilder
//   }
// TODO: It can be used by other operators (ReshapeOp, ConvertOP) conversion to
// reduce code repetition!
// TODO: rename to `genSparseIterationLoop`?
static void genSparseCOOIterationLoop(
    ConversionPatternRewriter &rewriter, Location loc, Value t,
    RankedTensorType tensorTp,
    function_ref<void(OpBuilder &, Location, Value, Value)> bodyBuilder) {
  auto enc = getSparseTensorEncoding(tensorTp);
  assert(enc && "Generating Sparse Tensor COO Loop on a Dense Tensor!");

  unsigned rank = tensorTp.getRank();
  Type elemTp = tensorTp.getElementType();

  // Start an iterator over the tensor (in original index order).
  auto noPerm = SparseTensorEncodingAttr::get(
      rewriter.getContext(), enc.getDimLevelType(), AffineMap(), AffineMap(),
      enc.getPointerBitWidth(), enc.getIndexBitWidth());
  SmallVector<Value, 4> sizes;
  sizesFromPtr(rewriter, sizes, loc, noPerm, tensorTp, t);
  Value iter = NewCallParams(rewriter, loc)
                   .genBuffers(noPerm, sizes, tensorTp)
                   .genNewCall(Action::kToIterator, t);

  // Construct a while loop over the iterator.
  Value srcIdx = genAlloca(rewriter, loc, rank, rewriter.getIndexType());
  Value elemPtr = genAllocaScalar(rewriter, loc, elemTp);
  SmallVector<Value> noArgs;
  SmallVector<Type> noTypes;
  auto whileOp = rewriter.create<scf::WhileOp>(loc, noTypes, noArgs);
  Block *before = rewriter.createBlock(&whileOp.getBefore(), {}, noTypes);
  rewriter.setInsertionPointToEnd(before);
  Value cond = genGetNextCall(rewriter, loc, iter, srcIdx, elemPtr);
  rewriter.create<scf::ConditionOp>(loc, cond, before->getArguments());
  Block *after = rewriter.createBlock(&whileOp.getAfter(), {}, noTypes);
  rewriter.setInsertionPointToStart(after);
  // Callback here to build loop body.
  bodyBuilder(rewriter, loc, srcIdx, elemPtr);
  rewriter.create<scf::YieldOp>(loc);
  // Finish generating loop.
  rewriter.setInsertionPointAfter(whileOp);

  // Free memory for iterator.
  genDelIteratorCall(rewriter, loc, elemTp, iter);
}

// Generate loop that iterates over a dense tensor.
//   for i1 in dim1
//    ..
//     for ik in dimk
//       val = a[i1,..,ik]
//       if val != 0
//         bodyBuilder(v, [i1, ..., ik])
// TODO: It can be used by other operators (ReshapeOp, ConvertOP) conversion to
// reduce code repetition!
static void genDenseTensorIterationLoop(
    ConversionPatternRewriter &rewriter, Location loc, Value t,
    RankedTensorType tensorTp,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  assert(!getSparseTensorEncoding(tensorTp) &&
         "Generating Dense Tensor Loop on a Sparse Tensor!");

  unsigned rank = tensorTp.getRank();
  Value zero = constantIndex(rewriter, loc, 0);
  Value one = constantIndex(rewriter, loc, 1);

  SmallVector<Value> lo;
  SmallVector<Value> hi;
  SmallVector<Value> st;

  // Fill out loop iteration information.
  for (unsigned i = 0; i < rank; i++) {
    lo.push_back(zero);
    hi.push_back(linalg::createOrFoldDimOp(rewriter, loc, t, i));
    st.push_back(one);
  }

  scf::buildLoopNest(rewriter, loc, lo, hi, st, {},
                     [&](OpBuilder &builder, Location loc, ValueRange ivs,
                         ValueRange args) -> scf::ValueVector {
                       // Invoke callback to build the body of the loop.
                       bodyBuilder(builder, loc, ivs);
                       return {};
                     });
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

/// Sparse conversion rule for dimension accesses.
class SparseTensorToDimSizeConverter
    : public OpConversionPattern<tensor::DimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only rewrite annotated DimOp with constant index.
    auto enc = getSparseTensorEncoding(op.getSource().getType());
    if (!enc)
      return failure();
    Optional<int64_t> index = op.getConstantIndex();
    if (!index)
      return failure();
    // Generate the call.
    Value src = adaptor.getOperands()[0];
    int64_t idx = *index;
    rewriter.replaceOp(op,
                       genDimSizeCall(rewriter, op->getLoc(), enc, src, idx));
    return success();
  }
};

/// Sparse conversion rule for trivial tensor casts.
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

/// Sparse conversion rule for a reshape operator.
template <typename ReshapeOp>
class SparseReshapeConverter : public OpConversionPattern<ReshapeOp> {
public:
  using OpAdaptor = typename OpConversionPattern<ReshapeOp>::OpAdaptor;
  using OpConversionPattern<ReshapeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return genSparse2SparseReshape(op, adaptor, rewriter);
  }
};

/// Sparse conversion rule for the new operator.
class SparseTensorNewConverter : public OpConversionPattern<NewOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resType = op.getType();
    auto enc = getSparseTensorEncoding(resType);
    if (!enc)
      return failure();
    // Generate the call to construct tensor from ptr. The sizes are
    // inferred from the result type of the new operator.
    SmallVector<Value, 4> sizes;
    ShapedType stp = resType.cast<ShapedType>();
    sizesFromType(rewriter, sizes, loc, stp);
    Value ptr = adaptor.getOperands()[0];
    rewriter.replaceOp(op, NewCallParams(rewriter, loc)
                               .genBuffers(enc, sizes, stp)
                               .genNewCall(Action::kFromFile, ptr));
    return success();
  }
};

/// Sparse conversion rule for the alloc operator.
class SparseTensorAllocConverter
    : public OpConversionPattern<bufferization::AllocTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(bufferization::AllocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getCopy())
      return rewriter.notifyMatchFailure(op,
                                         "sparse tensor copy not implemented");
    Location loc = op.getLoc();
    RankedTensorType resType = op.getType();
    auto enc = getSparseTensorEncoding(resType);
    if (!enc)
      return failure();
    // Gather all dimension sizes as SSA values.
    SmallVector<Value> sizes;
    unsigned int operandCtr = 0;
    for (int64_t i = 0; i < resType.getRank(); ++i) {
      if (resType.isDynamicDim(i)) {
        sizes.push_back(adaptor.getOperands()[operandCtr++]);
      } else {
        sizes.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, op.getStaticSize(i)));
      }
    }
    // Generate the call to construct empty tensor. The sizes are
    // explicitly defined by the arguments to the alloc operator.
    rewriter.replaceOp(op,
                       NewCallParams(rewriter, loc)
                           .genBuffers(enc, sizes, resType.cast<ShapedType>())
                           .genNewCall(Action::kEmpty));
    return success();
  }
};

/// Sparse conversion rule for the convert operator.
class SparseTensorConvertConverter : public OpConversionPattern<ConvertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  SparseTensorConvertConverter(MLIRContext *context,
                               SparseTensorConversionOptions o)
      : OpConversionPattern<ConvertOp>(context), options(o) {}
  SparseTensorConvertConverter(TypeConverter &typeConv, MLIRContext *context,
                               SparseTensorConversionOptions o)
      : OpConversionPattern<ConvertOp>(typeConv, context), options(o) {}

  LogicalResult
  matchAndRewrite(ConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type resType = op.getType();
    Type srcType = op.getSource().getType();
    auto encDst = getSparseTensorEncoding(resType);
    auto encSrc = getSparseTensorEncoding(srcType);
    Value src = adaptor.getOperands()[0];
    if (encDst && encSrc) {
      // This is a sparse => sparse conversion, which is handled as follows:
      //   t = src->toCOO();         ; src to COO in dst order
      //   dst = newSparseTensor(t)
      // Using the coordinate scheme as an intermediate does not always
      // yield the fastest conversion but avoids the need for a full
      // O(N^2) conversion matrix.
      if (encDst == encSrc) {
        rewriter.replaceOp(op, adaptor.getOperands()); // hidden nop cast
        return success();
      }
      SmallVector<Value, 4> sizes;
      NewCallParams params(rewriter, loc);
      ShapedType stp = srcType.cast<ShapedType>();
      sizesFromPtr(rewriter, sizes, loc, encSrc, stp, src);
      bool useDirectConversion;
      switch (options.sparseToSparseStrategy) {
      case SparseToSparseConversionStrategy::kViaCOO:
        useDirectConversion = false;
        break;
      case SparseToSparseConversionStrategy::kDirect:
        useDirectConversion = true;
        assert(canUseDirectConversion(encDst.getDimLevelType()) &&
               "Unsupported target for direct sparse-to-sparse conversion");
        break;
      case SparseToSparseConversionStrategy::kAuto:
        useDirectConversion = canUseDirectConversion(encDst.getDimLevelType());
        break;
      }
      if (useDirectConversion) {
        rewriter.replaceOp(op, params.genBuffers(encDst, sizes, stp)
                                   .genNewCall(Action::kSparseToSparse, src));
      } else { // use via-COO conversion.
        // Set up encoding with right mix of src and dst so that the two
        // method calls can share most parameters, while still providing
        // the correct sparsity information to either of them.
        auto enc = SparseTensorEncodingAttr::get(
            op->getContext(), encDst.getDimLevelType(), encDst.getDimOrdering(),
            encDst.getHigherOrdering(), encSrc.getPointerBitWidth(),
            encSrc.getIndexBitWidth());
        // TODO: This is the only place where `kToCOO` (or `kToIterator`)
        // is called with a non-identity permutation.  Is there any clean
        // way to push the permutation over to the `kFromCOO` side instead?
        Value coo =
            params.genBuffers(enc, sizes, stp).genNewCall(Action::kToCOO, src);
        Value dst = params.setTemplateTypes(encDst, stp)
                        .genNewCall(Action::kFromCOO, coo);
        genDelCOOCall(rewriter, loc, stp.getElementType(), coo);
        rewriter.replaceOp(op, dst);
      }
      return success();
    }
    if (!encDst && encSrc) {
      // This is sparse => dense conversion, which is handled as follows:
      //   dst = new Tensor(0);
      //   iter = new SparseTensorIterator(src);
      //   while (elem = iter->getNext()) {
      //     dst[elem.indices] = elem.value;
      //   }
      //   delete iter;
      RankedTensorType dstTensorTp = resType.cast<RankedTensorType>();
      RankedTensorType srcTensorTp = srcType.cast<RankedTensorType>();
      unsigned rank = dstTensorTp.getRank();
      Type elemTp = dstTensorTp.getElementType();
      // Fabricate a no-permutation encoding for NewCallParams
      // The pointer/index types must be those of `src`.
      // The dimLevelTypes aren't actually used by Action::kToIterator.
      encDst = SparseTensorEncodingAttr::get(
          op->getContext(),
          SmallVector<DimLevelType>(rank, DimLevelType::Dense), AffineMap(),
          AffineMap(), encSrc.getPointerBitWidth(), encSrc.getIndexBitWidth());
      SmallVector<Value, 4> sizes;
      sizesFromPtr(rewriter, sizes, loc, encSrc, srcTensorTp, src);
      Value iter = NewCallParams(rewriter, loc)
                       .genBuffers(encDst, sizes, dstTensorTp)
                       .genNewCall(Action::kToIterator, src);
      Value ind = genAlloca(rewriter, loc, rank, rewriter.getIndexType());
      Value elemPtr = genAllocaScalar(rewriter, loc, elemTp);
      Block *insertionBlock = rewriter.getInsertionBlock();
      // TODO: Dense buffers should be allocated/deallocated via the callback
      // in BufferizationOptions.
      Value dst = allocDenseTensor(rewriter, loc, dstTensorTp, sizes);
      SmallVector<Value> noArgs;
      SmallVector<Type> noTypes;
      auto whileOp = rewriter.create<scf::WhileOp>(loc, noTypes, noArgs);
      Block *before = rewriter.createBlock(&whileOp.getBefore(), {}, noTypes);
      rewriter.setInsertionPointToEnd(before);
      Value cond = genGetNextCall(rewriter, loc, iter, ind, elemPtr);
      rewriter.create<scf::ConditionOp>(loc, cond, before->getArguments());
      Block *after = rewriter.createBlock(&whileOp.getAfter(), {}, noTypes);
      rewriter.setInsertionPointToStart(after);
      SmallVector<Value, 4> ivs = loadIndices(rewriter, loc, rank, ind);
      insertScalarIntoDenseTensor(rewriter, loc, elemPtr, dst, ivs);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointAfter(whileOp);
      genDelIteratorCall(rewriter, loc, elemTp, iter);
      rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, resType, dst);
      // Deallocate the buffer.
      if (bufferization::allocationDoesNotEscape(op->getOpResult(0))) {
        rewriter.setInsertionPoint(insertionBlock->getTerminator());
        deallocDenseTensor(rewriter, loc, dst);
      }
      return success();
    }
    if (!encDst && !encSrc) {
      // dense => dense
      return failure();
    }
    // This is a dense => sparse conversion or a sparse constant in COO =>
    // sparse conversion, which is handled as follows:
    //   t = newSparseCOO()
    //   ...code to fill the COO tensor t...
    //   s = newSparseTensor(t)
    //
    // To fill the COO tensor from a dense tensor:
    //   for i1 in dim1
    //    ..
    //     for ik in dimk
    //       val = a[i1,..,ik]
    //       if val != 0
    //         t->add(val, [i1,..,ik], [p1,..,pk])
    //
    // To fill the COO tensor from a sparse constant in COO format:
    //   for i in range(NNZ)
    //     val = values[i]
    //     [i1,..,ik] = indices[i]
    //     t->add(val, [i1,..,ik], [p1,..,pk])
    //
    // Note that the dense tensor traversal code is actually implemented
    // using MLIR IR to avoid having to expose too much low-level
    // memref traversal details to the runtime support library.
    // Also note that the code below only generates the "new" ops and
    // the loop-nest per se; whereas the entire body of the innermost
    // loop is generated by genAddElt().
    ShapedType stp = resType.cast<ShapedType>();
    unsigned rank = stp.getRank();
    SmallVector<Value, 4> sizes;
    sizesFromSrc(rewriter, sizes, loc, src);
    NewCallParams params(rewriter, loc);
    Value coo =
        params.genBuffers(encDst, sizes, stp).genNewCall(Action::kEmptyCOO);
    Value ind = genAlloca(rewriter, loc, rank, rewriter.getIndexType());
    Value perm = params.getDim2LvlMap();
    Type eltType = stp.getElementType();
    Value elemPtr = genAllocaScalar(rewriter, loc, eltType);
    genDenseTensorOrSparseConstantIterLoop(
        rewriter, loc, src, rank,
        [&](OpBuilder &builder, Location loc, Value val, ValueRange indices) {
          for (unsigned i = 0; i < rank; i++) {
            Value idx = constantIndex(builder, loc, i);
            builder.create<memref::StoreOp>(loc, indices[i], ind, idx);
          }
          builder.create<memref::StoreOp>(loc, val, elemPtr);
          genAddEltCall(builder, loc, eltType, coo, elemPtr, ind, perm);
        });
    // Final call to construct sparse tensor storage.
    Value dst = params.genNewCall(Action::kFromCOO, coo);
    genDelCOOCall(rewriter, loc, eltType, coo);
    rewriter.replaceOp(op, dst);
    return success();
  }

private:
  /// Options to control sparse code generation.
  SparseTensorConversionOptions options;
};

/// Sparse conversion rule for the dealloc operator.
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
    StringRef name = "delSparseTensor";
    createFuncCall(rewriter, op->getLoc(), name, {}, adaptor.getOperands(),
                   EmitCInterface::Off);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Sparse conversion rule for pointer accesses.
class SparseTensorToPointersConverter
    : public OpConversionPattern<ToPointersOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToPointersOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Type ptrType = resType.cast<ShapedType>().getElementType();
    SmallString<16> name{"sparsePointers", overheadTypeFunctionSuffix(ptrType)};
    Value dim =
        constantIndex(rewriter, op->getLoc(), op.getDimension().getZExtValue());
    replaceOpWithFuncCall(rewriter, op, name, resType,
                          {adaptor.getTensor(), dim}, EmitCInterface::On);
    return success();
  }
};

/// Sparse conversion rule for index accesses.
class SparseTensorToIndicesConverter : public OpConversionPattern<ToIndicesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToIndicesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Type indType = resType.cast<ShapedType>().getElementType();
    SmallString<15> name{"sparseIndices", overheadTypeFunctionSuffix(indType)};
    Value dim =
        constantIndex(rewriter, op->getLoc(), op.getDimension().getZExtValue());
    replaceOpWithFuncCall(rewriter, op, name, resType,
                          {adaptor.getTensor(), dim}, EmitCInterface::On);
    return success();
  }
};

/// Sparse conversion rule for value accesses.
class SparseTensorToValuesConverter : public OpConversionPattern<ToValuesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToValuesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = op.getType().cast<ShapedType>();
    rewriter.replaceOp(op, genValuesCall(rewriter, op.getLoc(), resType,
                                         adaptor.getOperands()));
    return success();
  }
};

/// Sparse conversion rule for number of entries operator.
class SparseNumberOfEntriesConverter
    : public OpConversionPattern<NumberOfEntriesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NumberOfEntriesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Query values array size for the actually stored values size.
    Type eltType = op.getTensor().getType().cast<ShapedType>().getElementType();
    auto resTp = MemRefType::get({ShapedType::kDynamicSize}, eltType);
    Value values = genValuesCall(rewriter, loc, resTp, adaptor.getOperands());
    rewriter.replaceOpWithNewOp<memref::DimOp>(op, values,
                                               constantIndex(rewriter, loc, 0));
    return success();
  }
};

/// Sparse conversion rule for tensor rematerialization.
class SparseTensorLoadConverter : public OpConversionPattern<LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getHasInserts()) {
      // Finalize any pending insertions.
      StringRef name = "endInsert";
      createFuncCall(rewriter, op->getLoc(), name, {}, adaptor.getOperands(),
                     EmitCInterface::Off);
    }
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Sparse conversion rule for the insertion operator.
class SparseTensorInsertConverter : public OpConversionPattern<InsertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Note that the current regime only allows for strict lexicographic
    // index order. All values are passed by reference through stack
    // allocated memrefs.
    Location loc = op->getLoc();
    auto tp = op.getTensor().getType().cast<RankedTensorType>();
    auto elemTp = tp.getElementType();
    unsigned rank = tp.getRank();
    auto mref = genAlloca(rewriter, loc, rank, rewriter.getIndexType());
    auto vref = genAllocaScalar(rewriter, loc, elemTp);
    for (unsigned i = 0; i < rank; i++)
      rewriter.create<memref::StoreOp>(loc, adaptor.getIndices()[i], mref,
                                       constantIndex(rewriter, loc, i));
    rewriter.create<memref::StoreOp>(loc, adaptor.getValue(), vref);
    SmallString<12> name{"lexInsert", primaryTypeFunctionSuffix(elemTp)};
    createFuncCall(rewriter, loc, name, {}, {adaptor.getTensor(), mref, vref},
                   EmitCInterface::On);
    rewriter.replaceOp(op, adaptor.getTensor());
    return success();
  }
};

/// Sparse conversion rule for the expand operator.
class SparseTensorExpandConverter : public OpConversionPattern<ExpandOp> {
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
    // dimension size, translated back to original dimension).
    auto enc = getSparseTensorEncoding(srcType);
    unsigned innerDim = toOrigDim(srcType, srcType.getRank() - 1);
    auto sz = sizeFromPtrAtDim(rewriter, loc, enc, srcType, adaptor.getTensor(),
                               innerDim);
    // Allocate temporary buffers for values, filled-switch, and indices.
    // We do not use stack buffers for this, since the expanded size may
    // be rather large (as it envelops a single expanded dense dimension).
    Value values = genAlloc(rewriter, loc, sz, eltType);
    Value filled = genAlloc(rewriter, loc, sz, boolType);
    Value indices = genAlloc(rewriter, loc, sz, idxType);
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
    rewriter.replaceOp(op, {values, filled, indices, zero});
    return success();
  }
};

/// Sparse conversion rule for the compress operator.
class SparseTensorCompressConverter : public OpConversionPattern<CompressOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // Note that this method call resets the values/filled-switch back to
    // all-zero/false by only iterating over the set elements, so the
    // complexity remains proportional to the sparsity of the expanded
    // access pattern.
    Value values = adaptor.getValues();
    Value filled = adaptor.getFilled();
    Value added = adaptor.getAdded();
    Value count = adaptor.getCount();
    Value tensor = adaptor.getTensor();
    auto tp = op.getTensor().getType().cast<RankedTensorType>();
    Type elemTp = tp.getElementType();
    unsigned rank = tp.getRank();
    auto mref = genAlloca(rewriter, loc, rank, rewriter.getIndexType());
    for (unsigned i = 0; i < rank - 1; i++)
      rewriter.create<memref::StoreOp>(loc, adaptor.getIndices()[i], mref,
                                       constantIndex(rewriter, loc, i));
    SmallString<12> name{"expInsert", primaryTypeFunctionSuffix(elemTp)};
    createFuncCall(rewriter, loc, name, {},
                   {tensor, mref, values, filled, added, count},
                   EmitCInterface::On);
    rewriter.replaceOp(op, adaptor.getTensor());
    // Deallocate the buffers on exit of the loop nest.
    Operation *parent = getTop(op);
    rewriter.setInsertionPointAfter(parent);
    rewriter.create<memref::DeallocOp>(loc, values);
    rewriter.create<memref::DeallocOp>(loc, filled);
    rewriter.create<memref::DeallocOp>(loc, added);
    return success();
  }
};

/// Sparse conversion rule for the concatenate operator.
class SparseTensorConcatConverter : public OpConversionPattern<ConcatenateOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConcatenateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The conversion works as follow:
    // (1). When output is sparse, and mix of inputs:
    //    a_sparse = concat (b_dense, c_sparse, ....)
    // =>
    //    coo_for_a = newSparseCOO(shapeOf(a))
    //    for i, j, k // dense input
    //      coo->add(adjustForOffset(i,j,k), b[i,j,k])
    //
    //    for elem in sparse_input
    //      coo->add(adjustForOffset(elem.indices), elem.value)
    //    ...
    //    a = newSparseTensor(coo_for_a)
    //    return a
    //
    // (2). When output is dense, and mix of inputs:
    //    a_dense = concat (b_dense, c_sparse, ....)
    // =>
    //    a = malloc(shapeOf(a))
    //    for i, j, k // dense input
    //      a[ adjustForOffset(i,j,k) ] = b[i,j,k]
    //
    //    for elem in sparse_input
    //      a[ adjustForOffset(elem.indices) ] = elem.value
    //    return a
    Location loc = op.getLoc();
    auto dstTp = op.getType().cast<RankedTensorType>();
    auto encDst = getSparseTensorEncoding(dstTp);
    Type elemTp = dstTp.getElementType();
    uint64_t concatDim = op.getDimension().getZExtValue();
    unsigned rank = dstTp.getRank();

    Value dst;     // destination tensor
    Value dstPerm; // destination tensor permutation (if sparse out)
    // A pointer to the value being inserted (if dense => sparse)
    Value elemPtr;
    // Memory that holds the COO for destination tensor (if sparse out)
    Value dstIdx;
    // The offset applied to the dimenstion to be concated (starting from 0)
    Value offset = constantIndex(rewriter, loc, 0);

    SmallVector<Value, 4> sizes;
    NewCallParams params(rewriter, loc);
    concatSizesFromInputs(rewriter, sizes, loc, dstTp, op.getInputs(),
                          concatDim);

    if (encDst) {
      // Start a new COO for the destination tensor.
      dst =
          params.genBuffers(encDst, sizes, dstTp).genNewCall(Action::kEmptyCOO);
      dstPerm = params.getDim2LvlMap();
      elemPtr = genAllocaScalar(rewriter, loc, elemTp);
      dstIdx = genAlloca(rewriter, loc, rank, rewriter.getIndexType());
    } else {
      // TODO: Dense buffers should be allocated/deallocated via the callback
      // in BufferizationOptions.
      dst = allocDenseTensor(rewriter, loc, dstTp, sizes);
    }
    for (auto it : llvm::zip(op.getInputs(), adaptor.getInputs())) {
      Value orignalOp = std::get<0>(it); // Input (with encoding) from Op
      Value adaptedOp = std::get<1>(it); // Input (type converted) from adaptor
      RankedTensorType srcTp = orignalOp.getType().cast<RankedTensorType>();
      auto encSrc = getSparseTensorEncoding(srcTp);
      if (encSrc) {
        genSparseCOOIterationLoop(
            rewriter, loc, adaptedOp, srcTp,
            [&](OpBuilder &builder, Location loc, Value idx,
                Value elemPtr) -> void {
              auto indVec =
                  loadIndices(builder, loc, rank, idx, concatDim, offset);
              if (encDst) {
                // Case: sparse => sparse
                storeIndices(builder, loc, rank, dstIdx, indVec);
                genAddEltCall(builder, loc, elemTp, dst, elemPtr, dstIdx,
                              dstPerm);
              } else {
                // Case: sparse => dense
                insertScalarIntoDenseTensor(builder, loc, elemPtr, dst, indVec);
              }
            });
      } else {
        genDenseTensorIterationLoop(
            rewriter, loc, adaptedOp, srcTp,
            [&](OpBuilder &builder, Location loc, ValueRange idx) -> void {
              if (encDst) {
                // Case: dense => sparse
                storeIndices(builder, loc, rank, dstIdx, idx, concatDim,
                             offset);
                Value val = genValueForDense(builder, loc, adaptedOp, idx);
                builder.create<memref::StoreOp>(loc, val, elemPtr);
                genAddEltCall(builder, loc, elemTp, dst, elemPtr, dstIdx,
                              dstPerm);
              } else {
                // Case: dense => dense
                Value val = genValueForDense(builder, loc, adaptedOp, idx);
                SmallVector<Value, 4> indVec(idx);
                // Apply offset.
                indVec[concatDim] = builder.create<arith::AddIOp>(
                    loc, indVec[concatDim], offset);
                builder.create<memref::StoreOp>(loc, val, dst, indVec);
              }
            });
      }
      // Accumulate offset.
      // TODO: avoid calling sparseDimSize multiple times by caching the result!
      Value curDim = encSrc ? sizeFromPtrAtDim(rewriter, loc, encSrc, srcTp,
                                               adaptedOp, concatDim)
                            : linalg::createOrFoldDimOp(rewriter, loc,
                                                        adaptedOp, concatDim);

      offset = rewriter.create<arith::AddIOp>(loc, offset, curDim);
    }
    if (encDst) {
      // In sparse output case, the destination holds the COO.
      Value coo = dst;
      dst = params.genNewCall(Action::kFromCOO, coo);
      // Release resources.
      genDelCOOCall(rewriter, loc, elemTp, coo);
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, dstTp, dst);
    }
    return success();
  }
};

/// Sparse conversion rule for the output operator.
class SparseTensorOutConverter : public OpConversionPattern<OutOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    ShapedType srcType = op.getTensor().getType().cast<ShapedType>();
    // Convert to default permuted COO.
    Value src = adaptor.getOperands()[0];
    auto encSrc = getSparseTensorEncoding(srcType);
    SmallVector<Value, 4> sizes;
    sizesFromPtr(rewriter, sizes, loc, encSrc, srcType, src);
    auto enc = SparseTensorEncodingAttr::get(
        op->getContext(), encSrc.getDimLevelType(), AffineMap(), AffineMap(),
        encSrc.getPointerBitWidth(), encSrc.getIndexBitWidth());
    Value coo = NewCallParams(rewriter, loc)
                    .genBuffers(enc, sizes, srcType)
                    .genNewCall(Action::kToCOO, src);
    // Then output the tensor to external file with indices in the externally
    // visible lexicographic index order. A sort is required if the source was
    // not in that order yet (note that the sort can be dropped altogether if
    // external format does not care about the order at all, but here we assume
    // it does).
    Value sort = constantI1(rewriter, loc,
                            encSrc.getDimOrdering() &&
                                !encSrc.getDimOrdering().isIdentity());
    SmallVector<Value, 3> outParams{coo, adaptor.getOperands()[1], sort};
    Type eltType = srcType.getElementType();
    SmallString<18> name{"outSparseTensor", primaryTypeFunctionSuffix(eltType)};
    createFuncCall(rewriter, loc, name, {}, outParams, EmitCInterface::Off);
    genDelCOOCall(rewriter, loc, eltType, coo);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Sparse tensor type conversion into opaque pointer.
//===----------------------------------------------------------------------===//

mlir::SparseTensorTypeToPtrConverter::SparseTensorTypeToPtrConverter() {
  addConversion([](Type type) { return type; });
  addConversion(convertSparseTensorTypes);
}

//===----------------------------------------------------------------------===//
// Public method for populating conversion rules.
//===----------------------------------------------------------------------===//

/// Populates the given patterns list with conversion rules required for
/// the sparsification of linear algebra operations.
void mlir::populateSparseTensorConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const SparseTensorConversionOptions &options) {
  patterns.add<SparseReturnConverter, SparseTensorToDimSizeConverter,
               SparseCastConverter, SparseTensorNewConverter,
               SparseReshapeConverter<tensor::ExpandShapeOp>,
               SparseReshapeConverter<tensor::CollapseShapeOp>,
               SparseTensorConcatConverter, SparseTensorAllocConverter,
               SparseTensorDeallocConverter, SparseTensorToPointersConverter,
               SparseTensorToIndicesConverter, SparseTensorToValuesConverter,
               SparseNumberOfEntriesConverter, SparseTensorLoadConverter,
               SparseTensorInsertConverter, SparseTensorExpandConverter,
               SparseTensorCompressConverter, SparseTensorOutConverter>(
      typeConverter, patterns.getContext());

  patterns.add<SparseTensorConvertConverter>(typeConverter,
                                             patterns.getContext(), options);
}
