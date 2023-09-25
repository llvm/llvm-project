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
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
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
static std::optional<Type> convertSparseTensorTypes(Type type) {
  if (getSparseTensorEncoding(type) != nullptr)
    return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
  return std::nullopt;
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

/// Generates call to lookup a level-size.  N.B., this only generates
/// the raw function call, and therefore (intentionally) does not perform
/// any dim<->lvl conversion or other logic.
static Value genLvlSizeCall(OpBuilder &builder, Location loc, Value tensor,
                            uint64_t lvl) {
  StringRef name = "sparseLvlSize";
  SmallVector<Value, 2> params{tensor, constantIndex(builder, loc, lvl)};
  Type iTp = builder.getIndexType();
  return createFuncCall(builder, loc, name, iTp, params, EmitCInterface::Off)
      .getResult(0);
}

/// Generates call to lookup a dimension-size.  N.B., this only generates
/// the raw function call, and therefore (intentionally) does not perform
/// any dim<->lvl conversion or other logic.
static Value genDimSizeCall(OpBuilder &builder, Location loc, Value tensor,
                            uint64_t dim) {
  StringRef name = "sparseDimSize";
  SmallVector<Value, 2> params{tensor, constantIndex(builder, loc, dim)};
  Type iTp = builder.getIndexType();
  return createFuncCall(builder, loc, name, iTp, params, EmitCInterface::Off)
      .getResult(0);
}

/// Looks up a level-size by returning a statically-computed constant
/// (when possible), or by calling `genLvlSizeCall` (when dynamic).
static Value createOrFoldLvlCall(OpBuilder &builder, Location loc,
                                 SparseTensorType stt, Value tensor,
                                 Level lvl) {
  // Only sparse tensors have "levels" to query.
  assert(stt.hasEncoding());
  // TODO: The following implementation only handles permutations;
  // we'll need to generalize this to handle arbitrary AffineExpr.
  //
  // There's no need to assert `isPermutation` here: because
  // `getDimPosition` checks that the expr isa `AffineDimExpr`,
  // which is all we care about (for supporting permutations).
  const Dimension dim =
      stt.isIdentity() ? lvl : stt.getDimToLvl().getDimPosition(lvl);
  if (const auto sz = stt.getStaticDimSize(dim))
    return constantIndex(builder, loc, *sz);
  // If we cannot statically compute the size from the shape, then we
  // must dynamically query it.  (In principle we could also dynamically
  // compute it, but since we already did so to construct the `tensor`
  // in the first place, we might as well query rather than recompute.)
  return genLvlSizeCall(builder, loc, tensor, lvl);
}

/// Looks up a dimension-size by returning a constant from the shape
/// (for static sizes), or by calling `genDimSizeCall` (for dynamic sizes
/// of sparse tensors) or `linalg::createOrFoldDimOp` (for dynamic sizes
/// of dense tensors).
static Value createOrFoldDimCall(OpBuilder &builder, Location loc,
                                 SparseTensorType stt, Value tensor,
                                 Dimension dim) {
  if (const auto sz = stt.getStaticDimSize(dim))
    return constantIndex(builder, loc, *sz);
  if (stt.hasEncoding())
    return genDimSizeCall(builder, loc, tensor, dim);
  return linalg::createOrFoldDimOp(builder, loc, tensor, dim);
}

/// Populates the array with the dimension-sizes of the given tensor.
static void fillDimSizes(OpBuilder &builder, Location loc, SparseTensorType stt,
                         Value tensor, SmallVectorImpl<Value> &out) {
  const Dimension dimRank = stt.getDimRank();
  out.clear();
  out.reserve(dimRank);
  for (Dimension d = 0; d < dimRank; d++)
    out.push_back(createOrFoldDimCall(builder, loc, stt, tensor, d));
}

/// Returns an array with the dimension-sizes of the given tensor.
/// If the *tensor* parameters is null, the tensor type is assumed to have a
/// static shape.
static SmallVector<Value> getDimSizes(OpBuilder &builder, Location loc,
                                      SparseTensorType stt,
                                      Value tensor = Value()) {
  SmallVector<Value> out;
  fillDimSizes(builder, loc, stt, tensor, out);
  return out;
}

/// Populates the array with the dimension-shape of the given
/// `SparseTensorType`, where dynamic sizes are represented by zero.
static void fillDimShape(OpBuilder &builder, Location loc, SparseTensorType stt,
                         SmallVectorImpl<Value> &out) {
  out.clear();
  out.reserve(stt.getDimRank());
  for (const DynSize sh : stt.getDimShape()) {
    const auto s = ShapedType::isDynamic(sh) ? 0 : sh;
    out.push_back(constantIndex(builder, loc, s));
  }
}

/// Returns an array with the dimension-shape of the given `SparseTensorType`,
/// where dynamic sizes are represented by zero.
static SmallVector<Value> getDimShape(OpBuilder &builder, Location loc,
                                      SparseTensorType stt) {
  SmallVector<Value> out;
  fillDimShape(builder, loc, stt, out);
  return out;
}

/// Populates the given sizes array for concatenation from type (for static
/// sizes) and from an already-converted opaque pointer source (for dynamic
/// sizes).
static void concatDimSizesFromInputs(OpBuilder &builder, Location loc,
                                     SparseTensorType dstTp, ValueRange srcs,
                                     Dimension dim,
                                     SmallVectorImpl<Value> &dimSizes) {
  assert(dim < dstTp.getDimRank() && "Dimension is out of bounds");
  dimSizes.clear();

  // We first fills the sizes from an input tensor, and then
  // compute the size of the concatenation dimension if necessary.
  const auto srcTp = getSparseTensorType(srcs[0]);
  if (srcTp.hasEncoding())
    // Reuses sizes from an arbitrary input tensor is fine.
    fillDimSizes(builder, loc, srcTp, srcs[0], dimSizes);
  else
    sizesFromSrc(builder, dimSizes, loc, srcs[0]);

  if (const auto sz = dstTp.getStaticDimSize(dim)) {
    // Faithfully take the static size.
    dimSizes[dim] = constantIndex(builder, loc, *sz);
  } else {
    // Else, dynamically compute the size.
    for (const auto src : srcs.drop_front()) {
      const auto srcTp = getSparseTensorType(src);
      Value srcSz = createOrFoldDimCall(builder, loc, srcTp, src, dim);
      dimSizes[dim] = builder.create<arith::AddIOp>(loc, dimSizes[dim], srcSz);
    }
  }
}

/// Generates an uninitialized buffer of the given size and type,
/// but returns it as type `memref<? x $tp>` (rather than as type
/// `memref<$sz x $tp>`). Unlike temporary buffers on the stack,
/// this buffer must be explicitly deallocated by client.
static Value genAlloc(RewriterBase &rewriter, Location loc, Value sz, Type tp) {
  auto memTp = MemRefType::get({ShapedType::kDynamic}, tp);
  return rewriter.create<memref::AllocOp>(loc, memTp, ValueRange{sz});
}

/// Generates a temporary buffer for the level-types of the given encoding.
static Value genLvlTypesBuffer(OpBuilder &builder, Location loc,
                               SparseTensorType stt) {
  SmallVector<Value> lvlTypes;
  lvlTypes.reserve(stt.getLvlRank());
  for (const auto dlt : stt.getEncoding().getLvlTypes())
    lvlTypes.push_back(constantDimLevelTypeEncoding(builder, loc, dlt));
  return allocaBuffer(builder, loc, lvlTypes);
}

/// Extracts the bare (aligned) pointers that point to the tensor.
static Value extractBarePtrFromTensor(OpBuilder &builder, Location loc,
                                      Value tensor) {
  auto buf = genToMemref(builder, loc, tensor);
  return builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buf);
}

/// Generates a temporary buffer for the level-types of the given encoding.
static Value genLvlPtrsBuffers(OpBuilder &builder, Location loc,
                               ValueRange lvlTensors, Value valTensor) {
  SmallVector<Value> lvlBarePtrs;
  lvlBarePtrs.reserve(lvlTensors.size() + 1);
  // Passing in lvl buffer pointers.
  for (const auto lvl : lvlTensors)
    lvlBarePtrs.push_back(extractBarePtrFromTensor(builder, loc, lvl));

  // Passing in value buffer pointers.
  lvlBarePtrs.push_back(extractBarePtrFromTensor(builder, loc, valTensor));
  Value idxPtr = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, allocaBuffer(builder, loc, lvlBarePtrs));
  Value idxCast =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), idxPtr);
  return builder.create<LLVM::IntToPtrOp>(loc, getOpaquePointerType(builder),
                                          idxCast);
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
  NewCallParams &genBuffers(SparseTensorType stt, ValueRange dimSizes);

  /// (Re)sets the C++ template type parameters, and returns `this`
  /// for method chaining.  This is already done as part of `genBuffers`,
  /// but is factored out so that it can also be called independently
  /// whenever subsequent `genNewCall` calls want to reuse the same
  /// buffers but different type parameters.
  //
  // TODO: This is only ever used by sparse2sparse-viaCOO `ConvertOp`;
  // is there a better way to handle that than this one-off setter method?
  NewCallParams &setTemplateTypes(SparseTensorType stt) {
    const auto enc = stt.getEncoding();
    params[kParamPosTp] = constantPosTypeEncoding(builder, loc, enc);
    params[kParamCrdTp] = constantCrdTypeEncoding(builder, loc, enc);
    params[kParamValTp] =
        constantPrimaryTypeEncoding(builder, loc, stt.getElementType());
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
  Value getDimToLvl() const {
    assert(isInitialized() && "Must initialize before getDimToLvl");
    return params[kParamDimToLvl];
  }

  /// Generates a function call, with the current static parameters
  /// and the given dynamic arguments.
  Value genNewCall(Action action, Value ptr = Value()) {
    assert(isInitialized() && "Must initialize before genNewCall");
    StringRef name = "newSparseTensor";
    params[kParamAction] = constantAction(builder, loc, action);
    params[kParamPtr] = ptr ? ptr : builder.create<LLVM::ZeroOp>(loc, pTp);
    return createFuncCall(builder, loc, name, pTp, params, EmitCInterface::On)
        .getResult(0);
  }

private:
  static constexpr unsigned kNumStaticParams = 8;
  static constexpr unsigned kNumDynamicParams = 2;
  static constexpr unsigned kNumParams = kNumStaticParams + kNumDynamicParams;
  static constexpr unsigned kParamDimSizes = 0;
  static constexpr unsigned kParamLvlSizes = 1;
  static constexpr unsigned kParamLvlTypes = 2;
  static constexpr unsigned kParamLvlToDim = 3;
  static constexpr unsigned kParamDimToLvl = 4;
  static constexpr unsigned kParamPosTp = 5;
  static constexpr unsigned kParamCrdTp = 6;
  static constexpr unsigned kParamValTp = 7;
  static constexpr unsigned kParamAction = 8;
  static constexpr unsigned kParamPtr = 9;

  OpBuilder &builder;
  Location loc;
  Type pTp;
  Value params[kNumParams];
};

// TODO: see the note at `_mlir_ciface_newSparseTensor` about how
// the meaning of the various arguments (e.g., "sizes" vs "shapes")
// is inconsistent between the different actions.
NewCallParams &NewCallParams::genBuffers(SparseTensorType stt,
                                         ValueRange dimSizes) {
  const Level lvlRank = stt.getLvlRank();
  const Dimension dimRank = stt.getDimRank();
  // Sparsity annotations.
  params[kParamLvlTypes] = genLvlTypesBuffer(builder, loc, stt);
  // Dimension-sizes array of the enveloping tensor.  Useful for either
  // verification of external data, or for construction of internal data.
  assert(dimSizes.size() == static_cast<size_t>(dimRank) &&
         "Dimension-rank mismatch");
  params[kParamDimSizes] = allocaBuffer(builder, loc, dimSizes);
  // The level-sizes array must be passed as well, since for arbitrary
  // dimToLvl mappings it cannot be trivially reconstructed at runtime.
  // For now however, since we're still assuming permutations, we will
  // initialize this parameter alongside the `dimToLvl` and `lvlToDim`
  // parameters below.  We preinitialize `lvlSizes` for code symmetry.
  SmallVector<Value> lvlSizes(lvlRank);
  // The dimension-to-level mapping and its inverse.  We must preinitialize
  // `dimToLvl` so that the true branch below can perform random-access
  // `operator[]` assignment.  We preinitialize `lvlToDim` for code symmetry.
  SmallVector<Value> dimToLvl(dimRank);
  SmallVector<Value> lvlToDim(lvlRank);
  if (!stt.isIdentity()) {
    const auto dimToLvlMap = stt.getDimToLvl();
    assert(dimToLvlMap.isPermutation());
    for (Level l = 0; l < lvlRank; l++) {
      // The `d`th source variable occurs in the `l`th result position.
      const Dimension d = dimToLvlMap.getDimPosition(l);
      dimToLvl[d] = constantIndex(builder, loc, l);
      lvlToDim[l] = constantIndex(builder, loc, d);
      lvlSizes[l] = dimSizes[d];
    }
  } else {
    // The `SparseTensorType` ctor already ensures `dimRank == lvlRank`
    // when `isIdentity`; so no need to re-assert it here.
    for (Level l = 0; l < lvlRank; l++) {
      dimToLvl[l] = lvlToDim[l] = constantIndex(builder, loc, l);
      lvlSizes[l] = dimSizes[l];
    }
  }
  params[kParamLvlSizes] = allocaBuffer(builder, loc, lvlSizes);
  params[kParamLvlToDim] = allocaBuffer(builder, loc, lvlToDim);
  params[kParamDimToLvl] = stt.isIdentity()
                               ? params[kParamLvlToDim]
                               : allocaBuffer(builder, loc, dimToLvl);
  // Secondary and primary types encoding.
  setTemplateTypes(stt);
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
                          Value lvlCOO, Value valPtr, Value dimCoords,
                          Value dimToLvl) {
  SmallString<9> name{"addElt", primaryTypeFunctionSuffix(eltType)};
  SmallVector<Value, 4> params{lvlCOO, valPtr, dimCoords, dimToLvl};
  Type pTp = getOpaquePointerType(builder);
  createFuncCall(builder, loc, name, pTp, params, EmitCInterface::On);
}

/// Generates a call to `iter->getNext()`.  If there is a next element,
/// then it is copied into the out-parameters `coords` and `elemPtr`,
/// and the return value is true.  If there isn't a next element, then
/// the return value is false.
///
/// The `coords` argument uses the same coordinate-space as the `iter`
/// (which can be either dim- or lvl-coords, depending on context).
static Value genGetNextCall(OpBuilder &builder, Location loc, Value iter,
                            Value coords, Value elemPtr) {
  Type elemTp = cast<ShapedType>(elemPtr.getType()).getElementType();
  SmallString<10> name{"getNext", primaryTypeFunctionSuffix(elemTp)};
  SmallVector<Value, 3> params{iter, coords, elemPtr};
  Type i1 = builder.getI1Type();
  return createFuncCall(builder, loc, name, i1, params, EmitCInterface::On)
      .getResult(0);
}

/// Loads the value stored in `elemPtr`, and stores it at the coordinates
/// `cvs` into a dense tensor created by `allocDenseTensor`.
static void insertScalarIntoDenseTensor(OpBuilder &builder, Location loc,
                                        Value elemPtr, Value tensor,
                                        ValueRange cvs) {
  Value elemV = builder.create<memref::LoadOp>(loc, elemPtr);
  builder.create<memref::StoreOp>(loc, elemV, tensor, cvs);
}

/// Determine if the runtime library supports direct conversion to the
/// given target `dimTypes`.
static bool canUseDirectConversion(ArrayRef<DimLevelType> dimTypes) {
  bool alreadyCompressed = false;
  for (const auto dlt : dimTypes) {
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

/// Helper method to translate coordinates during a reshaping operation.
/// TODO: provide as general utility to MLIR at large?
static void reshapeCoords(Location loc, OpBuilder &builder,
                          ArrayRef<ReassociationIndices> reassociation,
                          ValueRange srcSizes, Value srcCoords,
                          ValueRange dstSizes, Value dstCoords) {
  const auto srcCvs = loadAll(builder, loc, srcSizes.size(), srcCoords);
  SmallVector<Value> dstCvs;
  reshapeCvs(builder, loc, reassociation, srcSizes, srcCvs, dstSizes, dstCvs);
  assert(dstCvs.size() == dstSizes.size());
  storeAll(builder, loc, dstCoords, dstCvs);
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
///     coo->add(reshape(elem.coords), elem.value)
///   }
///   s = newSparseTensor(coo)
template <typename ReshapeOp>
static LogicalResult
genSparse2SparseReshape(ReshapeOp op, typename ReshapeOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  const auto srcTp = getSparseTensorType(op.getSrc());
  const auto dstTp = getSparseTensorType(op.getResult());
  if (!srcTp.hasEncoding() || !dstTp.hasEncoding())
    return failure();
  Type elemTp = srcTp.getElementType();
  assert(elemTp == dstTp.getElementType() &&
         "reshape should not change element type");
  // Start an iterator over the source tensor (in coordinate order).
  SmallVector<Value> srcDimSizes =
      getDimSizes(rewriter, loc, srcTp, adaptor.getSrc());
  NewCallParams params(rewriter, loc);
  Value iter = params.genBuffers(srcTp.withoutDimToLvl(), srcDimSizes)
                   .genNewCall(Action::kToIterator, adaptor.getSrc());
  // Start a new COO for the destination tensor.
  SmallVector<Value> dstDimSizes;
  if (dstTp.hasStaticDimShape())
    // Static "shapes" are in fact "sizes".
    fillDimShape(rewriter, loc, dstTp, dstDimSizes);
  else
    genReshapeDstShape(rewriter, loc, dstDimSizes, srcDimSizes,
                       dstTp.getDimShape(), op.getReassociationIndices());
  const Value coo =
      params.genBuffers(dstTp, dstDimSizes).genNewCall(Action::kEmptyCOO);
  const Value dstDimToLvl = params.getDimToLvl();
  // Construct a while loop over the iterator.
  const Type iTp = rewriter.getIndexType();
  const Value srcDimCoords = genAlloca(rewriter, loc, srcTp.getDimRank(), iTp);
  const Value dstDimCoords = genAlloca(rewriter, loc, dstTp.getDimRank(), iTp);
  const Value elemPtr = genAllocaScalar(rewriter, loc, elemTp);
  const SmallVector<Value> noArgs;
  const SmallVector<Type> noTypes;
  auto whileOp = rewriter.create<scf::WhileOp>(loc, noTypes, noArgs);
  Block *before = rewriter.createBlock(&whileOp.getBefore(), {}, noTypes);
  rewriter.setInsertionPointToEnd(before);
  Value cond = genGetNextCall(rewriter, loc, iter, srcDimCoords, elemPtr);
  rewriter.create<scf::ConditionOp>(loc, cond, before->getArguments());
  // Translate coordinates from source to target and insert. Note that we do
  // not need to store the value in elemPtr, as the value is still there.
  Block *after = rewriter.createBlock(&whileOp.getAfter(), {}, noTypes);
  rewriter.setInsertionPointToStart(after);
  // We probably don't need these assertions, but better safe than sorry.
  assert(srcTp.getDimRank() == srcDimSizes.size());
  assert(dstTp.getDimRank() == dstDimSizes.size());
  reshapeCoords(loc, rewriter, op.getReassociationIndices(), srcDimSizes,
                srcDimCoords, dstDimSizes, dstDimCoords);
  genAddEltCall(rewriter, loc, elemTp, coo, elemPtr, dstDimCoords, dstDimToLvl);
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
    SparseTensorType stt,
    function_ref<void(OpBuilder &, Location, Value, Value)> bodyBuilder) {
  assert(stt.hasEncoding() &&
         "Generating Sparse Tensor COO Loop on a Dense Tensor!");
  const Dimension dimRank = stt.getDimRank();
  const Type elemTp = stt.getElementType();

  // Start an iterator over the tensor (in coordinate order).
  const auto noPerm = stt.withoutDimToLvl();
  SmallVector<Value> dimSizes = getDimSizes(rewriter, loc, noPerm, t);
  Value iter = NewCallParams(rewriter, loc)
                   .genBuffers(noPerm, dimSizes)
                   .genNewCall(Action::kToIterator, t);

  // Construct a while loop over the iterator.
  const Type iTp = rewriter.getIndexType();
  Value srcDimCoords = genAlloca(rewriter, loc, dimRank, iTp);
  Value elemPtr = genAllocaScalar(rewriter, loc, elemTp);
  const SmallVector<Value> noArgs;
  const SmallVector<Type> noTypes;
  auto whileOp = rewriter.create<scf::WhileOp>(loc, noTypes, noArgs);
  Block *before = rewriter.createBlock(&whileOp.getBefore(), {}, noTypes);
  rewriter.setInsertionPointToEnd(before);
  Value cond = genGetNextCall(rewriter, loc, iter, srcDimCoords, elemPtr);
  rewriter.create<scf::ConditionOp>(loc, cond, before->getArguments());
  Block *after = rewriter.createBlock(&whileOp.getAfter(), {}, noTypes);
  rewriter.setInsertionPointToStart(after);

  const bool hasDenseDim =
      llvm::any_of(stt.getEncoding().getLvlTypes(), isDenseDLT);
  if (hasDenseDim) {
    Value elemV = rewriter.create<memref::LoadOp>(loc, elemPtr);
    Value isZero = genIsNonzero(rewriter, loc, elemV);
    scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, isZero, /*else*/ false);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }
  // Callback here to build loop body.
  bodyBuilder(rewriter, loc, srcDimCoords, elemPtr);

  // Exit the scope from the IfOp.
  if (hasDenseDim)
    rewriter.setInsertionPointToEnd(after);

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
    SparseTensorType stt,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  assert(!stt.hasEncoding() &&
         "Generating Dense Tensor Loop on a Sparse Tensor!");

  const Dimension dimRank = stt.getDimRank();
  Value zero = constantIndex(rewriter, loc, 0);
  Value one = constantIndex(rewriter, loc, 1);

  SmallVector<Value> lo;
  SmallVector<Value> hi;
  SmallVector<Value> st;

  // Fill out loop iteration information.
  for (Dimension d = 0; d < dimRank; d++) {
    lo.push_back(zero);
    hi.push_back(linalg::createOrFoldDimOp(rewriter, loc, t, d));
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

/// Sparse conversion rule for accessing dimension-sizes.
class SparseTensorToDimSizeConverter
    : public OpConversionPattern<tensor::DimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto stt = getSparseTensorType(op.getSource());
    // Only rewrite sparse DimOp.
    if (!stt.hasEncoding())
      return failure();
    // Only rewrite DimOp with constant index.
    std::optional<int64_t> dim = op.getConstantIndex();
    if (!dim)
      return failure();
    // Generate the call.
    Value src = adaptor.getOperands()[0];
    rewriter.replaceOp(
        op, createOrFoldDimCall(rewriter, op->getLoc(), stt, src, *dim));
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
    const auto stt = getSparseTensorType(op);
    if (!stt.hasEncoding())
      return failure();
    const Dimension dimRank = stt.getDimRank();
    const Level lvlRank = stt.getLvlRank();
    // Construct the dimShape.
    SmallVector<Value> dimShapeValues = getDimShape(rewriter, loc, stt);
    Value dimShapeBuffer = allocaBuffer(rewriter, loc, dimShapeValues);
    // Allocate `SparseTensorReader` and perform all initial setup that
    // does not depend on lvlSizes (nor dimToLvl, lvlToDim, etc).
    Type opaqueTp = getOpaquePointerType(rewriter);
    Value valTp =
        constantPrimaryTypeEncoding(rewriter, loc, stt.getElementType());
    Value reader =
        createFuncCall(rewriter, loc, "createCheckedSparseTensorReader",
                       opaqueTp,
                       {adaptor.getOperands()[0], dimShapeBuffer, valTp},
                       EmitCInterface::On)
            .getResult(0);
    // Construct the lvlSizes.  If the dimShape is static, then it's
    // identical to dimSizes: so we can compute lvlSizes entirely at
    // compile-time.  If dimShape is dynamic, then we'll need to generate
    // code for computing lvlSizes from the `reader`'s actual dimSizes.
    //
    // TODO: For now we're still assuming `dimToLvl` is a permutation.
    // But since we're computing lvlSizes here (rather than in the runtime),
    // we can easily generalize that simply by adjusting this code.
    //
    // FIXME: reduce redundancy vs `NewCallParams::genBuffers`.
    Value dimSizesBuffer;
    if (stt.hasDynamicDimShape()) {
      Type indexTp = rewriter.getIndexType();
      auto memTp = MemRefType::get({ShapedType::kDynamic}, indexTp);
      dimSizesBuffer =
          createFuncCall(rewriter, loc, "getSparseTensorReaderDimSizes", memTp,
                         reader, EmitCInterface::On)
              .getResult(0);
    }
    Value lvlSizesBuffer;
    Value lvlToDimBuffer;
    Value dimToLvlBuffer;
    if (!stt.isIdentity()) {
      const auto dimToLvl = stt.getDimToLvl();
      assert(dimToLvl.isPermutation() && "Got non-permutation");
      // We preinitialize `dimToLvlValues` since we need random-access writing.
      // And we preinitialize the others for stylistic consistency.
      SmallVector<Value> lvlSizeValues(lvlRank);
      SmallVector<Value> lvlToDimValues(lvlRank);
      SmallVector<Value> dimToLvlValues(dimRank);
      for (Level l = 0; l < lvlRank; l++) {
        // The `d`th source variable occurs in the `l`th result position.
        Dimension d = dimToLvl.getDimPosition(l);
        Value lvl = constantIndex(rewriter, loc, l);
        Value dim = constantIndex(rewriter, loc, d);
        dimToLvlValues[d] = lvl;
        lvlToDimValues[l] = dim;
        lvlSizeValues[l] =
            stt.isDynamicDim(d)
                ? rewriter.create<memref::LoadOp>(loc, dimSizesBuffer, dim)
                : dimShapeValues[d];
      }
      lvlSizesBuffer = allocaBuffer(rewriter, loc, lvlSizeValues);
      lvlToDimBuffer = allocaBuffer(rewriter, loc, lvlToDimValues);
      dimToLvlBuffer = allocaBuffer(rewriter, loc, dimToLvlValues);
    } else {
      // The `SparseTensorType` ctor already ensures `dimRank == lvlRank`
      // when `isIdentity`; so no need to re-assert it here.
      SmallVector<Value> iotaValues;
      iotaValues.reserve(lvlRank);
      for (Level l = 0; l < lvlRank; l++)
        iotaValues.push_back(constantIndex(rewriter, loc, l));
      lvlSizesBuffer = dimSizesBuffer ? dimSizesBuffer : dimShapeBuffer;
      dimToLvlBuffer = lvlToDimBuffer = allocaBuffer(rewriter, loc, iotaValues);
    }
    // Use the `reader` to parse the file.
    SmallVector<Value, 8> params{
        reader,
        lvlSizesBuffer,
        genLvlTypesBuffer(rewriter, loc, stt),
        lvlToDimBuffer,
        dimToLvlBuffer,
        constantPosTypeEncoding(rewriter, loc, stt.getEncoding()),
        constantCrdTypeEncoding(rewriter, loc, stt.getEncoding()),
        valTp};
    Value tensor = createFuncCall(rewriter, loc, "newSparseTensorFromReader",
                                  opaqueTp, params, EmitCInterface::On)
                       .getResult(0);
    // Free the memory for `reader`.
    createFuncCall(rewriter, loc, "delSparseTensorReader", {}, {reader},
                   EmitCInterface::Off);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

/// Sparse conversion rule for the alloc operator.
/// TODO(springerm): remove when bufferization.alloc_tensor is gone
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
    const auto stt = getSparseTensorType(op);
    if (!stt.hasEncoding())
      return failure();
    // Gather all dimension sizes as SSA values.
    const Dimension dimRank = stt.getDimRank();
    SmallVector<Value> dimSizes;
    dimSizes.reserve(dimRank);
    unsigned operandCtr = 0;
    for (Dimension d = 0; d < dimRank; ++d) {
      dimSizes.push_back(
          stt.isDynamicDim(d)
              ? adaptor.getOperands()[operandCtr++]
              : constantIndex(rewriter, loc, op.getStaticSize(d)));
    }
    // Generate the call to construct empty tensor. The sizes are
    // explicitly defined by the arguments to the alloc operator.
    rewriter.replaceOp(op, NewCallParams(rewriter, loc)
                               .genBuffers(stt, dimSizes)
                               .genNewCall(Action::kEmpty));
    return success();
  }
};

/// Sparse conversion rule for the empty tensor.
class SparseTensorEmptyConverter : public OpConversionPattern<tensor::EmptyOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    const auto stt = getSparseTensorType(op);
    if (!stt.hasEncoding())
      return failure();
    // Gather all dimension sizes as SSA values.
    const Dimension dimRank = stt.getDimRank();
    SmallVector<Value> dimSizes;
    dimSizes.reserve(dimRank);
    auto shape = op.getType().getShape();
    unsigned operandCtr = 0;
    for (Dimension d = 0; d < dimRank; ++d) {
      dimSizes.push_back(stt.isDynamicDim(d)
                             ? adaptor.getOperands()[operandCtr++]
                             : constantIndex(rewriter, loc, shape[d]));
    }
    // Generate the call to construct empty tensor. The sizes are
    // explicitly defined by the arguments to the alloc operator.
    rewriter.replaceOp(op, NewCallParams(rewriter, loc)
                               .genBuffers(stt, dimSizes)
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
    const Location loc = op->getLoc();
    const auto srcTp = getSparseTensorType(op.getSource());
    const auto dstTp = getSparseTensorType(op);
    if (!srcTp.hasEncoding() && !dstTp.hasEncoding())
      return failure();

    const Dimension dimRank = srcTp.getDimRank();
    const Type elemTp = srcTp.getElementType();
    const Value src = adaptor.getOperands()[0];
    if (srcTp.hasEncoding() && dstTp.hasEncoding()) {
      const auto srcEnc = srcTp.getEncoding();
      const auto dstEnc = dstTp.getEncoding();
      // This is a sparse => sparse conversion, which is handled as follows:
      //   t = src->toCOO();         ; src to COO in dst order
      //   dst = newSparseTensor(t)
      // Using the coordinate scheme as an intermediate does not always
      // yield the fastest conversion but avoids the need for a full
      // O(N^2) conversion matrix.
      if (dstEnc == srcEnc) {
        rewriter.replaceOp(op, adaptor.getOperands()); // hidden nop cast
        return success();
      }
      NewCallParams params(rewriter, loc);
      SmallVector<Value> dimSizes = getDimSizes(rewriter, loc, srcTp, src);
      bool useDirectConversion;
      switch (options.sparseToSparseStrategy) {
      case SparseToSparseConversionStrategy::kViaCOO:
        useDirectConversion = false;
        break;
      case SparseToSparseConversionStrategy::kDirect:
        useDirectConversion = true;
        assert(canUseDirectConversion(dstEnc.getLvlTypes()) &&
               "Unsupported target for direct sparse-to-sparse conversion");
        break;
      case SparseToSparseConversionStrategy::kAuto:
        useDirectConversion = canUseDirectConversion(dstEnc.getLvlTypes());
        break;
      }
      if (useDirectConversion) {
        rewriter.replaceOp(
            op, params.genBuffers(srcTp.withEncoding(dstEnc), dimSizes)
                    .genNewCall(Action::kSparseToSparse, src));
      } else { // use via-COO conversion.
        // Set up encoding with right mix of src and dst so that the two
        // method calls can share most parameters, while still providing
        // the correct sparsity information to either of them.
        const auto mixedEnc =
            dstEnc.withBitWidths(srcEnc.getPosWidth(), srcEnc.getCrdWidth());
        // TODO: This is the only place where `kToCOO` (or `kToIterator`)
        // is called with a non-identity permutation.  Is there any clean
        // way to push the permutation over to the `kFromCOO` side instead?
        Value coo = params.genBuffers(srcTp.withEncoding(mixedEnc), dimSizes)
                        .genNewCall(Action::kToCOO, src);
        Value dst = params.setTemplateTypes(srcTp.withEncoding(dstEnc))
                        .genNewCall(Action::kFromCOO, coo);
        genDelCOOCall(rewriter, loc, elemTp, coo);
        rewriter.replaceOp(op, dst);
      }
      return success();
    }
    if (srcTp.hasEncoding() && !dstTp.hasEncoding()) {
      const auto srcEnc = srcTp.getEncoding();
      // This is sparse => dense conversion, which is handled as follows:
      //   dst = new Tensor(0);
      //   iter = new SparseTensorIterator(src);
      //   while (elem = iter->getNext()) {
      //     dst[elem.coords] = elem.value;
      //   }
      //   delete iter;
      //
      // Fabricate a no-permutation encoding for NewCallParams
      // The position/coordinate types must be those of `src`.
      // The dimLevelTypes aren't actually used by Action::kToIterator.
      const auto dstEnc = SparseTensorEncodingAttr::get(
          op->getContext(),
          SmallVector<DimLevelType>(dimRank, DimLevelType::Dense), AffineMap(),
          AffineMap(), srcEnc.getPosWidth(), srcEnc.getCrdWidth());
      SmallVector<Value> dimSizes = getDimSizes(rewriter, loc, srcTp, src);
      Value iter = NewCallParams(rewriter, loc)
                       .genBuffers(dstTp.withEncoding(dstEnc), dimSizes)
                       .genNewCall(Action::kToIterator, src);
      const Type iTp = rewriter.getIndexType();
      Value dimCoords = genAlloca(rewriter, loc, dimRank, iTp);
      Value elemPtr = genAllocaScalar(rewriter, loc, elemTp);
      // TODO: Dense buffers should be allocated/deallocated via the callback
      // in BufferizationOptions.
      Value dst = allocDenseTensor(rewriter, loc, dstTp, dimSizes);
      const SmallVector<Value> noArgs;
      const SmallVector<Type> noTypes;
      auto whileOp = rewriter.create<scf::WhileOp>(loc, noTypes, noArgs);
      Block *before = rewriter.createBlock(&whileOp.getBefore(), {}, noTypes);
      rewriter.setInsertionPointToEnd(before);
      Value cond = genGetNextCall(rewriter, loc, iter, dimCoords, elemPtr);
      rewriter.create<scf::ConditionOp>(loc, cond, before->getArguments());
      Block *after = rewriter.createBlock(&whileOp.getAfter(), {}, noTypes);
      rewriter.setInsertionPointToStart(after);
      const auto dcvs = loadAll(rewriter, loc, dimRank, dimCoords);
      insertScalarIntoDenseTensor(rewriter, loc, elemPtr, dst, dcvs);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointAfter(whileOp);
      genDelIteratorCall(rewriter, loc, elemTp, iter);
      rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
          op, dstTp.getRankedTensorType(), dst);
      return success();
    }
    assert(!srcTp.hasEncoding() && dstTp.hasEncoding());
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
    //     [i1,..,ik] = coordinates[i]
    //     t->add(val, [i1,..,ik], [p1,..,pk])
    //
    // Note that the dense tensor traversal code is actually implemented
    // using MLIR IR to avoid having to expose too much low-level
    // memref traversal details to the runtime support library.
    // Also note that the code below only generates the "new" ops and
    // the loop-nest per se; whereas the entire body of the innermost
    // loop is generated by genAddElt().
    SmallVector<Value> dimSizes;
    sizesFromSrc(rewriter, dimSizes, loc, src);
    NewCallParams params(rewriter, loc);
    Value coo =
        params.genBuffers(dstTp, dimSizes).genNewCall(Action::kEmptyCOO);
    const Type iTp = rewriter.getIndexType();
    Value dimCoords = genAlloca(rewriter, loc, dimRank, iTp);
    Value dimToLvl = params.getDimToLvl();
    Value elemPtr = genAllocaScalar(rewriter, loc, elemTp);
    genDenseTensorOrSparseConstantIterLoop(
        rewriter, loc, src, dimRank,
        [&](OpBuilder &builder, Location loc, Value val, ValueRange dcvs) {
          assert(dcvs.size() == static_cast<size_t>(dimRank));
          storeAll(builder, loc, dimCoords, dcvs);
          builder.create<memref::StoreOp>(loc, val, elemPtr);
          genAddEltCall(builder, loc, elemTp, coo, elemPtr, dimCoords,
                        dimToLvl);
        });
    // Final call to construct sparse tensor storage.
    Value dst = params.genNewCall(Action::kFromCOO, coo);
    genDelCOOCall(rewriter, loc, elemTp, coo);
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
    if (!getSparseTensorType(op.getTensor()).hasEncoding())
      return failure();
    StringRef name = "delSparseTensor";
    createFuncCall(rewriter, op->getLoc(), name, {}, adaptor.getOperands(),
                   EmitCInterface::Off);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Sparse conversion rule for position accesses.
class SparseTensorToPositionsConverter
    : public OpConversionPattern<ToPositionsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToPositionsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resTp = op.getType();
    Type posTp = cast<ShapedType>(resTp).getElementType();
    SmallString<17> name{"sparsePositions", overheadTypeFunctionSuffix(posTp)};
    Value lvl = constantIndex(rewriter, op->getLoc(), op.getLevel());
    replaceOpWithFuncCall(rewriter, op, name, resTp, {adaptor.getTensor(), lvl},
                          EmitCInterface::On);
    return success();
  }
};

/// Sparse conversion rule for coordinate accesses.
class SparseTensorToCoordinatesConverter
    : public OpConversionPattern<ToCoordinatesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToCoordinatesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: use `SparseTensorType::getCrdType` instead.
    Type resType = op.getType();
    const Type crdTp = cast<ShapedType>(resType).getElementType();
    SmallString<19> name{"sparseCoordinates",
                         overheadTypeFunctionSuffix(crdTp)};
    Location loc = op->getLoc();
    Value lvl = constantIndex(rewriter, loc, op.getLevel());

    // The function returns a MemRef without a layout.
    MemRefType callRetType = get1DMemRefType(crdTp, false);
    SmallVector<Value> operands{adaptor.getTensor(), lvl};
    auto fn = getFunc(op->getParentOfType<ModuleOp>(), name, callRetType,
                      operands, EmitCInterface::On);
    Value callRet =
        rewriter.create<func::CallOp>(loc, callRetType, fn, operands)
            .getResult(0);

    // Cast the MemRef type to the type expected by the users, though these
    // two types should be compatible at runtime.
    if (resType != callRetType)
      callRet = rewriter.create<memref::CastOp>(loc, resType, callRet);
    rewriter.replaceOp(op, callRet);

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
    auto resType = cast<ShapedType>(op.getType());
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
    Type eltType = cast<ShapedType>(op.getTensor().getType()).getElementType();
    auto resTp = MemRefType::get({ShapedType::kDynamic}, eltType);
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
    // coordinate order. All values are passed by reference through stack
    // allocated memrefs.
    Location loc = op->getLoc();
    const auto stt = getSparseTensorType(op.getTensor());
    const auto elemTp = stt.getElementType();
    const Level lvlRank = stt.getLvlRank();
    auto lvlCoords = genAlloca(rewriter, loc, lvlRank, rewriter.getIndexType());
    auto vref = genAllocaScalar(rewriter, loc, elemTp);
    storeAll(rewriter, loc, lvlCoords, adaptor.getLvlCoords());
    rewriter.create<memref::StoreOp>(loc, adaptor.getValue(), vref);
    SmallString<12> name{"lexInsert", primaryTypeFunctionSuffix(elemTp)};
    createFuncCall(rewriter, loc, name, {},
                   {adaptor.getTensor(), lvlCoords, vref}, EmitCInterface::On);
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
    const auto srcTp = getSparseTensorType(op.getTensor());
    Type eltType = srcTp.getElementType();
    Type boolType = rewriter.getIntegerType(1);
    Type idxType = rewriter.getIndexType();
    // All initialization should be done on entry of the loop nest.
    rewriter.setInsertionPointAfter(op.getTensor().getDefiningOp());
    // Get the cardinality of valid coordinates for the innermost level.
    Value sz = createOrFoldLvlCall(rewriter, loc, srcTp, adaptor.getTensor(),
                                   srcTp.getLvlRank() - 1);
    // Allocate temporary buffers for values, filled-switch, and coordinates.
    // We do not use stack buffers for this, since the expanded size may
    // be rather large (as it envelops a single expanded dense dimension).
    Value values = genAlloc(rewriter, loc, sz, eltType);
    Value filled = genAlloc(rewriter, loc, sz, boolType);
    Value lastLvlCoordinates = genAlloc(rewriter, loc, sz, idxType);
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
    // Replace expansion op with these buffers and initial coordinate.
    assert(op.getNumResults() == 4);
    rewriter.replaceOp(op, {values, filled, lastLvlCoordinates, zero});
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
    const auto stt = getSparseTensorType(op.getTensor());
    const Type elemTp = stt.getElementType();
    const Level lvlRank = stt.getLvlRank();
    auto lvlCoords = genAlloca(rewriter, loc, lvlRank, rewriter.getIndexType());
    storeAll(rewriter, loc, lvlCoords, adaptor.getLvlCoords());
    SmallString<12> name{"expInsert", primaryTypeFunctionSuffix(elemTp)};
    createFuncCall(rewriter, loc, name, {},
                   {tensor, lvlCoords, values, filled, added, count},
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
    // (1). When output is sparse and not all dims are dense, and mix of inputs:
    //    a_sparse = concat (b_dense, c_sparse, ....)
    // =>
    //    coo_for_a = newSparseCOO(shapeOf(a))
    //    for i, j, k // dense input
    //      coo->add(adjustForOffset(i,j,k), b[i,j,k])
    //
    //    for elem in sparse_input
    //      coo->add(adjustForOffset(elem.coords), elem.value)
    //    ...
    //    a = newSparseTensor(coo_for_a)
    //    return a
    //
    // (2). When output is dense or annotated all dense, and mix of inputs:
    //    a_dense = concat (b_dense, c_sparse, ....)
    // =>
    //    a = malloc(shapeOf(a)) or newSparseAllDense(shapeOf(a))
    //    for i, j, k // dense input
    //      a[ adjustForOffset(i,j,k) ] = b[i,j,k]
    //
    //    for elem in sparse_input
    //      a[ adjustForOffset(elem.coords) ] = elem.value
    //    return a
    Location loc = op.getLoc();
    const auto dstTp = getSparseTensorType(op);
    const auto dstEnc = dstTp.getEncoding();
    const Type elemTp = dstTp.getElementType();
    const Dimension concatDim = op.getDimension();
    const Dimension dimRank = dstTp.getDimRank();

    Value dst;         // destination tensor
    Value dstDimToLvl; // destination tensor permutation (if sparse out)
    // A pointer to the value being inserted (if dense => sparse)
    Value elemPtr;
    // Memory that holds the dim-coords for destination tensor (if sparse out)
    Value dstDimCoords;
    // The offset applied to the dimension to be concated (starting from 0)
    Value offset = constantIndex(rewriter, loc, 0);

    SmallVector<Value> dimSizes;
    concatDimSizesFromInputs(rewriter, loc, dstTp, op.getInputs(), concatDim,
                             dimSizes);

    NewCallParams params(rewriter, loc);
    const bool allDense = dstTp.hasEncoding() && dstTp.isAllDense();
    Value dstTensor;
    if (dstTp.hasEncoding()) {
      // Start a new COO or an initialized annotated all dense sparse tensor.
      dst = params.genBuffers(dstTp, dimSizes)
                .genNewCall(allDense ? Action::kEmpty : Action::kEmptyCOO);
      dstDimCoords = genAlloca(rewriter, loc, dimRank, rewriter.getIndexType());
      if (allDense) {
        dstTensor = dst;
        // Get the values buffer for the sparse tensor and reshape it to the
        // corresponding dense tensor shape.
        dst = genValuesCall(rewriter, loc,
                            MemRefType::get({ShapedType::kDynamic}, elemTp),
                            {dst});
        // Pass the `dstDimCoords` buffer for `reshapeValuesToLevels`
        // to reuse for storing level-sizes (yes, "level-sizes").
        // This is safe to do because `dstTp` is a dense-tensor type,
        // and therefore lvlRank == dimRank.
        dst = reshapeValuesToLevels(rewriter, loc, dstEnc, dimSizes, dst,
                                    dstDimCoords);
      } else {
        dstDimToLvl = params.getDimToLvl();
        elemPtr = genAllocaScalar(rewriter, loc, elemTp);
      }
    } else {
      // TODO: Dense buffers should be allocated/deallocated via the callback
      // in BufferizationOptions.
      dst = allocDenseTensor(rewriter, loc, dstTp, dimSizes);
    }
    const Level lvlRank = dstTp.getLvlRank();
    const auto dcvs2lcvs = [&](ValueRange dcvs) -> SmallVector<Value> {
      SmallVector<Value> lcvs;
      lcvs.reserve(lvlRank);
      for (Level l = 0; l < lvlRank; l++)
        // FIXME: `toOrigDim` is deprecated
        lcvs.push_back(dcvs[toOrigDim(dstEnc, l)]);
      return lcvs;
    };
    for (const auto &it : llvm::zip(op.getInputs(), adaptor.getInputs())) {
      Value orignalOp = std::get<0>(it); // Input (with encoding) from Op
      Value adaptedOp = std::get<1>(it); // Input (type converted) from adaptor
      const auto srcTp = getSparseTensorType(orignalOp);
      if (srcTp.hasEncoding()) {
        genSparseCOOIterationLoop(
            rewriter, loc, adaptedOp, srcTp,
            [&](OpBuilder &builder, Location loc, Value dimCoords,
                Value elemPtr) -> void {
              const auto dcvs =
                  loadAll(builder, loc, dimRank, dimCoords, concatDim, offset);
              if (dstTp.hasEncoding() && !allDense) {
                // Case: sparse => sparse, except for annotated all dense.
                storeAll(builder, loc, dstDimCoords, dcvs);
                genAddEltCall(builder, loc, elemTp, dst, elemPtr, dstDimCoords,
                              dstDimToLvl);
              } else {
                // Case: sparse => dense, or annotated all dense.
                const auto lcvs = allDense ? dcvs2lcvs(dcvs) : dcvs;
                insertScalarIntoDenseTensor(builder, loc, elemPtr, dst, lcvs);
              }
            });
      } else {
        genDenseTensorIterationLoop(
            rewriter, loc, adaptedOp, srcTp,
            [&](OpBuilder &builder, Location loc, ValueRange dcvs) -> void {
              if (dstTp.hasEncoding() && !allDense) {
                // Case: dense => sparse, except for annotated all dense.
                assert(dcvs.size() == static_cast<size_t>(dimRank));
                storeAll(builder, loc, dstDimCoords, dcvs, concatDim, offset);
                Value val = genValueForDense(builder, loc, adaptedOp, dcvs);
                builder.create<memref::StoreOp>(loc, val, elemPtr);
                genAddEltCall(builder, loc, elemTp, dst, elemPtr, dstDimCoords,
                              dstDimToLvl);
              } else {
                // Case: dense => dense, or annotated all dense.
                Value val = genValueForDense(builder, loc, adaptedOp, dcvs);
                // Despite the name, this isn't actually level-cvs until
                // after the `dcvs2lcvs` call.
                SmallVector<Value> lcvs(dcvs);
                // Apply offset.
                lcvs[concatDim] =
                    builder.create<arith::AddIOp>(loc, lcvs[concatDim], offset);
                if (allDense)
                  lcvs = dcvs2lcvs(lcvs);
                builder.create<memref::StoreOp>(loc, val, dst, lcvs);
              }
            });
      }
      // Accumulate offset.
      // TODO: avoid calling sparseDimSize multiple times by caching the result!
      Value curDim =
          createOrFoldDimCall(rewriter, loc, srcTp, adaptedOp, concatDim);
      offset = rewriter.create<arith::AddIOp>(loc, offset, curDim);
    }
    if (!dstTp.hasEncoding()) {
      rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
          op, dstTp.getRankedTensorType(), dst);
    } else if (allDense) {
      rewriter.replaceOp(op, dstTensor);
    } else {
      // In sparse output case, the destination holds the COO.
      Value coo = dst;
      dst = params.genNewCall(Action::kFromCOO, coo);
      // Release resources.
      genDelCOOCall(rewriter, loc, elemTp, coo);
      rewriter.replaceOp(op, dst);
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
    const Location loc = op->getLoc();
    const auto srcTp = getSparseTensorType(op.getTensor());
    // Convert to default permuted COO.
    Value src = adaptor.getOperands()[0];
    SmallVector<Value> dimSizes = getDimSizes(rewriter, loc, srcTp, src);
    Value coo = NewCallParams(rewriter, loc)
                    .genBuffers(srcTp.withoutDimToLvl(), dimSizes)
                    .genNewCall(Action::kToCOO, src);
    // Then output the tensor to external file with coordinates in the
    // externally visible lexicographic coordinate order.  A sort is
    // required if the source was not in that order yet (note that the
    // sort can be dropped altogether if external format does not care
    // about the order at all, but here we assume it does).
    const Value sort = constantI1(rewriter, loc, !srcTp.isIdentity());
    SmallVector<Value, 3> outParams{coo, adaptor.getOperands()[1], sort};
    const Type elemTp = srcTp.getElementType();
    SmallString<18> name{"outSparseTensor", primaryTypeFunctionSuffix(elemTp)};
    createFuncCall(rewriter, loc, name, {}, outParams, EmitCInterface::Off);
    genDelCOOCall(rewriter, loc, elemTp, coo);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Sparse conversion rule for the sparse_tensor.pack operator.
class SparseTensorPackConverter : public OpConversionPattern<PackOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const Location loc = op->getLoc();
    const auto dstTp = getSparseTensorType(op.getResult());
    // PackOps always returns a static shaped tensor result.
    assert(dstTp.hasStaticDimShape());
    SmallVector<Value> dimSizes = getDimSizes(rewriter, loc, dstTp);
    Value dst =
        NewCallParams(rewriter, loc)
            .genBuffers(dstTp.withoutDimToLvl(), dimSizes)
            .genNewCall(Action::kPack,
                        genLvlPtrsBuffers(rewriter, loc, adaptor.getLevels(),
                                          adaptor.getValues()));
    rewriter.replaceOp(op, dst);
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
  patterns
      .add<SparseReturnConverter, SparseTensorToDimSizeConverter,
           SparseCastConverter, SparseTensorNewConverter,
           SparseReshapeConverter<tensor::ExpandShapeOp>,
           SparseReshapeConverter<tensor::CollapseShapeOp>,
           SparseTensorConcatConverter, SparseTensorAllocConverter,
           SparseTensorEmptyConverter, SparseTensorDeallocConverter,
           SparseTensorToPositionsConverter, SparseTensorToCoordinatesConverter,
           SparseTensorToValuesConverter, SparseNumberOfEntriesConverter,
           SparseTensorLoadConverter, SparseTensorInsertConverter,
           SparseTensorExpandConverter, SparseTensorCompressConverter,
           SparseTensorOutConverter, SparseTensorPackConverter>(
          typeConverter, patterns.getContext());
  patterns.add<SparseTensorConvertConverter>(typeConverter,
                                             patterns.getContext(), options);
}
