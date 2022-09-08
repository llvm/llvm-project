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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/SparseTensorUtils.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

/// Shorthand aliases for the `emitCInterface` argument to `getFunc()`,
/// `createFuncCall()`, and `replaceOpWithFuncCall()`.
enum class EmitCInterface : bool { Off = false, On = true };

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// Returns the equivalent of `void*` for opaque arguments to the
/// execution engine.
static Type getOpaquePointerType(OpBuilder &builder) {
  return LLVM::LLVMPointerType::get(builder.getI8Type());
}

/// Maps each sparse tensor type to an opaque pointer.
static Optional<Type> convertSparseTensorTypes(Type type) {
  if (getSparseTensorEncoding(type) != nullptr)
    return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
  return llvm::None;
}

/// Returns a function reference (first hit also inserts into module). Sets
/// the "_emit_c_interface" on the function declaration when requested,
/// so that LLVM lowering generates a wrapper function that takes care
/// of ABI complications with passing in and returning MemRefs to C functions.
static FlatSymbolRefAttr getFunc(ModuleOp module, StringRef name,
                                 TypeRange resultType, ValueRange operands,
                                 EmitCInterface emitCInterface) {
  MLIRContext *context = module.getContext();
  auto result = SymbolRefAttr::get(context, name);
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<func::FuncOp>(
        module.getLoc(), name,
        FunctionType::get(context, operands.getTypes(), resultType));
    func.setPrivate();
    if (static_cast<bool>(emitCInterface))
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));
  }
  return result;
}

/// Creates a `CallOp` to the function reference returned by `getFunc()` in
/// the builder's module.
static func::CallOp createFuncCall(OpBuilder &builder, Location loc,
                                   StringRef name, TypeRange resultType,
                                   ValueRange operands,
                                   EmitCInterface emitCInterface) {
  auto module = builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  auto fn = getFunc(module, name, resultType, operands, emitCInterface);
  return builder.create<func::CallOp>(loc, resultType, fn, operands);
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
                            int64_t idx) {
  // Permute the index according to an optional dimension ordering.
  if (AffineMap p = enc.getDimOrdering())
    idx = p.getPermutedPosition(idx);
  // Generate the call.
  StringRef name = "sparseDimSize";
  SmallVector<Value, 2> params{src, constantIndex(builder, loc, idx)};
  Type iTp = builder.getIndexType();
  return createFuncCall(builder, loc, name, iTp, params, EmitCInterface::Off)
      .getResult(0);
}

/// Generates a call into the "swiss army knife" method of the sparse runtime
/// support library for materializing sparse tensors into the computation.
static Value genNewCall(OpBuilder &builder, Location loc,
                        ArrayRef<Value> params) {
  StringRef name = "newSparseTensor";
  Type pTp = getOpaquePointerType(builder);
  return createFuncCall(builder, loc, name, pTp, params, EmitCInterface::On)
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

/// Populates given sizes array from source.
static void sizesFromSrc(OpBuilder &builder, SmallVector<Value, 4> &sizes,
                         Location loc, Value src) {
  unsigned rank = src.getType().cast<ShapedType>().getRank();
  for (unsigned i = 0; i < rank; i++)
    sizes.push_back(linalg::createOrFoldDimOp(builder, loc, src, i));
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

/// Generates an uninitialized temporary buffer of the given size and
/// type, but returns it as type `memref<? x $tp>` (rather than as type
/// `memref<$sz x $tp>`).
static Value genAlloca(OpBuilder &builder, Location loc, Value sz, Type tp) {
  auto memTp = MemRefType::get({ShapedType::kDynamicSize}, tp);
  return builder.create<memref::AllocaOp>(loc, memTp, ValueRange{sz});
}

/// Generates an uninitialized buffer of the given size and type,
/// but returns it as type `memref<? x $tp>` (rather than as type
/// `memref<$sz x $tp>`). Unlike temporary buffers on the stack,
/// this buffer must be explicitly deallocated by client.
static Value genAlloc(RewriterBase &rewriter, Location loc, Value sz, Type tp) {
  auto memTp = MemRefType::get({ShapedType::kDynamicSize}, tp);
  return rewriter.create<memref::AllocOp>(loc, memTp, ValueRange{sz});
}

/// Generates an uninitialized temporary buffer of the given size and
/// type, but returns it as type `memref<? x $tp>` (rather than as type
/// `memref<$sz x $tp>`).
static Value genAlloca(OpBuilder &builder, Location loc, unsigned sz, Type tp) {
  return genAlloca(builder, loc, constantIndex(builder, loc, sz), tp);
}

/// Generates an uninitialized temporary buffer with room for one value
/// of the given type, and returns the `memref<$tp>`.
static Value genAllocaScalar(OpBuilder &builder, Location loc, Type tp) {
  return builder.create<memref::AllocaOp>(loc, MemRefType::get({}, tp));
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

/// Populates parameters required to call the "swiss army knife" method of the
/// sparse runtime support library for materializing sparse tensors into the
/// computation.
static void newParams(OpBuilder &builder, SmallVector<Value, 8> &params,
                      Location loc, ShapedType stp,
                      SparseTensorEncodingAttr &enc, Action action,
                      ValueRange szs, Value ptr = Value()) {
  ArrayRef<SparseTensorEncodingAttr::DimLevelType> dlt = enc.getDimLevelType();
  unsigned sz = dlt.size();
  // Sparsity annotations.
  SmallVector<Value, 4> attrs;
  for (unsigned i = 0; i < sz; i++)
    attrs.push_back(constantDimLevelTypeEncoding(builder, loc, dlt[i]));
  params.push_back(genBuffer(builder, loc, attrs));
  // Dimension sizes array of the enveloping tensor. Useful for either
  // verification of external data, or for construction of internal data.
  params.push_back(genBuffer(builder, loc, szs));
  // Dimension order permutation array. This is the "identity" permutation by
  // default, or otherwise the "reverse" permutation of a given ordering, so
  // that indices can be mapped quickly to the right position.
  SmallVector<Value, 4> rev(sz);
  if (AffineMap p = enc.getDimOrdering()) {
    for (unsigned i = 0; i < sz; i++)
      rev[p.getDimPosition(i)] = constantIndex(builder, loc, i);
  } else {
    for (unsigned i = 0; i < sz; i++)
      rev[i] = constantIndex(builder, loc, i);
  }
  params.push_back(genBuffer(builder, loc, rev));
  // Secondary and primary types encoding.
  Type elemTp = stp.getElementType();
  params.push_back(constantPointerTypeEncoding(builder, loc, enc));
  params.push_back(constantIndexTypeEncoding(builder, loc, enc));
  params.push_back(constantPrimaryTypeEncoding(builder, loc, elemTp));
  // User action.
  params.push_back(constantAction(builder, loc, action));
  // Payload pointer.
  if (!ptr)
    ptr = builder.create<LLVM::NullOp>(loc, getOpaquePointerType(builder));
  params.push_back(ptr);
}

/// Generates the code to read the value from tensor[ivs].The generated code
/// looks like the following and the insertion point after this routine is
/// inside the if-then branch behind the assignment to ind.
///    if (tensor[ivs] != 0)
///      insert_point
static Value genValueForDense(OpBuilder &builder, Location loc, Value tensor,
                              ValueRange ivs) {
  Value val = builder.create<tensor::ExtractOp>(loc, tensor, ivs);
  Value cond = genIsNonzero(builder, loc, val);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, cond, /*else*/ false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return val;
}

/// Generates the code to read the value from tensor[ivs], and conditionally
/// stores the indices ivs to the memory in ind. The generated code looks like
/// the following and the insertion point after this routine is inside the
/// if-then branch behind the assignment to ind. This is to ensure that the
/// addEltX call generated after is inside the if-then branch.
///    if (tensor[ivs] != 0)
///      ind = ivs
static Value genIndexAndValueForDense(OpBuilder &builder, Location loc,
                                      Value tensor, Value ind, ValueRange ivs) {
  Value val = genValueForDense(builder, loc, tensor, ivs);
  unsigned i = 0;
  for (auto iv : ivs) {
    Value idx = constantIndex(builder, loc, i++);
    builder.create<memref::StoreOp>(loc, iv, ind, idx);
  }
  return val;
}

/// Generates a call to release/delete a `SparseTensorCOO`.
static void genDelCOOCall(OpBuilder &builder, Location loc, Type elemTp,
                          Value coo) {
  SmallString<21> name{"delSparseTensorCOO", primaryTypeFunctionSuffix(elemTp)};
  createFuncCall(builder, loc, name, {}, coo, EmitCInterface::Off);
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
/// the memory for `iter` is freed and the return value is false.
static Value genGetNextCall(OpBuilder &builder, Location loc, Value iter,
                            Value ind, Value elemPtr) {
  Type elemTp = elemPtr.getType().cast<ShapedType>().getElementType();
  SmallString<10> name{"getNext", primaryTypeFunctionSuffix(elemTp)};
  SmallVector<Value, 3> params{iter, ind, elemPtr};
  Type i1 = builder.getI1Type();
  return createFuncCall(builder, loc, name, i1, params, EmitCInterface::On)
      .getResult(0);
}

/// If the tensor is a sparse constant, generates and returns the pair of
/// the constants for the indices and the values.
static Optional<std::pair<Value, Value>>
genSplitSparseConstant(OpBuilder &builder, Location loc, Value tensor) {
  if (auto constOp = tensor.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = constOp.getValue().dyn_cast<SparseElementsAttr>()) {
      DenseElementsAttr indicesAttr = attr.getIndices();
      Value indices = builder.create<arith::ConstantOp>(loc, indicesAttr);
      DenseElementsAttr valuesAttr = attr.getValues();
      Value values = builder.create<arith::ConstantOp>(loc, valuesAttr);
      return std::make_pair(indices, values);
    }
  }
  return {};
}

/// Generates the code to copy the index at indices[ivs] to ind, and return
/// the value at value[ivs].
static Value genIndexAndValueForSparse(OpBuilder &builder, Location loc,
                                       Value indices, Value values, Value ind,
                                       ValueRange ivs, unsigned rank) {
  for (unsigned i = 0; i < rank; i++) {
    Value idx = constantIndex(builder, loc, i);
    Value val = builder.create<tensor::ExtractOp>(loc, indices,
                                                  ValueRange{ivs[0], idx});
    val = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), val);
    builder.create<memref::StoreOp>(loc, val, ind, idx);
  }
  return builder.create<tensor::ExtractOp>(loc, values, ivs[0]);
}

/// Generates code to allocate a buffer of the given type, and zero
/// initialize it.  If the buffer type has any dynamic sizes, then the
/// `sizes` parameter should be as filled by sizesFromPtr(); that way
/// we can reuse the genDimSizeCall() results generated by sizesFromPtr().
static Value allocDenseTensor(OpBuilder &builder, Location loc,
                              RankedTensorType tensorTp, ValueRange sizes) {
  Type elemTp = tensorTp.getElementType();
  auto shape = tensorTp.getShape();
  auto memTp = MemRefType::get(shape, elemTp);
  SmallVector<Value> dynamicSizes;
  for (unsigned i = 0, rank = tensorTp.getRank(); i < rank; i++) {
    if (shape[i] == ShapedType::kDynamicSize)
      dynamicSizes.push_back(sizes[i]);
  }
  Value mem = builder.create<memref::AllocOp>(loc, memTp, dynamicSizes);
  Value zero = constantZero(builder, loc, elemTp);
  builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{mem});
  return mem;
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
static bool canUseDirectConversion(
    ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimTypes) {
  bool alreadyCompressed = false;
  for (uint64_t rank = dimTypes.size(), r = 0; r < rank; r++) {
    switch (dimTypes[r]) {
    case SparseTensorEncodingAttr::DimLevelType::Compressed:
      if (alreadyCompressed)
        return false; // Multiple compressed dimensions not yet supported.
      alreadyCompressed = true;
      break;
    case SparseTensorEncodingAttr::DimLevelType::Dense:
      if (alreadyCompressed)
        return false; // Dense after Compressed not yet supported.
      break;
    default: // TODO: investigate
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
                             Value srcIdx) {
  unsigned dstRank = dstTp.getRank();
  unsigned srcRank = srcTp.getRank();
  unsigned start = 0;
  unsigned i = 0;
  bool isExpand = srcRank > dstRank;
  ArrayRef<int64_t> shape = isExpand ? srcTp.getShape() : dstTp.getShape();
  // Iterate over reassociation map.
  for (const auto &map : llvm::enumerate(reassociation)) {
    // Prepare strides information in dimension slice.
    uint64_t linear = 1;
    for (unsigned j = start, end = start + map.value().size(); j < end; j++) {
      assert(!ShapedType::isDynamic(shape[j]));
      linear *= shape[j];
    }
    // Start collapse.
    Value idx = constantIndex(rewriter, loc, i++);
    Value val;
    if (!isExpand)
      val = rewriter.create<memref::LoadOp>(loc, srcIdx, idx);
    // Iterate over dimension slice.
    for (unsigned j = start, end = start + map.value().size(); j < end; j++) {
      linear /= shape[j];
      Value stride = constantIndex(rewriter, loc, linear);
      Value jdx = constantIndex(rewriter, loc, j);
      if (isExpand) {
        Value old = rewriter.create<memref::LoadOp>(loc, srcIdx, jdx);
        Value mul = linear == 1
                        ? old
                        : rewriter.create<arith::MulIOp>(loc, old, stride);
        val = val ? rewriter.create<arith::AddIOp>(loc, val, mul) : mul;
      } else {
        Value old = val;
        if (linear != 1)
          val = rewriter.create<arith::DivUIOp>(loc, val, stride);
        rewriter.create<memref::StoreOp>(loc, val, dstIdx, jdx);
        if (linear != 1)
          val = rewriter.create<arith::RemUIOp>(loc, old, stride);
      }
    }
    // Finalize expansion.
    if (isExpand)
      rewriter.create<memref::StoreOp>(loc, val, dstIdx, idx);
    start += map.value().size();
  }
  // Sanity.
  assert((isExpand && i == dstRank) || (!isExpand && i == srcRank));
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
static LogicalResult
genSparse2SparseReshape(Operation *op, ConversionPatternRewriter &rewriter,
                        ArrayRef<ReassociationIndices> reassociation, Value src,
                        RankedTensorType dstTp, RankedTensorType srcTp) {
  Location loc = op->getLoc();
  auto encDst = getSparseTensorEncoding(dstTp);
  auto encSrc = getSparseTensorEncoding(srcTp);
  assert(encDst && encSrc);
  unsigned srcRank = srcTp.getRank();
  unsigned dstRank = dstTp.getRank();
  Type elemTp = srcTp.getElementType();
  assert(elemTp == dstTp.getElementType() &&
         "reshape should not change element type");
  // Start an iterator over the source tensor (in original index order).
  auto noPerm = SparseTensorEncodingAttr::get(
      op->getContext(), encSrc.getDimLevelType(), AffineMap(),
      encSrc.getPointerBitWidth(), encSrc.getIndexBitWidth());
  SmallVector<Value, 4> sizes;
  SmallVector<Value, 8> params;
  sizesFromPtr(rewriter, sizes, loc, noPerm, srcTp, src);
  newParams(rewriter, params, loc, srcTp, noPerm, Action::kToIterator, sizes,
            src);
  Value iter = genNewCall(rewriter, loc, params);
  // Start a new COO for the destination tensor.
  sizes.clear();
  params.clear();
  sizesFromPtr(rewriter, sizes, loc, encDst, dstTp, src);
  newParams(rewriter, params, loc, dstTp, encDst, Action::kEmptyCOO, sizes);
  Value coo = genNewCall(rewriter, loc, params);
  Value dstPerm = params[2];
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
  translateIndices(loc, rewriter, reassociation, dstTp, srcTp, dstIdx, srcIdx);
  genAddEltCall(rewriter, loc, elemTp, coo, elemPtr, dstIdx, dstPerm);
  rewriter.create<scf::YieldOp>(loc);
  // Final call to construct sparse tensor storage and free temporary resources.
  rewriter.setInsertionPointAfter(whileOp);
  params[6] = constantAction(rewriter, loc, Action::kFromCOO);
  params[7] = coo;
  Value dst = genNewCall(rewriter, loc, params);
  genDelCOOCall(rewriter, loc, elemTp, coo);
  genDelCOOCall(rewriter, loc, elemTp, iter);
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
      rewriter.getContext(), enc.getDimLevelType(), AffineMap(),
      enc.getPointerBitWidth(), enc.getIndexBitWidth());
  SmallVector<Value, 4> sizes;
  SmallVector<Value, 8> params;
  sizesFromPtr(rewriter, sizes, loc, noPerm, tensorTp, t);
  newParams(rewriter, params, loc, tensorTp, noPerm, Action::kToIterator, sizes,
            t);
  Value iter = genNewCall(rewriter, loc, params);

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
  genDelCOOCall(rewriter, loc, elemTp, iter);
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
    Type dstType = op.getResult().getType();
    Type srcType = op.getSrc().getType();
    auto encDst = getSparseTensorEncoding(dstType);
    auto encSrc = getSparseTensorEncoding(srcType);
    if (encDst && encSrc)
      return genSparse2SparseReshape(
          op, rewriter, op.getReassociationIndices(), adaptor.getOperands()[0],
          dstType.cast<RankedTensorType>(), srcType.cast<RankedTensorType>());
    return failure(); // handled elsewhere
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
    SmallVector<Value, 8> params;
    ShapedType stp = resType.cast<ShapedType>();
    sizesFromType(rewriter, sizes, loc, stp);
    Value ptr = adaptor.getOperands()[0];
    newParams(rewriter, params, loc, stp, enc, Action::kFromFile, sizes, ptr);
    rewriter.replaceOp(op, genNewCall(rewriter, loc, params));
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
    SmallVector<Value, 8> params;
    ShapedType stp = resType.cast<ShapedType>();
    newParams(rewriter, params, loc, stp, enc, Action::kEmpty, sizes);
    rewriter.replaceOp(op, genNewCall(rewriter, loc, params));
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
      SmallVector<Value, 8> params;
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
        newParams(rewriter, params, loc, stp, encDst, Action::kSparseToSparse,
                  sizes, src);
        rewriter.replaceOp(op, genNewCall(rewriter, loc, params));
      } else { // use via-COO conversion.
        // Set up encoding with right mix of src and dst so that the two
        // method calls can share most parameters, while still providing
        // the correct sparsity information to either of them.
        auto enc = SparseTensorEncodingAttr::get(
            op->getContext(), encDst.getDimLevelType(), encDst.getDimOrdering(),
            encSrc.getPointerBitWidth(), encSrc.getIndexBitWidth());
        newParams(rewriter, params, loc, stp, enc, Action::kToCOO, sizes, src);
        Value coo = genNewCall(rewriter, loc, params);
        params[3] = constantPointerTypeEncoding(rewriter, loc, encDst);
        params[4] = constantIndexTypeEncoding(rewriter, loc, encDst);
        params[6] = constantAction(rewriter, loc, Action::kFromCOO);
        params[7] = coo;
        Value dst = genNewCall(rewriter, loc, params);
        genDelCOOCall(rewriter, loc, stp.getElementType(), coo);
        rewriter.replaceOp(op, dst);
      }
      return success();
    }
    if (!encDst && encSrc) {
      // This is sparse => dense conversion, which is handled as follows:
      //   dst = new Tensor(0);
      //   iter = src->toCOO();
      //   iter->startIterator();
      //   while (elem = iter->getNext()) {
      //     dst[elem.indices] = elem.value;
      //   }
      RankedTensorType dstTensorTp = resType.cast<RankedTensorType>();
      RankedTensorType srcTensorTp = srcType.cast<RankedTensorType>();
      unsigned rank = dstTensorTp.getRank();
      Type elemTp = dstTensorTp.getElementType();
      // Fabricate a no-permutation encoding for newParams().
      // The pointer/index types must be those of `src`.
      // The dimLevelTypes aren't actually used by Action::kToIterator.
      encDst = SparseTensorEncodingAttr::get(
          op->getContext(),
          SmallVector<SparseTensorEncodingAttr::DimLevelType>(
              rank, SparseTensorEncodingAttr::DimLevelType::Dense),
          AffineMap(), encSrc.getPointerBitWidth(), encSrc.getIndexBitWidth());
      SmallVector<Value, 4> sizes;
      SmallVector<Value, 8> params;
      sizesFromPtr(rewriter, sizes, loc, encSrc, srcTensorTp, src);
      newParams(rewriter, params, loc, dstTensorTp, encDst, Action::kToIterator,
                sizes, src);
      Value iter = genNewCall(rewriter, loc, params);
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
      genDelCOOCall(rewriter, loc, elemTp, iter);
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
    SmallVector<Value, 8> params;
    sizesFromSrc(rewriter, sizes, loc, src);
    newParams(rewriter, params, loc, stp, encDst, Action::kEmptyCOO, sizes);
    Value coo = genNewCall(rewriter, loc, params);
    Value ind = genAlloca(rewriter, loc, rank, rewriter.getIndexType());
    Value perm = params[2];
    SmallVector<Value> lo;
    SmallVector<Value> hi;
    SmallVector<Value> st;
    Value zero = constantIndex(rewriter, loc, 0);
    Value one = constantIndex(rewriter, loc, 1);
    auto indicesValues = genSplitSparseConstant(rewriter, loc, src);
    bool isCOOConstant = indicesValues.has_value();
    Value indices;
    Value values;
    if (isCOOConstant) {
      indices = indicesValues->first;
      values = indicesValues->second;
      lo.push_back(zero);
      hi.push_back(linalg::createOrFoldDimOp(rewriter, loc, values, 0));
      st.push_back(one);
    } else {
      for (unsigned i = 0; i < rank; i++) {
        lo.push_back(zero);
        hi.push_back(linalg::createOrFoldDimOp(rewriter, loc, src, i));
        st.push_back(one);
      }
    }
    Type eltType = stp.getElementType();
    Value elemPtr = genAllocaScalar(rewriter, loc, eltType);
    scf::buildLoopNest(
        rewriter, op.getLoc(), lo, hi, st, {},
        [&](OpBuilder &builder, Location loc, ValueRange ivs,
            ValueRange args) -> scf::ValueVector {
          Value val;
          if (isCOOConstant)
            val = genIndexAndValueForSparse(rewriter, loc, indices, values, ind,
                                            ivs, rank);
          else
            val = genIndexAndValueForDense(rewriter, loc, src, ind, ivs);
          builder.create<memref::StoreOp>(loc, val, elemPtr);
          genAddEltCall(rewriter, loc, eltType, coo, elemPtr, ind, perm);
          return {};
        });
    // Final call to construct sparse tensor storage.
    params[6] = constantAction(rewriter, loc, Action::kFromCOO);
    params[7] = coo;
    Value dst = genNewCall(rewriter, loc, params);
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
    Type resType = op.getType();
    Type eltType = resType.cast<ShapedType>().getElementType();
    SmallString<15> name{"sparseValues", primaryTypeFunctionSuffix(eltType)};
    replaceOpWithFuncCall(rewriter, op, name, resType, adaptor.getOperands(),
                          EmitCInterface::On);
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
    // index order.
    Type elemTp = op.getTensor().getType().cast<ShapedType>().getElementType();
    SmallString<12> name{"lexInsert", primaryTypeFunctionSuffix(elemTp)};
    replaceOpWithFuncCall(rewriter, op, name, {}, adaptor.getOperands(),
                          EmitCInterface::On);
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
    Value sz = rewriter.create<tensor::DimOp>(loc, op.getTensor(), innerDim);
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
    Type elemTp = op.getTensor().getType().cast<ShapedType>().getElementType();
    SmallString<12> name{"expInsert", primaryTypeFunctionSuffix(elemTp)};
    replaceOpWithFuncCall(rewriter, op, name, {}, adaptor.getOperands(),
                          EmitCInterface::On);
    // Deallocate the buffers on exit of the loop nest.
    Operation *parent = op;
    for (; isa<scf::ForOp>(parent->getParentOp()) ||
           isa<scf::WhileOp>(parent->getParentOp()) ||
           isa<scf::ParallelOp>(parent->getParentOp()) ||
           isa<scf::IfOp>(parent->getParentOp());
         parent = parent->getParentOp())
      ;
    rewriter.setInsertionPointAfter(parent);
    rewriter.create<memref::DeallocOp>(loc, adaptor.getOperands()[2]);
    rewriter.create<memref::DeallocOp>(loc, adaptor.getOperands()[3]);
    rewriter.create<memref::DeallocOp>(loc, adaptor.getOperands()[4]);
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
    SmallVector<Value, 8> params;
    concatSizesFromInputs(rewriter, sizes, loc, dstTp, op.getInputs(),
                          concatDim);

    if (encDst) {
      // Start a new COO for the destination tensor.
      newParams(rewriter, params, loc, dstTp, encDst, Action::kEmptyCOO, sizes);
      dst = genNewCall(rewriter, loc, params);
      dstPerm = params[2];
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
      params[6] = constantAction(rewriter, loc, Action::kFromCOO);
      // In sparse output case, the destination holds the COO.
      Value coo = dst;
      params[7] = coo;
      dst = genNewCall(rewriter, loc, params);
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
    SmallVector<Value, 8> params;
    sizesFromPtr(rewriter, sizes, loc, encSrc, srcType, src);
    auto enc = SparseTensorEncodingAttr::get(
        op->getContext(), encSrc.getDimLevelType(), AffineMap(),
        encSrc.getPointerBitWidth(), encSrc.getIndexBitWidth());
    newParams(rewriter, params, loc, srcType, enc, Action::kToCOO, sizes, src);
    Value coo = genNewCall(rewriter, loc, params);
    // Then output the tensor to external file with indices in the externally
    // visible lexicographic index order. A sort is required if the source was
    // not in that order yet (note that the sort can be dropped altogether if
    // external format does not care about the order at all, but here we assume
    // it does).
    bool sort =
        encSrc.getDimOrdering() && !encSrc.getDimOrdering().isIdentity();
    params.clear();
    params.push_back(coo);
    params.push_back(adaptor.getOperands()[1]);
    params.push_back(constantI1(rewriter, loc, sort));
    Type eltType = srcType.getElementType();
    SmallString<18> name{"outSparseTensor", primaryTypeFunctionSuffix(eltType)};
    createFuncCall(rewriter, loc, name, {}, params, EmitCInterface::Off);
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
               SparseTensorLoadConverter, SparseTensorInsertConverter,
               SparseTensorExpandConverter, SparseTensorCompressConverter,
               SparseTensorOutConverter>(typeConverter, patterns.getContext());

  patterns.add<SparseTensorConvertConverter>(typeConverter,
                                             patterns.getContext(), options);
}
