//===- SparseTensorConversion.cpp - Sparse tensor primitives conversion ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert sparse tensor primitives to calls into a runtime support library.
// Note that this is a current implementation choice to keep the conversion
// simple. In principle, these primitives could also be converted to actual
// elaborate IR code that implements the primitives on the selected sparse
// tensor storage schemes.
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

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

/// Returns a function reference (first hit also inserts into module). Sets
/// the "_emit_c_interface" on the function declaration when requested,
/// so that LLVM lowering generates a wrapper function that takes care
/// of ABI complications with passing in and returning MemRefs to C functions.
static FlatSymbolRefAttr getFunc(Operation *op, StringRef name,
                                 TypeRange resultType, ValueRange operands,
                                 EmitCInterface emitCInterface) {
  MLIRContext *context = op->getContext();
  auto module = op->getParentOfType<ModuleOp>();
  auto result = SymbolRefAttr::get(context, name);
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<func::FuncOp>(
        op->getLoc(), name,
        FunctionType::get(context, operands.getTypes(), resultType));
    func.setPrivate();
    if (static_cast<bool>(emitCInterface))
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));
  }
  return result;
}

/// Creates a `CallOp` to the function reference returned by `getFunc()`.
static func::CallOp createFuncCall(OpBuilder &builder, Operation *op,
                                   StringRef name, TypeRange resultType,
                                   ValueRange operands,
                                   EmitCInterface emitCInterface) {
  auto fn = getFunc(op, name, resultType, operands, emitCInterface);
  return builder.create<func::CallOp>(op->getLoc(), resultType, fn, operands);
}

/// Replaces the `op` with  a `CallOp` to the function reference returned
/// by `getFunc()`.
static func::CallOp replaceOpWithFuncCall(RewriterBase &rewriter, Operation *op,
                                          StringRef name, TypeRange resultType,
                                          ValueRange operands,
                                          EmitCInterface emitCInterface) {
  auto fn = getFunc(op, name, resultType, operands, emitCInterface);
  return rewriter.replaceOpWithNewOp<func::CallOp>(op, resultType, fn,
                                                   operands);
}

/// Generates dimension size call.
static Value genDimSizeCall(OpBuilder &builder, Operation *op,
                            SparseTensorEncodingAttr &enc, Value src,
                            int64_t idx) {
  // Permute the index according to an optional dimension ordering.
  if (AffineMap p = enc.getDimOrdering())
    idx = p.getPermutedPosition(idx);
  // Generate the call.
  StringRef name = "sparseDimSize";
  SmallVector<Value, 2> params{src, constantIndex(builder, op->getLoc(), idx)};
  Type iTp = builder.getIndexType();
  return createFuncCall(builder, op, name, iTp, params, EmitCInterface::Off)
      .getResult(0);
}

/// Generates a call into the "swiss army knife" method of the sparse runtime
/// support library for materializing sparse tensors into the computation.
static Value genNewCall(OpBuilder &builder, Operation *op,
                        ArrayRef<Value> params) {
  StringRef name = "newSparseTensor";
  Type pTp = getOpaquePointerType(builder);
  return createFuncCall(builder, op, name, pTp, params, EmitCInterface::On)
      .getResult(0);
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

/// Populates given sizes array from type (for static sizes) and from
/// an already converted into opague pointer source (for dynamic sizes).
static void sizesFromPtr(OpBuilder &builder, SmallVector<Value, 4> &sizes,
                         Operation *op, SparseTensorEncodingAttr &enc,
                         ShapedType stp, Value src) {
  Location loc = op->getLoc();
  auto shape = stp.getShape();
  for (unsigned i = 0, rank = stp.getRank(); i < rank; i++)
    if (shape[i] == ShapedType::kDynamicSize)
      sizes.push_back(genDimSizeCall(builder, op, enc, src, i));
    else
      sizes.push_back(constantIndex(builder, loc, shape[i]));
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
                      Operation *op, ShapedType stp,
                      SparseTensorEncodingAttr &enc, Action action,
                      ValueRange szs, Value ptr = Value()) {
  Location loc = op->getLoc();
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

/// Generates the code to read the value from tensor[ivs], and conditionally
/// stores the indices ivs to the memory in ind. The generated code looks like
/// the following and the insertion point after this routine is inside the
/// if-then branch behind the assignment to ind. This is to ensure that the
/// addEltX call generated after is inside the if-then branch.
///    if (tensor[ivs]!=0) {
///      ind = ivs
static Value genIndexAndValueForDense(OpBuilder &builder, Location loc,
                                      Value tensor, Value ind, ValueRange ivs) {
  Value val = builder.create<tensor::ExtractOp>(loc, tensor, ivs);
  Value cond = genIsNonzero(builder, loc, val);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, cond, /*else*/ false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  unsigned i = 0;
  for (auto iv : ivs) {
    Value idx = constantIndex(builder, loc, i++);
    builder.create<memref::StoreOp>(loc, iv, ind, idx);
  }
  return val;
}

/// Generates a call to release/delete a `SparseTensorCOO`.
static void genDelCOOCall(OpBuilder &builder, Operation *op, Type elemTp,
                          Value coo) {
  SmallString<21> name{"delSparseTensorCOO", primaryTypeFunctionSuffix(elemTp)};
  TypeRange noTp;
  createFuncCall(builder, op, name, noTp, coo, EmitCInterface::Off);
}

/// Generates a call that adds one element to a coordinate scheme.
/// In particular, this generates code like the following:
///   val = a[i1,..,ik];
///   if val != 0
///     t->add(&val, [i1,..,ik], [p1,..,pk]);
static void genAddEltCall(OpBuilder &builder, Operation *op, Type eltType,
                          Value ptr, Value valPtr, Value ind, Value perm) {
  SmallString<9> name{"addElt", primaryTypeFunctionSuffix(eltType)};
  SmallVector<Value, 4> params{ptr, valPtr, ind, perm};
  Type pTp = getOpaquePointerType(builder);
  createFuncCall(builder, op, name, pTp, params, EmitCInterface::On);
}

/// Generates a call to `iter->getNext()`.  If there is a next element,
/// then it is copied into the out-parameters `ind` and `elemPtr`,
/// and the return value is true.  If there isn't a next element, then
/// the memory for `iter` is freed and the return value is false.
static Value genGetNextCall(OpBuilder &builder, Operation *op, Value iter,
                            Value ind, Value elemPtr) {
  Type elemTp = elemPtr.getType().cast<ShapedType>().getElementType();
  SmallString<10> name{"getNext", primaryTypeFunctionSuffix(elemTp)};
  SmallVector<Value, 3> params{iter, ind, elemPtr};
  Type i1 = builder.getI1Type();
  return createFuncCall(builder, op, name, i1, params, EmitCInterface::On)
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

/// Generates code to allocate a tensor of the given type, and zero
/// initialize it.  If the tensor type has any dynamic sizes, then the
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

/// Inserts the element returned by genGetNextCall(_, ind, elemPtr) into
/// the tensor created by allocDenseTensor().  The `rank` is the rank
/// of the `tensor` and the length of `ind`.
static void insertScalarIntoDenseTensor(OpBuilder &builder, Location loc,
                                        Value elemPtr, Value tensor,
                                        unsigned rank, Value ind) {
  SmallVector<Value, 4> ivs;
  ivs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) {
    Value idx = constantIndex(builder, loc, i);
    ivs.push_back(builder.create<memref::LoadOp>(loc, ind, idx));
  }
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
    case SparseTensorEncodingAttr::DimLevelType::Singleton:
      // Although Singleton isn't generally supported yet, the direct
      // conversion method doesn't have any particular problems with
      // singleton after compressed.
      break;
    }
  }
  return true;
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
    rewriter.replaceOp(op, genDimSizeCall(rewriter, op, enc, src, idx));
    return success();
  }
};

/// Sparse conversion rule for trivial tensor casts.
class SparseCastConverter : public OpConversionPattern<tensor::CastOp> {
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

/// Sparse conversion rule for the new operator.
class SparseTensorNewConverter : public OpConversionPattern<NewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    auto enc = getSparseTensorEncoding(resType);
    if (!enc)
      return failure();
    // Generate the call to construct tensor from ptr. The sizes are
    // inferred from the result type of the new operator.
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 8> params;
    ShapedType stp = resType.cast<ShapedType>();
    sizesFromType(rewriter, sizes, op.getLoc(), stp);
    Value ptr = adaptor.getOperands()[0];
    newParams(rewriter, params, op, stp, enc, Action::kFromFile, sizes, ptr);
    rewriter.replaceOp(op, genNewCall(rewriter, op, params));
    return success();
  }
};

/// Sparse conversion rule for the alloc operator.
class SparseTensorAllocConverter
    : public OpConversionPattern<bufferization::AllocTensorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(bufferization::AllocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
        sizes.push_back(rewriter.create<arith::ConstantIndexOp>(
            op.getLoc(), op.getStaticSize(i)));
      }
    }
    // Generate the call to construct empty tensor. The sizes are
    // explicitly defined by the arguments to the alloc operator.
    SmallVector<Value, 8> params;
    ShapedType stp = resType.cast<ShapedType>();
    newParams(rewriter, params, op, stp, enc, Action::kEmpty, sizes);
    rewriter.replaceOp(op, genNewCall(rewriter, op, params));
    return success();
  }
};

/// Sparse conversion rule for the convert operator.
class SparseTensorConvertConverter : public OpConversionPattern<ConvertOp> {
  /// Options to control sparse code generation.
  SparseTensorConversionOptions options;

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
      sizesFromPtr(rewriter, sizes, op, encSrc, stp, src);
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
        newParams(rewriter, params, op, stp, encDst, Action::kSparseToSparse,
                  sizes, src);
        rewriter.replaceOp(op, genNewCall(rewriter, op, params));
      } else { // use via-COO conversion.
        // Set up encoding with right mix of src and dst so that the two
        // method calls can share most parameters, while still providing
        // the correct sparsity information to either of them.
        auto enc = SparseTensorEncodingAttr::get(
            op->getContext(), encDst.getDimLevelType(), encDst.getDimOrdering(),
            encSrc.getPointerBitWidth(), encSrc.getIndexBitWidth());
        newParams(rewriter, params, op, stp, enc, Action::kToCOO, sizes, src);
        Value coo = genNewCall(rewriter, op, params);
        params[3] = constantPointerTypeEncoding(rewriter, loc, encDst);
        params[4] = constantIndexTypeEncoding(rewriter, loc, encDst);
        params[6] = constantAction(rewriter, loc, Action::kFromCOO);
        params[7] = coo;
        Value dst = genNewCall(rewriter, op, params);
        genDelCOOCall(rewriter, op, stp.getElementType(), coo);
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
      sizesFromPtr(rewriter, sizes, op, encSrc, srcTensorTp, src);
      newParams(rewriter, params, op, dstTensorTp, encDst, Action::kToIterator,
                sizes, src);
      Value iter = genNewCall(rewriter, op, params);
      Value ind = genAlloca(rewriter, loc, rank, rewriter.getIndexType());
      Value elemPtr = genAllocaScalar(rewriter, loc, elemTp);
      Value dst = allocDenseTensor(rewriter, loc, dstTensorTp, sizes);
      SmallVector<Value> noArgs;
      SmallVector<Type> noTypes;
      auto whileOp = rewriter.create<scf::WhileOp>(loc, noTypes, noArgs);
      Block *before = rewriter.createBlock(&whileOp.getBefore(), {}, noTypes);
      rewriter.setInsertionPointToEnd(before);
      Value cond = genGetNextCall(rewriter, op, iter, ind, elemPtr);
      rewriter.create<scf::ConditionOp>(loc, cond, before->getArguments());
      Block *after = rewriter.createBlock(&whileOp.getAfter(), {}, noTypes);
      rewriter.setInsertionPointToStart(after);
      insertScalarIntoDenseTensor(rewriter, loc, elemPtr, dst, rank, ind);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointAfter(whileOp);
      genDelCOOCall(rewriter, op, elemTp, iter);
      rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, resType, dst);
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
    newParams(rewriter, params, op, stp, encDst, Action::kEmptyCOO, sizes);
    Value coo = genNewCall(rewriter, op, params);
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
          genAddEltCall(rewriter, op, eltType, coo, elemPtr, ind, perm);
          return {};
        });
    // Final call to construct sparse tensor storage.
    params[6] = constantAction(rewriter, loc, Action::kFromCOO);
    params[7] = coo;
    Value dst = genNewCall(rewriter, op, params);
    genDelCOOCall(rewriter, op, eltType, coo);
    rewriter.replaceOp(op, dst);
    return success();
  }
};

/// Sparse conversion rule for the release operator.
class SparseTensorReleaseConverter : public OpConversionPattern<ReleaseOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReleaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef name = "delSparseTensor";
    TypeRange noTp;
    createFuncCall(rewriter, op, name, noTp, adaptor.getOperands(),
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
    replaceOpWithFuncCall(rewriter, op, name, resType, adaptor.getOperands(),
                          EmitCInterface::On);
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
    replaceOpWithFuncCall(rewriter, op, name, resType, adaptor.getOperands(),
                          EmitCInterface::On);
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
      TypeRange noTp;
      createFuncCall(rewriter, op, name, noTp, adaptor.getOperands(),
                     EmitCInterface::Off);
    }
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Sparse conversion rule for inserting in lexicographic index order.
class SparseTensorLexInsertConverter : public OpConversionPattern<LexInsertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LexInsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elemTp = op.getTensor().getType().cast<ShapedType>().getElementType();
    SmallString<12> name{"lexInsert", primaryTypeFunctionSuffix(elemTp)};
    TypeRange noTp;
    replaceOpWithFuncCall(rewriter, op, name, noTp, adaptor.getOperands(),
                          EmitCInterface::On);
    return success();
  }
};

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
    // Determine the size for access expansion.
    auto enc = getSparseTensorEncoding(srcType);
    Value src = adaptor.getOperands()[0];
    Value sz = genDimSizeCall(rewriter, op, enc, src, srcType.getRank() - 1);
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
    TypeRange noTp;
    replaceOpWithFuncCall(rewriter, op, name, noTp, adaptor.getOperands(),
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
    sizesFromPtr(rewriter, sizes, op, encSrc, srcType, src);
    auto enc = SparseTensorEncodingAttr::get(
        op->getContext(), encSrc.getDimLevelType(), AffineMap(),
        encSrc.getPointerBitWidth(), encSrc.getIndexBitWidth());
    newParams(rewriter, params, op, srcType, enc, Action::kToCOO, sizes, src);
    Value coo = genNewCall(rewriter, op, params);
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
    TypeRange noTp;
    createFuncCall(rewriter, op, name, noTp, params, EmitCInterface::Off);
    genDelCOOCall(rewriter, op, eltType, coo);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

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
               SparseTensorAllocConverter, SparseTensorReleaseConverter,
               SparseTensorToPointersConverter, SparseTensorToIndicesConverter,
               SparseTensorToValuesConverter, SparseTensorLoadConverter,
               SparseTensorLexInsertConverter, SparseTensorExpandConverter,
               SparseTensorCompressConverter, SparseTensorOutConverter>(
      typeConverter, patterns.getContext());
  patterns.add<SparseTensorConvertConverter>(typeConverter,
                                             patterns.getContext(), options);
}
