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

/// Replaces the `op` with a `CallOp` to the `getFunc()` function reference.
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
/// for materializing sparse tensors into the computation. This abstraction
/// reduces the need for modifications when the API changes.
class NewCallParams final {
public:
  /// Allocates the `ValueRange` for the `func::CallOp` parameters.
  NewCallParams(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc), pTp(getOpaquePointerType(builder)) {}

  /// Initializes all static parameters (i.e., those which indicate
  /// type-level information such as the encoding and sizes), generating
  /// MLIR buffers as needed, and returning `this` for method chaining.
  NewCallParams &genBuffers(SparseTensorType stt,
                            ArrayRef<Value> dimSizesValues) {
    assert(dimSizesValues.size() == static_cast<size_t>(stt.getDimRank()));
    // Sparsity annotations.
    params[kParamLvlTypes] = genLvlTypesBuffer(builder, loc, stt);
    // Construct dimSizes, lvlSizes, dim2lvl, and lvl2dim buffers.
    params[kParamDimSizes] = allocaBuffer(builder, loc, dimSizesValues);
    params[kParamLvlSizes] =
        genMapBuffers(builder, loc, stt, dimSizesValues, params[kParamDimSizes],
                      params[kParamDim2Lvl], params[kParamLvl2Dim]);
    // Secondary and primary types encoding.
    setTemplateTypes(stt);
    // Finally, make note that initialization is complete.
    assert(isInitialized() && "Initialization failed");
    // And return `this` for method chaining.
    return *this;
  }

  /// (Re)sets the C++ template type parameters, and returns `this`
  /// for method chaining. This is already done as part of `genBuffers`,
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
  static constexpr unsigned kParamDim2Lvl = 3;
  static constexpr unsigned kParamLvl2Dim = 4;
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

/// Generates a call with a forwarding insertion.
static void genForwardingInsert(OpBuilder &builder, Location loc, Type eltType,
                                Value tensor, Value valPtr, Value dimCoords) {
  SmallString<9> name{"forwardingInsert", primaryTypeFunctionSuffix(eltType)};
  SmallVector<Value, 4> params{tensor, valPtr, dimCoords};
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
    // Construct the reader opening method calls.
    SmallVector<Value> dimShapesValues;
    Value dimSizesBuffer;
    Value reader = genReader(rewriter, loc, stt, adaptor.getOperands()[0],
                             dimShapesValues, dimSizesBuffer);
    // Now construct the lvlSizes, dim2lvl, and lvl2dim buffers.
    Value dim2lvlBuffer;
    Value lvl2dimBuffer;
    Value lvlSizesBuffer =
        genMapBuffers(rewriter, loc, stt, dimShapesValues, dimSizesBuffer,
                      dim2lvlBuffer, lvl2dimBuffer);
    // Use the `reader` to parse the file.
    Type opaqueTp = getOpaquePointerType(rewriter);
    Type eltTp = stt.getElementType();
    Value valTp = constantPrimaryTypeEncoding(rewriter, loc, eltTp);
    SmallVector<Value, 8> params{
        reader,
        lvlSizesBuffer,
        genLvlTypesBuffer(rewriter, loc, stt),
        dim2lvlBuffer,
        lvl2dimBuffer,
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
    // This is a "dense => sparse conversion" or a "sparse constant => sparse
    // conversion" which is conceptually handled as follows, with an additional
    // test for nonzero values for the dense case.
    //
    //   st = newSparseTensor()                  ; ST with forwarding COO
    //   for i1 in dim1, ..., ik in dimk         ; loop nest or range(NNZ) loop
    //     val = a[i1,..,ik]                     ;
    //     st->insertForwarding(val, [i1,..,ik]) ; maps dim to level
    //   st->endForwardingInsert()               ; finalize forwarding
    //
    // Note that the traversal code is actually implemented using MLIR IR to
    // avoid having to expose too much low-level memref traversal details to
    // the runtime support library.
    SmallVector<Value> dimSizes;
    sizesFromSrc(rewriter, dimSizes, loc, src);
    NewCallParams params(rewriter, loc);
    Value tensor =
        params.genBuffers(dstTp, dimSizes).genNewCall(Action::kEmptyForward);
    const Type iTp = rewriter.getIndexType();
    Value dimCoords = genAlloca(rewriter, loc, dimRank, iTp);
    Value elemPtr = genAllocaScalar(rewriter, loc, elemTp);
    genDenseTensorOrSparseConstantIterLoop(
        rewriter, loc, src, dimRank,
        [&](OpBuilder &builder, Location loc, Value val, ValueRange dcvs) {
          assert(dcvs.size() == static_cast<size_t>(dimRank));
          storeAll(builder, loc, dimCoords, dcvs);
          builder.create<memref::StoreOp>(loc, val, elemPtr);
          genForwardingInsert(builder, loc, elemTp, tensor, elemPtr, dimCoords);
        });
    // Final call to construct sparse tensor storage.
    StringRef name = "endForwardingInsert";
    createFuncCall(rewriter, op->getLoc(), name, {}, tensor,
                   EmitCInterface::Off);
    rewriter.replaceOp(op, tensor);
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
      StringRef name = "endLexInsert";
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
class SparseTensorAssembleConverter : public OpConversionPattern<AssembleOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AssembleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const Location loc = op->getLoc();
    const auto dstTp = getSparseTensorType(op.getResult());
    // AssembleOps always returns a static shaped tensor result.
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
  patterns.add<SparseReturnConverter, SparseTensorToDimSizeConverter,
               SparseCastConverter, SparseTensorNewConverter,
               SparseTensorAllocConverter, SparseTensorEmptyConverter,
               SparseTensorDeallocConverter, SparseTensorToPositionsConverter,
               SparseTensorToCoordinatesConverter,
               SparseTensorToValuesConverter, SparseNumberOfEntriesConverter,
               SparseTensorLoadConverter, SparseTensorInsertConverter,
               SparseTensorExpandConverter, SparseTensorCompressConverter,
               SparseTensorOutConverter, SparseTensorAssembleConverter>(
      typeConverter, patterns.getContext());
  patterns.add<SparseTensorConvertConverter>(typeConverter,
                                             patterns.getContext(), options);
}
