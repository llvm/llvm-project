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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

/// New tensor storage action. Keep these values consistent with
/// the sparse runtime support library.
enum Action : uint32_t {
  kEmpty = 0,
  kFromFile = 1,
  kFromCOO = 2,
  kEmptyCOO = 3,
  kToCOO = 4
};

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// Returns internal type encoding for primary storage. Keep these
/// values consistent with the sparse runtime support library.
static uint32_t getPrimaryTypeEncoding(Type tp) {
  if (tp.isF64())
    return 1;
  if (tp.isF32())
    return 2;
  if (tp.isInteger(64))
    return 3;
  if (tp.isInteger(32))
    return 4;
  if (tp.isInteger(16))
    return 5;
  if (tp.isInteger(8))
    return 6;
  return 0;
}

/// Returns internal type encoding for overhead storage. Keep these
/// values consistent with the sparse runtime support library.
static uint32_t getOverheadTypeEncoding(unsigned width) {
  switch (width) {
  default:
    return 1;
  case 32:
    return 2;
  case 16:
    return 3;
  case 8:
    return 4;
  }
}

/// Returns internal dimension level type encoding. Keep these
/// values consistent with the sparse runtime support library.
static uint32_t
getDimLevelTypeEncoding(SparseTensorEncodingAttr::DimLevelType dlt) {
  switch (dlt) {
  case SparseTensorEncodingAttr::DimLevelType::Dense:
    return 0;
  case SparseTensorEncodingAttr::DimLevelType::Compressed:
    return 1;
  case SparseTensorEncodingAttr::DimLevelType::Singleton:
    return 2;
  }
  llvm_unreachable("Unknown SparseTensorEncodingAttr::DimLevelType");
}

/// Generates a constant zero of the given type.
inline static Value constantZero(ConversionPatternRewriter &rewriter,
                                 Location loc, Type t) {
  return rewriter.create<arith::ConstantOp>(loc, t, rewriter.getZeroAttr(t));
}

/// Generates a constant of `index` type.
inline static Value constantIndex(ConversionPatternRewriter &rewriter,
                                  Location loc, int64_t i) {
  return rewriter.create<arith::ConstantIndexOp>(loc, i);
}

/// Generates a constant of `i32` type.
inline static Value constantI32(ConversionPatternRewriter &rewriter,
                                Location loc, int32_t i) {
  return rewriter.create<arith::ConstantIntOp>(loc, i, 32);
}

/// Generates a constant of `i8` type.
inline static Value constantI8(ConversionPatternRewriter &rewriter,
                               Location loc, int8_t i) {
  return rewriter.create<arith::ConstantIntOp>(loc, i, 8);
}

/// Returns a function reference (first hit also inserts into module). Sets
/// the "_emit_c_interface" on the function declaration when requested,
/// so that LLVM lowering generates a wrapper function that takes care
/// of ABI complications with passing in and returning MemRefs to C functions.
static FlatSymbolRefAttr getFunc(Operation *op, StringRef name,
                                 TypeRange resultType, ValueRange operands,
                                 bool emitCInterface = false) {
  MLIRContext *context = op->getContext();
  auto module = op->getParentOfType<ModuleOp>();
  auto result = SymbolRefAttr::get(context, name);
  auto func = module.lookupSymbol<FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<FuncOp>(
        op->getLoc(), name,
        FunctionType::get(context, operands.getTypes(), resultType));
    func.setPrivate();
    if (emitCInterface)
      func->setAttr("llvm.emit_c_interface", UnitAttr::get(context));
  }
  return result;
}

/// Generates dimension size call.
static Value genDimSizeCall(ConversionPatternRewriter &rewriter, Operation *op,
                            SparseTensorEncodingAttr &enc, Value src,
                            int64_t idx) {
  // Permute the index according to an optional dimension ordering.
  if (AffineMap p = enc.getDimOrdering())
    idx = p.getPermutedPosition(idx);
  // Generate the call.
  Location loc = op->getLoc();
  StringRef name = "sparseDimSize";
  SmallVector<Value, 2> params;
  params.push_back(src);
  params.push_back(constantIndex(rewriter, loc, idx));
  Type iTp = rewriter.getIndexType();
  auto fn = getFunc(op, name, iTp, params);
  return rewriter.create<CallOp>(loc, iTp, fn, params).getResult(0);
}

/// Generates a call into the "swiss army knife" method of the sparse runtime
/// support library for materializing sparse tensors into the computation.
static Value genNewCall(ConversionPatternRewriter &rewriter, Operation *op,
                        ArrayRef<Value> params) {
  Location loc = op->getLoc();
  StringRef name = "newSparseTensor";
  Type pTp = LLVM::LLVMPointerType::get(rewriter.getI8Type());
  auto fn = getFunc(op, name, pTp, params, /*emitCInterface=*/true);
  auto call = rewriter.create<CallOp>(loc, pTp, fn, params);
  return call.getResult(0);
}

/// Populates given sizes array from type.
static void sizesFromType(ConversionPatternRewriter &rewriter,
                          SmallVector<Value, 4> &sizes, Location loc,
                          ShapedType stp) {
  auto shape = stp.getShape();
  for (unsigned i = 0, rank = stp.getRank(); i < rank; i++) {
    uint64_t s = shape[i] == ShapedType::kDynamicSize ? 0 : shape[i];
    sizes.push_back(constantIndex(rewriter, loc, s));
  }
}

/// Populates given sizes array from source.
static void sizesFromSrc(ConversionPatternRewriter &rewriter,
                         SmallVector<Value, 4> &sizes, Location loc,
                         Value src) {
  ShapedType stp = src.getType().cast<ShapedType>();
  for (unsigned i = 0, rank = stp.getRank(); i < rank; i++)
    sizes.push_back(linalg::createOrFoldDimOp(rewriter, loc, src, i));
}

/// Populates given sizes array from type (for static sizes) and from
/// an already converted into opague pointer source (for dynamic sizes).
static void sizesFromPtr(ConversionPatternRewriter &rewriter,
                         SmallVector<Value, 4> &sizes, Operation *op,
                         SparseTensorEncodingAttr &enc, ShapedType stp,
                         Value src) {
  auto shape = stp.getShape();
  for (unsigned i = 0, rank = stp.getRank(); i < rank; i++)
    if (shape[i] == ShapedType::kDynamicSize)
      sizes.push_back(genDimSizeCall(rewriter, op, enc, src, i));
    else
      sizes.push_back(constantIndex(rewriter, op->getLoc(), shape[i]));
}

/// Generates a temporary buffer of the given size and type.
static Value genAlloca(ConversionPatternRewriter &rewriter, Location loc,
                       unsigned sz, Type tp) {
  auto memTp = MemRefType::get({ShapedType::kDynamicSize}, tp);
  Value a = constantIndex(rewriter, loc, sz);
  return rewriter.create<memref::AllocaOp>(loc, memTp, ValueRange{a});
}

/// Generates a temporary buffer of the given type and given contents.
static Value genBuffer(ConversionPatternRewriter &rewriter, Location loc,
                       ArrayRef<Value> values) {
  unsigned sz = values.size();
  assert(sz >= 1);
  Value buffer = genAlloca(rewriter, loc, sz, values[0].getType());
  for (unsigned i = 0; i < sz; i++) {
    Value idx = constantIndex(rewriter, loc, i);
    rewriter.create<memref::StoreOp>(loc, values[i], buffer, idx);
  }
  return buffer;
}

/// Populates parameters required to call the "swiss army knife" method of the
/// sparse runtime support library for materializing sparse tensors into the
/// computation.
static void newParams(ConversionPatternRewriter &rewriter,
                      SmallVector<Value, 8> &params, Operation *op,
                      SparseTensorEncodingAttr &enc, uint32_t action,
                      ValueRange szs, Value ptr = Value()) {
  Location loc = op->getLoc();
  ArrayRef<SparseTensorEncodingAttr::DimLevelType> dlt = enc.getDimLevelType();
  unsigned sz = dlt.size();
  // Sparsity annotations.
  SmallVector<Value, 4> attrs;
  for (unsigned i = 0; i < sz; i++)
    attrs.push_back(constantI8(rewriter, loc, getDimLevelTypeEncoding(dlt[i])));
  params.push_back(genBuffer(rewriter, loc, attrs));
  // Dimension sizes array of the enveloping tensor. Useful for either
  // verification of external data, or for construction of internal data.
  SmallVector<Value, 4> sizes;
  for (Value s : szs)
    sizes.push_back(s);
  params.push_back(genBuffer(rewriter, loc, sizes));
  // Dimension order permutation array. This is the "identity" permutation by
  // default, or otherwise the "reverse" permutation of a given ordering, so
  // that indices can be mapped quickly to the right position.
  SmallVector<Value, 4> rev(sz);
  if (AffineMap p = enc.getDimOrdering()) {
    for (unsigned i = 0; i < sz; i++)
      rev[p.getDimPosition(i)] = constantIndex(rewriter, loc, i);
  } else {
    for (unsigned i = 0; i < sz; i++)
      rev[i] = constantIndex(rewriter, loc, i);
  }
  params.push_back(genBuffer(rewriter, loc, rev));
  // Secondary and primary types encoding.
  ShapedType resType = op->getResult(0).getType().cast<ShapedType>();
  uint32_t secPtr = getOverheadTypeEncoding(enc.getPointerBitWidth());
  uint32_t secInd = getOverheadTypeEncoding(enc.getIndexBitWidth());
  uint32_t primary = getPrimaryTypeEncoding(resType.getElementType());
  assert(primary);
  params.push_back(constantI32(rewriter, loc, secPtr));
  params.push_back(constantI32(rewriter, loc, secInd));
  params.push_back(constantI32(rewriter, loc, primary));
  // User action and pointer.
  Type pTp = LLVM::LLVMPointerType::get(rewriter.getI8Type());
  if (!ptr)
    ptr = rewriter.create<LLVM::NullOp>(loc, pTp);
  params.push_back(constantI32(rewriter, loc, action));
  params.push_back(ptr);
}

/// Generates the comparison `v != 0` where `v` is of numeric type `t`.
/// For floating types, we use the "unordered" comparator (i.e., returns
/// true if `v` is NaN).
static Value genIsNonzero(ConversionPatternRewriter &rewriter, Location loc,
                          Value v) {
  Type t = v.getType();
  Value zero = constantZero(rewriter, loc, t);
  if (t.isa<FloatType>())
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, v,
                                          zero);
  if (t.isIntOrIndex())
    return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, v,
                                          zero);
  llvm_unreachable("Unknown element type");
}

/// Generates the code to read the value from tensor[ivs], and conditionally
/// stores the indices ivs to the memory in ind. The generated code looks like
/// the following and the insertion point after this routine is inside the
/// if-then branch behind the assignment to ind. This is to ensure that the
/// addEltX call generated after is inside the if-then branch.
///    if (tensor[ivs]!=0) {
///      ind = ivs
static Value genIndexAndValueForDense(ConversionPatternRewriter &rewriter,
                                      Location loc, Value tensor, Value ind,
                                      ValueRange ivs) {
  Value val = rewriter.create<tensor::ExtractOp>(loc, tensor, ivs);
  Value cond = genIsNonzero(rewriter, loc, val);
  scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, cond, /*else*/ false);
  rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
  unsigned i = 0;
  for (auto iv : ivs) {
    Value idx = constantIndex(rewriter, loc, i++);
    rewriter.create<memref::StoreOp>(loc, iv, ind, idx);
  }
  return val;
}

/// Generates a call that adds one element to a coordinate scheme.
/// In particular, this generates code like the following:
///   val = a[i1,..,ik];
///   if val != 0
///     t->add(val, [i1,..,ik], [p1,..,pk]);
static void genAddEltCall(ConversionPatternRewriter &rewriter, Operation *op,
                          Type eltType, Value ptr, Value val, Value ind,
                          Value perm) {
  Location loc = op->getLoc();
  StringRef name;
  if (eltType.isF64())
    name = "addEltF64";
  else if (eltType.isF32())
    name = "addEltF32";
  else if (eltType.isInteger(64))
    name = "addEltI64";
  else if (eltType.isInteger(32))
    name = "addEltI32";
  else if (eltType.isInteger(16))
    name = "addEltI16";
  else if (eltType.isInteger(8))
    name = "addEltI8";
  else
    llvm_unreachable("Unknown element type");
  SmallVector<Value, 8> params;
  params.push_back(ptr);
  params.push_back(val);
  params.push_back(ind);
  params.push_back(perm);
  Type pTp = LLVM::LLVMPointerType::get(rewriter.getI8Type());
  auto fn = getFunc(op, name, pTp, params, /*emitCInterface=*/true);
  rewriter.create<CallOp>(loc, pTp, fn, params);
}

/// If the tensor is a sparse constant, generates and returns the pair of
/// the constants for the indices and the values.
static Optional<std::pair<Value, Value>>
genSplitSparseConstant(ConversionPatternRewriter &rewriter, Location loc,
                       Value tensor) {
  if (auto constOp = tensor.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = constOp.value().dyn_cast<SparseElementsAttr>()) {
      DenseElementsAttr indicesAttr = attr.getIndices();
      Value indices = rewriter.create<arith::ConstantOp>(loc, indicesAttr);
      DenseElementsAttr valuesAttr = attr.getValues();
      Value values = rewriter.create<arith::ConstantOp>(loc, valuesAttr);
      return std::make_pair(indices, values);
    }
  }
  return {};
}

/// Generates the code to copy the index at indices[ivs] to ind, and return
/// the value at value[ivs].
static Value genIndexAndValueForSparse(ConversionPatternRewriter &rewriter,
                                       Location loc, Value indices,
                                       Value values, Value ind, ValueRange ivs,
                                       unsigned rank) {
  for (unsigned i = 0; i < rank; i++) {
    Value idx = constantIndex(rewriter, loc, i);
    Value val = rewriter.create<tensor::ExtractOp>(loc, indices,
                                                   ValueRange{ivs[0], idx});
    val =
        rewriter.create<arith::IndexCastOp>(loc, val, rewriter.getIndexType());
    rewriter.create<memref::StoreOp>(loc, val, ind, idx);
  }
  return rewriter.create<tensor::ExtractOp>(loc, values, ivs[0]);
}

//===----------------------------------------------------------------------===//
// Conversion rules.
//===----------------------------------------------------------------------===//

/// Sparse conversion rule for returns.
class SparseReturnConverter : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getOperands());
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
    auto enc = getSparseTensorEncoding(op.source().getType());
    if (!enc)
      return failure();
    Optional<int64_t> index = op.getConstantIndex();
    if (!index.hasValue())
      return failure();
    // Generate the call.
    Value src = adaptor.getOperands()[0];
    int64_t idx = index.getValue();
    rewriter.replaceOp(op, genDimSizeCall(rewriter, op, enc, src, idx));
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
    sizesFromType(rewriter, sizes, op.getLoc(), resType.cast<ShapedType>());
    Value ptr = adaptor.getOperands()[0];
    newParams(rewriter, params, op, enc, kFromFile, sizes, ptr);
    rewriter.replaceOp(op, genNewCall(rewriter, op, params));
    return success();
  }
};

/// Sparse conversion rule for the init operator.
class SparseTensorInitConverter : public OpConversionPattern<InitOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(InitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    auto enc = getSparseTensorEncoding(resType);
    if (!enc)
      return failure();
    // Generate the call to construct empty tensor. The sizes are
    // explicitly defined by the arguments to the init operator.
    SmallVector<Value, 8> params;
    newParams(rewriter, params, op, enc, kEmpty, adaptor.getOperands());
    rewriter.replaceOp(op, genNewCall(rewriter, op, params));
    return success();
  }
};

/// Sparse conversion rule for the convert operator.
class SparseTensorConvertConverter : public OpConversionPattern<ConvertOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type resType = op.getType();
    Type srcType = op.source().getType();
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
      SmallVector<Value, 4> sizes;
      SmallVector<Value, 8> params;
      sizesFromPtr(rewriter, sizes, op, encSrc, srcType.cast<ShapedType>(),
                   src);
      newParams(rewriter, params, op, encDst, kToCOO, sizes, src);
      Value coo = genNewCall(rewriter, op, params);
      params[6] = constantI32(rewriter, loc, kFromCOO);
      params[7] = coo;
      rewriter.replaceOp(op, genNewCall(rewriter, op, params));
      return success();
    }
    if (!encDst || encSrc) {
      // TODO: sparse => dense
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
    newParams(rewriter, params, op, encDst, kEmptyCOO, sizes);
    Value ptr = genNewCall(rewriter, op, params);
    Value ind = genAlloca(rewriter, loc, rank, rewriter.getIndexType());
    Value perm = params[2];
    SmallVector<Value> lo;
    SmallVector<Value> hi;
    SmallVector<Value> st;
    Value zero = constantIndex(rewriter, loc, 0);
    Value one = constantIndex(rewriter, loc, 1);
    auto indicesValues = genSplitSparseConstant(rewriter, loc, src);
    bool isCOOConstant = indicesValues.hasValue();
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
          genAddEltCall(rewriter, op, eltType, ptr, val, ind, perm);
          return {};
        });
    // Final call to construct sparse tensor storage.
    params[6] = constantI32(rewriter, loc, kFromCOO);
    params[7] = ptr;
    rewriter.replaceOp(op, genNewCall(rewriter, op, params));
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
    TypeRange none;
    auto fn = getFunc(op, name, none, adaptor.getOperands());
    rewriter.create<CallOp>(op.getLoc(), none, fn, adaptor.getOperands());
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
    Type eltType = resType.cast<ShapedType>().getElementType();
    StringRef name;
    if (eltType.isIndex())
      name = "sparsePointers";
    else if (eltType.isInteger(64))
      name = "sparsePointers64";
    else if (eltType.isInteger(32))
      name = "sparsePointers32";
    else if (eltType.isInteger(16))
      name = "sparsePointers16";
    else if (eltType.isInteger(8))
      name = "sparsePointers8";
    else
      return failure();
    auto fn = getFunc(op, name, resType, adaptor.getOperands(),
                      /*emitCInterface=*/true);
    rewriter.replaceOpWithNewOp<CallOp>(op, resType, fn, adaptor.getOperands());
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
    Type eltType = resType.cast<ShapedType>().getElementType();
    StringRef name;
    if (eltType.isIndex())
      name = "sparseIndices";
    else if (eltType.isInteger(64))
      name = "sparseIndices64";
    else if (eltType.isInteger(32))
      name = "sparseIndices32";
    else if (eltType.isInteger(16))
      name = "sparseIndices16";
    else if (eltType.isInteger(8))
      name = "sparseIndices8";
    else
      return failure();
    auto fn = getFunc(op, name, resType, adaptor.getOperands(),
                      /*emitCInterface=*/true);
    rewriter.replaceOpWithNewOp<CallOp>(op, resType, fn, adaptor.getOperands());
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
    StringRef name;
    if (eltType.isF64())
      name = "sparseValuesF64";
    else if (eltType.isF32())
      name = "sparseValuesF32";
    else if (eltType.isInteger(64))
      name = "sparseValuesI64";
    else if (eltType.isInteger(32))
      name = "sparseValuesI32";
    else if (eltType.isInteger(16))
      name = "sparseValuesI16";
    else if (eltType.isInteger(8))
      name = "sparseValuesI8";
    else
      return failure();
    auto fn = getFunc(op, name, resType, adaptor.getOperands(),
                      /*emitCInterface=*/true);
    rewriter.replaceOpWithNewOp<CallOp>(op, resType, fn, adaptor.getOperands());
    return success();
  }
};

/// Sparse conversion rule for tensor reconstruction.
class SparseTensorToTensorConverter : public OpConversionPattern<ToTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  // Simply fold the operator into the pointer to the sparse storage scheme.
  matchAndRewrite(ToTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check that all arguments of the tensor reconstruction operators are calls
    // into the support library that query exactly the same opaque pointer.
    Value ptr;
    for (Value op : adaptor.getOperands()) {
      if (auto call = op.getDefiningOp<CallOp>()) {
        Value arg = call.getOperand(0);
        if (!arg.getType().isa<LLVM::LLVMPointerType>())
          return failure();
        if (!ptr)
          ptr = arg;
        else if (arg != ptr)
          return failure();
      }
    }
    // If a single opaque pointer is found, perform the folding.
    if (!ptr)
      return failure();
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public method for populating conversion rules.
//===----------------------------------------------------------------------===//

/// Populates the given patterns list with conversion rules required for
/// the sparsification of linear algebra operations.
void mlir::populateSparseTensorConversionPatterns(TypeConverter &typeConverter,
                                                  RewritePatternSet &patterns) {
  patterns.add<SparseReturnConverter, SparseTensorToDimSizeConverter,
               SparseTensorNewConverter, SparseTensorInitConverter,
               SparseTensorConvertConverter, SparseTensorReleaseConverter,
               SparseTensorToPointersConverter, SparseTensorToIndicesConverter,
               SparseTensorToValuesConverter, SparseTensorToTensorConverter>(
      typeConverter, patterns.getContext());
}
