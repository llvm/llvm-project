//===- SparseTensorLowering.cpp - Sparse tensor primitives conversion -----===//
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

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// Returns internal type encoding for primary storage. Keep these
/// values consistent with the sparse runtime support library.
static unsigned getPrimaryTypeEncoding(Type tp) {
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
static unsigned getOverheadTypeEncoding(unsigned width) {
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
static unsigned
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

/// Returns integers of given width and values as a constant tensor.
/// We cast the static shape into a dynamic shape to ensure that the
/// method signature remains uniform accross different tensor dimensions.
static Value getTensor(ConversionPatternRewriter &rewriter, unsigned width,
                       Location loc, ArrayRef<APInt> values) {
  Type etp = rewriter.getIntegerType(width);
  unsigned sz = values.size();
  RankedTensorType tt1 = RankedTensorType::get({sz}, etp);
  RankedTensorType tt2 = RankedTensorType::get({ShapedType::kDynamicSize}, etp);
  auto elts =
      rewriter.create<ConstantOp>(loc, DenseElementsAttr::get(tt1, values));
  return rewriter.create<tensor::CastOp>(loc, tt2, elts);
}

/// Returns function reference (first hit also inserts into module).
static FlatSymbolRefAttr getFunc(Operation *op, StringRef name, Type resultType,
                                 ValueRange operands) {
  MLIRContext *context = op->getContext();
  auto module = op->getParentOfType<ModuleOp>();
  auto result = SymbolRefAttr::get(context, name);
  auto func = module.lookupSymbol<FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    moduleBuilder
        .create<FuncOp>(
            op->getLoc(), name,
            FunctionType::get(context, operands.getTypes(), resultType))
        .setPrivate();
  }
  return result;
}

/// Generates a call into the "swiss army knife" method of the sparse runtime
/// support library for materializing sparse tensors into the computation. The
/// method returns the call value and assigns the permutation to 'perm'.
static Value genNewCall(ConversionPatternRewriter &rewriter, Operation *op,
                        SparseTensorEncodingAttr &enc, uint32_t action,
                        Value &perm, Value ptr = Value()) {
  Location loc = op->getLoc();
  ShapedType resType = op->getResult(0).getType().cast<ShapedType>();
  SmallVector<Value, 8> params;
  // Sparsity annotations in tensor constant form.
  SmallVector<APInt, 4> attrs;
  unsigned sz = enc.getDimLevelType().size();
  for (unsigned i = 0; i < sz; i++)
    attrs.push_back(
        APInt(8, getDimLevelTypeEncoding(enc.getDimLevelType()[i])));
  params.push_back(getTensor(rewriter, 8, loc, attrs));
  // Dimension sizes array of the enveloping *dense* tensor. Useful for either
  // verification of external data, or for construction of internal data.
  auto shape = resType.getShape();
  SmallVector<APInt, 4> sizes;
  for (unsigned i = 0; i < sz; i++) {
    uint64_t s = shape[i] == ShapedType::kDynamicSize ? 0 : shape[i];
    sizes.push_back(APInt(64, s));
  }
  params.push_back(getTensor(rewriter, 64, loc, sizes));
  // Dimension order permutation array. This is the "identity" permutation by
  // default, or otherwise the "reverse" permutation of a given ordering, so
  // that indices can be mapped quickly to the right position.
  SmallVector<APInt, 4> rev(sz);
  if (AffineMap p = enc.getDimOrdering()) {
    for (unsigned i = 0; i < sz; i++)
      rev[p.getDimPosition(i)] = APInt(64, i);
  } else {
    for (unsigned i = 0; i < sz; i++)
      rev[i] = APInt(64, i);
  }
  perm = getTensor(rewriter, 64, loc, rev);
  params.push_back(perm);
  // Secondary and primary types encoding.
  unsigned secPtr = getOverheadTypeEncoding(enc.getPointerBitWidth());
  unsigned secInd = getOverheadTypeEncoding(enc.getIndexBitWidth());
  unsigned primary = getPrimaryTypeEncoding(resType.getElementType());
  assert(primary);
  params.push_back(
      rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(secPtr)));
  params.push_back(
      rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(secInd)));
  params.push_back(
      rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(primary)));
  // User action and pointer.
  Type pTp = LLVM::LLVMPointerType::get(IntegerType::get(op->getContext(), 8));
  if (!ptr)
    ptr = rewriter.create<LLVM::NullOp>(loc, pTp);
  params.push_back(
      rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(action)));
  params.push_back(ptr);
  // Generate the call to create new tensor.
  StringRef name = "newSparseTensor";
  auto call =
      rewriter.create<CallOp>(loc, pTp, getFunc(op, name, pTp, params), params);
  return call.getResult(0);
}

/// Generates a call that adds one element to a coordinate scheme.
static void genAddEltCall(ConversionPatternRewriter &rewriter, Operation *op,
                          Value ptr, Value tensor, Value ind, Value perm,
                          ValueRange ivs) {
  Location loc = op->getLoc();
  StringRef name;
  Type eltType = tensor.getType().cast<ShapedType>().getElementType();
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
  Value val = rewriter.create<tensor::ExtractOp>(loc, tensor, ivs);
  // TODO: add if here?
  unsigned i = 0;
  for (auto iv : ivs) {
    Value idx = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i++));
    rewriter.create<memref::StoreOp>(loc, iv, ind, idx);
  }
  SmallVector<Value, 8> params;
  params.push_back(ptr);
  params.push_back(val);
  params.push_back(ind);
  params.push_back(perm);
  Type pTp = LLVM::LLVMPointerType::get(IntegerType::get(op->getContext(), 8));
  rewriter.create<CallOp>(loc, pTp, getFunc(op, name, pTp, params), params);
}

//===----------------------------------------------------------------------===//
// Conversion rules.
//===----------------------------------------------------------------------===//

/// Sparse conversion rule for returns.
class SparseReturnConverter : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, operands);
    return success();
  }
};

/// Sparse conversion rule for dimension accesses.
class SparseTensorToDimSizeConverter
    : public OpConversionPattern<tensor::DimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::DimOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    auto enc = getSparseTensorEncoding(op.source().getType());
    if (!enc)
      return failure();
    // Permute the dim index.
    Optional<int64_t> index = op.getConstantIndex();
    if (!index.hasValue())
      return failure();
    int64_t idx = index.getValue();
    if (AffineMap p = enc.getDimOrdering())
      idx = p.getPermutedPosition(idx);
    // Generate the call.
    StringRef name = "sparseDimSize";
    SmallVector<Value, 2> params;
    params.push_back(operands[0]);
    params.push_back(
        rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(idx)));
    rewriter.replaceOpWithNewOp<CallOp>(
        op, resType, getFunc(op, name, resType, params), params);
    return success();
  }
};

/// Sparse conversion rule for the new operator.
class SparseTensorNewConverter : public OpConversionPattern<NewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NewOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    auto enc = getSparseTensorEncoding(resType);
    if (!enc)
      return failure();
    Value perm;
    rewriter.replaceOp(op, genNewCall(rewriter, op, enc, 0, perm, operands[0]));
    return success();
  }
};

/// Sparse conversion rule for the convert operator.
class SparseTensorConvertConverter : public OpConversionPattern<ConvertOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConvertOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    auto encDst = getSparseTensorEncoding(resType);
    auto encSrc = getSparseTensorEncoding(op.source().getType());
    if (encDst && encSrc) {
      // This is a sparse => sparse conversion, which is handled as follows:
      //   t = src->asCOO();         ; src to COO in dst order
      //   dst = newSparseTensor(t)
      // Using the coordinate scheme as an intermediate does not always
      // yield the fastest conversion but avoids the need for a full
      // O(N^2) conversion matrix.
      Value perm;
      Value coo = genNewCall(rewriter, op, encDst, 3, perm, operands[0]);
      rewriter.replaceOp(op, genNewCall(rewriter, op, encDst, 1, perm, coo));
      return success();
    }
    if (!encDst || encSrc) {
      // TODO: sparse => dense
      return failure();
    }
    // This is a dense => sparse conversion, which is handled as follows:
    //   t = newSparseCOO()
    //   for i1 in dim1
    //    ..
    //     for ik in dimk
    //       val = a[i1,..,ik]
    //       if val != 0
    //         t->add(val, [i1,..,ik], [p1,..,pk])
    //   s = newSparseTensor(t)
    // Note that the dense tensor traversal code is actually implemented
    // using MLIR IR to avoid having to expose too much low-level
    // memref traversal details to the runtime support library.
    Location loc = op->getLoc();
    ShapedType shape = resType.cast<ShapedType>();
    auto memTp =
        MemRefType::get({ShapedType::kDynamicSize}, rewriter.getIndexType());
    Value perm;
    Value ptr = genNewCall(rewriter, op, encDst, 2, perm);
    Value tensor = operands[0];
    Value arg = rewriter.create<ConstantOp>(
        loc, rewriter.getIndexAttr(shape.getRank()));
    Value ind = rewriter.create<memref::AllocaOp>(loc, memTp, ValueRange{arg});
    SmallVector<Value> lo;
    SmallVector<Value> hi;
    SmallVector<Value> st;
    Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
    Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
    for (unsigned i = 0, rank = shape.getRank(); i < rank; i++) {
      lo.push_back(zero);
      hi.push_back(linalg::createOrFoldDimOp(rewriter, loc, tensor, i));
      st.push_back(one);
    }
    scf::buildLoopNest(rewriter, op.getLoc(), lo, hi, st, {},
                       [&](OpBuilder &builder, Location loc, ValueRange ivs,
                           ValueRange args) -> scf::ValueVector {
                         genAddEltCall(rewriter, op, ptr, tensor, ind, perm,
                                       ivs);
                         return {};
                       });
    rewriter.replaceOp(op, genNewCall(rewriter, op, encDst, 1, perm, ptr));
    return success();
  }
};

/// Sparse conversion rule for pointer accesses.
class SparseTensorToPointersConverter
    : public OpConversionPattern<ToPointersOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToPointersOp op, ArrayRef<Value> operands,
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
    rewriter.replaceOpWithNewOp<CallOp>(
        op, resType, getFunc(op, name, resType, operands), operands);
    return success();
  }
};

/// Sparse conversion rule for index accesses.
class SparseTensorToIndicesConverter : public OpConversionPattern<ToIndicesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToIndicesOp op, ArrayRef<Value> operands,
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
    rewriter.replaceOpWithNewOp<CallOp>(
        op, resType, getFunc(op, name, resType, operands), operands);
    return success();
  }
};

/// Sparse conversion rule for value accesses.
class SparseTensorToValuesConverter : public OpConversionPattern<ToValuesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToValuesOp op, ArrayRef<Value> operands,
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
    rewriter.replaceOpWithNewOp<CallOp>(
        op, resType, getFunc(op, name, resType, operands), operands);
    return success();
  }
};

/// Sparse conversion rule for tensor reconstruction.
class SparseTensorToTensorConverter : public OpConversionPattern<ToTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  // Simply fold the operator into the pointer to the sparse storage scheme.
  matchAndRewrite(ToTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Check that all arguments of the tensor reconstruction operators are calls
    // into the support library that query exactly the same opaque pointer.
    Value ptr;
    for (Value op : operands) {
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
               SparseTensorNewConverter, SparseTensorConvertConverter,
               SparseTensorToPointersConverter, SparseTensorToIndicesConverter,
               SparseTensorToValuesConverter, SparseTensorToTensorConverter>(
      typeConverter, patterns.getContext());
}
