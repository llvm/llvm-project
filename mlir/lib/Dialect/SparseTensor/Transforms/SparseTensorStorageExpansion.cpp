//===- SparseTensorStorageExpansion.cpp - Sparse tensor storage expansion ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The sparse tensor storage expansion pass expands the compound storage for
// sparse tensors (using tuple) to flattened SSA values.
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

/// Expands sparse tensor storage tuple.
static Optional<LogicalResult>
convertSparseTensorStorageTuple(Type t, SmallVectorImpl<Type> &result) {
  if (auto tuple = t.dyn_cast<TupleType>()) {
    // Note that it does not handle nest tuples, but it is fine
    // for sparse compiler as they will not be generated.
    result.append(tuple.getTypes().begin(), tuple.getTypes().end());
    return success();
  }
  return llvm::None;
}

/// Flatten a list of operands that may contain tuples.
static void flattenOperands(ValueRange operands,
                            SmallVectorImpl<Value> &flattened) {
  // In case of
  // tuple<a, b>, c, tuple<d, e>
  // ==>
  // a, b, c, d, e
  for (auto operand : operands) {
    if (auto cast =
            dyn_cast<UnrealizedConversionCastOp>(operand.getDefiningOp());
        cast && cast->getResultTypes()[0].isa<TupleType>())
      // An unrealized_conversion_cast will be inserted by type converter to
      // inter-mix the gap between 1:N conversion between tuple and types.
      // In this case, take the operands in the cast and replace the tuple
      // output with the flattened type array.
      flattened.append(cast.getOperands().begin(), cast.getOperands().end());
    else
      flattened.push_back(operand);
  }
}
//===----------------------------------------------------------------------===//
// Conversion rules.
//===----------------------------------------------------------------------===//

/// Sparse tensor storage conversion rule for sparse_tensor::storage.
class SparseStorageConversion : public OpConversionPattern<StorageOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(StorageOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Simply convert it to a unrealize_conversion_cast.
    // We should guarantee that all uses of sparse_tensor.storage op will
    // be eventually eliminated by accessing the flattened SSA values directly.
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, TypeRange{op.getType()}, adaptor.getInputs());
    return success();
  }
};

/// Sparse tensor storage conversion rule for sparse_tensor::storage_get.
class SparseStorageGetConverter : public OpConversionPattern<StorageGetOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(StorageGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto castOp =
        cast<UnrealizedConversionCastOp>(adaptor.getStorage().getDefiningOp());
    uint64_t idx = op.getIdx().getZExtValue();
    assert(idx < castOp.getOperands().size());

    rewriter.replaceOp(op, castOp.getOperand(idx));
    return success();
  }
};

/// Sparse tensor storage conversion rule for sparse_tensor::storage_set.
class SparseStorageSetConverter : public OpConversionPattern<StorageSetOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(StorageSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto castOp =
        cast<UnrealizedConversionCastOp>(adaptor.getStorage().getDefiningOp());
    uint64_t idx = op.getIdx().getZExtValue();

    SmallVector<Value, 8> values(castOp.getOperands());
    assert(idx < values.size());

    // Updates the corresponding element.
    values[idx] = adaptor.getValue();
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, TypeRange{op.getType()}, values);
    return success();
  }
};

/// Sparse tensor storage conversion rule for returns.
class SparseStorageReturnConverter
    : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> flattened;
    flattenOperands(adaptor.getOperands(), flattened);
    // Create a return with the flattened value extracted from tuple.
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, flattened);
    return success();
  }
};

/// Sparse tensor storage conversion rule for calls.
class SparseStorageCallConverter : public OpConversionPattern<func::CallOp> {
public:
  // The default CallOp converter can not handle 1:N type conversion properly
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // In case of:
    //  tuple(a, b), f, tuple(c, d) = call @foo(...)
    // ==>
    //  a, b, f, c, d = call @foo(...)
    //  cast(a, b)->tuple, f, cast(c,d)->tuple
    SmallVector<Type, 8> finalRetTy;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), finalRetTy)))
      return failure();

    // (1) Genereates new call with flattened return value.
    SmallVector<Value, 8> flattened;
    flattenOperands(adaptor.getOperands(), flattened);
    auto newCall = rewriter.create<func::CallOp>(loc, op.getCallee(),
                                                 finalRetTy, flattened);

    // (2) Create cast operation for tuple returns.
    SmallVector<Value, 4> castedRet;
    // Tracks the offset of current return value (of the orignal call)
    // relative to the new call (after tuple flattening);
    unsigned retOffset = 0;
    for (auto ret : op.getResults()) {
      assert(retOffset < newCall.getNumResults());
      auto tupleRet = ret.getType().dyn_cast<TupleType>();
      if (tupleRet) {
        auto tupleSize = tupleRet.size();
        // NOTE: The range is computed under the assumption of non-recursive
        // tuple type.
        ValueRange tupleElem(iterator_range<ResultRange::iterator>(
            newCall.result_begin() + retOffset,
            newCall.result_begin() + retOffset + tupleSize));
        auto castOp = rewriter.create<UnrealizedConversionCastOp>(
            loc, TypeRange({tupleRet}), tupleElem);
        castedRet.push_back(castOp.getResult(0));
        retOffset += tupleSize;
      } else {
        // If this not a tuple, simply add it into returned values.
        castedRet.push_back(ret);
        retOffset++;
      }
    }

    assert(castedRet.size() == op.getNumResults());
    rewriter.replaceOp(op, castedRet);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Sparse tensor storage expansion
//===----------------------------------------------------------------------===//

mlir::SparseTensorStorageTupleExpander::SparseTensorStorageTupleExpander() {
  addConversion([](Type type) { return type; });
  addConversion(convertSparseTensorStorageTuple);
}

//===----------------------------------------------------------------------===//
// Public method for populating conversion rules.
//===----------------------------------------------------------------------===//

/// Populates the given patterns list with conversion rules required
/// to expand compounded sparse tensor tuples.
void mlir::populateSparseTensorStorageExpansionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<SparseStorageConversion, SparseStorageGetConverter,
               SparseStorageSetConverter, SparseStorageReturnConverter,
               SparseStorageCallConverter>(typeConverter,
                                           patterns.getContext());
}
