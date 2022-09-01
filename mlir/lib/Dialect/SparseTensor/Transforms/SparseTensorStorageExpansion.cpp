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

//===----------------------------------------------------------------------===//
// Conversion rules.
//===----------------------------------------------------------------------===//

/// Sparse tensor storage conversion rule for returns.
class SparseStorageReturnConverter
    : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> flattened;
    for (auto operand : adaptor.getOperands()) {
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
    // Create a return with the flattened value extracted from tuple.
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, flattened);
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
  patterns.add<SparseStorageReturnConverter>(typeConverter,
                                             patterns.getContext());
}
