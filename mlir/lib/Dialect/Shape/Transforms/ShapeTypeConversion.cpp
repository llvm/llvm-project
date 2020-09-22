//=====------- ShapeTypeConversion.cpp - Shape Type Conversions ----------*- C++
//-*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines patterns to convert types of inputs and outputs to shape
// operations to be memrefs instead of tensors.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferPlacement.h"

using namespace mlir;
using namespace mlir::shape;

namespace {
// Propagate tensor to memref conversions through shape.assuming ops.
class TypeConversionAssumingOpConverter
    : public BufferAssignmentOpConversionPattern<shape::AssumingOp> {
public:
  using BufferAssignmentOpConversionPattern<
      shape::AssumingOp>::BufferAssignmentOpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::AssumingOp assumingOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type, 2> newResultTypes;
    newResultTypes.reserve(assumingOp.getNumResults());
    for (auto result : assumingOp.getResults()) {
      auto originalType = result.getType();
      Type convertedType = converter->convertType(originalType);
      newResultTypes.push_back(convertedType);
    }

    auto newAssumingOp = rewriter.create<shape::AssumingOp>(
        assumingOp.getLoc(), newResultTypes, assumingOp.witness());

    // Handle the region transfer carefully here to avoid assertions that both
    // operations are valid at replacement time.
    newAssumingOp.doRegion().push_back(new Block());
    rewriter.replaceOp(assumingOp, newAssumingOp.getResults());
    newAssumingOp.doRegion().takeBody(assumingOp.doRegion());

    return success();
  }
};

struct ShapeTensorToMemrefPass
    : public ShapeTensorToMemrefBase<ShapeTensorToMemrefPass> {
  void runOnFunction() override {
    MLIRContext &ctx = getContext();

    OwningRewritePatternList patterns;
    BufferAssignmentTypeConverter converter;
    populateShapeTypeConversionPatterns(&ctx, &converter, &patterns);

    ConversionTarget target(getContext());
    auto isMemRefType = [](Type type) { return type.isa<BaseMemRefType>(); };

    target.addDynamicallyLegalOp<AssumingOp>([&](shape::AssumingOp op) {
      return std::all_of(op.result_type_begin(), op.result_type_end(),
                         isMemRefType);
    });

    if (failed(mlir::applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();
  }
};

} // namespace

/// Populates `patterns` with the conversion patterns of tensor->memref.
//
// TODO(tpopp): Change this to work generally with any type conversions.
void mlir::populateShapeTypeConversionPatterns(
    MLIRContext *context, BufferAssignmentTypeConverter *converter,
    OwningRewritePatternList *patterns) {
  patterns->insert<TypeConversionAssumingOpConverter>(context, converter);
}

//===----------------------------------------------------------------------===//
// ShapeTensorToMemrefPass construction
//===----------------------------------------------------------------------===//

std::unique_ptr<FunctionPass> mlir::createShapeTensorToMemrefPass() {
  return std::make_unique<ShapeTensorToMemrefPass>();
}
