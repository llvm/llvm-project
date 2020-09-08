//===- TensorsToBuffers.cpp - Transformation from tensors to buffers ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the conversion from tensors to buffers on Linalg
// operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferPlacement.h"

using namespace mlir;

namespace {
/// A pattern to convert Generic Linalg operations which work on tensors to
/// use buffers. A buffer is allocated using BufferAssignmentPlacer for
/// each operation result. BufferPlacement pass should be later used to move
/// Alloc operations to the correct positions and insert the missing Dealloc
/// operations in the correct places.
class GenericOpConverter
    : public BufferAssignmentOpConversionPattern<linalg::GenericOp> {
public:
  using BufferAssignmentOpConversionPattern<
      linalg::GenericOp>::BufferAssignmentOpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ResultRange results = op.getOperation()->getResults();
    SmallVector<Value, 2> newArgs, newResults;
    newArgs.reserve(operands.size() + results.size());
    newArgs.append(operands.begin(), operands.end());
    newResults.reserve(results.size());

    // Update all types to memref types.
    for (auto result : results) {
      auto type = result.getType().cast<ShapedType>();
      assert(type && "tensor to buffer conversion expects ranked results");
      if (!type.hasStaticShape())
        return rewriter.notifyMatchFailure(
            op, "dynamic shapes not currently supported");
      auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
      auto alloc = rewriter.create<AllocOp>(loc, memrefType);
      newArgs.push_back(alloc);
      newResults.push_back(alloc);
    }

    // Generate a new linalg operation that works on buffers.
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, llvm::None, newArgs, rewriter.getI64IntegerAttr(operands.size()),
        rewriter.getI64IntegerAttr(results.size()), op.indexing_maps(),
        op.iterator_types(), op.docAttr(), op.library_callAttr(),
        op.symbol_sourceAttr());

    // Create a new block in the region of the new Generic Op.
    Block &oldBlock = op.getRegion().front();
    Region &newRegion = linalgOp.region();
    Block *newBlock = rewriter.createBlock(&newRegion, newRegion.begin(),
                                           oldBlock.getArgumentTypes());

    // Add the result arguments to the new block.
    for (auto result : newResults)
      newBlock->addArgument(
          result.getType().cast<ShapedType>().getElementType());

    // Clone the body of the old block to the new block.
    BlockAndValueMapping mapping;
    for (unsigned i = 0; i < oldBlock.getNumArguments(); i++)
      mapping.map(oldBlock.getArgument(i), newBlock->getArgument(i));
    rewriter.setInsertionPointToEnd(newBlock);
    for (auto &op : oldBlock.getOperations()) {
      Operation *clonedOp = rewriter.clone(op, mapping);
      mapping.map(op.getResults(), clonedOp->getResults());
    }

    // Replace the results of the old Generic Op with the results of the new
    // one.
    rewriter.replaceOp(op, newResults);
    return success();
  }
};

/// Populate the given list with patterns to convert Linalg operations on
/// tensors to buffers.
static void populateConvertLinalgOnTensorsToBuffersPattern(
    MLIRContext *context, BufferAssignmentTypeConverter *converter,
    OwningRewritePatternList *patterns) {
  populateWithBufferAssignmentOpConversionPatterns<
      mlir::ReturnOp, mlir::ReturnOp, linalg::CopyOp>(context, converter,
                                                      patterns);
  patterns->insert<GenericOpConverter>(context, converter);
}

/// Converts Linalg operations that work on tensor-type operands or results to
/// work on buffers.
struct ConvertLinalgOnTensorsToBuffers
    : public LinalgOnTensorsToBuffersBase<ConvertLinalgOnTensorsToBuffers> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget target(context);
    BufferAssignmentTypeConverter converter;

    // Mark all Standard operations legal.
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<ModuleTerminatorOp>();

    // Mark all Linalg operations illegal as long as they work on tensors.
    auto isLegalOperation = [&](Operation *op) {
      return converter.isLegal(op);
    };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
        Optional<ConversionTarget::DynamicLegalityCallbackFn>(
            isLegalOperation));

    // Mark Standard Return operations illegal as long as one operand is tensor.
    target.addDynamicallyLegalOp<mlir::ReturnOp>([&](mlir::ReturnOp returnOp) {
      return converter.isLegal(returnOp.getOperandTypes());
    });

    // Mark the function operation illegal as long as an argument is tensor.
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
      return converter.isSignatureLegal(funcOp.getType()) &&
             llvm::none_of(funcOp.getType().getResults(),
                           [&](Type type) { return type.isa<MemRefType>(); }) &&
             converter.isLegal(&funcOp.getBody());
    });

    converter.setResultConversionKind<RankedTensorType, MemRefType>(
        BufferAssignmentTypeConverter::AppendToArgumentsList);

    OwningRewritePatternList patterns;
    populateConvertLinalgOnTensorsToBuffersPattern(&context, &converter,
                                                   &patterns);
    if (failed(applyFullConversion(this->getOperation(), target, patterns)))
      this->signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLinalgOnTensorsToBuffersPass() {
  return std::make_unique<ConvertLinalgOnTensorsToBuffers>();
}
