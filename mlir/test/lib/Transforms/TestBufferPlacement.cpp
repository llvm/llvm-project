//===- TestBufferPlacement.cpp - Test for buffer placement ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing buffer placement including its
// utility converters.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/BufferPlacement.h"

using namespace mlir;

namespace {
/// This pass tests the computeAllocPosition helper method and buffer assignment
/// operation converters. Furthermore, this pass converts linalg operations on
/// tensors to linalg operations on buffers to prepare them for the
/// BufferPlacement pass that can be applied afterwards.
/// `allowMemrefFunctionResults` informs the buffer placement to allow functions
/// that have memref typed results. Buffer assignment operation converters will
/// be adapted respectively. It will also allow memref typed results to escape
/// from the deallocation.
template <bool allowMemrefFunctionResults>
struct TestBufferPlacementPreparationPass
    : mlir::PassWrapper<
          TestBufferPlacementPreparationPass<allowMemrefFunctionResults>,
          OperationPass<ModuleOp>> {

  /// Converts tensor-type generic linalg operations to memref ones using
  /// buffer assignment.
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
        ShapedType type = result.getType().cast<ShapedType>();
        assert(type && "Generic operations with non-shaped typed results are "
                       "not currently supported.");
        if (!type.hasStaticShape())
          return rewriter.notifyMatchFailure(
              op, "dynamic shapes not currently supported");
        auto memrefType =
            MemRefType::get(type.getShape(), type.getElementType());

        // Compute alloc position and insert a custom allocation node.
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.restoreInsertionPoint(
            bufferAssignment->computeAllocPosition(result));
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

      // Map the old block arguments to the new ones.
      BlockAndValueMapping mapping;
      mapping.map(oldBlock.getArguments(), newBlock->getArguments());

      // Add the result arguments to the new block.
      for (auto result : newResults)
        newBlock->addArgument(
            result.getType().cast<ShapedType>().getElementType());

      // Clone the body of the old block to the new block.
      rewriter.setInsertionPointToEnd(newBlock);
      for (auto &op : oldBlock.getOperations())
        rewriter.clone(op, mapping);

      // Replace the results of the old Generic Op with the results of the new
      // one.
      rewriter.replaceOp(op, newResults);
      return success();
    }
  };

  void populateTensorLinalgToBufferLinalgConversionPattern(
      MLIRContext *context, BufferAssignmentPlacer *placer,
      TypeConverter *converter, OwningRewritePatternList *patterns) {
    populateWithBufferAssignmentOpConversionPatterns<
        mlir::ReturnOp, mlir::ReturnOp, linalg::CopyOp,
        allowMemrefFunctionResults>(context, placer, converter, patterns);
    patterns->insert<GenericOpConverter>(context, placer, converter);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = this->getContext();
    ConversionTarget target(context);
    BufferAssignmentTypeConverter converter;

    // Mark all Standard operations legal.
    target.addLegalDialect<StandardOpsDialect>();

    // Mark all Linalg operations illegal as long as they work on tensors.
    auto isLegalOperation = [&](Operation *op) {
      return converter.isLegal(op);
    };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(isLegalOperation);

    // Mark Standard Return operations illegal as long as one operand is tensor.
    target.addDynamicallyLegalOp<mlir::ReturnOp>([&](mlir::ReturnOp returnOp) {
      return converter.isLegal(returnOp.getOperandTypes());
    });

    // Mark Standard Call Operation illegal as long as it operates on tensor.
    target.addDynamicallyLegalOp<mlir::CallOp>(
        [&](mlir::CallOp callOp) { return converter.isLegal(callOp); });

    // Mark the function whose arguments are in tensor-type illegal.
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
      return converter.isSignatureLegal(funcOp.getType()) &&
             converter.isLegal(&funcOp.getBody());
    });

    // Walk over all the functions to apply buffer assignment.
    this->getOperation().walk([&](FuncOp function) -> WalkResult {
      OwningRewritePatternList patterns;
      BufferAssignmentPlacer placer(function);
      populateTensorLinalgToBufferLinalgConversionPattern(
          &context, &placer, &converter, &patterns);

      // Applying full conversion
      return applyFullConversion(function, target, patterns);
    });
  };
};
} // end anonymous namespace

namespace mlir {
void registerTestBufferPlacementPreparationPass() {
  PassRegistration<
      TestBufferPlacementPreparationPass</*allowMemrefFunctionResults=*/false>>(
      "test-buffer-placement-preparation",
      "Tests buffer placement helper methods including its "
      "operation-conversion patterns");
}

void registerTestPreparationPassWithAllowedMemrefResults() {
  PassRegistration<
      TestBufferPlacementPreparationPass</*allowMemrefFunctionResults=*/true>>(
      "test-buffer-placement-preparation-with-allowed-memref-results",
      "Tests the helper operation converters of buffer placement for allowing "
      "functions to have memref typed results.");
}
} // end namespace mlir
