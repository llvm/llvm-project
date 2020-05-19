//===- TestBufferPlacement.cpp - Test for buffer placement 0----*- C++ -*-===//
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
/// This pass tests the computeAllocPosition helper method and two provided
/// operation converters, FunctionAndBlockSignatureConverter and
/// NoBufferOperandsReturnOpConverter. Furthermore, this pass converts linalg
/// operations on tensors to linalg operations on buffers to prepare them for
/// the BufferPlacement pass that can be applied afterwards.
struct TestBufferPlacementPreparationPass
    : mlir::PassWrapper<TestBufferPlacementPreparationPass,
                        OperationPass<ModuleOp>> {

  /// Converts tensor-type generic linalg operations to memref ones using buffer
  /// assignment.
  class GenericOpConverter
      : public BufferAssignmentOpConversionPattern<linalg::GenericOp> {
  public:
    using BufferAssignmentOpConversionPattern<
        linalg::GenericOp>::BufferAssignmentOpConversionPattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final {
      auto loc = op.getLoc();
      SmallVector<Value, 4> args(operands.begin(), operands.end());

      // Update all types to memref types.
      auto results = op.getOperation()->getResults();
      for (auto result : results) {
        auto type = result.getType().cast<ShapedType>();
        if (!type)
          op.emitOpError()
              << "tensor to buffer conversion expects ranked results";
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
        result.replaceAllUsesWith(alloc);
        args.push_back(alloc);
      }

      // Generate a new linalg operation that works on buffers.
      auto linalgOp = rewriter.create<linalg::GenericOp>(
          loc, llvm::None, args, rewriter.getI64IntegerAttr(operands.size()),
          rewriter.getI64IntegerAttr(results.size()), op.indexing_maps(),
          op.iterator_types(), op.docAttr(), op.library_callAttr());

      // Move regions from the old operation to the new one.
      auto &region = linalgOp.region();
      rewriter.inlineRegionBefore(op.region(), region, region.end());

      // TODO: verify the internal memref-based linalg functionality.
      auto &entryBlock = region.front();
      for (auto result : results) {
        auto type = result.getType().cast<ShapedType>();
        entryBlock.addArgument(type.getElementType());
      }
      rewriter.eraseOp(op);
      return success();
    }
  };

  void populateTensorLinalgToBufferLinalgConversionPattern(
      MLIRContext *context, BufferAssignmentPlacer *placer,
      TypeConverter *converter, OwningRewritePatternList *patterns) {
    // clang-format off
    patterns->insert<
                   FunctionAndBlockSignatureConverter,
                   GenericOpConverter,
                   NoBufferOperandsReturnOpConverter<
                      ReturnOp, ReturnOp, linalg::CopyOp>
    >(context, placer, converter);
    // clang-format on
  }

  void runOnOperation() override {
    auto &context = getContext();
    ConversionTarget target(context);
    BufferAssignmentTypeConverter converter;
    target.addLegalDialect<StandardOpsDialect>();

    // Make all linalg operations illegal as long as they work on tensors.
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
        Optional<ConversionTarget::DynamicLegalityCallbackFn>(
            [&](Operation *op) {
              auto isIllegalType = [&](Type type) {
                return !converter.isLegal(type);
              };
              return llvm::none_of(op->getOperandTypes(), isIllegalType) &&
                     llvm::none_of(op->getResultTypes(), isIllegalType);
            }));

    // Mark std.ReturnOp illegal as long as an operand is tensor or buffer.
    target.addDynamicallyLegalOp<mlir::ReturnOp>([&](mlir::ReturnOp returnOp) {
      return llvm::none_of(returnOp.getOperandTypes(), [&](Type type) {
        return type.isa<MemRefType>() || !converter.isLegal(type);
      });
    });

    // Mark the function whose arguments are in tensor-type illegal.
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
      return converter.isSignatureLegal(funcOp.getType());
    });

    // Walk over all the functions to apply buffer assignment.
    getOperation().walk([&](FuncOp function) {
      OwningRewritePatternList patterns;
      BufferAssignmentPlacer placer(function);
      populateTensorLinalgToBufferLinalgConversionPattern(
          &context, &placer, &converter, &patterns);

      // Applying full conversion
      return failed(applyFullConversion(function, target, patterns, &converter))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
  };
};
} // end anonymous namespace

namespace mlir {
void registerTestBufferPlacementPreparationPass() {
  PassRegistration<TestBufferPlacementPreparationPass>(
      "test-buffer-placement-preparation",
      "Tests buffer placement helper methods including its "
      "operation-conversion patterns");
}
} // end namespace mlir