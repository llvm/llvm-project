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
    linalg::GenericOpAdaptor adaptor(operands,
                                     op.getOperation()->getAttrDictionary());

    // All inputs need to be turned into buffers first. Until then, bail out.
    if (llvm::any_of(adaptor.inputs(),
                     [](Value in) { return !in.getType().isa<MemRefType>(); }))
      return failure();

    // All init_tensors need to be turned into buffers first. Until then, bail
    // out.
    if (llvm::any_of(adaptor.init_tensors(),
                     [](Value in) { return !in.getType().isa<MemRefType>(); }))
      return failure();

    Location loc = op.getLoc();
    SmallVector<Value, 2> newOutputBuffers;
    newOutputBuffers.reserve(op.getNumOutputs());
    newOutputBuffers.append(adaptor.output_buffers().begin(),
                            adaptor.output_buffers().end());

    // Update all types to memref types.
    // Assume the init tensors fold onto the first results.
    // TODO: update this assumption because the reality is more complex under
    // linalg on tensor based transformations.
    for (auto en : llvm::enumerate(op.getResultTypes())) {
      auto type = en.value().cast<ShapedType>();
      if (!type.hasStaticShape())
        return rewriter.notifyMatchFailure(
            op, "dynamic shapes not currently supported");
      auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
      bool foldedInitTensor = en.index() < op.getNumInitTensors();
      if (foldedInitTensor) {
        // Dealing with an init tensor requires distinguishing between 1-use
        // and many-use cases which would create aliasing and WAR hazards.
        Value initTensor = op.getInitTensor(en.index());
        Value initBuffer = adaptor.init_tensors()[en.index()];
        if (initTensor.hasOneUse()) {
          newOutputBuffers.push_back(initBuffer);
          continue;
        }
        auto alloc = rewriter.create<AllocOp>(loc, memrefType);
        rewriter.create<linalg::CopyOp>(loc, initBuffer, alloc);
        newOutputBuffers.push_back(alloc);
      } else {
        auto alloc = rewriter.create<AllocOp>(loc, memrefType);
        newOutputBuffers.push_back(alloc);
      }
    }

    // Generate a new linalg operation that works on buffers.
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/ArrayRef<Type>{},
        /*inputs=*/adaptor.inputs(),
        /*outputBuffers=*/newOutputBuffers,
        /*initTensors=*/ValueRange{}, op.indexing_maps(), op.iterator_types(),
        op.docAttr(), op.library_callAttr(), op.symbol_sourceAttr());

    // Create a new block in the region of the new Generic Op.
    Block &oldBlock = op.getRegion().front();
    Region &newRegion = linalgOp.region();
    Block *newBlock = rewriter.createBlock(&newRegion, newRegion.begin(),
                                           oldBlock.getArgumentTypes());

    // Add the result arguments that do not come from init_tensors to the new
    // block.
    // TODO: update this assumption because the reality is more complex under
    // linalg on tensor based transformations.
    for (Value v :
         ValueRange(newOutputBuffers).drop_front(adaptor.init_tensors().size()))
      newBlock->addArgument(v.getType().cast<MemRefType>().getElementType());

    // Clone the body of the old block to the new block.
    BlockAndValueMapping mapping;
    for (unsigned i = 0; i < oldBlock.getNumArguments(); i++)
      mapping.map(oldBlock.getArgument(i), newBlock->getArgument(i));

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(newBlock);
    for (auto &op : oldBlock.getOperations()) {
      Operation *clonedOp = rewriter.clone(op, mapping);
      mapping.map(op.getResults(), clonedOp->getResults());
    }

    // Replace the results of the old op with the new output buffers.
    rewriter.replaceOp(op, newOutputBuffers);
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
