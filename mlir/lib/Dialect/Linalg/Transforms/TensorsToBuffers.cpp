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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferPlacement.h"

namespace {

using namespace ::mlir;
using namespace ::mlir::linalg;

SmallVector<Range, 4>
computeLoopRanges(Location loc, linalg::GenericOp linalgOp, OpBuilder *b) {
  auto indexingMaps = llvm::to_vector<4>(
      linalgOp.indexing_maps().getAsValueRange<AffineMapAttr>());
  auto inputIndexingMaps =
      llvm::makeArrayRef(indexingMaps).take_front(linalgOp.getNumInputs());

  mlir::edsc::ScopedContext scope(*b, loc);
  return emitLoopRanges(scope.getBuilderRef(), loc,
                        concatAffineMaps(inputIndexingMaps),
                        getShape(*b, linalgOp));
}

Value maybeConvertToIndex(Location loc, Value val, OpBuilder *b) {
  if (val.getType().isIndex())
    return val;
  return b->create<IndexCastOp>(loc, val, b->getIndexType());
}

LogicalResult allocateBuffersForResults(Location loc,
                                        linalg::GenericOp linalgOp,
                                        linalg::GenericOpAdaptor &adaptor,
                                        SmallVectorImpl<Value> *resultBuffers,
                                        OpBuilder *b) {
  // Lazily compute loopRanges.
  SmallVector<Range, 4> loopRanges;

  // Allocate a buffer for every tensor result.
  for (auto en : llvm::enumerate(linalgOp.getResultTypes())) {
    size_t resultIndex = en.index();
    Type resultType = en.value();

    auto tensorType = resultType.dyn_cast<RankedTensorType>();
    if (tensorType == nullptr) {
      linalgOp.emitOpError()
          << "tensor to buffer conversion expects ranked tensor results";
      return failure();
    }
    auto tensorShape = tensorType.getShape();
    auto memrefType = MemRefType::get(tensorShape, tensorType.getElementType());

    // Allocate buffers for init tensors that are assumed to fold onto the first
    // results.
    // TODO: update this assumption because the reality is more complex
    // under linalg on tensor based transformations.
    bool foldedInitTensor = resultIndex < linalgOp.getNumInitTensors();
    if (foldedInitTensor) {
      // Dealing with an init tensor requires distinguishing between 1-use
      // and many-use cases which would create aliasing and WAR hazards.
      Value initTensor = linalgOp.getInitTensor(resultIndex);
      Value initBuffer = adaptor.init_tensors()[resultIndex];
      if (initTensor.hasOneUse()) {
        resultBuffers->push_back(initBuffer);
        continue;
      }
      SmallVector<Value, 4> dynOperands;
      for (auto dim : llvm::enumerate(tensorShape)) {
        if (dim.value() == TensorType::kDynamicSize) {
          dynOperands.push_back(b->create<DimOp>(loc, initTensor, dim.index()));
        }
      }
      auto alloc = b->create<AllocOp>(loc, memrefType, dynOperands);
      b->create<linalg::CopyOp>(loc, initBuffer, alloc);
      resultBuffers->push_back(alloc);
      continue;
    }

    // Allocate buffers for statically-shaped results.
    if (memrefType.hasStaticShape()) {
      resultBuffers->push_back(b->create<AllocOp>(loc, memrefType));
      continue;
    }

    // Perform a naive shape inference for the dynamically-shaped results.
    // Extract the required element out of the vector.
    SmallVector<Value, 4> dynOperands;
    auto resultIndexingMap = linalgOp.getOutputIndexingMap(resultIndex);
    for (auto shapeElement : llvm::enumerate(tensorType.getShape())) {
      if (loopRanges.empty())
        loopRanges = computeLoopRanges(loc, linalgOp, b);

      if (shapeElement.value() != ShapedType::kDynamicSize)
        continue;

      AffineExpr expr = resultIndexingMap.getResult(shapeElement.index());
      switch (expr.getKind()) {
      case AffineExprKind::DimId: {
        int64_t loopIndex = expr.cast<AffineDimExpr>().getPosition();
        Value size = maybeConvertToIndex(loc, loopRanges[loopIndex].size, b);
        dynOperands.push_back(size);
        break;
      }
      default:
        return failure();
      }
    }
    resultBuffers->push_back(b->create<AllocOp>(loc, memrefType, dynOperands));
  }
  return success();
}

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
  matchAndRewrite(linalg::GenericOp linalgOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    linalg::GenericOpAdaptor adaptor(
        operands, linalgOp.getOperation()->getAttrDictionary());

    // All inputs need to be turned into buffers first. Until then, bail out.
    if (llvm::any_of(adaptor.inputs(),
                     [](Value in) { return !in.getType().isa<MemRefType>(); }))
      return failure();

    // All init_tensors need to be turned into buffers first. Until then, bail
    // out.
    if (llvm::any_of(adaptor.init_tensors(),
                     [](Value in) { return !in.getType().isa<MemRefType>(); }))
      return failure();

    Location loc = linalgOp.getLoc();
    SmallVector<Value, 2> newOutputBuffers(adaptor.output_buffers().begin(),
                                           adaptor.output_buffers().end());

    if (failed(allocateBuffersForResults(loc, linalgOp, adaptor,
                                         &newOutputBuffers, &rewriter))) {
      linalgOp.emitOpError()
          << "Failed to allocate buffers for tensor results.";
      return failure();
    }

    // Generate a new linalg operation that works on buffers.
    auto newLinalgOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/llvm::None,
        /*inputs=*/adaptor.inputs(),
        /*outputBuffers=*/newOutputBuffers,
        /*initTensors=*/llvm::None, linalgOp.indexing_maps(),
        linalgOp.iterator_types(), linalgOp.docAttr(),
        linalgOp.library_callAttr(), linalgOp.symbol_sourceAttr());

    // Create a new block in the region of the new Generic Op.
    Block *oldBlock = linalgOp.getBody();
    Region &newRegion = newLinalgOp.region();
    Block *newBlock = rewriter.createBlock(&newRegion, newRegion.begin(),
                                           oldBlock->getArgumentTypes());

    // Add the result arguments to the new block.
    for (Value v : newOutputBuffers)
      newBlock->addArgument(v.getType().cast<MemRefType>().getElementType());

    // Clone the body of the old block to the new block.
    BlockAndValueMapping mapping;
    mapping.map(oldBlock->getArguments(), newBlock->getArguments());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(newBlock);
    for (auto &op : oldBlock->getOperations()) {
      Operation *clonedOp = rewriter.clone(op, mapping);
      mapping.map(op.getResults(), clonedOp->getResults());
    }

    // Replace the results of the old op with the new output buffers.
    rewriter.replaceOp(linalgOp, newOutputBuffers);
    return success();
  }
};

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
    populateConvertLinalgOnTensorsToBuffersPatterns(&context, &converter,
                                                    &patterns);
    populateWithBufferAssignmentOpConversionPatterns<
        mlir::ReturnOp, mlir::ReturnOp, linalg::CopyOp>(&context, &converter,
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

void mlir::linalg::populateConvertLinalgOnTensorsToBuffersPatterns(
    MLIRContext *context, BufferAssignmentTypeConverter *converter,
    OwningRewritePatternList *patterns) {
  patterns->insert<GenericOpConverter>(context, converter);
}
