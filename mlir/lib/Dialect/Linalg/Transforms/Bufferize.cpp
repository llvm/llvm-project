//===- Bufferize.cpp - Bufferization of linalg ops ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Bufferize.h"
#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace ::mlir;
using namespace ::mlir::linalg;

static SmallVector<Range, 4> computeLoopRanges(Location loc, LinalgOp linalgOp,
                                               OpBuilder &b) {
  auto indexingMaps = llvm::to_vector<4>(
      linalgOp.indexing_maps().getAsValueRange<AffineMapAttr>());
  auto inputIndexingMaps =
      llvm::makeArrayRef(indexingMaps).take_front(linalgOp.getNumInputs());

  mlir::edsc::ScopedContext scope(b, loc);
  return emitLoopRanges(scope.getBuilderRef(), loc,
                        concatAffineMaps(inputIndexingMaps),
                        getShape(b, linalgOp));
}

static Value maybeConvertToIndex(Location loc, Value val, OpBuilder &b) {
  if (val.getType().isIndex())
    return val;
  return b.create<IndexCastOp>(loc, val, b.getIndexType());
}

static LogicalResult
allocateBuffersForResults(Location loc, LinalgOp linalgOp,
                          linalg::GenericOpAdaptor &adaptor,
                          SmallVectorImpl<Value> &resultBuffers, OpBuilder &b) {
  // Lazily compute loopRanges.
  SmallVector<Range, 4> loopRanges;

  // Allocate a buffer for every tensor result.
  for (auto en : llvm::enumerate(linalgOp.getOperation()->getResultTypes())) {
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
        resultBuffers.push_back(initBuffer);
        continue;
      }
      SmallVector<Value, 4> dynOperands;
      for (auto dim : llvm::enumerate(tensorShape)) {
        if (dim.value() == TensorType::kDynamicSize) {
          dynOperands.push_back(b.create<DimOp>(loc, initTensor, dim.index()));
        }
      }
      auto alloc = b.create<AllocOp>(loc, memrefType, dynOperands);
      b.create<linalg::CopyOp>(loc, initBuffer, alloc);
      resultBuffers.push_back(alloc);
      continue;
    }

    // Allocate buffers for statically-shaped results.
    if (memrefType.hasStaticShape()) {
      resultBuffers.push_back(b.create<AllocOp>(loc, memrefType));
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
    resultBuffers.push_back(b.create<AllocOp>(loc, memrefType, dynOperands));
  }
  return success();
}

// Specialization for `linalg::GenericOp`.
/// A pattern to convert Generic Linalg operations which work on tensors to
/// use buffers. BufferPlacement pass should be later used to move
/// Alloc operations to the correct positions and insert the missing Dealloc
/// operations in the correct places.
static void finalizeBufferAllocation(ConversionPatternRewriter &rewriter,
                                     linalg::GenericOp genericOp,
                                     ValueRange inputs, ValueRange outputs) {
  // Generate a new linalg operation that works on buffers.
  auto newGenericOp = rewriter.create<linalg::GenericOp>(
      genericOp.getLoc(),
      /*resultTensorTypes=*/llvm::None,
      /*inputs=*/inputs,
      /*outputBuffers=*/outputs,
      /*initTensors=*/llvm::None, genericOp.indexing_maps(),
      genericOp.iterator_types(), genericOp.docAttr(),
      genericOp.library_callAttr(), genericOp.symbol_sourceAttr());

  // Create a new block in the region of the new Generic Op.
  Block *oldBlock = genericOp.getBody();
  Region &newRegion = newGenericOp.region();
  Block *newBlock = rewriter.createBlock(&newRegion, newRegion.begin(),
                                         oldBlock->getArgumentTypes());

  // Add the result arguments to the new block.
  for (Value v : ValueRange(outputs).drop_front(genericOp.getNumInitTensors()))
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
  rewriter.replaceOp(genericOp, outputs);
}

// TODO: Specialization for `linalg::IndexedGenericOp`.

// Specialization for all other `linalg::LinalgOp`.
static void finalizeBufferAllocation(ConversionPatternRewriter &rewriter,
                                     linalg::LinalgOp linalgOp,
                                     ValueRange inputs, ValueRange outputs) {
  assert(!isa<linalg::GenericOp>(linalgOp.getOperation()));
  assert(!isa<linalg::IndexedGenericOp>(linalgOp.getOperation()));
  SmallVector<Value, 8> newOperands = inputs;
  newOperands.append(outputs.begin(), outputs.end());
  auto otherOperands = linalgOp.getAssumedNonShapedOperands();
  newOperands.append(otherOperands.begin(), otherOperands.end());
  LinalgOp res = cast<LinalgOp>(linalgOp.clone(rewriter, linalgOp.getLoc(),
                                               /*resultTypes=*/ArrayRef<Type>{},
                                               newOperands));
  // Need to mutate the operands_segment_sizes in the resulting op.
  res.setNumOutputBuffers(outputs.size());
  res.setNumInitTensors(0);
  // Replace the results of the old op with the new output buffers.
  rewriter.replaceOp(linalgOp, outputs);
}

//===----------------------------------------------------------------------===//
// Buffer allocation patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Generic BufferizeConversionPattern that matches any Operation* and
/// dispatches internally. This avoids template instantiating one pattern for
/// each LinalgOp op.
class LinalgOpConverter : public BufferizeConversionPattern {
public:
  LinalgOpConverter(MLIRContext *context, BufferizeTypeConverter &converter)
      : BufferizeConversionPattern(context, converter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      return failure();

    // We abuse the GenericOpAdaptor here.
    // TODO: Manually create an Adaptor that captures inputs, output_buffers and
    // init_tensors for all linalg::LinalgOp interface ops.
    linalg::GenericOpAdaptor adaptor(operands, op->getAttrDictionary());

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
                                         newOutputBuffers, rewriter))) {
      linalgOp.emitOpError()
          << "Failed to allocate buffers for tensor results.";
      return failure();
    }

    // Delegate to the linalg generic pattern.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      finalizeBufferAllocation(rewriter, genericOp, adaptor.inputs(),
                               newOutputBuffers);
      return success();
    }

    finalizeBufferAllocation(rewriter, linalgOp, adaptor.inputs(),
                             newOutputBuffers);
    return success();
  }
};
} // namespace

namespace {
/// TensorConstantOp conversion inserts a linearized 1-D vector constant that is
/// stored in memory. A linalg.reshape is introduced to convert to the desired
/// n-D buffer form.
class TensorConstantOpConverter
    : public BufferizeOpConversionPattern<ConstantOp> {
public:
  using BufferizeOpConversionPattern<ConstantOp>::BufferizeOpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    RankedTensorType rankedTensorType =
        op.getType().dyn_cast<RankedTensorType>();
    if (!rankedTensorType)
      return failure();
    if (llvm::any_of(rankedTensorType.getShape(), [](int64_t s) {
          return s == 0 || ShapedType::isDynamic(s);
        }))
      return failure();

    int64_t nElements = 1;
    for (int64_t s : rankedTensorType.getShape())
      nElements *= s;
    Type elementType = rankedTensorType.getElementType();
    MemRefType memrefType =
        converter.convertType(op.getType()).cast<MemRefType>();
    VectorType flatVectorType = VectorType::get({nElements}, elementType);
    MemRefType memrefOfFlatVectorType = MemRefType::get({}, flatVectorType);
    MemRefType flatMemrefType = MemRefType::get({nElements}, elementType);

    Location loc = op.getLoc();
    auto attr = op.getValue().cast<DenseElementsAttr>();
    Value alloc =
        rewriter.create<AllocOp>(loc, memrefOfFlatVectorType, ValueRange{});
    Value cstVec = rewriter.create<ConstantOp>(loc, flatVectorType,
                                               attr.reshape(flatVectorType));
    rewriter.create<StoreOp>(loc, cstVec, alloc);

    Value memref =
        rewriter.create<vector::TypeCastOp>(loc, flatMemrefType, alloc);
    if (rankedTensorType.getRank() > 1) {
      // Introduce a linalg.reshape to flatten the memref.
      AffineMap collapseAllDims = AffineMap::getMultiDimIdentityMap(
          /*numDims=*/rankedTensorType.getRank(), op.getContext());
      memref = rewriter.create<linalg::ReshapeOp>(
          loc, memrefType, memref,
          rewriter.getAffineMapArrayAttr(collapseAllDims));
    }
    rewriter.replaceOp(op, memref);

    return success();
  }
};
} // namespace

namespace {

/// Converts Linalg operations that work on tensor-type operands or results to
/// work on buffers.
struct LinalgBufferizePass : public LinalgBufferizeBase<LinalgBufferizePass> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget target(context);
    BufferizeTypeConverter converter;

    // Mark all Standard operations legal.
    target.addLegalDialect<StandardOpsDialect, vector::VectorDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<ModuleTerminatorOp>();

    // Mark all Linalg operations illegal as long as they work on tensors.
    auto isLegalOperation = [&](Operation *op) {
      return converter.isLegal(op);
    };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
        Optional<ConversionTarget::DynamicLegalityCallbackFn>(
            isLegalOperation));

    // Mark operations that consume or return tensors illegal.
    auto isLegal = [&](Operation *op) {
      if (llvm::any_of(op->getOperandTypes(),
                       [&](Type t) { return !converter.isLegal(t); }))
        return false;
      if (llvm::any_of(op->getResultTypes(),
                       [&](Type t) { return !converter.isLegal(t); }))
        return false;
      return true;
    };
    target.addDynamicallyLegalOp<
        // clang-format off
        CallOp,
        ConstantOp,
        ConstantIntOp,
        ConstantIndexOp,
        ConstantFloatOp,
        ReturnOp,
        TensorCastOp
        // clang-format on
        >(isLegal);

    // Mark the function operation illegal as long as an argument is tensor.
    // TODO: if the FuncOp is a FuncOp that only has a declaration (e.g. to an
    // externally defined symbol like an external library calls), only convert
    // if some special attribute is set. This will allow more control of interop
    // across ABI boundaries.
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
      return converter.isSignatureLegal(funcOp.getType()) &&
             llvm::none_of(funcOp.getType().getResults(),
                           [&](Type type) { return type.isa<MemRefType>(); }) &&
             converter.isLegal(&funcOp.getBody());
    });

    converter.setResultConversionKind<RankedTensorType, MemRefType>(
        BufferizeTypeConverter::AppendToArgumentsList);

    OwningRewritePatternList patterns;
    populateLinalgBufferizePatterns(&context, converter, patterns);
    populateStdBufferizePatterns(&context, converter, patterns);
    populateWithBufferizeOpConversionPatterns<mlir::ReturnOp, mlir::ReturnOp,
                                              linalg::CopyOp>(
        &context, converter, patterns);
    if (failed(applyFullConversion(this->getOperation(), target,
                                   std::move(patterns))))
      this->signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createLinalgBufferizePass() {
  return std::make_unique<LinalgBufferizePass>();
}
void mlir::linalg::populateLinalgBufferizePatterns(
    MLIRContext *context, BufferizeTypeConverter &converter,
    OwningRewritePatternList &patterns) {
  patterns.insert<
      // clang-format off
      LinalgOpConverter,
      TensorConstantOpConverter
      // clang-format on
      >(context, converter);
}
