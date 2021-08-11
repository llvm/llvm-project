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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace ::mlir;
using namespace ::mlir::linalg;

static Value cloneMemref(Location loc, Value memref, OpBuilder &b) {
  auto memrefType = memref.getType().cast<MemRefType>();
  auto alloc = b.create<memref::AllocOp>(loc, memrefType,
                                         getDynOperands(loc, memref, b));
  b.create<linalg::CopyOp>(loc, memref, alloc);
  return alloc;
}

static LogicalResult
allocateBuffersForResults(Location loc, LinalgOp linalgOp, ValueRange outputs,
                          SmallVectorImpl<Value> &resultBuffers, OpBuilder &b) {
  // Lazily compute loopRanges.
  SmallVector<Range, 4> loopRanges;

  // Allocate a buffer for every tensor result.
  assert(linalgOp.getNumOutputs() == linalgOp->getNumResults());
  for (auto en : llvm::enumerate(linalgOp->getResultTypes())) {
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
    Value resultTensor = outputs[resultIndex];

    // Clone output buffers whose value is actually used.
    OpOperand *tiedOpOperand = linalgOp.getOutputOperand(resultIndex);
    if (linalgOp.payloadUsesValueFromOperand(tiedOpOperand)) {
      resultBuffers.push_back(cloneMemref(loc, resultTensor, b));
      continue;
    }

    // Allocate buffers for statically-shaped results.
    if (memrefType.hasStaticShape()) {
      resultBuffers.push_back(b.create<memref::AllocOp>(loc, memrefType));
      continue;
    }

    resultBuffers.push_back(b.create<memref::AllocOp>(
        loc, memrefType, getDynOperands(loc, resultTensor, b)));
  }
  return success();
}

/// Specialization for `linalg::GenericOp`.
/// A pattern to convert Generic Linalg operations which work on tensors to
/// use buffers. BufferPlacement pass should be later used to move
/// Alloc operations to the correct positions and insert the missing Dealloc
/// operations in the correct places.
static void
finalizeBufferAllocationForGenericOp(ConversionPatternRewriter &rewriter,
                                     GenericOp genericOp, ValueRange inputs,
                                     ValueRange outputs) {
  // Generate a new linalg operation that works on buffers.
  auto newGenericOp = rewriter.create<GenericOp>(
      genericOp.getLoc(),
      /*resultTensorTypes=*/llvm::None,
      /*inputs=*/inputs,
      /*outputs=*/outputs, genericOp.indexing_maps(),
      genericOp.iterator_types(), genericOp.docAttr(),
      genericOp.library_callAttr());

  // Create a new block in the region of the new Generic Op.
  Block *oldBlock = genericOp.getBody();
  Region &newRegion = newGenericOp.region();
  Block *newBlock = rewriter.createBlock(&newRegion, newRegion.begin(),
                                         oldBlock->getArgumentTypes());

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

/// Specialization for all other `linalg::LinalgOp`.
static void finalizeBufferAllocation(ConversionPatternRewriter &rewriter,
                                     linalg::LinalgOp linalgOp,
                                     ValueRange inputs, ValueRange outputs) {
  assert(!isa<linalg::GenericOp>(linalgOp.getOperation()));
  SmallVector<Value, 8> newOperands = inputs;
  newOperands.append(outputs.begin(), outputs.end());
  linalgOp.clone(rewriter, linalgOp.getLoc(),
                 /*resultTypes=*/ArrayRef<Type>{}, newOperands);
  // Replace the results of the old op with the new output buffers.
  rewriter.replaceOp(linalgOp, outputs);
}

//===----------------------------------------------------------------------===//
// Bufferization patterns.
//===----------------------------------------------------------------------===//

namespace {

/// Conversion pattern that replaces `linalg.init_tensor` with allocation.
class BufferizeInitTensorOp : public OpConversionPattern<InitTensorOp> {
public:
  using OpConversionPattern<InitTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InitTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    linalg::InitTensorOpAdaptor adaptor(operands, op->getAttrDictionary());
    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, getTypeConverter()->convertType(op.getType()).cast<MemRefType>(),
        adaptor.sizes());
    return success();
  }
};

/// Conversion pattern that replaces `linalg.tensor_reshape` with
/// `linalg.reshape`.
template <typename TensorReshapeOp,
          typename Adaptor = typename TensorReshapeOp::Adaptor>
class BufferizeTensorReshapeOp : public OpConversionPattern<TensorReshapeOp> {
public:
  using OpConversionPattern<TensorReshapeOp>::OpConversionPattern;
  using ReshapeOp = typename std::conditional_t<
      std::is_same<TensorReshapeOp, TensorExpandShapeOp>::value,
      memref::ExpandShapeOp, memref::CollapseShapeOp>;

  LogicalResult
  matchAndRewrite(TensorReshapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Adaptor adaptor(operands, op->getAttrDictionary());
    rewriter.replaceOpWithNewOp<ReshapeOp>(op,
                                           this->getTypeConverter()
                                               ->convertType(op.getType())
                                               .template cast<MemRefType>(),
                                           adaptor.src(),
                                           adaptor.reassociation());
    return success();
  }
};

/// Conversion pattern that bufferizes `linalg.fill` operation.
class BufferizeFillOp : public OpConversionPattern<FillOp> {
public:
  using OpConversionPattern<FillOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FillOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    linalg::FillOpAdaptor adaptor(operands, op->getAttrDictionary());
    if (!op.output().getType().isa<TensorType>())
      return rewriter.notifyMatchFailure(op,
                                         "operand must be of a tensor type");

    rewriter.create<FillOp>(op.getLoc(), adaptor.value(), adaptor.output());
    rewriter.replaceOp(op, adaptor.output());

    return success();
  }
};

/// Generic conversion pattern that matches any LinalgOp. This avoids template
/// instantiating one pattern for each LinalgOp.
class BufferizeAnyLinalgOp : public OpInterfaceConversionPattern<LinalgOp> {
public:
  using OpInterfaceConversionPattern<LinalgOp>::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(LinalgOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // GenericOpAdaptor below expects an `operand_segment_sizes` attribute.
    if (!op->hasAttr("operand_segment_sizes"))
      return failure();

    // We abuse the GenericOpAdaptor here.
    // TODO: Manually create an Adaptor that captures inputs and outputs for all
    // linalg::LinalgOp interface ops.
    linalg::GenericOpAdaptor adaptor(operands, op->getAttrDictionary());

    Location loc = op.getLoc();
    SmallVector<Value, 2> newOutputBuffers;

    if (op->getParentOfType<TiledLoopOp>()) {
      newOutputBuffers = adaptor.outputs();
    } else if (failed(allocateBuffersForResults(loc, op, adaptor.outputs(),
                                                newOutputBuffers, rewriter))) {
      return op.emitOpError()
             << "Failed to allocate buffers for tensor results.";
    }

    // Delegate to the linalg generic pattern.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(*op)) {
      finalizeBufferAllocationForGenericOp(rewriter, genericOp,
                                           adaptor.inputs(), newOutputBuffers);
      return success();
    }

    finalizeBufferAllocation(rewriter, op, adaptor.inputs(), newOutputBuffers);
    return success();
  }
};

bool IsBlockArgOfTiledLoop(Value tensor) {
  if (auto tensorLoad = tensor.getDefiningOp<memref::TensorLoadOp>())
    if (auto blockArgument = tensorLoad.memref().dyn_cast<BlockArgument>())
      if (isa<TiledLoopOp>(blockArgument.getOwner()->getParentOp()))
        return true;
  return false;
}

/// Convert `extract_slice %t [offsets][sizes][strides] -> %st` to an
/// alloc + copy pattern.
/// ```
///   %a = alloc(sizes)
///   %sv = subview %source [offsets][sizes][strides]
///   linalg_copy(%sv, %a)
/// ```
///
/// This pattern is arguable a std pattern once linalg::CopyOp becomes
/// std::CopyOp.
class ExtractSliceOpConverter
    : public OpConversionPattern<tensor::ExtractSliceOp> {
public:
  using OpConversionPattern<tensor::ExtractSliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    tensor::ExtractSliceOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value sourceMemref = adaptor.source();
    assert(sourceMemref.getType().isa<MemRefType>());

    // Block arguments of the tiled_loop can be bufferized inplace.
    if (IsBlockArgOfTiledLoop(op.source())) {
      Value subView = rewriter.create<memref::SubViewOp>(
          op.getLoc(), sourceMemref, op.getMixedOffsets(), op.getMixedSizes(),
          op.getMixedStrides());
      rewriter.replaceOp(op, subView);
      return success();
    }

    MemRefType subviewMemRefType =
        getTypeConverter()->convertType(op.getType()).cast<MemRefType>();
    // op.sizes() capture exactly the dynamic alloc operands matching the
    // subviewMemRefType thanks to subview/slice canonicalization and
    // verification.
    Value alloc = rewriter.create<memref::AllocOp>(
        op.getLoc(), subviewMemRefType, op.sizes());
    Value subView = rewriter.create<memref::SubViewOp>(
        op.getLoc(), sourceMemref, op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
    rewriter.create<linalg::CopyOp>(op.getLoc(), subView, alloc);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

/// Convert `insert_slice %source into %dest [offsets][sizes][strides] ->
/// %t` to an buffer_cast + subview + copy + tensor_load pattern.
/// buffer_cast and tensor_load are inserted automatically by the
/// conversion infra:
/// ```
///   %sv = subview %dest [offsets][sizes][strides]
///   linalg_copy(%source, %sv)
///   // replace with %dest
/// ```
///
/// This pattern is arguable a std pattern once linalg::CopyOp becomes
/// std::CopyOp.
class InsertSliceOpConverter
    : public OpConversionPattern<tensor::InsertSliceOp> {
public:
  using OpConversionPattern<tensor::InsertSliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    tensor::InsertSliceOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value sourceMemRef = adaptor.source();
    assert(sourceMemRef.getType().isa<MemRefType>());

    // For now, be conservative and copy the converted input memref.
    // In general, the converted input memref here could be aliased or could
    // point into constant memory, so mutating it would lead to miscompilations.
    // Block arguments of the tiled_loop can be bufferized inplace.
    Value destMemRef;
    if (IsBlockArgOfTiledLoop(op.dest()))
      destMemRef = adaptor.dest();
    else
      destMemRef = cloneMemref(op.getLoc(), adaptor.dest(), rewriter);
    assert(destMemRef.getType().isa<MemRefType>());

    // Take a subview to copy the small memref.
    Value subview = rewriter.create<memref::SubViewOp>(
        op.getLoc(), destMemRef, op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
    // Copy the small memref.
    rewriter.create<linalg::CopyOp>(op.getLoc(), sourceMemRef, subview);
    rewriter.replaceOp(op, destMemRef);
    return success();
  }
};

class TiledLoopOpConverter : public OpConversionPattern<TiledLoopOp> {
public:
  using OpConversionPattern<TiledLoopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TiledLoopOp tiledLoop, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    TiledLoopOp::Adaptor adaptor(operands, tiledLoop->getAttrDictionary());
    Location loc = tiledLoop.getLoc();
    if (tiledLoop.getNumResults() == 0)
      return failure();
    auto newTiledLoop = rewriter.create<TiledLoopOp>(
        loc, adaptor.lowerBound(), adaptor.upperBound(), adaptor.step(),
        adaptor.inputs(), adaptor.outputs(), adaptor.iterator_types(),
        adaptor.distribution_types());
    // Clone the region.
    BlockAndValueMapping bvm;
    bvm.map(tiledLoop.getInductionVars(), newTiledLoop.getInductionVars());

    OpBuilder innerBuilder =
        OpBuilder::atBlockEnd(newTiledLoop.getBody(), rewriter.getListener());

    // Remap input block arguments.
    SmallVector<Value, 2> inputs;
    for (auto en : llvm::zip(newTiledLoop.getRegionInputArgs(),
                             tiledLoop.getRegionInputArgs())) {
      auto &newInputArg = std::get<0>(en);
      if (!newInputArg.getType().isa<ShapedType>()) {
        inputs.push_back(std::get<0>(en));
        continue;
      }
      inputs.push_back(
          innerBuilder.create<memref::TensorLoadOp>(loc, newInputArg));
    }
    bvm.map(tiledLoop.getRegionInputArgs(), inputs);

    // Remap output block arguments.
    SmallVector<Value, 2> outputs;
    for (auto en : llvm::zip(newTiledLoop.getRegionOutputArgs(),
                             tiledLoop.getRegionOutputArgs())) {
      auto &newOutputArg = std::get<0>(en);
      if (!newOutputArg.getType().isa<ShapedType>()) {
        outputs.push_back(std::get<0>(en));
        continue;
      }
      outputs.push_back(
          innerBuilder.create<memref::TensorLoadOp>(loc, newOutputArg));
    }
    bvm.map(tiledLoop.getRegionOutputArgs(), outputs);

    for (auto &op : tiledLoop.getBody()->without_terminator())
      innerBuilder.clone(op, bvm);
    innerBuilder.create<linalg::YieldOp>(loc);
    rewriter.replaceOp(tiledLoop, newTiledLoop.outputs());
    return success();
  }
};

class VectorTransferReadOpConverter
    : public OpConversionPattern<vector::TransferReadOp> {
public:
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (readOp.getShapedType().isa<MemRefType>())
      return failure();
    vector::TransferReadOp::Adaptor adaptor(operands,
                                            readOp->getAttrDictionary());
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        readOp, readOp.getType(), adaptor.source(), adaptor.indices(),
        adaptor.permutation_map(), adaptor.padding(), adaptor.mask(),
        adaptor.in_bounds());
    return success();
  }
};

class VectorTransferWriteOpConverter
    : public OpConversionPattern<vector::TransferWriteOp> {
public:
  using OpConversionPattern<vector::TransferWriteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferWriteOp writeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (writeOp.getShapedType().isa<MemRefType>())
      return failure();
    vector::TransferWriteOp::Adaptor adaptor(operands,
                                             writeOp->getAttrDictionary());
    rewriter.create<vector::TransferWriteOp>(
        writeOp.getLoc(), adaptor.vector(), adaptor.source(), adaptor.indices(),
        adaptor.permutation_map(),
        adaptor.in_bounds() ? adaptor.in_bounds() : ArrayAttr());
    rewriter.replaceOp(writeOp, adaptor.source());
    return success();
  }
};
} // namespace

static Value materializeTensorLoad(OpBuilder &builder, TensorType type,
                                   ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(inputs[0].getType().isa<BaseMemRefType>());
  return builder.create<memref::TensorLoadOp>(loc, type, inputs[0]);
}

namespace {

/// A helper type converter class that automatically populates the relevant
/// materializations and type conversions for bufferization.
//
// The default BufferizeTypeConverter defined in "Transforms/Bufferize.h" does
// not properly support memrefs with non-default layout. Whenever a layout of
// memref changes during bufferization, target materialization call back would
// assert that the non-matching type is a tensor.
// There was an attempt to fix this behavior of dialect conversion in a more
// principal way in https://reviews.llvm.org/D93126 but it had to be reverted
// due to test failures outside of MLIR Core. It might make sense to revive this
// PR.
class CustomBufferizeTypeConverter : public BufferizeTypeConverter {
public:
  CustomBufferizeTypeConverter() {
    // Keep all types unchanged.
    addConversion([](Type type) { return type; });
    // Convert RankedTensorType to MemRefType.
    addConversion([](RankedTensorType type) -> Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });
    // Convert UnrankedTensorType to UnrankedMemRefType.
    addConversion([](UnrankedTensorType type) -> Type {
      return UnrankedMemRefType::get(type.getElementType(), 0);
    });
    addArgumentMaterialization(materializeTensorLoad);
    addSourceMaterialization(materializeTensorLoad);
    addTargetMaterialization([](OpBuilder &builder, BaseMemRefType type,
                                ValueRange inputs, Location loc) -> Value {
      assert(inputs.size() == 1);
      // Target materialization is invoked if the new operand type does not
      // match the expected type. A special case is when the new operand type is
      // a memref with a specified layout, i.e. non-empty affine map.
      // TODO(pifon) : Change how target materialization is invoked in dialect
      // conversion.
      if (auto memrefType = inputs[0].getType().dyn_cast<MemRefType>()) {
        assert(!memrefType.getAffineMaps().empty());
        return inputs[0];
      }
      assert(inputs[0].getType().isa<TensorType>());
      return builder.create<memref::BufferCastOp>(loc, type, inputs[0]);
    });
  }
};

/// Converts Linalg operations that work on tensor-type operands or results to
/// work on buffers.
struct LinalgBufferizePass : public LinalgBufferizeBase<LinalgBufferizePass> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget target(context);
    CustomBufferizeTypeConverter typeConverter;

    // Mark all Standard operations legal.
    target.addLegalDialect<AffineDialect, math::MathDialect,
                           memref::MemRefDialect, StandardOpsDialect,
                           tensor::TensorDialect>();
    target.addIllegalOp<InitTensorOp, tensor::ExtractSliceOp,
                        tensor::InsertSliceOp, PadTensorOp>();

    // Mark all Linalg operations illegal as long as they work on tensors.
    auto isLegalOperation = [&](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(isLegalOperation);
    target.addDynamicallyLegalOp<ConstantOp, vector::TransferReadOp,
                                 vector::TransferWriteOp>(isLegalOperation);

    RewritePatternSet patterns(&context);
    populateLinalgBufferizePatterns(typeConverter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgBufferizePass() {
  return std::make_unique<LinalgBufferizePass>();
}

void mlir::linalg::populateLinalgBufferizePatterns(
    BufferizeTypeConverter &typeConverter, RewritePatternSet &patterns) {
  // TODO: Drop this once tensor constants work in standard.
  // clang-format off
  patterns.add<
      BufferizeAnyLinalgOp,
      BufferizeFillOp,
      BufferizeInitTensorOp,
      BufferizeTensorReshapeOp<TensorExpandShapeOp>,
      BufferizeTensorReshapeOp<TensorCollapseShapeOp>,
      ExtractSliceOpConverter,
      InsertSliceOpConverter,
      TiledLoopOpConverter,
      VectorTransferReadOpConverter,
      VectorTransferWriteOpConverter
    >(typeConverter, patterns.getContext());
  // clang-format on
  patterns.add<GeneralizePadTensorOpPattern>(patterns.getContext());
}
