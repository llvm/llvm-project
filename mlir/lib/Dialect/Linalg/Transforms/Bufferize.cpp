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
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace ::mlir;
using namespace ::mlir::linalg;

static Value cloneMemref(Location loc, Value memref, OpBuilder &b) {
  auto memrefType = memref.getType().cast<MemRefType>();
  auto alloc =
      b.create<AllocOp>(loc, memrefType, getDynOperands(loc, memref, b));
  b.create<linalg::CopyOp>(loc, memref, alloc);
  return alloc;
}

static LogicalResult
allocateBuffersForResults(Location loc, LinalgOp linalgOp,
                          linalg::GenericOpAdaptor &adaptor,
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
    Value resultTensor = adaptor.outputs()[resultIndex];

    // Clone output buffers whose value is actually used.
    if (linalgOp.payloadUsesValueFromOutputOperandIndex(resultIndex)) {
      resultBuffers.push_back(cloneMemref(loc, resultTensor, b));
      continue;
    }

    if (auto alloc = resultTensor.getDefiningOp<AllocOp>()) {
      resultBuffers.push_back(resultTensor);
      continue;
    }
    // Allocate buffers for statically-shaped results.
    if (memrefType.hasStaticShape()) {
      resultBuffers.push_back(b.create<AllocOp>(loc, memrefType));
      continue;
    }

    resultBuffers.push_back(b.create<AllocOp>(
        loc, memrefType, getDynOperands(loc, resultTensor, b)));
  }
  return success();
}

/// Specialization for `linalg::GenericOp` and `linalg::IndexedGenericOp`.
/// A pattern to convert Generic Linalg operations which work on tensors to
/// use buffers. BufferPlacement pass should be later used to move
/// Alloc operations to the correct positions and insert the missing Dealloc
/// operations in the correct places.
template <typename GenericOpTy>
static void
finalizeBufferAllocationForGenericOp(ConversionPatternRewriter &rewriter,
                                     GenericOpTy genericOp, ValueRange inputs,
                                     ValueRange outputs) {
  // Generate a new linalg operation that works on buffers.
  auto newGenericOp = rewriter.create<GenericOpTy>(
      genericOp.getLoc(),
      /*resultTensorTypes=*/llvm::None,
      /*inputs=*/inputs,
      /*outputs=*/outputs, genericOp.indexing_maps(),
      genericOp.iterator_types(), genericOp.docAttr(),
      genericOp.library_callAttr(), genericOp.sparseAttr());

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
  assert(!isa<linalg::IndexedGenericOp>(linalgOp.getOperation()));
  SmallVector<Value, 8> newOperands = inputs;
  newOperands.append(outputs.begin(), outputs.end());
  auto otherOperands = linalgOp.getAssumedNonShapedOperands();
  newOperands.append(otherOperands.begin(), otherOperands.end());
  linalgOp.clone(rewriter, linalgOp.getLoc(),
                 /*resultTypes=*/ArrayRef<Type>{}, newOperands);
  // Replace the results of the old op with the new output buffers.
  rewriter.replaceOp(linalgOp, outputs);
}

//===----------------------------------------------------------------------===//
// Bufferization patterns.
//===----------------------------------------------------------------------===//

namespace {

/// Generic conversion pattern that matches any LinalgOp. This avoids template
/// instantiating one pattern for each LinalgOp.
class BufferizeInitTensorOp : public OpConversionPattern<InitTensorOp> {
public:
  using OpConversionPattern<InitTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InitTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    linalg::InitTensorOpAdaptor adaptor(operands, op->getAttrDictionary());
    rewriter.replaceOpWithNewOp<AllocOp>(
        op, getTypeConverter()->convertType(op.getType()).cast<MemRefType>(),
        adaptor.sizes());
    return success();
  }
};

/// Generic conversion pattern that matches any LinalgOp. This avoids template
/// instantiating one pattern for each LinalgOp.
class BufferizeAnyLinalgOp : public ConversionPattern {
public:
  BufferizeAnyLinalgOp(TypeConverter &typeConverter)
      : ConversionPattern(/*benefit=*/1, typeConverter, MatchAnyOpTypeTag()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      return failure();

    // We abuse the GenericOpAdaptor here.
    // TODO: Manually create an Adaptor that captures inputs and outputs for all
    // linalg::LinalgOp interface ops.
    linalg::GenericOpAdaptor adaptor(operands, op->getAttrDictionary());

    Location loc = linalgOp.getLoc();
    SmallVector<Value, 2> newOutputBuffers;

    if (failed(allocateBuffersForResults(loc, linalgOp, adaptor,
                                         newOutputBuffers, rewriter))) {
      linalgOp.emitOpError()
          << "Failed to allocate buffers for tensor results.";
      return failure();
    }

    // Delegate to the linalg generic pattern.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      finalizeBufferAllocationForGenericOp<GenericOp>(
          rewriter, genericOp, adaptor.inputs(), newOutputBuffers);
      return success();
    }

    // Delegate to the linalg indexed generic pattern.
    if (auto genericOp = dyn_cast<linalg::IndexedGenericOp>(op)) {
      finalizeBufferAllocationForGenericOp<IndexedGenericOp>(
          rewriter, genericOp, adaptor.inputs(), newOutputBuffers);
      return success();
    }

    finalizeBufferAllocation(rewriter, linalgOp, adaptor.inputs(),
                             newOutputBuffers);
    return success();
  }
};

// Extract int64_t values from the assumed ArrayAttr of IntegerAttr.
static SmallVector<int64_t, 4> extractFromI64ArrayAttr(Attribute attr) {
  return llvm::to_vector<4>(
      llvm::map_range(attr.cast<ArrayAttr>(), [](Attribute a) -> int64_t {
        return a.cast<IntegerAttr>().getInt();
      }));
}

/// Convert `subtensor %t [offsets][sizes][strides] -> %st` to an alloc + copy
/// pattern.
/// ```
///   %a = alloc(sizes)
///   %sv = subview %source [offsets][sizes][strides]
///   linalg_copy(%sv, %a)
/// ```
///
/// This pattern is arguable a std pattern once linalg::CopyOp becomes
/// std::CopyOp.
class SubTensorOpConverter : public OpConversionPattern<SubTensorOp> {
public:
  using OpConversionPattern<SubTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SubTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    SubTensorOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value sourceMemref = adaptor.source();
    assert(sourceMemref.getType().isa<MemRefType>());

    MemRefType subviewMemRefType =
        getTypeConverter()->convertType(op.getType()).cast<MemRefType>();
    // op.sizes() capture exactly the dynamic alloc operands matching the
    // subviewMemRefType thanks to subview/subtensor canonicalization and
    // verification.
    Value alloc =
        rewriter.create<AllocOp>(op.getLoc(), subviewMemRefType, op.sizes());
    Value subView = rewriter.create<SubViewOp>(
        op.getLoc(), sourceMemref, extractFromI64ArrayAttr(op.static_offsets()),
        extractFromI64ArrayAttr(op.static_sizes()),
        extractFromI64ArrayAttr(op.static_strides()), op.offsets(), op.sizes(),
        op.strides());
    rewriter.create<linalg::CopyOp>(op.getLoc(), subView, alloc);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

/// Convert `subtensor_insert %source into %dest [offsets][sizes][strides] ->
/// %t` to an tensor_to_memref + subview + copy + tensor_load pattern.
/// tensor_to_memref and tensor_load are inserted automatically by the
/// conversion infra:
/// ```
///   %sv = subview %dest [offsets][sizes][strides]
///   linalg_copy(%source, %sv)
///   // replace with %dest
/// ```
///
/// This pattern is arguable a std pattern once linalg::CopyOp becomes
/// std::CopyOp.
class SubTensorInsertOpConverter
    : public OpConversionPattern<SubTensorInsertOp> {
public:
  using OpConversionPattern<SubTensorInsertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SubTensorInsertOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    SubTensorInsertOpAdaptor adaptor(operands, op->getAttrDictionary());
    Value sourceMemRef = adaptor.source();
    assert(sourceMemRef.getType().isa<MemRefType>());

    // For now, be conservative and copy the converted input memref.
    // In general, the converted input memref here could be aliased or could
    // point into constant memory, so mutating it would lead to miscompilations.
    Value destMemRef = cloneMemref(op.getLoc(), adaptor.dest(), rewriter);
    assert(destMemRef.getType().isa<MemRefType>());

    // Take a subview to copy the small memref.
    Value subview = rewriter.create<SubViewOp>(
        op.getLoc(), destMemRef, extractFromI64ArrayAttr(op.static_offsets()),
        extractFromI64ArrayAttr(op.static_sizes()),
        extractFromI64ArrayAttr(op.static_strides()), adaptor.offsets(),
        adaptor.sizes(), adaptor.strides());
    // Copy the small memref.
    rewriter.create<linalg::CopyOp>(op.getLoc(), sourceMemRef, subview);
    rewriter.replaceOp(op, destMemRef);
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
    BufferizeTypeConverter typeConverter;

    // Mark all Standard operations legal.
    target.addLegalDialect<AffineDialect, StandardOpsDialect>();
    target.addIllegalOp<InitTensorOp, SubTensorOp, SubTensorInsertOp>();

    // Mark all Linalg operations illegal as long as they work on tensors.
    auto isLegalOperation = [&](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(isLegalOperation);
    target.addDynamicallyLegalOp<ConstantOp>(isLegalOperation);

    OwningRewritePatternList patterns;
    populateLinalgBufferizePatterns(&context, typeConverter, patterns);
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
    MLIRContext *context, BufferizeTypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  patterns.insert<BufferizeAnyLinalgOp>(typeConverter);
  // TODO: Drop this once tensor constants work in standard.
  // clang-format off
  patterns.insert<
      BufferizeInitTensorOp,
      SubTensorOpConverter,
      SubTensorInsertOpConverter
    >(typeConverter, context);
  // clang-format on
}
