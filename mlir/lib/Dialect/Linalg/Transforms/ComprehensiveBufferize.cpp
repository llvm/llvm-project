//===- ComprehensiveBufferize.cpp - Single pass bufferization -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Perform inplace bufferization within function boundaries.
// This is a specialized pass that supports inplace analysis for a fixed subset
// of ops that have well-defined inplace semantics.
// This pass caters to high-performance codegen where buffer reuse is deemed
// necessary: the pass should fail if the bufferized form of the function needs
// to return any buffer.
// Generic control-flow and branching are unsupported.
// Composability with extensible set of ops is not a first-class concern.
//
// Bufferization occurs by:
//  a. performing an inPlace analysis `inPlaceAnalysisFuncOpInternals`
//     which marks each operation within the function with the
//     `kInPlaceResultsAttrName` attribute.
//  b. traversing each operation in the function and rewriting it in
//     buffer form and keeping a BlockAndValueMapping mapping of the
//     rewrites. New allocations are introduced during this step.
//     TODO: Allocation + depending op hoisting to outermost enclosing
//     sequential scope.
//  c. at the end of this bufferization, 2 cases may occur:
//     * inplaceable function arguments may be reused in place after the
//       function itself has been bufferized. This is encoded by IR resembling:
//
// ```
//   #map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
//   func @foo(%A: tensor<?xf32> {linalg.inplaceable = true}) -> tensor<?xf32> {
//     %0 = memref.buffer_cast %A : memref<?xf32, #map>
//     // ... uses of %0
//     %res = memref.tensor_load %0 : memref<?xf32, #map>
//     return %res : tensor<?xf32>
//   }
// ```
//
//       this is the cue for the bufferization of the function foo (and calls to
//       it) may bufferize to `func @foo(%A: memref<?xf32, some_layout>)`.
//       To fully achieve bufferization, an additional analysis is needed to
//       determine whether function argument/operand pairs bufferize to a single
//       inplace buffer argument (i.e. functions may return tensors in arbitrary
//       order that may not match argument numbers).
//     * results that don't map to an inplaceable function argument must be
//       allocated. Since memref semantics wrt ownership of the underlying
//       memory region are not well-defined, comprehensive bufferization chooses
//       to perform allocations in a scoped fashion: returning memrefs is always
//       considered illegal. Such scenarios are encoded by IR resembling:
//
// ```
//   #map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
//   func @foo(%A: tensor<?xf32> {linalg.inplaceable = true}) -> tensor<?xf32> {
//     %0 = memref.buffer_cast %A : memref<?xf32, #map>
//     %1 = memref.dim %0, %c0 : memref<?xf32, #map>
//     %2 = memref.alloc(%1) : memref<?xf32>
//     %3 = memref.cast %2 : memref<?xf32> to memref<?xf32, #map>
//     // ... uses of %3
//     memref.dealloc %2 : memref<?xf32, #map>
//     %res = memref.tensor_load %3 : memref<?xf32, #map>
//     return %res : tensor<?xf32>
//   }
// ```
//
//       this is the cue for the bufferization of the function foo (and calls to
//       it) that it must bufferize to
//       `func @foo(%A: memref<?xf32, some_layout>,
//                  %B: memref<?xf32, some_layout>)` (i.e. make a cloned
//       allocation of the result tensor)
//       To fully achieve bufferization, the alloc/dealloc pair must be lifted
//       out of the function at each call site.
//
//  Lastly, note that layout map chosen to bufferize is the most dynamic
//  canonical strided layout of the proper rank. This ensures compatibility with
//  expected layouts after transformations. Combinations of memref.cast +
//  canonicalization are responsible for clean ups.

#include "PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferUtils.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "comprehensive-func-bufferize"

using namespace mlir;
using namespace linalg;
using namespace tensor;

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

//===----------------------------------------------------------------------===//
// Op-specific semantics helper to retrieve matching inplaceable result.
//===----------------------------------------------------------------------===//

/// Return the OpResult that matches an operand.
/// Return null if no such result exists.
OpResult getMatchingOpResult(LinalgOp linalgOp, OpOperand &opOperand) {
  if (!opOperand.get().getType().isa<RankedTensorType>())
    return OpResult();
  // For now assume inputs are never inplaceable.
  // TODO: refine this.
  if (opOperand.getOperandNumber() < linalgOp.getNumInputs())
    return OpResult();
  // For now assume if the operand appears twice, it is not inplaceable.
  // TODO: refine this.
  for (auto &opOperand2 : linalgOp->getOpOperands()) {
    if (opOperand.getOperandNumber() == opOperand2.getOperandNumber())
      continue;
    if (opOperand.get() == opOperand2.get())
      return OpResult();
  }
  int64_t outputOperandIndex =
      opOperand.getOperandNumber() - linalgOp.getNumInputs();
  int64_t numOutputBuffers = 0;
  for (unsigned idx = 0; idx < outputOperandIndex; ++idx)
    if (!linalgOp.getOutputShapedType(idx).isa<TensorType>())
      ++numOutputBuffers;
  return linalgOp->getResult(outputOperandIndex - numOutputBuffers);
}

/// Return the OpResult that matches an operand.
/// Return null if no such result exists.
OpResult getMatchingOpResult(VectorTransferOpInterface op,
                             OpOperand &opOperand) {
  if (opOperand.get() != op.source() ||
      !op.source().getType().isa<TensorType>())
    return OpResult();
  return op->getResult(0);
}

/// Return the OpResult that matches an operand.
/// Return null if no such result exists.
OpResult getMatchingOpResult(SubTensorInsertOp op, OpOperand &opOperand) {
  if (opOperand.get() != op.dest())
    return OpResult();
  return op->getResult(0);
}

/// Determine which results may be reused inplace by the bufferization
/// patterns of `bufferizeFuncOpInternals`.
/// The inplace analysis uses this information along with interfering read
/// analysis to determine which op results reuse the same buffer as some
/// operand.
OpResult getMatchingOpResult(OpOperand &opOperand) {
  return llvm::TypeSwitch<Operation *, OpResult>(opOperand.getOwner())
      // clang-format off
        // Ops that perform destructive updates on operand(s) to produce
        // result(s).
        .Case<LinalgOp,
              SubTensorInsertOp,
              VectorTransferOpInterface>(
            [&](auto op) { return getMatchingOpResult(op, opOperand); })
        // Other ops.
        .Case<SubTensorOp>([&](auto op) { return OpResult(); })
        .Default([&](Operation *op) { return OpResult(); });
  // clang-format on
}

//===----------------------------------------------------------------------===//
// Bufferization-specific attribute manipulation.
//===----------------------------------------------------------------------===//

/// Attribute marker to specify op results that can be bufferized inPlace.
constexpr StringLiteral kInPlaceResultsAttrName = "__inplace_results_attr__";

// TODO: proper enum.
enum class InPlaceSpec {
  False,
  True,
  None,
};

static StringRef stringify(InPlaceSpec val) {
  switch (val) {
  case InPlaceSpec::False:
    return "false";
  case InPlaceSpec::True:
    return "true";
  case InPlaceSpec::None:
    return "none";
  }
  return "";
}

static Optional<InPlaceSpec> symbolize(StringRef str) {
  return StringSwitch<Optional<InPlaceSpec>>(str)
      .Case("false", InPlaceSpec::False)
      .Case("true", InPlaceSpec::True)
      .Case("none", InPlaceSpec::None)
      .Default(None);
}

/// Mark whether OpResult can actually be bufferized inplace. If `inPlace` is
/// `InPlaceSpec::True`, the use-def chain analysis has guaranteed that no
/// subsequent write would occur to the bufferized tensor value (i.e. the result
/// can be bufferized inPlace).
static void setInPlaceOpResult(OpResult opResult,
                               InPlaceSpec inPlace = InPlaceSpec::True) {
  if (!opResult)
    return;

  Operation *op = opResult.getOwner();
  auto attr =
      op->getAttr(kInPlaceResultsAttrName).dyn_cast_or_null<ArrayAttr>();
  SmallVector<StringRef> inPlaceVector =
      attr ? SmallVector<StringRef>(
                 llvm::to_vector<4>(attr.getAsValueRange<StringAttr>()))
           : SmallVector<StringRef>(op->getNumResults(),
                                    stringify(InPlaceSpec::None));
  LLVM_DEBUG(DBGS() << "Set inPlace=" << stringify(inPlace) << ": " << *op
                    << " @idx=" << opResult.getResultNumber() << "\n");
  inPlaceVector[opResult.getResultNumber()] = stringify(inPlace);
  op->setAttr(kInPlaceResultsAttrName,
              OpBuilder(op).getStrArrayAttr(inPlaceVector));
}

/// Get the InPlaceSpec attribute entry `kInPlaceResultsAttrName` for
/// `opResult`. If the result is `InPlaceSpec::True`, the use-def chain analysis
/// has guaranteed that no subsequent read of the tensor value occurs and the
/// result can be buferized inPlace.
/// If no InPlaceSpec attribute has been set for `opResult`, return
/// InPlaceSpec::None.
static InPlaceSpec getInPlace(OpResult opResult) {
  if (!opResult)
    return InPlaceSpec::None;

  Operation *op = opResult.getOwner();
  auto attr =
      op->getAttr(kInPlaceResultsAttrName).dyn_cast_or_null<ArrayAttr>();
  if (!attr)
    return InPlaceSpec::None;

  // Must return a proper value.
  return *symbolize(*(attr.getAsValueRange<StringAttr>().begin() +
                      opResult.getResultNumber()));
}

/// Get inPlace information for `bbArg`.
/// If it does not come from a function, return InPlaceSpec::False.
static InPlaceSpec getInPlace(BlockArgument bbArg) {
  auto funcOp = dyn_cast<FuncOp>(bbArg.getOwner()->getParentOp());
  if (!funcOp)
    return InPlaceSpec::False;
  auto attr = funcOp.getArgAttrOfType<BoolAttr>(
      bbArg.getArgNumber(), LinalgDialect::kInplaceableAttrName);
  if (!attr)
    return InPlaceSpec::None;
  return attr.getValue() ? InPlaceSpec::True : InPlaceSpec::False;
}

//===----------------------------------------------------------------------===//
// Bufferization-specific BlockAndValueMapping support with debugging.
//===----------------------------------------------------------------------===//

/// Wrapper for better debugging.
static void map(BlockAndValueMapping &bvm, ValueRange keys, ValueRange values) {
  assert(!keys.empty() && "Unexpected empty keys");
  LLVM_DEBUG(DBGS() << "Map: " << keys.front() << " to " << values.front()
                    << "\n");
  return bvm.map(keys, values);
}

/// Wrapper for better debugging.
static void map(BlockAndValueMapping &bvm, Value key, Value value) {
  LLVM_DEBUG(DBGS() << "Map: " << key << " to " << value << "\n");
  return bvm.map(key, value);
}

/// Wrapper for better debugging.
static Value lookup(BlockAndValueMapping &bvm, Value key) {
  // TODO: if key comes from bbArg, forward.
  assert(key.getType().isa<TensorType>());
  if (!bvm.lookupOrNull(key)) {
    if (auto bbArg = key.dyn_cast<BlockArgument>()) {
      if (isa<FuncOp>(key.getParentBlock()->getParentOp()))
        key.getParentBlock()->getParentOp()->dump();
      else
        key.getParentBlock()->getParentOp()->getParentOfType<FuncOp>()->dump();
      bbArg.getOwner()->getParentOp()->dump();
    } else {
      key.getDefiningOp()->getParentOfType<FuncOp>()->dump();
    }
    llvm::errs() << "NO VALUE FOR KEY: " << key << "\n";
    return Value();
  }
  return bvm.lookup(key);
}

//===----------------------------------------------------------------------===//
// Bufferization-specific MemRefType support.
//===----------------------------------------------------------------------===//

/// Return a contiguous MemRefType (i.e. with canonical/empty layout map) with
/// the same shape as `shapedType` and specified `layout` and `addressSpace`.
static MemRefType getContiguousMemRefType(ShapedType shapedType,
                                          ArrayRef<AffineMap> layout = {},
                                          unsigned addressSpace = 0) {
  if (RankedTensorType tensorType = shapedType.dyn_cast<RankedTensorType>())
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                           layout, addressSpace);
  MemRefType memrefType = shapedType.cast<MemRefType>();
  return MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                         layout, addressSpace);
}

/// Return a contiguous MemRefType (i.e. with canonical/empty layout map) with
/// the same shape as `shapedType` and specified `layout` and `addressSpace` or
/// an UnrankedMemRefType otherwise.
static Type getContiguousOrUnrankedMemRefType(Type type,
                                              ArrayRef<AffineMap> layout = {},
                                              unsigned addressSpace = 0) {
  if (type.isa<RankedTensorType, MemRefType>())
    return getContiguousMemRefType(type.cast<ShapedType>(), layout,
                                   addressSpace);
  assert(layout.empty() && "expected empty layout with UnrankedMemRefType");
  return UnrankedMemRefType::get(getElementTypeOrSelf(type), addressSpace);
}

/// Return a MemRefType to which the `tensorType` can be bufferized in a
/// composable fashion. The layout must be the most dynamic possible and
/// canonicalize away once bufferization is finished.
static MemRefType getDynamicMemRefType(RankedTensorType tensorType,
                                       unsigned addressSpace = 0) {
  // TODO: address space decisions to connect with the actual alloc.
  int64_t dynamicOffset = ShapedType::kDynamicStrideOrOffset;
  SmallVector<int64_t> dynamicStrides(tensorType.getRank(),
                                      ShapedType::kDynamicStrideOrOffset);
  AffineMap stridedLayout = makeStridedLinearLayoutMap(
      dynamicStrides, dynamicOffset, tensorType.getContext());
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         stridedLayout, addressSpace);
}

//===----------------------------------------------------------------------===//
// Bufferization-specific scoped alloc/dealloc insertion support.
//===----------------------------------------------------------------------===//

/// Create an Allocop/DeAllocOp pair, where the AllocOp is after
/// `shapedValue.getDefiningOp` (or at the top of the block in case of a bbArg)
/// and the DeallocOp is at the end of the block.
static Value createNewAllocDeallocPairForShapedValue(
    OpBuilder &b, Location loc, Value shapedValue,
    SmallVector<Value, 4> dynOperands = {}) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // TODO: non-zero address space.
  // TODO: layout information if relevant.
  // Cannot allocate an unranked memref so just always go for the contiguous
  // form.
  MemRefType allocMemRefType =
      getContiguousMemRefType(shapedValue.getType().cast<ShapedType>());
  assert(shapedValue.getType().isa<ShapedType>());
  MemRefType memRefType = shapedValue.getType().dyn_cast<MemRefType>();
  memRefType = memRefType ? memRefType : allocMemRefType;

  if (auto bbArg = shapedValue.dyn_cast<BlockArgument>()) {
    b.setInsertionPointToStart(bbArg.getOwner());
    loc = bbArg.getOwner()->getParentOp()->getLoc();
  } else {
    b.setInsertionPointAfter(shapedValue.getDefiningOp());
    loc = shapedValue.getDefiningOp()->getLoc();
  }

  // If the dynOperands are not passed explicity, copmpute them.
  // This circumvents currently missing dim(init_tensor) canonicalizations.
  // TODO: dim(init_tensor) canonicalization.
  if (dynOperands.empty()) {
    for (auto dim : llvm::enumerate(memRefType.getShape()))
      if (dim.value() == ShapedType::kDynamicSize)
        dynOperands.push_back(
            b.create<memref::DimOp>(loc, shapedValue, dim.index()));
  }

  Value allocated =
      b.create<memref::AllocOp>(loc, allocMemRefType, dynOperands);
  Value casted = allocated;
  if (memRefType != allocMemRefType)
    casted = b.create<memref::CastOp>(loc, memRefType, allocated);
  b.setInsertionPoint(allocated.getParentBlock()->getTerminator());
  b.create<memref::DeallocOp>(loc, allocated);
  return casted;
}

//===----------------------------------------------------------------------===//
// Bufferization as simple BlockAndValueMapping rewrites.
//===----------------------------------------------------------------------===//

/// Helper function for LinalgOp bufferization.
/// Operate on mixed tensor + buffer Linalg ops for progressive bufferization.
/// Allocate the output buffers for the remaining tensor output operands of
/// the Linalg op. If the tensor is an "init" tensor (i.e. its value is
/// actually used in the payload region), we additionally copy the original
/// value into the newly allocated buffer.
static LogicalResult
allocateBuffersForResults(OpBuilder &b, Location loc, LinalgOp op,
                          SmallVectorImpl<Value> &resultBuffers,
                          BlockAndValueMapping &bvm) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // Lazily compute loopRanges.
  SmallVector<Range, 4> loopRanges;

  // Linalg invariant: output tensors and result match 1-1.
  assert(op.getNumOutputTensors() == op->getNumResults());
  for (auto &opOperand : op.getOutputOpOperands()) {
    Value output = opOperand.get();
    if (output.getType().isa<MemRefType>()) {
      resultBuffers.push_back(output);
      continue;
    }

    // If output tensor is marked inPlace, just use the buffer.
    // The following uses internal knowledge of the position of tied operand /
    // results.
    OpResult tiedResult = getMatchingOpResult(op, opOperand);
    if (getInPlace(tiedResult) == InPlaceSpec::True) {
      Value v = lookup(bvm, output);
      if (!v)
        return failure();
      resultBuffers.push_back(v);
      continue;
    }

    Value dimTensor = bvm.lookupOrDefault(output);
    Value alloc = createNewAllocDeallocPairForShapedValue(b, loc, dimTensor);
    b.setInsertionPointAfter(alloc.getDefiningOp());
    resultBuffers.push_back(alloc);

    // Additionally, if the output buffer is used, clone its value for now.
    if (op.payloadUsesValueFromOpOperand(&opOperand)) {
      Value v = lookup(bvm, output);
      if (!v)
        return failure();
      b.create<CopyOp>(loc, v, alloc);
    }
  }
  if (op->getNumResults())
    map(bvm, op->getResults(), resultBuffers);

  return success();
}

static void finalizeBufferAllocation(OpBuilder &b, LinalgOp op,
                                     ValueRange inputs, ValueRange outputs,
                                     BlockAndValueMapping &bvm) {
  SmallVector<Value, 8> newOperands = inputs;
  newOperands.append(outputs.begin(), outputs.end());
  auto otherOperands = op.getAssumedNonShapedOperands();
  newOperands.append(otherOperands.begin(), otherOperands.end());
  Location loc = op.getLoc();
  op.clone(b, loc, /*resultTypes=*/TypeRange{}, newOperands);

  // Replace the results of the old op with the new output buffers.
  if (op->getNumResults())
    map(bvm, op->getResults(), outputs);
  if (!op.hasTensorSemantics())
    op->erase();
}

/// Generic conversion for any LinalgOp.
/// Operate on mixed tensor + buffer Linalg ops for progressive bufferization.
static LogicalResult bufferize(OpBuilder &b, LinalgOp op,
                               BlockAndValueMapping &bvm) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  if (op.hasBufferSemantics())
    return failure();

  LLVM_DEBUG(DBGS() << "bufferize: " << *op << "\n");

  b.setInsertionPoint(op);
  Location loc = op.getLoc();
  SmallVector<Value, 2> newInputBuffers;
  newInputBuffers.reserve(op.getNumInputs());
  for (Value in : op.getInputs()) {
    Value v = lookup(bvm, in);
    if (!v)
      return failure();
    newInputBuffers.push_back(v);
  }
  SmallVector<Value, 2> newOutputBuffers;
  if (failed(allocateBuffersForResults(b, loc, op, newOutputBuffers, bvm)))
    return failure();
  finalizeBufferAllocation(b, op, newInputBuffers, newOutputBuffers, bvm);
  return success();
}

/// DimOp tensor operand is modified inplace. This allows leaving dead tensors
/// behind that will get DCE'd.
static LogicalResult bufferize(OpBuilder &b, memref::DimOp dimOp,
                               BlockAndValueMapping &bvm) {
  if (dimOp.memrefOrTensor().getType().isa<RankedTensorType>()) {
    Value v = lookup(bvm, dimOp.memrefOrTensor());
    if (!v)
      return failure();
    dimOp.memrefOrTensorMutable().assign(v);
  }
  return success();
}

/// FuncOp always creates TensorToMemRef ops.
static LogicalResult bufferize(OpBuilder &b, FuncOp funcOp,
                               BlockAndValueMapping &bvm) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(&funcOp.body().front());
  for (auto bbArg : funcOp.getArguments()) {
    auto tensorType = bbArg.getType().dyn_cast<TensorType>();
    if (!tensorType)
      continue;
    auto rankedTensorType = tensorType.dyn_cast<RankedTensorType>();
    // Cast the tensor to the most dynamic buffer possible. Further
    // canonicalizations will clean up.
    Type memRefType = rankedTensorType
                          ? getDynamicMemRefType(rankedTensorType)
                          : getContiguousOrUnrankedMemRefType(tensorType);
    Value tensorToMemref =
        b.create<memref::BufferCastOp>(funcOp.getLoc(), memRefType, bbArg);
    map(bvm, bbArg, tensorToMemref);
  }
  return success();
}

/// ReturnOp always creates memref::TensorLoadOp.
static LogicalResult bufferize(OpBuilder &b, ReturnOp returnOp,
                               BlockAndValueMapping &bvm) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(returnOp);

  assert(isa<FuncOp>(returnOp->getParentOp()) &&
         "only support FuncOp parent for ReturnOp");
  for (OpOperand &operand : returnOp->getOpOperands()) {
    auto tensorType = operand.get().getType().dyn_cast<TensorType>();
    if (!tensorType)
      continue;
    Value v = lookup(bvm, operand.get());
    if (!v)
      return failure();
    operand.set(b.create<memref::TensorLoadOp>(returnOp.getLoc(), v));
  }
  return success();
}

/// Bufferize SubTensorOp to subview with optional alloc + copy depending on
/// whether or not it is marked inplaceable.
/// Note that `getMatchingOpResult` on a SubTensorOp always returns null.
/// As consequence a SubTensorOp always alloc + copy when taken in isolation.
static LogicalResult bufferize(OpBuilder &b, SubTensorOp subTensorOp,
                               BlockAndValueMapping &bvm) {
  LLVM_DEBUG(DBGS() << "bufferize: " << *subTensorOp << "\n");

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(subTensorOp);

  Location loc = subTensorOp.getLoc();
  // Bail if source was not bufferized.
  Value srcMemref = lookup(bvm, subTensorOp.source());
  if (!srcMemref)
    return failure();
  auto srcMemrefType = srcMemref.getType().cast<MemRefType>();
  auto dstTensorType = subTensorOp.result().getType().cast<RankedTensorType>();

  // If not inplaceable, alloc.
  Value alloc;
  auto inPlace = getInPlace(subTensorOp->getResult(0));
  if (inPlace != InPlaceSpec::True) {
    alloc =
        createNewAllocDeallocPairForShapedValue(b, loc, subTensorOp.result());
    b.setInsertionPointAfter(alloc.getDefiningOp());
  }

  // Bufferize to subview.
  auto subviewMemRefType =
      memref::SubViewOp::inferRankReducedResultType(
          dstTensorType.getRank(), srcMemrefType, subTensorOp.getMixedOffsets(),
          subTensorOp.getMixedSizes(), subTensorOp.getMixedStrides())
          .cast<MemRefType>();
  Value subView = b.create<memref::SubViewOp>(
      loc, subviewMemRefType, srcMemref, subTensorOp.getMixedOffsets(),
      subTensorOp.getMixedSizes(), subTensorOp.getMixedStrides());

  /// If not inplaceable, copy.
  if (alloc) {
    b.create<CopyOp>(subTensorOp.getLoc(), subView, alloc);
    subView = alloc;
  }

  map(bvm, subTensorOp.result(), subView);
  return success();
}

static LogicalResult bufferize(OpBuilder &b,
                               SubTensorInsertOp subTensorInsertOp,
                               BlockAndValueMapping &bvm) {
  LLVM_DEBUG(DBGS() << "bufferize: " << *subTensorInsertOp << "\n");

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(subTensorInsertOp);
  Location loc = subTensorInsertOp.getLoc();

  Value dstMemref = lookup(bvm, subTensorInsertOp.dest());
  if (!dstMemref)
    return failure();
  auto inPlace = getInPlace(subTensorInsertOp->getResult(0));
  if (inPlace != InPlaceSpec::True) {
    // Since subtensor_insert arise from tiling and introducing loops, this case
    // is generally a deal breaker. When used with loops, this ends up cloning
    // the whole tensor on every single iteration and is a symptom of a
    // catastrophically bad scheduling decision.
    // TODO: be very loud about it or even consider failing the pass.
    Value newDstMemref = createNewAllocDeallocPairForShapedValue(
        b, loc, subTensorInsertOp.result());
    b.setInsertionPointAfter(newDstMemref.getDefiningOp());
    b.create<CopyOp>(subTensorInsertOp.getLoc(), dstMemref, newDstMemref);
    dstMemref = newDstMemref;
  }
  auto dstMemrefType = dstMemref.getType().cast<MemRefType>();

  Value srcMemref = lookup(bvm, subTensorInsertOp.source());
  if (!srcMemref)
    return failure();
  auto subviewMemRefType =
      memref::SubViewOp::inferRankReducedResultType(
          subTensorInsertOp.getSourceType().getRank(), dstMemrefType,
          subTensorInsertOp.getMixedOffsets(),
          subTensorInsertOp.getMixedSizes(),
          subTensorInsertOp.getMixedStrides())
          .cast<MemRefType>();

  // A copy of the source buffer is needed if either:
  //   - The producer of `source` is not inplace. This is the case where a
  //     subtensor is computed out of place into the inplace full tensor.
  //   - The result is not inplace. This is the case where the whole tensor is
  //     cloned and the clone needs to be updated.
  Value source = subTensorInsertOp.source();
  InPlaceSpec inPlaceProducer = InPlaceSpec::None;
  if (auto opResult = source.dyn_cast<OpResult>())
    inPlaceProducer = getInPlace(opResult);
  else
    inPlaceProducer = getInPlace(source.cast<BlockArgument>());
  if (inPlaceProducer != InPlaceSpec::True) {
    LLVM_DEBUG(DBGS() << "subtensor_insert needs extra source copy: " << source
                      << " -> copy\n");
    // Take a subview of the dst.
    Value subView = b.create<memref::SubViewOp>(
        loc, subviewMemRefType, dstMemref, subTensorInsertOp.getMixedOffsets(),
        subTensorInsertOp.getMixedSizes(), subTensorInsertOp.getMixedStrides());
    b.create<CopyOp>(subTensorInsertOp.getLoc(), srcMemref, subView);
  }

  map(bvm, subTensorInsertOp.result(), dstMemref);

  return success();
}

static LogicalResult bufferize(OpBuilder &b, VectorTransferOpInterface op,
                               BlockAndValueMapping &bvm) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Location loc = op.getLoc();

  if (op.getShapedType().isa<MemRefType>())
    return failure();

  LLVM_DEBUG(DBGS() << "bufferize: " << *op << "\n");

  /// transfer_read from buffer always reads from the bufferized op.source().
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op.getOperation())) {
    Value v = lookup(bvm, op.source());
    if (!v)
      return failure();
    readOp.sourceMutable().assign(v);
    return success();
  }

  auto inPlace = getInPlace(op->getResult(0));
  auto writeOp = cast<vector::TransferWriteOp>(op.getOperation());

  // If transfer_write is not inPlace, allocate a new buffer.
  Value newInputBuffer;
  if (inPlace != InPlaceSpec::True) {
    newInputBuffer =
        createNewAllocDeallocPairForShapedValue(b, loc, writeOp.result());
    b.setInsertionPointAfter(newInputBuffer.getDefiningOp());
    map(bvm, writeOp.result(), newInputBuffer);
  } else {
    // InPlace write will result in memref.tensor_load(x) which must
    // canonicalize away with one of it uses.
    newInputBuffer = lookup(bvm, writeOp.source());
    if (!newInputBuffer)
      return failure();
  }

  // Create a new transfer_write on buffer that doesn't have a return value.
  // Leave the previous transfer_write to dead code as it still has uses at
  // this point.
  b.create<vector::TransferWriteOp>(
      loc, writeOp.vector(), newInputBuffer, writeOp.indices(),
      writeOp.permutation_map(),
      writeOp.in_bounds() ? *writeOp.in_bounds() : ArrayAttr());

  map(bvm, op->getResult(0), newInputBuffer);

  return success();
}

//===----------------------------------------------------------------------===//
// Functions and calls bufferization support.
//===----------------------------------------------------------------------===//

/// Determine whether any subsequent read of the tensor `opOperand` may occur.
/// For now, this assumes any use is a read. If any use of the tensor does not
/// properly dominate `opOperand.getOwner()`, then the tensor cannot be
/// bufferized inPlace.
// TODO: For now, this assumes any use is a read. Refine this.
bool hasInterferingTensorRead(OpOperand &opOperand,
                              const DominanceInfo &domInfo) {
  if (!opOperand.get().getType().isa<RankedTensorType>())
    return false;
  for (auto &use : opOperand.get().getUses()) {
    Operation *user = use.getOwner();
    // If properly dominate, there  is a clear sequence point and we can dismiss
    // read.
    if (domInfo.properlyDominates(user, opOperand.getOwner()))
      continue;
    // Otherwise, we need to analyze self-dependencies, for now just let it go.
    // TODO: proper self-dependence analysis.
    if (domInfo.dominates(user, opOperand.getOwner()))
      continue;
    if (user == opOperand.getOwner() &&
        use.getOperandNumber() == opOperand.getOperandNumber())
      continue;
    LLVM_DEBUG(DBGS() << "found interfering read operand #"
                      << opOperand.getOperandNumber()
                      << " in op: " << *opOperand.getOwner() << "\n");
    return true;
  }
  LLVM_DEBUG(DBGS() << "no interfering read\n");
  return false;
}

/// Return false if either:
/// 1. `opOperand` is produced by a constant op. For now this is assumed to be
///    bufferized to a GlobalMemrefOp that cannot be written. Generalize in the
///    future.
/// 2.`opOperand` is a BlockArgument of a FuncOp that is not known to be
///    bufferizable inplace.
/// Return true otherwise.
static bool bufferizeToWriteable(OpOperand &opOperand) {
  // Constant tensors are deemed not bufferizable for now.
  if (auto constantOp =
          dyn_cast_or_null<ConstantOp>(opOperand.get().getDefiningOp()))
    return !constantOp.getResult().getType().isa<RankedTensorType>();
  if (auto bbArg = opOperand.get().dyn_cast<BlockArgument>()) {
    // Uses of function arguments that may not be written-to need to be copied.
    // If the function argument itself is not inplaceable, early return false.
    // If is is inplaceable, interfering tensor read need to be checked.
    //
    // TODO: better propagate the fact that we want a single clone inside the
    // function. Atm every user that wants to write inplace will create its own
    // alloc, irrespective of whether or not interfering reads occur.
    if (isa<FuncOp>(bbArg.getOwner()->getParentOp())) {
      if (getInPlace(bbArg) != InPlaceSpec::True)
        return false;
    } else {
      // Conservatively dump any other block argument for now.
      return false;
    }
  }
  return true;
}

/// Return false if either:
/// 1. `opOperand` is produced by a constant op. For now this is assumed to be
///    bufferized to a GlobalMemrefOp that cannot be written. Generalize in the
///    future.
/// 2.`opOperand` is a BlockArgument of a FuncOp that is not known to be
///    bufferizable inplace.
/// 3.`opOperand` has an interfering tensor read.
/// Return true otherwise.
static bool isBufferizableInPlace(OpOperand &opOperand,
                                  const DominanceInfo &domInfo) {
  return bufferizeToWriteable(opOperand) &&
         !hasInterferingTensorRead(opOperand, domInfo);
}

/// Return true if `operand` bufferizes to a buffer that is known to never be
/// written.
static bool bufferizeToReadOnly(OpOperand &operand) {
  return llvm::TypeSwitch<Operation *, bool>(operand.getOwner())
      .Case([&](LinalgOp linalgOp) { return linalgOp.isInputTensor(&operand); })
      .Default([&](Operation *op) { return false; });
}

/// Assume operand is a use of a `subTensorOp`.
/// Return true if this use bufferizes to a buffer that is known to never be
/// written.
/// Note: This function takes into consideration uses of subTensorOp and whether
/// the owner of those uses is inplaceable. This needs to be run in postorder to
/// provide the most accurate analysis; otherwise it is conservative.
static bool subTensorUseBufferizesToReadOnly(OpOperand &operand) {
  assert(operand.get().getDefiningOp<SubTensorOp>() && "expected subtensor op");
  if (auto subTensorInsertOp =
          dyn_cast<SubTensorInsertOp>(operand.getOwner())) {
    return operand.getOperandNumber() == 0 /* source of the subTensorInsert*/ &&
           // If the subTensorInsertOp is not inplace, there is no possible
           // internal aliasing with subTensorOp, which is inplaceable.
           getInPlace(subTensorInsertOp->getResult(0)) != InPlaceSpec::True;
  }
  return bufferizeToReadOnly(operand);
}

/// Return true if `dominator.getOwner()` dominates all other uses of
/// `dominator.get()`.
static bool dominatesAllOtherUses(OpOperand &dominator,
                                  const DominanceInfo &domInfo) {
  for (OpOperand &use : dominator.get().getUses()) {
    // Same use.
    if (use.getOwner() == dominator.getOwner() &&
        use.getOperandNumber() == dominator.getOperandNumber())
      continue;
    if (!domInfo.properlyDominates(dominator.getOwner(), use.getOwner()))
      return false;
  }
  return true;
}

/// SubTensorOp introduces potential aliasing and a combination of things need
/// to occur to determine whether it is inplaceable.
static void analyzeInPlaceSubTensor(SubTensorOp subTensorOp,
                                    const DominanceInfo &domInfo) {
  // Case 1:
  //   a. All uses are known to bufferize to readonly buffers.
  //   b. The source has no use that is not dominated by subTensorOp.
  // This can skip bufferizeToWriteable analysis / function boundary annotation.
  if (llvm::all_of(subTensorOp.result().getUses(),
                   subTensorUseBufferizesToReadOnly) &&
      dominatesAllOtherUses(subTensorOp->getOpOperand(0), domInfo))
    return setInPlaceOpResult(subTensorOp->getResult(0), InPlaceSpec::True);

  // TODO: Implement more advanced use cases.There is a notion of transitivity
  // and interference sets lurking.
}

/// Analyze the internals of a FuncOp to determine inplaceable ops.
static void inPlaceAnalysisFuncOpInternals(FuncOp funcOp,
                                           const DominanceInfo &domInfo) {
  assert(funcOp && funcOp->getNumRegions() > 0 && !funcOp.body().empty() &&
         "expected a funcOp definition with a body");

  funcOp.walk([&](Operation *op) {
    // Skip SubTensorOp in a first pass.
    if (auto subTensorOp = dyn_cast<SubTensorOp>(op))
      return analyzeInPlaceSubTensor(subTensorOp, domInfo);

    // All other ops are checked for `isBufferizableInPlace`.
    for (OpOperand &opOperand : op->getOpOperands()) {
      OpResult result = getMatchingOpResult(opOperand);
      if (result && isBufferizableInPlace(opOperand, domInfo)) {
        LLVM_DEBUG(DBGS() << "bufferizable inplace operand #"
                          << opOperand.getOperandNumber() << " in " << *op);
        setInPlaceOpResult(result);
      }
    }
  });
}

static LogicalResult bufferizeFuncOpInternals(
    FuncOp funcOp, BlockAndValueMapping &bvm,
    const DenseMap<FuncOp, SmallVector<int64_t>> &tiedResultsMap) {
  OpBuilder b(funcOp->getContext());
  /// Start by bufferizing `funcOp` arguments.
  if (failed(bufferize(b, funcOp, bvm)))
    return failure();
  WalkResult result = funcOp.walk<WalkOrder::PostOrder>([&](Operation *op) {
    LogicalResult status =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            // Skip BufferCast and TensorLoad ops.
            // clang-format off
            .Case<memref::BufferCastOp,
                  memref::TensorLoadOp>(
                [&](auto) { return success(); })
            .Case<memref::DimOp,
                  LinalgOp,
                  ReturnOp,
                  SubTensorOp,
                  SubTensorInsertOp,
                  VectorTransferOpInterface>(
                [&](auto op) { return bufferize(b, op, bvm); })
            // clang-format on
            .Default([&](Operation *op) {
              auto isaTensor = [](Type t) { return t.isa<TensorType>(); };
              if (llvm::any_of(op->getOperandTypes(), isaTensor) ||
                  llvm::any_of(op->getResultTypes(), isaTensor))
                return failure();
              return success();
            });
    if (failed(status)) {
      op->emitError("Failed bufferization");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return success();
}

namespace {
struct LinalgComprehensiveFuncBufferize
    : public LinalgComprehensiveFuncBufferizeBase<
          LinalgComprehensiveFuncBufferize> {
  void runOnFunction() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect>();
  }
};
} // end namespace

void LinalgComprehensiveFuncBufferize::runOnFunction() {
  auto funcOp = getFunction();
  DominanceInfo domInfo(funcOp);
  BlockAndValueMapping bvm;
  DenseMap<FuncOp, SmallVector<int64_t>> tiedResultsMap;
  LLVM_DEBUG(llvm::dbgs() << "\n\n");
  LLVM_DEBUG(DBGS() << "Begin InPlaceAnalysisFuncOpInternals:\n"
                    << funcOp << "\n");
  inPlaceAnalysisFuncOpInternals(funcOp, domInfo);
  LLVM_DEBUG(DBGS() << "End InPlaceAnalysisFuncOpInternals:\n"
                    << funcOp << "\n");

  if (testAnalysisOnly)
    return;

  LLVM_DEBUG(llvm::dbgs() << "\n\n");
  LLVM_DEBUG(DBGS() << "Begin BufferizeFuncOpInternals:\n" << funcOp << "\n");
  auto guard = llvm::make_scope_exit([&] {
    funcOp.walk(
        [&](Operation *op) { op->removeAttr(kInPlaceResultsAttrName); });
    LLVM_DEBUG(DBGS() << "End BufferizeFuncOpInternals:\n" << funcOp << "\n");
  });
  if (failed(bufferizeFuncOpInternals(funcOp, bvm, tiedResultsMap)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createLinalgComprehensiveFuncBufferizePass() {
  return std::make_unique<LinalgComprehensiveFuncBufferize>();
}
