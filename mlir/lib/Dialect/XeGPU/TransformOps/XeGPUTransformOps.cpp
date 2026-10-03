//===- XeGPUTransformOps.cpp - Implementation of XeGPU transformation ops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include <optional>

#include "llvm/Support/DebugLog.h"
#define DEBUG_TYPE "xegpu-transforms"

using namespace mlir;
using namespace mlir::transform;

/// Assuming that `ofr` is an index attr or a param of index type
/// or a transform dialect handle mapped to exactly one op
/// with one index result, get that value and cast it to int type.
static DiagnosedSilenceableFailure convertMixedValuesToInt(
    transform::TransformState &state, TransformOpInterface transformOp,
    SmallVectorImpl<int32_t> &result, ArrayRef<OpFoldResult> ofrs) {
  for (OpFoldResult ofr : ofrs) {
    // Attribute case.
    if (auto attr = dyn_cast<Attribute>(ofr)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        result.push_back(intAttr.getInt());
        continue;
      }
      return transformOp.emitDefiniteFailure() << "expected IntegerAttr";
    }

    // Transform param case.
    Value transformValue = cast<Value>(ofr);
    if (isa<TransformParamTypeInterface>(transformValue.getType())) {
      ArrayRef<Attribute> params = state.getParams(transformValue);
      if (params.size() != 1)
        return transformOp.emitDefiniteFailure()
               << "requires exactly one parameter associated";
      result.push_back(
          cast<IntegerAttr>(params.front()).getValue().getSExtValue());
      continue;
    }

    // Payload value case.
    auto payloadOps = state.getPayloadOps(transformValue);
    if (!llvm::hasSingleElement(payloadOps)) {
      DiagnosedSilenceableFailure diag =
          transformOp.emitSilenceableError()
          << "handle must be mapped to exactly one payload op";
      diag.attachNote(transformValue.getLoc())
          << "mapped to " << llvm::range_size(payloadOps) << " payload ops";
      return diag;
    }

    Operation *op = *payloadOps.begin();
    if (op->getNumResults() != 1 || !op->getResult(0).getType().isIndex()) {
      DiagnosedSilenceableFailure diag =
          transformOp.emitSilenceableError()
          << "payload op must have exactly 1 index result";
      diag.attachNote(op->getLoc())
          << "has " << op->getNumResults() << " results";
      return diag;
    }

    IntegerAttr intAttr;
    if (!matchPattern(op->getResult(0), m_Constant(&intAttr)))
      return transformOp.emitSilenceableError()
             << "requires param or handle to be the result of a constant like "
                "op";

    result.push_back(intAttr.getInt());
  }
  return DiagnosedSilenceableFailure::success();
}

/// Find producer operation of type T for the given value.
/// It's assumed that producer ops are chained through their first operand.
/// Producer chain is traced trough loop block arguments (init values).
template <typename T>
static std::optional<T> findProducerOfType(Value val) {
  Value currentValue = val;
  if (!currentValue.getDefiningOp()) {
    // Value may be a block argument initialized outside a loop.
    if (val.getNumUses() == 0) {
      LDBG() << "Failed to find producer op, value has no uses.";
      return std::nullopt;
    }
    auto userOp = val.getUsers().begin();
    auto parentLoop = userOp->getParentOfType<LoopLikeOpInterface>();
    if (!parentLoop) {
      LDBG() << "Failed to find producer op, not in a loop.";
      return std::nullopt;
    }
    int64_t iterArgIdx;
    if (auto iterArg = llvm::dyn_cast<BlockArgument>(currentValue)) {
      auto numInductionVars = parentLoop.getLoopInductionVars()->size();
      iterArgIdx = iterArg.getArgNumber() - numInductionVars;
      currentValue = parentLoop.getInits()[iterArgIdx];
    } else {
      LDBG() << "Failed to find producer op, value not in init values.";
      return std::nullopt;
    }
  }
  Operation *producerOp = currentValue.getDefiningOp();

  if (auto matchingOp = dyn_cast<T>(producerOp))
    return matchingOp;

  if (producerOp->getNumOperands() == 0)
    return std::nullopt;

  return findProducerOfType<T>(producerOp->getOperand(0));
}

/// Create a layout attribute from the given parameters.
static xegpu::LayoutAttr createLayoutAttr(
    MLIRContext *ctx, ArrayRef<int32_t> sgLayout, ArrayRef<int32_t> sgData,
    std::optional<ArrayRef<int32_t>> instData, ArrayRef<int32_t> order) {
  return xegpu::LayoutAttr::get(
      ctx, DenseI32ArrayAttr::get(ctx, sgLayout),
      DenseI32ArrayAttr::get(ctx, sgData),
      instData ? DenseI32ArrayAttr::get(ctx, instData.value()) : nullptr,
      /*lane_layout=*/nullptr,
      /*lane_data=*/nullptr,
      /*order=*/order.empty() ? nullptr : DenseI32ArrayAttr::get(ctx, order));
}

/// Generate `xegpu::LayoutAttr` from op mixed layout values.
DiagnosedSilenceableFailure
getLayoutAttrFromOperands(MLIRContext *ctx, transform::TransformState &state,
                          TransformOpInterface transformOp,
                          ArrayRef<::mlir::OpFoldResult> mixedSgLayout,
                          ArrayRef<::mlir::OpFoldResult> mixedSgData,
                          ArrayRef<::mlir::OpFoldResult> mixedInstData,
                          ArrayRef<int32_t> order,
                          xegpu::LayoutAttr &layoutAttr) {
  SmallVector<int32_t> sgLayout, sgData, instData;
  auto status =
      convertMixedValuesToInt(state, transformOp, sgLayout, mixedSgLayout);
  if (!status.succeeded())
    return status;

  status = convertMixedValuesToInt(state, transformOp, sgData, mixedSgData);
  if (!status.succeeded())
    return status;

  status = convertMixedValuesToInt(state, transformOp, instData, mixedInstData);
  if (!status.succeeded())
    return status;
  auto maybeInstData = instData.empty()
                           ? std::nullopt
                           : std::optional<ArrayRef<int32_t>>(instData);

  layoutAttr = createLayoutAttr(ctx, sgLayout, sgData, maybeInstData, order);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform::GetLoadOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  auto targetValues = state.getPayloadValues(getTarget());
  if (!llvm::hasSingleElement(targetValues)) {
    return emitDefiniteFailure()
           << "requires exactly one target value handle (got "
           << llvm::range_size(targetValues) << ")";
  }

  Operation *loadOp = nullptr;
  auto maybeLoadNdOp =
      findProducerOfType<xegpu::LoadNdOp>(*targetValues.begin());
  if (maybeLoadNdOp) {
    loadOp = maybeLoadNdOp->getOperation();
  } else {
    auto maybeLoadOp =
        findProducerOfType<xegpu::LoadGatherOp>(*targetValues.begin());
    if (maybeLoadOp) {
      loadOp = maybeLoadOp->getOperation();
    } else {
      return emitSilenceableFailure(getLoc())
             << "Could not find a matching xegpu.load_nd or xegpu.load op when "
                "walking the "
                "producer chain of the first operand.";
    }
  }

  results.set(llvm::cast<OpResult>(getResult()), {loadOp});
  return DiagnosedSilenceableFailure::success();
}

void transform::SetAnchorLayoutOp::build(
    OpBuilder &builder, OperationState &ostate, Value target, int64_t index,
    ArrayRef<OpFoldResult> mixedSgLayout, ArrayRef<OpFoldResult> mixedSgData,
    ArrayRef<OpFoldResult> mixedInstData, ArrayRef<int32_t> order,
    ArrayRef<int64_t> sliceDims) {
  SmallVector<int64_t> staticSgLayout, staticSgData, staticInstData;
  SmallVector<Value> dynamicSgLayout, dynamicSgData, dynamicInstData;
  dispatchIndexOpFoldResults(mixedSgLayout, dynamicSgLayout, staticSgLayout);
  dispatchIndexOpFoldResults(mixedSgData, dynamicSgData, staticSgData);
  dispatchIndexOpFoldResults(mixedInstData, dynamicInstData, staticInstData);
  build(builder, ostate, target.getType(),
        /*target=*/target,
        /*index=*/index,
        /*sg_layout=*/dynamicSgLayout,
        /*sg_data=*/dynamicSgData,
        /*inst_data=*/dynamicInstData,
        /*static_sg_layout=*/staticSgLayout,
        /*static_sg_data=*/staticSgData,
        /*static_inst_data=*/staticInstData,
        /*order=*/order,
        /*slice_dims=*/sliceDims);
}

DiagnosedSilenceableFailure
transform::SetAnchorLayoutOp::apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {
  auto targetOps = state.getPayloadOps(getTarget());
  int64_t index = getIndex();

  // Construct layout attribute.
  xegpu::LayoutAttr layoutAttr = nullptr;
  auto status = getLayoutAttrFromOperands(
      getContext(), state, (*this), getMixedSgLayout(), getMixedSgData(),
      getMixedInstData(), getOrder(), layoutAttr);
  if (!status.succeeded())
    return status;

  xegpu::DistributeLayoutAttr layout = layoutAttr;
  auto sliceDims = getSliceDims();
  if (sliceDims.size() > 0) {
    // Wrap layoutAttr in a slice attribute.
    layout = xegpu::SliceAttr::get(
        getContext(), layout, DenseI64ArrayAttr::get(getContext(), sliceDims));
  }

  // Apply the layout to all target ops.
  for (Operation *target : targetOps) {
    // Set layout attribute
    if (auto dpasOp = dyn_cast<xegpu::DpasOp>(target)) {
      // dpas op is a special case where layout needs to be set for A, B, and C
      if (index == 0)
        dpasOp.getProperties().layout_a = layout;
      else if (index == 1)
        dpasOp.getProperties().layout_b = layout;
      else if (index == 2)
        dpasOp.getProperties().layout_cd = layout;
      else {
        auto diag = emitSilenceableFailure(getLoc())
                    << "Invalid index for setting dpas op layout: " << index;
        diag.attachNote(target->getLoc()) << "target op";
        return diag;
      }
    } else {
      // op's anchor layout.
      auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(target);
      if (!anchorOp) {
        auto diag = emitSilenceableFailure(getLoc())
                    << "Cannot set anchor layout to op: " << target->getName();
        diag.attachNote(target->getLoc()) << "target op";
        return diag;
      }
      anchorOp.setAnchorLayout(layout);
    }
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::SetAnchorLayoutOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  onlyReadsHandle(getSgLayoutMutable(), effects);
  onlyReadsHandle(getSgDataMutable(), effects);
  onlyReadsHandle(getInstDataMutable(), effects);
  modifiesPayload(effects);
}

void transform::SetGPULaunchThreadsOp::build(
    OpBuilder &builder, OperationState &ostate, Value target,
    ArrayRef<OpFoldResult> mixedThreads) {
  SmallVector<int64_t> staticThreads;
  SmallVector<Value> dynamicThreads;
  dispatchIndexOpFoldResults(mixedThreads, dynamicThreads, staticThreads);
  build(builder, ostate, target.getType(),
        /*target=*/target,
        /*threads=*/dynamicThreads,
        /*static_threads=*/staticThreads);
}

DiagnosedSilenceableFailure
transform::SetGPULaunchThreadsOp::apply(transform::TransformRewriter &rewriter,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  auto targetOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(targetOps)) {
    return emitDefiniteFailure() << "Requires exactly one targetOp handle (got "
                                 << llvm::range_size(targetOps) << ")";
  }
  Operation *target = *targetOps.begin();

  auto launchOp = dyn_cast<gpu::LaunchOp>(target);
  if (!launchOp) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Expected a gpu.launch op, but got: " << target->getName();
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  SmallVector<int32_t> threads;
  DiagnosedSilenceableFailure status =
      convertMixedValuesToInt(state, (*this), threads, getMixedThreads());
  if (!status.succeeded())
    return status;

  if (threads.size() != 3) {
    return emitSilenceableFailure(getLoc())
           << "Expected threads argument to consist of three values (got "
           << threads.size() << ")";
  }

  rewriter.setInsertionPoint(launchOp);
  auto createConstValue = [&](int value) {
    return arith::ConstantIndexOp::create(rewriter, launchOp.getLoc(), value);
  };

  // Replace threads in-place.
  launchOp.getBlockSizeXMutable().assign(createConstValue(threads[0]));
  launchOp.getBlockSizeYMutable().assign(createConstValue(threads[1]));
  launchOp.getBlockSizeZMutable().assign(createConstValue(threads[2]));

  return DiagnosedSilenceableFailure::success();
}

void transform::SetGPULaunchThreadsOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  onlyReadsHandle(getThreadsMutable(), effects);
  modifiesPayload(effects);
}

/// Collect the pure loop-local producer slice (i.e. set of ops) needed to
/// recreate a descriptor for another iteration. Values defined above the loop
/// are reused.
static LogicalResult
collectLoopLocalDescriptorDependancyChain(Value value, scf::ForOp forOp,
                                          SmallPtrSetImpl<Operation *> &seen,
                                          SmallVectorImpl<Operation *> &slice) {
  if (auto arg = dyn_cast<BlockArgument>(value))
    return forOp.isDefinedOutsideOfLoop(arg) || arg == forOp.getInductionVar()
               ? success()
               : failure();

  Operation *def = value.getDefiningOp();
  if (forOp.isDefinedOutsideOfLoop(value))
    return success();
  if (def->getBlock() != forOp.getBody() || def->getNumRegions() != 0 ||
      !isMemoryEffectFree(def))
    return failure();
  if (!seen.insert(def).second)
    return success();
  for (Value operand : def->getOperands())
    if (failed(collectLoopLocalDescriptorDependancyChain(operand, forOp, seen,
                                                         slice)))
      return failure();
  slice.push_back(def);
  return success();
}

/// Clone the loop-local producer slice with the current induction variable
/// replaced by the iteration that should be prefetched.
static xegpu::CreateNdDescOp
clonePrefetchDescriptor(transform::TransformRewriter &rewriter,
                        scf::ForOp forOp, xegpu::CreateNdDescOp descOp,
                        ArrayRef<Operation *> slice, Value iteration) {
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), iteration);
  for (Operation *op : slice)
    rewriter.clone(*op, mapping);
  return mapping.lookup(descOp.getResult())
      .getDefiningOp<xegpu::CreateNdDescOp>();
}

DiagnosedSilenceableFailure
transform::InsertPrefetchOp::apply(transform::TransformRewriter &rewriter,
                                   transform::TransformResults &results,
                                   transform::TransformState &state) {
  auto targetOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(targetOps))
    return emitDefiniteFailure()
           << "requires exactly one target op handle (got "
           << llvm::range_size(targetOps) << ")";
  auto target = *targetOps.begin();

  int64_t nbPrefetch = getStaticNbPrefetch();
  if (getDynamicNbPrefetch()) {
    // Get dynamic prefetch count from transform param or handle.
    SmallVector<int32_t> dynamicNbPrefetch;
    auto status = convertMixedValuesToInt(state, (*this), dynamicNbPrefetch,
                                          {getDynamicNbPrefetch()});
    if (!status.succeeded())
      return status;
    if (dynamicNbPrefetch.size() != 1)
      return emitDefiniteFailure()
             << "requires exactly one value for dynamic_nb_prefetch";
    nbPrefetch = dynamicNbPrefetch[0];
  }
  if (nbPrefetch <= 0)
    return emitSilenceableFailure(getLoc())
           << "nb_prefetch must be a positive integer.";

  // Cast target to load op.
  auto maybeLoadOp = dyn_cast<xegpu::LoadNdOp>(target);
  if (!maybeLoadOp) {
    return emitSilenceableFailure(getLoc())
           << "Expected xegpu.load_nd op, got " << target->getName();
  }
  auto loadOp = maybeLoadOp;
  if (loadOp.getMixedOffsets().size() == 0) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Load op must have offsets.";
    diag.attachNote(loadOp.getLoc()) << "load op";
    return diag;
  }

  // Find the parent scf.for loop.
  auto forOp = loadOp->getParentOfType<scf::ForOp>();
  if (!forOp) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Load op is not contained in a scf.for loop.";
    diag.attachNote(loadOp.getLoc()) << "load op";
    return diag;
  }

  // Find the load's descriptor and distinguish the two supported forms:
  // (1) a descriptor outside the loop with the loop induction variable in a
  //     load offset, or
  // (2) a descriptor is created off of a subview. This subview must carry the
  //     loop induction variable in its offsets. It is expected that load
  //     offsets are zero.
  auto maybeDescOp =
      findProducerOfType<xegpu::CreateNdDescOp>(loadOp.getResult());
  if (!maybeDescOp)
    return emitSilenceableFailure(getLoc()) << "Could not find descriptor op.";
  auto descOp = *maybeDescOp;
  bool loopLocalDesc = forOp->isAncestor(descOp);
  SmallVector<OpFoldResult> loadOffsets = loadOp.getMixedOffsets();
  if (loopLocalDesc) {
    // Case 2: the loop position must come from a subview, not load_nd.
    if (!areAllConstantIntValue(loadOffsets, 0))
      return emitSilenceableFailure(getLoc())
             << "loop-local prefetch requires zero load offsets";
  } else {
    // Case 1: at least one load offset must be the induction variable itself.
    // Every other offset must already be available before the loop.
    bool loadUsesIV = false;
    for (OpFoldResult offset : loadOffsets) {
      auto value = dyn_cast<Value>(offset);
      if (!value)
        continue;
      if (value == forOp.getInductionVar()) {
        loadUsesIV = true;
        continue;
      }
      if (!forOp.isDefinedOutsideOfLoop(value))
        return emitSilenceableFailure(getLoc())
               << "prefetch load offsets must use the induction variable "
                  "directly or be loop-invariant";
    }
    if (!loadUsesIV)
      return emitSilenceableFailure(getLoc())
             << "prefetch load offsets must directly use the loop induction "
                "variable";
  }

  SmallVector<Operation *> descSlice;
  if (loopLocalDesc) {
    // Case 2: cloning a future subview needs a boundary-checking descriptor
    // and a side-effect-free chain of producers from within the loop body.
    if (!descOp.getType().getBoundaryCheck())
      return emitSilenceableFailure(getLoc())
             << "loop-local prefetch requires descriptor boundary_check";

    SmallPtrSet<Operation *, 8> seen;
    if (failed(collectLoopLocalDescriptorDependancyChain(
            descOp.getResult(), forOp, seen, descSlice)))
      return emitSilenceableFailure(getLoc())
             << "prefetch descriptor depends on unsupported loop-local values";

    // A local subview must carry the induction variable directly in an
    // offset. Its remaining offsets must be reusable in the prologue.
    bool hasSubview = false;
    bool subviewUsesIV = false;
    for (Operation *op : descSlice) {
      auto subview = dyn_cast<memref::SubViewOp>(op);
      if (!subview)
        continue;
      hasSubview = true;
      for (OpFoldResult offset : subview.getMixedOffsets()) {
        auto value = dyn_cast<Value>(offset);
        if (!value)
          continue;
        if (value == forOp.getInductionVar()) {
          subviewUsesIV = true;
          continue;
        }
        if (!forOp.isDefinedOutsideOfLoop(value))
          return emitSilenceableFailure(getLoc())
                 << "prefetch subview offsets must use the induction variable "
                    "directly or be loop-invariant";
      }
    }
    if (!hasSubview || !subviewUsesIV)
      return emitSilenceableFailure(getLoc())
             << "loop-local prefetch requires a subview offset that directly "
                "uses the loop induction variable";
  }

  // Case 1 shares one descriptor across all prefetches. Case 2 recreates its
  // loop-local descriptor at each prefetch site below.
  rewriter.setInsertionPoint(forOp);
  xegpu::CreateNdDescOp newDescOp;
  if (!loopLocalDesc)
    newDescOp =
        cast<xegpu::CreateNdDescOp>(rewriter.clone(*descOp.getOperation()));

  // Warm up the first nbPrefetch tiles before the load loop. Case 2 clamps
  // this prologue to the runtime bound so its subviews remain in-bounds.
  auto nbPrefetchCst =
      arith::ConstantIndexOp::create(rewriter, forOp.getLoc(), nbPrefetch);
  auto nbStep = rewriter.createOrFold<arith::MulIOp>(
      forOp.getLoc(), nbPrefetchCst, forOp.getStep());
  auto initUpBound = rewriter.createOrFold<arith::AddIOp>(
      forOp.getLoc(), forOp.getLowerBound(), nbStep);
  if (loopLocalDesc)
    initUpBound = arith::MinSIOp::create(rewriter, forOp.getLoc(), initUpBound,
                                         forOp.getUpperBound());
  auto initForOp =
      scf::ForOp::create(rewriter, forOp.getLoc(), forOp.getLowerBound(),
                         initUpBound, forOp.getStep());

  auto ctx = rewriter.getContext();
  auto readCacheHint =
      xegpu::CachePolicyAttr::get(ctx, xegpu::CachePolicy::CACHED);

  // Case 1 substitutes the iteration into load_nd's direct-IV offset. In
  // case 2 the cloned subview carries the iteration, so load offsets stay zero.
  auto getPrefetchOffsets =
      [&](Value replacementVal) -> SmallVector<OpFoldResult> {
    if (loopLocalDesc)
      return getAsIndexOpFoldResult(
          ctx, SmallVector<int64_t>(loadOffsets.size(), 0));
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), replacementVal);
    SmallVector<Value> dynamicOffsets =
        llvm::map_to_vector(loadOp.getOffsets(), [&](Value v) {
          return mapping.lookupOrDefault(v);
        });
    auto constOffsets = loadOp.getConstOffsets();
    return getMixedValues(constOffsets, dynamicOffsets, ctx);
  };

  // Issue warmup prefetches. Case 2 clones the subview and descriptor with the
  // prologue induction variable replacing the load loop's variable.
  rewriter.setInsertionPointToStart(initForOp.getBody());
  xegpu::CreateNdDescOp initDescOp =
      loopLocalDesc
          ? clonePrefetchDescriptor(rewriter, forOp, descOp, descSlice,
                                    initForOp.getInductionVar())
          : newDescOp;
  xegpu::PrefetchNdOp::create(rewriter, initDescOp.getLoc(),
                              initDescOp.getResult(),
                              getPrefetchOffsets(initForOp.getInductionVar()),
                              readCacheHint, readCacheHint, readCacheHint,
                              /*layout=*/nullptr);

  // Prefetch nbPrefetch steps ahead at the start of each load-loop iteration.
  rewriter.setInsertionPointToStart(forOp.getBody());
  auto prefetchOffset = arith::AddIOp::create(rewriter, forOp.getLoc(),
                                              forOp.getInductionVar(), nbStep);
  xegpu::CreateNdDescOp mainDescOp = newDescOp;
  if (loopLocalDesc) {
    // A descriptor's boundary check does not make an out-of-bounds subview
    // valid. Only build the future subview while its iteration is in range.
    auto inBounds = arith::CmpIOp::create(
        rewriter, forOp.getLoc(), arith::CmpIPredicate::slt, prefetchOffset,
        forOp.getUpperBound());
    auto ifOp = scf::IfOp::create(rewriter, forOp.getLoc(), inBounds,
                                  /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    mainDescOp = clonePrefetchDescriptor(rewriter, forOp, descOp, descSlice,
                                         prefetchOffset);
  }
  // Replace induction var with the future tile's offset.
  xegpu::PrefetchNdOp::create(rewriter, mainDescOp.getLoc(),
                              mainDescOp.getResult(),
                              getPrefetchOffsets(prefetchOffset), readCacheHint,
                              readCacheHint, readCacheHint, /*layout=*/nullptr);

  // Case 1 has a fixed-count prologue that can be unrolled. Case 2 keeps its
  // runtime-bounded prologue as a loop.
  if (!loopLocalDesc && failed(loopUnrollFull(initForOp)))
    return emitSilenceableFailure(getLoc()) << "Failed to unroll the loop";

  // Return both descriptors for case 2 so callers can annotate both prefetch
  // sites; case 1 has only the shared descriptor.
  if (loopLocalDesc)
    results.set(llvm::cast<OpResult>(getResult()), {initDescOp, mainDescOp});
  else
    results.set(llvm::cast<OpResult>(getResult()), {newDescOp});

  return DiagnosedSilenceableFailure::success();
}

void transform::InsertPrefetchOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  onlyReadsHandle(getDynamicNbPrefetchMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

void transform::ConvertLayoutOp::build(
    OpBuilder &builder, OperationState &ostate, Value target,
    ArrayRef<OpFoldResult> mixedInputSgLayout,
    ArrayRef<OpFoldResult> mixedInputSgData,
    ArrayRef<OpFoldResult> mixedInputInstData, ArrayRef<int32_t> inputOrder,
    ArrayRef<OpFoldResult> mixedTargetSgLayout,
    ArrayRef<OpFoldResult> mixedTargetSgData,
    ArrayRef<OpFoldResult> mixedTargetInstData, ArrayRef<int32_t> targetOrder) {
  SmallVector<int64_t> staticInputSgLayout, staticInputSgData,
      staticInputInstData;
  SmallVector<Value> dynamicInputSgLayout, dynamicInputSgData,
      dynamicInputInstData;
  dispatchIndexOpFoldResults(mixedInputSgLayout, dynamicInputSgLayout,
                             staticInputSgLayout);
  dispatchIndexOpFoldResults(mixedInputSgData, dynamicInputSgData,
                             staticInputSgData);
  dispatchIndexOpFoldResults(mixedInputInstData, dynamicInputInstData,
                             staticInputInstData);
  SmallVector<int64_t> staticTargetSgLayout, staticTargetSgData,
      staticTargetInstData;
  SmallVector<Value> dynamicTargetSgLayout, dynamicTargetSgData,
      dynamicTargetInstData;
  dispatchIndexOpFoldResults(mixedTargetSgLayout, dynamicTargetSgLayout,
                             staticTargetSgLayout);
  dispatchIndexOpFoldResults(mixedTargetSgData, dynamicTargetSgData,
                             staticTargetSgData);
  dispatchIndexOpFoldResults(mixedTargetInstData, dynamicTargetInstData,
                             staticTargetInstData);
  build(builder, ostate, target.getType(),
        /*target=*/target,
        /*input_sg_layout=*/dynamicInputSgLayout,
        /*input_sg_data=*/dynamicInputSgData,
        /*input_inst_data=*/dynamicInputInstData,
        /*target_sg_layout=*/dynamicTargetSgLayout,
        /*target_sg_data=*/dynamicTargetSgData,
        /*target_inst_data=*/dynamicTargetInstData,
        /*input_order=*/inputOrder,
        /*static_input_sg_layout=*/staticInputSgLayout,
        /*static_input_sg_data=*/staticInputSgData,
        /*static_input_inst_data=*/staticInputInstData,
        /*static_target_sg_layout=*/staticTargetSgLayout,
        /*static_target_sg_data=*/staticTargetSgData,
        /*static_target_inst_data=*/staticTargetInstData,
        /*target_order=*/targetOrder);
}

DiagnosedSilenceableFailure
transform::ConvertLayoutOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  auto targetValues = state.getPayloadValues(getTarget());
  if (!llvm::hasSingleElement(targetValues))
    return emitDefiniteFailure()
           << "requires exactly one target value handle (got "
           << llvm::range_size(targetValues) << ")";
  auto value = *targetValues.begin();

  // Construct layout attributes.
  xegpu::LayoutAttr inputLayoutAttr = nullptr;
  auto status = getLayoutAttrFromOperands(
      getContext(), state, (*this), getMixedInputSgLayout(),
      getMixedInputSgData(), getMixedInputInstData(), getInputOrder(),
      inputLayoutAttr);
  if (!status.succeeded())
    return status;

  xegpu::LayoutAttr targetLayoutAttr = nullptr;
  status = getLayoutAttrFromOperands(
      getContext(), state, (*this), getMixedTargetSgLayout(),
      getMixedTargetSgData(), getMixedTargetInstData(), getTargetOrder(),
      targetLayoutAttr);
  if (!status.succeeded())
    return status;

  // Find first user op to define insertion point for layout conversion.
  if (value.use_empty())
    return emitSilenceableFailure(getLoc())
           << "Value has no users to insert layout conversion.";
  Operation *userOp = *value.getUsers().begin();

  // Emit convert_layout op.
  rewriter.setInsertionPoint(userOp);
  auto convLayoutOp =
      xegpu::ConvertLayoutOp::create(rewriter, value.getLoc(), value.getType(),
                                     value, inputLayoutAttr, targetLayoutAttr);
  // Replace load op result with the converted layout.
  rewriter.replaceUsesWithIf(
      value, convLayoutOp.getResult(), [&](OpOperand &use) {
        return use.getOwner() != convLayoutOp.getOperation();
      });

  results.set(llvm::cast<OpResult>(getResult()), {convLayoutOp});
  return DiagnosedSilenceableFailure::success();
}

void transform::ConvertLayoutOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  onlyReadsHandle(getInputSgLayoutMutable(), effects);
  onlyReadsHandle(getInputSgDataMutable(), effects);
  onlyReadsHandle(getInputInstDataMutable(), effects);
  onlyReadsHandle(getTargetSgLayoutMutable(), effects);
  onlyReadsHandle(getTargetSgDataMutable(), effects);
  onlyReadsHandle(getTargetInstDataMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

namespace {
class XeGPUTransformDialectExtension
    : public transform::TransformDialectExtension<
          XeGPUTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XeGPUTransformDialectExtension)

  using Base::Base;

  void init();
};

void XeGPUTransformDialectExtension::init() {
  declareGeneratedDialect<scf::SCFDialect>();
  declareGeneratedDialect<arith::ArithDialect>();
  declareGeneratedDialect<xegpu::XeGPUDialect>();

  registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.cpp.inc"
      >();
}
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.cpp.inc"

void mlir::xegpu::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<XeGPUTransformDialectExtension>();
}
