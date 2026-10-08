//===- XeGPUTransformOps.cpp - Implementation of XeGPU transformation ops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
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

  // Find descriptor op.
  auto maybeDescOp =
      findProducerOfType<xegpu::CreateNdDescOp>(loadOp.getResult());
  if (!maybeDescOp)
    return emitSilenceableFailure(getLoc()) << "Could not find descriptor op.";
  auto descOp = *maybeDescOp;

  // Clone desc op outside the loop.
  rewriter.setInsertionPoint(forOp);
  auto newDescOp =
      cast<xegpu::CreateNdDescOp>(rewriter.clone(*descOp.getOperation()));

  // Clone reduction loop to emit initial prefetches.
  // Compute upper bound of the init loop: start + nbPrefetch * step.
  auto nbPrefetchCst =
      arith::ConstantIndexOp::create(rewriter, forOp.getLoc(), nbPrefetch);
  auto nbStep = rewriter.createOrFold<arith::MulIOp>(
      forOp.getLoc(), nbPrefetchCst, forOp.getStep());
  auto initUpBound = rewriter.createOrFold<arith::AddIOp>(
      forOp.getLoc(), forOp.getLowerBound(), nbStep);
  auto initForOp =
      scf::ForOp::create(rewriter, forOp.getLoc(), forOp.getLowerBound(),
                         initUpBound, forOp.getStep());

  auto ctx = rewriter.getContext();
  auto readCacheHint =
      xegpu::CachePolicyAttr::get(ctx, xegpu::CachePolicy::CACHED);

  // Modify loadOp mixedOffsets by replacing the for loop induction variable
  // with the given value.
  auto getPrefetchOffsets =
      [&](Value replacementVal) -> SmallVector<OpFoldResult> {
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), replacementVal);
    SmallVector<Value> dynamicOffsets =
        llvm::map_to_vector(loadOp.getOffsets(), [&](Value v) {
          return mapping.lookupOrDefault(v);
        });
    auto constOffsets = loadOp.getConstOffsets();
    return getMixedValues(constOffsets, dynamicOffsets, ctx);
  };

  // Insert prefetch op in init loop.
  // Replace induction var with the init loop induction var.
  rewriter.setInsertionPointToStart(initForOp.getBody());
  xegpu::PrefetchNdOp::create(rewriter, newDescOp.getLoc(),
                              newDescOp.getResult(),
                              getPrefetchOffsets(initForOp.getInductionVar()),
                              readCacheHint, readCacheHint, readCacheHint,
                              /*layout=*/nullptr);

  // Insert prefetch op in main loop.
  // Calculate prefetch offset after the init prefetches have been issued.
  rewriter.setInsertionPointToStart(forOp.getBody());
  auto prefetchOffset = arith::AddIOp::create(rewriter, forOp.getLoc(),
                                              forOp.getInductionVar(), nbStep);
  // Replace induction var with correct offset.
  xegpu::PrefetchNdOp::create(rewriter, newDescOp.getLoc(),
                              newDescOp.getResult(),
                              getPrefetchOffsets(prefetchOffset), readCacheHint,
                              readCacheHint, readCacheHint, /*layout=*/nullptr);

  // Unroll the init loop.
  if (failed(loopUnrollFull(initForOp)))
    return emitSilenceableFailure(getLoc()) << "Failed to unroll the loop";

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
