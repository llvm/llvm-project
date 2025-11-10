//===- XeGPUTransformOps.cpp - Implementation of XeGPU transformation ops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"

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
static xegpu::LayoutAttr
createLayoutAttr(MLIRContext *ctx, ArrayRef<int32_t> sgLayout,
                 ArrayRef<int32_t> sgData,
                 std::optional<ArrayRef<int32_t>> instData) {
  return xegpu::LayoutAttr::get(
      ctx, DenseI32ArrayAttr::get(ctx, sgLayout),
      DenseI32ArrayAttr::get(ctx, sgData),
      instData ? DenseI32ArrayAttr::get(ctx, instData.value()) : nullptr,
      /*lane_layout=*/nullptr,
      /*lane_data=*/nullptr,
      /*order=*/nullptr);
}

/// Replace xegpu.create_nd_desc op with a new one with the given layout.
static xegpu::CreateNdDescOp
setDescLayout(transform::TransformRewriter &rewriter,
              xegpu::CreateNdDescOp descOp, xegpu::LayoutAttr layout) {
  assert(descOp.getMixedOffsets().size() == 0 &&
         "create desc op with offsets is not supported");
  auto oldTensorDesc = descOp.getType();
  auto descType = xegpu::TensorDescType::get(
      oldTensorDesc.getShape(), oldTensorDesc.getElementType(),
      /*array_length=*/oldTensorDesc.getArrayLength(),
      /*boundary_check=*/oldTensorDesc.getBoundaryCheck(),
      /*memory_space=*/oldTensorDesc.getMemorySpace(),
      /*layout=*/layout);

  rewriter.setInsertionPointAfter(descOp);
  auto newDescOp = rewriter.replaceOpWithNewOp<xegpu::CreateNdDescOp>(
      descOp, descType, descOp.getSource(), descOp.getMixedSizes(),
      descOp.getMixedStrides());
  return newDescOp;
}

DiagnosedSilenceableFailure
transform::GetDescOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  auto targetValues = state.getPayloadValues(getTarget());
  if (!llvm::hasSingleElement(targetValues)) {
    return emitDefiniteFailure()
           << "requires exactly one target value handle (got "
           << llvm::range_size(targetValues) << ")";
  }

  auto maybeDescOp =
      findProducerOfType<xegpu::CreateNdDescOp>(*targetValues.begin());
  if (!maybeDescOp) {
    return emitSilenceableFailure(getLoc())
           << "Could not find a matching descriptor op when walking the "
              "producer chain of the first operand.";
  }

  results.set(llvm::cast<OpResult>(getResult()), {*maybeDescOp});
  return DiagnosedSilenceableFailure::success();
}

void transform::SetDescLayoutOp::build(OpBuilder &builder,
                                       OperationState &result, Value target,
                                       ArrayRef<OpFoldResult> mixedSgLayout,
                                       ArrayRef<OpFoldResult> mixedSgData,
                                       ArrayRef<OpFoldResult> mixedInstData) {
  SmallVector<int64_t> staticSgLayout, staticSgData, staticInstData;
  SmallVector<Value> dynamicSgLayout, dynamicSgData, dynamicInstData;
  dispatchIndexOpFoldResults(mixedSgLayout, dynamicSgLayout, staticSgLayout);
  dispatchIndexOpFoldResults(mixedSgData, dynamicSgData, staticSgData);
  dispatchIndexOpFoldResults(mixedInstData, dynamicInstData, staticInstData);
  build(builder, result, target.getType(),
        /*target=*/target,
        /*sg_layout=*/dynamicSgLayout,
        /*sg_data=*/dynamicSgData,
        /*inst_data=*/dynamicInstData,
        /*static_sg_layout=*/staticSgLayout,
        /*static_sg_data=*/staticSgData,
        /*static_inst_data=*/staticInstData);
}

DiagnosedSilenceableFailure
transform::SetDescLayoutOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  auto targetOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(targetOps)) {
    return emitDefiniteFailure() << "requires exactly one targetOp handle (got "
                                 << llvm::range_size(targetOps) << ")";
  }
  Operation *target = *targetOps.begin();

  SmallVector<int32_t> sgLayout;
  DiagnosedSilenceableFailure status =
      convertMixedValuesToInt(state, (*this), sgLayout, getMixedSgLayout());
  if (!status.succeeded())
    return status;

  SmallVector<int32_t> sgData;
  status = convertMixedValuesToInt(state, (*this), sgData, getMixedSgData());
  if (!status.succeeded())
    return status;

  SmallVector<int32_t> instData;
  status =
      convertMixedValuesToInt(state, (*this), instData, getMixedInstData());
  if (!status.succeeded())
    return status;
  auto maybeInstData = instData.empty()
                           ? std::nullopt
                           : std::optional<ArrayRef<int32_t>>(instData);

  // For now only create_nd_desc op is supported.
  auto descOp = dyn_cast<xegpu::CreateNdDescOp>(target);
  if (!descOp) {
    auto diag = emitSilenceableFailure(getLoc())
                << "Expected a xegpu.create_nd_desc op, but got: "
                << target->getName();
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  // Set layout attr in desc op's return type. Replaces old desc op.
  auto layoutAttr =
      createLayoutAttr(rewriter.getContext(), sgLayout, sgData, maybeInstData);
  auto newdescOp = setDescLayout(rewriter, descOp, layoutAttr);

  // Map result handles.
  results.set(cast<OpResult>(getTransformed()), {newdescOp.getOperation()});

  return DiagnosedSilenceableFailure::success();
}

void transform::SetDescLayoutOp::getEffects(
    ::llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  onlyReadsHandle(getSgLayoutMutable(), effects);
  onlyReadsHandle(getSgDataMutable(), effects);
  onlyReadsHandle(getInstDataMutable(), effects);
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
