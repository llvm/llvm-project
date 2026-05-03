//===- ACCComputeLowering.cpp - Lower ACC compute to compute_region -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass decomposes OpenACC compute constructs into a representation that
// separates the data environment from the compute portion and prepares for
// parallelism assignment and privatization at the appropriate level.
//
// Overview:
// ---------
// Each compute construct (`acc.parallel`, `acc.serial`, `acc.kernels`) is
// lowered to (1) `acc.kernel_environment`, which captures the data environment
// and (2) `acc.compute_region`, which holds the compute body. Inside the
// compute region, acc.loop is converted to SCF loops (`scf.parallel` or
// `scf.for`) with any predetermined parallelism expressed as `par_dims`. This
// decomposition allows later phases to assign parallelism and handle
// privatization at the right granularity.
//
// Transformations:
// ----------------
// 1. Compute constructs: acc.parallel, acc.serial, and acc.kernels are
//    replaced by acc.kernel_environment containing a single acc.compute_region.
//    For acc.parallel / acc.kernels, launch arguments (num_gangs, num_workers,
//    vector_length) become acc.par_width ops (each result is `index`) and are
//    passed as compute_region launch operands. Compute regions with
//    num_gangs(1), num_workers(1), and vector_length(1) and acc serial use a
//    single sequential acc.par_width launch operand.
//
// 2. acc.loop: Converted according to context and attributes:
//    - Unstructured: body wrapped in scf.execute_region.
//    - Sequential (serial region, seq clause, or compute region with
//    num_gangs(1), num_workers(1), and vector_length(1)):
//      scf.parallel with par_dims = sequential.
//    - Auto (in parallel/kernels): scf.for with collapse when
//    multi-dimensional.
//    - Orphan (not inside a compute construct): scf.for, no collapse.
//    - Independent (in parallel/kernels): scf.parallel with par_dims from
//      gang/worker/vector mapping (e.g. block_x).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCParMapping.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsLoop.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCCOMPUTELOWERING
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-compute-lowering"

using namespace mlir;
using namespace mlir::acc;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Strip index_cast operations from a value before checking for a constant.
static Value stripIndexCasts(Value val) {
  while (auto castOp = val.getDefiningOp<arith::IndexCastOp>())
    val = castOp.getIn();
  return val;
}

template <typename ComputeOpT>
static bool isGangWorkerVectorAllOne(ComputeOpT op) {
  auto numGangs = op.getNumGangsValues();
  if (numGangs.empty())
    return false;
  for (Value gangSize : numGangs) {
    if (!isConstantIntValue(stripIndexCasts(gangSize), 1))
      return false;
  }
  Value numWorkers = op.getNumWorkersValue();
  if (!numWorkers)
    return false;
  Value vectorLength = op.getVectorLengthValue();
  if (!vectorLength)
    return false;
  return isConstantIntValue(stripIndexCasts(numWorkers), 1) &&
         isConstantIntValue(stripIndexCasts(vectorLength), 1);
}

/// A compute construct is "effectively serial" when it specifies
/// num_gangs(1), num_workers(1), and vector_length(1). This is because
/// these are the only parallelism dimensions expressible from OpenACC spec
/// point-of-view and is consistent with how `serial` semantics are defined.
template <typename ComputeOpT>
static bool isEffectivelySerial(ComputeOpT op) {
  return isGangWorkerVectorAllOne(op);
}

static bool isOpInComputeRegion(Operation *op) {
  Region *region = op->getBlock()->getParent();
  return getEnclosingComputeOp(*region) != nullptr;
}

static bool isOpInSerialRegion(Operation *op) {
  if (auto parallelOp = op->getParentOfType<ParallelOp>())
    return isEffectivelySerial(parallelOp);
  if (auto kernelsOp = op->getParentOfType<KernelsOp>())
    return isEffectivelySerial(kernelsOp);
  if (op->getParentOfType<SerialOp>())
    return true;
  if (auto computeRegion = op->getParentOfType<ComputeRegionOp>())
    return computeRegion.isEffectivelySerial();
  if (auto funcOp = op->getParentOfType<FunctionOpInterface>()) {
    if (isSpecializedAccRoutine(funcOp)) {
      auto attr = funcOp->getAttrOfType<SpecializedRoutineAttr>(
          getSpecializedRoutineAttrName());
      if (attr && attr.getLevel().getValue() == ParLevel::seq)
        return true;
    }
  }
  return false;
}

static void setParDimsAttr(Operation *op, GPUParallelDimsAttr attr) {
  op->setAttr(GPUParallelDimsAttr::name, attr);
}

/// Clone defining ops of constant live-in values into `region`, rewrite uses
/// inside the region to the clones, and remove those values from
/// `liveInValues` so they are not threaded through `acc.compute_region` ins.
static void materializeConstantLiveInsIntoRegion(Region &region,
                                                 SetVector<Value> &liveInValues,
                                                 RewriterBase &rewriter) {
  SmallVector<Value> constantLiveIns;
  for (Value v : liveInValues) {
    Operation *defOp = v.getDefiningOp();
    if (defOp && matchPattern(defOp, m_Constant())) {
      // As per the definition of ConstantLike trait, constants must have a
      // single result.
      assert(defOp->getNumResults() == 1 &&
             "constants must have a single result");
      constantLiveIns.push_back(v);
    }
  }
  if (constantLiveIns.empty())
    return;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&region.front());

  for (Value v : constantLiveIns) {
    Value newV = rewriter.clone(*v.getDefiningOp())->getResult(0);
    replaceAllUsesInRegionWith(v, newV, region);
    liveInValues.remove(v);
  }
}

/// Insert a parallel dimension into the list, maintaining order by
/// GPUParallelDimAttr::getOrder (descending).
static void insertParDim(SmallVectorImpl<GPUParallelDimAttr> &parDims,
                         GPUParallelDimAttr parDim) {
  GPUParallelDimAttr *lb = llvm::lower_bound(
      parDims, parDim,
      [](const GPUParallelDimAttr &a, const GPUParallelDimAttr &b) {
        return a.getOrder() > b.getOrder();
      });
  if (lb == parDims.end() || *lb != parDim)
    parDims.insert(lb, parDim);
}

/// Map loop parallelism clauses (gang/worker/vector) to GPU parallel
/// dimensions using the given mapping policy.
static SmallVector<GPUParallelDimAttr>
getParallelDimensions(LoopOp loopOp, const ACCToGPUMappingPolicy &policy,
                      DeviceType deviceType) {
  SmallVector<GPUParallelDimAttr> parDims;
  auto *ctx = loopOp->getContext();

  if (loopOp.hasVector(deviceType))
    insertParDim(parDims, policy.vectorDim(ctx));
  if (loopOp.hasWorker(deviceType))
    insertParDim(parDims, policy.workerDim(ctx));
  if (auto gangDimValue = loopOp.getGangValue(GangArgType::Dim, deviceType)) {
    if (auto gangDimDefOp =
            gangDimValue.getDefiningOp<arith::ConstantIntOp>()) {
      auto gangLevel = getGangParLevel(gangDimDefOp.value());
      insertParDim(parDims, policy.gangDim(ctx, gangLevel));
    }
  } else if (loopOp.hasGang(deviceType)) {
    insertParDim(parDims, policy.gangDim(ctx, ParLevel::gang_dim1));
  }
  return parDims;
}

/// Build `acc.compute_region` launch operands: one sequential `acc.par_width`
/// for `acc.serial`, for `acc.parallel` / `acc.kernels` when every num_gangs
/// operand and num_workers / vector_length are the constant 1, and otherwise
/// `acc.par_width` from gang/worker/vector (device-type operands first, then
/// default DeviceType::None).
template <typename ComputeConstructT>
static SmallVector<Value>
assignKnownLaunchArgs(ComputeConstructT computeOp, DeviceType deviceType,
                      RewriterBase &rewriter,
                      const ACCToGPUMappingPolicy &policy) {
  auto *ctx = rewriter.getContext();
  auto loc = computeOp->getLoc();

  if constexpr (std::is_same_v<ComputeConstructT, SerialOp>) {
    return {ParWidthOp::create(rewriter, loc, Value(), policy.seqDim(ctx))};
  } else if constexpr (llvm::is_one_of<ComputeConstructT, ParallelOp,
                                       KernelsOp>::value) {
    if (isEffectivelySerial(computeOp))
      return {ParWidthOp::create(rewriter, loc, Value(), policy.seqDim(ctx))};

    SmallVector<Value> values;
    auto indexTy = rewriter.getIndexType();

    auto numGangs = computeOp.getNumGangsValues(deviceType);
    if (numGangs.empty())
      numGangs = computeOp.getNumGangsValues();
    for (auto [gangDimIdx, gangSize] : llvm::enumerate(numGangs)) {
      auto gangLevel = getGangParLevel(gangDimIdx + 1);
      values.push_back(ParWidthOp::create(
          rewriter, loc,
          getValueOrCreateCastToIndexLike(rewriter, gangSize.getLoc(), indexTy,
                                          gangSize),
          policy.gangDim(ctx, gangLevel)));
    }

    Value numWorkers = computeOp.getNumWorkersValue(deviceType);
    if (!numWorkers)
      numWorkers = computeOp.getNumWorkersValue();
    if (numWorkers) {
      values.push_back(ParWidthOp::create(
          rewriter, loc,
          getValueOrCreateCastToIndexLike(rewriter, numWorkers.getLoc(),
                                          indexTy, numWorkers),
          policy.workerDim(ctx)));
    }

    Value vectorLength = computeOp.getVectorLengthValue(deviceType);
    if (!vectorLength)
      vectorLength = computeOp.getVectorLengthValue();
    if (vectorLength) {
      values.push_back(ParWidthOp::create(
          rewriter, loc,
          getValueOrCreateCastToIndexLike(rewriter, vectorLength.getLoc(),
                                          indexTy, vectorLength),
          policy.vectorDim(ctx)));
    }
    return values;
  } else {
    llvm_unreachable("assignKnownLaunchArgs: expected parallel, kernels, or "
                     "serial");
  }
}

//===----------------------------------------------------------------------===//
// Loop conversion pattern
//===----------------------------------------------------------------------===//

class ACCLoopConversion : public OpRewritePattern<LoopOp> {
public:
  ACCLoopConversion(MLIRContext *ctx, const ACCToGPUMappingPolicy &policy,
                    DeviceType deviceType)
      : OpRewritePattern<LoopOp>(ctx), policy(policy), deviceType(deviceType) {}

  LogicalResult matchAndRewrite(LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    if (loopOp.getUnstructured()) {
      auto executeRegion =
          convertUnstructuredACCLoopToSCFExecuteRegion(loopOp, rewriter);
      if (!executeRegion)
        return failure();
      rewriter.replaceOp(loopOp, executeRegion);
      return success();
    }

    LoopParMode parMode = loopOp.getDefaultOrDeviceTypeParallelism(deviceType);

    if (parMode == LoopParMode::loop_seq || isOpInSerialRegion(loopOp)) {
      // Although it might seem unintuitive, scf.parallel is used here because
      // the parallelism of the loop is already predetermined (as sequential).
      // scf.for will become a candidate for auto-parallelization analysis.
      auto parallelOp = convertACCLoopToSCFParallel(loopOp, rewriter);
      if (!parallelOp)
        return failure();
      setParDimsAttr(parallelOp,
                     GPUParallelDimsAttr::seq(loopOp->getContext()));
      rewriter.replaceOp(loopOp, parallelOp);
    } else if (parMode == LoopParMode::loop_auto) {
      // All loops in serial regions should have already been handled.
      assert(!isOpInSerialRegion(loopOp) &&
             "Expected loop to be in non-serial region");
      // Mark as scf.for to allow auto-parallelization analysis later.
      auto forOp =
          convertACCLoopToSCFFor(loopOp, rewriter, /*enableCollapse=*/true);
      if (!forOp)
        return failure();
      rewriter.replaceOp(loopOp, forOp);
    } else if (!isOpInComputeRegion(loopOp) &&
               !isSpecializedAccRoutine(
                   loopOp->getParentOfType<FunctionOpInterface>())) {
      // This loop is an orphan `acc loop` but it is not in any sort
      // of compute region. Thus it is just a sequential non-accelerator loop.
      auto forOp =
          convertACCLoopToSCFFor(loopOp, rewriter, /*enableCollapse=*/false);
      if (!forOp)
        return failure();
      rewriter.replaceOp(loopOp, forOp);
    } else {
      assert(parMode == LoopParMode::loop_independent &&
             "Expected loop to be independent");
      auto parallelOp = convertACCLoopToSCFParallel(loopOp, rewriter);
      if (!parallelOp)
        return failure();

      SmallVector<GPUParallelDimAttr> parDims =
          getParallelDimensions(loopOp, policy, deviceType);
      if (!parDims.empty()) {
        auto parDimsAttr =
            GPUParallelDimsAttr::get(loopOp->getContext(), parDims);
        setParDimsAttr(parallelOp, parDimsAttr);
      }

      rewriter.replaceOp(loopOp, parallelOp);
    }
    return success();
  }

private:
  const ACCToGPUMappingPolicy &policy;
  DeviceType deviceType;
};

//===----------------------------------------------------------------------===//
// Compute construct conversion pattern
//===----------------------------------------------------------------------===//

template <typename ComputeConstructT>
class ComputeOpConversion : public OpRewritePattern<ComputeConstructT> {
public:
  ComputeOpConversion(MLIRContext *ctx, const ACCToGPUMappingPolicy &policy,
                      DeviceType deviceType)
      : OpRewritePattern<ComputeConstructT>(ctx), policy(policy),
        deviceType(deviceType) {}

  LogicalResult matchAndRewrite(ComputeConstructT computeOp,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(computeOp);
    auto kernelEnv =
        KernelEnvironmentOp::createAndPopulate(computeOp, rewriter);
    auto launchArgs =
        assignKnownLaunchArgs(computeOp, deviceType, rewriter, policy);
    Region &region = computeOp.getRegion();
    SetVector<Value> liveInValues;
    getUsedValuesDefinedAbove(region, region, liveInValues);
    materializeConstantLiveInsIntoRegion(region, liveInValues, rewriter);
    IRMapping mapping;
    auto computeRegion = buildComputeRegion(
        computeOp->getLoc(), launchArgs, liveInValues.getArrayRef(),
        ComputeConstructT::getOperationName(), region, rewriter, mapping);
    if (!computeRegion) {
      rewriter.eraseOp(kernelEnv);
      return failure();
    }
    rewriter.eraseOp(computeOp);
    return success();
  }

private:
  const ACCToGPUMappingPolicy &policy;
  DeviceType deviceType;
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

class ACCComputeLowering
    : public acc::impl::ACCComputeLoweringBase<ACCComputeLowering> {
public:
  using ACCComputeLoweringBase::ACCComputeLoweringBase;

  void runOnOperation() override {
    auto op = getOperation();
    auto *context = op.getContext();

    DefaultACCToGPUMappingPolicy policy;

    // Part 1: Convert acc.loop to scf.parallel/scf.for while the parent
    // compute construct is still present (needed to determine conversion
    // strategy).
    RewritePatternSet loopPatterns(context);
    loopPatterns.insert<ACCLoopConversion>(context, policy, deviceType);
    if (failed(applyPatternsGreedily(op, std::move(loopPatterns))))
      return signalPassFailure();

    // Part 2: Convert acc.parallel, acc.kernels, and acc.serial to
    // acc.kernel_environment { acc.compute_region { ... } }.
    RewritePatternSet computePatterns(context);
    computePatterns
        .insert<ComputeOpConversion<ParallelOp>, ComputeOpConversion<KernelsOp>,
                ComputeOpConversion<SerialOp>>(context, policy, deviceType);
    if (failed(applyPatternsGreedily(op, std::move(computePatterns))))
      return signalPassFailure();
  }
};

} // namespace
