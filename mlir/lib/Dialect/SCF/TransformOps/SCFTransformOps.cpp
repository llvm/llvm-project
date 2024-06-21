//===- SCFTransformOps.cpp - Implementation of SCF transformation ops -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"

using namespace mlir;
using namespace mlir::affine;

//===----------------------------------------------------------------------===//
// Apply...PatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyForLoopCanonicalizationPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
}

void transform::ApplySCFStructuralConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  scf::populateSCFStructuralTypeConversions(typeConverter, patterns);
}

void transform::ApplySCFStructuralConversionPatternsOp::
    populateConversionTargetRules(const TypeConverter &typeConverter,
                                  ConversionTarget &conversionTarget) {
  scf::populateSCFStructuralTypeConversionTarget(typeConverter,
                                                 conversionTarget);
}

//===----------------------------------------------------------------------===//
// ForallToForOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ForallToForOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {
  auto payload = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payload))
    return emitSilenceableError() << "expected a single payload op";

  auto target = dyn_cast<scf::ForallOp>(*payload.begin());
  if (!target) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError() << "expected the payload to be scf.forall";
    diag.attachNote((*payload.begin())->getLoc()) << "payload op";
    return diag;
  }

  if (!target.getOutputs().empty()) {
    return emitSilenceableError()
           << "unsupported shared outputs (didn't bufferize?)";
  }

  SmallVector<OpFoldResult> lbs = target.getMixedLowerBound();

  if (getNumResults() != lbs.size()) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError()
        << "op expects as many results (" << getNumResults()
        << ") as payload has induction variables (" << lbs.size() << ")";
    diag.attachNote(target.getLoc()) << "payload op";
    return diag;
  }

  SmallVector<Operation *> opResults;
  if (failed(scf::forallToForLoop(rewriter, target, &opResults))) {
    DiagnosedSilenceableFailure diag = emitSilenceableError()
                                       << "failed to convert forall into for";
    return diag;
  }

  for (auto &&[i, res] : llvm::enumerate(opResults)) {
    results.set(cast<OpResult>(getTransformed()[i]), {res});
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ForallToForOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ForallToParallelOp::apply(transform::TransformRewriter &rewriter,
                                     transform::TransformResults &results,
                                     transform::TransformState &state) {
  auto payload = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payload))
    return emitSilenceableError() << "expected a single payload op";

  auto target = dyn_cast<scf::ForallOp>(*payload.begin());
  if (!target) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError() << "expected the payload to be scf.forall";
    diag.attachNote((*payload.begin())->getLoc()) << "payload op";
    return diag;
  }

  if (!target.getOutputs().empty()) {
    return emitSilenceableError()
           << "unsupported shared outputs (didn't bufferize?)";
  }

  if (getNumResults() != 1) {
    DiagnosedSilenceableFailure diag = emitSilenceableError()
                                       << "op expects one result, given "
                                       << getNumResults();
    diag.attachNote(target.getLoc()) << "payload op";
    return diag;
  }

  scf::ParallelOp opResult;
  if (failed(scf::forallToParallelLoop(rewriter, target, &opResult))) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError() << "failed to convert forall into parallel";
    return diag;
  }

  results.set(cast<OpResult>(getTransformed()[0]), {opResult});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LoopOutlineOp
//===----------------------------------------------------------------------===//

/// Wraps the given operation `op` into an `scf.execute_region` operation. Uses
/// the provided rewriter for all operations to remain compatible with the
/// rewriting infra, as opposed to just splicing the op in place.
static scf::ExecuteRegionOp wrapInExecuteRegion(RewriterBase &b,
                                                Operation *op) {
  if (op->getNumRegions() != 1)
    return nullptr;
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  scf::ExecuteRegionOp executeRegionOp =
      b.create<scf::ExecuteRegionOp>(op->getLoc(), op->getResultTypes());
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&executeRegionOp.getRegion().emplaceBlock());
    Operation *clonedOp = b.cloneWithoutRegions(*op);
    Region &clonedRegion = clonedOp->getRegions().front();
    assert(clonedRegion.empty() && "expected empty region");
    b.inlineRegionBefore(op->getRegions().front(), clonedRegion,
                         clonedRegion.end());
    b.create<scf::YieldOp>(op->getLoc(), clonedOp->getResults());
  }
  b.replaceOp(op, executeRegionOp.getResults());
  return executeRegionOp;
}

DiagnosedSilenceableFailure
transform::LoopOutlineOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {
  SmallVector<Operation *> functions;
  SmallVector<Operation *> calls;
  DenseMap<Operation *, SymbolTable> symbolTables;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    Location location = target->getLoc();
    Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(target);
    scf::ExecuteRegionOp exec = wrapInExecuteRegion(rewriter, target);
    if (!exec) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "failed to outline";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    func::CallOp call;
    FailureOr<func::FuncOp> outlined = outlineSingleBlockRegion(
        rewriter, location, exec.getRegion(), getFuncName(), &call);

    if (failed(outlined))
      return emitDefaultDefiniteFailure(target);

    if (symbolTableOp) {
      SymbolTable &symbolTable =
          symbolTables.try_emplace(symbolTableOp, symbolTableOp)
              .first->getSecond();
      symbolTable.insert(*outlined);
      call.setCalleeAttr(FlatSymbolRefAttr::get(*outlined));
    }
    functions.push_back(*outlined);
    calls.push_back(call);
  }
  results.set(cast<OpResult>(getFunction()), functions);
  results.set(cast<OpResult>(getCall()), calls);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LoopPeelOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::LoopPeelOp::applyToOne(transform::TransformRewriter &rewriter,
                                  scf::ForOp target,
                                  transform::ApplyToEachResultList &results,
                                  transform::TransformState &state) {
  scf::ForOp result;
  if (getPeelFront()) {
    LogicalResult status =
        scf::peelForLoopFirstIteration(rewriter, target, result);
    if (failed(status)) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "failed to peel the first iteration";
      return diag;
    }
  } else {
    LogicalResult status =
        scf::peelForLoopAndSimplifyBounds(rewriter, target, result);
    if (failed(status)) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "failed to peel the last iteration";
      return diag;
    }
  }

  results.push_back(target);
  results.push_back(result);

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LoopPipelineOp
//===----------------------------------------------------------------------===//

/// Callback for PipeliningOption. Populates `schedule` with the mapping from an
/// operation to its logical time position given the iteration interval and the
/// read latency. The latter is only relevant for vector transfers.
static void
loopScheduling(scf::ForOp forOp,
               std::vector<std::pair<Operation *, unsigned>> &schedule,
               unsigned iterationInterval, unsigned readLatency) {
  auto getLatency = [&](Operation *op) -> unsigned {
    if (isa<vector::TransferReadOp>(op))
      return readLatency;
    return 1;
  };

  std::optional<int64_t> ubConstant = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> lbConstant = getConstantIntValue(forOp.getLowerBound());
  DenseMap<Operation *, unsigned> opCycles;
  std::map<unsigned, std::vector<Operation *>> wrappedSchedule;
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    unsigned earlyCycle = 0;
    for (Value operand : op.getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;
      if (ubConstant && lbConstant) {
        unsigned ubInt = ubConstant.value();
        unsigned lbInt = lbConstant.value();
        auto minLatency = std::min(ubInt - lbInt - 1, getLatency(def));
        earlyCycle = std::max(earlyCycle, opCycles[def] + minLatency);
      } else {
        earlyCycle = std::max(earlyCycle, opCycles[def] + getLatency(def));
      }
    }
    opCycles[&op] = earlyCycle;
    wrappedSchedule[earlyCycle % iterationInterval].push_back(&op);
  }
  for (const auto &it : wrappedSchedule) {
    for (Operation *op : it.second) {
      unsigned cycle = opCycles[op];
      schedule.emplace_back(op, cycle / iterationInterval);
    }
  }
}

DiagnosedSilenceableFailure
transform::LoopPipelineOp::applyToOne(transform::TransformRewriter &rewriter,
                                      scf::ForOp target,
                                      transform::ApplyToEachResultList &results,
                                      transform::TransformState &state) {
  scf::PipeliningOption options;
  options.getScheduleFn =
      [this](scf::ForOp forOp,
             std::vector<std::pair<Operation *, unsigned>> &schedule) mutable {
        loopScheduling(forOp, schedule, getIterationInterval(),
                       getReadLatency());
      };
  scf::ForLoopPipeliningPattern pattern(options, target->getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<scf::ForOp> patternResult =
      scf::pipelineForLoop(rewriter, target, options);
  if (succeeded(patternResult)) {
    results.push_back(*patternResult);
    return DiagnosedSilenceableFailure::success();
  }
  return emitDefaultSilenceableFailure(target);
}

//===----------------------------------------------------------------------===//
// LoopPromoteIfOneIterationOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::LoopPromoteIfOneIterationOp::applyToOne(
    transform::TransformRewriter &rewriter, LoopLikeOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  (void)target.promoteIfSingleIteration(rewriter);
  return DiagnosedSilenceableFailure::success();
}

void transform::LoopPromoteIfOneIterationOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// LoopUnrollOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::LoopUnrollOp::applyToOne(transform::TransformRewriter &rewriter,
                                    Operation *op,
                                    transform::ApplyToEachResultList &results,
                                    transform::TransformState &state) {
  LogicalResult result(failure());
  if (scf::ForOp scfFor = dyn_cast<scf::ForOp>(op))
    result = loopUnrollByFactor(scfFor, getFactor());
  else if (AffineForOp affineFor = dyn_cast<AffineForOp>(op))
    result = loopUnrollByFactor(affineFor, getFactor());
  else
    return emitSilenceableError()
           << "failed to unroll, incorrect type of payload";

  if (failed(result))
    return emitSilenceableError() << "failed to unroll";

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LoopUnrollAndJamOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::LoopUnrollAndJamOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *op,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  LogicalResult result(failure());
  if (scf::ForOp scfFor = dyn_cast<scf::ForOp>(op))
    result = loopUnrollJamByFactor(scfFor, getFactor());
  else if (AffineForOp affineFor = dyn_cast<AffineForOp>(op))
    result = loopUnrollJamByFactor(affineFor, getFactor());
  else
    return emitSilenceableError()
           << "failed to unroll and jam, incorrect type of payload";

  if (failed(result))
    return emitSilenceableError() << "failed to unroll and jam";

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LoopCoalesceOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::LoopCoalesceOp::applyToOne(transform::TransformRewriter &rewriter,
                                      Operation *op,
                                      transform::ApplyToEachResultList &results,
                                      transform::TransformState &state) {
  LogicalResult result(failure());
  if (scf::ForOp scfForOp = dyn_cast<scf::ForOp>(op))
    result = coalescePerfectlyNestedSCFForLoops(scfForOp);
  else if (AffineForOp affineForOp = dyn_cast<AffineForOp>(op))
    result = coalescePerfectlyNestedAffineLoops(affineForOp);

  results.push_back(op);
  if (failed(result)) {
    DiagnosedSilenceableFailure diag = emitSilenceableError()
                                       << "failed to coalesce";
    return diag;
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// TakeAssumedBranchOp
//===----------------------------------------------------------------------===//
/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(RewriterBase &rewriter, Operation *op,
                                Region &region) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, /*blockArgs=*/{});
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

DiagnosedSilenceableFailure transform::TakeAssumedBranchOp::applyToOne(
    transform::TransformRewriter &rewriter, scf::IfOp ifOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  rewriter.setInsertionPoint(ifOp);
  Region &region =
      getTakeElseBranch() ? ifOp.getElseRegion() : ifOp.getThenRegion();
  if (!llvm::hasSingleElement(region)) {
    return emitDefiniteFailure()
           << "requires an scf.if op with a single-block "
           << ((getTakeElseBranch()) ? "`else`" : "`then`") << " region";
  }
  replaceOpWithRegion(rewriter, ifOp, region);
  return DiagnosedSilenceableFailure::success();
}

void transform::TakeAssumedBranchOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// LoopFuseSiblingOp
//===----------------------------------------------------------------------===//

/// Check if `target` and `source` are siblings, in the context that `target`
/// is being fused into `source`.
///
/// This is a simple check that just checks if both operations are in the same
/// block and some checks to ensure that the fused IR does not violate
/// dominance.
static DiagnosedSilenceableFailure isOpSibling(Operation *target,
                                               Operation *source) {
  // Check if both operations are same.
  if (target == source)
    return emitSilenceableFailure(source)
           << "target and source need to be different loops";

  // Check if both operations are in the same block.
  if (target->getBlock() != source->getBlock())
    return emitSilenceableFailure(source)
           << "target and source are not in the same block";

  // Check if fusion will violate dominance.
  DominanceInfo domInfo(source);
  if (target->isBeforeInBlock(source)) {
    // Since `target` is before `source`, all users of results of `target`
    // need to be dominated by `source`.
    for (Operation *user : target->getUsers()) {
      if (!domInfo.properlyDominates(source, user, /*enclosingOpOk=*/false)) {
        return emitSilenceableFailure(target)
               << "user of results of target should be properly dominated by "
                  "source";
      }
    }
  } else {
    // Since `target` is after `source`, all values used by `target` need
    // to dominate `source`.

    // Check if operands of `target` are dominated by `source`.
    for (Value operand : target->getOperands()) {
      Operation *operandOp = operand.getDefiningOp();
      // Operands without defining operations are block arguments. When `target`
      // and `source` occur in the same block, these operands dominate `source`.
      if (!operandOp)
        continue;

      // Operand's defining operation should properly dominate `source`.
      if (!domInfo.properlyDominates(operandOp, source,
                                     /*enclosingOpOk=*/false))
        return emitSilenceableFailure(target)
               << "operands of target should be properly dominated by source";
    }

    // Check if values used by `target` are dominated by `source`.
    bool failed = false;
    OpOperand *failedValue = nullptr;
    visitUsedValuesDefinedAbove(target->getRegions(), [&](OpOperand *operand) {
      Operation *operandOp = operand->get().getDefiningOp();
      if (operandOp && !domInfo.properlyDominates(operandOp, source,
                                                  /*enclosingOpOk=*/false)) {
        // `operand` is not an argument of an enclosing block and the defining
        // op of `operand` is outside `target` but does not dominate `source`.
        failed = true;
        failedValue = operand;
      }
    });

    if (failed)
      return emitSilenceableFailure(failedValue->getOwner())
             << "values used inside regions of target should be properly "
                "dominated by source";
  }

  return DiagnosedSilenceableFailure::success();
}

/// Check if `target` scf.forall can be fused into `source` scf.forall.
///
/// This simply checks if both loops have the same bounds, steps and mapping.
/// No attempt is made at checking that the side effects of `target` and
/// `source` are independent of each other.
static bool isForallWithIdenticalConfiguration(Operation *target,
                                               Operation *source) {
  auto targetOp = dyn_cast<scf::ForallOp>(target);
  auto sourceOp = dyn_cast<scf::ForallOp>(source);
  if (!targetOp || !sourceOp)
    return false;

  return targetOp.getMixedLowerBound() == sourceOp.getMixedLowerBound() &&
         targetOp.getMixedUpperBound() == sourceOp.getMixedUpperBound() &&
         targetOp.getMixedStep() == sourceOp.getMixedStep() &&
         targetOp.getMapping() == sourceOp.getMapping();
}

/// Check if `target` scf.for can be fused into `source` scf.for.
///
/// This simply checks if both loops have the same bounds and steps. No attempt
/// is made at checking that the side effects of `target` and `source` are
/// independent of each other.
static bool isForWithIdenticalConfiguration(Operation *target,
                                            Operation *source) {
  auto targetOp = dyn_cast<scf::ForOp>(target);
  auto sourceOp = dyn_cast<scf::ForOp>(source);
  if (!targetOp || !sourceOp)
    return false;

  return targetOp.getLowerBound() == sourceOp.getLowerBound() &&
         targetOp.getUpperBound() == sourceOp.getUpperBound() &&
         targetOp.getStep() == sourceOp.getStep();
}

DiagnosedSilenceableFailure
transform::LoopFuseSiblingOp::apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {
  auto targetOps = state.getPayloadOps(getTarget());
  auto sourceOps = state.getPayloadOps(getSource());

  if (!llvm::hasSingleElement(targetOps) ||
      !llvm::hasSingleElement(sourceOps)) {
    return emitDefiniteFailure()
           << "requires exactly one target handle (got "
           << llvm::range_size(targetOps) << ") and exactly one "
           << "source handle (got " << llvm::range_size(sourceOps) << ")";
  }

  Operation *target = *targetOps.begin();
  Operation *source = *sourceOps.begin();

  // Check if the target and source are siblings.
  DiagnosedSilenceableFailure diag = isOpSibling(target, source);
  if (!diag.succeeded())
    return diag;

  Operation *fusedLoop;
  /// TODO: Support fusion for loop-like ops besides scf.for and scf.forall.
  if (isForWithIdenticalConfiguration(target, source)) {
    fusedLoop = fuseIndependentSiblingForLoops(
        cast<scf::ForOp>(target), cast<scf::ForOp>(source), rewriter);
  } else if (isForallWithIdenticalConfiguration(target, source)) {
    fusedLoop = fuseIndependentSiblingForallLoops(
        cast<scf::ForallOp>(target), cast<scf::ForallOp>(source), rewriter);
  } else
    return emitSilenceableFailure(target->getLoc())
           << "operations cannot be fused";

  assert(fusedLoop && "failed to fuse operations");

  results.set(cast<OpResult>(getFusedLoop()), {fusedLoop});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class SCFTransformDialectExtension
    : public transform::TransformDialectExtension<
          SCFTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<func::FuncDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.cpp.inc"

void mlir::scf::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<SCFTransformDialectExtension>();
}
