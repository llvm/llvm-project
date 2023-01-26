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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// GetParentForOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform::GetParentForOp::apply(transform::TransformResults &results,
                                 transform::TransformState &state) {
  SetVector<Operation *> parents;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    Operation *loop, *current = target;
    for (unsigned i = 0, e = getNumLoops(); i < e; ++i) {
      loop = getAffine()
                 ? current->getParentOfType<AffineForOp>().getOperation()
                 : current->getParentOfType<scf::ForOp>().getOperation();
      if (!loop) {
        DiagnosedSilenceableFailure diag =
            emitSilenceableError()
            << "could not find an '"
            << (getAffine() ? AffineForOp::getOperationName()
                            : scf::ForOp::getOperationName())
            << "' parent";
        diag.attachNote(target->getLoc()) << "target op";
        return diag;
      }
      current = loop;
    }
    parents.insert(loop);
  }
  results.set(getResult().cast<OpResult>(), parents.getArrayRef());
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
transform::LoopOutlineOp::apply(transform::TransformResults &results,
                                transform::TransformState &state) {
  SmallVector<Operation *> transformed;
  DenseMap<Operation *, SymbolTable> symbolTables;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    Location location = target->getLoc();
    Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(target);
    TrivialPatternRewriter rewriter(getContext());
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
    transformed.push_back(*outlined);
  }
  results.set(getTransformed().cast<OpResult>(), transformed);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LoopPeelOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::LoopPeelOp::applyToOne(scf::ForOp target,
                                  transform::ApplyToEachResultList &results,
                                  transform::TransformState &state) {
  scf::ForOp result;
  IRRewriter rewriter(target->getContext());
  // This helper returns failure when peeling does not occur (i.e. when the IR
  // is not modified). This is not a failure for the op as the postcondition:
  //    "the loop trip count is divisible by the step"
  // is valid.
  LogicalResult status =
      scf::peelAndCanonicalizeForLoop(rewriter, target, result);
  // TODO: Return both the peeled loop and the remainder loop.
  results.push_back(failed(status) ? target : result);
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
      earlyCycle = std::max(earlyCycle, opCycles[def] + getLatency(def));
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
transform::LoopPipelineOp::applyToOne(scf::ForOp target,
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
  TrivialPatternRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<scf::ForOp> patternResult =
      pattern.returningMatchAndRewrite(target, rewriter);
  if (succeeded(patternResult)) {
    results.push_back(*patternResult);
    return DiagnosedSilenceableFailure::success();
  }
  return emitDefaultSilenceableFailure(target);
}

//===----------------------------------------------------------------------===//
// LoopUnrollOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::LoopUnrollOp::applyToOne(Operation *op,
                                    transform::ApplyToEachResultList &results,
                                    transform::TransformState &state) {
  LogicalResult result(failure());
  if (scf::ForOp scfFor = dyn_cast<scf::ForOp>(op))
    result = loopUnrollByFactor(scfFor, getFactor());
  else if (AffineForOp affineFor = dyn_cast<AffineForOp>(op))
    result = loopUnrollByFactor(affineFor, getFactor());

  if (failed(result)) {
    DiagnosedSilenceableFailure diag = emitSilenceableError()
                                       << "failed to unroll";
    return diag;
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LoopCoalesceOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::LoopCoalesceOp::applyToOne(Operation *op,
                                      transform::ApplyToEachResultList &results,
                                      transform::TransformState &state) {
  LogicalResult result(failure());
  if (scf::ForOp scfForOp = dyn_cast<scf::ForOp>(op))
    result = coalescePerfectlyNestedLoops(scfForOp);
  else if (AffineForOp affineForOp = dyn_cast<AffineForOp>(op))
    result = coalescePerfectlyNestedLoops(affineForOp);

  results.push_back(op);
  if (failed(result)) {
    DiagnosedSilenceableFailure diag = emitSilenceableError()
                                       << "failed to coalesce";
    return diag;
  }
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
    declareGeneratedDialect<AffineDialect>();
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
