//===- DataFlowFramework.cpp - A generic framework for data-flow analysis -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "dataflow"
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
#define DATAFLOW_DEBUG(X) LLVM_DEBUG(X)
#else
#define DATAFLOW_DEBUG(X)
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

using namespace mlir;

//===----------------------------------------------------------------------===//
// GenericLatticeAnchor
//===----------------------------------------------------------------------===//

GenericLatticeAnchor::~GenericLatticeAnchor() = default;

//===----------------------------------------------------------------------===//
// AnalysisState
//===----------------------------------------------------------------------===//

AnalysisState::~AnalysisState() = default;

void AnalysisState::addDependency(ProgramPoint *dependent,
                                  DataFlowAnalysis *analysis) {
  auto inserted = dependents.insert({dependent, analysis});
  (void)inserted;
  DATAFLOW_DEBUG({
    if (inserted) {
      LDBG() << "Creating dependency between " << debugName << " of " << anchor
             << "\nand " << debugName << " on " << *dependent;
    }
  });
}

void AnalysisState::dump() const { print(llvm::errs()); }

//===----------------------------------------------------------------------===//
// ProgramPoint
//===----------------------------------------------------------------------===//

void ProgramPoint::print(raw_ostream &os) const {
  if (isNull()) {
    os << "<NULL POINT>";
    return;
  }
  if (!isBlockStart()) {
    os << "<after operation>:"
       << OpWithFlags(getPrevOp(), OpPrintingFlags().skipRegions());
    return;
  }
  os << "<before operation>:"
     << OpWithFlags(getNextOp(), OpPrintingFlags().skipRegions());
}

//===----------------------------------------------------------------------===//
// LatticeAnchor
//===----------------------------------------------------------------------===//

void LatticeAnchor::print(raw_ostream &os) const {
  if (isNull()) {
    os << "<NULL POINT>";
    return;
  }
  if (auto *latticeAnchor = llvm::dyn_cast<GenericLatticeAnchor *>(*this))
    return latticeAnchor->print(os);
  if (auto value = llvm::dyn_cast<Value>(*this)) {
    return value.print(os, OpPrintingFlags().skipRegions());
  }

  return llvm::cast<ProgramPoint *>(*this)->print(os);
}

Location LatticeAnchor::getLoc() const {
  if (auto *latticeAnchor = llvm::dyn_cast<GenericLatticeAnchor *>(*this))
    return latticeAnchor->getLoc();
  if (auto value = llvm::dyn_cast<Value>(*this))
    return value.getLoc();

  ProgramPoint *pp = llvm::cast<ProgramPoint *>(*this);
  if (!pp->isBlockStart())
    return pp->getPrevOp()->getLoc();
  return pp->getBlock()->getParent()->getLoc();
}

//===----------------------------------------------------------------------===//
// DataFlowSolver
//===----------------------------------------------------------------------===//

LogicalResult DataFlowSolver::initializeAndRun(Operation *top) {
  // Enable enqueue to the worklist.
  isRunning = true;
  auto guard = llvm::make_scope_exit([&]() { isRunning = false; });

  // Initialize equivalent lattice anchors.
  for (DataFlowAnalysis &analysis : llvm::make_pointee_range(childAnalyses)) {
    analysis.initializeEquivalentLatticeAnchor(top);
  }

  // Initialize the analyses.
  for (DataFlowAnalysis &analysis : llvm::make_pointee_range(childAnalyses)) {
    DATAFLOW_DEBUG(LDBG() << "Priming analysis: " << analysis.debugName);
    if (failed(analysis.initialize(top)))
      return failure();
  }

  // Run the analysis until fixpoint.
  // Iterate until all states are in some initialized state and the worklist
  // is exhausted.
  while (!worklist.empty()) {
    auto [point, analysis] = worklist.front();
    worklist.pop();

    DATAFLOW_DEBUG(LDBG() << "Invoking '" << analysis->debugName
                          << "' on: " << *point);
    if (failed(analysis->visit(point)))
      return failure();
  }

  return success();
}

void DataFlowSolver::propagateIfChanged(AnalysisState *state,
                                        ChangeResult changed) {
  assert(isRunning &&
         "DataFlowSolver is not running, should not use propagateIfChanged");
  if (changed == ChangeResult::Change) {
    DATAFLOW_DEBUG(LDBG() << "Propagating update to " << state->debugName
                          << " of " << state->anchor << "\n"
                          << "Value: " << *state);
    state->onUpdate(this);
  }
}

//===----------------------------------------------------------------------===//
// DataFlowAnalysis
//===----------------------------------------------------------------------===//

DataFlowAnalysis::~DataFlowAnalysis() = default;

DataFlowAnalysis::DataFlowAnalysis(DataFlowSolver &solver) : solver(solver) {}

void DataFlowAnalysis::addDependency(AnalysisState *state,
                                     ProgramPoint *point) {
  state->addDependency(point, this);
}

void DataFlowAnalysis::propagateIfChanged(AnalysisState *state,
                                          ChangeResult changed) {
  solver.propagateIfChanged(state, changed);
}
