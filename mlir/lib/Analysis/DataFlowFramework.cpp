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
#include "llvm/ADT/iterator.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
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
      llvm::dbgs() << "Creating dependency between " << debugName << " of "
                   << anchor << "\nand " << debugName << " on " << dependent
                   << "\n";
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
    os << "<after operation>:";
    return getPrevOp()->print(os, OpPrintingFlags().skipRegions());
  }
  os << "<before operation>:";
  return getNextOp()->print(os, OpPrintingFlags().skipRegions());
}

//===----------------------------------------------------------------------===//
// LatticeAnchor
//===----------------------------------------------------------------------===//

void LatticeAnchor::print(raw_ostream &os) const {
  if (isNull()) {
    os << "<NULL POINT>";
    return;
  }
  if (auto *LatticeAnchor = llvm::dyn_cast<GenericLatticeAnchor *>(*this))
    return LatticeAnchor->print(os);
  if (auto value = llvm::dyn_cast<Value>(*this)) {
    return value.print(os, OpPrintingFlags().skipRegions());
  }

  return get<ProgramPoint *>()->print(os);
}

Location LatticeAnchor::getLoc() const {
  if (auto *LatticeAnchor = llvm::dyn_cast<GenericLatticeAnchor *>(*this))
    return LatticeAnchor->getLoc();
  if (auto value = llvm::dyn_cast<Value>(*this))
    return value.getLoc();

  ProgramPoint *pp = get<ProgramPoint *>();
  if (!pp->isBlockStart())
    return pp->getPrevOp()->getLoc();
  return pp->getBlock()->getParent()->getLoc();
}

//===----------------------------------------------------------------------===//
// DataFlowSolver
//===----------------------------------------------------------------------===//

LogicalResult DataFlowSolver::initializeAndRun(Operation *top) {
  // Initialize the analyses.
  for (DataFlowAnalysis &analysis : llvm::make_pointee_range(childAnalyses)) {
    DATAFLOW_DEBUG(llvm::dbgs()
                   << "Priming analysis: " << analysis.debugName << "\n");
    if (failed(analysis.initialize(top)))
      return failure();
  }

  // Run the analysis until fixpoint.
  do {
    // Exhaust the worklist.
    while (!worklist.empty()) {
      auto [point, analysis] = worklist.front();
      worklist.pop();

      DATAFLOW_DEBUG(llvm::dbgs() << "Invoking '" << analysis->debugName
                                  << "' on: " << point << "\n");
      if (failed(analysis->visit(point)))
        return failure();
    }

    // Iterate until all states are in some initialized state and the worklist
    // is exhausted.
  } while (!worklist.empty());

  return success();
}

void DataFlowSolver::propagateIfChanged(AnalysisState *state,
                                        ChangeResult changed) {
  if (changed == ChangeResult::Change) {
    DATAFLOW_DEBUG(llvm::dbgs() << "Propagating update to " << state->debugName
                                << " of " << state->anchor << "\n"
                                << "Value: " << *state << "\n");
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
