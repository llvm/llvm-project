//===- DataFlowFramework.cpp - A generic framework for data-flow analysis -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlowFramework.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dataflow"
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
#define DATAFLOW_DEBUG(X) LLVM_DEBUG(X)
#else
#define DATAFLOW_DEBUG(X)
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

using namespace mlir;

//===----------------------------------------------------------------------===//
// GenericProgramPoint
//===----------------------------------------------------------------------===//

GenericProgramPoint::~GenericProgramPoint() = default;

//===----------------------------------------------------------------------===//
// AnalysisState
//===----------------------------------------------------------------------===//

AnalysisState::~AnalysisState() = default;

//===----------------------------------------------------------------------===//
// ProgramPoint
//===----------------------------------------------------------------------===//

void ProgramPoint::print(raw_ostream &os) const {
  if (isNull()) {
    os << "<NULL POINT>";
    return;
  }
  if (auto *programPoint = dyn_cast<GenericProgramPoint *>())
    return programPoint->print(os);
  if (auto *op = dyn_cast<Operation *>())
    return op->print(os);
  if (auto value = dyn_cast<Value>())
    return value.print(os);
  if (auto *block = dyn_cast<Block *>())
    return block->print(os);
  auto *region = get<Region *>();
  os << "{\n";
  for (Block &block : *region) {
    block.print(os);
    os << "\n";
  }
  os << "}";
}

Location ProgramPoint::getLoc() const {
  if (auto *programPoint = dyn_cast<GenericProgramPoint *>())
    return programPoint->getLoc();
  if (auto *op = dyn_cast<Operation *>())
    return op->getLoc();
  if (auto value = dyn_cast<Value>())
    return value.getLoc();
  if (auto *block = dyn_cast<Block *>())
    return block->getParent()->getLoc();
  return get<Region *>()->getLoc();
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
  ProgramPoint point;
  DataFlowAnalysis *analysis;

  do {
    // Exhaust the worklist.
    while (!worklist.empty()) {
      std::tie(point, analysis) = worklist.front();
      worklist.pop();

      DATAFLOW_DEBUG(llvm::dbgs() << "Invoking '" << analysis->debugName
                                  << "' on: " << point << "\n");
      if (failed(analysis->visit(point)))
        return failure();
    }

    // "Nudge" the state of the analysis by forcefully initializing states that
    // are still uninitialized. All uninitialized states in the graph can be
    // initialized in any order because the analysis reached fixpoint, meaning
    // that there are no work items that would have further nudged the analysis.
    for (AnalysisState &state :
         llvm::make_pointee_range(llvm::make_second_range(analysisStates))) {
      if (!state.isUninitialized())
        continue;
      DATAFLOW_DEBUG(llvm::dbgs() << "Default initializing " << state.debugName
                                  << " of " << state.point << "\n");
      propagateIfChanged(&state, state.defaultInitialize());
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
                                << " of " << state->point << "\n"
                                << "Value: " << *state << "\n");
    for (const WorkItem &item : state->dependents)
      enqueue(item);
    state->onUpdate(this);
  }
}

void DataFlowSolver::addDependency(AnalysisState *state,
                                   DataFlowAnalysis *analysis,
                                   ProgramPoint point) {
  auto inserted = state->dependents.insert({point, analysis});
  (void)inserted;
  DATAFLOW_DEBUG({
    if (inserted) {
      llvm::dbgs() << "Creating dependency between " << state->debugName
                   << " of " << state->point << "\nand " << analysis->debugName
                   << " on " << point << "\n";
    }
  });
}

//===----------------------------------------------------------------------===//
// DataFlowAnalysis
//===----------------------------------------------------------------------===//

DataFlowAnalysis::~DataFlowAnalysis() = default;

DataFlowAnalysis::DataFlowAnalysis(DataFlowSolver &solver) : solver(solver) {}

void DataFlowAnalysis::addDependency(AnalysisState *state, ProgramPoint point) {
  solver.addDependency(state, this, point);
}

void DataFlowAnalysis::propagateIfChanged(AnalysisState *state,
                                          ChangeResult changed) {
  solver.propagateIfChanged(state, changed);
}
