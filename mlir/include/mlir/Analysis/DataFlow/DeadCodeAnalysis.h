//===- DeadCodeAnalysis.h - Dead code analysis ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements dead code analysis using the data-flow analysis
// framework. This analysis uses the results of constant propagation to
// determine live blocks, control-flow edges, and control-flow predecessors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_DEADCODEANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOW_DEADCODEANALYSIS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {

class CallOpInterface;
class CallableOpInterface;
class BranchOpInterface;
class RegionBranchOpInterface;

namespace dataflow {

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

/// This is a simple analysis state that represents whether the associated
/// program point (either a block or a control-flow edge) is live.
class Executable : public AnalysisState {
public:
  using AnalysisState::AnalysisState;

  /// The state is initialized by default.
  bool isUninitialized() const override { return false; }

  /// The state is always initialized.
  ChangeResult defaultInitialize() override { return ChangeResult::NoChange; }

  /// Set the state of the program point to live.
  ChangeResult setToLive();

  /// Get whether the program point is live.
  bool isLive() const { return live; }

  /// Print the liveness.
  void print(raw_ostream &os) const override;

  /// When the state of the program point is changed to live, re-invoke
  /// subscribed analyses on the operations in the block and on the block
  /// itself.
  void onUpdate(DataFlowSolver *solver) const override;

  /// Subscribe an analysis to changes to the liveness.
  void blockContentSubscribe(DataFlowAnalysis *analysis) {
    subscribers.insert(analysis);
  }

private:
  /// Whether the program point is live. Optimistically assume that the program
  /// point is dead.
  bool live = false;

  /// A set of analyses that should be updated when this state changes.
  SetVector<DataFlowAnalysis *, SmallVector<DataFlowAnalysis *, 4>,
            SmallPtrSet<DataFlowAnalysis *, 4>>
      subscribers;
};

//===----------------------------------------------------------------------===//
// PredecessorState
//===----------------------------------------------------------------------===//

/// This analysis state represents a set of live control-flow "predecessors" of
/// a program point (either an operation or a block), which are the last
/// operations along all execution paths that pass through this point.
///
/// For example, in dead-code analysis, an operation with region control-flow
/// can be the predecessor of a region's entry block or itself, the exiting
/// terminator of a region can be the predecessor of the parent operation or
/// another region's entry block, the callsite of a callable operation can be
/// the predecessor to its entry block, and the exiting terminator or a callable
/// operation can be the predecessor of the call operation.
///
/// The state can indicate that it is underdefined, meaning that not all live
/// control-flow predecessors can be known.
class PredecessorState : public AnalysisState {
public:
  using AnalysisState::AnalysisState;

  /// The state is initialized by default.
  bool isUninitialized() const override { return false; }

  /// The state is always initialized.
  ChangeResult defaultInitialize() override { return ChangeResult::NoChange; }

  /// Print the known predecessors.
  void print(raw_ostream &os) const override;

  /// Returns true if all predecessors are known.
  bool allPredecessorsKnown() const { return allKnown; }

  /// Indicate that there are potentially unknown predecessors.
  ChangeResult setHasUnknownPredecessors() {
    return std::exchange(allKnown, false) ? ChangeResult::Change
                                          : ChangeResult::NoChange;
  }

  /// Get the known predecessors.
  ArrayRef<Operation *> getKnownPredecessors() const {
    return knownPredecessors.getArrayRef();
  }

  /// Add a known predecessor.
  ChangeResult join(Operation *predecessor) {
    return knownPredecessors.insert(predecessor) ? ChangeResult::Change
                                                 : ChangeResult::NoChange;
  }

private:
  /// Whether all predecessors are known. Optimistically assume that we know
  /// all predecessors.
  bool allKnown = true;

  /// The known control-flow predecessors of this program point.
  SetVector<Operation *, SmallVector<Operation *, 4>,
            SmallPtrSet<Operation *, 4>>
      knownPredecessors;
};

//===----------------------------------------------------------------------===//
// CFGEdge
//===----------------------------------------------------------------------===//

/// This program point represents a control-flow edge between a block and one
/// of its successors.
class CFGEdge
    : public GenericProgramPointBase<CFGEdge, std::pair<Block *, Block *>> {
public:
  using Base::Base;

  /// Get the block from which the edge originates.
  Block *getFrom() const { return getValue().first; }
  /// Get the target block.
  Block *getTo() const { return getValue().second; }

  /// Print the blocks between the control-flow edge.
  void print(raw_ostream &os) const override;
  /// Get a fused location of both blocks.
  Location getLoc() const override;
};

//===----------------------------------------------------------------------===//
// DeadCodeAnalysis
//===----------------------------------------------------------------------===//

/// Dead code analysis analyzes control-flow, as understood by
/// `RegionBranchOpInterface` and `BranchOpInterface`, and the callgraph, as
/// understood by `CallableOpInterface` and `CallOpInterface`.
///
/// This analysis uses known constant values of operands to determine the
/// liveness of each block and each edge between a block and its predecessors.
/// For region control-flow, this analysis determines the predecessor operations
/// for region entry blocks and region control-flow operations. For the
/// callgraph, this analysis determines the callsites and live returns of every
/// function.
class DeadCodeAnalysis : public DataFlowAnalysis {
public:
  explicit DeadCodeAnalysis(DataFlowSolver &solver);

  /// Initialize the analysis by visiting every operation with potential
  /// control-flow semantics.
  LogicalResult initialize(Operation *top) override;

  /// Visit an operation with control-flow semantics and deduce which of its
  /// successors are live.
  LogicalResult visit(ProgramPoint point) override;

private:
  /// Find and mark symbol callables with potentially unknown callsites as
  /// having overdefined predecessors. `top` is the top-level operation that the
  /// analysis is operating on.
  void initializeSymbolCallables(Operation *top);

  /// Recursively Initialize the analysis on nested regions.
  LogicalResult initializeRecursively(Operation *op);

  /// Visit the given call operation and compute any necessary lattice state.
  void visitCallOperation(CallOpInterface call);

  /// Visit the given branch operation with successors and try to determine
  /// which are live from the current block.
  void visitBranchOperation(BranchOpInterface branch);

  /// Visit the given region branch operation, which defines regions, and
  /// compute any necessary lattice state. This also resolves the lattice state
  /// of both the operation results and any nested regions.
  void visitRegionBranchOperation(RegionBranchOpInterface branch);

  /// Visit the given terminator operation that exits a region under an
  /// operation with control-flow semantics. These are terminators with no CFG
  /// successors.
  void visitRegionTerminator(Operation *op, RegionBranchOpInterface branch);

  /// Visit the given terminator operation that exits a callable region. These
  /// are terminators with no CFG successors.
  void visitCallableTerminator(Operation *op, CallableOpInterface callable);

  /// Mark the edge between `from` and `to` as executable.
  void markEdgeLive(Block *from, Block *to);

  /// Mark the entry blocks of the operation as executable.
  void markEntryBlocksLive(Operation *op);

  /// Get the constant values of the operands of the operation. Returns none if
  /// any of the operand lattices are uninitialized.
  Optional<SmallVector<Attribute>> getOperandValues(Operation *op);

  /// A symbol table used for O(1) symbol lookups during simplification.
  SymbolTableCollection symbolTable;
};

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_DEADCODEANALYSIS_H
