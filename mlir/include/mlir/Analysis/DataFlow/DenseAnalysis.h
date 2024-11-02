//===- DenseAnalysis.h - Dense data-flow analysis -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements dense data-flow analysis using the data-flow analysis
// framework. The analysis is forward and conditional and uses the results of
// dead code analysis to prune dead code during the analysis.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DENSEDATAFLOWANALYSIS_H
#define MLIR_ANALYSIS_DENSEDATAFLOWANALYSIS_H

#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir {

class RegionBranchOpInterface;

namespace dataflow {

//===----------------------------------------------------------------------===//
// AbstractDenseLattice
//===----------------------------------------------------------------------===//

/// This class represents a dense lattice. A dense lattice is attached to
/// operations to represent the program state after their execution or to blocks
/// to represent the program state at the beginning of the block. A dense
/// lattice is propagated through the IR by dense data-flow analysis.
class AbstractDenseLattice : public AnalysisState {
public:
  /// A dense lattice can only be created for operations and blocks.
  using AnalysisState::AnalysisState;

  /// Join the lattice across control-flow or callgraph edges.
  virtual ChangeResult join(const AbstractDenseLattice &rhs) = 0;
};

//===----------------------------------------------------------------------===//
// AbstractDenseDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// Base class for dense data-flow analyses. Dense data-flow analysis attaches a
/// lattice between the execution of operations and implements a transfer
/// function from the lattice before each operation to the lattice after. The
/// lattice contains information about the state of the program at that point.
///
/// In this implementation, a lattice attached to an operation represents the
/// state of the program after its execution, and a lattice attached to block
/// represents the state of the program right before it starts executing its
/// body.
class AbstractDenseDataFlowAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  /// Initialize the analysis by visiting every program point whose execution
  /// may modify the program state; that is, every operation and block.
  LogicalResult initialize(Operation *top) override;

  /// Visit a program point that modifies the state of the program. If this is a
  /// block, then the state is propagated from control-flow predecessors or
  /// callsites. If this is a call operation or region control-flow operation,
  /// then the state after the execution of the operation is set by control-flow
  /// or the callgraph. Otherwise, this function invokes the operation transfer
  /// function.
  LogicalResult visit(ProgramPoint point) override;

protected:
  /// Propagate the dense lattice before the execution of an operation to the
  /// lattice after its execution.
  virtual void visitOperationImpl(Operation *op,
                                  const AbstractDenseLattice &before,
                                  AbstractDenseLattice *after) = 0;

  /// Get the dense lattice after the execution of the given program point.
  virtual AbstractDenseLattice *getLattice(ProgramPoint point) = 0;

  /// Get the dense lattice after the execution of the given program point and
  /// add it as a dependency to a program point.
  const AbstractDenseLattice *getLatticeFor(ProgramPoint dependent,
                                            ProgramPoint point);

  /// Set the dense lattice at control flow entry point and propagate an update
  /// if it changed.
  virtual void setToEntryState(AbstractDenseLattice *lattice) = 0;

  /// Join a lattice with another and propagate an update if it changed.
  void join(AbstractDenseLattice *lhs, const AbstractDenseLattice &rhs) {
    propagateIfChanged(lhs, lhs->join(rhs));
  }

private:
  /// Visit an operation. If this is a call operation or region control-flow
  /// operation, then the state after the execution of the operation is set by
  /// control-flow or the callgraph. Otherwise, this function invokes the
  /// operation transfer function.
  void visitOperation(Operation *op);

  /// Visit a block. The state at the start of the block is propagated from
  /// control-flow predecessors or callsites
  void visitBlock(Block *block);

  /// Visit a program point within a region branch operation with predecessors
  /// in it. This can either be an entry block of one of the regions of the
  /// parent operation itself.
  void visitRegionBranchOperation(ProgramPoint point,
                                  RegionBranchOpInterface branch,
                                  AbstractDenseLattice *after);
};

//===----------------------------------------------------------------------===//
// DenseDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// A dense (forward) data-flow analysis for propagating lattices before and
/// after the execution of every operation across the IR by implementing
/// transfer functions for operations.
///
/// `StateT` is expected to be a subclass of `AbstractDenseLattice`.
template <typename LatticeT>
class DenseDataFlowAnalysis : public AbstractDenseDataFlowAnalysis {
  static_assert(
      std::is_base_of<AbstractDenseLattice, LatticeT>::value,
      "analysis state class expected to subclass AbstractDenseLattice");

public:
  using AbstractDenseDataFlowAnalysis::AbstractDenseDataFlowAnalysis;

  /// Visit an operation with the dense lattice before its execution. This
  /// function is expected to set the dense lattice after its execution.
  virtual void visitOperation(Operation *op, const LatticeT &before,
                              LatticeT *after) = 0;

protected:
  /// Get the dense lattice after this program point.
  LatticeT *getLattice(ProgramPoint point) override {
    return getOrCreate<LatticeT>(point);
  }

  /// Set the dense lattice at control flow entry point and propagate an update
  /// if it changed.
  virtual void setToEntryState(LatticeT *lattice) = 0;
  void setToEntryState(AbstractDenseLattice *lattice) override {
    setToEntryState(static_cast<LatticeT *>(lattice));
  }

private:
  /// Type-erased wrappers that convert the abstract dense lattice to a derived
  /// lattice and invoke the virtual hooks operating on the derived lattice.
  void visitOperationImpl(Operation *op, const AbstractDenseLattice &before,
                          AbstractDenseLattice *after) override {
    visitOperation(op, static_cast<const LatticeT &>(before),
                   static_cast<LatticeT *>(after));
  }
};

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DENSEDATAFLOWANALYSIS_H
