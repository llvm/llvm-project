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
#include "mlir/IR/SymbolTable.h"

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
  virtual ChangeResult join(const AbstractDenseLattice &rhs) {
    return ChangeResult::NoChange;
  }

  virtual ChangeResult meet(const AbstractDenseLattice &rhs) {
    return ChangeResult::NoChange;
  }
};

//===----------------------------------------------------------------------===//
// AbstractDenseDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// Base class for dense (forward) data-flow analyses. Dense data-flow analysis
/// attaches a lattice between the execution of operations and implements a
/// transfer function from the lattice before each operation to the lattice
/// after. The lattice contains information about the state of the program at
/// that point.
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
  /// add it as a dependency to a program point. That is, every time the lattice
  /// after point is updated, the dependent program point must be visited, and
  /// the newly triggered visit might update the lattice after dependent.
  const AbstractDenseLattice *getLatticeFor(ProgramPoint dependent,
                                            ProgramPoint point);

  /// Set the dense lattice at control flow entry point and propagate an update
  /// if it changed.
  virtual void setToEntryState(AbstractDenseLattice *lattice) = 0;

  /// Join a lattice with another and propagate an update if it changed.
  void join(AbstractDenseLattice *lhs, const AbstractDenseLattice &rhs) {
    propagateIfChanged(lhs, lhs->join(rhs));
  }

  /// Visit an operation. If this is a call operation or region control-flow
  /// operation, then the state after the execution of the operation is set by
  /// control-flow or the callgraph. Otherwise, this function invokes the
  /// operation transfer function.
  virtual void processOperation(Operation *op);

  /// Visit a program point within a region branch operation with predecessors
  /// in it. This can either be an entry block of one of the regions of the
  /// parent operation itself.
  void visitRegionBranchOperation(ProgramPoint point,
                                  RegionBranchOpInterface branch,
                                  AbstractDenseLattice *after);

private:
  /// Visit a block. The state at the start of the block is propagated from
  /// control-flow predecessors or callsites.
  void visitBlock(Block *block);
};

//===----------------------------------------------------------------------===//
// DenseDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// A dense (forward) data-flow analysis for propagating lattices before and
/// after the execution of every operation across the IR by implementing
/// transfer functions for operations.
///
/// `LatticeT` is expected to be a subclass of `AbstractDenseLattice`.
template <typename LatticeT>
class DenseDataFlowAnalysis : public AbstractDenseDataFlowAnalysis {
  static_assert(
      std::is_base_of<AbstractDenseLattice, LatticeT>::value,
      "analysis state class expected to subclass AbstractDenseLattice");

public:
  using AbstractDenseDataFlowAnalysis::AbstractDenseDataFlowAnalysis;

  /// Visit an operation with the dense lattice before its execution. This
  /// function is expected to set the dense lattice after its execution and
  /// trigger change propagation in case of change.
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

  /// Type-erased wrappers that convert the abstract dense lattice to a derived
  /// lattice and invoke the virtual hooks operating on the derived lattice.
  void visitOperationImpl(Operation *op, const AbstractDenseLattice &before,
                          AbstractDenseLattice *after) override {
    visitOperation(op, static_cast<const LatticeT &>(before),
                   static_cast<LatticeT *>(after));
  }
};

//===----------------------------------------------------------------------===//
// AbstractDenseBackwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// Base class for dense backward dataflow analyses. Such analyses attach a
/// lattice between the execution of operations and implement a transfer
/// function from the lattice after the operation ot the lattice before it, thus
/// propagating backward.
///
/// In this implementation, a lattice attached to an operation represents the
/// state of the program before its execution, and a lattice attached to a block
/// represents the state of the program before the end of the block, i.e., after
/// its terminator.
class AbstractDenseBackwardDataFlowAnalysis : public DataFlowAnalysis {
public:
  /// Construct the analysis in the given solver. Takes a symbol table
  /// collection that is used to cache symbol resolution in interprocedural part
  /// of the analysis. The symbol table need not be prefilled.
  AbstractDenseBackwardDataFlowAnalysis(DataFlowSolver &solver,
                                        SymbolTableCollection &symbolTable)
      : DataFlowAnalysis(solver), symbolTable(symbolTable) {}

  /// Initialize the analysis by visiting every program point whose execution
  /// may modify the program state; that is, every operation and block.
  LogicalResult initialize(Operation *top) override;

  /// Visit a program point that modifies the state of the program. The state is
  /// propagated along control flow directions for branch-, region- and
  /// call-based control flow using the respective interfaces. For other
  /// operations, the state is propagated using the transfer function
  /// (visitOperation).
  ///
  /// Note: the transfer function is currently *not* invoked for operations with
  /// region or call interface, but *is* invoked for block terminators.
  LogicalResult visit(ProgramPoint point) override;

protected:
  /// Propagate the dense lattice after the execution of an operation to the
  /// lattice before its execution.
  virtual void visitOperationImpl(Operation *op,
                                  const AbstractDenseLattice &after,
                                  AbstractDenseLattice *before) = 0;

  /// Get the dense lattice before the execution of the program point. That is,
  /// before the execution of the given operation or after the execution of the
  /// block.
  virtual AbstractDenseLattice *getLattice(ProgramPoint point) = 0;

  /// Get the dense lattice before the execution of the program point `point`
  /// and declare that the `dependent` program point must be updated every time
  /// `point` is.
  const AbstractDenseLattice *getLatticeFor(ProgramPoint dependent,
                                            ProgramPoint point);

  /// Set the dense lattice before at the control flow exit point and propagate
  /// the update if it changed.
  virtual void setToExitState(AbstractDenseLattice *lattice) = 0;

  /// Meet a lattice with another lattice and propagate an update if it changed.
  void meet(AbstractDenseLattice *lhs, const AbstractDenseLattice &rhs) {
    propagateIfChanged(lhs, lhs->meet(rhs));
  }

  /// Visit an operation. If this is a call operation or region control-flow
  /// operation, then the state after the execution of the operation is set by
  /// control-flow or the callgraph. Otherwise, this function invokes the
  /// operation transfer function.
  virtual void processOperation(Operation *op);

  /// Visit a program point within a region branch operation with successors
  /// (from which the state is propagated) in or after it. `regionNo` indicates
  /// the region that contains the successor, `nullopt` indicating the successor
  /// of the branch operation itself.
  void visitRegionBranchOperation(ProgramPoint point,
                                  RegionBranchOpInterface branch,
                                  std::optional<unsigned> regionNo,
                                  AbstractDenseLattice *before);

private:
  /// VIsit a block. The state and the end of the block is propagated from
  /// control-flow successors of the block or callsites.
  void visitBlock(Block *block);

  /// Symbol table for call-level control flow.
  SymbolTableCollection &symbolTable;
};

//===----------------------------------------------------------------------===//
// DenseBackwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// A dense backward dataflow analysis propagating lattices after and before the
/// execution of every operation across the IR by implementing transfer
/// functions for opreations.
///
/// `LatticeT` is expected to be a subclass of `AbstractDenseLattice`.
template <typename LatticeT>
class DenseBackwardDataFlowAnalysis
    : public AbstractDenseBackwardDataFlowAnalysis {
  static_assert(std::is_base_of_v<AbstractDenseLattice, LatticeT>,
                "analysis state expected to subclass AbstractDenseLattice");

public:
  using AbstractDenseBackwardDataFlowAnalysis::
      AbstractDenseBackwardDataFlowAnalysis;

  /// Transfer function. Visits an operation with the dense lattice after its
  /// execution. This function is expected to set the dense lattice before its
  /// execution and trigger propagation in case of change.
  virtual void visitOperation(Operation *op, const LatticeT &after,
                              LatticeT *before) = 0;

protected:
  /// Get the dense lattice at the given program point.
  LatticeT *getLattice(ProgramPoint point) override {
    return getOrCreate<LatticeT>(point);
  }

  /// Set the dense lattice at control flow exit point (after the terminator)
  /// and propagate an update if it changed.
  virtual void setToExitState(LatticeT *lattice) = 0;
  void setToExitState(AbstractDenseLattice *lattice) override {
    setToExitState(static_cast<LatticeT *>(lattice));
  }

  /// Type-erased wrapper that convert the abstract dense lattice to a derived
  /// lattice and invoke the virtual hooks operating on the derived lattice.
  void visitOperationImpl(Operation *op, const AbstractDenseLattice &after,
                          AbstractDenseLattice *before) override {
    visitOperation(op, static_cast<const LatticeT &>(after),
                   static_cast<LatticeT *>(before));
  }
};

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DENSEDATAFLOWANALYSIS_H
