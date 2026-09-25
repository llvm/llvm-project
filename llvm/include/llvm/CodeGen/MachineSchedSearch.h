//===- MachineSchedSearch.h - Complete schedule search ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides common support for exploring complete instruction orders.
// A complete-schedule optimizer can either replace the normal scheduling
// strategy through a replay adapter or refine the schedule materialized by an
// existing strategy through ScheduleDAGMI's post-scheduling hook.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINESCHEDSEARCH_H
#define LLVM_CODEGEN_MACHINESCHEDSEARCH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

/// A read-only, region-local view of a fully constructed machine scheduling
/// DAG for complete-schedule exploration.
///
/// Nodes are identified by stable ordinals in [0, size()). Candidate schedules
/// are complete permutations of those ordinals. Only strong dependencies are
/// legality constraints; weak edges remain scheduling preferences.
///
/// This view is intended for search-based schedulers that evaluate multiple
/// complete schedules without mutating the MachineFunction. It is valid only
/// while the underlying SUnit storage remains alive and unchanged.
class LLVM_ABI MachineSchedSearchRegion {
public:
  struct MoveRange {
    /// First legal final position after removing and reinserting the node.
    unsigned Begin;
    /// Last legal final position after removing and reinserting the node.
    unsigned End;
  };

private:
  const ScheduleDAGMI *DAG = nullptr;
  ArrayRef<SUnit> Nodes;
  SmallVector<SmallVector<unsigned, 4>, 0> Predecessors;
  SmallVector<SmallVector<unsigned, 4>, 0> Successors;

public:
  explicit MachineSchedSearchRegion(ArrayRef<SUnit> Nodes);
  explicit MachineSchedSearchRegion(ScheduleDAGMI &DAG);

  /// Return the underlying scheduler DAG, or nullptr when the view was
  /// constructed directly from SUnit storage.
  const ScheduleDAGMI *getDAG() const { return DAG; }
  unsigned size() const { return Predecessors.size(); }
  const SUnit &getSUnit(unsigned Node) const;

  ArrayRef<unsigned> predecessors(unsigned Node) const {
    return Predecessors[Node];
  }
  ArrayRef<unsigned> successors(unsigned Node) const {
    return Successors[Node];
  }

  /// Return the order in which nodes appeared when the DAG was built.
  SmallVector<unsigned, 0> getInitialOrder() const;

  /// Return a stable topological order, preferring lower node ordinals when
  /// more than one node is ready.
  SmallVector<unsigned, 0> getTopologicalOrder() const;

  /// Return whether \p Order is a complete permutation that preserves every
  /// strong dependency in the scheduling DAG.
  bool isLegalOrder(ArrayRef<unsigned> Order) const;

  /// Return the inclusive range of final positions to which \p Node may be
  /// relocated while all other nodes retain their relative order. Returns
  /// false if the input order is not legal or the node ordinal is invalid.
  bool getLegalMoveRange(ArrayRef<unsigned> Order, unsigned Node,
                         MoveRange &Range) const;
};

/// Adapter for schedulers that compute a complete schedule before LLVM begins
/// applying scheduling decisions.
///
/// An owned MachineSchedCompleteScheduleOptimizer receives the incoming legal
/// order as its founder and may return a replacement. The selected order is
/// validated before use and replayed top-down through ScheduleDAGMI's
/// incremental MachineSchedStrategy interface. When used with
/// ScheduleDAGMILive, instruction movement, LiveIntervals, and
/// register-pressure accounting therefore remain owned by the existing
/// scheduler.
///
/// This is not a common base for all search-based schedulers. Strategies that
/// choose each node from the current ready set should implement
/// MachineSchedStrategy directly or derive from another incremental strategy.
///
/// If the optimizer declines to provide an order or returns an invalid one, the
/// existing order is preserved when legal. Otherwise, a stable topological
/// order is used.
class LLVM_ABI MachineSchedCompleteScheduleReplayer
    : public MachineSchedStrategy {
  std::unique_ptr<MachineSchedCompleteScheduleOptimizer> Optimizer;
  SmallVector<SUnit *, 0> CompleteSchedule;
  unsigned NextNodeToReplay = 0;
  bool UsedOptimizedSchedule = false;

public:
  explicit MachineSchedCompleteScheduleReplayer(
      std::unique_ptr<MachineSchedCompleteScheduleOptimizer> Optimizer);
  ~MachineSchedCompleteScheduleReplayer() override;

  void initialize(ScheduleDAGMI *DAG) override;
  SUnit *pickNode(bool &IsTopNode) override;
  void schedNode(SUnit *, bool) override {}
  void releaseTopNode(SUnit *) override {}
  void releaseBottomNode(SUnit *) override {}

  bool usedOptimizedSchedule() const { return UsedOptimizedSchedule; }
};

} // namespace llvm

#endif // LLVM_CODEGEN_MACHINESCHEDSEARCH_H
