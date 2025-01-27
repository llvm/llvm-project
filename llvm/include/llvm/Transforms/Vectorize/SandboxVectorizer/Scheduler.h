//===- Scheduler.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the bottom-up list scheduler used by the vectorizer. It is used for
// checking the legality of vectorization and for scheduling instructions in
// such a way that makes vectorization possible, if legal.
//
// The legality check is performed by `trySchedule(Instrs)`, which will try to
// schedule the IR until all instructions in `Instrs` can be scheduled together
// back-to-back. If this fails then it is illegal to vectorize `Instrs`.
//
// Internally the scheduler uses the vectorizer-specific DependencyGraph class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SCHEDULER_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SCHEDULER_H

#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/DependencyGraph.h"
#include <queue>

namespace llvm::sandboxir {

class PriorityCmp {
public:
  bool operator()(const DGNode *N1, const DGNode *N2) {
    // TODO: This should be a hierarchical comparator.
    return N1->getInstruction()->comesBefore(N2->getInstruction());
  }
};

/// The list holding nodes that are ready to schedule. Used by the scheduler.
class ReadyListContainer {
  PriorityCmp Cmp;
  /// Control/Other dependencies are not modeled by the DAG to save memory.
  /// These have to be modeled in the ready list for correctness.
  /// This means that the list will hold back nodes that need to meet such
  /// unmodeled dependencies.
  std::priority_queue<DGNode *, std::vector<DGNode *>, PriorityCmp> List;

public:
  ReadyListContainer() : List(Cmp) {}
  void insert(DGNode *N) { List.push(N); }
  DGNode *pop() {
    auto *Back = List.top();
    List.pop();
    return Back;
  }
  bool empty() const { return List.empty(); }
  void clear() { List = {}; }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif // NDEBUG
};

/// The nodes that need to be scheduled back-to-back in a single scheduling
/// cycle form a SchedBundle.
class SchedBundle {
public:
  using ContainerTy = SmallVector<DGNode *, 4>;

private:
  ContainerTy Nodes;

  /// Called by the DGNode destructor to avoid accessing freed memory.
  void eraseFromBundle(DGNode *N) { Nodes.erase(find(Nodes, N)); }
  friend DGNode::~DGNode(); // For eraseFromBundle().

public:
  SchedBundle() = default;
  SchedBundle(ContainerTy &&Nodes) : Nodes(std::move(Nodes)) {
    for (auto *N : this->Nodes)
      N->setSchedBundle(*this);
  }
  /// Copy CTOR (unimplemented).
  SchedBundle(const SchedBundle &Other) = delete;
  /// Copy Assignment (unimplemented).
  SchedBundle &operator=(const SchedBundle &Other) = delete;
  ~SchedBundle() {
    for (auto *N : this->Nodes)
      N->clearSchedBundle();
  }
  bool empty() const { return Nodes.empty(); }
  DGNode *back() const { return Nodes.back(); }
  using iterator = ContainerTy::iterator;
  using const_iterator = ContainerTy::const_iterator;
  iterator begin() { return Nodes.begin(); }
  iterator end() { return Nodes.end(); }
  const_iterator begin() const { return Nodes.begin(); }
  const_iterator end() const { return Nodes.end(); }
  /// \Returns the bundle node that comes before the others in program order.
  DGNode *getTop() const;
  /// \Returns the bundle node that comes after the others in program order.
  DGNode *getBot() const;
  /// Move all bundle instructions to \p Where back-to-back.
  void cluster(BasicBlock::iterator Where);
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

/// The list scheduler.
class Scheduler {
  /// This is a list-scheduler and this is the list containing the instructions
  /// that are ready, meaning that all their dependency successors have already
  /// been scheduled.
  ReadyListContainer ReadyList;
  /// The dependency graph is used by the scheduler to determine the legal
  /// ordering of instructions.
  DependencyGraph DAG;
  /// This is the top of the schedule, i.e. the location where the scheduler
  /// is about to place the scheduled instructions. It gets updated as we
  /// schedule.
  std::optional<BasicBlock::iterator> ScheduleTopItOpt;
  // TODO: This is wasting memory in exchange for fast removal using a raw ptr.
  DenseMap<SchedBundle *, std::unique_ptr<SchedBundle>> Bndls;
  /// The BB that we are currently scheduling.
  BasicBlock *ScheduledBB = nullptr;

  /// \Returns a scheduling bundle containing \p Instrs.
  SchedBundle *createBundle(ArrayRef<Instruction *> Instrs);
  void eraseBundle(SchedBundle *SB);
  /// Schedule nodes until we can schedule \p Instrs back-to-back.
  bool tryScheduleUntil(ArrayRef<Instruction *> Instrs);
  /// Schedules all nodes in \p Bndl, marks them as scheduled, updates the
  /// UnscheduledSuccs counter of all dependency predecessors, and adds any of
  /// them that become ready to the ready list.
  void scheduleAndUpdateReadyList(SchedBundle &Bndl);
  /// The scheduling state of the instructions in the bundle.
  enum class BndlSchedState {
    NoneScheduled, ///> No instruction in the bundle was previously scheduled.
    PartiallyOrDifferentlyScheduled, ///> Only some of the instrs in the bundle
                                     /// were previously scheduled, or all of
                                     /// them were but not in the same
                                     /// SchedBundle.
    FullyScheduled, ///> All instrs in the bundle were previously scheduled and
                    /// were in the same SchedBundle.
  };
  /// \Returns whether none/some/all of \p Instrs have been scheduled.
  BndlSchedState getBndlSchedState(ArrayRef<Instruction *> Instrs) const;
  /// Destroy the top-most part of the schedule that includes \p Instrs.
  void trimSchedule(ArrayRef<Instruction *> Instrs);
  /// Disable copies.
  Scheduler(const Scheduler &) = delete;
  Scheduler &operator=(const Scheduler &) = delete;

public:
  Scheduler(AAResults &AA, Context &Ctx) : DAG(AA, Ctx) {}
  ~Scheduler() {}
  /// Tries to build a schedule that includes all of \p Instrs scheduled at the
  /// same scheduling cycle. This essentially checks that there are no
  /// dependencies among \p Instrs. This function may involve scheduling
  /// intermediate instructions or canceling and re-scheduling if needed.
  /// \Returns true on success, false otherwise.
  bool trySchedule(ArrayRef<Instruction *> Instrs);
  /// Clear the scheduler's state, including the DAG.
  void clear() {
    Bndls.clear();
    // TODO: clear view once it lands.
    DAG.clear();
    ReadyList.clear();
    ScheduleTopItOpt = std::nullopt;
    ScheduledBB = nullptr;
    assert(Bndls.empty() && DAG.empty() && ReadyList.empty() &&
           !ScheduleTopItOpt && ScheduledBB == nullptr &&
           "Expected empty state!");
  }

#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SCHEDULER_H
