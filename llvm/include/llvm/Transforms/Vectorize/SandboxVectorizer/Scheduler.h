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
    // Given that the DAG does not model dependencies such that PHIs are always
    // at the top, or terminators always at the bottom, we need to force the
    // priority here in the comparator of the ready list container.
    auto *I1 = N1->getInstruction();
    auto *I2 = N2->getInstruction();
    bool IsTerm1 = I1->isTerminator();
    bool IsTerm2 = I2->isTerminator();
    if (IsTerm1 != IsTerm2)
      // Terminators have the lowest priority.
      return IsTerm1 > IsTerm2;
    bool IsPHI1 = isa<PHINode>(I1);
    bool IsPHI2 = isa<PHINode>(I2);
    if (IsPHI1 != IsPHI2)
      // PHIs have the highest priority.
      return IsPHI1 < IsPHI2;
    // Otherwise rely on the instruction order.
    return I2->comesBefore(I1);
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
  void insert(DGNode *N) {
#ifndef NDEBUG
    assert(!N->scheduled() && "Don't insert a scheduled node!");
    auto ListCopy = List;
    while (!ListCopy.empty()) {
      DGNode *Top = ListCopy.top();
      ListCopy.pop();
      assert(Top != N && "Node already exists in ready list!");
    }
#endif
    List.push(N);
  }
  DGNode *pop() {
    auto *Back = List.top();
    List.pop();
    return Back;
  }
  bool empty() const { return List.empty(); }
  void clear() { List = {}; }
  /// \Removes \p N if found in the ready list.
  void remove(DGNode *N) {
    // TODO: Use a more efficient data-structure for the ready list because the
    // priority queue does not support fast removals.
    SmallVector<DGNode *, 8> Keep;
    Keep.reserve(List.size());
    while (!List.empty()) {
      auto *Top = List.top();
      List.pop();
      if (Top == N)
        break;
      Keep.push_back(Top);
    }
    for (auto *KeepN : Keep)
      List.push(KeepN);
  }
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
  void eraseFromBundle(DGNode *N) {
    Nodes.erase(std::remove(Nodes.begin(), Nodes.end(), N), Nodes.end());
  }
  friend void DGNode::setSchedBundle(SchedBundle &); // For eraseFromBunde().
  friend DGNode::~DGNode();                          // For eraseFromBundle().

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
  /// Singleton bundles are created when scheduling instructions temporarily to
  /// fill in the schedule until we schedule the vector bundle. These are
  /// non-vector bundles containing just a single instruction.
  bool isSingleton() const { return Nodes.size() == 1u; }
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
  /// \Returns true if all nodes in the bundle are ready.
  bool ready() const {
    return all_of(Nodes, [](const auto *N) { return N->ready(); });
  }
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
  friend class SchedulerInternalsAttorney; // For DAG.
  Context &Ctx;
  /// This is the top of the schedule, i.e. the location where the scheduler
  /// is about to place the scheduled instructions. It gets updated as we
  /// schedule.
  std::optional<BasicBlock::iterator> ScheduleTopItOpt;
  // TODO: This is wasting memory in exchange for fast removal using a raw ptr.
  DenseMap<SchedBundle *, std::unique_ptr<SchedBundle>> Bndls;
  /// The BB that we are currently scheduling.
  BasicBlock *ScheduledBB = nullptr;
  /// The ID of the callback we register with Sandbox IR.
  std::optional<Context::CallbackID> CreateInstrCB;
  /// Called by Sandbox IR's callback system, after \p I has been created.
  /// NOTE: This should run after DAG's callback has run.
  // TODO: Perhaps call DAG's notify function from within this one?
  void notifyCreateInstr(Instruction *I);

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
    AlreadyScheduled, ///> At least one instruction in the bundle belongs to a
                      /// different non-singleton scheduling bundle.
    TemporarilyScheduled, ///> Instructions were temporarily scheduled as
                          /// singleton bundles or some of them were not
                          /// scheduled at all. None of them were in a vector
                          ///(non-singleton) bundle.
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
  Scheduler(AAResults &AA, Context &Ctx) : DAG(AA, Ctx), Ctx(Ctx) {
    // NOTE: The scheduler's callback depends on the DAG's callback running
    // before it and updating the DAG accordingly.
    CreateInstrCB = Ctx.registerCreateInstrCallback(
        [this](Instruction *I) { notifyCreateInstr(I); });
  }
  ~Scheduler() {
    if (CreateInstrCB)
      Ctx.unregisterCreateInstrCallback(*CreateInstrCB);
  }
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

/// A client-attorney class for accessing the Scheduler's internals (used for
/// unit tests).
class SchedulerInternalsAttorney {
public:
  static DependencyGraph &getDAG(Scheduler &Sched) { return Sched.DAG; }
  using BndlSchedState = Scheduler::BndlSchedState;
  static BndlSchedState getBndlSchedState(const Scheduler &Sched,
                                          ArrayRef<Instruction *> Instrs) {
    return Sched.getBndlSchedState(Instrs);
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SCHEDULER_H
