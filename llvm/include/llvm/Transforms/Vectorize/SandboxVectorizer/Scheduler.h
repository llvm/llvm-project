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
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/DependencyGraph.h"
#include <queue>
#include <variant>

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

enum class SchedDirection {
  BottomUp,
  TopDown,
};
#ifndef NDEBUG
StringLiteral schedDirectionToStr(SchedDirection Dir);
#endif

/// The nodes that need to be scheduled back-to-back in a single scheduling
/// cycle form a SchedBundle.
class SchedBundle {
public:
  using ContainerTy = SmallVector<DGNode *, 4>;

private:
  ContainerTy Nodes;

  /// Called by the DGNode destructor to avoid accessing freed memory.
  void eraseFromBundle(DGNode *N) { llvm::erase(Nodes, N); }
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
  LLVM_ABI DGNode *getTop() const;
  /// \Returns the bundle node that comes after the others in program order.
  LLVM_ABI DGNode *getBot() const;
  /// Move all bundle instructions to \p Where back-to-back.
  LLVM_ABI void cluster(BasicBlock::iterator Where);
  /// \Returns true if all nodes in the bundle are ready.
  bool ready(SchedDirection Dir) const {
    return all_of(Nodes, [Dir](const auto *N) {
      return Dir == SchedDirection::BottomUp ? N->readyBottomUp()
                                             : N->readyTopDown();
    });
  }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

/// The scheduling point in the context of the Scheduler points to the
/// top-of-schedule (i.e., the top-most instruction of the top bundle) during
/// bottom-up scheduling or the bottom of the schedule (i.e., the bottom-most
/// instruction of the bottom bundle) during top-down.
///
/// This class can be thought of as an extended BB::iterator, one that can
/// not only point to after the last instruction in a BB (i.e., BB.end()), but
/// also before the first instruction (i.e., something equivalent to
/// prev(BB.begin()), which is not a legal BasicBlock::iterator).
///
/// This is needed for symmetric implementations of top-down and bottom-up
/// scheduling. More specifically, if this is the first scheduling attempt we
/// need the scheduling front to still point to a hypothetical last scheduling
/// point. In bottom-up this can be at BB.end() but in top-down this can be
/// before BB.begin(). This is why a BasicBlock::iterator is not suitable for
/// this.
class SchedulingPoint {
  /// If Where contains a Block, then we are pointing before BB.begin(),
  /// otherwise if it contains an iterator then we point to anywhere in the BB
  /// or at BB.end().
  std::variant<BasicBlock::iterator, BasicBlock *> Where;

  /// Creates a scheduling point pointing before the beginning of BB.
  SchedulingPoint(BasicBlock &BB) : Where(&BB) {}

public:
  /// Creates a scheduling point pointing at \p It, meaning any instruction in a
  /// BB or BB.end().
  SchedulingPoint(BasicBlock::iterator It) : Where(It) {}
  /// Returns a SchedulingPoint that points to \p It.
  static SchedulingPoint createAt(BasicBlock::iterator It) {
    return SchedulingPoint(It);
  }
  /// Returns a SchedulingPoint that points to one element before \p It.
  static SchedulingPoint createBefore(BasicBlock::iterator It) {
    BasicBlock &BB = *It.getNodeParent();
    if (It == BB.begin())
      return SchedulingPoint(BB);
    return SchedulingPoint(std::prev(It));
  }
  /// Returns a SchedulingPoint that points to one element after \p It.
  static SchedulingPoint createAfter(BasicBlock::iterator It) {
    assert(It != It.getNodeParent()->end() && "Already at end!");
    return SchedulingPoint(std::next(It));
  }

  /// If the SchedulingPoint points to before the beginning of a BB, then this
  /// returns that BB, else returns nullptr.
  BasicBlock *atBeforeBeginOrNull() const {
    if (std::holds_alternative<BasicBlock::iterator>(Where))
      return nullptr;
    return std::get<BasicBlock *>(Where);
  }
  /// If the SchedulingPoint points after the last instruction in the BB then
  /// this returns the corresponding BasicBlock, nullptr otherwise.
  BasicBlock *atEndOrNull() const {
    if (std::holds_alternative<BasicBlock *>(Where))
      return nullptr;
    auto It = std::get<BasicBlock::iterator>(Where);
    return It == It.getNodeParent()->end() ? It.getNodeParent() : nullptr;
  }
  /// Returns the instruction pointed to by this SchedulingPoint or null if we
  /// are before/after BB.
  Instruction *atInstrOrNull() const {
    if (atBeforeBeginOrNull() || atEndOrNull())
      return nullptr;
    return &*std::get<BasicBlock::iterator>(Where);
  }
  /// Cast to Instruction *. Asserts that we are pointing to an instruction and
  /// not before/after the beginning/end of a BB.
  operator Instruction *() const { return atInstrOrNull(); }
  /// Returns the corresponding BB::iterator. Asserts that we are not pointing
  /// before BB begin.
  BasicBlock::iterator getIterator() const {
    assert(!atBeforeBeginOrNull() && "Expected in/after BB!");
    return std::get<BasicBlock::iterator>(Where);
  }
  operator BasicBlock::iterator() const { return getIterator(); }
  /// Returns the SchedulingPoint pointing after this.
  SchedulingPoint getNext() const {
    assert(!atEndOrNull() && "Expected before/in BB!");
    if (BasicBlock *BB = atBeforeBeginOrNull())
      return BB->begin();
    return std::next(getIterator());
  }
  /// Returns the SchedulingPoint pointing before this.
  SchedulingPoint getPrev() const {
    assert(!atBeforeBeginOrNull() && "Expected in/after BB!");
    auto It = getIterator();
    auto *BB = It.getNodeParent();
    if (It == BB->begin())
      return *BB;
    return std::prev(It);
  }
  bool operator==(const SchedulingPoint &Other) const {
    return Where == Other.Where;
  }
#ifndef NDEBUG
  void print(raw_ostream &OS) const;
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
  /// This is the top of the schedule during bottom-up scheduling and the bottom
  /// of the schedule during top-down. It points to the position of the last
  /// top-most/bottom-most instruction scheduled. It may get updated after every
  /// trySchedule() attempt, regardless of whether scheduling succeeded or not.
  /// It is nullopt if we have not scheduled before.
  std::optional<SchedulingPoint> ScheduleTopItOpt;
  // TODO: This is wasting memory in exchange for fast removal using a raw ptr.
  DenseMap<SchedBundle *, std::unique_ptr<SchedBundle>> Bndls;
  /// The BB that we are currently scheduling.
  BasicBlock *ScheduledBB = nullptr;
  /// The ID of the callback we register with Sandbox IR.
  std::optional<Context::CallbackID> CreateInstrCB;
  /// Called by Sandbox IR's callback system, after \p I has been created.
  /// NOTE: This should run after DAG's callback has run.
  // TODO: Perhaps call DAG's notify function from within this one?
  LLVM_ABI void notifyCreateInstr(Instruction *I);

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
  LLVM_ABI BndlSchedState
  getBndlSchedState(ArrayRef<Instruction *> Instrs) const;
  /// Destroy the top-most part of the schedule that includes \p Instrs.
  void trimSchedule(ArrayRef<Instruction *> Instrs);
  /// Disable copies.
  Scheduler(const Scheduler &) = delete;
  Scheduler &operator=(const Scheduler &) = delete;

private:
  SchedDirection Dir = SchedDirection::BottomUp;

public:
  Scheduler(AAResults &AA, Context &Ctx, SchedDirection Dir)
      : DAG(AA, Ctx), Ctx(Ctx), Dir(Dir) {
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
  LLVM_ABI bool trySchedule(ArrayRef<Instruction *> Instrs);
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
