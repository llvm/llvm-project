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

public:
  SchedBundle() = default;
  SchedBundle(ContainerTy &&Nodes) : Nodes(std::move(Nodes)) {}
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
  ReadyListContainer ReadyList;
  DependencyGraph DAG;
  std::optional<BasicBlock::iterator> ScheduleTopItOpt;
  SmallVector<std::unique_ptr<SchedBundle>> Bndls;

  /// \Returns a scheduling bundle containing \p Instrs.
  SchedBundle *createBundle(ArrayRef<Instruction *> Instrs);
  /// Schedule nodes until we can schedule \p Instrs back-to-back.
  bool tryScheduleUntil(ArrayRef<Instruction *> Instrs);
  /// Schedules all nodes in \p Bndl, marks them as scheduled, updates the
  /// UnscheduledSuccs counter of all dependency predecessors, and adds any of
  /// them that become ready to the ready list.
  void scheduleAndUpdateReadyList(SchedBundle &Bndl);

  /// Disable copies.
  Scheduler(const Scheduler &) = delete;
  Scheduler &operator=(const Scheduler &) = delete;

public:
  Scheduler(AAResults &AA) : DAG(AA) {}
  ~Scheduler() {}

  bool trySchedule(ArrayRef<Instruction *> Instrs);

#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SCHEDULER_H
