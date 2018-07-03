//===- llvm/Analysis/TapirTaskInfo.h - Tapir Task Calculator ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TapirTaskInfo class that is used to identify parallel
// tasks as represented in Tapir.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TAPIRTASKINFO_H
#define LLVM_ANALYSIS_TAPIRTASKINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Allocator.h"
#include <algorithm>
#include <utility>

namespace llvm {

class PHINode;
class raw_ostream;
class Spindle;
class Task;
class TaskInfo;

//===----------------------------------------------------------------------===//
/// In Tapir, the basic blocks in a function can be partitioned into
/// spindles.  A spindle is a connected set of basic blocks with a
/// single entry point for parallel control flow.  When executed, all
/// blocks within a spindle are guaranteed to execute sequentially on
/// one worker.
///
class Spindle {
public:
  enum SPType { Entry, Detach, Sync, Phi };

private:
  SPType Ty;

  Task *ParentTask;

  // The list of basic blocks in this spindle.  The first entry is the entry
  // block of the spindle.
  std::vector<BasicBlock *> Blocks;

  SmallPtrSet<const BasicBlock *, 8> DenseBlockSet;

  // Predecessor and successor spindles.
  using SpindleEdge = std::pair<Spindle *, BasicBlock *>;
  SmallVector<SpindleEdge, 8> Predecessors;
  SmallVector<SpindleEdge, 8> Successors;
  
  Spindle(const Spindle &) = delete;
  const Spindle &operator=(const Spindle &) = delete;

public:
  BasicBlock *getEntry() const { return getBlocks().front(); }
  bool isEntry(const BasicBlock *B) const { return (getBlocks().front() == B); }
  Task *getParentTask() const { return ParentTask; }

  void setParentTask(Task *T) { ParentTask = T; }

  SPType getType() const { return Ty; }
  bool isSync() const { return Sync == Ty; }
  bool isPhi() const { return Phi == Ty; }

  /// Return true if the specified basic block is in this task.
  bool contains(const BasicBlock *BB) const {
    return DenseBlockSet.count(BB);
  }

  /// Return true if the specified instruction is in this task.
  bool contains(const Instruction *Inst) const {
    return contains(Inst->getParent());
  }

  /// Get a list of the basic blocks which make up this task.
  ArrayRef<BasicBlock *> getBlocks() const {
    return Blocks;
  }
  using iterator = ArrayRef<BasicBlock *>::const_iterator;
  iterator block_begin() const { return getBlocks().begin(); }
  iterator block_end() const { return getBlocks().end(); }
  inline iterator_range<iterator> blocks() const {
    return make_range(block_begin(), block_end());
  }

  /// Get the number of blocks in this task in constant time.
  unsigned getNumBlocks() const {
    return Blocks.size();
  }

  /// Return a direct, mutable handle to the blocks vector so that we can
  /// mutate it efficiently with techniques like `std::remove`.
  std::vector<BasicBlock *> &getBlocksVector() {
    return Blocks;
  }
  /// Return a direct, mutable handle to the blocks set so that we can
  /// mutate it efficiently.
  SmallPtrSetImpl<const BasicBlock *> &getBlocksSet() {
    return DenseBlockSet;
  }

  /// True if terminator in the block can branch to another block that is
  /// outside of this spindle.
  bool isSpindleExiting(const BasicBlock *BB) const {
    if (BB->getTerminator()->getNumSuccessors() == 0)
      return true;
    for (const auto &Succ : children<const BasicBlock *>(BB))
      if (!contains(Succ))
        return true;
    return false;
  }

  /// Iterator to walk just the exiting basic blocks of the spindle.
  template<typename SpindleT = Spindle *>
  class exitblock_iterator_impl
    : public iterator_facade_base<exitblock_iterator_impl<SpindleT>,
                                  std::forward_iterator_tag, BasicBlock> {
    SpindleT S;
    unsigned i;

    void findNextExit() {
      while (!atEnd() && !S->isSpindleExiting(S->getBlocks()[i])) ++i;
    }

  public:
    explicit exitblock_iterator_impl(SpindleT S, unsigned i = 0) : S(S), i(i) {
      findNextExit();
    }

    // Allow default construction to build variables, but this doesn't build
    // a useful iterator.
    exitblock_iterator_impl() = default;

    // Allow conversion between instantiations where valid.
    exitblock_iterator_impl(const exitblock_iterator_impl<SpindleT> &Arg)
        : S(Arg.S), i(Arg.i) {}

    static exitblock_iterator_impl end(SpindleT S) {
      return exitblock_iterator_impl(S, S->getNumBlocks());
    }

    bool operator==(const exitblock_iterator_impl &Arg) const {
      return (S == Arg.S) && (i == Arg.i);
    }
    bool operator!=(const exitblock_iterator_impl &Arg) const {
      return !operator==(Arg);
    }
    bool atEnd() const { return i >= S->getNumBlocks(); }

    BasicBlock *operator->() const { return &operator*(); }
    BasicBlock &operator*() const {
      assert(!atEnd() && "exitblock_iterator out of range");
      return *S->getBlocks()[i];
    }

    // using exitblock_iterator_impl::iterator_facade_base::operator++;
    inline exitblock_iterator_impl<SpindleT> &operator++() {
      assert(!atEnd() && "Cannot increment the end iterator!");
      ++i;
      findNextExit();
      return *this;
    }
    inline exitblock_iterator_impl<SpindleT> operator++(int) { // Postincrement
      exitblock_iterator_impl tmp = *this;
      ++*this;
      return tmp;
    }
  };
  using exitblock_iterator = exitblock_iterator_impl<>;
  using const_exitblock_iterator = exitblock_iterator_impl<const Spindle *>;

  /// Returns a range that iterates over the exit blocks in the spindle.
  iterator_range<const_exitblock_iterator> exits() const {
    return make_range<const_exitblock_iterator>(
        const_exitblock_iterator(this), const_exitblock_iterator::end(this));
  }
  iterator_range<exitblock_iterator> exits() {
    return make_range<exitblock_iterator>(
        exitblock_iterator(this), exitblock_iterator::end(this));
  }

  using spedge_iterator = SmallVectorImpl<SpindleEdge>::iterator;
  using spedge_const_iterator = SmallVectorImpl<SpindleEdge>::const_iterator;
  using spedge_range = iterator_range<spedge_iterator>;
  using spedge_const_range = iterator_range<spedge_const_iterator>;

  inline spedge_iterator pred_begin() { return Predecessors.begin(); }
  inline spedge_const_iterator pred_begin() const {
    return Predecessors.begin();
  }
  inline spedge_iterator pred_end() { return Predecessors.end(); }
  inline spedge_const_iterator pred_end() const {
    return Predecessors.end();
  }

  inline spedge_iterator succ_begin() { return Successors.begin(); }
  inline spedge_const_iterator succ_begin() const {
    return Successors.begin();
  }
  inline spedge_iterator succ_end() { return Successors.end(); }
  inline spedge_const_iterator succ_end() const {
    return Successors.end();
  }

  /// Print spindle with all the BBs inside it.
  void print(raw_ostream &OS, bool TaskExit = false,
             bool Verbose = false) const;

  /// Raw method to add block B to this spindle.
  void addBlock(BasicBlock &B) {
    Blocks.push_back(&B);
    DenseBlockSet.insert(&B);
  }

  bool blockPrecedesSpindle(BasicBlock *B) {
    for (BasicBlock *SB : successors(B))
      if (SB == getEntry())
        return true;
    return false;
  }

  // Raw method to add spindle S as a predecessor of this spindle.
  void addSpindleEdgeTo(Spindle *Succ, BasicBlock *FromExit) {
    assert(contains(FromExit) &&
           "Cannot add spindle edge from block not in this spindle");
    assert(Succ->blockPrecedesSpindle(FromExit) &&
           "FromExit must precede successor spindle");
    Successors.push_back(SpindleEdge(Succ, FromExit));
    Succ->Predecessors.push_back(SpindleEdge(this, FromExit));
  }

protected:
  friend class Task;
  friend class TaskInfo;

  /// This creates an empty spindle.
  Spindle() : ParentTask(nullptr) {}

  explicit Spindle(BasicBlock *BB, SPType Ty)
      : Ty(Ty), ParentTask(nullptr) {
    Blocks.push_back(BB);
    DenseBlockSet.insert(BB);
  }

  // To allow passes like SCEV to key analysis results off of `Task` pointers,
  // we disallow re-use of pointers within a task pass manager.  This means task
  // passes should not be `delete` ing `Task` objects directly (and risk a later
  // `Task` allocation re-using the address of a previous one) but should be
  // using TaskInfo::markAsRemoved, which keeps around the `Task` pointer till
  // the end of the lifetime of the `TaskInfo` object.
  //
  // To make it easier to follow this rule, we mark the destructor as
  // non-public.
  ~Spindle() {
    Blocks.clear();
    DenseBlockSet.clear();
    ParentTask = nullptr;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const Spindle &S);

// // Provide specializations of GraphTraits to be able to treat a function
// // as a graph of spindles.

// template <> struct GraphTraits<const Spindle *> {
//   using NodeRef = const Spindle *;
//   using ChildIteratorType = const_iterator;

//   static NodeRef getEntryNode(const Spindle *S) { return S; }
//   static ChildIteratorType child_begin(NodeRef N) { return N->succ_begin(); }
//   static ChildIteratorType child_end(NodeRef N) { return N->succ_end(); }
// };

// template <> struct GraphTraits<TaskInfo *> {
//   using NodeRef = Spindle *;
//   using ChildIteratorType = succ_spindle_iterator;

//   static NodeRef getEntryNode(Spindle *S) { return S; }
//   static ChildIteratorType child_begin(NodeRef N) { return succ_begin(N); }
//   static ChildIteratorType child_end(NodeRef N) { return succ_end(N); }
// };

// template <> struct GraphTraits<Inverse<const Spindle *>> {
//   using NodeRef = const Spindle *;
//   using ChildIteratorType = const_pred_spindle_iterator;

//   static NodeRef getEntryNode(Inverse<const Spindle *> G) { return G.Graph; }
//   static ChildIteratorType child_begin(NodeRef N) { return pred_begin(N); }
//   static ChildIteratorType child_end(NodeRef N) { return pred_end(N); }
// };

// template <> struct GraphTraits<Inverse<Spindle *>> {
//   using NodeRef = Spindle *;
//   using ChildIteratorType = pred_spindle_iterator;

//   static NodeRef getEntryNode(Inverse<TaskInfo *> G) { return G.Graph; }
//   static ChildIteratorType child_begin(NodeRef N) { return pred_begin(N); }
//   static ChildIteratorType child_end(NodeRef N) { return pred_end(N); }
// };

//===----------------------------------------------------------------------===//
/// Instances of this class are used to represent Tapir tasks that are detected
/// in the flow graph.
///
class Task {
  Task *ParentTask;
  // Dominator tree
  DominatorTree &DomTree;
  // Tasks contained entirely within this one.
  std::vector<Task *> SubTasks;

  // List of spindles that make up this task.
  std::vector<Spindle *> Spindles;
  SmallPtrSet<const Spindle *, 8> DenseSpindleSet;

  // List of shared exception-handling spindles associated with this task.
  SmallVector<Spindle *, 1> SharedSubTaskEH;
  SmallPtrSet<const Spindle *, 8> DenseEHSpindleSet;

  Task(const Task &) = delete;
  const Task &operator=(const Task &) = delete;

public:
  /// Return the nesting level of this task.  An outer-most task has depth 1,
  /// for consistency with task depth values used for basic blocks, where depth
  /// 0 is used for blocks not inside any tasks.
  unsigned getTaskDepth() const {
    unsigned D = 0;
    for (const Task *CurTask = ParentTask; CurTask;
         CurTask = CurTask->ParentTask)
      ++D;
    return D;
  }
  Spindle *getEntrySpindle() const {
    return getSpindles().front();
  }
  BasicBlock *getEntry() const {
    return getEntrySpindle()->getEntry();
  }
  Task *getParentTask() const { return ParentTask; }
  void setParentTask(Task *T) { ParentTask = T; }

  /// Return true if this task is "serial," meaning it does not itself perform a
  /// detach.  This method does not preclude functions called by this task from
  /// performing a detach.
  bool isSerial() const { return SubTasks.empty(); }

  /// Return true if the specified spindle is in this task.
  bool contains(const Spindle *S) const {
    return DenseSpindleSet.count(S);
  }

  /// Return true if the specified spindle is a shared EH spindle dominated by
  /// this task.
  bool containsSharedEH(const Spindle *S) const {
    return DenseEHSpindleSet.count(S);
  }

  /// Return the tasks contained entirely within this task.
  const std::vector<Task *> &getSubTasks() const {
    return SubTasks;
  }
  std::vector<Task *> &getSubTasksVector() {
    return SubTasks;
  }
  using iterator = typename std::vector<Task *>::const_iterator;
  using reverse_iterator =
                       typename std::vector<Task *>::const_reverse_iterator;
  iterator begin() const { return getSubTasks().begin(); }
  iterator end() const { return getSubTasks().end(); }
  reverse_iterator rbegin() const { return getSubTasks().rbegin(); }
  reverse_iterator rend() const { return getSubTasks().rend(); }
  bool empty() const { return getSubTasks().empty(); }

  /// Return the spindles contained entirely within this task.
  const std::vector<Spindle *> &getSpindles() const {
    return Spindles;
  }
  std::vector<Spindle *> &getSpindlesVector() {
    return Spindles;
  }
  typedef typename std::vector<Spindle *>::const_iterator spindle_iterator;
  spindle_iterator spindle_begin() const { return getSpindles().begin(); }
  spindle_iterator spindle_end() const { return getSpindles().end(); }
  inline iterator_range<spindle_iterator> spindles() const {
    return make_range(spindle_begin(), spindle_end());
  }

  /// Get the number of spindles in this task in constant time.
  unsigned getNumSpindles() const {
    return Spindles.size();
  }
  /// Get the number of shared EH spindles in this task in constant time.
  unsigned getNumSharedEHSpindles() const {
    return SharedSubTaskEH.size();
  }

  /// Get a list of the basic blocks which make up this task.
  void getBlocks(SmallVectorImpl<BasicBlock *> &Blocks) const {
    DomTree.getDescendants(getEntry(), Blocks);
  }

  /// Return true if specified task contains the specified basic block.
  bool contains(const BasicBlock *BB) const {
    // dbgs() << "Checking if " << BB->getName() <<
    //   " is in task at depth " << getTaskDepth() << "\n";
    return DomTree.dominates(getEntry(), BB);
  }

  /// True if terminator in the block can branch to another block that is
  /// outside of the current task.
  bool isTaskExiting(const BasicBlock *BB) const {
    if (BB->getTerminator()->getNumSuccessors() == 0)
      return true;
    for (const auto &Succ : children<const BasicBlock *>(BB)) {
      if (!contains(Succ))
        return true;
    }
    return false;
  }

  /// True if the spindle can exit to a block that is outside of the current
  /// task.
  bool isTaskExiting(const Spindle *S) const {
    for (const auto &Exit : S->exits()) {
      if (isTaskExiting(&Exit))
        return true;
    }
    return false;
  }

  /// Verify task structure
  void verify(const TaskInfo *TI, const BasicBlock *Entry,
              const DominatorTree &DT) const;

  /// Print task with all the BBs inside it.
  void print(raw_ostream &OS, unsigned Depth = 0, bool Verbose = false) const;

  void dump() const;
  void dumpVerbose() const;

  /// Raw method to add spindle S to this task.
  void addSpindle(Spindle &S) {
    Spindles.push_back(&S);
    DenseSpindleSet.insert(&S);
  }

  /// Raw method to add a shared exception-handling spindle S to this task.
  void addEHSpindle(Spindle &S) {
    SharedSubTaskEH.push_back(&S);
    DenseEHSpindleSet.insert(&S);
  }

protected:
  friend class TaskInfo;

  explicit Task(Spindle &Entry, DominatorTree &DomTree)
      : ParentTask(nullptr), DomTree(DomTree) {
    Spindles.push_back(&Entry);
    DenseSpindleSet.insert(&Entry);
  }

  // To allow passes like SCEV to key analysis results off of `Task` pointers,
  // we disallow re-use of pointers within a task pass manager.  This means task
  // passes should not be `delete` ing `Task` objects directly (and risk a later
  // `Task` allocation re-using the address of a previous one) but should be
  // using TaskInfo::markAsRemoved, which keeps around the `Task` pointer till
  // the end of the lifetime of the `TaskInfo` object.
  //
  // To make it easier to follow this rule, we mark the destructor as
  // non-public.
  ~Task() {
    for (auto *SubTask : SubTasks)
      SubTask->~Task();

    for (auto *Spindle : Spindles)
      Spindle->~Spindle();

    SubTasks.clear();
    ParentTask = nullptr;
  }
};

//===----------------------------------------------------------------------===//
/// This class builds and contains all of the top-level task
/// structures in the specified function.
///

class TaskInfo {
  // BBMap - Mapping of basic blocks to the innermost spindle they occur in
  DenseMap<const BasicBlock *, Spindle *> BBMap;
  // SpindleMap - Mapping of spindles to the innermost task they occur in
  DenseMap<const Spindle *, Task *> SpindleMap;

  Task *RootTask;
  // Dominator tree
  DominatorTree &DomTree;

  BumpPtrAllocator TaskAllocator;

  void operator=(const TaskInfo &) = delete;
  TaskInfo(const TaskInfo &) = delete;

public:
  explicit TaskInfo(DominatorTree &DomTree) : DomTree(DomTree) {}
  ~TaskInfo() { releaseMemory(); }

  TaskInfo(TaskInfo &&Arg)
      : BBMap(std::move(Arg.BBMap)),
        SpindleMap(std::move(Arg.SpindleMap)),
        RootTask(std::move(Arg.RootTask)),
        DomTree(Arg.DomTree),
        TaskAllocator(std::move(Arg.TaskAllocator)) {
  }
  TaskInfo &operator=(TaskInfo &&RHS) {
    BBMap = std::move(RHS.BBMap);
    SpindleMap = std::move(RHS.SpindleMap);
    RootTask->~Task();
    RootTask = std::move(RHS.RootTask);
    TaskAllocator = std::move(RHS.TaskAllocator);
    DomTree = std::move(RHS.DomTree);
    return *this;
  }

  void releaseMemory() {
    BBMap.clear();
    SpindleMap.clear();
    RootTask->~Task();
    TaskAllocator.Reset();
  }

  template <typename... ArgsTy> Spindle *AllocateSpindle(ArgsTy &&... Args) {
    Spindle *Storage = TaskAllocator.Allocate<Spindle>();
    return new (Storage) Spindle(std::forward<ArgsTy>(Args)...);
  }
  template <typename... ArgsTy> Task *AllocateTask(ArgsTy &&... Args) {
    Task *Storage = TaskAllocator.Allocate<Task>();
    return new (Storage) Task(std::forward<ArgsTy>(Args)...);
  }

  /// Return true if this function is "serial," meaning it does not itself
  /// perform a detach.  This method does not preclude functions called by this
  /// function from performing a detach.
  bool isSerial() const { return RootTask->SubTasks.empty(); }

  Task *getRootTask() const { return RootTask; }

  /// iterator/begin/end - The interface to the top-level tasks in the current
  /// function.
  ///
  using iterator = Task::iterator;
  using reverse_iterator = Task::reverse_iterator;
  iterator begin() const { return getRootTask()->begin(); }
  iterator end() const { return getRootTask()->end(); }
  reverse_iterator rbegin() const { return getRootTask()->rbegin(); }
  reverse_iterator rend() const { return getRootTask()->rend(); }
  bool empty() const { return getRootTask()->empty(); }

  /// Return the innermost spindle that BB lives in.
  // const Spindle *getSpindleFor(const BasicBlock *BB) const {
  //   return const_cast<Spindle *>(getSpindleFor(BB));
  // }
  Spindle *getSpindleFor(const BasicBlock *BB) const {
    return BBMap.lookup(BB);
  }

  /// Return the innermost task that spindle F lives in.
  Task *getTaskFor(const Spindle *S) const { return SpindleMap.lookup(S); }
  /// Same as getTaskFor(S).
  const Task *operator[](const Spindle *S) const { return getTaskFor(S); }

  /// Return the innermost task that BB lives in.
  Task *getTaskFor(const BasicBlock *BB) const {
    return getTaskFor(getSpindleFor(BB));
  }
  /// Same as getTaskFor(BB).
  const Task *operator[](const BasicBlock *BB) const { return getTaskFor(BB); }

  /// Return the innermost task that encompases both basic blocks BB1 and BB2.
  Task *getEnclosingTask(const BasicBlock *BB1, const BasicBlock *BB2) const {
    return getTaskFor(DomTree.findNearestCommonDominator(BB1, BB2));
  }

  /// Return the innermost task that encompases both spindles S1 and S2.
  Task *getEnclosingTask(const Spindle *S1, const Spindle *S2) const {
    return getTaskFor(DomTree.findNearestCommonDominator(
                          S1->getEntry(), S2->getEntry()));
  }

  /// Return true if task T1 contains task T2.
  bool contains(const Task *T1, const Task *T2) const {
    if (!T1 || !T2) return false;
    return DomTree.dominates(T1->getEntry(), T2->getEntry());
  }

  /// Return true if specified task contains the specified basic block.
  bool contains(const Task *T, const BasicBlock *BB) const {
    if (!T) return false;
    assert(&DomTree == &(T->DomTree) &&
           "Task's DomTree does not match TaskInfo DomTree.");
    return T->contains(BB);
  }

  /// Return true if the specified instruction is in this task.
  bool contains(const Task *T, const Instruction *Inst) const {
    return contains(T, Inst->getParent());
  }

  /// Return the task nesting level of the specified block. A depth of 0 means
  /// the block is in the root task.
  unsigned getTaskDepth(const BasicBlock *BB) const {
    return getTaskFor(BB)->getTaskDepth();
  }

  // True if the block is a task entry block
  bool isTaskEntry(const BasicBlock *BB) const {
    return getTaskFor(BB)->getEntry() == BB;
  }

  //===----------------------------------------------------------------------===//
  // Spindle pred_iterator definition
  //===----------------------------------------------------------------------===//
  template<typename TaskInfoT = TaskInfo *, typename SpindleT = Spindle *,
           typename PredIteratorT = pred_iterator>
  class pred_spindle_iterator_impl
    : public iterator_facade_base<
    pred_spindle_iterator_impl<TaskInfoT, SpindleT, PredIteratorT>,
    std::forward_iterator_tag, SpindleT> {
    using SelfT = pred_spindle_iterator_impl<TaskInfoT, SpindleT,
                                             PredIteratorT>;
    TaskInfoT TI;
    PredIteratorT PIter;

  public:
    pred_spindle_iterator_impl(TaskInfoT TI, SpindleT S)
        : TI(TI), PIter(S->getEntry())
    {}
    pred_spindle_iterator_impl(TaskInfoT TI, SpindleT S, bool)
        : TI(TI), PIter(S->getEntry(), true)
    {}

    // Allow default construction to build variables, but this doesn't build
    // a useful iterator.
    pred_spindle_iterator_impl() = default;

    // Allow conversion between instantiations where valid.
    pred_spindle_iterator_impl(const SelfT &Arg)
        : TI(Arg.TI), PIter(Arg.PIter) {}

    bool operator==(const pred_spindle_iterator_impl &Arg) const {
      assert(TI == Arg.TI && "Incomparable pred_spindle_iterators");
      return (PIter == Arg.PIter);
    }
    bool operator!=(const pred_spindle_iterator_impl &Arg) const {
      assert(TI == Arg.TI && "Incomparable pred_spindle_iterators");
      return (PIter != Arg.PIter);
    }

    SpindleT operator*() const { return TI->getSpindleFor(*PIter); }

    using pred_spindle_iterator_impl::iterator_facade_base::operator++;
    pred_spindle_iterator_impl &operator++() {
      PIter++;
      return *this;
    }
  };
  using pred_spindle_iterator = pred_spindle_iterator_impl<>;
  using const_pred_spindle_iterator =
    pred_spindle_iterator_impl<const TaskInfo *, const Spindle *,
                               const_pred_iterator>;
  using pred_spindle_range = iterator_range<pred_spindle_iterator>;
  using const_pred_spindle_range = iterator_range<const_pred_spindle_iterator>;

  pred_spindle_iterator pred_begin(Spindle *S) {
    return pred_spindle_iterator(this, S);
  }
  const_pred_spindle_iterator pred_begin(const Spindle *S) const {
    return const_pred_spindle_iterator(this, S);
  }

  pred_spindle_iterator pred_end(Spindle *S) {
    return pred_spindle_iterator(this, S, true);
  }
  const_pred_spindle_iterator pred_end(const Spindle *S) const {
    return const_pred_spindle_iterator(this, S, true);
  }

  bool pred_empty(const Spindle *S) const {
    return pred_begin(S) == pred_end(S);
  }
  pred_spindle_range predecessors(Spindle *S) {
    return pred_spindle_range(pred_begin(S), pred_end(S));
  }
  const_pred_spindle_range predecessors(const Spindle *S) const {
    return const_pred_spindle_range(pred_begin(S), pred_end(S));
  }

  //===----------------------------------------------------------------------===//
  // Spindle succ_iterator definition
  //===----------------------------------------------------------------------===//

  /// Iterator to walk just the successors of a spindle.
  template<typename TaskInfoT = TaskInfo *, typename SpindleT = Spindle *,
           typename SPExitIteratorT = Spindle::exitblock_iterator,
           typename SuccIteratorT = succ_iterator>
  class succ_spindle_iterator_impl
    : public iterator_facade_base<
    succ_spindle_iterator_impl<TaskInfoT, SpindleT, SPExitIteratorT,
                               SuccIteratorT>,
    std::forward_iterator_tag, SpindleT> {

    using SelfT = succ_spindle_iterator_impl<TaskInfoT, SpindleT,
                                             SPExitIteratorT, SuccIteratorT>;

    TaskInfoT TI;
    SPExitIteratorT SPExit;
    SuccIteratorT Succ, End;

    void findNextSuccessor() {
      while(!SPExit.atEnd() && SPExit->getTerminator()->getNumSuccessors() == 0)
        ++SPExit;
      if (!SPExit.atEnd()) {
        // dbgs() << "New SPExit: " << SPExit->getName() <<
        //   ", terminator " << *SPExit->getTerminator() << "\n";
        Succ = SuccIteratorT(SPExit->getTerminator());
        End = SuccIteratorT(SPExit->getTerminator(), true);
      } // else
        // dbgs() << "End SPExit reached\n";
    }

  public:
    succ_spindle_iterator_impl(TaskInfoT TI, SpindleT S)
        : TI(TI), SPExit(S), Succ(nullptr, true), End(nullptr, true) {
      // dbgs() << "Initial exit: " << SPExit->getName() <<
      //   ", terminator " << *SPExit->getTerminator() << "\n";
      findNextSuccessor();
    }
    succ_spindle_iterator_impl(TaskInfoT TI, SpindleT S, bool)
        : TI(TI), SPExit(SPExitIteratorT::end(S)), Succ(nullptr, true),
          End(nullptr, true)
    {}

    // Allow default construction to build variables, but this doesn't build
    // a useful iterator.
    succ_spindle_iterator_impl() = default;

    // Allow conversion between instantiations where valid.
    succ_spindle_iterator_impl(const SelfT &Arg)
        : TI(Arg.TI), SPExit(Arg.SPExit), Succ(Arg.Succ), End(Arg.End) {}

    bool operator==(const succ_spindle_iterator_impl &Arg) const {
      assert(TI == Arg.TI && "Incomparable succ_spindle_iterators");
      return (SPExit == Arg.SPExit) && (SPExit.atEnd() || (Succ == Arg.Succ));
    }
    bool operator!=(const succ_spindle_iterator_impl &Arg) const {
      assert(TI == Arg.TI && "Incomparable succ_spindle_iterators");
      return (SPExit != Arg.SPExit) || (!SPExit.atEnd() && (Succ != Arg.Succ));
    }

    SpindleT operator*() const {
      assert(!SPExit.atEnd() && "succ_spindle_iterator out of range!");
      return TI->getSpindleFor(*Succ);
    }

    using succ_spindle_iterator_impl::iterator_facade_base::operator++;
    succ_spindle_iterator_impl &operator++() {
      if (++Succ != End)
        return *this;
      ++SPExit;
      findNextSuccessor();
      return *this;
    }
  };
  using succ_spindle_iterator = succ_spindle_iterator_impl<>;
  using const_succ_spindle_iterator =
    succ_spindle_iterator_impl<const TaskInfo *, const Spindle *,
                               Spindle::const_exitblock_iterator,
                               succ_const_iterator>;
  using succ_spindle_range = iterator_range<succ_spindle_iterator>;
  using const_succ_spindle_range = iterator_range<const_succ_spindle_iterator>;

  succ_spindle_iterator succ_begin(Spindle *S) {
    return succ_spindle_iterator(this, S);
  }
  const_succ_spindle_iterator succ_begin(const Spindle *S) const {
    return const_succ_spindle_iterator(this, S);
  }

  succ_spindle_iterator succ_end(Spindle *S) {
    return succ_spindle_iterator(this, S, true);
  }
  const_succ_spindle_iterator succ_end(const Spindle *S) const {
    return const_succ_spindle_iterator(this, S, true);
  }

  bool succ_empty(const Spindle *S) {
    return succ_begin(S) == succ_end(S);
  }
  succ_spindle_range successors(Spindle *S) {
    return succ_spindle_range(succ_begin(S), succ_end(S));
  }
  const_succ_spindle_range successors(const Spindle *S) const {
    return const_succ_spindle_range(succ_begin(S), succ_end(S));
  }
  
  /// Create the task forest using a stable algorithm.
  void analyze(Function &F);

  // void determineSyncState(Spindle *S,
  //                         DenseMap<const Spindle *, unsigned> &State,
  //                         Value *SyncRegion = nullptr);

  /// Handle invalidation explicitly.
  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &);

  // Debugging
  void print(raw_ostream &OS) const;

  void verify(const DominatorTree &DomTree) const;

  /// Destroy a task that has been removed from the `TaskInfo` nest.
  ///
  /// This runs the destructor of the task object making it invalid to
  /// reference afterward. The memory is retained so that the *pointer* to the
  /// task remains valid.
  ///
  /// The caller is responsible for removing this task from the task nest and
  /// otherwise disconnecting it from the broader `TaskInfo` data structures.
  /// Callers that don't naturally handle this themselves should probably call
  /// `erase' instead.
  void destroy(Task *T) {
    T->~Task();

    // Since TaskAllocator is a BumpPtrAllocator, this Deallocate only poisons
    // \c T, but the pointer remains valid for non-dereferencing uses.
    TaskAllocator.Deallocate(T);
  }

  // Create a spindle with entry block B and type Ty.
  Spindle *createSpindleWithEntry(BasicBlock *B, Spindle::SPType Ty) {
    Spindle *S = AllocateSpindle(B, Ty);
    assert(!BBMap[B] && "BasicBlock already in a spindle!");
    BBMap[B] = S;
    return S;
  }

  // Create a task with spindle entry S.
  Task *createTaskWithEntry(Spindle *S) {
    Task *T = AllocateTask(*S, DomTree);
    S->setParentTask(T);
    assert(!SpindleMap[S] && "Spindle already in a task!");
    SpindleMap[S] = T;
    return T;
  }

  // Add spindle S to task T.
  void addSpindleToTask(Spindle *S, Task *T) {
    assert(!SpindleMap[S] && "Spindle already mapped to a task.");
    T->addSpindle(*S);
    S->setParentTask(T);
    SpindleMap[S] = T;
  }

  // Add spindle S to task T, where S is a shared exception-handling spindle
  // among subtasks of T.
  void addEHSpindleToTask(Spindle *S, Task *T) {
    assert(!SpindleMap[S] && "Spindle already mapped to a task.");
    T->addEHSpindle(*S);
    S->setParentTask(T);
    SpindleMap[S] = T;
  }

  // Add basic block B to spindle S.
  void addBlockToSpindle(BasicBlock &B, Spindle *S) {
    assert(!BBMap[&B] && "Block already mapped to a spindle.");
    S->addBlock(B);
    BBMap[&B] = S;
  }
};

// Allow clients to walk the list of nested tasks.
template <> struct GraphTraits<const Task *> {
  using NodeRef = const Task *;
  using ChildIteratorType = TaskInfo::iterator;

  static NodeRef getEntryNode(const Task *T) { return T; }
  static ChildIteratorType child_begin(NodeRef N) { return N->begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->end(); }
};

template <> struct GraphTraits<Task *> {
  using NodeRef = Task *;
  using ChildIteratorType = TaskInfo::iterator;

  static NodeRef getEntryNode(Task *T) { return T; }
  static ChildIteratorType child_begin(NodeRef N) { return N->begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->end(); }
};

/// \brief Analysis pass that exposes the \c TaskInfo for a function.
class TaskAnalysis : public AnalysisInfoMixin<TaskAnalysis> {
  friend AnalysisInfoMixin<TaskAnalysis>;
  static AnalysisKey Key;

public:
  using Result = TaskInfo;

  TaskInfo run(Function &F, FunctionAnalysisManager &AM);
};

/// \brief Printer pass for the \c TaskAnalysis results.
class TaskPrinterPass : public PassInfoMixin<TaskPrinterPass> {
  raw_ostream &OS;

public:
  explicit TaskPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// \brief Verifier pass for the \c TaskAnalysis results.
struct TaskVerifierPass : public PassInfoMixin<TaskVerifierPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// \brief The legacy pass manager's analysis pass to compute task information.
class TaskInfoWrapperPass : public FunctionPass {
  std::unique_ptr<TaskInfo> TI;

public:
  static char ID; // Pass identification, replacement for typeid

  TaskInfoWrapperPass() : FunctionPass(ID) {
    initializeTaskInfoWrapperPassPass(*PassRegistry::getPassRegistry());
  }

  TaskInfo &getTaskInfo() { return *TI; }
  const TaskInfo &getTaskInfo() const { return *TI; }

  /// \brief Calculate the natural task information for a given function.
  bool runOnFunction(Function &F) override;

  void verifyAnalysis() const override;

  void releaseMemory() override { TI.reset(); }

  void print(raw_ostream &O, const Module *M = nullptr) const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

/// Function to print a task's contents as LLVM's text IR assembly.
void printTask(Task &T, raw_ostream &OS, const std::string &Banner = "");

} // End llvm namespace

#endif
