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
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
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
class Loop;
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
  using SpindleEdge = std::pair<Spindle *, BasicBlock *>;

private:
  SPType Ty;

  Task *ParentTask = nullptr;

  // The list of basic blocks in this spindle.  The first entry is the entry
  // block of the spindle.
  std::vector<BasicBlock *> Blocks;

  SmallPtrSet<const BasicBlock *, 8> DenseBlockSet;

  // Predecessor and successor spindles.
  SmallVector<SpindleEdge, 8> Incoming;
  SmallVector<SpindleEdge, 8> Outgoing;
  
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

  /// Return true if this spindle is a shared EH spindle.
  bool isSharedEH() const;

  /// Return true if this spindle is the continuation of a detached task.
  bool isTaskContinuation() const;

  /// Return true if the predecessor spindle Pred is part of a different task
  /// from this spindle.
  bool predInDifferentTask(const Spindle *Pred) const {
    return (getParentTask() != Pred->getParentTask()) && !isSharedEH();
  }
  /// Return true if the successor spindle Succ is part of the same task as this
  /// spindle.
  bool succInSameTask(const Spindle *Succ) const;

  /// Return true if the successor spindle Succ is part of the same task as this
  /// spindle.
  bool succInSubTask(const Spindle *Succ) const;

  /// Get a list of the basic blocks which make up this task.
  ArrayRef<BasicBlock *> getBlocks() const {
    return Blocks;
  }
  using iterator = typename ArrayRef<BasicBlock *>::const_iterator;
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

  /// Helper class for iterator to walk just the exiting basic blocks of the
  /// spindle.
  class SpindleExitingFilter {
    const Spindle *S = nullptr;
  public:
    SpindleExitingFilter() {}
    SpindleExitingFilter(const Spindle *S) : S(S) {}
    bool operator()(const BasicBlock *B) const {
      return S->isSpindleExiting(B);
    }
  };
  inline iterator_range<
    filter_iterator<ArrayRef<BasicBlock *>::iterator,
                    SpindleExitingFilter>> spindle_exits() {
    return make_filter_range(blocks(), SpindleExitingFilter(this));
  }
  inline iterator_range<
    filter_iterator<ArrayRef<BasicBlock *>::const_iterator,
                    SpindleExitingFilter>> spindle_exits() const {
    return make_filter_range(blocks(), SpindleExitingFilter(this));
  }

  // Iterators for the incoming and outgoing edges of this spindle.
  using spedge_iterator = typename SmallVectorImpl<SpindleEdge>::iterator;
  using spedge_const_iterator =
    typename SmallVectorImpl<SpindleEdge>::const_iterator;
  using spedge_range = iterator_range<spedge_iterator>;
  using spedge_const_range = iterator_range<spedge_const_iterator>;

  inline spedge_iterator in_begin() { return Incoming.begin(); }
  inline spedge_const_iterator in_begin() const {
    return Incoming.begin();
  }
  inline spedge_iterator in_end() { return Incoming.end(); }
  inline spedge_const_iterator in_end() const {
    return Incoming.end();
  }
  inline spedge_range in_edges() {
    return make_range<spedge_iterator>(in_begin(), in_end());
  }
  inline spedge_const_range in_edges() const {
    return make_range<spedge_const_iterator>(in_begin(), in_end());
  }

  inline spedge_iterator out_begin() { return Outgoing.begin(); }
  inline spedge_const_iterator out_begin() const {
    return Outgoing.begin();
  }
  inline spedge_iterator out_end() { return Outgoing.end(); }
  inline spedge_const_iterator out_end() const {
    return Outgoing.end();
  }
  inline spedge_range out_edges() {
    return make_range<spedge_iterator>(out_begin(), out_end());
  }
  inline spedge_const_range out_edges() const {
    return make_range<spedge_const_iterator>(out_begin(), out_end());
  }

  template<typename SPEdgeIt = spedge_iterator, typename SpindleT = Spindle *>
  class adj_iterator_impl
    : public iterator_adaptor_base<
    adj_iterator_impl<SPEdgeIt, SpindleT>, SPEdgeIt,
    typename std::iterator_traits<SPEdgeIt>::iterator_category,
    SpindleT, std::ptrdiff_t, SpindleT *, SpindleT> {

    using BaseT = iterator_adaptor_base<
      adj_iterator_impl<SPEdgeIt, SpindleT>, SPEdgeIt,
      typename std::iterator_traits<SPEdgeIt>::iterator_category,
      SpindleT, std::ptrdiff_t, SpindleT *, SpindleT>;

  public:
    adj_iterator_impl(SPEdgeIt Begin) : BaseT(Begin) {}
    inline SpindleT operator*() const { return BaseT::I->first; }
  };

  using adj_iterator = adj_iterator_impl<>;
  using adj_const_iterator =
    adj_iterator_impl<spedge_const_iterator, const Spindle *>;
  using adj_range = iterator_range<adj_iterator>;
  using adj_const_range = iterator_range<adj_const_iterator>;

  /// Print spindle with all the BBs inside it.
  void print(raw_ostream &OS, bool Verbose = false) const;

  /// Raw method to add block B to this spindle.
  void addBlock(BasicBlock &B) {
    Blocks.push_back(&B);
    DenseBlockSet.insert(&B);
  }

  // Returns true if the basic block B predeces this spindle.
  bool blockPrecedesSpindle(const BasicBlock *B) const {
    for (const BasicBlock *SB : successors(B))
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
    Outgoing.push_back(SpindleEdge(Succ, FromExit));
    Succ->Incoming.push_back(SpindleEdge(this, FromExit));
  }

protected:
  friend class Task;
  friend class TaskInfo;

  /// This creates an empty spindle.
  Spindle() {}

  explicit Spindle(BasicBlock *BB, SPType Ty) : Ty(Ty) {
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
    Incoming.clear();
    Outgoing.clear();
    ParentTask = nullptr;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const Spindle &S);

// Iterators for the predecessors of a Spindle, using the Spindle edges.
using pred_spindle_iterator = typename Spindle::adj_iterator;
using pred_spindle_const_iterator = typename Spindle::adj_const_iterator;
using pred_spindle_range = iterator_range<pred_spindle_iterator>;
using pred_spindle_const_range = iterator_range<pred_spindle_const_iterator>;

inline pred_spindle_iterator pred_begin(Spindle *S) {
  return pred_spindle_iterator(S->in_begin());
}
inline pred_spindle_const_iterator pred_begin(const Spindle *S) {
  return pred_spindle_const_iterator(S->in_begin());
}
inline pred_spindle_iterator pred_end(Spindle *S) {
  return pred_spindle_iterator(S->in_end());
}
inline pred_spindle_const_iterator pred_end(const Spindle *S) {
  return pred_spindle_const_iterator(S->in_end());
}
inline pred_spindle_range predecessors(Spindle *S) {
  return pred_spindle_range(pred_begin(S), pred_end(S));
}
inline pred_spindle_const_range predecessors(const Spindle *S) {
  return pred_spindle_const_range(pred_begin(S), pred_end(S));
}

// Iterators for the successors of a Spindle, using the Spindle edges.
using succ_spindle_iterator = typename Spindle::adj_iterator;
using succ_spindle_const_iterator = typename Spindle::adj_const_iterator;
using succ_spindle_range = iterator_range<succ_spindle_iterator>;
using succ_spindle_const_range = iterator_range<succ_spindle_const_iterator>;

inline succ_spindle_iterator succ_begin(Spindle *S) {
  return succ_spindle_iterator(S->out_begin());
}
inline succ_spindle_const_iterator succ_begin(const Spindle *S) {
  return succ_spindle_const_iterator(S->out_begin());
}
inline succ_spindle_iterator succ_end(Spindle *S) {
  return succ_spindle_iterator(S->out_end());
}
inline succ_spindle_const_iterator succ_end(const Spindle *S) {
  return succ_spindle_const_iterator(S->out_end());
}
inline succ_spindle_range successors(Spindle *S) {
  return succ_spindle_range(succ_begin(S), succ_end(S));
}
inline succ_spindle_const_range successors(const Spindle *S) {
  return succ_spindle_const_range(succ_begin(S), succ_end(S));
}

// Helper class for iterating over spindles within the same task.
class InTaskFilter {
  const Spindle *S = nullptr;
public:
  InTaskFilter() {}
  InTaskFilter(const Spindle *S) : S(S) {}
  bool operator()(const Spindle *Succ) const {
    return S->succInSameTask(Succ);
  }
};

//===--------------------------------------------------------------------===//
// GraphTraits specializations for spindle graphs
//===--------------------------------------------------------------------===//

// Provide specializations of GraphTraits to be able to treat a function
// as a graph of spindles.

template <> struct GraphTraits<Spindle *> {
  using NodeRef = Spindle *;
  using ChildIteratorType = succ_spindle_iterator;

  static NodeRef getEntryNode(Spindle *S) { return S; }
  static ChildIteratorType child_begin(NodeRef N) { return succ_begin(N); }
  static ChildIteratorType child_end(NodeRef N) { return succ_end(N); }
};

template <> struct GraphTraits<const Spindle *> {
  using NodeRef = const Spindle *;
  using ChildIteratorType = succ_spindle_const_iterator;

  static NodeRef getEntryNode(const Spindle *S) { return S; }
  static ChildIteratorType child_begin(NodeRef N) { return succ_begin(N); }
  static ChildIteratorType child_end(NodeRef N) { return succ_end(N); }
};

// Provide specializations of GraphTrais to be able to treat a function as a
// graph of spindles and walk it in inverse order.  Inverse order in this case
// is considered to be when traversing the predecessor edges of a spindle
// instead of the successor edges.

template <> struct GraphTraits<Inverse<Spindle *>> {
  using NodeRef = Spindle *;
  using ChildIteratorType = pred_spindle_iterator;

  static NodeRef getEntryNode(Inverse<Spindle *> G) { return G.Graph; }
  static ChildIteratorType child_begin(NodeRef N) { return pred_begin(N); }
  static ChildIteratorType child_end(NodeRef N) { return pred_end(N); }
};

template <> struct GraphTraits<Inverse<const Spindle *>> {
  using NodeRef = const Spindle *;
  using ChildIteratorType = pred_spindle_const_iterator;

  static NodeRef getEntryNode(Inverse<const Spindle *> G) { return G.Graph; }
  static ChildIteratorType child_begin(NodeRef N) { return pred_begin(N); }
  static ChildIteratorType child_end(NodeRef N) { return pred_end(N); }
};

// Special type of GraphTrait that uses a filter on the successors of a spindle.
// This GraphTrait is used to build the InTask and UnderTask GraphTraits.

template <typename SpindlePtrT, typename Filter>
using FilteredSuccessorSpindles = std::pair<SpindlePtrT, Filter>;

template <typename Filter>
struct GraphTraits<FilteredSuccessorSpindles<Spindle *, Filter>> {
  using NodeRef = Spindle *;
  using ChildIteratorType = filter_iterator<succ_spindle_iterator, Filter>;

  static NodeRef getEntryNode(FilteredSuccessorSpindles<Spindle *, Filter> S) {
    return S.first;
  }
  static ChildIteratorType child_begin(NodeRef N) {
    return make_filter_range(successors(N), Filter(N)).begin();
  }
  static ChildIteratorType child_end(NodeRef N) {
    return make_filter_range(successors(N), Filter(N)).end();
  }
};

template <typename Filter>
struct GraphTraits<FilteredSuccessorSpindles<const Spindle *, Filter>> {
  using NodeRef = const Spindle *;
  using ChildIteratorType =
    filter_iterator<succ_spindle_const_iterator, Filter>;

  static NodeRef getEntryNode(
      FilteredSuccessorSpindles<const Spindle *, Filter> S) {
    return S.first;
  }
  static ChildIteratorType child_begin(NodeRef N) {
    return make_filter_range(successors(N), Filter(N)).begin();
  }
  static ChildIteratorType child_end(NodeRef N) {
    return make_filter_range(successors(N), Filter(N)).end();
  }
};

// Wrapper to allow traversal of only those spindles within a task, excluding
// all subtasks of that task.
template <typename SpindlePtrT>
struct InTask
  : public FilteredSuccessorSpindles<SpindlePtrT,
                                     InTaskFilter> {
  inline InTask(SpindlePtrT S)
      : FilteredSuccessorSpindles<SpindlePtrT, InTaskFilter>
      (S, InTaskFilter(S)) {}
};

template<> struct GraphTraits<InTask<Spindle *>> :
    public GraphTraits<FilteredSuccessorSpindles<
                         Spindle *, InTaskFilter>> {
  using NodeRef = Spindle *;
  static NodeRef getEntryNode(InTask<Spindle *> G) {
    return G.first;
  }
};
template<> struct GraphTraits<InTask<const Spindle *>> :
    public GraphTraits<FilteredSuccessorSpindles<
                         const Spindle *, InTaskFilter>> {
  using NodeRef = const Spindle *;
  static NodeRef getEntryNode(InTask<const Spindle *> G) {
    return G.first;
  }
};

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
  SmallPtrSet<const Spindle *, 1> DenseEHSpindleSet;

  // Pointers to the continuation and exceptional-continuation spindles for this
  // task.
  Spindle *Continuation = nullptr;
  Spindle *EHContinuation = nullptr;
  // The exceptional continuation of the task might not be a landingpad, due to
  // transformations on exception-handling code.  Hence we keep track of the
  // value of landingpad at the exceptional continuation.
  Value *LPadValueInEHContinuation = nullptr;

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

  /// Return true if this task is a "root" task, meaning that it has no parent task.
  bool isRootTask() const { return nullptr == ParentTask; }

  /// Return the detach instruction that created this task, or nullptr if this
  /// task is a root task.
  DetachInst *getDetach() const {
    if (isRootTask()) return nullptr;
    BasicBlock *Detacher = getEntry()->getSinglePredecessor();
    assert(Detacher &&
           "Entry block of non-root task should have a single predecessor");
    assert(isa<DetachInst>(Detacher->getTerminator()) &&
           "Single predecessor of a task should be terminated by a detach");
    return dyn_cast<DetachInst>(Detacher->getTerminator());
  }

  /// Get the spindle for the continuation of this task.  Returns nullptr if
  /// this task is a root task, meaning it has no continuation spindle.
  Spindle *getContinuationSpindle() const {
    assert(((isRootTask() && !Continuation) || (!isRootTask() && Continuation))
           && "Task should have a continuation spindle iff not a root task.");
    return Continuation;
  }

  /// Get the spindle for the exceptional continuation o fthis task.  Returns
  /// nullptr if this task is a root task or the detach for this task does not
  /// have an unwind destination.
  Spindle *getEHContinuationSpindle() const {
    assert(((isRootTask() && !EHContinuation) ||
            (!isRootTask() &&
             (getDetach()->hasUnwindDest() && EHContinuation) ||
             (!getDetach()->hasUnwindDest() && !EHContinuation))) &&
           "Task should have a EH continuation spindle iff not a root task and "
           "detach has an unwind destination.");
    return EHContinuation;
  }

  /// Get the spindle for the exceptional continuation o fthis task.  Returns
  /// nullptr if this task is a root task or the detach for this task does not
  /// have an unwind destination.
  Value *getLPadValueInEHContinuationSpindle() const {
    assert(((isRootTask() && !LPadValueInEHContinuation) ||
            (!isRootTask() &&
             (getDetach()->hasUnwindDest() && LPadValueInEHContinuation) ||
             (!getDetach()->hasUnwindDest() && !LPadValueInEHContinuation))) &&
           "Task should have a EH continuation spindle iff not a root task and "
           "detach has an unwind destination.");
    return LPadValueInEHContinuation;
  }

  /// Return true if spindle S is in this task.
  bool contains(const Spindle *S) const {
    return DenseSpindleSet.count(S);
  }

  /// Return true if spindle S is a shared EH spindle dominated by this task.
  bool containsSharedEH(const Spindle *S) const {
    return DenseEHSpindleSet.count(S);
  }

  /// Return true if basic block B is in a shared EH spindle dominated by this
  /// task.
  bool containsSharedEH(const BasicBlock *B) const {
    for (const Spindle *S : SharedSubTaskEH)
      if (S->contains(B))
        return true;
    return false;
  }

  /// Return the tasks contained entirely within this task.
  ArrayRef<Task *> getSubTasks() const {
    return SubTasks;
  }
  std::vector<Task *> &getSubTasksVector() {
    return SubTasks;
  }
  using iterator = typename std::vector<Task *>::const_iterator;
  using const_iterator = iterator;
  using reverse_iterator =
                       typename std::vector<Task *>::const_reverse_iterator;
  using const_reverse_iterator = reverse_iterator;
  inline iterator begin() const { return SubTasks.begin(); }
  inline iterator end() const { return SubTasks.end(); }
  inline reverse_iterator rbegin() const { return SubTasks.rbegin(); }
  inline reverse_iterator rend() const { return SubTasks.rend(); }
  inline bool empty() const { return SubTasks.empty(); }
  inline iterator_range<iterator> subtasks() const {
    return make_range(begin(), end());
  }

  /// Get the number of spindles in this task in constant time.
  unsigned getNumSpindles() const {
    return Spindles.size();
  }

  /// Return the spindles contained within this task and no subtask.
  ArrayRef<Spindle *> getSpindles() const {
    return Spindles;
  }
  std::vector<Spindle *> &getSpindlesVector() {
    return Spindles;
  }
  SmallPtrSetImpl<const Spindle *> &getSpindlesSet() {
    return DenseSpindleSet;
  }

  using spindle_iterator = typename std::vector<Spindle *>::const_iterator;
  inline spindle_iterator spindle_begin() const {
    return Spindles.begin();
  }
  inline spindle_iterator spindle_end() const {
    return Spindles.end();
  }
  inline iterator_range<spindle_iterator> spindles() const {
    return make_range(spindle_begin(), spindle_end());
  }

  /// Returns true if this task exits to a shared EH spindle.
  bool hasSharedEHExit() const {
    if (isRootTask()) return false;
    if (!getParentTask()->tracksSharedEHSpindles()) return false;

    for (Spindle *S : getSpindles())
      for (Spindle *Succ : successors(S))
        if (getParentTask()->containsSharedEH(Succ))
          return true;

    return false;
  }

  /// Returns true if SharedEH is a shared EH exit of this task.
  bool isSharedEHExit(const Spindle *SharedEH) const;

  /// Get the shared EH spindles that this task can exit to and append them to
  /// SpindleVec.
  void getSharedEHExits(SmallVectorImpl<Spindle *> &SpindleVec) const;

  /// Returns true if this task tracks any shared EH spindles for its subtasks.
  bool tracksSharedEHSpindles() const {
    return !SharedSubTaskEH.empty();
  }
  /// Get the number of shared EH spindles in this task in constant time.
  unsigned getNumSharedEHSpindles() const {
    return SharedSubTaskEH.size();
  }

  /// Return the shared EH spindles contained within this task.
  const SmallVectorImpl<Spindle *> &getSharedEHSpindles() const {
    return SharedSubTaskEH;
  }
  SmallVectorImpl<Spindle *> &getSharedEHSpindles() {
    return SharedSubTaskEH;
  }
  /// Get the shared EH spindle containing basic block B, if it exists.
  const Spindle *getSharedEHContaining(const BasicBlock *B) const {
    for (const Spindle *S : SharedSubTaskEH)
      if (S->contains(B))
        return S;
    return nullptr;
  }
  Spindle *getSharedEHContaining(BasicBlock *B) const {
    for (Spindle *S : SharedSubTaskEH)
      if (S->contains(B))
        return S;
    return nullptr;
  }

  using shared_eh_spindle_iterator =
    typename SmallVectorImpl<Spindle *>::const_iterator;
  shared_eh_spindle_iterator shared_eh_spindle_begin() const {
    return getSharedEHSpindles().begin();
  }
  shared_eh_spindle_iterator shared_eh_spindle_end() const {
    return getSharedEHSpindles().end();
  }
  inline iterator_range<shared_eh_spindle_iterator>
  shared_eh_spindles() const {
    return make_range(shared_eh_spindle_begin(), shared_eh_spindle_end());
  }

  /// Get a list of all basic blocks in this task, including blocks in
  /// descendant tasks.
  void getDominatedBlocks(SmallVectorImpl<BasicBlock *> &Blocks) const {
    DomTree.getDescendants(getEntry(), Blocks);
  }

  /// Returns true if this task encloses basic block BB simply, that is, without
  /// checking any shared EH exits of this task.
  bool simplyEncloses(const BasicBlock *BB) const {
    return DomTree.dominates(getEntry(), BB);
  }

  /// Return true if specified task encloses basic block BB.
  bool encloses(const BasicBlock *BB) const {
    if (simplyEncloses(BB))
      return true;
    if (ParentTask && ParentTask->tracksSharedEHSpindles())
      if (const Spindle *SharedEH = ParentTask->getSharedEHContaining(BB))
        return isSharedEHExit(SharedEH);
    return false;
  }

  /// Returns either the representative subtask of this task that encloses basic
  /// block B or the this task itself if no subtask encloses B.  This task must
  /// enclose B.
  ///
  /// These representatives are useful for studying series-parallel
  /// relationships between basic blocks in a function when those basic blocks
  /// might appear in nested subtasks.
  const Task *getSubTaskEnclosing(const BasicBlock *BB) const {
    assert(encloses(BB) && "Task does not enclose given BasicBlock");
    for (Task *SubT : subtasks())
      if (SubT->encloses(BB))
        return SubT;
    return this;
  }

  /// True if terminator in the block can branch to another block that is
  /// outside of the current task.
  bool isTaskExiting(const BasicBlock *BB) const {
    if (BB->getTerminator()->getNumSuccessors() == 0)
      return true;
    for (const auto &Succ : children<const BasicBlock *>(BB))
      if (!encloses(Succ))
        return true;
    return false;
  }

  /// True if the spindle can exit to a block that is outside of the current
  /// task.
  bool isTaskExiting(const Spindle *S) const {
    for (const BasicBlock *Exit : S->spindle_exits())
      if (isTaskExiting(Exit))
        return true;
    return false;
  }

  // Returns true if the specified value is defined in the parent of this task.
  bool definedInParent(const Value *V) const {
    if (isa<Argument>(V)) return true;
    if (const Instruction *I = dyn_cast<Instruction>(V))
      return !encloses(I->getParent());
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

  // Add task ST as a subtask of this task.
  void addSubTask(Task *ST) {
    assert(!ST->ParentTask && "SubTask already has a parent task.");
    ST->setParentTask(this);
    SubTasks.push_back(ST);
  }

  // Set Spindle S to be the continuation spindle of this task.
  void setContinuationSpindle(Spindle *S) {
    assert(!isRootTask() && "Root task cannot have a continuation spindle.");
    Continuation = S;
  }

  // Set S to be the exceptional continuation spindle of this task.
  void setEHContinuationSpindle(Spindle *S, Value *LPadVal) {
    assert((!isRootTask() || getDetach()->hasUnwindDest()) &&
           "Task should not have an exceptional continuation.");
    EHContinuation = S;
    LPadValueInEHContinuation = LPadVal;
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

    for (auto *SharedEH : SharedSubTaskEH)
      SharedEH->~Spindle();

    SubTasks.clear();
    Spindles.clear();
    SharedSubTaskEH.clear();
    DenseSpindleSet.clear();
    DenseEHSpindleSet.clear();
    ParentTask = nullptr;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const Task &T);

//===--------------------------------------------------------------------===//
// GraphTraits specializations for task spindle graphs
//===--------------------------------------------------------------------===//

// Allow clients to walk the list of nested tasks.
template <> struct GraphTraits<const Task *> {
  using NodeRef = const Task *;
  using ChildIteratorType = Task::const_iterator;

  static NodeRef getEntryNode(const Task *T) { return T; }
  static ChildIteratorType child_begin(NodeRef N) { return N->begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->end(); }
};

template <> struct GraphTraits<Task *> {
  using NodeRef = Task *;
  using ChildIteratorType = Task::iterator;

  static NodeRef getEntryNode(Task *T) { return T; }
  static ChildIteratorType child_begin(NodeRef N) { return N->begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->end(); }
};

// Filter for spindle successors in the same task or a subtask.
class UnderTaskFilter {
  const Spindle *S = nullptr;
public:
  UnderTaskFilter() {}
  UnderTaskFilter(const Spindle *S) : S(S) {}
  bool operator()(const Spindle *Succ) const {
    return S->succInSameTask(Succ) ||
      (Succ->getParentTask()->getParentTask() == S->getParentTask());
  }
};

// Wrapper to allow traversal of only those spindles within a task, including
// all subtasks of that task.
template <typename SpindlePtrT>
struct UnderTask
  : public FilteredSuccessorSpindles<SpindlePtrT,
                                     UnderTaskFilter> {
  inline UnderTask(SpindlePtrT S)
      : FilteredSuccessorSpindles<SpindlePtrT, UnderTaskFilter>
      (S, UnderTaskFilter(S)) {}
};

template<> struct GraphTraits<UnderTask<Spindle *>> :
    public GraphTraits<FilteredSuccessorSpindles<
                         Spindle *, UnderTaskFilter>> {
  using NodeRef = Spindle *;
  static NodeRef getEntryNode(UnderTask<Spindle *> G) {
    return G.first;
  }
};
template<> struct GraphTraits<UnderTask<const Spindle *>> :
    public GraphTraits<FilteredSuccessorSpindles<
                         const Spindle *, UnderTaskFilter>> {
  using NodeRef = const Spindle *;
  static NodeRef getEntryNode(UnderTask<const Spindle *> G) {
    return G.first;
  }
};

// Structure to record the synced state of each spindle.
struct IsSyncedState {
  enum class SyncInfo {
    Unsynced = 0,
    Synced = 1,
    TaskEntry = 2,
    NoUnsync = Synced | TaskEntry,
    Incomplete = 4,
  };

  static inline bool isUnsynced(const SyncInfo SyncI) {
    return (static_cast<int>(SyncI) & static_cast<int>(SyncInfo::NoUnsync)) ==
      static_cast<int>(SyncInfo::Unsynced);
  }
  static inline bool isSynced(const SyncInfo SyncI) {
    return !isUnsynced(SyncI);
  }
  static inline bool isIncomplete(const SyncInfo SyncI) {
    return (static_cast<int>(SyncI) & static_cast<int>(SyncInfo::Incomplete)) ==
      static_cast<int>(SyncInfo::Incomplete);
  }
  static inline SyncInfo setUnsynced(const SyncInfo SyncI) {
    // Once a sync state is set to unsynced, it's complete.
    return SyncInfo(static_cast<int>(SyncI) &
                    static_cast<int>(SyncInfo::Unsynced));
  }
  static inline SyncInfo setIncomplete(const SyncInfo SyncI) {
    return SyncInfo(static_cast<int>(SyncI) |
                    static_cast<int>(SyncInfo::Incomplete));
  }
  static inline SyncInfo setComplete(const SyncInfo SyncI) {
    return SyncInfo(static_cast<int>(SyncI) &
                    ~static_cast<int>(SyncInfo::Incomplete));
  }

  DenseMap<const Spindle *, SyncInfo> SyncedState;

  bool markDefiningSpindle(const Spindle *S);
  bool evaluate(const Spindle *S, unsigned EvalNum);
};

using MPTaskListTy = DenseMap<const Spindle *, SmallPtrSet<const Task *, 2>>;

// Structure to record the set of child tasks that might be in parallel with
// this spindle.
struct MaybeParallelTasks {
  MPTaskListTy TaskList;

  // TODO: Use a bitvector representation to perform the analysis.

  bool markDefiningSpindle(const Spindle *S);
  bool evaluate(const Spindle *S, unsigned EvalNum);
};

//===----------------------------------------------------------------------===//
/// This class builds and contains all of the top-level task structures in the
/// specified function.
///
class TaskInfo {
  // BBMap - Mapping of basic blocks to the innermost spindle they occur in
  DenseMap<const BasicBlock *, Spindle *> BBMap;
  // SpindleMap - Mapping of spindles to the innermost task they occur in
  DenseMap<const Spindle *, Task *> SpindleMap;
  // Pointer to the root task for the function.  All tasks detached within this
  // function body are descendants of this root task.
  Task *RootTask = nullptr;

  // Cache storing maybe-parallel-task state.  This cache is initialized lazily
  // by calls to the mayHappenInParallel method.
  mutable std::unique_ptr<MaybeParallelTasks> MPTasks;

  BumpPtrAllocator TaskAllocator;

  void operator=(const TaskInfo &) = delete;
  TaskInfo(const TaskInfo &) = delete;

public:
  TaskInfo() {}
  ~TaskInfo() { releaseMemory(); }

  TaskInfo(TaskInfo &&Arg)
      : BBMap(std::move(Arg.BBMap)),
        SpindleMap(std::move(Arg.SpindleMap)),
        RootTask(std::move(Arg.RootTask)),
        MPTasks(std::move(Arg.MPTasks)),
        TaskAllocator(std::move(Arg.TaskAllocator)) {
    Arg.RootTask = nullptr;
  }
  TaskInfo &operator=(TaskInfo &&RHS) {
    BBMap = std::move(RHS.BBMap);
    SpindleMap = std::move(RHS.SpindleMap);
    if (RootTask)
      RootTask->~Task();
    RootTask = std::move(RHS.RootTask);
    MPTasks = std::move(RHS.MPTasks);
    TaskAllocator = std::move(RHS.TaskAllocator);
    RHS.RootTask = nullptr;
    return *this;
  }

  void releaseMemory() {
    BBMap.clear();
    SpindleMap.clear();
    if (RootTask)
      RootTask->~Task();
    RootTask = nullptr;
    if (MPTasks) {
      MPTasks->TaskList.clear();
      MPTasks.release();
    }
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

  Task *getRootTask() const { return RootTask; }

  /// Return true if this function is "serial," meaning it does not itself
  /// perform a detach.  This method does not preclude functions called by this
  /// function from performing a detach.
  bool isSerial() const {
    assert(getRootTask() && "Null root task\n");
    return getRootTask()->isSerial();
  }

  /// iterator/begin/end - The interface to the top-level tasks in the current
  /// function.
  ///
  using iterator = typename Task::iterator;
  using const_iterator = typename Task::const_iterator;
  using reverse_iterator = typename Task::reverse_iterator;
  using const_reverse_iterator = typename Task::const_reverse_iterator;
  inline iterator begin() const { return getRootTask()->begin(); }
  inline iterator end() const { return getRootTask()->end(); }
  inline reverse_iterator rbegin() const { return getRootTask()->rbegin(); }
  inline reverse_iterator rend() const { return getRootTask()->rend(); }
  inline bool empty() const { return getRootTask()->empty(); }

  /// Return the innermost spindle that BB lives in.
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
    return getTaskFor(
        getRootTask()->DomTree.findNearestCommonDominator(BB1, BB2));
  }

  /// Return the innermost task that encompases both spindles S1 and S2.
  Task *getEnclosingTask(const Spindle *S1, const Spindle *S2) const {
    return getEnclosingTask(S1->getEntry(), S2->getEntry());
  }

  /// Return true if task T1 encloses task T2.
  bool encloses(const Task *T1, const Task *T2) const {
    if (!T1 || !T2) return false;
    return getRootTask()->DomTree.dominates(T1->getEntry(), T2->getEntry());
  }

  /// Return true if task T encloses basic block BB.
  bool encloses(const Task *T, const BasicBlock *BB) const {
    if (!T) return false;
    return T->encloses(BB);
  }

  /// Return true if the task T encloses instruction Inst.
  bool encloses(const Task *T, const Instruction *Inst) const {
    return encloses(T, Inst->getParent());
  }

  /// Return the task nesting level of basic block BB. A depth of 0 means the
  /// block is in the root task.
  unsigned getTaskDepth(const BasicBlock *BB) const {
    return getTaskFor(BB)->getTaskDepth();
  }

  /// True if basic block BB is a task entry block
  bool isTaskEntry(const BasicBlock *BB) const {
    return getTaskFor(BB)->getEntry() == BB;
  }

  /// Traverse the graph of spindles to evaluate some parallel state.
  template<typename StateT>
  void evaluateParallelState(StateT &State) const {
    SetVector<Spindle *> ToProcess;

    // This method performs the work-list algorithm for data-flow analysis on
    // spindles.

    // First mark all defining spindles and spindles whose state is eagerly
    // updated.
    {
      // Get the spindles in post order, so we can traverse them in RPO.
      SmallVector<Spindle *, 16> POSpindles;
      for (Spindle *S : post_order(getRootTask()->getEntrySpindle()))
        POSpindles.push_back(S);
      // SetVector<Spindle *> DefSpindles;
      for (Spindle *S : llvm::reverse(POSpindles))
        // If we find a defining spindle (or a spindle with an eagerly-updated
        // state), add its successors for processing.
        if (State.markDefiningSpindle(S))
          for (Spindle *Succ : successors(S))
            ToProcess.insert(Succ);
    }

    // Perform the work-list algorithm to propagate data-flow information among
    // the spindles.
    {
      SmallVector<Spindle *, 8> NextToProcess;
      unsigned EvalNum = 0;
      while (!ToProcess.empty()) {
        // Process all spindles that need processing.
        for (Spindle *Curr : ToProcess)
          if (!State.evaluate(Curr, EvalNum))
            // If the state of this spindle changed, add its successors for
            // future processing.
            for (Spindle *Succ : successors(Curr))
              NextToProcess.push_back(Succ);

        // Get ready to Process the next set of spindles.
        ToProcess.clear();
        ToProcess.insert(NextToProcess.begin(), NextToProcess.end());
        NextToProcess.clear();
        ++EvalNum;
      }
    }
  }

  /// Check if a alloca AI is promotable based on task structure.
  bool isAllocaParallelPromotable(const AllocaInst *AI) const;

  /// Check if the two basic blocks B1 and B2 may execute in parallel.
  bool mayHappenInParallel(const BasicBlock *B1, const BasicBlock *B2) const {
    // Common case: No blocks execute in parallel in a serial function.
    if (isSerial())
      return false;

    // if (getTaskFor(B1) == getTaskFor(B2))
    //   return false;

    // If necessary, compute which tasks may execute in parallel.
    if (!MPTasks) {
      MPTasks.reset(new MaybeParallelTasks());
      evaluateParallelState<MaybeParallelTasks>(*MPTasks);
    }

    // Get the task Encl that encloses both basic blocks.
    const Task *Encl = getEnclosingTask(B1, B2);

    // For each basic block, get the representative subtask of Encl that
    // encloses that basic block.
    const Task *B1Task = Encl->getSubTaskEnclosing(B1);
    const Task *B2Task = Encl->getSubTaskEnclosing(B2);

    // Translate these representative tasks into spindles.
    const Spindle *B1Spindle = getSpindleFor(B1);
    const Spindle *B2Spindle = getSpindleFor(B2);
    if (B1Task != Encl)
      B1Spindle = getSpindleFor(B1Task->getDetach()->getParent());
    if (B2Task != Encl)
      B2Spindle = getSpindleFor(B2Task->getDetach()->getParent());

    // Evaluate the maybe-parallel task lists for the two representative
    // spindles to determine if the blocks may execute in parallel.
    return MPTasks->TaskList[B1Spindle].count(B2Task) ||
      MPTasks->TaskList[B2Spindle].count(B1Task);
  }

  /// Create the task forest using a stable algorithm.
  void analyze(Function &F, DominatorTree &DomTree);

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
    assert(T && "Cannot destroy a null task.");
    T->~Task();

    // Since TaskAllocator is a BumpPtrAllocator, this Deallocate only poisons
    // \c T, but the pointer remains valid for non-dereferencing uses.
    TaskAllocator.Deallocate(T);
  }

  // Manually recalculate TaskInfo from the given dominator tree.
  void recalculate(Function &F, DominatorTree &DomTree) {
    releaseMemory();
    analyze(F, DomTree);
  }

  // Create a spindle with entry block B and type Ty.
  Spindle *createSpindleWithEntry(BasicBlock *B, Spindle::SPType Ty) {
    Spindle *S = AllocateSpindle(B, Ty);
    assert(!BBMap.count(B) && "BasicBlock already in a spindle!");
    BBMap[B] = S;
    return S;
  }

  // Create a task with spindle entry S.
  Task *createTaskWithEntry(Spindle *S, DominatorTree &DomTree) {
    Task *T = AllocateTask(*S, DomTree);
    S->setParentTask(T);
    assert(!SpindleMap.count(S) && "Spindle already in a task!");
    SpindleMap[S] = T;
    return T;
  }

  // Add spindle S to task T.
  void addSpindleToTask(Spindle *S, Task *T) {
    assert(!SpindleMap.count(S) && "Spindle already mapped to a task.");
    T->addSpindle(*S);
    S->setParentTask(T);
    SpindleMap[S] = T;
  }

  // Add spindle S to task T, where S is a shared exception-handling spindle
  // among subtasks of T.
  void addEHSpindleToTask(Spindle *S, Task *T) {
    assert(!SpindleMap.count(S) && "Spindle already mapped to a task.");
    T->addEHSpindle(*S);
    S->setParentTask(T);
    SpindleMap[S] = T;
  }

  // Add basic block B to spindle S.
  void addBlockToSpindle(BasicBlock &B, Spindle *S) {
    assert(!BBMap.count(&B) && "Block already mapped to a spindle.");
    S->addBlock(B);
    BBMap[&B] = S;
  }
};

/// Analysis pass that exposes the \c TaskInfo for a function.
class TaskAnalysis : public AnalysisInfoMixin<TaskAnalysis> {
  friend AnalysisInfoMixin<TaskAnalysis>;
  static AnalysisKey Key;

public:
  using Result = TaskInfo;

  TaskInfo run(Function &F, FunctionAnalysisManager &AM);
};

/// Printer pass for the \c TaskAnalysis results.
class TaskPrinterPass : public PassInfoMixin<TaskPrinterPass> {
  raw_ostream &OS;

public:
  explicit TaskPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// Verifier pass for the \c TaskAnalysis results.
struct TaskVerifierPass : public PassInfoMixin<TaskVerifierPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// The legacy pass manager's analysis pass to compute task information.
class TaskInfoWrapperPass : public FunctionPass {
  TaskInfo TI;

public:
  static char ID; // Pass identification, replacement for typeid

  TaskInfoWrapperPass() : FunctionPass(ID) {
    initializeTaskInfoWrapperPassPass(*PassRegistry::getPassRegistry());
  }

  TaskInfo &getTaskInfo() { return TI; }
  const TaskInfo &getTaskInfo() const { return TI; }

  /// Calculate the natural task information for a given function.
  bool runOnFunction(Function &F) override;

  void verifyAnalysis() const override;

  void releaseMemory() override { TI.releaseMemory(); }

  void print(raw_ostream &O, const Module *M = nullptr) const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

/// Function to print a task's contents as LLVM's text IR assembly.
void printTask(Task &T, raw_ostream &OS, const std::string &Banner = "");

/// Examine a given loop to determine if it is structurally a Tapir loop.
/// Returns the Task that encodes the loop body if so, or nullptr if not.
Task *getTaskIfTapirLoopStructure(const Loop *L, TaskInfo *TI);

} // End llvm namespace

#endif
