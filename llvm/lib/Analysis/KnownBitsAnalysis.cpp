//===- KnownBitsAnalysis.cpp - An analysis that caches KnownBits ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/KnownBitsAnalysis.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"

using namespace llvm;

template <typename NodeRef, typename ChildIteratorType>
struct NodeGraphTraitsBase {
  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) { // NOLINT
    return N->user_begin();
  }
  static ChildIteratorType child_end(NodeRef N) { // NOLINT
    return N->user_end();
  }
};

template <>
struct GraphTraits<Value *>
    : public NodeGraphTraitsBase<Value *, Value::user_iterator> {
  using NodeRef = Value *;
  using ChildIteratorType = Value::user_iterator;
};

template <>
struct GraphTraits<const Value *>
    : public NodeGraphTraitsBase<const Value *, Value::const_user_iterator> {
  using NodeRef = const Value *;
  using ChildIteratorType = Value::const_user_iterator;
};

/// A wrapper around make_filter_range, that filters \p R for integer, pointer,
/// or vector thereof, types, excluding values that match \p ExcludeFn.
template <typename RangeT>
static auto filter_range( // NOLINT
    RangeT R, std::function<bool(const Value *)> ExcludeFn = [](const Value *) {
      return false;
    }) {
  return make_filter_range(R, [&](const Value *V) {
    return !ExcludeFn(V) && V->getType()->getScalarType()->isIntOrPtrTy();
  });
}

static bool isLeaf(const Value *V) { return filter_range(V->users()).empty(); }

ArrayRef<const Value *> KnownBitsDataflow::getRoots() const { return Roots; }

SmallVector<const Value *>
KnownBitsDataflow::getLeaves(ArrayRef<const Value *> Roots) const {
  SetVector<const Value *> Leaves;
  for (const Value *R : Roots)
    for (const Value *N : filter_range(depth_first(R)))
      if (isLeaf(N))
        Leaves.insert(N);
  return Leaves.takeVector();
}

SmallVector<const Value *> KnownBitsDataflow::getLeaves() const {
  return getLeaves(getRoots());
}

void KnownBitsDataflow::emplace_all_conflict(const Value *V) {
  emplace_or_assign(V, KnownBits::getAllConflict(DL.getTypeSizeInBits(
                           V->getType()->getScalarType())));
}

template <typename RangeT>
SmallVector<Value *> KnownBitsDataflow::insert_range(RangeT R) {
  SmallVector<Value *> Filtered(
      filter_range(R, bind_front(&KnownBitsDataflow::contains, this)));
  for (Value *V : Filtered)
    emplace_all_conflict(V);
  return Filtered;
}

void KnownBitsDataflow::recurseInsertChildren(ArrayRef<Value *> R) {
  for (Value *V : R)
    recurseInsertChildren(insert_range(
        map_range(V->users(), [](User *U) -> Value * { return U; })));
}

template <typename RangeT> void KnownBitsDataflow::initialize(RangeT R) {
  for (auto *V : R) {
    emplace_all_conflict(V);
    recurseInsertChildren(V);
  }
  append_range(Roots, R);
}

void KnownBitsDataflow::initialize(Function &F) {
  // First, initialize with function arguments.
  initialize(filter_range(llvm::make_pointer_range(F.args())));
  for (BasicBlock &BB : F) {
    // Now initialize with all Instructions in the BB that weren't seen.
    initialize(filter_range(llvm::make_pointer_range(BB),
                            bind_front(&KnownBitsDataflow::contains, this)));
  }
}

KnownBitsDataflow::KnownBitsDataflow(Function &F) : DL(F.getDataLayout()) {
  initialize(F);
}

SmallVector<const Value *> KnownBitsDataflow::invalidate(const Value *V) {
  SmallVector<const Value *> Leaves;
  for (const Value *N : filter_range(depth_first(V))) {
    setAllConflict(N);
    if (isLeaf(N))
      Leaves.push_back(N);
  }
  return Leaves;
}

void KnownBitsDataflow::print(raw_ostream &OS) const {
  for (const Value *R : getRoots()) {
    OS << "^ ";
    R->print(OS);
    OS << " | ";
    at(R).print(OS);
    OS << "\n";
    for (const Value *V : filter_range(drop_begin(depth_first(R)))) {
      if (isLeaf(V))
        OS << "$ ";
      else
        OS << "  ";
      V->print(OS);
      OS << " | ";
      at(V).print(OS);
      OS << "\n";
    }
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void KnownBitsDataflow::dump() const { print(dbgs()); }
#endif
