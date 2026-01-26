//===- Semilattice.cpp - A semilattice of KnownBits -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Semilattice.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using NodeT = SemilatticeNode;

static bool isSupportedValue(Value *V) {
  // Pointer types are useful for inferring alignment.
  return V->getType()->isIntOrIntVectorTy() ||
         V->getType()->isPtrOrPtrVectorTy();
}

SemilatticeNode::SemilatticeNode(Value *V, const DataLayout &DL)
    : ValHasKnownBits(V, 0),
      Known(DL.getTypeSizeInBits(V->getType()->getScalarType())) {
  assert(V && "Attempting to create a node with an empty Value");
}

void Semilattice::initialize(ArrayRef<Value *> Roots, const DataLayout &DL) {
  auto ToInsert = make_filter_range(
      Roots, [&](Value *V) { return !contains(V) && isSupportedValue(V); });
  for (Value *V : ToInsert)
    recurseInsertChildren(insert(V, DL), DL);
}

void Semilattice::initialize(Function &F) {
  SmallVector<Value *> Args(llvm::make_pointer_range(F.args()));
  initialize(Args, F.getDataLayout());
  for (BasicBlock &BB : F) {
    SmallVector<Value *> Args(llvm::make_pointer_range(BB));
    initialize(Args, F.getDataLayout());
  }
}

Semilattice::Semilattice(Function &F) : RootNode(create()) { initialize(F); }

NodeT *Semilattice::insert(Value *V, const DataLayout &DL, NodeT *Parent) {
  assert(isSupportedValue(V) && "Cannot insert non-integral non-pointer types");
  NodeT *Node = getOrCreate(V, DL);
  NodeMap.try_emplace(V, Node);
  NodeT *ParentNode = Parent ? Parent : RootNode;
  ParentNode->addChild(Node);
  return Node->addParent(ParentNode);
}

SmallVector<NodeT *, 4> Semilattice::insert_range(NodeT *Parent,
                                                  ArrayRef<User *> R,
                                                  const DataLayout &DL) {
  SmallVector<NodeT *, 4> Ret;
  auto Users = make_filter_range(R, [&](User *U) {
    return (!contains(U) || !lookup(U)->hasParent(Parent)) &&
           isSupportedValue(U);
  });
  for (User *U : Users)
    Ret.push_back(insert(U, DL, Parent));
  return Ret;
}

void Semilattice::recurseInsertChildren(ArrayRef<NodeT *> Parents,
                                        const DataLayout &DL) {
  for (NodeT *P : Parents)
    recurseInsertChildren(
        insert_range(P, to_vector(P->getValue()->users()), DL), DL);
}

SmallVector<NodeT *> Semilattice::invalidateKnownBits(Value *V) {
  if (!contains(V))
    return {};
  SetVector<NodeT *> ToUpdate;
  for (NodeT *N : breadth_first(lookup(V))) {
    N->resetKnownBits();
    N->setHasKnownBits();
    ToUpdate.insert(N);
  }
  SmallVector<NodeT *> Ret(reverse(ToUpdate.takeVector()));
  return Ret;
}

SmallVector<NodeT *> Semilattice::rauw(Value *OldV, Value *NewV) {
  if (!contains(OldV))
    return {};
  NodeMap.emplace_or_assign(NewV, lookup(OldV)->rauw(NewV));
  NodeMap.erase(OldV);
  return invalidateKnownBits(NewV);
}

void Semilattice::print(raw_ostream &OS) const {
  for (NodeT *N : drop_begin(depth_first(RootNode))) {
    if (N->hasParent(RootNode))
      OS << "^ ";
    else if (N->isLeaf())
      OS << "$ ";
    else
      OS << "  ";
    N->getValue()->print(OS);
    if (N->hasKnownBits()) {
      OS << " | ";
      N->Known.print(OS);
    }
    OS << "\n";
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void Semilattice::dump() const { print(dbgs()); }
#endif
