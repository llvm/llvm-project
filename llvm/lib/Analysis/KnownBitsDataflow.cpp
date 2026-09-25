//===- KnownBitsDataflow.cpp - Cache and invalidate KnownBits -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/KnownBitsDataflow.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "known-bits-dataflow"

// Pin the vtable.
void KnownBitsVH::anchor() {}

void KnownBitsVH::deleted() {
  // This is called in the destructor of ValueHandleBase. Carefully avoid
  // constructing a new ValueHandle, and avoid calling virtual functions.
  [[maybe_unused]] bool Removed = KBD->remove_if(
      [&](const auto &It) { return It.first.getValPtr() == getValPtr(); });
  assert(Removed && "Expected to find ValPtr in map");
  clearValPtr();
}

void KnownBitsVH::allUsesReplacedWith(Value *New) {
  // This is called in ValueHandleBase before any uses are replaced.
  KBD->invalidate(*this);
  setValPtr(New);
}

/// A wrapper around make_filter_range, that filters \p R on scalar types that
/// are either integer or pointer type, as these are the only types handled by
/// computeKnownBits.
template <typename RangeT>
static auto make_knownbits_range(RangeT &&R) { // NOLINT
  return make_filter_range(R, [](const auto &V) {
    return V->getType()->getScalarType()->isIntOrPtrTy();
  });
}

unsigned KnownBitsDataflow::getBitWidth(Type *Ty, const DataLayout &DL) {
  if (unsigned BitWidth = Ty->getScalarSizeInBits())
    return BitWidth;
  return DL.getPointerTypeSizeInBits(Ty);
}

LLVM_ABI_FOR_TEST SmallVector<const Value *>
KnownBitsDataflow::forwardDataflow(ArrayRef<KnownBitsVH> Roots) const {
  SetVector<const Value *> Collected;
  for (const KnownBitsVH &V : Roots)
    Collected.insert_range(forwardDataflow(V));
  return Collected.takeVector();
}

SmallVector<KnownBitsVH>
KnownBitsDataflow::computeRoots(const Function &F) const {
  SmallVector<KnownBitsVH> Roots;

  // First, collect function arguments.
  for (const Value *V : make_knownbits_range(make_pointer_range(F.args())))
    if (contains(V))
      Roots.emplace_back(key_as(V));

  // A helper to find out whether a Value is reachable from Roots that computes
  // the reachability information just in time, as Roots are updated.
  auto IsReachableFromRoots = [&](const Value *V) {
    for (const KnownBitsVH &R : Roots)
      for (const Value *N : make_knownbits_range(depth_first(R.getValPtr())))
        if (N == V)
          return true;
    return false;
  };

  // Now collect all Instructions that aren't reachable from the function's
  // arguments, updating Roots, as we test for unreachability.
  for (const BasicBlock &BB : F)
    for (const Value *V : make_knownbits_range(make_pointer_range(BB)))
      if (!IsReachableFromRoots(V) && contains(V))
        Roots.emplace_back(key_as(V));

  return Roots;
}

void KnownBitsDataflow::print(const Function &F, raw_ostream &OS) const {
  auto IsLeaf = [](const Value *V) {
    return make_knownbits_range(V->users()).empty();
  };
  SmallVector<KnownBitsVH> Roots = computeRoots(F);
  for (const Value *V : forwardDataflow(Roots)) {
    if (is_contained(Roots, V))
      OS << "^ ";
    else if (IsLeaf(V))
      OS << "$ ";
    else
      OS << "  ";
    V->print(OS);
    OS << " | ";
    value_as(V).print(OS);
    OS << "\n";
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void KnownBitsDataflow::dump(const Function &F) const {
  print(F, dbgs());
}
#endif

bool KnownBitsDataflow::invalidate(Function &, const PreservedAnalyses &PA,
                                   FunctionAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<KnownBitsDataflowAnalysis>();
  return !PAC.preserved();
}

AnalysisKey KnownBitsDataflowAnalysis::Key;

KnownBitsDataflow KnownBitsDataflowAnalysis::run(Function &F,
                                                 FunctionAnalysisManager &) {
  return {};
}

// Legacy PM wrapper pass.
char KnownBitsDataflowAnalysisWrapperPass::ID = 0;

KnownBitsDataflowAnalysisWrapperPass::KnownBitsDataflowAnalysisWrapperPass()
    : FunctionPass(ID) {}

void KnownBitsDataflowAnalysisWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool KnownBitsDataflowAnalysisWrapperPass::runOnFunction(Function &F) {
  Result.reset(new KnownBitsDataflow());
  return false;
}

INITIALIZE_PASS(KnownBitsDataflowAnalysisWrapperPass, "known-bits-dataflow",
                "KnownBits Dataflow", false, true)
