//===- VecUtils.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrMaps.h"

namespace llvm::sandboxir {

static cl::opt<unsigned> MaxUsersToConsider(
    "sbvec-max-users-to-consider", cl::init(16), cl::Hidden,
    cl::desc("Limit the number of a seed's users that getNextUserBundles() "
             "will examine as candidates for a matching bundle, to cap "
             "compilation time."));

static SmallVector<unsigned, 2> getOperandIndicesInUser(User *U, Value *Op) {
  SmallVector<unsigned, 2> OpIdxVec;
  for (unsigned Idx : seq<unsigned>(U->getNumOperands()))
    if (U->getOperand(Idx) == Op)
      OpIdxVec.push_back(Idx);
  return OpIdxVec;
}

static std::optional<BundleTy>
getMatchingBundle(ArrayRef<Value *> Bndl, const InstrMaps &IMaps, Value *Seed,
                  Instruction *SeedUserInst,
                  SmallPtrSet<Instruction *, 4> &Claimed) {
  SmallVector<unsigned, 2> OpIdxVec0 =
      getOperandIndicesInUser(SeedUserInst, Seed);
  assert(!OpIdxVec0.empty() && "U0 does not use Seed!");
  BundleTy NextUserBndl;
  NextUserBndl.push_back(SeedUserInst);
  Claimed.insert(SeedUserInst);
  for (Value *V : drop_begin(Bndl)) {
    Instruction *Match = nullptr;
    for (User *U : V->users()) {
      auto *UI = dyn_cast<Instruction>(U);
      if (!UI || IMaps.isVectorized(UI) || Claimed.contains(UI) ||
          UI->getOpcode() != SeedUserInst->getOpcode() ||
          UI->getType() != SeedUserInst->getType() ||
          UI->getParent() != SeedUserInst->getParent() ||
          getOperandIndicesInUser(UI, V) != OpIdxVec0)
        continue;

      Match = UI;
      break;
    }
    if (!Match)
      return std::nullopt;
    NextUserBndl.push_back(Match);
  }

  for (auto *I : NextUserBndl)
    Claimed.insert(cast<Instruction>(I));
  return NextUserBndl;
}

SmallVector<BundleTy>
VecUtils::getNextUserBundles(ArrayRef<Value *> Bndl, const InstrMaps &IMaps,
                             SmallPtrSet<Instruction *, 4> &Claimed) {
  SmallVector<BundleTy> Bundles;
  if (Bndl.empty())
    return Bundles;

  Value *V0 = Bndl[0];
  DenseSet<User *> SeenUsers;
  // For each user U0 of lane 0, try to form a bundle of matching users across
  // all lanes. Cap the number of users considered to bound compilation time,
  // since each one may trigger an O(Bndl.size()) search across the other
  // lanes' users.
  for (User *U0 : V0->users()) {
    if (SeenUsers.size() >= MaxUsersToConsider)
      break;
    if (!SeenUsers.insert(U0).second)
      continue;
    auto *UI0 = dyn_cast<Instruction>(U0);
    if (!UI0 || IMaps.isVectorized(UI0) || Claimed.contains(UI0))
      continue;
    std::optional<BundleTy> NextUserBndl =
        getMatchingBundle(Bndl, IMaps, V0, UI0, Claimed);
    if (NextUserBndl)
      Bundles.emplace_back(std::move(*NextUserBndl));
  }
  return Bundles;
}

unsigned VecUtils::getFloorPowerOf2(unsigned Num) {
  if (Num == 0)
    return Num;
  unsigned Mask = Num;
  Mask >>= 1;
  for (unsigned ShiftBy = 1; ShiftBy < sizeof(Num) * 8; ShiftBy <<= 1)
    Mask |= Mask >> ShiftBy;
  return Num & ~Mask;
}

#ifndef NDEBUG
template <typename T> static void dumpImpl(ArrayRef<T *> Bndl) {
  for (auto [Idx, V] : enumerate(Bndl))
    dbgs() << Idx << "." << *V << "\n";
}
void VecUtils::dump(ArrayRef<Value *> Bndl) { dumpImpl(Bndl); }
void VecUtils::dump(ArrayRef<Instruction *> Bndl) { dumpImpl(Bndl); }
#endif // NDEBUG

} // namespace llvm::sandboxir
