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
#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrMaps.h"

namespace llvm::sandboxir {

SmallVector<Value *, 4> VecUtils::getNextUserBundle(ArrayRef<Value *> Bndl,
                                                    User *U0, Value *V0,
                                                    InstrMaps &IMaps) {
  auto *UI0 = dyn_cast<Instruction>(U0);
  if (!UI0 || IMaps.isVectorized(UI0))
    return {};

  // Find the operand index at which U0 uses lane 0.
  unsigned OpIdx = UI0->getNumOperands();
  for (unsigned Idx : seq<unsigned>(UI0->getNumOperands())) {
    if (UI0->getOperand(Idx) == V0) {
      OpIdx = Idx;
      break;
    }
  }
  if (OpIdx == UI0->getNumOperands())
    return {};

  // Find a distinct matching user for each of the remaining lanes.
  SmallVector<Value *, 4> NextUserBndl;
  NextUserBndl.push_back(UI0);
  SmallPtrSet<Instruction *, 4> Claimed;
  Claimed.insert(UI0);
  for (Value *V : drop_begin(Bndl)) {
    Instruction *Match = nullptr;
    for (User *U : V->users()) {
      auto *UI = dyn_cast<Instruction>(U);
      if (!UI || IMaps.isVectorized(UI) || Claimed.contains(UI))
        continue;
      if (UI->getOpcode() != UI0->getOpcode() ||
          UI->getType() != UI0->getType())
        continue;
      // The whole bundle must live in the same block.
      if (UI->getParent() != UI0->getParent())
        continue;
      // The user must consume this lane at the same operand index.
      if (OpIdx >= UI->getNumOperands() || UI->getOperand(OpIdx) != V)
        continue;
      Match = UI;
      break;
    }
    if (!Match)
      return {};
    Claimed.insert(Match);
    NextUserBndl.push_back(Match);
  }
  return NextUserBndl;
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
