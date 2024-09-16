//===- Region.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Region.h"

using namespace llvm;

sandboxir::Region::Region(sandboxir::Context &Ctx, sandboxir::BasicBlock &SBBB)
    : Ctx(Ctx), SBBB(SBBB) {
  static unsigned StaticRegionID;
  RegionID = StaticRegionID++;
}

sandboxir::Region::~Region() {}

void sandboxir::Region::add(sandboxir::Instruction *SBI) { Insts.insert(SBI); }

void sandboxir::Region::remove(sandboxir::Instruction *SBI) {
  Insts.remove(SBI);
}

#ifndef NDEBUG
bool sandboxir::Region::operator==(const sandboxir::Region &Other) const {
  if (Insts.size() != Other.Insts.size())
    return false;
  if (!std::is_permutation(Insts.begin(), Insts.end(), Other.Insts.begin()))
    return false;
  return true;
}

void sandboxir::Region::dump(raw_ostream &OS) const {
  OS << "RegionID: " << getID() << " ScalarCost=" << ScalarCost
     << " VectorCost=" << VectorCost << "\n";
  for (auto *I : Insts)
    OS << *I << "\n";
}

void sandboxir::Region::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

#endif // NDEBUG
