//===- Region.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Region.h"

namespace llvm::sandboxir {

Region::Region(Context &Ctx) : Ctx(Ctx) {
  static unsigned StaticRegionID;
  RegionID = StaticRegionID++;
}

Region::~Region() {}

void Region::add(Instruction *I) { Insts.insert(I); }

void Region::remove(Instruction *I) { Insts.remove(I); }

#ifndef NDEBUG
bool Region::operator==(const Region &Other) const {
  if (Insts.size() != Other.Insts.size())
    return false;
  if (!std::is_permutation(Insts.begin(), Insts.end(), Other.Insts.begin()))
    return false;
  return true;
}

void Region::dump(raw_ostream &OS) const {
  OS << "RegionID: " << getID() << "\n";
  for (auto *I : Insts)
    OS << *I << "\n";
}

void Region::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

} // namespace llvm::sandboxir
