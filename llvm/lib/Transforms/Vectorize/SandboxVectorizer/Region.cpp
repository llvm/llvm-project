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

  LLVMContext &LLVMCtx = Ctx.LLVMCtx;
  auto *RegionStrMD = MDString::get(LLVMCtx, RegionStr);
  auto *RegionIDMD = llvm::ConstantAsMetadata::get(
      llvm::ConstantInt::get(LLVMCtx, APInt(sizeof(RegionID) * 8, RegionID)));
  RegionMDN = MDNode::get(LLVMCtx, {RegionStrMD, RegionIDMD});
}

Region::~Region() {}

void Region::add(Instruction *I) {
  Insts.insert(I);
  cast<llvm::Instruction>(I->Val)->setMetadata(MDKind, RegionMDN);
}

void Region::remove(Instruction *I) {
  Insts.remove(I);
  cast<llvm::Instruction>(I->Val)->setMetadata(MDKind, nullptr);
}

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

SmallVector<std::unique_ptr<Region>> Region::createRegionsFromMD(Function &F) {
  SmallVector<std::unique_ptr<Region>> Regions;
  DenseMap<MDNode *, Region *> MDNToRegion;
  auto &Ctx = F.getContext();
  for (BasicBlock &BB : F) {
    for (Instruction &Inst : BB) {
      if (auto *MDN = cast<llvm::Instruction>(Inst.Val)->getMetadata(MDKind)) {
        Region *R = nullptr;
        auto It = MDNToRegion.find(MDN);
        if (It == MDNToRegion.end()) {
          Regions.push_back(std::make_unique<Region>(Ctx));
          R = Regions.back().get();
          MDNToRegion[MDN] = R;
        } else {
          R = It->second;
        }
        R->add(&Inst);
      }
    }
  }
  return Regions;
}

} // namespace llvm::sandboxir
