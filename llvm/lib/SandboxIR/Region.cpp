//===- Region.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Region.h"

namespace llvm::sandboxir {

Region::Region(Context &Ctx) : Ctx(Ctx) {
  LLVMContext &LLVMCtx = Ctx.LLVMCtx;
  auto *RegionStrMD = MDString::get(LLVMCtx, RegionStr);
  RegionMDN = MDNode::getDistinct(LLVMCtx, {RegionStrMD});
}

Region::~Region() {}

void Region::add(Instruction *I) {
  Insts.insert(I);
  // TODO: Consider tagging instructions lazily.
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
