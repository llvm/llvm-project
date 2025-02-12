//===- Region.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Region.h"
#include "llvm/SandboxIR/Function.h"

namespace llvm::sandboxir {

InstructionCost ScoreBoard::getCost(Instruction *I) const {
  auto *LLVMI = cast<llvm::Instruction>(I->Val);
  SmallVector<const llvm::Value *> Operands(LLVMI->operands());
  return TTI.getInstructionCost(LLVMI, Operands, CostKind);
}

void ScoreBoard::remove(Instruction *I) {
  auto Cost = getCost(I);
  if (Rgn.contains(I))
    // If `I` is one the newly added ones, then we should adjust `AfterCost`
    AfterCost -= Cost;
  else
    // If `I` is one of the original instructions (outside the region) then it
    // is part of the original code, so adjust `BeforeCost`.
    BeforeCost += Cost;
}

#ifndef NDEBUG
void ScoreBoard::dump() const { dump(dbgs()); }
#endif

Region::Region(Context &Ctx, TargetTransformInfo &TTI)
    : Ctx(Ctx), Scoreboard(*this, TTI) {
  LLVMContext &LLVMCtx = Ctx.LLVMCtx;
  auto *RegionStrMD = MDString::get(LLVMCtx, RegionStr);
  RegionMDN = MDNode::getDistinct(LLVMCtx, {RegionStrMD});

  CreateInstCB = Ctx.registerCreateInstrCallback(
      [this](Instruction *NewInst) { add(NewInst); });
  EraseInstCB = Ctx.registerEraseInstrCallback(
      [this](Instruction *ErasedInst) { remove(ErasedInst); });
}

Region::~Region() {
  Ctx.unregisterCreateInstrCallback(CreateInstCB);
  Ctx.unregisterEraseInstrCallback(EraseInstCB);
}

void Region::add(Instruction *I) {
  Insts.insert(I);
  // TODO: Consider tagging instructions lazily.
  cast<llvm::Instruction>(I->Val)->setMetadata(MDKind, RegionMDN);
  // Keep track of the instruction cost.
  Scoreboard.add(I);
}

void Region::remove(Instruction *I) {
  // Keep track of the instruction cost. This need to be done *before* we remove
  // `I` from the region.
  Scoreboard.remove(I);

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

SmallVector<std::unique_ptr<Region>>
Region::createRegionsFromMD(Function &F, TargetTransformInfo &TTI) {
  SmallVector<std::unique_ptr<Region>> Regions;
  DenseMap<MDNode *, Region *> MDNToRegion;
  auto &Ctx = F.getContext();
  for (BasicBlock &BB : F) {
    for (Instruction &Inst : BB) {
      if (auto *MDN = cast<llvm::Instruction>(Inst.Val)->getMetadata(MDKind)) {
        auto [It, Inserted] = MDNToRegion.try_emplace(MDN);
        if (Inserted) {
          Regions.push_back(std::make_unique<Region>(Ctx, TTI));
          It->second = Regions.back().get();
        }
        It->second->add(&Inst);
      }
    }
  }
  return Regions;
}

} // namespace llvm::sandboxir
