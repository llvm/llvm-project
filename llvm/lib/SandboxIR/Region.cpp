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

void Region::setAux(ArrayRef<Instruction *> Aux) {
  this->Aux = SmallVector<Instruction *>(Aux);
  auto &LLVMCtx = Ctx.LLVMCtx;
  for (auto [Idx, I] : enumerate(Aux)) {
    llvm::ConstantInt *IdxC =
        llvm::ConstantInt::get(LLVMCtx, llvm::APInt(32, Idx, false));
    assert(cast<llvm::Instruction>(I->Val)->getMetadata(AuxMDKind) == nullptr &&
           "Instruction already in Aux!");
    cast<llvm::Instruction>(I->Val)->setMetadata(
        AuxMDKind, MDNode::get(LLVMCtx, ConstantAsMetadata::get(IdxC)));
  }
}

void Region::setAux(unsigned Idx, Instruction *I) {
  assert((Idx >= Aux.size() || Aux[Idx] == nullptr) &&
         "There is already an Instruction at Idx in Aux!");
  unsigned ExpectedSz = Idx + 1;
  if (Aux.size() < ExpectedSz) {
    auto SzBefore = Aux.size();
    Aux.resize(ExpectedSz);
    // Initialize the gap with nullptr.
    for (unsigned Idx = SzBefore; Idx + 1 < ExpectedSz; ++Idx)
      Aux[Idx] = nullptr;
  }
  Aux[Idx] = I;
}

void Region::clearAux() {
  for (unsigned Idx : seq<unsigned>(0, Aux.size())) {
    auto *LLVMI = cast<llvm::Instruction>(Aux[Idx]->Val);
    LLVMI->setMetadata(AuxMDKind, nullptr);
  }
  Aux.clear();
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
  if (!Aux.empty()) {
    OS << "\nAux:\n";
    for (auto *I : Aux) {
      if (I == nullptr)
        OS << "NULL\n";
      else
        OS << *I << "\n";
    }
  }
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
      auto *LLVMI = cast<llvm::Instruction>(Inst.Val);
      if (auto *MDN = LLVMI->getMetadata(MDKind)) {
        Region *R = nullptr;
        auto [It, Inserted] = MDNToRegion.try_emplace(MDN);
        if (Inserted) {
          Regions.push_back(std::make_unique<Region>(Ctx, TTI));
          R = Regions.back().get();
          It->second = R;
        } else {
          R = It->second;
        }
        R->add(&Inst);

        if (auto *AuxMDN = LLVMI->getMetadata(AuxMDKind)) {
          llvm::Constant *IdxC =
              dyn_cast<ConstantAsMetadata>(AuxMDN->getOperand(0))->getValue();
          auto Idx = cast<llvm::ConstantInt>(IdxC)->getSExtValue();
          R->setAux(Idx, &Inst);
        }
      }
    }
  }
#ifndef NDEBUG
  // Check that there are no gaps in the Aux vector.
  for (auto &RPtr : Regions)
    for (auto *I : RPtr->getAux())
      assert(I != nullptr && "Gap in Aux!");
#endif
  return Regions;
}

} // namespace llvm::sandboxir
