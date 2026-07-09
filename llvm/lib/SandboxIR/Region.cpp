//===- Region.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Region.h"

namespace llvm::sandboxir {

Region::Region(Context &Ctx, RegionClassID ID) : Ctx(Ctx), ID(ID) {
  LLVMContext &LLVMCtx = Ctx.LLVMCtx;
  auto *RegionStrMD = MDString::get(LLVMCtx, RegionStr);
  RegionMDN = MDNode::getDistinct(LLVMCtx, {RegionStrMD});

  CreateInstCB = Ctx.registerCreateInstrCallback(
      [this](Instruction *NewInst) { addRaw(NewInst); });
  EraseInstCB = Ctx.registerEraseInstrCallback([this](Instruction *ErasedInst) {
    remove(ErasedInst);
    removeFromAux(ErasedInst);
  });
}

Region::~Region() {
  Ctx.unregisterCreateInstrCallback(CreateInstCB);
  Ctx.unregisterEraseInstrCallback(EraseInstCB);
}

void Region::setAux(ArrayRef<Instruction *> Aux) {
  this->Aux = SmallVector<Instruction *>(Aux);
  auto &LLVMCtx = Ctx.LLVMCtx;
  for (auto [Idx, I] : enumerate(Aux)) {
    llvm::ConstantInt *IdxC =
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(LLVMCtx), Idx, false);
    assert(cast<llvm::Instruction>(I->Val)->getMetadata(AuxMDKind) == nullptr &&
           "Instruction already in Aux!");
    cast<llvm::Instruction>(I->Val)->setMetadata(
        AuxMDKind, MDNode::get(LLVMCtx, ConstantAsMetadata::get(IdxC)));
    // Aux instrs should always be in a region.
    addRaw(I);
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
  // Aux instrs should always be in a region.
  addRaw(I);
}

void Region::dropAuxMetadata(Instruction *I) {
  auto *LLVMI = cast<llvm::Instruction>(I->Val);
  LLVMI->setMetadata(AuxMDKind, nullptr);
}

void Region::removeFromAux(Instruction *I) {
  auto It = find(Aux, I);
  if (It == Aux.end())
    return;
  dropAuxMetadata(I);
  Aux.erase(It);
}

void Region::clearAux() {
  for (unsigned Idx : seq<unsigned>(0, Aux.size()))
    dropAuxMetadata(Aux[Idx]);
  Aux.clear();
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

SmallVector<std::unique_ptr<Region>> Region::createRegionsFromMD(Function &F) {
  return Region::createRegionsFromMD<Region>(
      F, [&F]() { return std::make_unique<Region>(F.getContext()); });
}

} // namespace llvm::sandboxir
