//===- Tracker.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Tracker.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/SandboxIR/Instruction.h"
#include <sstream>

using namespace llvm::sandboxir;

#ifndef NDEBUG
void UseSet::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void UseSwap::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

PHIRemoveIncoming::PHIRemoveIncoming(PHINode *PHI, unsigned RemovedIdx)
    : PHI(PHI), RemovedIdx(RemovedIdx) {
  RemovedV = PHI->getIncomingValue(RemovedIdx);
  RemovedBB = PHI->getIncomingBlock(RemovedIdx);
}

void PHIRemoveIncoming::revert(Tracker &Tracker) {
  // Special case: if the PHI is now empty, as we don't need to care about the
  // order of the incoming values.
  unsigned NumIncoming = PHI->getNumIncomingValues();
  if (NumIncoming == 0) {
    PHI->addIncoming(RemovedV, RemovedBB);
    return;
  }
  // Shift all incoming values by one starting from the end until `Idx`.
  // Start by adding a copy of the last incoming values.
  unsigned LastIdx = NumIncoming - 1;
  PHI->addIncoming(PHI->getIncomingValue(LastIdx),
                   PHI->getIncomingBlock(LastIdx));
  for (unsigned Idx = LastIdx; Idx > RemovedIdx; --Idx) {
    auto *PrevV = PHI->getIncomingValue(Idx - 1);
    auto *PrevBB = PHI->getIncomingBlock(Idx - 1);
    PHI->setIncomingValue(Idx, PrevV);
    PHI->setIncomingBlock(Idx, PrevBB);
  }
  PHI->setIncomingValue(RemovedIdx, RemovedV);
  PHI->setIncomingBlock(RemovedIdx, RemovedBB);
}

#ifndef NDEBUG
void PHIRemoveIncoming::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

PHIAddIncoming::PHIAddIncoming(PHINode *PHI)
    : PHI(PHI), Idx(PHI->getNumIncomingValues()) {}

void PHIAddIncoming::revert(Tracker &Tracker) { PHI->removeIncomingValue(Idx); }

#ifndef NDEBUG
void PHIAddIncoming::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Tracker::~Tracker() {
  assert(Changes.empty() && "You must accept or revert changes!");
}

EraseFromParent::EraseFromParent(std::unique_ptr<sandboxir::Value> &&ErasedIPtr)
    : ErasedIPtr(std::move(ErasedIPtr)) {
  auto *I = cast<Instruction>(this->ErasedIPtr.get());
  auto LLVMInstrs = I->getLLVMInstrs();
  // Iterate in reverse program order.
  for (auto *LLVMI : reverse(LLVMInstrs)) {
    SmallVector<llvm::Value *> Operands;
    Operands.reserve(LLVMI->getNumOperands());
    for (auto [OpNum, Use] : enumerate(LLVMI->operands()))
      Operands.push_back(Use.get());
    InstrData.push_back({Operands, LLVMI});
  }
  assert(is_sorted(InstrData,
                   [](const auto &D0, const auto &D1) {
                     return D0.LLVMI->comesBefore(D1.LLVMI);
                   }) &&
         "Expected reverse program order!");
  auto *BotLLVMI = cast<llvm::Instruction>(I->Val);
  if (BotLLVMI->getNextNode() != nullptr)
    NextLLVMIOrBB = BotLLVMI->getNextNode();
  else
    NextLLVMIOrBB = BotLLVMI->getParent();
}

void EraseFromParent::accept() {
  for (const auto &IData : InstrData)
    IData.LLVMI->deleteValue();
}

void EraseFromParent::revert(Tracker &Tracker) {
  // Place the bottom-most instruction first.
  auto [Operands, BotLLVMI] = InstrData[0];
  if (auto *NextLLVMI = NextLLVMIOrBB.dyn_cast<llvm::Instruction *>()) {
    BotLLVMI->insertBefore(NextLLVMI);
  } else {
    auto *LLVMBB = NextLLVMIOrBB.get<llvm::BasicBlock *>();
    BotLLVMI->insertInto(LLVMBB, LLVMBB->end());
  }
  for (auto [OpNum, Op] : enumerate(Operands))
    BotLLVMI->setOperand(OpNum, Op);

  // Go over the rest of the instructions and stack them on top.
  for (auto [Operands, LLVMI] : drop_begin(InstrData)) {
    LLVMI->insertBefore(BotLLVMI);
    for (auto [OpNum, Op] : enumerate(Operands))
      LLVMI->setOperand(OpNum, Op);
    BotLLVMI = LLVMI;
  }
  Tracker.getContext().registerValue(std::move(ErasedIPtr));
}

#ifndef NDEBUG
void EraseFromParent::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

RemoveFromParent::RemoveFromParent(Instruction *RemovedI) : RemovedI(RemovedI) {
  if (auto *NextI = RemovedI->getNextNode())
    NextInstrOrBB = NextI;
  else
    NextInstrOrBB = RemovedI->getParent();
}

void RemoveFromParent::revert(Tracker &Tracker) {
  if (auto *NextI = NextInstrOrBB.dyn_cast<Instruction *>()) {
    RemovedI->insertBefore(NextI);
  } else {
    auto *BB = NextInstrOrBB.get<BasicBlock *>();
    RemovedI->insertInto(BB, BB->end());
  }
}

#ifndef NDEBUG
void RemoveFromParent::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

CatchSwitchAddHandler::CatchSwitchAddHandler(CatchSwitchInst *CSI)
    : CSI(CSI), HandlerIdx(CSI->getNumHandlers()) {}

void CatchSwitchAddHandler::revert(Tracker &Tracker) {
  // TODO: This should ideally use sandboxir::CatchSwitchInst::removeHandler()
  // once it gets implemented.
  auto *LLVMCSI = cast<llvm::CatchSwitchInst>(CSI->Val);
  LLVMCSI->removeHandler(LLVMCSI->handler_begin() + HandlerIdx);
}

void SwitchRemoveCase::revert(Tracker &Tracker) { Switch->addCase(Val, Dest); }

#ifndef NDEBUG
void SwitchRemoveCase::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

void SwitchAddCase::revert(Tracker &Tracker) {
  auto It = Switch->findCaseValue(Val);
  Switch->removeCase(It);
}

#ifndef NDEBUG
void SwitchAddCase::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

MoveInstr::MoveInstr(Instruction *MovedI) : MovedI(MovedI) {
  if (auto *NextI = MovedI->getNextNode())
    NextInstrOrBB = NextI;
  else
    NextInstrOrBB = MovedI->getParent();
}

void MoveInstr::revert(Tracker &Tracker) {
  if (auto *NextI = NextInstrOrBB.dyn_cast<Instruction *>()) {
    MovedI->moveBefore(NextI);
  } else {
    auto *BB = NextInstrOrBB.get<BasicBlock *>();
    MovedI->moveBefore(*BB, BB->end());
  }
}

#ifndef NDEBUG
void MoveInstr::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

void InsertIntoBB::revert(Tracker &Tracker) { InsertedI->removeFromParent(); }

InsertIntoBB::InsertIntoBB(Instruction *InsertedI) : InsertedI(InsertedI) {}

#ifndef NDEBUG
void InsertIntoBB::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

void CreateAndInsertInst::revert(Tracker &Tracker) { NewI->eraseFromParent(); }

#ifndef NDEBUG
void CreateAndInsertInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

ShuffleVectorSetMask::ShuffleVectorSetMask(ShuffleVectorInst *SVI)
    : SVI(SVI), PrevMask(SVI->getShuffleMask()) {}

void ShuffleVectorSetMask::revert(Tracker &Tracker) {
  SVI->setShuffleMask(PrevMask);
}

#ifndef NDEBUG
void ShuffleVectorSetMask::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

CmpSwapOperands::CmpSwapOperands(CmpInst *Cmp) : Cmp(Cmp) {}

void CmpSwapOperands::revert(Tracker &Tracker) { Cmp->swapOperands(); }
#ifndef NDEBUG
void CmpSwapOperands::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

void Tracker::save() { State = TrackerState::Record; }

void Tracker::revert() {
  assert(State == TrackerState::Record && "Forgot to save()!");
  State = TrackerState::Disabled;
  for (auto &Change : reverse(Changes))
    Change->revert(*this);
  Changes.clear();
}

void Tracker::accept() {
  assert(State == TrackerState::Record && "Forgot to save()!");
  State = TrackerState::Disabled;
  for (auto &Change : Changes)
    Change->accept();
  Changes.clear();
}

#ifndef NDEBUG
void Tracker::dump(raw_ostream &OS) const {
  for (auto [Idx, ChangePtr] : enumerate(Changes)) {
    OS << Idx << ". ";
    ChangePtr->dump(OS);
    OS << "\n";
  }
}
void Tracker::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG
