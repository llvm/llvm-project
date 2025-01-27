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
#include "llvm/IR/Module.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/SandboxIR/Instruction.h"
#include <sstream>

using namespace llvm::sandboxir;

#ifndef NDEBUG

std::string IRSnapshotChecker::dumpIR(const llvm::Function &F) const {
  std::string Result;
  raw_string_ostream SS(Result);
  F.print(SS, /*AssemblyAnnotationWriter=*/nullptr);
  return Result;
}

IRSnapshotChecker::ContextSnapshot IRSnapshotChecker::takeSnapshot() const {
  ContextSnapshot Result;
  for (const auto &Entry : Ctx.LLVMModuleToModuleMap)
    for (const auto &F : *Entry.first) {
      FunctionSnapshot Snapshot;
      Snapshot.Hash = StructuralHash(F, /*DetailedHash=*/true);
      Snapshot.TextualIR = dumpIR(F);
      Result[&F] = Snapshot;
    }
  return Result;
}

bool IRSnapshotChecker::diff(const ContextSnapshot &Orig,
                             const ContextSnapshot &Curr) const {
  bool DifferenceFound = false;
  for (const auto &[F, OrigFS] : Orig) {
    auto CurrFSIt = Curr.find(F);
    if (CurrFSIt == Curr.end()) {
      DifferenceFound = true;
      dbgs() << "Function " << F->getName() << " not found in current IR.\n";
      dbgs() << OrigFS.TextualIR << "\n";
      continue;
    }
    const FunctionSnapshot &CurrFS = CurrFSIt->second;
    if (OrigFS.Hash != CurrFS.Hash) {
      DifferenceFound = true;
      dbgs() << "Found IR difference in Function " << F->getName() << "\n";
      dbgs() << "Original:\n" << OrigFS.TextualIR << "\n";
      dbgs() << "Current:\n" << CurrFS.TextualIR << "\n";
    }
  }
  // Check that Curr doesn't contain any new functions.
  for (const auto &[F, CurrFS] : Curr) {
    if (!Orig.contains(F)) {
      DifferenceFound = true;
      dbgs() << "Function " << F->getName()
             << " found in current IR but not in original snapshot.\n";
      dbgs() << CurrFS.TextualIR << "\n";
    }
  }
  return DifferenceFound;
}

void IRSnapshotChecker::save() { OrigContextSnapshot = takeSnapshot(); }

void IRSnapshotChecker::expectNoDiff() {
  ContextSnapshot CurrContextSnapshot = takeSnapshot();
  if (diff(OrigContextSnapshot, CurrContextSnapshot)) {
    llvm_unreachable(
        "Original and current IR differ! Probably a checkpointing bug.");
  }
}

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
  if (auto *NextLLVMI = dyn_cast<llvm::Instruction *>(NextLLVMIOrBB)) {
    BotLLVMI->insertBefore(NextLLVMI->getIterator());
  } else {
    auto *LLVMBB = cast<llvm::BasicBlock *>(NextLLVMIOrBB);
    BotLLVMI->insertInto(LLVMBB, LLVMBB->end());
  }
  for (auto [OpNum, Op] : enumerate(Operands))
    BotLLVMI->setOperand(OpNum, Op);

  // Go over the rest of the instructions and stack them on top.
  for (auto [Operands, LLVMI] : drop_begin(InstrData)) {
    LLVMI->insertBefore(BotLLVMI->getIterator());
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
  if (auto *NextI = dyn_cast<Instruction *>(NextInstrOrBB)) {
    RemovedI->insertBefore(NextI);
  } else {
    auto *BB = cast<BasicBlock *>(NextInstrOrBB);
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

SwitchRemoveCase::SwitchRemoveCase(SwitchInst *Switch) : Switch(Switch) {
  for (const auto &C : Switch->cases())
    Cases.push_back({C.getCaseValue(), C.getCaseSuccessor()});
}

void SwitchRemoveCase::revert(Tracker &Tracker) {
  // SwitchInst::removeCase doesn't provide any guarantees about the order of
  // cases after removal. In order to preserve the original ordering, we save
  // all of them and, when reverting, clear them all then insert them in the
  // desired order. This still relies on the fact that `addCase` will insert
  // them at the end, but it is documented to invalidate `case_end()` so it's
  // probably okay.
  unsigned NumCases = Switch->getNumCases();
  for (unsigned I = 0; I < NumCases; ++I)
    Switch->removeCase(Switch->case_begin());
  for (auto &Case : Cases)
    Switch->addCase(Case.Val, Case.Dest);
}

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
  if (auto *NextI = dyn_cast<Instruction *>(NextInstrOrBB)) {
    MovedI->moveBefore(NextI);
  } else {
    auto *BB = cast<BasicBlock *>(NextInstrOrBB);
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

void Tracker::save() {
  State = TrackerState::Record;
#if !defined(NDEBUG) && defined(EXPENSIVE_CHECKS)
  SnapshotChecker.save();
#endif
}

void Tracker::revert() {
  assert(State == TrackerState::Record && "Forgot to save()!");
  State = TrackerState::Disabled;
  for (auto &Change : reverse(Changes))
    Change->revert(*this);
  Changes.clear();
#if !defined(NDEBUG) && defined(EXPENSIVE_CHECKS)
  SnapshotChecker.expectNoDiff();
#endif
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
