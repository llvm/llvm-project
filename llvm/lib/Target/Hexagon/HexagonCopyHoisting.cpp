//===--------- HexagonCopyHoisting.cpp - Hexagon Copy Hoisting  ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The purpose of this pass is to move the copy instructions that are
// present in all the successor of a basic block (BB) to the end of BB.
//===----------------------------------------------------------------------===//

#include "HexagonTargetMachine.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "CopyHoist"

using namespace llvm;

static cl::opt<std::string> CPHoistFn("cphoistfn", cl::Hidden, cl::desc(""),
                                      cl::init(""));

namespace llvm {
void initializeHexagonCopyHoistingPass(PassRegistry &Registry);
FunctionPass *createHexagonCopyHoisting();
} // namespace llvm

namespace {

class HexagonCopyHoisting : public MachineFunctionPass {

public:
  static char ID;
  HexagonCopyHoisting() : MachineFunctionPass(ID), MFN(0), MRI(0) {
    initializeHexagonCopyHoistingPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Hexagon Copy Hoisting"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<SlotIndexes>();
    AU.addRequired<LiveIntervals>();
    AU.addPreserved<SlotIndexes>();
    AU.addPreserved<LiveIntervals>();
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;
  void collectCopyInst();
  void addMItoCopyList(MachineInstr *MI);
  bool analyzeCopy(MachineBasicBlock *BB);
  bool isSafetoMove(MachineInstr *CandMI);
  void moveCopyInstr(MachineBasicBlock *DestBB, StringRef key,
                     MachineInstr *MI);

  MachineFunction *MFN;
  MachineRegisterInfo *MRI;
  StringMap<MachineInstr *> CopyMI;
  std::vector<StringMap<MachineInstr *>> CopyMIList;
};

} // namespace

char HexagonCopyHoisting::ID = 0;

namespace llvm {
char &HexagonCopyHoistingID = HexagonCopyHoisting::ID;
}

bool HexagonCopyHoisting::runOnMachineFunction(MachineFunction &Fn) {

  if ((CPHoistFn != "") && (CPHoistFn != Fn.getFunction().getName()))
    return false;

  MFN = &Fn;
  MRI = &Fn.getRegInfo();

  LLVM_DEBUG(dbgs() << "\nCopy Hoisting:"
                    << "\'" << Fn.getName() << "\'\n");

  CopyMIList.clear();
  CopyMIList.resize(Fn.getNumBlockIDs());

  // Traverse through all basic blocks and collect copy instructions.
  collectCopyInst();

  // Traverse through the basic blocks again and move the COPY instructions
  // that are present in all the successors of BB to BB.
  bool changed = false;
  for (auto I = po_begin(&Fn), E = po_end(&Fn); I != E; ++I) {
    MachineBasicBlock &BB = **I;
    if (!BB.empty()) {
      if (BB.pred_size() != 1) //
        continue;
      auto &BBCopyInst = CopyMIList[BB.getNumber()];
      if (BBCopyInst.size() > 0)
        changed |= analyzeCopy(*BB.pred_begin());
    }
  }
  // Re-compute liveness
  if (changed) {
    LiveIntervals &LIS = getAnalysis<LiveIntervals>();
    SlotIndexes *SI = LIS.getSlotIndexes();
    SI->releaseMemory();
    SI->runOnMachineFunction(Fn);
    LIS.releaseMemory();
    LIS.runOnMachineFunction(Fn);
  }
  return changed;
}

//===----------------------------------------------------------------------===//
// Save all COPY instructions for each basic block in CopyMIList vector.
//===----------------------------------------------------------------------===//
void HexagonCopyHoisting::collectCopyInst() {
  for (auto BI = MFN->begin(), BE = MFN->end(); BI != BE; ++BI) {
    MachineBasicBlock *BB = &*BI;
#ifndef NDEBUG
    auto &BBCopyInst = CopyMIList[BB->getNumber()];
    LLVM_DEBUG(dbgs() << "Visiting BB#" << BB->getNumber() << ":\n");
#endif

    for (auto MII = BB->begin(), MIE = BB->end(); MII != MIE; ++MII) {
      MachineInstr *MI = &*MII;
      if (MI->getOpcode() == TargetOpcode::COPY)
        addMItoCopyList(MI);
    }
    LLVM_DEBUG(dbgs() << "\tNumber of copies: " << BBCopyInst.size() << "\n");
  }
}

void HexagonCopyHoisting::addMItoCopyList(MachineInstr *MI) {
  unsigned BBNum = MI->getParent()->getNumber();
  auto &BBCopyInst = CopyMIList[BBNum];
  unsigned DstReg = MI->getOperand(0).getReg();
  unsigned SrcReg = MI->getOperand(1).getReg();

  if (!Register::isVirtualRegister(DstReg) ||
      !Register::isVirtualRegister(SrcReg) ||
      MRI->getRegClass(DstReg) != &Hexagon::IntRegsRegClass ||
      MRI->getRegClass(SrcReg) != &Hexagon::IntRegsRegClass)
    return;

  StringRef key;
  SmallString<256> TmpData("");
  (void)Twine(Register::virtReg2Index(DstReg)).toStringRef(TmpData);
  TmpData += '=';
  key = Twine(Register::virtReg2Index(SrcReg)).toStringRef(TmpData);
  BBCopyInst[key] = MI;
#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "\tAdding Copy Instr to the list: " << MI << "\n");
  for (auto II = BBCopyInst.begin(), IE = BBCopyInst.end(); II != IE; ++II) {
    MachineInstr *TempMI = (*II).getValue();
    LLVM_DEBUG(dbgs() << "\tIn the list: " << TempMI << "\n");
  }
#endif
}

//===----------------------------------------------------------------------===//
// Look at the COPY instructions of all the successors of BB. If the same
// instruction is present in every successor and can be safely moved,
// pull it into BB.
//===----------------------------------------------------------------------===//
bool HexagonCopyHoisting::analyzeCopy(MachineBasicBlock *BB) {

  bool changed = false;
  if (BB->succ_size() < 2)
    return false;

  for (auto I = BB->succ_begin(), E = BB->succ_end(); I != E; ++I) {
    MachineBasicBlock *SB = *I;
    if (SB->pred_size() != 1 || SB->isEHPad() || SB->hasAddressTaken())
      return false;
  }

  auto SuccI = BB->succ_begin(), SuccE = BB->succ_end();

  MachineBasicBlock *SBB1 = *SuccI;
  ++SuccI;
  auto &BBCopyInst1 = CopyMIList[SBB1->getNumber()];

  for (auto II = BBCopyInst1.begin(), IE = BBCopyInst1.end(); II != IE; ++II) {
    StringRef key = (*II).getKeyData();
    MachineInstr *MI = (*II).getValue();
    bool IsSafetoMove = true;
    for (SuccI = BB->succ_begin(); SuccI != SuccE; ++SuccI) {
      MachineBasicBlock *SuccBB = *SuccI;
      auto &SuccBBCopyInst = CopyMIList[SuccBB->getNumber()];
      if (!SuccBBCopyInst.count(key)) {
        // Same copy not present in this successor
        IsSafetoMove = false;
        break;
      }
      // If present, make sure that it's safe to pull this copy instruction
      // into the predecessor.
      MachineInstr *SuccMI = SuccBBCopyInst[key];
      if (!isSafetoMove(SuccMI)) {
        IsSafetoMove = false;
        break;
      }
    }
    // If we have come this far, this copy instruction can be safely
    // moved to the predecessor basic block.
    if (IsSafetoMove) {
      LLVM_DEBUG(dbgs() << "\t\t Moving instr to BB#" << BB->getNumber() << ": "
                        << MI << "\n");
      moveCopyInstr(BB, key, MI);
      // Add my into BB copyMI list.
      changed = true;
    }
  }

#ifndef NDEBUG
  auto &BBCopyInst = CopyMIList[BB->getNumber()];
  for (auto II = BBCopyInst.begin(), IE = BBCopyInst.end(); II != IE; ++II) {
    MachineInstr *TempMI = (*II).getValue();
    LLVM_DEBUG(dbgs() << "\tIn the list: " << TempMI << "\n");
  }
#endif
  return changed;
}

bool HexagonCopyHoisting::isSafetoMove(MachineInstr *CandMI) {
  // Make sure that it's safe to move this 'copy' instruction to the predecessor
  // basic block.
  assert(CandMI->getOperand(0).isReg() && CandMI->getOperand(1).isReg());
  unsigned DefR = CandMI->getOperand(0).getReg();
  unsigned UseR = CandMI->getOperand(1).getReg();

  MachineBasicBlock *BB = CandMI->getParent();
  // There should not be a def/use of DefR between the start of BB and CandMI.
  MachineBasicBlock::iterator MII, MIE;
  for (MII = BB->begin(), MIE = CandMI; MII != MIE; ++MII) {
    MachineInstr *otherMI = &*MII;
    for (MachineInstr::mop_iterator Mo = otherMI->operands_begin(),
                                    E = otherMI->operands_end();
         Mo != E; ++Mo)
      if (Mo->isReg() && Mo->getReg() == DefR)
        return false;
  }
  // There should not be a def of UseR between the start of BB and CandMI.
  for (MII = BB->begin(), MIE = CandMI; MII != MIE; ++MII) {
    MachineInstr *otherMI = &*MII;
    for (MachineInstr::mop_iterator Mo = otherMI->operands_begin(),
                                    E = otherMI->operands_end();
         Mo != E; ++Mo)
      if (Mo->isReg() && Mo->isDef() && Mo->getReg() == UseR)
        return false;
  }
  return true;
}

void HexagonCopyHoisting::moveCopyInstr(MachineBasicBlock *DestBB,
                                        StringRef key, MachineInstr *MI) {
  MachineBasicBlock::iterator FirstTI = DestBB->getFirstTerminator();
  assert(FirstTI != DestBB->end());

  DestBB->splice(FirstTI, MI->getParent(), MI);

  addMItoCopyList(MI);
  auto I = ++(DestBB->succ_begin()), E = DestBB->succ_end();
  for (; I != E; ++I) {
    MachineBasicBlock *SuccBB = *I;
    auto &BBCopyInst = CopyMIList[SuccBB->getNumber()];
    MachineInstr *SuccMI = BBCopyInst[key];
    SuccMI->eraseFromParent();
    BBCopyInst.erase(key);
  }
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

INITIALIZE_PASS(HexagonCopyHoisting, "hexagon-move-phicopy",
                "Hexagon move phi copy", false, false)

FunctionPass *llvm::createHexagonCopyHoisting() {
  return new HexagonCopyHoisting();
}
