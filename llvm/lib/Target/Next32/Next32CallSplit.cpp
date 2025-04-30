//===-- Next32CallSplit.cpp - use or replace LEA instructions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass that finds instructions that can be
// re-written as LEA instructions in order to reduce pipeline delays.
// When optimizing for size it replaces suitable LEAs with INC or DEC.
//
//===----------------------------------------------------------------------===//

#include "Next32.h"
#include "Next32InstrInfo.h"
#include "Next32PassTrace.h"
#include "Next32Subtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace llvm {
void initializeCallSplitPassPass(PassRegistry &);
}

#define CALLSPLIT_DESC "Next32 CallSplit Fixup"
#define CALLSPLIT_NAME "Next32-callsplit"

#define DEBUG_TYPE CALLSPLIT_NAME

namespace {
class CallSplitPass : public MachineFunctionPass {

public:
  static char ID;

  StringRef getPassName() const override { return CALLSPLIT_DESC; }

  CallSplitPass() : MachineFunctionPass(ID) {
    initializeCallSplitPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &Func) override;

private:
  MachineFunction *MF;
  const Next32InstrInfo *TII;

  bool ProcessBasicBlock(MachineBasicBlock &MBB);
  void Split(MachineBasicBlock &MBB, MachineInstr &CTMI, MachineInstr &CMI);
};
} // namespace

char CallSplitPass::ID = 0;

INITIALIZE_PASS(CallSplitPass, CALLSPLIT_NAME, CALLSPLIT_DESC, false, false)

FunctionPass *llvm::createNext32CallSplits() { return new CallSplitPass(); }

bool CallSplitPass::runOnMachineFunction(MachineFunction &Func) {
  Next32PassTrace TFunc(DEBUG_TYPE, Func);
  TII = Func.getSubtarget<Next32Subtarget>().getInstrInfo();
  MF = &Func;
  bool Changed = false;
  for (auto &MBB : TFunc)
    if (!MBB.empty())
      Changed |= ProcessBasicBlock(MBB);

  return Changed;
}

bool CallSplitPass::ProcessBasicBlock(MachineBasicBlock &MBB) {
  for (auto I = MBB.begin(); I != MBB.end(); ++I) {
    if ((I->getOpcode() != Next32::CALL) && (I->getOpcode() != Next32::CALLc) &&
        (I->getOpcode() != Next32::CALLPTRWRAPPER))
      continue;

    if (I == MBB.end())
      return false;

    auto CT = I;
    ++CT;
    if ((CT->getOpcode() == Next32::CALL) ||
        (CT->getOpcode() == Next32::CALLc) ||
        (CT->getOpcode() == Next32::CALLPTRWRAPPER))
      continue;

    Split(MBB, *CT, *I);
    break;
  }
  return true;
}

void CallSplitPass::Split(MachineBasicBlock &MBB, MachineInstr &CTMI,
                          MachineInstr &CMI) {
  if (!MBB.isLiveIn(Next32::RET_FID))
    MBB.addLiveIn(Next32::RET_FID);

  MachineBasicBlock *NewMBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  NewMBB->transferSuccessorsAndUpdatePHIs(&MBB);
  MBB.addSuccessor(NewMBB);
  MF->insert(std::next(MachineFunction::iterator(MBB)), NewMBB);
  NewMBB->splice(NewMBB->end(), &MBB, CTMI, MBB.end());
  NewMBB->addLiveIn(Next32::TID);
  // Add return basic-block to CALL instruction
  CMI.addOperand(MachineOperand::CreateMBB(NewMBB));
}
