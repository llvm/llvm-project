//===-- Next32AddRetFid.cpp - use or replace LEA instructions -----------===//
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
void initializeAddRetFidPassPass(PassRegistry &);
}

#define ADDRETFID_DESC "Next32 AddRetFid Fixup"
#define ADDRETFID_NAME "Next32-addretfid"

#define DEBUG_TYPE ADDRETFID_NAME

namespace {
class AddRetFidPass : public MachineFunctionPass {

public:
  static char ID;

  StringRef getPassName() const override { return ADDRETFID_DESC; }

  AddRetFidPass() : MachineFunctionPass(ID) {
    initializeAddRetFidPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &Func) override;

private:
  MachineFunction *MF;
  const Next32InstrInfo *TII;

  bool ProcessBasicBlock(MachineBasicBlock &MBB);
  void Split(MachineBasicBlock &MBB, MachineInstr &CTMI, MachineInstr &CMI);
};
} // namespace

char AddRetFidPass::ID = 0;

INITIALIZE_PASS(AddRetFidPass, ADDRETFID_NAME, ADDRETFID_DESC, false, false)

FunctionPass *llvm::createNext32AddRetFid() { return new AddRetFidPass(); }

bool AddRetFidPass::runOnMachineFunction(MachineFunction &Func) {
  Next32PassTrace TFunc(DEBUG_TYPE, Func);
  TII = Func.getSubtarget<Next32Subtarget>().getInstrInfo();
  MF = &Func;
  bool Changed = false;
  for (auto &MBB : TFunc)
    Changed |= ProcessBasicBlock(MBB);
  return Changed;
}

bool AddRetFidPass::ProcessBasicBlock(MachineBasicBlock &MBB) {
  if (MBB.isLiveIn(Next32::RET_FID))
    return false;

  MBB.addLiveIn(Next32::RET_FID);
  return true;
}
