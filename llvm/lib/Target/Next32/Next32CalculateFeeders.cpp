//===-- Next32CalculateFeeders.cpp - Describe live-ins via feeders --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass that
//
//===----------------------------------------------------------------------===//

#include "Next32.h"
#include "Next32InstrInfo.h"
#include "Next32PassTrace.h"
#include "Next32Subtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace llvm {
void initializeCalculateFeedersPassPass(PassRegistry &);
}

#define CALCULATEFEEDERS_DESC "Next32 CalculateFeeders"
#define CALCULATEFEEDERS_NAME "Next32-calculatefeeder"

#define DEBUG_TYPE CALCULATEFEEDERS_NAME

namespace {
class CalculateFeedersPass : public MachineFunctionPass {

public:
  static char ID;

  StringRef getPassName() const override { return CALCULATEFEEDERS_DESC; }

  CalculateFeedersPass() : MachineFunctionPass(ID) {
    initializeCalculateFeedersPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  const Next32RegisterInfo *TRI;
  const Next32InstrInfo *TII;
  MachineFunction *MF;

  void ProcessBasicBlock(MachineBasicBlock &MBB);
  unsigned int GetFeederOpcode(uint16_t PhysReg, MachineBasicBlock &MBB) const;
  void BuildFeederMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     DebugLoc DL, bool Passthrough, Register LiveInReg,
                     MachineInstr *DbgVal);
};
} // namespace

char CalculateFeedersPass::ID = 0;

INITIALIZE_PASS(CalculateFeedersPass, CALCULATEFEEDERS_NAME,
                CALCULATEFEEDERS_DESC, false, false)

FunctionPass *llvm::createNext32CalculateFeeders() {
  return new CalculateFeedersPass();
}

bool CalculateFeedersPass::runOnMachineFunction(MachineFunction &Func) {
  Next32PassTrace TFunc(DEBUG_TYPE, Func);
  TII = Func.getSubtarget<Next32Subtarget>().getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MF = &Func;

  for (auto &MBB : TFunc) {
    if (MBB.getNumber() == Func.begin()->getNumber())
      continue;
    ProcessBasicBlock(MBB);
  }
  return false;
}

void CalculateFeedersPass::BuildFeederMI(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator I,
                                         DebugLoc DL, bool Passthrough,
                                         Register LiveInReg,
                                         MachineInstr *DbgVal) {
  unsigned Opcode = Passthrough ? Next32::FEEDERP : Next32::FEEDER;

  if (DbgVal) {
    BuildMI(MBB, I, DbgVal->getDebugLoc(), TII->get(TargetOpcode::DBG_VALUE),
            DbgVal->isIndirectDebugValue(), DbgVal->getOperand(0),
            DbgVal->getDebugVariable(), DbgVal->getDebugExpression());
    LLVM_DEBUG(dbgs() << "\tCreating feeder debug value for reg: " << LiveInReg
                      << "\n");
  }

  BuildMI(MBB, I, DL, TII->get(Opcode), LiveInReg)
      .addReg(LiveInReg)
      .addImm(llvm::Next32Constants::InstructionSize::InstructionSize32);
  LLVM_DEBUG(dbgs() << "\tCreating Feeder for: " << LiveInReg << " (" << Opcode
                    << ")\n");
}

void CalculateFeedersPass::ProcessBasicBlock(MachineBasicBlock &MBB) {
  if (MBB.empty()) {
    BuildMI(&MBB, DebugLoc(), TII->get(Next32::FEEDER), Next32::TID)
        .addReg(Next32::TID)
        .addImm(llvm::Next32Constants::InstructionSize::InstructionSize32);
    BuildMI(&MBB, DebugLoc(), TII->get(Next32::FEEDER), Next32::RET_FID)
        .addReg(Next32::RET_FID)
        .addImm(llvm::Next32Constants::InstructionSize::InstructionSize32);
    return;
  }

  if (MBB.livein_empty())
    MBB.addLiveIn(Next32::RET_FID);

  for (auto &MI : MBB) {
    if (MI.getOpcode() == Next32::SYM_INSTR)
      break;
    if (MI.getOpcode() == Next32::FEEDER || MI.getOpcode() == Next32::FEEDERP)
      MBB.removeLiveIn(MI.getOperand(1).getReg());
  }

  std::vector<DebugLoc> RegDLs(TRI->getNumRegs());
  std::vector<MachineInstr *> RegDbgVals(TRI->getNumRegs());
  auto UpdateRegDL = [](DebugLoc &DL, const MachineInstr &I) {
    if (!DL)
      DL = I.getDebugLoc();
  };
  auto UpdateRegDbgVal = [](MachineInstr *&DbgVal, MachineInstr &I) {
    if (!DbgVal)
      DbgVal = &I;
  };

  for (auto &I : MBB) {
    if (I.isCall()) {
      UpdateRegDL(RegDLs[Next32::TID], I);
    } else if (I.isDebugValue() && I.getOperand(0).isReg()) {
      auto &O = I.getOperand(0);
      UpdateRegDbgVal(RegDbgVals[O.getReg()], I);
    }

    for (auto &O : I.operands()) {
      if (!O.isReg())
        continue;

      UpdateRegDL(RegDLs[O.getReg()], I);
    }
  }

  DebugLoc FallbackDL = MBB.findDebugLoc(MBB.begin());
  MachineBasicBlock::iterator I = MBB.begin();

  auto BuildFeeder = [&](Register R) {
    const DebugLoc &DL = RegDLs[R];
    BuildFeederMI(MBB, I, DL ? DL : FallbackDL, /*Passthrough*/ !DL, R,
                  RegDbgVals[R]);
  };

  if (MBB.isLiveIn(Next32::TID))
    BuildFeeder(Next32::TID);

  for (auto &LiveIn : MBB.liveins()) {
    if (LiveIn.PhysReg == Next32::TID)
      continue;

    BuildFeeder(LiveIn.PhysReg);
  }
}
