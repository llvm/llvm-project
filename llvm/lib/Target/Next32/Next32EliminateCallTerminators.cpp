//===-- Next32EliminateCallTerminators.cpp - replace CT instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass that finds call terminator instructions and
// replaces them with feeder instructions
//
//===----------------------------------------------------------------------===//

#include "Next32.h"
#include "Next32InstrInfo.h"
#include "Next32PassTrace.h"
#include "Next32Subtarget.h"
#include "TargetInfo/Next32BaseInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace llvm {
void initializeNext32EliminateCallTerminatorsPassPass(PassRegistry &);
}

#define ELIMINATECALLTERMINATOR_DESC "Next32 EliminateCallTerminators Fixup"
#define ELIMINATECALLTERMINATOR_NAME "Next32-eliminatecallterminators"

#define DEBUG_TYPE ELIMINATECALLTERMINATOR_NAME

namespace {
class Next32EliminateCallTerminatorsPass : public MachineFunctionPass {

public:
  static char ID;

  StringRef getPassName() const override {
    return ELIMINATECALLTERMINATOR_DESC;
  }

  Next32EliminateCallTerminatorsPass() : MachineFunctionPass(ID) {
    initializeNext32EliminateCallTerminatorsPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &Func) override;

  // This pass runs after regalloc and doesn't support VReg operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  unsigned int GetFeederOpcode(uint16_t PhysReg, MachineBasicBlock &MBB) const;
};
} // namespace

char Next32EliminateCallTerminatorsPass::ID = 0;

INITIALIZE_PASS(Next32EliminateCallTerminatorsPass,
                ELIMINATECALLTERMINATOR_NAME, ELIMINATECALLTERMINATOR_DESC,
                false, false)

FunctionPass *llvm::createNext32EliminateCallTerminators() {
  return new Next32EliminateCallTerminatorsPass();
}

bool Next32EliminateCallTerminatorsPass::runOnMachineFunction(
    MachineFunction &Func) {
  Next32PassTrace TFunc(DEBUG_TYPE, Func);
  const Next32InstrInfo *TII =
      Func.getSubtarget<Next32Subtarget>().getInstrInfo();
  std::vector<MachineInstr *> EraseFromMI;

  for (auto &MBBI : TFunc) {
    for (auto &I : MBBI) {
      SmallVector<unsigned int, 4> Regs;
      SmallVector<unsigned int, 4> Sizes;

      if (I.getOpcode() == Next32::CALL_TERMINATOR) {
        unsigned NumDefs = I.getNumDefs();
        for (unsigned i = 0; i < NumDefs; ++i) {
          // Collect all registers and sizes needed for feeders
          // generation.
          Regs.push_back(I.getOperand(i).getReg());
          Sizes.push_back(I.getOperand(i + NumDefs).getImm());
        }
      } else if (I.getOpcode() == Next32::CALL_TERMINATOR_TID) {
        Regs.push_back(Next32::TID);
        Sizes.push_back(I.getOperand(0).getImm());
      } else
        continue;

      for (size_t i = 0; i < Regs.size(); i++) {
        unsigned Opcode = GetFeederOpcode(Regs[i], MBBI);
        BuildMI(MBBI, I, I.getDebugLoc(), TII->get(Opcode), Regs[i])
            .addReg(Regs[i])
            .addImm(Sizes[i]);
      }
      EraseFromMI.push_back(&I);
    }
  }

  for (auto &I : EraseFromMI)
    I->eraseFromParent();

  return true;
}

unsigned int Next32EliminateCallTerminatorsPass::GetFeederOpcode(
    uint16_t PhysReg, MachineBasicBlock &MBB) const {
  for (auto &I : MBB) {
    if (I.getOpcode() == Next32::CALL_TERMINATOR)
      continue;

    if (I.isCall() && PhysReg == Next32::TID)
      return Next32::FEEDER;

    for (auto &O : I.operands()) {
      if (!O.isReg())
        continue;
      if (O.getReg() == PhysReg)
        return Next32::FEEDER;
    }
  }
  return Next32::FEEDERP;
}
