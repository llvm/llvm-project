//===-- SIShrinkInstructions.cpp - Shrink Instructions --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// The pass tries to use the 32-bit encoding for instructions when possible.
//===----------------------------------------------------------------------===//
//

#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "si-shrink-instructions"

STATISTIC(NumInstructionsShrunk,
          "Number of 64-bit instruction reduced to 32-bit.");

namespace llvm {
  void initializeSIShrinkInstructionsPass(PassRegistry&);
}

using namespace llvm;

namespace {

class SIShrinkInstructions : public MachineFunctionPass {
public:
  static char ID;

public:
  SIShrinkInstructions() : MachineFunctionPass(ID) {
  }

  virtual bool runOnMachineFunction(MachineFunction &MF) override;

  virtual const char *getPassName() const override {
    return "SI Shrink Instructions";
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIShrinkInstructions, DEBUG_TYPE,
                      "SI Lower il Copies", false, false)
INITIALIZE_PASS_END(SIShrinkInstructions, DEBUG_TYPE,
                    "SI Lower il Copies", false, false)

char SIShrinkInstructions::ID = 0;

FunctionPass *llvm::createSIShrinkInstructionsPass() {
  return new SIShrinkInstructions();
}

static bool isVGPR(const MachineOperand *MO, const SIRegisterInfo &TRI,
                   const MachineRegisterInfo &MRI) {
  if (!MO->isReg())
    return false;

  if (TargetRegisterInfo::isVirtualRegister(MO->getReg()))
    return TRI.hasVGPRs(MRI.getRegClass(MO->getReg()));

  return TRI.hasVGPRs(TRI.getPhysRegClass(MO->getReg()));
}

static bool canShrink(MachineInstr &MI, const SIInstrInfo *TII,
                      const SIRegisterInfo &TRI,
                      const MachineRegisterInfo &MRI) {

  const MachineOperand *Src2 = TII->getNamedOperand(MI, AMDGPU::OpName::src2);
  // Can't shrink instruction with three operands.
  if (Src2)
    return false;

  const MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
  const MachineOperand *Src1Mod =
      TII->getNamedOperand(MI, AMDGPU::OpName::src1_modifiers);

  if (Src1 && (!isVGPR(Src1, TRI, MRI) || Src1Mod->getImm() != 0))
    return false;

  // We don't need to check src0, all input types are legal, so just make
  // sure src0 isn't using any modifiers.
  const MachineOperand *Src0Mod =
      TII->getNamedOperand(MI, AMDGPU::OpName::src0_modifiers);
  if (Src0Mod && Src0Mod->getImm() != 0)
    return false;

  // Check output modifiers
  const MachineOperand *Omod = TII->getNamedOperand(MI, AMDGPU::OpName::omod);
  if (Omod && Omod->getImm() != 0)
    return false;

  const MachineOperand *Clamp = TII->getNamedOperand(MI, AMDGPU::OpName::clamp);
  return !Clamp || Clamp->getImm() == 0;
}

bool SIShrinkInstructions::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(
      MF.getTarget().getInstrInfo());
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  std::vector<unsigned> I1Defs;

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
                                                  BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    MachineBasicBlock::iterator I, Next;
    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);
      MachineInstr &MI = *I;

      if (!TII->hasVALU32BitEncoding(MI.getOpcode()))
        continue;

      if (!canShrink(MI, TII, TRI, MRI)) {
        // Try commtuing the instruction and see if that enables us to shrink
        // it.
        if (!MI.isCommutable() || !TII->commuteInstruction(&MI) ||
            !canShrink(MI, TII, TRI, MRI))
          continue;
      }

      int Op32 = AMDGPU::getVOPe32(MI.getOpcode());

      // Op32 could be -1 here if we started with an instruction that had a
      // a 32-bit encoding and then commuted it to an instruction that did not.
      if (Op32 == -1)
        continue;

      if (TII->isVOPC(Op32)) {
        unsigned DstReg = MI.getOperand(0).getReg();
        if (TargetRegisterInfo::isVirtualRegister(DstReg)) {
          // VOPC instructions can only write to the VCC register.  We can't
          // force them to use VCC here, because the register allocator
          // has trouble with sequences like this, which cause the allocator
          // to run out of registes if vreg0 and vreg1 belong to the VCCReg
          // register class:
          // vreg0 = VOPC;
          // vreg1 = VOPC;
          // S_AND_B64 vreg0, vreg1
          //
          // So, instead of forcing the instruction to write to VCC, we provide a
          // hint to the register allocator to use VCC and then we
          // we will run this pass again after RA and shrink it if it outpus to
          // VCC.
          MRI.setRegAllocationHint(MI.getOperand(0).getReg(), 0, AMDGPU::VCC);
          continue;
        }
        if (DstReg != AMDGPU::VCC)
          continue;
      }

      // We can shrink this instruction
      DEBUG(dbgs() << "Shrinking "; MI.dump(); dbgs() << "\n";);

      MachineInstrBuilder MIB =
          BuildMI(MBB, I, MI.getDebugLoc(), TII->get(Op32));

      // dst
      MIB.addOperand(MI.getOperand(0));

      MIB.addOperand(*TII->getNamedOperand(MI, AMDGPU::OpName::src0));

      const MachineOperand *Src1 =
          TII->getNamedOperand(MI, AMDGPU::OpName::src1);
      if (Src1)
        MIB.addOperand(*Src1);

      for (const MachineOperand &MO : MI.implicit_operands())
        MIB.addOperand(MO);

      DEBUG(dbgs() << "e32 MI = "; MI.dump(); dbgs() << "\n";);
      ++NumInstructionsShrunk;
      MI.eraseFromParent();
    }
  }
  return false;
}
