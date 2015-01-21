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
#include "AMDGPUMCInstLower.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "si-shrink-instructions"

STATISTIC(NumInstructionsShrunk,
          "Number of 64-bit instruction reduced to 32-bit.");
STATISTIC(NumLiteralConstantsFolded,
          "Number of literal constants folded into 32-bit instructions.");

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

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Shrink Instructions";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
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

  if (Src1 && (!isVGPR(Src1, TRI, MRI) || (Src1Mod && Src1Mod->getImm() != 0)))
    return false;

  // We don't need to check src0, all input types are legal, so just make sure
  // src0 isn't using any modifiers.
  if (TII->hasModifiersSet(MI, AMDGPU::OpName::src0_modifiers))
    return false;

  // Check output modifiers
  if (TII->hasModifiersSet(MI, AMDGPU::OpName::omod))
    return false;

  if (TII->hasModifiersSet(MI, AMDGPU::OpName::clamp))
    return false;

  return true;
}

/// \brief This function checks \p MI for operands defined by a move immediate
/// instruction and then folds the literal constant into the instruction if it
/// can.  This function assumes that \p MI is a VOP1, VOP2, or VOPC instruction
/// and will only fold literal constants if we are still in SSA.
static void foldImmediates(MachineInstr &MI, const SIInstrInfo *TII,
                           MachineRegisterInfo &MRI, bool TryToCommute = true) {

  if (!MRI.isSSA())
    return;

  assert(TII->isVOP1(MI.getOpcode()) || TII->isVOP2(MI.getOpcode()) ||
         TII->isVOPC(MI.getOpcode()));

  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);

  // Only one literal constant is allowed per instruction, so if src0 is a
  // literal constant then we can't do any folding.
  if (Src0->isImm() && TII->isLiteralConstant(*Src0))
    return;


  // Literal constants and SGPRs can only be used in Src0, so if Src0 is an
  // SGPR, we cannot commute the instruction, so we can't fold any literal
  // constants.
  if (Src0->isReg() && !isVGPR(Src0, TRI, MRI))
    return;

  // Try to fold Src0
  if (Src0->isReg()) {
    unsigned Reg = Src0->getReg();
    MachineInstr *Def = MRI.getUniqueVRegDef(Reg);
    if (Def && Def->isMoveImmediate()) {
      MachineOperand &MovSrc = Def->getOperand(1);
      bool ConstantFolded = false;

      if (MovSrc.isImm() && isUInt<32>(MovSrc.getImm())) {
        Src0->ChangeToImmediate(MovSrc.getImm());
        ConstantFolded = true;
      }
      if (ConstantFolded) {
        if (MRI.use_empty(Reg))
          Def->eraseFromParent();
        ++NumLiteralConstantsFolded;
        return;
      }
    }
  }

  // We have failed to fold src0, so commute the instruction and try again.
  if (TryToCommute && MI.isCommutable() && TII->commuteInstruction(&MI))
    foldImmediates(MI, TII, MRI, false);

}

bool SIShrinkInstructions::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  std::vector<unsigned> I1Defs;

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
                                                  BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    MachineBasicBlock::iterator I, Next;
    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);
      MachineInstr &MI = *I;

      // Try to use S_MOVK_I32, which will save 4 bytes for small immediates.
      if (MI.getOpcode() == AMDGPU::S_MOV_B32) {
        const MachineOperand &Src = MI.getOperand(1);

        if (Src.isImm()) {
          if (isInt<16>(Src.getImm()) && !TII->isInlineConstant(Src))
            MI.setDesc(TII->get(AMDGPU::S_MOVK_I32));
        }

        continue;
      }

      if (!TII->hasVALU32BitEncoding(MI.getOpcode()))
        continue;

      if (!canShrink(MI, TII, TRI, MRI)) {
        // Try commuting the instruction and see if that enables us to shrink
        // it.
        if (!MI.isCommutable() || !TII->commuteInstruction(&MI) ||
            !canShrink(MI, TII, TRI, MRI))
          continue;
      }

      // getVOPe32 could be -1 here if we started with an instruction that had
      // a 32-bit encoding and then commuted it to an instruction that did not.
      if (!TII->hasVALU32BitEncoding(MI.getOpcode()))
        continue;

      int Op32 = AMDGPU::getVOPe32(MI.getOpcode());

      if (TII->isVOPC(Op32)) {
        unsigned DstReg = MI.getOperand(0).getReg();
        if (TargetRegisterInfo::isVirtualRegister(DstReg)) {
          // VOPC instructions can only write to the VCC register.  We can't
          // force them to use VCC here, because the register allocator has
          // trouble with sequences like this, which cause the allocator to run
          // out of registers if vreg0 and vreg1 belong to the VCCReg register
          // class:
          // vreg0 = VOPC;
          // vreg1 = VOPC;
          // S_AND_B64 vreg0, vreg1
          //
          // So, instead of forcing the instruction to write to VCC, we provide
          // a hint to the register allocator to use VCC and then we we will run
          // this pass again after RA and shrink it if it outputs to VCC.
          MRI.setRegAllocationHint(MI.getOperand(0).getReg(), 0, AMDGPU::VCC);
          continue;
        }
        if (DstReg != AMDGPU::VCC)
          continue;
      }

      // We can shrink this instruction
      DEBUG(dbgs() << "Shrinking "; MI.dump(); dbgs() << '\n';);

      MachineInstrBuilder Inst32 =
          BuildMI(MBB, I, MI.getDebugLoc(), TII->get(Op32));

      // dst
      Inst32.addOperand(MI.getOperand(0));

      Inst32.addOperand(*TII->getNamedOperand(MI, AMDGPU::OpName::src0));

      const MachineOperand *Src1 =
          TII->getNamedOperand(MI, AMDGPU::OpName::src1);
      if (Src1)
        Inst32.addOperand(*Src1);

      ++NumInstructionsShrunk;
      MI.eraseFromParent();

      foldImmediates(*Inst32, TII, MRI);
      DEBUG(dbgs() << "e32 MI = " << *Inst32 << '\n');


    }
  }
  return false;
}
