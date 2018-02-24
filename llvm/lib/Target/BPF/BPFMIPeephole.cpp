//===-------------- BPFMIPeephole.cpp - MI Peephole Cleanups  -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs peephole optimizations to cleanup ugly code sequences at
// MachineInstruction layer.
//
// Currently, the only optimization in this pass is to eliminate type promotion
// sequences, those zero extend 32-bit subregisters to 64-bit registers, if the
// compiler could prove the subregisters is defined by 32-bit operations in
// which case the upper half of the underlying 64-bit registers were zeroed
// implicitly.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFInstrInfo.h"
#include "BPFTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-mi-promotion-elim"

STATISTIC(CmpPromotionElemNum, "Number of shifts for CMP promotion eliminated");

namespace {

struct BPFMIPeephole : public MachineFunctionPass {

  static char ID;
  const BPFInstrInfo *TII;
  MachineFunction *MF;
  MachineRegisterInfo *MRI;

  BPFMIPeephole() : MachineFunctionPass(ID) {
    initializeBPFMIPeepholePass(*PassRegistry::getPassRegistry());
  }

private:
  // Initialize class variables.
  void initialize(MachineFunction &MFParm);

  bool eliminateCmpPromotionSeq(void);
  MachineInstr *getInsnDefZExtSubReg(unsigned Reg) const;
  void updateInsnSeq(MachineBasicBlock &MBB, MachineInstr &MI,
                     unsigned Reg) const;

public:

  // Main entry point for this pass.
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;

    initialize(MF);

    return eliminateCmpPromotionSeq();
  }
};

// Initialize class variables.
void BPFMIPeephole::initialize(MachineFunction &MFParm) {
  MF = &MFParm;
  MRI = &MF->getRegInfo();
  TII = MF->getSubtarget<BPFSubtarget>().getInstrInfo();
  DEBUG(dbgs() << "*** BPF MI peephole pass ***\n\n");
}

MachineInstr *BPFMIPeephole::getInsnDefZExtSubReg(unsigned Reg) const {
  MachineInstr *Insn = MRI->getVRegDef(Reg);

  if (!Insn ||
      Insn->isPHI() ||
      Insn->getOpcode() != BPF::SRL_ri ||
      Insn->getOperand(2).getImm() != 32)
    return nullptr;

  Insn = MRI->getVRegDef(Insn->getOperand(1).getReg());
  if (!Insn ||
      Insn->isPHI() ||
      Insn->getOpcode() != BPF::SLL_ri ||
      Insn->getOperand(2).getImm() != 32)
    return nullptr;

  Insn = MRI->getVRegDef(Insn->getOperand(1).getReg());
  if (!Insn ||
      Insn->isPHI() ||
      Insn->getOpcode() != BPF::MOV_32_64)
    return nullptr;

  return Insn;
}

void
BPFMIPeephole::updateInsnSeq(MachineBasicBlock &MBB, MachineInstr &MI,
                             unsigned Reg) const {
  MachineInstr *Mov, *Lshift, *Rshift;
  unsigned SubReg;
  DebugLoc DL;

  Rshift = MRI->getVRegDef(Reg);
  Lshift = MRI->getVRegDef(Rshift->getOperand(1).getReg());
  Mov = MRI->getVRegDef(Lshift->getOperand(1).getReg());
  SubReg = Mov->getOperand(1).getReg();
  DL = MI.getDebugLoc();
  BuildMI(MBB, Rshift, DL, TII->get(BPF::SUBREG_TO_REG), Reg)
    .addImm(0).addReg(SubReg).addImm(BPF::sub_32);
  Rshift->eraseFromParent();
  Lshift->eraseFromParent();
  Mov->eraseFromParent();

  CmpPromotionElemNum++;
}

bool BPFMIPeephole::eliminateCmpPromotionSeq(void) {
  bool Eliminated = false;
  MachineInstr *Mov;
  unsigned Reg;

  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      switch (MI.getOpcode()) {
      default:
        break;
      case BPF::JUGT_rr:
      case BPF::JUGE_rr:
      case BPF::JULT_rr:
      case BPF::JULE_rr:
      case BPF::JEQ_rr:
      case BPF::JNE_rr:
        Reg = MI.getOperand(1).getReg();
        Mov = getInsnDefZExtSubReg(Reg);
        if (!Mov)
          break;

	updateInsnSeq(MBB, MI, Reg);
	Eliminated = true;

        // Fallthrough
      case BPF::JUGT_ri:
      case BPF::JUGE_ri:
      case BPF::JULT_ri:
      case BPF::JULE_ri:
      case BPF::JEQ_ri:
      case BPF::JNE_ri:
        Reg = MI.getOperand(0).getReg();
        Mov = getInsnDefZExtSubReg(Reg);
        if (!Mov)
          break;

	updateInsnSeq(MBB, MI, Reg);
	Eliminated = true;
        break;
      }
    }
  }

  return Eliminated;
}

} // end default namespace

INITIALIZE_PASS(BPFMIPeephole, DEBUG_TYPE, "BPF MI Peephole Optimization",
                false, false)

char BPFMIPeephole::ID = 0;
FunctionPass* llvm::createBPFMIPeepholePass() { return new BPFMIPeephole(); }
