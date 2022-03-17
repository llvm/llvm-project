//===-------------- BPFMIPeephole.cpp - MI Peephole Cleanups  -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs peephole optimizations to cleanup ugly code sequences at
// MachineInstruction layer.
//
// Currently, there are two optimizations implemented:
//  - One pre-RA MachineSSA pass to eliminate type promotion sequences, those
//    zero extend 32-bit subregisters to 64-bit registers, if the compiler
//    could prove the subregisters is defined by 32-bit operations in which
//    case the upper half of the underlying 64-bit registers were zeroed
//    implicitly.
//
//  - One post-RA PreEmit pass to do final cleanup on some redundant
//    instructions generated due to bad RA on subregister.
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFInstrInfo.h"
#include "BPFTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include <set>

using namespace llvm;

#define DEBUG_TYPE "bpf-mi-zext-elim"

STATISTIC(ZExtElemNum, "Number of zero extension shifts eliminated");

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

  bool isCopyFrom32Def(MachineInstr *CopyMI);
  bool isInsnFrom32Def(MachineInstr *DefInsn);
  bool isPhiFrom32Def(MachineInstr *MovMI);
  bool isMovFrom32Def(MachineInstr *MovMI);
  bool eliminateZExtSeq();
  bool eliminateZExt();

  std::set<MachineInstr *> PhiInsns;

public:

  // Main entry point for this pass.
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;

    initialize(MF);

    // First try to eliminate (zext, lshift, rshift) and then
    // try to eliminate zext.
    bool ZExtSeqExist, ZExtExist;
    ZExtSeqExist = eliminateZExtSeq();
    ZExtExist = eliminateZExt();
    return ZExtSeqExist || ZExtExist;
  }
};

// Initialize class variables.
void BPFMIPeephole::initialize(MachineFunction &MFParm) {
  MF = &MFParm;
  MRI = &MF->getRegInfo();
  TII = MF->getSubtarget<BPFSubtarget>().getInstrInfo();
  LLVM_DEBUG(dbgs() << "*** BPF MachineSSA ZEXT Elim peephole pass ***\n\n");
}

bool BPFMIPeephole::isCopyFrom32Def(MachineInstr *CopyMI)
{
  MachineOperand &opnd = CopyMI->getOperand(1);

  if (!opnd.isReg())
    return false;

  // Return false if getting value from a 32bit physical register.
  // Most likely, this physical register is aliased to
  // function call return value or current function parameters.
  Register Reg = opnd.getReg();
  if (!Register::isVirtualRegister(Reg))
    return false;

  if (MRI->getRegClass(Reg) == &BPF::GPRRegClass)
    return false;

  MachineInstr *DefInsn = MRI->getVRegDef(Reg);
  if (!isInsnFrom32Def(DefInsn))
    return false;

  return true;
}

bool BPFMIPeephole::isPhiFrom32Def(MachineInstr *PhiMI)
{
  for (unsigned i = 1, e = PhiMI->getNumOperands(); i < e; i += 2) {
    MachineOperand &opnd = PhiMI->getOperand(i);

    if (!opnd.isReg())
      return false;

    MachineInstr *PhiDef = MRI->getVRegDef(opnd.getReg());
    if (!PhiDef)
      return false;
    if (PhiDef->isPHI()) {
      if (PhiInsns.find(PhiDef) != PhiInsns.end())
        return false;
      PhiInsns.insert(PhiDef);
      if (!isPhiFrom32Def(PhiDef))
        return false;
    }
    if (PhiDef->getOpcode() == BPF::COPY && !isCopyFrom32Def(PhiDef))
      return false;
  }

  return true;
}

// The \p DefInsn instruction defines a virtual register.
bool BPFMIPeephole::isInsnFrom32Def(MachineInstr *DefInsn)
{
  if (!DefInsn)
    return false;

  if (DefInsn->isPHI()) {
    if (PhiInsns.find(DefInsn) != PhiInsns.end())
      return false;
    PhiInsns.insert(DefInsn);
    if (!isPhiFrom32Def(DefInsn))
      return false;
  } else if (DefInsn->getOpcode() == BPF::COPY) {
    if (!isCopyFrom32Def(DefInsn))
      return false;
  }

  return true;
}

bool BPFMIPeephole::isMovFrom32Def(MachineInstr *MovMI)
{
  MachineInstr *DefInsn = MRI->getVRegDef(MovMI->getOperand(1).getReg());

  LLVM_DEBUG(dbgs() << "  Def of Mov Src:");
  LLVM_DEBUG(DefInsn->dump());

  PhiInsns.clear();
  if (!isInsnFrom32Def(DefInsn))
    return false;

  LLVM_DEBUG(dbgs() << "  One ZExt elim sequence identified.\n");

  return true;
}

bool BPFMIPeephole::eliminateZExtSeq() {
  MachineInstr* ToErase = nullptr;
  bool Eliminated = false;

  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      // If the previous instruction was marked for elimination, remove it now.
      if (ToErase) {
        ToErase->eraseFromParent();
        ToErase = nullptr;
      }

      // Eliminate the 32-bit to 64-bit zero extension sequence when possible.
      //
      //   MOV_32_64 rB, wA
      //   SLL_ri    rB, rB, 32
      //   SRL_ri    rB, rB, 32
      if (MI.getOpcode() == BPF::SRL_ri &&
          MI.getOperand(2).getImm() == 32) {
        Register DstReg = MI.getOperand(0).getReg();
        Register ShfReg = MI.getOperand(1).getReg();
        MachineInstr *SllMI = MRI->getVRegDef(ShfReg);

        LLVM_DEBUG(dbgs() << "Starting SRL found:");
        LLVM_DEBUG(MI.dump());

        if (!SllMI ||
            SllMI->isPHI() ||
            SllMI->getOpcode() != BPF::SLL_ri ||
            SllMI->getOperand(2).getImm() != 32)
          continue;

        LLVM_DEBUG(dbgs() << "  SLL found:");
        LLVM_DEBUG(SllMI->dump());

        MachineInstr *MovMI = MRI->getVRegDef(SllMI->getOperand(1).getReg());
        if (!MovMI ||
            MovMI->isPHI() ||
            MovMI->getOpcode() != BPF::MOV_32_64)
          continue;

        LLVM_DEBUG(dbgs() << "  Type cast Mov found:");
        LLVM_DEBUG(MovMI->dump());

        Register SubReg = MovMI->getOperand(1).getReg();
        if (!isMovFrom32Def(MovMI)) {
          LLVM_DEBUG(dbgs()
                     << "  One ZExt elim sequence failed qualifying elim.\n");
          continue;
        }

        BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(BPF::SUBREG_TO_REG), DstReg)
          .addImm(0).addReg(SubReg).addImm(BPF::sub_32);

        SllMI->eraseFromParent();
        MovMI->eraseFromParent();
        // MI is the right shift, we can't erase it in it's own iteration.
        // Mark it to ToErase, and erase in the next iteration.
        ToErase = &MI;
        ZExtElemNum++;
        Eliminated = true;
      }
    }
  }

  return Eliminated;
}

bool BPFMIPeephole::eliminateZExt() {
  MachineInstr* ToErase = nullptr;
  bool Eliminated = false;

  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      // If the previous instruction was marked for elimination, remove it now.
      if (ToErase) {
        ToErase->eraseFromParent();
        ToErase = nullptr;
      }

      if (MI.getOpcode() != BPF::MOV_32_64)
        continue;

      // Eliminate MOV_32_64 if possible.
      //   MOV_32_64 rA, wB
      //
      // If wB has been zero extended, replace it with a SUBREG_TO_REG.
      // This is to workaround BPF programs where pkt->{data, data_end}
      // is encoded as u32, but actually the verifier populates them
      // as 64bit pointer. The MOV_32_64 will zero out the top 32 bits.
      LLVM_DEBUG(dbgs() << "Candidate MOV_32_64 instruction:");
      LLVM_DEBUG(MI.dump());

      if (!isMovFrom32Def(&MI))
        continue;

      LLVM_DEBUG(dbgs() << "Removing the MOV_32_64 instruction\n");

      Register dst = MI.getOperand(0).getReg();
      Register src = MI.getOperand(1).getReg();

      // Build a SUBREG_TO_REG instruction.
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(BPF::SUBREG_TO_REG), dst)
        .addImm(0).addReg(src).addImm(BPF::sub_32);

      ToErase = &MI;
      Eliminated = true;
    }
  }

  return Eliminated;
}

} // end default namespace

INITIALIZE_PASS(BPFMIPeephole, DEBUG_TYPE,
                "BPF MachineSSA Peephole Optimization For ZEXT Eliminate",
                false, false)

char BPFMIPeephole::ID = 0;
FunctionPass* llvm::createBPFMIPeepholePass() { return new BPFMIPeephole(); }

STATISTIC(RedundantMovElemNum, "Number of redundant moves eliminated");

namespace {

struct BPFMIPreEmitPeephole : public MachineFunctionPass {

  static char ID;
  MachineFunction *MF;
  const TargetRegisterInfo *TRI;

  BPFMIPreEmitPeephole() : MachineFunctionPass(ID) {
    initializeBPFMIPreEmitPeepholePass(*PassRegistry::getPassRegistry());
  }

private:
  // Initialize class variables.
  void initialize(MachineFunction &MFParm);

  bool eliminateRedundantMov();

public:

  // Main entry point for this pass.
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;

    initialize(MF);

    return eliminateRedundantMov();
  }
};

// Initialize class variables.
void BPFMIPreEmitPeephole::initialize(MachineFunction &MFParm) {
  MF = &MFParm;
  TRI = MF->getSubtarget<BPFSubtarget>().getRegisterInfo();
  LLVM_DEBUG(dbgs() << "*** BPF PreEmit peephole pass ***\n\n");
}

bool BPFMIPreEmitPeephole::eliminateRedundantMov() {
  MachineInstr* ToErase = nullptr;
  bool Eliminated = false;

  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      // If the previous instruction was marked for elimination, remove it now.
      if (ToErase) {
        LLVM_DEBUG(dbgs() << "  Redundant Mov Eliminated:");
        LLVM_DEBUG(ToErase->dump());
        ToErase->eraseFromParent();
        ToErase = nullptr;
      }

      // Eliminate identical move:
      //
      //   MOV rA, rA
      //
      // Note that we cannot remove
      //   MOV_32_64  rA, wA
      //   MOV_rr_32  wA, wA
      // as these two instructions having side effects, zeroing out
      // top 32 bits of rA.
      unsigned Opcode = MI.getOpcode();
      if (Opcode == BPF::MOV_rr) {
        Register dst = MI.getOperand(0).getReg();
        Register src = MI.getOperand(1).getReg();

        if (dst != src)
          continue;

        ToErase = &MI;
        RedundantMovElemNum++;
        Eliminated = true;
      }
    }
  }

  return Eliminated;
}

} // end default namespace

INITIALIZE_PASS(BPFMIPreEmitPeephole, "bpf-mi-pemit-peephole",
                "BPF PreEmit Peephole Optimization", false, false)

char BPFMIPreEmitPeephole::ID = 0;
FunctionPass* llvm::createBPFMIPreEmitPeepholePass()
{
  return new BPFMIPreEmitPeephole();
}

STATISTIC(TruncElemNum, "Number of truncation eliminated");

namespace {

struct BPFMIPeepholeTruncElim : public MachineFunctionPass {

  static char ID;
  const BPFInstrInfo *TII;
  MachineFunction *MF;
  MachineRegisterInfo *MRI;

  BPFMIPeepholeTruncElim() : MachineFunctionPass(ID) {
    initializeBPFMIPeepholeTruncElimPass(*PassRegistry::getPassRegistry());
  }

private:
  // Initialize class variables.
  void initialize(MachineFunction &MFParm);

  bool eliminateTruncSeq();

public:

  // Main entry point for this pass.
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;

    initialize(MF);

    return eliminateTruncSeq();
  }
};

static bool TruncSizeCompatible(int TruncSize, unsigned opcode)
{
  if (TruncSize == 1)
    return opcode == BPF::LDB || opcode == BPF::LDB32;

  if (TruncSize == 2)
    return opcode == BPF::LDH || opcode == BPF::LDH32;

  if (TruncSize == 4)
    return opcode == BPF::LDW || opcode == BPF::LDW32;

  return false;
}

// Initialize class variables.
void BPFMIPeepholeTruncElim::initialize(MachineFunction &MFParm) {
  MF = &MFParm;
  MRI = &MF->getRegInfo();
  TII = MF->getSubtarget<BPFSubtarget>().getInstrInfo();
  LLVM_DEBUG(dbgs() << "*** BPF MachineSSA TRUNC Elim peephole pass ***\n\n");
}

// Reg truncating is often the result of 8/16/32bit->64bit or
// 8/16bit->32bit conversion. If the reg value is loaded with
// masked byte width, the AND operation can be removed since
// BPF LOAD already has zero extension.
//
// This also solved a correctness issue.
// In BPF socket-related program, e.g., __sk_buff->{data, data_end}
// are 32-bit registers, but later on, kernel verifier will rewrite
// it with 64-bit value. Therefore, truncating the value after the
// load will result in incorrect code.
bool BPFMIPeepholeTruncElim::eliminateTruncSeq() {
  MachineInstr* ToErase = nullptr;
  bool Eliminated = false;

  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      // The second insn to remove if the eliminate candidate is a pair.
      MachineInstr *MI2 = nullptr;
      Register DstReg, SrcReg;
      MachineInstr *DefMI;
      int TruncSize = -1;

      // If the previous instruction was marked for elimination, remove it now.
      if (ToErase) {
        ToErase->eraseFromParent();
        ToErase = nullptr;
      }

      // AND A, 0xFFFFFFFF will be turned into SLL/SRL pair due to immediate
      // for BPF ANDI is i32, and this case only happens on ALU64.
      if (MI.getOpcode() == BPF::SRL_ri &&
          MI.getOperand(2).getImm() == 32) {
        SrcReg = MI.getOperand(1).getReg();
        if (!MRI->hasOneNonDBGUse(SrcReg))
          continue;

        MI2 = MRI->getVRegDef(SrcReg);
        DstReg = MI.getOperand(0).getReg();

        if (!MI2 ||
            MI2->getOpcode() != BPF::SLL_ri ||
            MI2->getOperand(2).getImm() != 32)
          continue;

        // Update SrcReg.
        SrcReg = MI2->getOperand(1).getReg();
        DefMI = MRI->getVRegDef(SrcReg);
        if (DefMI)
          TruncSize = 4;
      } else if (MI.getOpcode() == BPF::AND_ri ||
                 MI.getOpcode() == BPF::AND_ri_32) {
        SrcReg = MI.getOperand(1).getReg();
        DstReg = MI.getOperand(0).getReg();
        DefMI = MRI->getVRegDef(SrcReg);

        if (!DefMI)
          continue;

        int64_t imm = MI.getOperand(2).getImm();
        if (imm == 0xff)
          TruncSize = 1;
        else if (imm == 0xffff)
          TruncSize = 2;
      }

      if (TruncSize == -1)
        continue;

      // The definition is PHI node, check all inputs.
      if (DefMI->isPHI()) {
        bool CheckFail = false;

        for (unsigned i = 1, e = DefMI->getNumOperands(); i < e; i += 2) {
          MachineOperand &opnd = DefMI->getOperand(i);
          if (!opnd.isReg()) {
            CheckFail = true;
            break;
          }

          MachineInstr *PhiDef = MRI->getVRegDef(opnd.getReg());
          if (!PhiDef || PhiDef->isPHI() ||
              !TruncSizeCompatible(TruncSize, PhiDef->getOpcode())) {
            CheckFail = true;
            break;
          }
        }

        if (CheckFail)
          continue;
      } else if (!TruncSizeCompatible(TruncSize, DefMI->getOpcode())) {
        continue;
      }

      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(BPF::MOV_rr), DstReg)
              .addReg(SrcReg);

      if (MI2)
        MI2->eraseFromParent();

      // Mark it to ToErase, and erase in the next iteration.
      ToErase = &MI;
      TruncElemNum++;
      Eliminated = true;
    }
  }

  return Eliminated;
}

} // end default namespace

INITIALIZE_PASS(BPFMIPeepholeTruncElim, "bpf-mi-trunc-elim",
                "BPF MachineSSA Peephole Optimization For TRUNC Eliminate",
                false, false)

char BPFMIPeepholeTruncElim::ID = 0;
FunctionPass* llvm::createBPFMIPeepholeTruncElimPass()
{
  return new BPFMIPeepholeTruncElim();
}
