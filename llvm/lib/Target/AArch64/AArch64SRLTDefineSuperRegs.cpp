//===- AArch64SRLTDefineSuperRegs.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// When SubRegister Liveness Tracking (SRLT) is enabled, this pass adds
// extra implicit-def's to instructions that define the low N bits of
// a GPR/FPR register to also define the top bits, because all AArch64
// instructions that write the low bits of a GPR/FPR also implicitly zero
// the top bits.  For example, 'mov w0, w1' writes zeroes to the top 32-bits of
// x0, so this pass adds a `implicit-def $x0` after register allocation.
//
// These semantics are originally represented in the MIR using `SUBREG_TO_REG`
// which expresses that the top bits have been defined by the preceding
// instructions, but during register coalescing this information is lost and in
// contrast to when SRTL is disabled, when rewriting virtual -> physical
// registers the implicit-defs are not added to the instruction.
//
// There have been several attempts to fix this in the coalescer [1], but each
// iteration has exposed new bugs and the patch had to be reverted.
// Additionally, the concept of adding 'implicit-def' of a virtual register is
// particularly fragile and many places don't expect it (for example in
// `X86::commuteInstructionImpl` the  code only looks at specific operands and
// does not consider implicit-defs. Similar in `SplitEditor::addDeadDef` where
// it traverses operand 'defs' rather than 'all_defs').
//
// We want a temporary solution that doesn't impact other targets and is simpler
// and less intrusive than the patch proposed for the register coalescer [1], so
// that we can enable SRLT for AArch64.
//
// The approach here is to just add the 'implicit-def' manually after rewriting
// virtual regs -> phsyical regs. This still means that during the register
// allocation process the dependences are not accurately represented in the MIR
// and LiveIntervals, but there are several reasons why we believe this isn't a
// problem in practice:
// (A) The register allocator only spills entire virtual registers.
//     This is additionally guarded by code in
//     AArch64InstrInfo::storeRegToStackSlot/loadRegFromStackSlot
//     where it checks if a register matches the expected register class.
// (B) Rematerialization only happens when the instruction writes the full
//     register.
// (C) The high bits of the AArch64 register cannot be written independently.
// (D) Instructions that write only part of a register always take that same
//     register as a tied input operand, to indicate it's a merging operation.
//
// (A) means that for two virtual registers of regclass GPR32 and GPR64, if the
// GPR32 register is coalesced into the GPR64 vreg then the full GPR64 would
// be spilled/filled even if only the low 32-bits would be required for the
// given liverange. (B) means that the top bits of a GPR64 would never be
// overwritten by rematerialising a GPR32 sub-register for a given liverange.
// (C-D) means that we can assume that the MIR as input to the register
// allocator correctly expresses the instruction behaviour and dependences
// between values, so unless the register allocator would violate (A) or (B),
// the MIR is otherwise sound.
//
// Alternative approaches have also been considered, such as:
// (1) Changing the AArch64 instruction definitions to write all bits and
//     extract the low N bits for the result.
// (2) Disabling coalescing of SUBREG_TO_REG and using regalloc hints to tell
//     the register allocator to favour the same register for the input/output.
// (3) Adding a new coalescer guard node with a tied-operand constraint, such
//     that when the SUBREG_TO_REG is removed, something still represents that
//     the top bits are defined. The node would get removed before rewriting
//     virtregs.
// (4) Using an explicit INSERT_SUBREG into a zero value and try to optimize
//     away the INSERT_SUBREG (this is a more explicit variant of (2) and (3))
// (5) Adding a new MachineOperand flag that represents the top bits would be
//     defined, but are not read nor undef.
//
// (1) would be the best approach but would be a significant effort as it
// requires rewriting most/all instruction definitions and fixing MIR passes
// that rely on the current definitions, whereas (2-4) result in sub-optimal
// code that can't really be avoided because the explicit nodes would stop
// rematerialization. (5) might be a way to mitigate the
// fragility of implicit-def's of virtual registers if we want to pursue
// landing [1], but then we'd rather choose approach (1) to avoid using
// SUBREG_TO_REG entirely.
//
// [1] https://github.com/llvm/llvm-project/pull/168353
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-srlt-define-superregs"
#define PASS_NAME "AArch64 SRLT Define Super-Regs Pass"

namespace {

struct AArch64SRLTDefineSuperRegs : public MachineFunctionPass {
  inline static char ID = 0;

  AArch64SRLTDefineSuperRegs() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  Register getWidestSuperReg(Register R, const BitVector &RequiredBaseRegUnits,
                             const BitVector &QHiRegUnits);

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreservedID(MachineLoopInfoID);
    AU.addPreservedID(MachineDominatorsID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  MachineFunction *MF = nullptr;
  const AArch64Subtarget *Subtarget = nullptr;
  const AArch64RegisterInfo *TRI = nullptr;
};

} // end anonymous namespace

INITIALIZE_PASS(AArch64SRLTDefineSuperRegs, DEBUG_TYPE, PASS_NAME, false, false)

// Returns the widest super-reg for a given reg, or NoRegister if no suitable
// wider super-reg has been found. For example:
//  W0    -> X0
//  B1    -> Q1 (without SVE)
//        -> Z1 (with SVE)
//  W1_W2 -> X1_X2
//  D0_D1 -> Q0_Q1 (without SVE)
//        -> Z0_Z1 (with SVE)
Register AArch64SRLTDefineSuperRegs::getWidestSuperReg(
    Register R, const BitVector &RequiredBaseRegUnits,
    const BitVector &QHiRegUnits) {
  assert(R.isPhysical() &&
         "Expected to be run straight after virtregrewriter!");

  BitVector Units(TRI->getNumRegUnits());
  for (MCRegUnit U : TRI->regunits(R))
    Units.set((unsigned)U);

  auto IsSuitableSuperReg = [&](Register SR) {
    for (MCRegUnit U : TRI->regunits(SR)) {
      // Avoid choosing z1 as super-reg of d1 if SVE is not available.
      // Q*_HI registers are only set for SVE registers, as those consist
      // of the Q* register for the low 128 bits and the Q*_HI (artificial)
      // register for the top (vscale-1) * 128 bits.
      if (QHiRegUnits.test((unsigned)U) &&
          !Subtarget->isSVEorStreamingSVEAvailable())
        return false;
      // We consider a super-reg as unsuitable if any of its reg units is not
      // artificial and not shared, as that would imply that U is a unit for a
      // different register, which means the candidate super-reg is likely
      // a register tuple.
      if (!TRI->isArtificialRegUnit(U) &&
          (!Units.test((unsigned)U) || !RequiredBaseRegUnits.test((unsigned)U)))
        return false;
    }
    return true;
  };

  Register LargestSuperReg = AArch64::NoRegister;
  for (Register SR : TRI->superregs(R))
    if (IsSuitableSuperReg(SR) && (LargestSuperReg == AArch64::NoRegister ||
                                   TRI->isSuperRegister(LargestSuperReg, SR)))
      LargestSuperReg = SR;

  return LargestSuperReg;
}

bool AArch64SRLTDefineSuperRegs::runOnMachineFunction(MachineFunction &MF) {
  this->MF = &MF;
  Subtarget = &MF.getSubtarget<AArch64Subtarget>();
  TRI = Subtarget->getRegisterInfo();
  const MachineRegisterInfo *MRI = &MF.getRegInfo();

  if (!MRI->subRegLivenessEnabled())
    return false;

  assert(!MRI->isSSA() && "Expected to be run after breaking down SSA form!");

  auto XRegs = seq_inclusive<unsigned>(AArch64::X0, AArch64::X28);
  auto ZRegs = seq_inclusive<unsigned>(AArch64::Z0, AArch64::Z31);
  constexpr unsigned FixedRegs[] = {AArch64::FP, AArch64::LR, AArch64::SP};

  BitVector RequiredBaseRegUnits(TRI->getNumRegUnits());
  for (Register R : concat<unsigned>(XRegs, ZRegs, FixedRegs))
    for (MCRegUnit U : TRI->regunits(R))
      RequiredBaseRegUnits.set((unsigned)U);

  BitVector QHiRegUnits(TRI->getNumRegUnits());
  for (Register R : seq_inclusive<unsigned>(AArch64::Q0_HI, AArch64::Q31_HI))
    for (MCRegUnit U : TRI->regunits(R))
      QHiRegUnits.set((unsigned)U);

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // PATCHPOINT may have a 'def' that's not a register, avoid this.
      if (MI.getOpcode() == TargetOpcode::PATCHPOINT)
        continue;
      // For each partial register write, also add an implicit-def for top bits
      // of the register (e.g. for w0 add a def of x0).
      SmallSet<Register, 8> SuperRegs;
      for (const MachineOperand &DefOp : MI.defs())
        if (Register R = getWidestSuperReg(DefOp.getReg(), RequiredBaseRegUnits,
                                           QHiRegUnits);
            R != AArch64::NoRegister)
          SuperRegs.insert(R);

      if (!SuperRegs.size())
        continue;

      LLVM_DEBUG(dbgs() << "Adding implicit-defs to: " << MI);
      for (Register R : SuperRegs) {
        LLVM_DEBUG(dbgs() << "  " << printReg(R, TRI) << "\n");
        bool IsRenamable = any_of(MI.defs(), [&](const MachineOperand &MO) {
          return MO.isRenamable() && TRI->regsOverlap(MO.getReg(), R);
        });
        bool IsDead = any_of(MI.defs(), [&](const MachineOperand &MO) {
          return MO.isDead() && TRI->regsOverlap(MO.getReg(), R);
        });
        MachineOperand DefOp = MachineOperand::CreateReg(
            R, /*isDef=*/true, /*isImp=*/true, /*isKill=*/false,
            /*isDead=*/IsDead, /*isUndef=*/false, /*isEarlyClobber=*/false,
            /*SubReg=*/0, /*isDebug=*/false, /*isInternalRead=*/false,
            /*isRenamable=*/IsRenamable);
        MI.addOperand(DefOp);
      }
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createAArch64SRLTDefineSuperRegsPass() {
  return new AArch64SRLTDefineSuperRegs();
}
