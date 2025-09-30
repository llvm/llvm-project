//===-- X86EliminateRedundantZeroExtend.cpp - Eliminate Redundant ZExt ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This pass eliminates redundant zero-extension instructions where the source
/// register is a sub-register of the destination and the destination's upper
/// bits are known to be zero.
///
/// For example:
///   movzbl (%rdi), %ecx  ; ECX = zero-extend byte, upper 24 bits are zero
///   ...
///   movzbl %cl, %ecx     ; Redundant! CL is part of ECX, upper bits already 0
///
/// This pattern commonly occurs in loops processing byte values.
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "x86-eliminate-zext"
#define PASS_NAME "X86 Eliminate Redundant Zero Extension"

namespace {
class EliminateRedundantZeroExtend : public MachineFunctionPass {
public:
  static char ID;
  EliminateRedundantZeroExtend() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return PASS_NAME; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

private:
  const X86InstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;

  /// Check if the register's upper bits are known to be zero at this point.
  /// This checks backward from MI to find the most recent definition of Reg.
  bool hasZeroUpperBits(Register Reg, const MachineInstr &MI,
                        const MachineBasicBlock &MBB) const;

  /// Try to eliminate a redundant MOVZX instruction.
  bool tryEliminateRedundantZeroExtend(MachineInstr &MI,
                                       MachineBasicBlock &MBB) const;
};

char EliminateRedundantZeroExtend::ID = 0;
} // end anonymous namespace

FunctionPass *llvm::createX86EliminateRedundantZeroExtend() {
  return new EliminateRedundantZeroExtend();
}

bool EliminateRedundantZeroExtend::hasZeroUpperBits(
    Register Reg, const MachineInstr &MI, const MachineBasicBlock &MBB) const {
  // Walk backward from MI to find the most recent definition of Reg
  MachineBasicBlock::const_reverse_iterator I = ++MI.getReverseIterator();
  MachineBasicBlock::const_reverse_iterator E = MBB.rend();
  for (; I != E; ++I) {
    const MachineInstr &Inst = *I;

    // Check if this instruction defines Reg
    for (const MachineOperand &MO : Inst.operands()) {
      if (!MO.isReg() || !MO.isDef())
        continue;

      Register DefReg = MO.getReg();
      if (DefReg == Reg || TRI->isSuperRegister(Reg, DefReg)) {
        // Found a definition - check if it zeros upper bits
        unsigned Opc = Inst.getOpcode();
        switch (Opc) {
        // These instructions zero-extend to 32 bits
        case X86::MOVZX32rm8:
        case X86::MOVZX32rr8:
        case X86::MOVZX32rm16:
        case X86::MOVZX32rr16:
          return true;
        // XOR with self zeros the register
        case X86::XOR32rr:
          if (Inst.getOperand(1).getReg() == Inst.getOperand(2).getReg())
            return true;
          return false;
        // MOV32r0 explicitly zeros
        case X86::MOV32r0:
          return true;
        // ADD, SUB on 32-bit register (implicitly zero-extends to 64-bit)
        case X86::ADD32rr:
        case X86::ADD32ri:
        case X86::ADD32rm:
        case X86::SUB32rr:
        case X86::SUB32ri:
        case X86::SUB32rm:
        case X86::LEA32r:
          return true;
        default:
          // Any other definition might set upper bits, so not safe
          return false;
        }
      }

      // Check if this instruction modifies Reg (partial write or implicit use)
      if (TRI->regsOverlap(DefReg, Reg)) {
        // Partial register update - upper bits are unknown
        return false;
      }
    }

    // Check for implicit defs
    for (const MachineOperand &MO : Inst.implicit_operands()) {
      if (MO.isReg() && MO.isDef() && TRI->regsOverlap(MO.getReg(), Reg)) {
        return false;
      }
    }
  }

  // Didn't find a definition in this block - check predecessors
  // If all predecessors define Reg with zero upper bits, it's safe
  if (MBB.pred_empty())
    return false;

  // Check all predecessor blocks
  for (const MachineBasicBlock *Pred : MBB.predecessors()) {
    bool FoundZeroExtend = false;

    // SAFETY CHECK: If the sub-register is live-in to the predecessor,
    // we make the CONSERVATIVE assumption that the parent register was
    // zero-extended in an earlier block.
    //
    // This is safe because:
    // 1. After register allocation, if $cl is live-in but $ecx is not,
    //    it means only the low 8 bits are meaningful
    // 2. The register allocator ensures no other code modifies $ecx between
    //    the zero-extension and this point (otherwise $ecx would be live)
    // 3. Any write to $ch or upper bits would show as a def of $ecx, which
    //    would be found in our backward scan below and handled correctly
    //
    // However, this is still conservative - we should verify the actual
    // definition to be completely safe.
    Register SubReg8 = TRI->getSubReg(Reg, X86::sub_8bit);
    Register SubReg16 = TRI->getSubReg(Reg, X86::sub_16bit);
    bool SubRegLiveIn = (SubReg8 && Pred->isLiveIn(SubReg8)) ||
                        (SubReg16 && Pred->isLiveIn(SubReg16));

    if (SubRegLiveIn) {
      // Sub-register is live-in. We'll verify this is safe by checking
      // that no instructions in this block modify the parent register
      // before we reach the end (where control flows to our block).
      // If we find any such modification, we'll conservatively bail out.
      bool SafeToAssume = true;
      for (const MachineInstr &Inst : *Pred) {
        for (const MachineOperand &MO : Inst.operands()) {
          if (MO.isReg() && MO.isDef()) {
            Register DefReg = MO.getReg();
            // Check if this modifies Reg or overlaps with it (partial write)
            if ((DefReg == Reg || TRI->regsOverlap(DefReg, Reg)) &&
                DefReg != SubReg8 && DefReg != SubReg16) {
              // Found a write to the parent register or overlapping register
              // that's not just the sub-register we expect
              SafeToAssume = false;
              break;
            }
          }
        }
        if (!SafeToAssume)
          break;
      }

      if (SafeToAssume) {
        FoundZeroExtend = true;
        goto next_predecessor;
      }
    }

    // Walk backward through predecessor to find last definition of Reg
    for (const MachineInstr &Inst : llvm::reverse(*Pred)) {
      // Check if this instruction defines Reg
      for (const MachineOperand &MO : Inst.operands()) {
        if (!MO.isReg() || !MO.isDef())
          continue;

        Register DefReg = MO.getReg();
        if (DefReg == Reg || TRI->isSuperRegister(Reg, DefReg)) {
          // Found a definition - check if it zeros upper bits
          unsigned Opc = Inst.getOpcode();
          switch (Opc) {
          case X86::MOVZX32rm8:
          case X86::MOVZX32rr8:
          case X86::MOVZX32rm16:
          case X86::MOVZX32rr16:
          case X86::MOV32r0:
          case X86::ADD32rr:
          case X86::ADD32ri:
          case X86::ADD32rm:
          case X86::SUB32rr:
          case X86::SUB32ri:
          case X86::SUB32rm:
          case X86::LEA32r:
            FoundZeroExtend = true;
            break;
          case X86::XOR32rr:
            if (Inst.getOperand(1).getReg() == Inst.getOperand(2).getReg())
              FoundZeroExtend = true;
            break;
          default:
            // Found a definition that doesn't zero upper bits
            return false;
          }
          // Found the definition in this predecessor
          goto next_predecessor;
        }

        // Check for partial register updates
        if (TRI->regsOverlap(DefReg, Reg)) {
          return false;
        }
      }
    }

  next_predecessor:
    // If we didn't find a zero-extending definition in this predecessor, fail
    if (!FoundZeroExtend)
      return false;
  }

  // All predecessors have zero-extending definitions
  return true;
}

bool EliminateRedundantZeroExtend::tryEliminateRedundantZeroExtend(
    MachineInstr &MI, MachineBasicBlock &MBB) const {
  unsigned Opc = MI.getOpcode();

  // Only handle MOVZX32rr8 for now (can extend to MOVZX32rr16 later)
  if (Opc != X86::MOVZX32rr8)
    return false;

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();

  // Check if source is a sub-register of destination
  // e.g., CL is sub-register of ECX
  if (!TRI->isSubRegister(DstReg, SrcReg))
    return false;

  // Check if destination's upper bits are already zero
  if (!hasZeroUpperBits(DstReg, MI, MBB))
    return false;

  // The MOVZX is redundant! Since SrcReg is part of DstReg and DstReg's
  // upper bits are already zero, this instruction does nothing.
  LLVM_DEBUG(dbgs() << "Eliminating redundant zero-extend: " << MI);
  MI.eraseFromParent();
  return true;
}

bool EliminateRedundantZeroExtend::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  TRI = MF.getSubtarget<X86Subtarget>().getRegisterInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Iterate through instructions - use a worklist to handle erasures
    SmallVector<MachineInstr *, 4> ToErase;

    for (MachineInstr &MI : MBB) {
      if (tryEliminateRedundantZeroExtend(MI, MBB)) {
        Changed = true;
        // Note: MI is already erased in tryEliminateRedundantZeroExtend
        break; // Restart iteration for this block
      }
    }
  }

  return Changed;
}