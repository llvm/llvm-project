//===- RISCVCopyCombine.cpp - Remove special copy for RISC-V --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass attempts a shrink-wrap optimization for special cases, which is
// effective when data types require extension.
//
// After finalize-isel:
//   bb0:
//   liveins: $x10, $x11
//     %1:gpr = COPY $x11   ---- will be delete in this pass
//     %0:gpr = COPY $x10
//     %2:gpr = COPY %1:gpr ---- without this pass, sink to bb1 in machine-sink,
//                               then delete at regalloc
//     BEQ %0:gpr, killed %3:gpr, %bb.3 PseudoBR %bb1
//
//   bb1:
//   bb2:
//     BNE %2:gpr, killed %5:gpr, %bb.2
// ...
// After regalloc
//  bb0:
//    liveins: $x10, $x11
//    renamable $x8 = COPY $x11
//    renamable $x11 = ADDI $x0, 57 --- def x11, so COPY can not be sink
//    BEQ killed renamable $x10, killed renamable $x11, %bb.4
//    PseudoBR %bb.1
//
//  bb1:
//  bb2:
//    BEQ killed renamable $x8, killed renamable $x10, %bb.4
//
// ----->
//
// After this pass:
//   bb0:
//   liveins: $x10, $x11
//     %0:gpr = COPY $x10
//     %2:gpr = COPY $x11
//     BEQ %0:gpr, killed %3:gpr, %bb.3
//     PseudoBR %bb1
//
//   bb1:
//   bb2:
//     BNE %2:gpr, killed %5:gpr, %bb.2
// ...
// After regalloc
//  bb0:
//    liveins: $x10, $x11
//    renamable $x12 = ADDI $x0, 57
//    renamable $x8 = COPY $x11
//    BEQ killed renamable $x10, killed renamable $x11, %bb.4
//    PseudoBR %bb.1
//
//  bb1:
//  bb2:
//    BEQ killed renamable $x8, killed renamable $x10, %bb.4
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;
#define DEBUG_TYPE "riscv-copy-combine"
#define RISCV_COPY_COMBINE "RISC-V Copy Combine"

STATISTIC(NumCopyDeleted, "Number of copy deleted");

namespace {
class RISCVCopyCombine : public MachineFunctionPass {
public:
  static char ID;
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;

  RISCVCopyCombine() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  StringRef getPassName() const override { return RISCV_COPY_COMBINE; }

private:
  bool optimizeBlock(MachineBasicBlock &MBB);
  bool copyCombine(MachineOperand &Op);
};
} // end anonymous namespace

char RISCVCopyCombine::ID = 0;
INITIALIZE_PASS(RISCVCopyCombine, DEBUG_TYPE, RISCV_COPY_COMBINE, false, false)

/// Check if it's safe to move From down to To, checking that no physical
/// registers are clobbered.
static bool isSafeToMove(const MachineInstr &From, const MachineInstr &To) {
  SmallVector<Register> PhysUses;
  for (const MachineOperand &MO : From.all_uses())
    if (MO.getReg().isPhysical())
      PhysUses.push_back(MO.getReg());
  bool SawStore = false;
  for (auto II = From.getIterator(); II != To.getIterator(); II++) {
    for (Register PhysReg : PhysUses)
      if (II->definesRegister(PhysReg, nullptr))
        return false;
    if (II->mayStore()) {
      SawStore = true;
      break;
    }
  }
  return From.isSafeToMove(SawStore);
}

bool RISCVCopyCombine::copyCombine(MachineOperand &Op) {
  if (!Op.isReg())
    return false;

  Register Reg = Op.getReg();
  if (!Reg.isVirtual())
    return false;

  MachineInstr *MI = MRI->getVRegDef(Reg);
  if (MI->getOpcode() != RISCV::COPY)
    return false;

  Register Op1reg = MI->getOperand(1).getReg();
  if (!MRI->hasOneUse(Op1reg) || !Op1reg.isVirtual() ||
      !MI->getOperand(0).getReg().isVirtual())
    return false;

  MachineInstr *Src = MRI->getVRegDef(Op1reg);
  if (!Src || Src->hasUnmodeledSideEffects() ||
      Src->getOpcode() != RISCV::COPY || Src->getParent() != MI->getParent() ||
      Src->getNumDefs() != 1)
    return false;

  if (!isSafeToMove(*Src, *MI))
    return false;

  Register SrcOp1reg = Src->getOperand(1).getReg();
  MRI->replaceRegWith(Op1reg, SrcOp1reg);
  MRI->clearKillFlags(SrcOp1reg);
  LLVM_DEBUG(dbgs() << "Deleting this copy instruction "; Src->print(dbgs()));
  ++NumCopyDeleted;
  Src->eraseFromParent();
  return true;
}

bool RISCVCopyCombine::optimizeBlock(MachineBasicBlock &MBB) {
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 3> Cond;
  if (TII->analyzeBranch(MBB, TBB, FBB, Cond, /*AllowModify*/ false) ||
      Cond.empty())
    return false;

  if (!TBB || Cond.size() != 3)
    return false;

  return copyCombine(Cond[1]) || copyCombine(Cond[2]);
}

bool RISCVCopyCombine::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  TII = MF.getSubtarget().getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = MRI->getTargetRegisterInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= optimizeBlock(MBB);

  return Changed;
}

FunctionPass *llvm::createRISCVCopyCombinePass() {
  return new RISCVCopyCombine();
}
