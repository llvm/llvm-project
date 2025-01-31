//===- RISCVVMV0Elimination.cpp - VMV0 Elimination -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// Mask operands in vector pseudos have to be in v0. We select them as a virtual
// register in the singleton vmv0 register class instead of copying them to $v0
// straight away, to make optimizing masks easier.
//
// However the register allocator struggles with singleton register classes and
// will run into errors like "ran out of registers during register allocation in
// function"
//
// This pass runs just before register allocation and replaces any uses* of vmv0
// with copies to $v0.
//
// %x:vrnov0 = PseudoVADD_VV_M1_MASK %0:vrnov0, %1:vr, %2:vr, %3:vmv0, ...
// ->
// $v0 = COPY %3:vr
// %x:vrnov0 = PseudoVADD_VV_M1_MASK %0:vrnov0, %1:vr, %2:vr, $0, ...
//
// * The only uses of vmv0 left behind are when used for inline asm with the vm
// constraint.
//
//===---------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#ifndef NDEBUG
#include "llvm/ADT/PostOrderIterator.h"
#endif
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-vmv0-elimination"

namespace {

class RISCVVMV0Elimination : public MachineFunctionPass {
public:
  static char ID;
  RISCVVMV0Elimination() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    // TODO: We could move this closer to regalloc, out of SSA, which would
    // allow scheduling past mask operands. We would need to preserve live
    // intervals.
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }
};

} // namespace

char RISCVVMV0Elimination::ID = 0;

INITIALIZE_PASS(RISCVVMV0Elimination, DEBUG_TYPE, "RISC-V VMV0 Elimination",
                false, false)

FunctionPass *llvm::createRISCVVMV0EliminationPass() {
  return new RISCVVMV0Elimination();
}

static bool isVMV0(const MCOperandInfo &MCOI) {
  return MCOI.RegClass == RISCV::VMV0RegClassID;
}

bool RISCVVMV0Elimination::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  // Skip if the vector extension is not enabled.
  const RISCVSubtarget *ST = &MF.getSubtarget<RISCVSubtarget>();
  if (!ST->hasVInstructions())
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo *TRI = MRI.getTargetRegisterInfo();
  const TargetInstrInfo *TII = ST->getInstrInfo();

#ifndef NDEBUG
  // Assert that we won't clobber any existing reads of v0 where we need to
  // insert copies.
  ReversePostOrderTraversal<MachineBasicBlock *> RPOT(&*MF.begin());
  for (MachineBasicBlock *MBB : RPOT) {
    bool V0Clobbered = false;
    for (MachineInstr &MI : *MBB) {
      assert(!(MI.readsRegister(RISCV::V0, TRI) && V0Clobbered) &&
             "Inserting a copy to v0 would clobber a read");
      if (MI.modifiesRegister(RISCV::V0, TRI))
        V0Clobbered = false;

      if (any_of(MI.getDesc().operands(), isVMV0))
        V0Clobbered = true;
    }

    assert(!(V0Clobbered &&
             any_of(MBB->successors(),
                    [](auto *Succ) { return Succ->isLiveIn(RISCV::V0); })) &&
           "Clobbered a v0 used in a successor");
  }
#endif

  bool MadeChange = false;

  // For any instruction with a vmv0 operand, replace it with a copy to v0.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      assert(count_if(MI.getDesc().operands(), isVMV0) < 2 &&
             "Expected only one or zero vmv0 operands");

      for (auto [OpNo, MCOI] : enumerate(MI.getDesc().operands())) {
        if (isVMV0(MCOI)) {
          MachineOperand &MO = MI.getOperand(OpNo);
          Register Src = MO.getReg();
          assert(MO.isUse() && MO.getSubReg() == RISCV::NoSubRegister &&
                 Src.isVirtual() && "vmv0 use in unexpected form");

          // Peek through a single copy to match what isel does.
          MachineInstr *SrcMI = MRI.getVRegDef(Src);
          if (SrcMI->isCopy() && SrcMI->getOperand(1).getReg().isVirtual()) {
            assert(SrcMI->getOperand(1).getSubReg() == RISCV::NoSubRegister);
            Src = SrcMI->getOperand(1).getReg();
          }

          BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(RISCV::COPY), RISCV::V0)
              .addReg(Src);

          MO.setReg(RISCV::V0);
          MadeChange = true;
          break;
        }
      }
    }
  }

  if (!MadeChange)
    return false;

  // Now that any constraints requiring vmv0 are gone, eliminate any uses of
  // vmv0 by recomputing the reg class.
  // The only remaining uses should be around inline asm.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : MI.uses()) {
        if (MO.isReg() && MO.getReg().isVirtual() &&
            MRI.getRegClass(MO.getReg()) == &RISCV::VMV0RegClass) {
          MRI.recomputeRegClass(MO.getReg());
          assert(MRI.getRegClass(MO.getReg()) != &RISCV::VMV0RegClass ||
                 MI.isInlineAsm() ||
                 MRI.getVRegDef(MO.getReg())->isInlineAsm() &&
                     "Non-inline-asm use of vmv0 left behind");
          MadeChange = true;
        }
      }
    }
  }

  return MadeChange;
}
