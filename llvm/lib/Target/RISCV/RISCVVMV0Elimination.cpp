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

  auto IsVMV0 = [](const MCOperandInfo &MCOI) {
    return MCOI.RegClass == RISCV::VMV0RegClassID;
  };

#ifndef NDEBUG
  // Assert that we won't clobber any existing reads of V0 where we need to
  // insert copies.
  ReversePostOrderTraversal<MachineBasicBlock *> RPOT(&*MF.begin());
  SmallPtrSet<MachineBasicBlock *, 8> V0ClobberedOnEntry;
  for (MachineBasicBlock *MBB : RPOT) {
    bool V0Clobbered = V0ClobberedOnEntry.contains(MBB);
    for (MachineInstr &MI : *MBB) {
      assert(!(MI.readsRegister(RISCV::V0, TRI) && V0Clobbered));
      if (MI.modifiesRegister(RISCV::V0, TRI))
        V0Clobbered = false;

      if (any_of(MI.getDesc().operands(), IsVMV0))
        V0Clobbered = true;
    }

    if (V0Clobbered)
      for (MachineBasicBlock *Succ : MBB->successors())
        V0ClobberedOnEntry.insert(Succ);
  }
#endif

  bool MadeChange = false;

  // For any instruction with a vmv0 operand, replace it with a copy to v0.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // An instruction should only have one or zero vmv0 operands.
      assert(count_if(MI.getDesc().operands(), IsVMV0) < 2);

      for (auto [OpNo, MCOI] : enumerate(MI.getDesc().operands())) {
        if (IsVMV0(MCOI)) {
          MachineOperand &MO = MI.getOperand(OpNo);
          BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(RISCV::COPY), RISCV::V0)
              .addReg(MO.getReg());
          MO.setReg(RISCV::V0);
          MadeChange = true;
          break;
        }
      }
    }
  }

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
                 MRI.getVRegDef(MO.getReg())->isInlineAsm());
          MadeChange = true;
        }
      }
    }
  }

  return MadeChange;
}
