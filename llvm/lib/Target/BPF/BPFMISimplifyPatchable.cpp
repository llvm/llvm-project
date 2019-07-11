//===----- BPFMISimplifyPatchable.cpp - MI Simplify Patchable Insts -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass targets a subset of instructions like below
//    ld_imm64 r1, @global
//    ldd r2, r1, 0
//    add r3, struct_base_reg, r2
//
// Here @global should either present a AMA (abstruct member access) or
// a patchable extern variable. And these two kinds of accesses
// are subject to bpf load time patching. After this pass, the
// code becomes
//    ld_imm64 r1, @global
//    add r3, struct_base_reg, r1
//
// Eventually, at BTF output stage, a relocation record will be generated
// for ld_imm64 which should be replaced later by bpf loader:
//    r1 = <calculated offset> or <to_be_patched_extern_val>
//    add r3, struct_base_reg, r1
// or
//    ld_imm64 r1, <to_be_patched_extern_val>
//    add r3, struct_base_reg, r1
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFCORE.h"
#include "BPFInstrInfo.h"
#include "BPFTargetMachine.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-mi-simplify-patchable"

namespace {

struct BPFMISimplifyPatchable : public MachineFunctionPass {

  static char ID;
  const BPFInstrInfo *TII;
  MachineFunction *MF;

  BPFMISimplifyPatchable() : MachineFunctionPass(ID) {
    initializeBPFMISimplifyPatchablePass(*PassRegistry::getPassRegistry());
  }

private:
  // Initialize class variables.
  void initialize(MachineFunction &MFParm);

  bool removeLD(void);

public:
  // Main entry point for this pass.
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!skipFunction(MF.getFunction())) {
      initialize(MF);
    }
    return removeLD();
  }
};

// Initialize class variables.
void BPFMISimplifyPatchable::initialize(MachineFunction &MFParm) {
  MF = &MFParm;
  TII = MF->getSubtarget<BPFSubtarget>().getInstrInfo();
  LLVM_DEBUG(dbgs() << "*** BPF simplify patchable insts pass ***\n\n");
}

/// Remove unneeded Load instructions.
bool BPFMISimplifyPatchable::removeLD() {
  MachineRegisterInfo *MRI = &MF->getRegInfo();
  MachineInstr *ToErase = nullptr;
  bool Changed = false;

  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      if (ToErase) {
        ToErase->eraseFromParent();
        ToErase = nullptr;
      }

      // Ensure the register format is LOAD <reg>, <reg>, 0
      if (MI.getOpcode() != BPF::LDD && MI.getOpcode() != BPF::LDW &&
          MI.getOpcode() != BPF::LDH && MI.getOpcode() != BPF::LDB &&
          MI.getOpcode() != BPF::LDW32 && MI.getOpcode() != BPF::LDH32 &&
          MI.getOpcode() != BPF::LDB32)
        continue;

      if (!MI.getOperand(0).isReg() || !MI.getOperand(1).isReg())
        continue;

      if (!MI.getOperand(2).isImm() || MI.getOperand(2).getImm())
        continue;

      unsigned DstReg = MI.getOperand(0).getReg();
      unsigned SrcReg = MI.getOperand(1).getReg();
      int64_t ImmVal = MI.getOperand(2).getImm();

      MachineInstr *DefInst = MRI->getUniqueVRegDef(SrcReg);
      if (!DefInst)
        continue;

      bool IsCandidate = false;
      if (DefInst->getOpcode() == BPF::LD_imm64) {
        const MachineOperand &MO = DefInst->getOperand(1);
        if (MO.isGlobal()) {
          const GlobalValue *GVal = MO.getGlobal();
          auto *GVar = dyn_cast<GlobalVariable>(GVal);
          if (GVar) {
            // Global variables representing structure offset or
            // patchable extern globals.
            if (GVar->hasAttribute(BPFCoreSharedInfo::AmaAttr)) {
              assert(ImmVal == 0);
              IsCandidate = true;
            } else if (!GVar->hasInitializer() && GVar->hasExternalLinkage() &&
                       GVar->getSection() ==
                           BPFCoreSharedInfo::PatchableExtSecName) {
              if (ImmVal == 0)
                IsCandidate = true;
              else
                errs() << "WARNING: unhandled patchable extern "
                       << GVar->getName() << " with load offset " << ImmVal
                       << "\n";
            }
          }
        }
      }

      if (!IsCandidate)
        continue;

      auto Begin = MRI->use_begin(DstReg), End = MRI->use_end();
      decltype(End) NextI;
      for (auto I = Begin; I != End; I = NextI) {
        NextI = std::next(I);
        I->setReg(SrcReg);
      }

      ToErase = &MI;
      Changed = true;
    }
  }

  return Changed;
}

} // namespace

INITIALIZE_PASS(BPFMISimplifyPatchable, DEBUG_TYPE,
                "BPF PreEmit SimplifyPatchable", false, false)

char BPFMISimplifyPatchable::ID = 0;
FunctionPass *llvm::createBPFMISimplifyPatchablePass() {
  return new BPFMISimplifyPatchable();
}
