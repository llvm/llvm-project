//===--------- PPCPreEmitPeephole.cpp - Late peephole optimizations -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A pre-emit peephole for catching opportunities introduced by late passes such
// as MachineBlockPlacement.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCInstrInfo.h"
#include "PPCSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "ppc-pre-emit-peephole"

STATISTIC(NumRRConvertedInPreEmit,
          "Number of r+r instructions converted to r+i in pre-emit peephole");
STATISTIC(NumRemovedInPreEmit,
          "Number of instructions deleted in pre-emit peephole");

static cl::opt<bool>
RunPreEmitPeephole("ppc-late-peephole", cl::Hidden, cl::init(true),
                   cl::desc("Run pre-emit peephole optimizations."));

namespace {
  class PPCPreEmitPeephole : public MachineFunctionPass {
  public:
    static char ID;
    PPCPreEmitPeephole() : MachineFunctionPass(ID) {
      initializePPCPreEmitPeepholePass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoVRegs);
    }

    bool runOnMachineFunction(MachineFunction &MF) override {
      if (skipFunction(MF.getFunction()) || !RunPreEmitPeephole)
        return false;
      bool Changed = false;
      const PPCInstrInfo *TII = MF.getSubtarget<PPCSubtarget>().getInstrInfo();
      SmallVector<MachineInstr *, 4> InstrsToErase;
      for (MachineBasicBlock &MBB : MF) {
        for (MachineInstr &MI : MBB) {
          MachineInstr *DefMIToErase = nullptr;
          if (TII->convertToImmediateForm(MI, &DefMIToErase)) {
            Changed = true;
            NumRRConvertedInPreEmit++;
            DEBUG(dbgs() << "Converted instruction to imm form: ");
            DEBUG(MI.dump());
            if (DefMIToErase) {
              InstrsToErase.push_back(DefMIToErase);
            }
          }
        }
      }
      for (MachineInstr *MI : InstrsToErase) {
        DEBUG(dbgs() << "PPC pre-emit peephole: erasing instruction: ");
        DEBUG(MI->dump());
        MI->eraseFromParent();
        NumRemovedInPreEmit++;
      }
      return Changed;
    }
  };
}

INITIALIZE_PASS(PPCPreEmitPeephole, DEBUG_TYPE, "PowerPC Pre-Emit Peephole",
                false, false)
char PPCPreEmitPeephole::ID = 0;

FunctionPass *llvm::createPPCPreEmitPeepholePass() {
  return new PPCPreEmitPeephole();
}
