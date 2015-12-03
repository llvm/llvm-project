//===-- WebAssemblyStoreResults.cpp - Optimize using store result values --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements an optimization pass using store result values.
///
/// WebAssembly's store instructions return the stored value, specifically to
/// enable the optimization of reducing get_local/set_local traffic, which is
/// what we're doing here.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-store-results"

namespace {
class WebAssemblyStoreResults final : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyStoreResults() : MachineFunctionPass(ID) {}

  const char *getPassName() const override {
    return "WebAssembly Store Results";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineBlockFrequencyInfo>();
    AU.addPreserved<MachineBlockFrequencyInfo>();
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
};
} // end anonymous namespace

char WebAssemblyStoreResults::ID = 0;
FunctionPass *llvm::createWebAssemblyStoreResults() {
  return new WebAssemblyStoreResults();
}

bool WebAssemblyStoreResults::runOnMachineFunction(MachineFunction &MF) {
  DEBUG({
    dbgs() << "********** Store Results **********\n"
           << "********** Function: " << MF.getName() << '\n';
  });

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();

  for (auto &MBB : MF) {
    DEBUG(dbgs() << "Basic Block: " << MBB.getName() << '\n');
    for (auto &MI : MBB)
      switch (MI.getOpcode()) {
      default:
        break;
      case WebAssembly::STORE8_I32:
      case WebAssembly::STORE16_I32:
      case WebAssembly::STORE8_I64:
      case WebAssembly::STORE16_I64:
      case WebAssembly::STORE32_I64:
      case WebAssembly::STORE_F32:
      case WebAssembly::STORE_F64:
      case WebAssembly::STORE_I32:
      case WebAssembly::STORE_I64:
        unsigned ToReg = MI.getOperand(0).getReg();
        unsigned FromReg = MI.getOperand(2).getReg();
        for (auto I = MRI.use_begin(FromReg), E = MRI.use_end(); I != E;) {
          MachineOperand &O = *I++;
          MachineInstr *Where = O.getParent();
          if (Where->getOpcode() == TargetOpcode::PHI)
            Where = Where->getOperand(&O - &Where->getOperand(0) + 1)
                        .getMBB()
                        ->getFirstTerminator();
          if (&MI == Where || !MDT.dominates(&MI, Where))
            continue;
          DEBUG(dbgs() << "Setting operand " << O << " in " << *Where <<
                " from " << MI <<"\n");
          O.setReg(ToReg);
        }
      }
  }

  return true;
}
