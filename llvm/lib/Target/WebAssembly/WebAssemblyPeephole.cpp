//===-- WebAssemblyPeephole.cpp - WebAssembly Peephole Optimiztions -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Late peephole optimizations for WebAssembly.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-peephole"

namespace {
class WebAssemblyPeephole final : public MachineFunctionPass {
  const char *getPassName() const override {
    return "WebAssembly late peephole optimizer";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID;
  WebAssemblyPeephole() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyPeephole::ID = 0;
FunctionPass *llvm::createWebAssemblyPeephole() {
  return new WebAssemblyPeephole();
}

bool WebAssemblyPeephole::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  WebAssemblyFunctionInfo &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();

  for (auto &MBB : MF)
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
      case WebAssembly::STORE_I64: {
        // Store instructions return their value operand. If we ended up using
        // the same register for both, replace it with a dead def so that it
        // can use $discard instead.
        MachineOperand &MO = MI.getOperand(0);
        unsigned OldReg = MO.getReg();
        // TODO: Handle SP/physregs
        if (OldReg == MI.getOperand(3).getReg() &&
            TargetRegisterInfo::isVirtualRegister(MI.getOperand(3).getReg())) {
          Changed = true;
          unsigned NewReg = MRI.createVirtualRegister(MRI.getRegClass(OldReg));
          MO.setReg(NewReg);
          MO.setIsDead();
          MFI.stackifyVReg(NewReg);
          MFI.addWAReg(NewReg, WebAssemblyFunctionInfo::UnusedReg);
        }
      }
      }

  return Changed;
}
