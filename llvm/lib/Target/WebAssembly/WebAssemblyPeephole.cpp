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
#include "WebAssemblySubtarget.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-peephole"

namespace {
class WebAssemblyPeephole final : public MachineFunctionPass {
  const char *getPassName() const override {
    return "WebAssembly late peephole optimizer";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
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

/// If desirable, rewrite NewReg to a discard register.
static bool MaybeRewriteToDiscard(unsigned OldReg, unsigned NewReg,
                                  MachineOperand &MO,
                                  WebAssemblyFunctionInfo &MFI,
                                  MachineRegisterInfo &MRI) {
  bool Changed = false;
  // TODO: Handle SP/physregs
  if (OldReg == NewReg && TargetRegisterInfo::isVirtualRegister(NewReg)) {
    Changed = true;
    unsigned NewReg = MRI.createVirtualRegister(MRI.getRegClass(OldReg));
    MO.setReg(NewReg);
    MO.setIsDead();
    MFI.stackifyVReg(NewReg);
    MFI.addWAReg(NewReg, WebAssemblyFunctionInfo::UnusedReg);
  }
  return Changed;
}

bool WebAssemblyPeephole::runOnMachineFunction(MachineFunction &MF) {
  DEBUG({
    dbgs() << "********** Store Results **********\n"
           << "********** Function: " << MF.getName() << '\n';
  });

  MachineRegisterInfo &MRI = MF.getRegInfo();
  WebAssemblyFunctionInfo &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  const WebAssemblyTargetLowering &TLI =
      *MF.getSubtarget<WebAssemblySubtarget>().getTargetLowering();
  auto &LibInfo = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  bool Changed = false;

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
        unsigned NewReg =
            MI.getOperand(WebAssembly::StoreValueOperandNo).getReg();
        Changed |= MaybeRewriteToDiscard(OldReg, NewReg, MO, MFI, MRI);
        break;
      }
      case WebAssembly::CALL_I32:
      case WebAssembly::CALL_I64: {
        MachineOperand &Op1 = MI.getOperand(1);
        if (Op1.isSymbol()) {
          StringRef Name(Op1.getSymbolName());
          if (Name == TLI.getLibcallName(RTLIB::MEMCPY) ||
              Name == TLI.getLibcallName(RTLIB::MEMMOVE) ||
              Name == TLI.getLibcallName(RTLIB::MEMSET)) {
            LibFunc::Func Func;
            if (LibInfo.getLibFunc(Name, Func)) {
              const auto &Op2 = MI.getOperand(2);
              if (!Op2.isReg())
                report_fatal_error("Peephole: call to builtin function with "
                                   "wrong signature, not consuming reg");
              MachineOperand &MO = MI.getOperand(0);
              unsigned OldReg = MO.getReg();
              unsigned NewReg = Op2.getReg();

              // TODO: Handle SP/physregs in MaybeRewriteToDiscard
              if (TargetRegisterInfo::isVirtualRegister(NewReg) &&
                  (MRI.getRegClass(NewReg) != MRI.getRegClass(OldReg)))
                report_fatal_error("Peephole: call to builtin function with "
                                   "wrong signature, from/to mismatch");
              Changed |= MaybeRewriteToDiscard(OldReg, NewReg, MO, MFI, MRI);
            }
          }
        }
      }
      }

  return Changed;
}
