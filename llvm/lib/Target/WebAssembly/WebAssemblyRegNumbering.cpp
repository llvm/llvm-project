//===-- WebAssemblyRegNumbering.cpp - Register Numbering ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a pass which assigns WebAssembly register
/// numbers for CodeGen virtual registers.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-reg-numbering"

namespace {
class WebAssemblyRegNumbering final : public MachineFunctionPass {
  const char *getPassName() const override {
    return "WebAssembly Register Numbering";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyRegNumbering() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyRegNumbering::ID = 0;
FunctionPass *llvm::createWebAssemblyRegNumbering() {
  return new WebAssemblyRegNumbering();
}

bool WebAssemblyRegNumbering::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "********** Register Numbering **********\n"
                  "********** Function: "
               << MF.getName() << '\n');

  WebAssemblyFunctionInfo &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  MFI.initWARegs();

  // WebAssembly argument registers are in the same index space as local
  // variables. Assign the numbers for them first.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      switch (MI.getOpcode()) {
      case WebAssembly::ARGUMENT_I32:
      case WebAssembly::ARGUMENT_I64:
      case WebAssembly::ARGUMENT_F32:
      case WebAssembly::ARGUMENT_F64:
        MFI.setWAReg(MI.getOperand(0).getReg(), MI.getOperand(1).getImm());
        break;
      default:
        break;
      }
    }
  }

  // Then assign regular WebAssembly registers for all remaining used
  // virtual registers.
  unsigned NumArgRegs = MFI.getParams().size();
  unsigned NumVRegs = MF.getRegInfo().getNumVirtRegs();
  unsigned CurReg = 0;
  for (unsigned VRegIdx = 0; VRegIdx < NumVRegs; ++VRegIdx) {
    unsigned VReg = TargetRegisterInfo::index2VirtReg(VRegIdx);
    // Skip unused registers.
    if (MRI.use_empty(VReg))
      continue;
    if (MFI.getWAReg(VReg) == WebAssemblyFunctionInfo::UnusedReg)
      MFI.setWAReg(VReg, NumArgRegs + CurReg++);
  }

  return true;
}
