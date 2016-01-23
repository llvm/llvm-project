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
#include "llvm/CodeGen/MachineFrameInfo.h"
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
  const MachineFrameInfo &FrameInfo = *MF.getFrameInfo();

  MFI.initWARegs();

  // WebAssembly argument registers are in the same index space as local
  // variables. Assign the numbers for them first.
  MachineBasicBlock &EntryMBB = MF.front();
  for (MachineInstr &MI : EntryMBB) {
    switch (MI.getOpcode()) {
    case WebAssembly::ARGUMENT_I32:
    case WebAssembly::ARGUMENT_I64:
    case WebAssembly::ARGUMENT_F32:
    case WebAssembly::ARGUMENT_F64:
      DEBUG(dbgs() << "Arg VReg " << MI.getOperand(0).getReg() << " -> WAReg "
                   << MI.getOperand(1).getImm() << "\n");
      MFI.setWAReg(MI.getOperand(0).getReg(), MI.getOperand(1).getImm());
      break;
    default:
      break;
    }
  }

  // Then assign regular WebAssembly registers for all remaining used
  // virtual registers. TODO: Consider sorting the registers by frequency of
  // use, to maximize usage of small immediate fields.
  unsigned NumVRegs = MF.getRegInfo().getNumVirtRegs();
  unsigned NumStackRegs = 0;
  // Start the numbering for locals after the arg regs
  unsigned CurReg = MFI.getParams().size();
  for (unsigned VRegIdx = 0; VRegIdx < NumVRegs; ++VRegIdx) {
    unsigned VReg = TargetRegisterInfo::index2VirtReg(VRegIdx);
    // Handle stackified registers.
    if (MFI.isVRegStackified(VReg)) {
      DEBUG(dbgs() << "VReg " << VReg << " -> WAReg "
                   << (INT32_MIN | NumStackRegs) << "\n");
      MFI.setWAReg(VReg, INT32_MIN | NumStackRegs++);
      continue;
    }
    // Skip unused registers.
    if (MRI.use_empty(VReg))
      continue;
    if (MFI.getWAReg(VReg) == WebAssemblyFunctionInfo::UnusedReg) {
      DEBUG(dbgs() << "VReg " << VReg << " -> WAReg " << CurReg << "\n");
      MFI.setWAReg(VReg, CurReg++);
    }
  }
  // Allocate locals for used physical registers
  if (FrameInfo.getStackSize() > 0) {
    DEBUG(dbgs() << "PReg SP " << CurReg << "\n");
    MFI.addPReg(WebAssembly::SP32, CurReg++);
  }

  return true;
}
