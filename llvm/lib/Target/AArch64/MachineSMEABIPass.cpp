//===- MachineSMEABIPass.cpp - MIR SME ABI lowerings ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass implements aspects of the SME ABI not known to be required till
// this stage in lowering. Currently this is just:
// * Saving VG for the unwinder (which depends on SMEPeepholeOpt running first).
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-machine-sme-abi"

namespace {

struct MachineSMEABI : public MachineFunctionPass {
  inline static char ID = 0;

  MachineSMEABI() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "Machine SME ABI pass"; }

  bool insertVGSaveForUnwinder(MachineFunction &MF) const;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

bool MachineSMEABI::insertVGSaveForUnwinder(MachineFunction &MF) const {
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const AArch64RegisterInfo &TRI = *Subtarget.getRegisterInfo();
  const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  if (!TRI.requiresSaveVG(MF))
    return false;

  SMEAttrs Attrs = AFI->getSMEFnAttrs();
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();

  DebugLoc DL;
  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();

  if (Attrs.hasStreamingBody() && !Attrs.hasStreamingInterface()) {
    // For locally-streaming functions, we need to store both the streaming
    // & non-streaming VG. Spill the streaming value first.
    Register RDSVLReg = MRI.createVirtualRegister(&AArch64::GPR64RegClass);
    Register StreamingVGReg =
        MRI.createVirtualRegister(&AArch64::GPR64RegClass);
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::RDSVLI_XI), RDSVLReg).addImm(1);
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::UBFMXri), StreamingVGReg)
        .addReg(RDSVLReg)
        .addImm(3)
        .addImm(63);
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::STRXui))
        .addReg(StreamingVGReg)
        .addTargetIndex(AArch64::SAVED_STREAMING_VG_SLOT)
        .addImm(0);
  }

  Register VGReg;
  if (MF.getSubtarget<AArch64Subtarget>().hasSVE()) {
    VGReg = MRI.createVirtualRegister(&AArch64::GPR64RegClass);
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::CNTD_XPiI), VGReg)
        .addImm(31)
        .addImm(1);
  } else {
    VGReg = AArch64::X0;
    const uint32_t *RegMask = TRI.getCallPreservedMask(
        MF, CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X1);
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::BL))
        .addExternalSymbol("__arm_get_current_vg")
        .addRegMask(RegMask)
        .addReg(AArch64::X0, RegState::ImplicitDefine);
  }

  BuildMI(MBB, MBBI, DL, TII.get(AArch64::STRXui))
      .addReg(VGReg)
      .addTargetIndex(AArch64::SAVED_VG_SLOT)
      .addImm(0);
  return true;
}

INITIALIZE_PASS(MachineSMEABI, "aarch64-machine-sme-abi",
                "Machine SME ABI pass", false, false)

bool MachineSMEABI::runOnMachineFunction(MachineFunction &MF) {
  assert(MF.getRegInfo().isSSA() && "Expected to be run on SSA form!");
  return insertVGSaveForUnwinder(MF);
}

FunctionPass *llvm::createMachineSMEABIPass() { return new MachineSMEABI(); }
