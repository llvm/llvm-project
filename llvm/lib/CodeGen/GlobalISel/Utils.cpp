//===- llvm/CodeGen/GlobalISel/Utils.cpp -------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file This file implements the utility functions used by the GlobalISel
/// pipeline.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

#define DEBUG_TYPE "globalisel-utils"

using namespace llvm;

unsigned llvm::constrainOperandRegClass(
    const MachineFunction &MF, const TargetRegisterInfo &TRI,
    MachineRegisterInfo &MRI, const TargetInstrInfo &TII,
    const RegisterBankInfo &RBI, MachineInstr &InsertPt, const MCInstrDesc &II,
    unsigned Reg, unsigned OpIdx) {
  // Assume physical registers are properly constrained.
  assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
         "PhysReg not implemented");

  const TargetRegisterClass *RegClass = TII.getRegClass(II, OpIdx, &TRI, MF);

  if (!RBI.constrainGenericRegister(Reg, *RegClass, MRI)) {
    unsigned NewReg = MRI.createVirtualRegister(RegClass);
    BuildMI(*InsertPt.getParent(), InsertPt, InsertPt.getDebugLoc(),
            TII.get(TargetOpcode::COPY), NewReg)
        .addReg(Reg);
    return NewReg;
  }

  return Reg;
}

bool llvm::isTriviallyDead(const MachineInstr &MI,
                           const MachineRegisterInfo &MRI) {
  // If we can move an instruction, we can remove it.  Otherwise, it has
  // a side-effect of some sort.
  bool SawStore = false;
  if (!MI.isSafeToMove(/*AA=*/nullptr, SawStore))
    return false;

  // Instructions without side-effects are dead iff they only define dead vregs.
  for (auto &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef())
      continue;

    unsigned Reg = MO.getReg();
    if (TargetRegisterInfo::isPhysicalRegister(Reg) ||
        !MRI.use_nodbg_empty(Reg))
      return false;
  }
  return true;
}

void llvm::reportGISelFailure(MachineFunction &MF, const TargetPassConfig &TPC,
                              MachineOptimizationRemarkEmitter &MORE,
                              MachineOptimizationRemarkMissed &R) {
  MF.getProperties().set(MachineFunctionProperties::Property::FailedISel);

  // Print the function name explicitly if we don't have a debug location (which
  // makes the diagnostic less useful) or if we're going to emit a raw error.
  if (!R.getLocation().isValid() || TPC.isGlobalISelAbortEnabled())
    R << (" (in function: " + MF.getName() + ")").str();

  if (TPC.isGlobalISelAbortEnabled())
    report_fatal_error(R.getMsg());
  else
    MORE.emit(R);
}

void llvm::reportGISelFailure(MachineFunction &MF, const TargetPassConfig &TPC,
                              MachineOptimizationRemarkEmitter &MORE,
                              const char *PassName, StringRef Msg,
                              const MachineInstr &MI) {
  MachineOptimizationRemarkMissed R(PassName, "GISelFailure: ",
                                    MI.getDebugLoc(), MI.getParent());
  R << Msg << ": " << ore::MNV("Inst", MI);
  reportGISelFailure(MF, TPC, MORE, R);
}

Optional<int64_t> llvm::getConstantVRegVal(unsigned VReg,
                                           const MachineRegisterInfo &MRI) {
  MachineInstr *MI = MRI.getVRegDef(VReg);
  if (MI->getOpcode() != TargetOpcode::G_CONSTANT)
    return None;

  if (MI->getOperand(1).isImm())
    return MI->getOperand(1).getImm();

  if (MI->getOperand(1).isCImm() &&
      MI->getOperand(1).getCImm()->getBitWidth() <= 64)
    return MI->getOperand(1).getCImm()->getSExtValue();

  return None;
}

const llvm::ConstantFP* llvm::getConstantFPVRegVal(unsigned VReg,
                                       const MachineRegisterInfo &MRI) {
  MachineInstr *MI = MRI.getVRegDef(VReg);
  if (TargetOpcode::G_FCONSTANT != MI->getOpcode())
    return nullptr;
  return MI->getOperand(1).getFPImm();
}
