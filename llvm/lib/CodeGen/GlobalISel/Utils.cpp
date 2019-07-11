//===- llvm/CodeGen/GlobalISel/Utils.cpp -------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file This file implements the utility functions used by the GlobalISel
/// pipeline.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Constants.h"

#define DEBUG_TYPE "globalisel-utils"

using namespace llvm;

unsigned llvm::constrainRegToClass(MachineRegisterInfo &MRI,
                                   const TargetInstrInfo &TII,
                                   const RegisterBankInfo &RBI, unsigned Reg,
                                   const TargetRegisterClass &RegClass) {
  if (!RBI.constrainGenericRegister(Reg, RegClass, MRI))
    return MRI.createVirtualRegister(&RegClass);

  return Reg;
}

unsigned llvm::constrainOperandRegClass(
    const MachineFunction &MF, const TargetRegisterInfo &TRI,
    MachineRegisterInfo &MRI, const TargetInstrInfo &TII,
    const RegisterBankInfo &RBI, MachineInstr &InsertPt,
    const TargetRegisterClass &RegClass, const MachineOperand &RegMO,
    unsigned OpIdx) {
  unsigned Reg = RegMO.getReg();
  // Assume physical registers are properly constrained.
  assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
         "PhysReg not implemented");

  unsigned ConstrainedReg = constrainRegToClass(MRI, TII, RBI, Reg, RegClass);
  // If we created a new virtual register because the class is not compatible
  // then create a copy between the new and the old register.
  if (ConstrainedReg != Reg) {
    MachineBasicBlock::iterator InsertIt(&InsertPt);
    MachineBasicBlock &MBB = *InsertPt.getParent();
    if (RegMO.isUse()) {
      BuildMI(MBB, InsertIt, InsertPt.getDebugLoc(),
              TII.get(TargetOpcode::COPY), ConstrainedReg)
          .addReg(Reg);
    } else {
      assert(RegMO.isDef() && "Must be a definition");
      BuildMI(MBB, std::next(InsertIt), InsertPt.getDebugLoc(),
              TII.get(TargetOpcode::COPY), Reg)
          .addReg(ConstrainedReg);
    }
  }
  return ConstrainedReg;
}

unsigned llvm::constrainOperandRegClass(
    const MachineFunction &MF, const TargetRegisterInfo &TRI,
    MachineRegisterInfo &MRI, const TargetInstrInfo &TII,
    const RegisterBankInfo &RBI, MachineInstr &InsertPt, const MCInstrDesc &II,
    const MachineOperand &RegMO, unsigned OpIdx) {
  unsigned Reg = RegMO.getReg();
  // Assume physical registers are properly constrained.
  assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
         "PhysReg not implemented");

  const TargetRegisterClass *RegClass = TII.getRegClass(II, OpIdx, &TRI, MF);
  // Some of the target independent instructions, like COPY, may not impose any
  // register class constraints on some of their operands: If it's a use, we can
  // skip constraining as the instruction defining the register would constrain
  // it.

  // We can't constrain unallocatable register classes, because we can't create
  // virtual registers for these classes, so we need to let targets handled this
  // case.
  if (RegClass && !RegClass->isAllocatable())
    RegClass = TRI.getConstrainedRegClassForOperand(RegMO, MRI);

  if (!RegClass) {
    assert((!isTargetSpecificOpcode(II.getOpcode()) || RegMO.isUse()) &&
           "Register class constraint is required unless either the "
           "instruction is target independent or the operand is a use");
    // FIXME: Just bailing out like this here could be not enough, unless we
    // expect the users of this function to do the right thing for PHIs and
    // COPY:
    //   v1 = COPY v0
    //   v2 = COPY v1
    // v1 here may end up not being constrained at all. Please notice that to
    // reproduce the issue we likely need a destination pattern of a selection
    // rule producing such extra copies, not just an input GMIR with them as
    // every existing target using selectImpl handles copies before calling it
    // and they never reach this function.
    return Reg;
  }
  return constrainOperandRegClass(MF, TRI, MRI, TII, RBI, InsertPt, *RegClass,
                                  RegMO, OpIdx);
}

bool llvm::constrainSelectedInstRegOperands(MachineInstr &I,
                                            const TargetInstrInfo &TII,
                                            const TargetRegisterInfo &TRI,
                                            const RegisterBankInfo &RBI) {
  assert(!isPreISelGenericOpcode(I.getOpcode()) &&
         "A selected instruction is expected");
  MachineBasicBlock &MBB = *I.getParent();
  MachineFunction &MF = *MBB.getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  for (unsigned OpI = 0, OpE = I.getNumExplicitOperands(); OpI != OpE; ++OpI) {
    MachineOperand &MO = I.getOperand(OpI);

    // There's nothing to be done on non-register operands.
    if (!MO.isReg())
      continue;

    LLVM_DEBUG(dbgs() << "Converting operand: " << MO << '\n');
    assert(MO.isReg() && "Unsupported non-reg operand");

    unsigned Reg = MO.getReg();
    // Physical registers don't need to be constrained.
    if (TRI.isPhysicalRegister(Reg))
      continue;

    // Register operands with a value of 0 (e.g. predicate operands) don't need
    // to be constrained.
    if (Reg == 0)
      continue;

    // If the operand is a vreg, we should constrain its regclass, and only
    // insert COPYs if that's impossible.
    // constrainOperandRegClass does that for us.
    MO.setReg(constrainOperandRegClass(MF, TRI, MRI, TII, RBI, I, I.getDesc(),
                                       MO, OpI));

    // Tie uses to defs as indicated in MCInstrDesc if this hasn't already been
    // done.
    if (MO.isUse()) {
      int DefIdx = I.getDesc().getOperandConstraint(OpI, MCOI::TIED_TO);
      if (DefIdx != -1 && !I.isRegTiedToUseOperand(DefIdx))
        I.tieOperands(DefIdx, OpI);
    }
  }
  return true;
}

bool llvm::isTriviallyDead(const MachineInstr &MI,
                           const MachineRegisterInfo &MRI) {
  // If we can move an instruction, we can remove it.  Otherwise, it has
  // a side-effect of some sort.
  bool SawStore = false;
  if (!MI.isSafeToMove(/*AA=*/nullptr, SawStore) && !MI.isPHI())
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
  R << Msg;
  // Printing MI is expensive;  only do it if expensive remarks are enabled.
  if (TPC.isGlobalISelAbortEnabled() || MORE.allowExtraAnalysis(PassName))
    R << ": " << ore::MNV("Inst", MI);
  reportGISelFailure(MF, TPC, MORE, R);
}

Optional<int64_t> llvm::getConstantVRegVal(unsigned VReg,
                                           const MachineRegisterInfo &MRI) {
  Optional<ValueAndVReg> ValAndVReg =
      getConstantVRegValWithLookThrough(VReg, MRI, /*LookThroughInstrs*/ false);
  assert((!ValAndVReg || ValAndVReg->VReg == VReg) &&
         "Value found while looking through instrs");
  if (!ValAndVReg)
    return None;
  return ValAndVReg->Value;
}

Optional<ValueAndVReg> llvm::getConstantVRegValWithLookThrough(
    unsigned VReg, const MachineRegisterInfo &MRI, bool LookThroughInstrs) {
  SmallVector<std::pair<unsigned, unsigned>, 4> SeenOpcodes;
  MachineInstr *MI;
  while ((MI = MRI.getVRegDef(VReg)) &&
         MI->getOpcode() != TargetOpcode::G_CONSTANT && LookThroughInstrs) {
    switch (MI->getOpcode()) {
    case TargetOpcode::G_TRUNC:
    case TargetOpcode::G_SEXT:
    case TargetOpcode::G_ZEXT:
      SeenOpcodes.push_back(std::make_pair(
          MI->getOpcode(),
          MRI.getType(MI->getOperand(0).getReg()).getSizeInBits()));
      VReg = MI->getOperand(1).getReg();
      break;
    case TargetOpcode::COPY:
      VReg = MI->getOperand(1).getReg();
      if (TargetRegisterInfo::isPhysicalRegister(VReg))
        return None;
      break;
    case TargetOpcode::G_INTTOPTR:
      VReg = MI->getOperand(1).getReg();
      break;
    default:
      return None;
    }
  }
  if (!MI || MI->getOpcode() != TargetOpcode::G_CONSTANT ||
      (!MI->getOperand(1).isImm() && !MI->getOperand(1).isCImm()))
    return None;

  const MachineOperand &CstVal = MI->getOperand(1);
  unsigned BitWidth = MRI.getType(MI->getOperand(0).getReg()).getSizeInBits();
  APInt Val = CstVal.isImm() ? APInt(BitWidth, CstVal.getImm())
                             : CstVal.getCImm()->getValue();
  assert(Val.getBitWidth() == BitWidth &&
         "Value bitwidth doesn't match definition type");
  while (!SeenOpcodes.empty()) {
    std::pair<unsigned, unsigned> OpcodeAndSize = SeenOpcodes.pop_back_val();
    switch (OpcodeAndSize.first) {
    case TargetOpcode::G_TRUNC:
      Val = Val.trunc(OpcodeAndSize.second);
      break;
    case TargetOpcode::G_SEXT:
      Val = Val.sext(OpcodeAndSize.second);
      break;
    case TargetOpcode::G_ZEXT:
      Val = Val.zext(OpcodeAndSize.second);
      break;
    }
  }

  if (Val.getBitWidth() > 64)
    return None;

  return ValueAndVReg{Val.getSExtValue(), VReg};
}

const llvm::ConstantFP* llvm::getConstantFPVRegVal(unsigned VReg,
                                       const MachineRegisterInfo &MRI) {
  MachineInstr *MI = MRI.getVRegDef(VReg);
  if (TargetOpcode::G_FCONSTANT != MI->getOpcode())
    return nullptr;
  return MI->getOperand(1).getFPImm();
}

llvm::MachineInstr *llvm::getDefIgnoringCopies(Register Reg,
                                               const MachineRegisterInfo &MRI) {
  auto *DefMI = MRI.getVRegDef(Reg);
  auto DstTy = MRI.getType(DefMI->getOperand(0).getReg());
  if (!DstTy.isValid())
    return nullptr;
  while (DefMI->getOpcode() == TargetOpcode::COPY) {
    unsigned SrcReg = DefMI->getOperand(1).getReg();
    auto SrcTy = MRI.getType(SrcReg);
    if (!SrcTy.isValid() || SrcTy != DstTy)
      break;
    DefMI = MRI.getVRegDef(SrcReg);
  }
  return DefMI;
}

llvm::MachineInstr *llvm::getOpcodeDef(unsigned Opcode, Register Reg,
                                       const MachineRegisterInfo &MRI) {
  MachineInstr *DefMI = getDefIgnoringCopies(Reg, MRI);
  return DefMI && DefMI->getOpcode() == Opcode ? DefMI : nullptr;
}

APFloat llvm::getAPFloatFromSize(double Val, unsigned Size) {
  if (Size == 32)
    return APFloat(float(Val));
  if (Size == 64)
    return APFloat(Val);
  if (Size != 16)
    llvm_unreachable("Unsupported FPConstant size");
  bool Ignored;
  APFloat APF(Val);
  APF.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &Ignored);
  return APF;
}

Optional<APInt> llvm::ConstantFoldBinOp(unsigned Opcode, const unsigned Op1,
                                        const unsigned Op2,
                                        const MachineRegisterInfo &MRI) {
  auto MaybeOp1Cst = getConstantVRegVal(Op1, MRI);
  auto MaybeOp2Cst = getConstantVRegVal(Op2, MRI);
  if (MaybeOp1Cst && MaybeOp2Cst) {
    LLT Ty = MRI.getType(Op1);
    APInt C1(Ty.getSizeInBits(), *MaybeOp1Cst, true);
    APInt C2(Ty.getSizeInBits(), *MaybeOp2Cst, true);
    switch (Opcode) {
    default:
      break;
    case TargetOpcode::G_ADD:
      return C1 + C2;
    case TargetOpcode::G_AND:
      return C1 & C2;
    case TargetOpcode::G_ASHR:
      return C1.ashr(C2);
    case TargetOpcode::G_LSHR:
      return C1.lshr(C2);
    case TargetOpcode::G_MUL:
      return C1 * C2;
    case TargetOpcode::G_OR:
      return C1 | C2;
    case TargetOpcode::G_SHL:
      return C1 << C2;
    case TargetOpcode::G_SUB:
      return C1 - C2;
    case TargetOpcode::G_XOR:
      return C1 ^ C2;
    case TargetOpcode::G_UDIV:
      if (!C2.getBoolValue())
        break;
      return C1.udiv(C2);
    case TargetOpcode::G_SDIV:
      if (!C2.getBoolValue())
        break;
      return C1.sdiv(C2);
    case TargetOpcode::G_UREM:
      if (!C2.getBoolValue())
        break;
      return C1.urem(C2);
    case TargetOpcode::G_SREM:
      if (!C2.getBoolValue())
        break;
      return C1.srem(C2);
    }
  }
  return None;
}

bool llvm::isKnownNeverNaN(Register Val, const MachineRegisterInfo &MRI,
                           bool SNaN) {
  const MachineInstr *DefMI = MRI.getVRegDef(Val);
  if (!DefMI)
    return false;

  if (DefMI->getFlag(MachineInstr::FmNoNans))
    return true;

  if (SNaN) {
    // FP operations quiet. For now, just handle the ones inserted during
    // legalization.
    switch (DefMI->getOpcode()) {
    case TargetOpcode::G_FPEXT:
    case TargetOpcode::G_FPTRUNC:
    case TargetOpcode::G_FCANONICALIZE:
      return true;
    default:
      return false;
    }
  }

  return false;
}

void llvm::getSelectionDAGFallbackAnalysisUsage(AnalysisUsage &AU) {
  AU.addPreserved<StackProtector>();
}
