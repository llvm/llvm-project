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
#include "llvm/ADT/APInt.h"
#include "llvm/CodeGen/CodeGenCommonISel.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/LostDebugLocObserver.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSizeOpts.h"
#include "llvm/CodeGen/RegisterBankInfo.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include <numeric>
#include <optional>

#define DEBUG_TYPE "globalisel-utils"

using namespace llvm;
using namespace MIPatternMatch;

Register llvm::constrainRegToClass(MachineRegisterInfo &MRI,
                                   const TargetInstrInfo &TII,
                                   const RegisterBankInfo &RBI, Register Reg,
                                   const TargetRegisterClass &RegClass) {
  if (!RBI.constrainGenericRegister(Reg, RegClass, MRI))
    return MRI.createVirtualRegister(&RegClass);

  return Reg;
}

Register llvm::constrainOperandRegClass(
    const MachineFunction &MF, const TargetRegisterInfo &TRI,
    MachineRegisterInfo &MRI, const TargetInstrInfo &TII,
    const RegisterBankInfo &RBI, MachineInstr &InsertPt,
    const TargetRegisterClass &RegClass, MachineOperand &RegMO) {
  Register Reg = RegMO.getReg();
  // Assume physical registers are properly constrained.
  assert(Reg.isVirtual() && "PhysReg not implemented");

  // Save the old register class to check whether
  // the change notifications will be required.
  // TODO: A better approach would be to pass
  // the observers to constrainRegToClass().
  auto *OldRegClass = MRI.getRegClassOrNull(Reg);
  Register ConstrainedReg = constrainRegToClass(MRI, TII, RBI, Reg, RegClass);
  // If we created a new virtual register because the class is not compatible
  // then create a copy between the new and the old register.
  if (ConstrainedReg != Reg) {
    MachineBasicBlock::iterator InsertIt(&InsertPt);
    MachineBasicBlock &MBB = *InsertPt.getParent();
    // FIXME: The copy needs to have the classes constrained for its operands.
    // Use operand's regbank to get the class for old register (Reg).
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
    if (GISelChangeObserver *Observer = MF.getObserver()) {
      Observer->changingInstr(*RegMO.getParent());
    }
    RegMO.setReg(ConstrainedReg);
    if (GISelChangeObserver *Observer = MF.getObserver()) {
      Observer->changedInstr(*RegMO.getParent());
    }
  } else if (OldRegClass != MRI.getRegClassOrNull(Reg)) {
    if (GISelChangeObserver *Observer = MF.getObserver()) {
      if (!RegMO.isDef()) {
        MachineInstr *RegDef = MRI.getVRegDef(Reg);
        Observer->changedInstr(*RegDef);
      }
      Observer->changingAllUsesOfReg(MRI, Reg);
      Observer->finishedChangingAllUsesOfReg();
    }
  }
  return ConstrainedReg;
}

Register llvm::constrainOperandRegClass(
    const MachineFunction &MF, const TargetRegisterInfo &TRI,
    MachineRegisterInfo &MRI, const TargetInstrInfo &TII,
    const RegisterBankInfo &RBI, MachineInstr &InsertPt, const MCInstrDesc &II,
    MachineOperand &RegMO, unsigned OpIdx) {
  Register Reg = RegMO.getReg();
  // Assume physical registers are properly constrained.
  assert(Reg.isVirtual() && "PhysReg not implemented");

  const TargetRegisterClass *OpRC = TII.getRegClass(II, OpIdx, &TRI, MF);
  // Some of the target independent instructions, like COPY, may not impose any
  // register class constraints on some of their operands: If it's a use, we can
  // skip constraining as the instruction defining the register would constrain
  // it.

  if (OpRC) {
    // Obtain the RC from incoming regbank if it is a proper sub-class. Operands
    // can have multiple regbanks for a superclass that combine different
    // register types (E.g., AMDGPU's VGPR and AGPR). The regbank ambiguity
    // resolved by targets during regbankselect should not be overridden.
    if (const auto *SubRC = TRI.getCommonSubClass(
            OpRC, TRI.getConstrainedRegClassForOperand(RegMO, MRI)))
      OpRC = SubRC;

    OpRC = TRI.getAllocatableClass(OpRC);
  }

  if (!OpRC) {
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
  return constrainOperandRegClass(MF, TRI, MRI, TII, RBI, InsertPt, *OpRC,
                                  RegMO);
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

    Register Reg = MO.getReg();
    // Physical registers don't need to be constrained.
    if (Reg.isPhysical())
      continue;

    // Register operands with a value of 0 (e.g. predicate operands) don't need
    // to be constrained.
    if (Reg == 0)
      continue;

    // If the operand is a vreg, we should constrain its regclass, and only
    // insert COPYs if that's impossible.
    // constrainOperandRegClass does that for us.
    constrainOperandRegClass(MF, TRI, MRI, TII, RBI, I, I.getDesc(), MO, OpI);

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

bool llvm::canReplaceReg(Register DstReg, Register SrcReg,
                         MachineRegisterInfo &MRI) {
  // Give up if either DstReg or SrcReg  is a physical register.
  if (DstReg.isPhysical() || SrcReg.isPhysical())
    return false;
  // Give up if the types don't match.
  if (MRI.getType(DstReg) != MRI.getType(SrcReg))
    return false;
  // Replace if either DstReg has no constraints or the register
  // constraints match.
  return !MRI.getRegClassOrRegBank(DstReg) ||
         MRI.getRegClassOrRegBank(DstReg) == MRI.getRegClassOrRegBank(SrcReg);
}

bool llvm::isTriviallyDead(const MachineInstr &MI,
                           const MachineRegisterInfo &MRI) {
  // FIXME: This logical is mostly duplicated with
  // DeadMachineInstructionElim::isDead. Why is LOCAL_ESCAPE not considered in
  // MachineInstr::isLabel?

  // Don't delete frame allocation labels.
  if (MI.getOpcode() == TargetOpcode::LOCAL_ESCAPE)
    return false;
  // LIFETIME markers should be preserved even if they seem dead.
  if (MI.getOpcode() == TargetOpcode::LIFETIME_START ||
      MI.getOpcode() == TargetOpcode::LIFETIME_END)
    return false;

  // If we can move an instruction, we can remove it.  Otherwise, it has
  // a side-effect of some sort.
  bool SawStore = false;
  if (!MI.isSafeToMove(/*AA=*/nullptr, SawStore) && !MI.isPHI())
    return false;

  // Instructions without side-effects are dead iff they only define dead vregs.
  for (const auto &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef())
      continue;

    Register Reg = MO.getReg();
    if (Reg.isPhysical() || !MRI.use_nodbg_empty(Reg))
      return false;
  }
  return true;
}

static void reportGISelDiagnostic(DiagnosticSeverity Severity,
                                  MachineFunction &MF,
                                  const TargetPassConfig &TPC,
                                  MachineOptimizationRemarkEmitter &MORE,
                                  MachineOptimizationRemarkMissed &R) {
  bool IsFatal = Severity == DS_Error &&
                 TPC.isGlobalISelAbortEnabled();
  // Print the function name explicitly if we don't have a debug location (which
  // makes the diagnostic less useful) or if we're going to emit a raw error.
  if (!R.getLocation().isValid() || IsFatal)
    R << (" (in function: " + MF.getName() + ")").str();

  if (IsFatal)
    report_fatal_error(Twine(R.getMsg()));
  else
    MORE.emit(R);
}

void llvm::reportGISelWarning(MachineFunction &MF, const TargetPassConfig &TPC,
                              MachineOptimizationRemarkEmitter &MORE,
                              MachineOptimizationRemarkMissed &R) {
  reportGISelDiagnostic(DS_Warning, MF, TPC, MORE, R);
}

void llvm::reportGISelFailure(MachineFunction &MF, const TargetPassConfig &TPC,
                              MachineOptimizationRemarkEmitter &MORE,
                              MachineOptimizationRemarkMissed &R) {
  MF.getProperties().set(MachineFunctionProperties::Property::FailedISel);
  reportGISelDiagnostic(DS_Error, MF, TPC, MORE, R);
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

std::optional<APInt> llvm::getIConstantVRegVal(Register VReg,
                                               const MachineRegisterInfo &MRI) {
  std::optional<ValueAndVReg> ValAndVReg = getIConstantVRegValWithLookThrough(
      VReg, MRI, /*LookThroughInstrs*/ false);
  assert((!ValAndVReg || ValAndVReg->VReg == VReg) &&
         "Value found while looking through instrs");
  if (!ValAndVReg)
    return std::nullopt;
  return ValAndVReg->Value;
}

std::optional<int64_t>
llvm::getIConstantVRegSExtVal(Register VReg, const MachineRegisterInfo &MRI) {
  std::optional<APInt> Val = getIConstantVRegVal(VReg, MRI);
  if (Val && Val->getBitWidth() <= 64)
    return Val->getSExtValue();
  return std::nullopt;
}

namespace {

typedef std::function<bool(const MachineInstr *)> IsOpcodeFn;
typedef std::function<std::optional<APInt>(const MachineInstr *MI)> GetAPCstFn;

std::optional<ValueAndVReg> getConstantVRegValWithLookThrough(
    Register VReg, const MachineRegisterInfo &MRI, IsOpcodeFn IsConstantOpcode,
    GetAPCstFn getAPCstValue, bool LookThroughInstrs = true,
    bool LookThroughAnyExt = false) {
  SmallVector<std::pair<unsigned, unsigned>, 4> SeenOpcodes;
  MachineInstr *MI;

  while ((MI = MRI.getVRegDef(VReg)) && !IsConstantOpcode(MI) &&
         LookThroughInstrs) {
    switch (MI->getOpcode()) {
    case TargetOpcode::G_ANYEXT:
      if (!LookThroughAnyExt)
        return std::nullopt;
      [[fallthrough]];
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
      if (VReg.isPhysical())
        return std::nullopt;
      break;
    case TargetOpcode::G_INTTOPTR:
      VReg = MI->getOperand(1).getReg();
      break;
    default:
      return std::nullopt;
    }
  }
  if (!MI || !IsConstantOpcode(MI))
    return std::nullopt;

  std::optional<APInt> MaybeVal = getAPCstValue(MI);
  if (!MaybeVal)
    return std::nullopt;
  APInt &Val = *MaybeVal;
  while (!SeenOpcodes.empty()) {
    std::pair<unsigned, unsigned> OpcodeAndSize = SeenOpcodes.pop_back_val();
    switch (OpcodeAndSize.first) {
    case TargetOpcode::G_TRUNC:
      Val = Val.trunc(OpcodeAndSize.second);
      break;
    case TargetOpcode::G_ANYEXT:
    case TargetOpcode::G_SEXT:
      Val = Val.sext(OpcodeAndSize.second);
      break;
    case TargetOpcode::G_ZEXT:
      Val = Val.zext(OpcodeAndSize.second);
      break;
    }
  }

  return ValueAndVReg{Val, VReg};
}

bool isIConstant(const MachineInstr *MI) {
  if (!MI)
    return false;
  return MI->getOpcode() == TargetOpcode::G_CONSTANT;
}

bool isFConstant(const MachineInstr *MI) {
  if (!MI)
    return false;
  return MI->getOpcode() == TargetOpcode::G_FCONSTANT;
}

bool isAnyConstant(const MachineInstr *MI) {
  if (!MI)
    return false;
  unsigned Opc = MI->getOpcode();
  return Opc == TargetOpcode::G_CONSTANT || Opc == TargetOpcode::G_FCONSTANT;
}

std::optional<APInt> getCImmAsAPInt(const MachineInstr *MI) {
  const MachineOperand &CstVal = MI->getOperand(1);
  if (CstVal.isCImm())
    return CstVal.getCImm()->getValue();
  return std::nullopt;
}

std::optional<APInt> getCImmOrFPImmAsAPInt(const MachineInstr *MI) {
  const MachineOperand &CstVal = MI->getOperand(1);
  if (CstVal.isCImm())
    return CstVal.getCImm()->getValue();
  if (CstVal.isFPImm())
    return CstVal.getFPImm()->getValueAPF().bitcastToAPInt();
  return std::nullopt;
}

} // end anonymous namespace

std::optional<ValueAndVReg> llvm::getIConstantVRegValWithLookThrough(
    Register VReg, const MachineRegisterInfo &MRI, bool LookThroughInstrs) {
  return getConstantVRegValWithLookThrough(VReg, MRI, isIConstant,
                                           getCImmAsAPInt, LookThroughInstrs);
}

std::optional<ValueAndVReg> llvm::getAnyConstantVRegValWithLookThrough(
    Register VReg, const MachineRegisterInfo &MRI, bool LookThroughInstrs,
    bool LookThroughAnyExt) {
  return getConstantVRegValWithLookThrough(
      VReg, MRI, isAnyConstant, getCImmOrFPImmAsAPInt, LookThroughInstrs,
      LookThroughAnyExt);
}

std::optional<FPValueAndVReg> llvm::getFConstantVRegValWithLookThrough(
    Register VReg, const MachineRegisterInfo &MRI, bool LookThroughInstrs) {
  auto Reg = getConstantVRegValWithLookThrough(
      VReg, MRI, isFConstant, getCImmOrFPImmAsAPInt, LookThroughInstrs);
  if (!Reg)
    return std::nullopt;
  return FPValueAndVReg{getConstantFPVRegVal(Reg->VReg, MRI)->getValueAPF(),
                        Reg->VReg};
}

const ConstantFP *
llvm::getConstantFPVRegVal(Register VReg, const MachineRegisterInfo &MRI) {
  MachineInstr *MI = MRI.getVRegDef(VReg);
  if (TargetOpcode::G_FCONSTANT != MI->getOpcode())
    return nullptr;
  return MI->getOperand(1).getFPImm();
}

std::optional<DefinitionAndSourceRegister>
llvm::getDefSrcRegIgnoringCopies(Register Reg, const MachineRegisterInfo &MRI) {
  Register DefSrcReg = Reg;
  auto *DefMI = MRI.getVRegDef(Reg);
  auto DstTy = MRI.getType(DefMI->getOperand(0).getReg());
  if (!DstTy.isValid())
    return std::nullopt;
  unsigned Opc = DefMI->getOpcode();
  while (Opc == TargetOpcode::COPY || isPreISelGenericOptimizationHint(Opc)) {
    Register SrcReg = DefMI->getOperand(1).getReg();
    auto SrcTy = MRI.getType(SrcReg);
    if (!SrcTy.isValid())
      break;
    DefMI = MRI.getVRegDef(SrcReg);
    DefSrcReg = SrcReg;
    Opc = DefMI->getOpcode();
  }
  return DefinitionAndSourceRegister{DefMI, DefSrcReg};
}

MachineInstr *llvm::getDefIgnoringCopies(Register Reg,
                                         const MachineRegisterInfo &MRI) {
  std::optional<DefinitionAndSourceRegister> DefSrcReg =
      getDefSrcRegIgnoringCopies(Reg, MRI);
  return DefSrcReg ? DefSrcReg->MI : nullptr;
}

Register llvm::getSrcRegIgnoringCopies(Register Reg,
                                       const MachineRegisterInfo &MRI) {
  std::optional<DefinitionAndSourceRegister> DefSrcReg =
      getDefSrcRegIgnoringCopies(Reg, MRI);
  return DefSrcReg ? DefSrcReg->Reg : Register();
}

MachineInstr *llvm::getOpcodeDef(unsigned Opcode, Register Reg,
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

std::optional<APInt> llvm::ConstantFoldBinOp(unsigned Opcode,
                                             const Register Op1,
                                             const Register Op2,
                                             const MachineRegisterInfo &MRI) {
  auto MaybeOp2Cst = getAnyConstantVRegValWithLookThrough(Op2, MRI, false);
  if (!MaybeOp2Cst)
    return std::nullopt;

  auto MaybeOp1Cst = getAnyConstantVRegValWithLookThrough(Op1, MRI, false);
  if (!MaybeOp1Cst)
    return std::nullopt;

  const APInt &C1 = MaybeOp1Cst->Value;
  const APInt &C2 = MaybeOp2Cst->Value;
  switch (Opcode) {
  default:
    break;
  case TargetOpcode::G_ADD:
  case TargetOpcode::G_PTR_ADD:
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
  case TargetOpcode::G_SMIN:
    return APIntOps::smin(C1, C2);
  case TargetOpcode::G_SMAX:
    return APIntOps::smax(C1, C2);
  case TargetOpcode::G_UMIN:
    return APIntOps::umin(C1, C2);
  case TargetOpcode::G_UMAX:
    return APIntOps::umax(C1, C2);
  }

  return std::nullopt;
}

std::optional<APFloat>
llvm::ConstantFoldFPBinOp(unsigned Opcode, const Register Op1,
                          const Register Op2, const MachineRegisterInfo &MRI) {
  const ConstantFP *Op2Cst = getConstantFPVRegVal(Op2, MRI);
  if (!Op2Cst)
    return std::nullopt;

  const ConstantFP *Op1Cst = getConstantFPVRegVal(Op1, MRI);
  if (!Op1Cst)
    return std::nullopt;

  APFloat C1 = Op1Cst->getValueAPF();
  const APFloat &C2 = Op2Cst->getValueAPF();
  switch (Opcode) {
  case TargetOpcode::G_FADD:
    C1.add(C2, APFloat::rmNearestTiesToEven);
    return C1;
  case TargetOpcode::G_FSUB:
    C1.subtract(C2, APFloat::rmNearestTiesToEven);
    return C1;
  case TargetOpcode::G_FMUL:
    C1.multiply(C2, APFloat::rmNearestTiesToEven);
    return C1;
  case TargetOpcode::G_FDIV:
    C1.divide(C2, APFloat::rmNearestTiesToEven);
    return C1;
  case TargetOpcode::G_FREM:
    C1.mod(C2);
    return C1;
  case TargetOpcode::G_FCOPYSIGN:
    C1.copySign(C2);
    return C1;
  case TargetOpcode::G_FMINNUM:
    return minnum(C1, C2);
  case TargetOpcode::G_FMAXNUM:
    return maxnum(C1, C2);
  case TargetOpcode::G_FMINIMUM:
    return minimum(C1, C2);
  case TargetOpcode::G_FMAXIMUM:
    return maximum(C1, C2);
  case TargetOpcode::G_FMINNUM_IEEE:
  case TargetOpcode::G_FMAXNUM_IEEE:
    // FIXME: These operations were unfortunately named. fminnum/fmaxnum do not
    // follow the IEEE behavior for signaling nans and follow libm's fmin/fmax,
    // and currently there isn't a nice wrapper in APFloat for the version with
    // correct snan handling.
    break;
  default:
    break;
  }

  return std::nullopt;
}

SmallVector<APInt>
llvm::ConstantFoldVectorBinop(unsigned Opcode, const Register Op1,
                              const Register Op2,
                              const MachineRegisterInfo &MRI) {
  auto *SrcVec2 = getOpcodeDef<GBuildVector>(Op2, MRI);
  if (!SrcVec2)
    return SmallVector<APInt>();

  auto *SrcVec1 = getOpcodeDef<GBuildVector>(Op1, MRI);
  if (!SrcVec1)
    return SmallVector<APInt>();

  SmallVector<APInt> FoldedElements;
  for (unsigned Idx = 0, E = SrcVec1->getNumSources(); Idx < E; ++Idx) {
    auto MaybeCst = ConstantFoldBinOp(Opcode, SrcVec1->getSourceReg(Idx),
                                      SrcVec2->getSourceReg(Idx), MRI);
    if (!MaybeCst)
      return SmallVector<APInt>();
    FoldedElements.push_back(*MaybeCst);
  }
  return FoldedElements;
}

bool llvm::isKnownNeverNaN(Register Val, const MachineRegisterInfo &MRI,
                           bool SNaN) {
  const MachineInstr *DefMI = MRI.getVRegDef(Val);
  if (!DefMI)
    return false;

  const TargetMachine& TM = DefMI->getMF()->getTarget();
  if (DefMI->getFlag(MachineInstr::FmNoNans) || TM.Options.NoNaNsFPMath)
    return true;

  // If the value is a constant, we can obviously see if it is a NaN or not.
  if (const ConstantFP *FPVal = getConstantFPVRegVal(Val, MRI)) {
    return !FPVal->getValueAPF().isNaN() ||
           (SNaN && !FPVal->getValueAPF().isSignaling());
  }

  if (DefMI->getOpcode() == TargetOpcode::G_BUILD_VECTOR) {
    for (const auto &Op : DefMI->uses())
      if (!isKnownNeverNaN(Op.getReg(), MRI, SNaN))
        return false;
    return true;
  }

  switch (DefMI->getOpcode()) {
  default:
    break;
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FREM:
  case TargetOpcode::G_FSIN:
  case TargetOpcode::G_FCOS:
  case TargetOpcode::G_FMA:
  case TargetOpcode::G_FMAD:
    if (SNaN)
      return true;

    // TODO: Need isKnownNeverInfinity
    return false;
  case TargetOpcode::G_FMINNUM_IEEE:
  case TargetOpcode::G_FMAXNUM_IEEE: {
    if (SNaN)
      return true;
    // This can return a NaN if either operand is an sNaN, or if both operands
    // are NaN.
    return (isKnownNeverNaN(DefMI->getOperand(1).getReg(), MRI) &&
            isKnownNeverSNaN(DefMI->getOperand(2).getReg(), MRI)) ||
           (isKnownNeverSNaN(DefMI->getOperand(1).getReg(), MRI) &&
            isKnownNeverNaN(DefMI->getOperand(2).getReg(), MRI));
  }
  case TargetOpcode::G_FMINNUM:
  case TargetOpcode::G_FMAXNUM: {
    // Only one needs to be known not-nan, since it will be returned if the
    // other ends up being one.
    return isKnownNeverNaN(DefMI->getOperand(1).getReg(), MRI, SNaN) ||
           isKnownNeverNaN(DefMI->getOperand(2).getReg(), MRI, SNaN);
  }
  }

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

Align llvm::inferAlignFromPtrInfo(MachineFunction &MF,
                                  const MachinePointerInfo &MPO) {
  auto PSV = dyn_cast_if_present<const PseudoSourceValue *>(MPO.V);
  if (auto FSPV = dyn_cast_or_null<FixedStackPseudoSourceValue>(PSV)) {
    MachineFrameInfo &MFI = MF.getFrameInfo();
    return commonAlignment(MFI.getObjectAlign(FSPV->getFrameIndex()),
                           MPO.Offset);
  }

  if (const Value *V = dyn_cast_if_present<const Value *>(MPO.V)) {
    const Module *M = MF.getFunction().getParent();
    return V->getPointerAlignment(M->getDataLayout());
  }

  return Align(1);
}

Register llvm::getFunctionLiveInPhysReg(MachineFunction &MF,
                                        const TargetInstrInfo &TII,
                                        MCRegister PhysReg,
                                        const TargetRegisterClass &RC,
                                        const DebugLoc &DL, LLT RegTy) {
  MachineBasicBlock &EntryMBB = MF.front();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register LiveIn = MRI.getLiveInVirtReg(PhysReg);
  if (LiveIn) {
    MachineInstr *Def = MRI.getVRegDef(LiveIn);
    if (Def) {
      // FIXME: Should the verifier check this is in the entry block?
      assert(Def->getParent() == &EntryMBB && "live-in copy not in entry block");
      return LiveIn;
    }

    // It's possible the incoming argument register and copy was added during
    // lowering, but later deleted due to being/becoming dead. If this happens,
    // re-insert the copy.
  } else {
    // The live in register was not present, so add it.
    LiveIn = MF.addLiveIn(PhysReg, &RC);
    if (RegTy.isValid())
      MRI.setType(LiveIn, RegTy);
  }

  BuildMI(EntryMBB, EntryMBB.begin(), DL, TII.get(TargetOpcode::COPY), LiveIn)
    .addReg(PhysReg);
  if (!EntryMBB.isLiveIn(PhysReg))
    EntryMBB.addLiveIn(PhysReg);
  return LiveIn;
}

std::optional<APInt> llvm::ConstantFoldExtOp(unsigned Opcode,
                                             const Register Op1, uint64_t Imm,
                                             const MachineRegisterInfo &MRI) {
  auto MaybeOp1Cst = getIConstantVRegVal(Op1, MRI);
  if (MaybeOp1Cst) {
    switch (Opcode) {
    default:
      break;
    case TargetOpcode::G_SEXT_INREG: {
      LLT Ty = MRI.getType(Op1);
      return MaybeOp1Cst->trunc(Imm).sext(Ty.getScalarSizeInBits());
    }
    }
  }
  return std::nullopt;
}

std::optional<APFloat>
llvm::ConstantFoldIntToFloat(unsigned Opcode, LLT DstTy, Register Src,
                             const MachineRegisterInfo &MRI) {
  assert(Opcode == TargetOpcode::G_SITOFP || Opcode == TargetOpcode::G_UITOFP);
  if (auto MaybeSrcVal = getIConstantVRegVal(Src, MRI)) {
    APFloat DstVal(getFltSemanticForLLT(DstTy));
    DstVal.convertFromAPInt(*MaybeSrcVal, Opcode == TargetOpcode::G_SITOFP,
                            APFloat::rmNearestTiesToEven);
    return DstVal;
  }
  return std::nullopt;
}

std::optional<SmallVector<unsigned>>
llvm::ConstantFoldCTLZ(Register Src, const MachineRegisterInfo &MRI) {
  LLT Ty = MRI.getType(Src);
  SmallVector<unsigned> FoldedCTLZs;
  auto tryFoldScalar = [&](Register R) -> std::optional<unsigned> {
    auto MaybeCst = getIConstantVRegVal(R, MRI);
    if (!MaybeCst)
      return std::nullopt;
    return MaybeCst->countl_zero();
  };
  if (Ty.isVector()) {
    // Try to constant fold each element.
    auto *BV = getOpcodeDef<GBuildVector>(Src, MRI);
    if (!BV)
      return std::nullopt;
    for (unsigned SrcIdx = 0; SrcIdx < BV->getNumSources(); ++SrcIdx) {
      if (auto MaybeFold = tryFoldScalar(BV->getSourceReg(SrcIdx))) {
        FoldedCTLZs.emplace_back(*MaybeFold);
        continue;
      }
      return std::nullopt;
    }
    return FoldedCTLZs;
  }
  if (auto MaybeCst = tryFoldScalar(Src)) {
    FoldedCTLZs.emplace_back(*MaybeCst);
    return FoldedCTLZs;
  }
  return std::nullopt;
}

bool llvm::isKnownToBeAPowerOfTwo(Register Reg, const MachineRegisterInfo &MRI,
                                  GISelKnownBits *KB) {
  std::optional<DefinitionAndSourceRegister> DefSrcReg =
      getDefSrcRegIgnoringCopies(Reg, MRI);
  if (!DefSrcReg)
    return false;

  const MachineInstr &MI = *DefSrcReg->MI;
  const LLT Ty = MRI.getType(Reg);

  switch (MI.getOpcode()) {
  case TargetOpcode::G_CONSTANT: {
    unsigned BitWidth = Ty.getScalarSizeInBits();
    const ConstantInt *CI = MI.getOperand(1).getCImm();
    return CI->getValue().zextOrTrunc(BitWidth).isPowerOf2();
  }
  case TargetOpcode::G_SHL: {
    // A left-shift of a constant one will have exactly one bit set because
    // shifting the bit off the end is undefined.

    // TODO: Constant splat
    if (auto ConstLHS = getIConstantVRegVal(MI.getOperand(1).getReg(), MRI)) {
      if (*ConstLHS == 1)
        return true;
    }

    break;
  }
  case TargetOpcode::G_LSHR: {
    if (auto ConstLHS = getIConstantVRegVal(MI.getOperand(1).getReg(), MRI)) {
      if (ConstLHS->isSignMask())
        return true;
    }

    break;
  }
  case TargetOpcode::G_BUILD_VECTOR: {
    // TODO: Probably should have a recursion depth guard since you could have
    // bitcasted vector elements.
    for (const MachineOperand &MO : llvm::drop_begin(MI.operands()))
      if (!isKnownToBeAPowerOfTwo(MO.getReg(), MRI, KB))
        return false;

    return true;
  }
  case TargetOpcode::G_BUILD_VECTOR_TRUNC: {
    // Only handle constants since we would need to know if number of leading
    // zeros is greater than the truncation amount.
    const unsigned BitWidth = Ty.getScalarSizeInBits();
    for (const MachineOperand &MO : llvm::drop_begin(MI.operands())) {
      auto Const = getIConstantVRegVal(MO.getReg(), MRI);
      if (!Const || !Const->zextOrTrunc(BitWidth).isPowerOf2())
        return false;
    }

    return true;
  }
  default:
    break;
  }

  if (!KB)
    return false;

  // More could be done here, though the above checks are enough
  // to handle some common cases.

  // Fall back to computeKnownBits to catch other known cases.
  KnownBits Known = KB->getKnownBits(Reg);
  return (Known.countMaxPopulation() == 1) && (Known.countMinPopulation() == 1);
}

void llvm::getSelectionDAGFallbackAnalysisUsage(AnalysisUsage &AU) {
  AU.addPreserved<StackProtector>();
}

LLT llvm::getLCMType(LLT OrigTy, LLT TargetTy) {
  const unsigned OrigSize = OrigTy.getSizeInBits();
  const unsigned TargetSize = TargetTy.getSizeInBits();

  if (OrigSize == TargetSize)
    return OrigTy;

  if (OrigTy.isVector()) {
    const LLT OrigElt = OrigTy.getElementType();

    if (TargetTy.isVector()) {
      const LLT TargetElt = TargetTy.getElementType();

      if (OrigElt.getSizeInBits() == TargetElt.getSizeInBits()) {
        int GCDElts =
            std::gcd(OrigTy.getNumElements(), TargetTy.getNumElements());
        // Prefer the original element type.
        ElementCount Mul = OrigTy.getElementCount() * TargetTy.getNumElements();
        return LLT::vector(Mul.divideCoefficientBy(GCDElts),
                           OrigTy.getElementType());
      }
    } else {
      if (OrigElt.getSizeInBits() == TargetSize)
        return OrigTy;
    }

    unsigned LCMSize = std::lcm(OrigSize, TargetSize);
    return LLT::fixed_vector(LCMSize / OrigElt.getSizeInBits(), OrigElt);
  }

  if (TargetTy.isVector()) {
    unsigned LCMSize = std::lcm(OrigSize, TargetSize);
    return LLT::fixed_vector(LCMSize / OrigSize, OrigTy);
  }

  unsigned LCMSize = std::lcm(OrigSize, TargetSize);

  // Preserve pointer types.
  if (LCMSize == OrigSize)
    return OrigTy;
  if (LCMSize == TargetSize)
    return TargetTy;

  return LLT::scalar(LCMSize);
}

LLT llvm::getCoverTy(LLT OrigTy, LLT TargetTy) {
  if (!OrigTy.isVector() || !TargetTy.isVector() || OrigTy == TargetTy ||
      (OrigTy.getScalarSizeInBits() != TargetTy.getScalarSizeInBits()))
    return getLCMType(OrigTy, TargetTy);

  unsigned OrigTyNumElts = OrigTy.getNumElements();
  unsigned TargetTyNumElts = TargetTy.getNumElements();
  if (OrigTyNumElts % TargetTyNumElts == 0)
    return OrigTy;

  unsigned NumElts = alignTo(OrigTyNumElts, TargetTyNumElts);
  return LLT::scalarOrVector(ElementCount::getFixed(NumElts),
                             OrigTy.getElementType());
}

LLT llvm::getGCDType(LLT OrigTy, LLT TargetTy) {
  const unsigned OrigSize = OrigTy.getSizeInBits();
  const unsigned TargetSize = TargetTy.getSizeInBits();

  if (OrigSize == TargetSize)
    return OrigTy;

  if (OrigTy.isVector()) {
    LLT OrigElt = OrigTy.getElementType();
    if (TargetTy.isVector()) {
      LLT TargetElt = TargetTy.getElementType();
      if (OrigElt.getSizeInBits() == TargetElt.getSizeInBits()) {
        int GCD = std::gcd(OrigTy.getNumElements(), TargetTy.getNumElements());
        return LLT::scalarOrVector(ElementCount::getFixed(GCD), OrigElt);
      }
    } else {
      // If the source is a vector of pointers, return a pointer element.
      if (OrigElt.getSizeInBits() == TargetSize)
        return OrigElt;
    }

    unsigned GCD = std::gcd(OrigSize, TargetSize);
    if (GCD == OrigElt.getSizeInBits())
      return OrigElt;

    // If we can't produce the original element type, we have to use a smaller
    // scalar.
    if (GCD < OrigElt.getSizeInBits())
      return LLT::scalar(GCD);
    return LLT::fixed_vector(GCD / OrigElt.getSizeInBits(), OrigElt);
  }

  if (TargetTy.isVector()) {
    // Try to preserve the original element type.
    LLT TargetElt = TargetTy.getElementType();
    if (TargetElt.getSizeInBits() == OrigSize)
      return OrigTy;
  }

  unsigned GCD = std::gcd(OrigSize, TargetSize);
  return LLT::scalar(GCD);
}

std::optional<int> llvm::getSplatIndex(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR &&
         "Only G_SHUFFLE_VECTOR can have a splat index!");
  ArrayRef<int> Mask = MI.getOperand(3).getShuffleMask();
  auto FirstDefinedIdx = find_if(Mask, [](int Elt) { return Elt >= 0; });

  // If all elements are undefined, this shuffle can be considered a splat.
  // Return 0 for better potential for callers to simplify.
  if (FirstDefinedIdx == Mask.end())
    return 0;

  // Make sure all remaining elements are either undef or the same
  // as the first non-undef value.
  int SplatValue = *FirstDefinedIdx;
  if (any_of(make_range(std::next(FirstDefinedIdx), Mask.end()),
             [&SplatValue](int Elt) { return Elt >= 0 && Elt != SplatValue; }))
    return std::nullopt;

  return SplatValue;
}

static bool isBuildVectorOp(unsigned Opcode) {
  return Opcode == TargetOpcode::G_BUILD_VECTOR ||
         Opcode == TargetOpcode::G_BUILD_VECTOR_TRUNC;
}

namespace {

std::optional<ValueAndVReg> getAnyConstantSplat(Register VReg,
                                                const MachineRegisterInfo &MRI,
                                                bool AllowUndef) {
  MachineInstr *MI = getDefIgnoringCopies(VReg, MRI);
  if (!MI)
    return std::nullopt;

  bool isConcatVectorsOp = MI->getOpcode() == TargetOpcode::G_CONCAT_VECTORS;
  if (!isBuildVectorOp(MI->getOpcode()) && !isConcatVectorsOp)
    return std::nullopt;

  std::optional<ValueAndVReg> SplatValAndReg;
  for (MachineOperand &Op : MI->uses()) {
    Register Element = Op.getReg();
    // If we have a G_CONCAT_VECTOR, we recursively look into the
    // vectors that we're concatenating to see if they're splats.
    auto ElementValAndReg =
        isConcatVectorsOp
            ? getAnyConstantSplat(Element, MRI, AllowUndef)
            : getAnyConstantVRegValWithLookThrough(Element, MRI, true, true);

    // If AllowUndef, treat undef as value that will result in a constant splat.
    if (!ElementValAndReg) {
      if (AllowUndef && isa<GImplicitDef>(MRI.getVRegDef(Element)))
        continue;
      return std::nullopt;
    }

    // Record splat value
    if (!SplatValAndReg)
      SplatValAndReg = ElementValAndReg;

    // Different constant than the one already recorded, not a constant splat.
    if (SplatValAndReg->Value != ElementValAndReg->Value)
      return std::nullopt;
  }

  return SplatValAndReg;
}

} // end anonymous namespace

bool llvm::isBuildVectorConstantSplat(const Register Reg,
                                      const MachineRegisterInfo &MRI,
                                      int64_t SplatValue, bool AllowUndef) {
  if (auto SplatValAndReg = getAnyConstantSplat(Reg, MRI, AllowUndef))
    return mi_match(SplatValAndReg->VReg, MRI, m_SpecificICst(SplatValue));
  return false;
}

bool llvm::isBuildVectorConstantSplat(const MachineInstr &MI,
                                      const MachineRegisterInfo &MRI,
                                      int64_t SplatValue, bool AllowUndef) {
  return isBuildVectorConstantSplat(MI.getOperand(0).getReg(), MRI, SplatValue,
                                    AllowUndef);
}

std::optional<APInt>
llvm::getIConstantSplatVal(const Register Reg, const MachineRegisterInfo &MRI) {
  if (auto SplatValAndReg =
          getAnyConstantSplat(Reg, MRI, /* AllowUndef */ false)) {
    std::optional<ValueAndVReg> ValAndVReg =
        getIConstantVRegValWithLookThrough(SplatValAndReg->VReg, MRI);
    return ValAndVReg->Value;
  }

  return std::nullopt;
}

std::optional<APInt>
llvm::getIConstantSplatVal(const MachineInstr &MI,
                           const MachineRegisterInfo &MRI) {
  return getIConstantSplatVal(MI.getOperand(0).getReg(), MRI);
}

std::optional<int64_t>
llvm::getIConstantSplatSExtVal(const Register Reg,
                               const MachineRegisterInfo &MRI) {
  if (auto SplatValAndReg =
          getAnyConstantSplat(Reg, MRI, /* AllowUndef */ false))
    return getIConstantVRegSExtVal(SplatValAndReg->VReg, MRI);
  return std::nullopt;
}

std::optional<int64_t>
llvm::getIConstantSplatSExtVal(const MachineInstr &MI,
                               const MachineRegisterInfo &MRI) {
  return getIConstantSplatSExtVal(MI.getOperand(0).getReg(), MRI);
}

std::optional<FPValueAndVReg>
llvm::getFConstantSplat(Register VReg, const MachineRegisterInfo &MRI,
                        bool AllowUndef) {
  if (auto SplatValAndReg = getAnyConstantSplat(VReg, MRI, AllowUndef))
    return getFConstantVRegValWithLookThrough(SplatValAndReg->VReg, MRI);
  return std::nullopt;
}

bool llvm::isBuildVectorAllZeros(const MachineInstr &MI,
                                 const MachineRegisterInfo &MRI,
                                 bool AllowUndef) {
  return isBuildVectorConstantSplat(MI, MRI, 0, AllowUndef);
}

bool llvm::isBuildVectorAllOnes(const MachineInstr &MI,
                                const MachineRegisterInfo &MRI,
                                bool AllowUndef) {
  return isBuildVectorConstantSplat(MI, MRI, -1, AllowUndef);
}

std::optional<RegOrConstant>
llvm::getVectorSplat(const MachineInstr &MI, const MachineRegisterInfo &MRI) {
  unsigned Opc = MI.getOpcode();
  if (!isBuildVectorOp(Opc))
    return std::nullopt;
  if (auto Splat = getIConstantSplatSExtVal(MI, MRI))
    return RegOrConstant(*Splat);
  auto Reg = MI.getOperand(1).getReg();
  if (any_of(make_range(MI.operands_begin() + 2, MI.operands_end()),
             [&Reg](const MachineOperand &Op) { return Op.getReg() != Reg; }))
    return std::nullopt;
  return RegOrConstant(Reg);
}

static bool isConstantScalar(const MachineInstr &MI,
                             const MachineRegisterInfo &MRI,
                             bool AllowFP = true,
                             bool AllowOpaqueConstants = true) {
  switch (MI.getOpcode()) {
  case TargetOpcode::G_CONSTANT:
  case TargetOpcode::G_IMPLICIT_DEF:
    return true;
  case TargetOpcode::G_FCONSTANT:
    return AllowFP;
  case TargetOpcode::G_GLOBAL_VALUE:
  case TargetOpcode::G_FRAME_INDEX:
  case TargetOpcode::G_BLOCK_ADDR:
  case TargetOpcode::G_JUMP_TABLE:
    return AllowOpaqueConstants;
  default:
    return false;
  }
}

bool llvm::isConstantOrConstantVector(MachineInstr &MI,
                                      const MachineRegisterInfo &MRI) {
  Register Def = MI.getOperand(0).getReg();
  if (auto C = getIConstantVRegValWithLookThrough(Def, MRI))
    return true;
  GBuildVector *BV = dyn_cast<GBuildVector>(&MI);
  if (!BV)
    return false;
  for (unsigned SrcIdx = 0; SrcIdx < BV->getNumSources(); ++SrcIdx) {
    if (getIConstantVRegValWithLookThrough(BV->getSourceReg(SrcIdx), MRI) ||
        getOpcodeDef<GImplicitDef>(BV->getSourceReg(SrcIdx), MRI))
      continue;
    return false;
  }
  return true;
}

bool llvm::isConstantOrConstantVector(const MachineInstr &MI,
                                      const MachineRegisterInfo &MRI,
                                      bool AllowFP, bool AllowOpaqueConstants) {
  if (isConstantScalar(MI, MRI, AllowFP, AllowOpaqueConstants))
    return true;

  if (!isBuildVectorOp(MI.getOpcode()))
    return false;

  const unsigned NumOps = MI.getNumOperands();
  for (unsigned I = 1; I != NumOps; ++I) {
    const MachineInstr *ElementDef = MRI.getVRegDef(MI.getOperand(I).getReg());
    if (!isConstantScalar(*ElementDef, MRI, AllowFP, AllowOpaqueConstants))
      return false;
  }

  return true;
}

std::optional<APInt>
llvm::isConstantOrConstantSplatVector(MachineInstr &MI,
                                      const MachineRegisterInfo &MRI) {
  Register Def = MI.getOperand(0).getReg();
  if (auto C = getIConstantVRegValWithLookThrough(Def, MRI))
    return C->Value;
  auto MaybeCst = getIConstantSplatSExtVal(MI, MRI);
  if (!MaybeCst)
    return std::nullopt;
  const unsigned ScalarSize = MRI.getType(Def).getScalarSizeInBits();
  return APInt(ScalarSize, *MaybeCst, true);
}

bool llvm::isNullOrNullSplat(const MachineInstr &MI,
                             const MachineRegisterInfo &MRI, bool AllowUndefs) {
  switch (MI.getOpcode()) {
  case TargetOpcode::G_IMPLICIT_DEF:
    return AllowUndefs;
  case TargetOpcode::G_CONSTANT:
    return MI.getOperand(1).getCImm()->isNullValue();
  case TargetOpcode::G_FCONSTANT: {
    const ConstantFP *FPImm = MI.getOperand(1).getFPImm();
    return FPImm->isZero() && !FPImm->isNegative();
  }
  default:
    if (!AllowUndefs) // TODO: isBuildVectorAllZeros assumes undef is OK already
      return false;
    return isBuildVectorAllZeros(MI, MRI);
  }
}

bool llvm::isAllOnesOrAllOnesSplat(const MachineInstr &MI,
                                   const MachineRegisterInfo &MRI,
                                   bool AllowUndefs) {
  switch (MI.getOpcode()) {
  case TargetOpcode::G_IMPLICIT_DEF:
    return AllowUndefs;
  case TargetOpcode::G_CONSTANT:
    return MI.getOperand(1).getCImm()->isAllOnesValue();
  default:
    if (!AllowUndefs) // TODO: isBuildVectorAllOnes assumes undef is OK already
      return false;
    return isBuildVectorAllOnes(MI, MRI);
  }
}

bool llvm::matchUnaryPredicate(
    const MachineRegisterInfo &MRI, Register Reg,
    std::function<bool(const Constant *ConstVal)> Match, bool AllowUndefs) {

  const MachineInstr *Def = getDefIgnoringCopies(Reg, MRI);
  if (AllowUndefs && Def->getOpcode() == TargetOpcode::G_IMPLICIT_DEF)
    return Match(nullptr);

  // TODO: Also handle fconstant
  if (Def->getOpcode() == TargetOpcode::G_CONSTANT)
    return Match(Def->getOperand(1).getCImm());

  if (Def->getOpcode() != TargetOpcode::G_BUILD_VECTOR)
    return false;

  for (unsigned I = 1, E = Def->getNumOperands(); I != E; ++I) {
    Register SrcElt = Def->getOperand(I).getReg();
    const MachineInstr *SrcDef = getDefIgnoringCopies(SrcElt, MRI);
    if (AllowUndefs && SrcDef->getOpcode() == TargetOpcode::G_IMPLICIT_DEF) {
      if (!Match(nullptr))
        return false;
      continue;
    }

    if (SrcDef->getOpcode() != TargetOpcode::G_CONSTANT ||
        !Match(SrcDef->getOperand(1).getCImm()))
      return false;
  }

  return true;
}

bool llvm::isConstTrueVal(const TargetLowering &TLI, int64_t Val, bool IsVector,
                          bool IsFP) {
  switch (TLI.getBooleanContents(IsVector, IsFP)) {
  case TargetLowering::UndefinedBooleanContent:
    return Val & 0x1;
  case TargetLowering::ZeroOrOneBooleanContent:
    return Val == 1;
  case TargetLowering::ZeroOrNegativeOneBooleanContent:
    return Val == -1;
  }
  llvm_unreachable("Invalid boolean contents");
}

bool llvm::isConstFalseVal(const TargetLowering &TLI, int64_t Val,
                           bool IsVector, bool IsFP) {
  switch (TLI.getBooleanContents(IsVector, IsFP)) {
  case TargetLowering::UndefinedBooleanContent:
    return ~Val & 0x1;
  case TargetLowering::ZeroOrOneBooleanContent:
  case TargetLowering::ZeroOrNegativeOneBooleanContent:
    return Val == 0;
  }
  llvm_unreachable("Invalid boolean contents");
}

int64_t llvm::getICmpTrueVal(const TargetLowering &TLI, bool IsVector,
                             bool IsFP) {
  switch (TLI.getBooleanContents(IsVector, IsFP)) {
  case TargetLowering::UndefinedBooleanContent:
  case TargetLowering::ZeroOrOneBooleanContent:
    return 1;
  case TargetLowering::ZeroOrNegativeOneBooleanContent:
    return -1;
  }
  llvm_unreachable("Invalid boolean contents");
}

bool llvm::shouldOptForSize(const MachineBasicBlock &MBB,
                            ProfileSummaryInfo *PSI, BlockFrequencyInfo *BFI) {
  const auto &F = MBB.getParent()->getFunction();
  return F.hasOptSize() || F.hasMinSize() ||
         llvm::shouldOptimizeForSize(MBB.getBasicBlock(), PSI, BFI);
}

void llvm::saveUsesAndErase(MachineInstr &MI, MachineRegisterInfo &MRI,
                            LostDebugLocObserver *LocObserver,
                            SmallInstListTy &DeadInstChain) {
  for (MachineOperand &Op : MI.uses()) {
    if (Op.isReg() && Op.getReg().isVirtual())
      DeadInstChain.insert(MRI.getVRegDef(Op.getReg()));
  }
  LLVM_DEBUG(dbgs() << MI << "Is dead; erasing.\n");
  DeadInstChain.remove(&MI);
  MI.eraseFromParent();
  if (LocObserver)
    LocObserver->checkpoint(false);
}

void llvm::eraseInstrs(ArrayRef<MachineInstr *> DeadInstrs,
                       MachineRegisterInfo &MRI,
                       LostDebugLocObserver *LocObserver) {
  SmallInstListTy DeadInstChain;
  for (MachineInstr *MI : DeadInstrs)
    saveUsesAndErase(*MI, MRI, LocObserver, DeadInstChain);

  while (!DeadInstChain.empty()) {
    MachineInstr *Inst = DeadInstChain.pop_back_val();
    if (!isTriviallyDead(*Inst, MRI))
      continue;
    saveUsesAndErase(*Inst, MRI, LocObserver, DeadInstChain);
  }
}

void llvm::eraseInstr(MachineInstr &MI, MachineRegisterInfo &MRI,
                      LostDebugLocObserver *LocObserver) {
  return eraseInstrs({&MI}, MRI, LocObserver);
}

void llvm::salvageDebugInfo(const MachineRegisterInfo &MRI, MachineInstr &MI) {
  for (auto &Def : MI.defs()) {
    assert(Def.isReg() && "Must be a reg");

    SmallVector<MachineOperand *, 16> DbgUsers;
    for (auto &MOUse : MRI.use_operands(Def.getReg())) {
      MachineInstr *DbgValue = MOUse.getParent();
      // Ignore partially formed DBG_VALUEs.
      if (DbgValue->isNonListDebugValue() && DbgValue->getNumOperands() == 4) {
        DbgUsers.push_back(&MOUse);
      }
    }

    if (!DbgUsers.empty()) {
      salvageDebugInfoForDbgValue(MRI, MI, DbgUsers);
    }
  }
}
