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
#include "llvm/ADT/Optional.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineSizeOpts.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/Target/TargetMachine.h"

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
  assert(Register::isVirtualRegister(Reg) && "PhysReg not implemented");

  Register ConstrainedReg = constrainRegToClass(MRI, TII, RBI, Reg, RegClass);
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
    if (GISelChangeObserver *Observer = MF.getObserver()) {
      Observer->changingInstr(*RegMO.getParent());
    }
    RegMO.setReg(ConstrainedReg);
    if (GISelChangeObserver *Observer = MF.getObserver()) {
      Observer->changedInstr(*RegMO.getParent());
    }
  } else {
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
  assert(Register::isVirtualRegister(Reg) && "PhysReg not implemented");

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
    if (Register::isPhysicalRegister(Reg))
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
  for (auto &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef())
      continue;

    Register Reg = MO.getReg();
    if (Register::isPhysicalRegister(Reg) || !MRI.use_nodbg_empty(Reg))
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
    report_fatal_error(R.getMsg());
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

Optional<APInt> llvm::getConstantVRegVal(Register VReg,
                                         const MachineRegisterInfo &MRI) {
  Optional<ValueAndVReg> ValAndVReg =
      getConstantVRegValWithLookThrough(VReg, MRI, /*LookThroughInstrs*/ false);
  assert((!ValAndVReg || ValAndVReg->VReg == VReg) &&
         "Value found while looking through instrs");
  if (!ValAndVReg)
    return None;
  return ValAndVReg->Value;
}

Optional<int64_t> llvm::getConstantVRegSExtVal(Register VReg,
                                               const MachineRegisterInfo &MRI) {
  Optional<APInt> Val = getConstantVRegVal(VReg, MRI);
  if (Val && Val->getBitWidth() <= 64)
    return Val->getSExtValue();
  return None;
}

Optional<ValueAndVReg> llvm::getConstantVRegValWithLookThrough(
    Register VReg, const MachineRegisterInfo &MRI, bool LookThroughInstrs,
    bool HandleFConstant, bool LookThroughAnyExt) {
  SmallVector<std::pair<unsigned, unsigned>, 4> SeenOpcodes;
  MachineInstr *MI;
  auto IsConstantOpcode = [HandleFConstant](unsigned Opcode) {
    return Opcode == TargetOpcode::G_CONSTANT ||
           (HandleFConstant && Opcode == TargetOpcode::G_FCONSTANT);
  };
  auto GetImmediateValue = [HandleFConstant,
                            &MRI](const MachineInstr &MI) -> Optional<APInt> {
    const MachineOperand &CstVal = MI.getOperand(1);
    if (!CstVal.isImm() && !CstVal.isCImm() &&
        (!HandleFConstant || !CstVal.isFPImm()))
      return None;
    if (!CstVal.isFPImm()) {
      unsigned BitWidth =
          MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      APInt Val = CstVal.isImm() ? APInt(BitWidth, CstVal.getImm())
                                 : CstVal.getCImm()->getValue();
      assert(Val.getBitWidth() == BitWidth &&
             "Value bitwidth doesn't match definition type");
      return Val;
    }
    return CstVal.getFPImm()->getValueAPF().bitcastToAPInt();
  };
  while ((MI = MRI.getVRegDef(VReg)) && !IsConstantOpcode(MI->getOpcode()) &&
         LookThroughInstrs) {
    switch (MI->getOpcode()) {
    case TargetOpcode::G_ANYEXT:
      if (!LookThroughAnyExt)
        return None;
      LLVM_FALLTHROUGH;
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
      if (Register::isPhysicalRegister(VReg))
        return None;
      break;
    case TargetOpcode::G_INTTOPTR:
      VReg = MI->getOperand(1).getReg();
      break;
    default:
      return None;
    }
  }
  if (!MI || !IsConstantOpcode(MI->getOpcode()))
    return None;

  Optional<APInt> MaybeVal = GetImmediateValue(*MI);
  if (!MaybeVal)
    return None;
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

const ConstantFP *
llvm::getConstantFPVRegVal(Register VReg, const MachineRegisterInfo &MRI) {
  MachineInstr *MI = MRI.getVRegDef(VReg);
  if (TargetOpcode::G_FCONSTANT != MI->getOpcode())
    return nullptr;
  return MI->getOperand(1).getFPImm();
}

Optional<DefinitionAndSourceRegister>
llvm::getDefSrcRegIgnoringCopies(Register Reg, const MachineRegisterInfo &MRI) {
  Register DefSrcReg = Reg;
  auto *DefMI = MRI.getVRegDef(Reg);
  auto DstTy = MRI.getType(DefMI->getOperand(0).getReg());
  if (!DstTy.isValid())
    return None;
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
  Optional<DefinitionAndSourceRegister> DefSrcReg =
      getDefSrcRegIgnoringCopies(Reg, MRI);
  return DefSrcReg ? DefSrcReg->MI : nullptr;
}

Register llvm::getSrcRegIgnoringCopies(Register Reg,
                                       const MachineRegisterInfo &MRI) {
  Optional<DefinitionAndSourceRegister> DefSrcReg =
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

Optional<APInt> llvm::ConstantFoldBinOp(unsigned Opcode, const Register Op1,
                                        const Register Op2,
                                        const MachineRegisterInfo &MRI) {
  auto MaybeOp2Cst = getConstantVRegVal(Op2, MRI);
  if (!MaybeOp2Cst)
    return None;

  auto MaybeOp1Cst = getConstantVRegVal(Op1, MRI);
  if (!MaybeOp1Cst)
    return None;

  const APInt &C1 = *MaybeOp1Cst;
  const APInt &C2 = *MaybeOp2Cst;
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

  return None;
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
  auto PSV = MPO.V.dyn_cast<const PseudoSourceValue *>();
  if (auto FSPV = dyn_cast_or_null<FixedStackPseudoSourceValue>(PSV)) {
    MachineFrameInfo &MFI = MF.getFrameInfo();
    return commonAlignment(MFI.getObjectAlign(FSPV->getFrameIndex()),
                           MPO.Offset);
  }

  if (const Value *V = MPO.V.dyn_cast<const Value *>()) {
    const Module *M = MF.getFunction().getParent();
    return V->getPointerAlignment(M->getDataLayout());
  }

  return Align(1);
}

Register llvm::getFunctionLiveInPhysReg(MachineFunction &MF,
                                        const TargetInstrInfo &TII,
                                        MCRegister PhysReg,
                                        const TargetRegisterClass &RC,
                                        LLT RegTy) {
  DebugLoc DL; // FIXME: Is no location the right choice?
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

Optional<APInt> llvm::ConstantFoldExtOp(unsigned Opcode, const Register Op1,
                                        uint64_t Imm,
                                        const MachineRegisterInfo &MRI) {
  auto MaybeOp1Cst = getConstantVRegVal(Op1, MRI);
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
  return None;
}

bool llvm::isKnownToBeAPowerOfTwo(Register Reg, const MachineRegisterInfo &MRI,
                                  GISelKnownBits *KB) {
  Optional<DefinitionAndSourceRegister> DefSrcReg =
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
    if (auto ConstLHS = getConstantVRegVal(MI.getOperand(1).getReg(), MRI)) {
      if (*ConstLHS == 1)
        return true;
    }

    break;
  }
  case TargetOpcode::G_LSHR: {
    if (auto ConstLHS = getConstantVRegVal(MI.getOperand(1).getReg(), MRI)) {
      if (ConstLHS->isSignMask())
        return true;
    }

    break;
  }
  default:
    break;
  }

  // TODO: Are all operands of a build vector constant powers of two?
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

static unsigned getLCMSize(unsigned OrigSize, unsigned TargetSize) {
  unsigned Mul = OrigSize * TargetSize;
  unsigned GCDSize = greatestCommonDivisor(OrigSize, TargetSize);
  return Mul / GCDSize;
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
        int GCDElts = greatestCommonDivisor(OrigTy.getNumElements(),
                                            TargetTy.getNumElements());
        // Prefer the original element type.
        int Mul = OrigTy.getNumElements() * TargetTy.getNumElements();
        return LLT::vector(Mul / GCDElts, OrigTy.getElementType());
      }
    } else {
      if (OrigElt.getSizeInBits() == TargetSize)
        return OrigTy;
    }

    unsigned LCMSize = getLCMSize(OrigSize, TargetSize);
    return LLT::vector(LCMSize / OrigElt.getSizeInBits(), OrigElt);
  }

  if (TargetTy.isVector()) {
    unsigned LCMSize = getLCMSize(OrigSize, TargetSize);
    return LLT::vector(LCMSize / OrigSize, OrigTy);
  }

  unsigned LCMSize = getLCMSize(OrigSize, TargetSize);

  // Preserve pointer types.
  if (LCMSize == OrigSize)
    return OrigTy;
  if (LCMSize == TargetSize)
    return TargetTy;

  return LLT::scalar(LCMSize);
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
        int GCD = greatestCommonDivisor(OrigTy.getNumElements(),
                                        TargetTy.getNumElements());
        return LLT::scalarOrVector(GCD, OrigElt);
      }
    } else {
      // If the source is a vector of pointers, return a pointer element.
      if (OrigElt.getSizeInBits() == TargetSize)
        return OrigElt;
    }

    unsigned GCD = greatestCommonDivisor(OrigSize, TargetSize);
    if (GCD == OrigElt.getSizeInBits())
      return OrigElt;

    // If we can't produce the original element type, we have to use a smaller
    // scalar.
    if (GCD < OrigElt.getSizeInBits())
      return LLT::scalar(GCD);
    return LLT::vector(GCD / OrigElt.getSizeInBits(), OrigElt);
  }

  if (TargetTy.isVector()) {
    // Try to preserve the original element type.
    LLT TargetElt = TargetTy.getElementType();
    if (TargetElt.getSizeInBits() == OrigSize)
      return OrigTy;
  }

  unsigned GCD = greatestCommonDivisor(OrigSize, TargetSize);
  return LLT::scalar(GCD);
}

Optional<int> llvm::getSplatIndex(MachineInstr &MI) {
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
    return None;

  return SplatValue;
}

static bool isBuildVectorOp(unsigned Opcode) {
  return Opcode == TargetOpcode::G_BUILD_VECTOR ||
         Opcode == TargetOpcode::G_BUILD_VECTOR_TRUNC;
}

// TODO: Handle mixed undef elements.
static bool isBuildVectorConstantSplat(const MachineInstr &MI,
                                       const MachineRegisterInfo &MRI,
                                       int64_t SplatValue) {
  if (!isBuildVectorOp(MI.getOpcode()))
    return false;

  const unsigned NumOps = MI.getNumOperands();
  for (unsigned I = 1; I != NumOps; ++I) {
    Register Element = MI.getOperand(I).getReg();
    if (!mi_match(Element, MRI, m_SpecificICst(SplatValue)))
      return false;
  }

  return true;
}

Optional<int64_t>
llvm::getBuildVectorConstantSplat(const MachineInstr &MI,
                                  const MachineRegisterInfo &MRI) {
  if (!isBuildVectorOp(MI.getOpcode()))
    return None;

  const unsigned NumOps = MI.getNumOperands();
  Optional<int64_t> Scalar;
  for (unsigned I = 1; I != NumOps; ++I) {
    Register Element = MI.getOperand(I).getReg();
    int64_t ElementValue;
    if (!mi_match(Element, MRI, m_ICst(ElementValue)))
      return None;
    if (!Scalar)
      Scalar = ElementValue;
    else if (*Scalar != ElementValue)
      return None;
  }

  return Scalar;
}

bool llvm::isBuildVectorAllZeros(const MachineInstr &MI,
                                 const MachineRegisterInfo &MRI) {
  return isBuildVectorConstantSplat(MI, MRI, 0);
}

bool llvm::isBuildVectorAllOnes(const MachineInstr &MI,
                                const MachineRegisterInfo &MRI) {
  return isBuildVectorConstantSplat(MI, MRI, -1);
}

Optional<RegOrConstant> llvm::getVectorSplat(const MachineInstr &MI,
                                             const MachineRegisterInfo &MRI) {
  unsigned Opc = MI.getOpcode();
  if (!isBuildVectorOp(Opc))
    return None;
  if (auto Splat = getBuildVectorConstantSplat(MI, MRI))
    return RegOrConstant(*Splat);
  auto Reg = MI.getOperand(1).getReg();
  if (any_of(make_range(MI.operands_begin() + 2, MI.operands_end()),
             [&Reg](const MachineOperand &Op) { return Op.getReg() != Reg; }))
    return None;
  return RegOrConstant(Reg);
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
