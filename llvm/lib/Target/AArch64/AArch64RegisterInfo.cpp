//===- AArch64RegisterInfo.cpp - AArch64 Register Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AArch64 implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#include "AArch64RegisterInfo.h"
#include "AArch64FrameLowering.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "AArch64GenRegisterInfo.inc"

AArch64RegisterInfo::AArch64RegisterInfo(const Triple &TT)
    : AArch64GenRegisterInfo(AArch64::LR), TT(TT) {}

const MCPhysReg *
AArch64RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  assert(MF && "Invalid MachineFunction pointer.");
  if (MF->getFunction()->getCallingConv() == CallingConv::GHC)
    // GHC set of callee saved regs is empty as all those regs are
    // used for passing STG regs around
    return CSR_AArch64_NoRegs_SaveList;
  if (MF->getFunction()->getCallingConv() == CallingConv::AnyReg)
    return CSR_AArch64_AllRegs_SaveList;
  else
    return CSR_AArch64_AAPCS_SaveList;
}

const uint32_t *
AArch64RegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                          CallingConv::ID CC) const {
  if (CC == CallingConv::GHC)
    // This is academic becase all GHC calls are (supposed to be) tail calls
    return CSR_AArch64_NoRegs_RegMask;
  if (CC == CallingConv::AnyReg)
    return CSR_AArch64_AllRegs_RegMask;
  else
    return CSR_AArch64_AAPCS_RegMask;
}

const uint32_t *AArch64RegisterInfo::getTLSCallPreservedMask() const {
  if (TT.isOSDarwin())
    return CSR_AArch64_TLS_Darwin_RegMask;

  assert(TT.isOSBinFormatELF() && "only expect Darwin or ELF TLS");
  return CSR_AArch64_TLS_ELF_RegMask;
}

const uint32_t *
AArch64RegisterInfo::getThisReturnPreservedMask(const MachineFunction &MF,
                                                CallingConv::ID CC) const {
  // This should return a register mask that is the same as that returned by
  // getCallPreservedMask but that additionally preserves the register used for
  // the first i64 argument (which must also be the register used to return a
  // single i64 return value)
  //
  // In case that the calling convention does not use the same register for
  // both, the function should return NULL (does not currently apply)
  assert(CC != CallingConv::GHC && "should not be GHC calling convention.");
  return CSR_AArch64_AAPCS_ThisReturn_RegMask;
}

BitVector
AArch64RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  const AArch64FrameLowering *TFI = getFrameLowering(MF);

  // FIXME: avoid re-calculating this every time.
  BitVector Reserved(getNumRegs());
  Reserved.set(AArch64::SP);
  Reserved.set(AArch64::XZR);
  Reserved.set(AArch64::WSP);
  Reserved.set(AArch64::WZR);

  if (TFI->hasFP(MF) || TT.isOSDarwin()) {
    Reserved.set(AArch64::FP);
    Reserved.set(AArch64::W29);
  }

  if (MF.getSubtarget<AArch64Subtarget>().isX18Reserved()) {
    Reserved.set(AArch64::X18); // Platform register
    Reserved.set(AArch64::W18);
  }

  if (hasBasePointer(MF)) {
    Reserved.set(AArch64::X19);
    Reserved.set(AArch64::W19);
  }

  return Reserved;
}

bool AArch64RegisterInfo::isReservedReg(const MachineFunction &MF,
                                      unsigned Reg) const {
  const AArch64FrameLowering *TFI = getFrameLowering(MF);

  switch (Reg) {
  default:
    break;
  case AArch64::SP:
  case AArch64::XZR:
  case AArch64::WSP:
  case AArch64::WZR:
    return true;
  case AArch64::X18:
  case AArch64::W18:
    return MF.getSubtarget<AArch64Subtarget>().isX18Reserved();
  case AArch64::FP:
  case AArch64::W29:
    return TFI->hasFP(MF) || TT.isOSDarwin();
  case AArch64::W19:
  case AArch64::X19:
    return hasBasePointer(MF);
  }

  return false;
}

const TargetRegisterClass *
AArch64RegisterInfo::getPointerRegClass(const MachineFunction &MF,
                                      unsigned Kind) const {
  return &AArch64::GPR64RegClass;
}

const TargetRegisterClass *
AArch64RegisterInfo::getCrossCopyRegClass(const TargetRegisterClass *RC) const {
  if (RC == &AArch64::CCRRegClass)
    return &AArch64::GPR64RegClass; // Only MSR & MRS copy NZCV.
  return RC;
}

unsigned AArch64RegisterInfo::getBaseRegister() const { return AArch64::X19; }

bool AArch64RegisterInfo::hasBasePointer(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();

  // In the presence of variable sized objects, if the fixed stack size is
  // large enough that referencing from the FP won't result in things being
  // in range relatively often, we can use a base pointer to allow access
  // from the other direction like the SP normally works.
  // Furthermore, if both variable sized objects are present, and the
  // stack needs to be dynamically re-aligned, the base pointer is the only
  // reliable way to reference the locals.
  if (MFI->hasVarSizedObjects()) {
    if (needsStackRealignment(MF))
      return true;
    // Conservatively estimate whether the negative offset from the frame
    // pointer will be sufficient to reach. If a function has a smallish
    // frame, it's less likely to have lots of spills and callee saved
    // space, so it's all more likely to be within range of the frame pointer.
    // If it's wrong, we'll materialize the constant and still get to the
    // object; it's just suboptimal. Negative offsets use the unscaled
    // load/store instructions, which have a 9-bit signed immediate.
    if (MFI->getLocalFrameSize() < 256)
      return false;
    return true;
  }

  return false;
}

unsigned
AArch64RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const AArch64FrameLowering *TFI = getFrameLowering(MF);
  return TFI->hasFP(MF) ? AArch64::FP : AArch64::SP;
}

bool AArch64RegisterInfo::requiresRegisterScavenging(
    const MachineFunction &MF) const {
  return true;
}

bool AArch64RegisterInfo::requiresVirtualBaseRegisters(
    const MachineFunction &MF) const {
  return true;
}

bool
AArch64RegisterInfo::useFPForScavengingIndex(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  // AArch64FrameLowering::resolveFrameIndexReference() can always fall back
  // to the stack pointer, so only put the emergency spill slot next to the
  // FP when there's no better way to access it (SP or base pointer).
  return MFI->hasVarSizedObjects() && !hasBasePointer(MF);
}

bool AArch64RegisterInfo::requiresFrameIndexScavenging(
    const MachineFunction &MF) const {
  return true;
}

bool
AArch64RegisterInfo::cannotEliminateFrame(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  // Only consider eliminating leaf frames.
  if (MFI->hasCalls() || (MF.getTarget().Options.DisableFramePointerElim(MF) &&
                          MFI->adjustsStack()))
    return true;
  return MFI->hasVarSizedObjects() || MFI->isFrameAddressTaken();
}

/// needsFrameBaseReg - Returns true if the instruction's frame index
/// reference would be better served by a base register other than FP
/// or SP. Used by LocalStackFrameAllocation to determine which frame index
/// references it should create new base registers for.
bool AArch64RegisterInfo::needsFrameBaseReg(MachineInstr *MI,
                                            int64_t Offset) const {
  for (unsigned i = 0; !MI->getOperand(i).isFI(); ++i)
    assert(i < MI->getNumOperands() &&
           "Instr doesn't have FrameIndex operand!");

  // It's the load/store FI references that cause issues, as it can be difficult
  // to materialize the offset if it won't fit in the literal field. Estimate
  // based on the size of the local frame and some conservative assumptions
  // about the rest of the stack frame (note, this is pre-regalloc, so
  // we don't know everything for certain yet) whether this offset is likely
  // to be out of range of the immediate. Return true if so.

  // We only generate virtual base registers for loads and stores, so
  // return false for everything else.
  if (!MI->mayLoad() && !MI->mayStore())
    return false;

  // Without a virtual base register, if the function has variable sized
  // objects, all fixed-size local references will be via the frame pointer,
  // Approximate the offset and see if it's legal for the instruction.
  // Note that the incoming offset is based on the SP value at function entry,
  // so it'll be negative.
  MachineFunction &MF = *MI->getParent()->getParent();
  const AArch64FrameLowering *TFI = getFrameLowering(MF);
  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Estimate an offset from the frame pointer.
  // Conservatively assume all GPR callee-saved registers get pushed.
  // FP, LR, X19-X28, D8-D15. 64-bits each.
  int64_t FPOffset = Offset - 16 * 20;
  // Estimate an offset from the stack pointer.
  // The incoming offset is relating to the SP at the start of the function,
  // but when we access the local it'll be relative to the SP after local
  // allocation, so adjust our SP-relative offset by that allocation size.
  Offset += MFI->getLocalFrameSize();
  // Assume that we'll have at least some spill slots allocated.
  // FIXME: This is a total SWAG number. We should run some statistics
  //        and pick a real one.
  Offset += 128; // 128 bytes of spill slots

  // If there is a frame pointer, try using it.
  // The FP is only available if there is no dynamic realignment. We
  // don't know for sure yet whether we'll need that, so we guess based
  // on whether there are any local variables that would trigger it.
  if (TFI->hasFP(MF) && isFrameOffsetLegal(MI, AArch64::FP, FPOffset))
    return false;

  // If we can reference via the stack pointer or base pointer, try that.
  // FIXME: This (and the code that resolves the references) can be improved
  //        to only disallow SP relative references in the live range of
  //        the VLA(s). In practice, it's unclear how much difference that
  //        would make, but it may be worth doing.
  if (isFrameOffsetLegal(MI, AArch64::SP, Offset))
    return false;

  // The offset likely isn't legal; we want to allocate a virtual base register.
  return true;
}

bool AArch64RegisterInfo::isFrameOffsetLegal(const MachineInstr *MI,
                                             unsigned BaseReg,
                                             int64_t Offset) const {
  assert(Offset <= INT_MAX && "Offset too big to fit in int.");
  assert(MI && "Unable to get the legal offset for nil instruction.");
  int SaveOffset = Offset;
  return isAArch64FrameOffsetLegal(*MI, SaveOffset) & AArch64FrameOffsetIsLegal;
}

/// Insert defining instruction(s) for BaseReg to be a pointer to FrameIdx
/// at the beginning of the basic block.
void AArch64RegisterInfo::materializeFrameBaseRegister(MachineBasicBlock *MBB,
                                                       unsigned BaseReg,
                                                       int FrameIdx,
                                                       int64_t Offset) const {
  MachineBasicBlock::iterator Ins = MBB->begin();
  DebugLoc DL; // Defaults to "unknown"
  if (Ins != MBB->end())
    DL = Ins->getDebugLoc();
  const MachineFunction &MF = *MBB->getParent();
  const AArch64InstrInfo *TII =
      MF.getSubtarget<AArch64Subtarget>().getInstrInfo();
  const MCInstrDesc &MCID = TII->get(AArch64::ADDXri);
  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
  MRI.constrainRegClass(BaseReg, TII->getRegClass(MCID, 0, this, MF));
  unsigned Shifter = AArch64_AM::getShifterImm(AArch64_AM::LSL, 0);

  BuildMI(*MBB, Ins, DL, MCID, BaseReg)
      .addFrameIndex(FrameIdx)
      .addImm(Offset)
      .addImm(Shifter);
}

void AArch64RegisterInfo::resolveFrameIndex(MachineInstr &MI, unsigned BaseReg,
                                            int64_t Offset) const {
  int Off = Offset; // ARM doesn't need the general 64-bit offsets
  unsigned i = 0;

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }
  const MachineFunction *MF = MI.getParent()->getParent();
  const AArch64InstrInfo *TII =
      MF->getSubtarget<AArch64Subtarget>().getInstrInfo();
  bool Done = rewriteAArch64FrameIndex(MI, i, BaseReg, Off, TII);
  assert(Done && "Unable to resolve frame index!");
  (void)Done;
}

void AArch64RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                              int SPAdj, unsigned FIOperandNum,
                                              RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected");

  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  const AArch64InstrInfo *TII =
      MF.getSubtarget<AArch64Subtarget>().getInstrInfo();
  const AArch64FrameLowering *TFI = getFrameLowering(MF);

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  unsigned FrameReg;
  int Offset;

  // Special handling of dbg_value, stackmap and patchpoint instructions.
  if (MI.isDebugValue() || MI.getOpcode() == TargetOpcode::STACKMAP ||
      MI.getOpcode() == TargetOpcode::PATCHPOINT) {
    Offset = TFI->resolveFrameIndexReference(MF, FrameIndex, FrameReg,
                                             /*PreferFP=*/true);
    Offset += MI.getOperand(FIOperandNum + 1).getImm();
    MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, false /*isDef*/);
    MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
    return;
  }

  // Modify MI as necessary to handle as much of 'Offset' as possible
  Offset = TFI->resolveFrameIndexReference(MF, FrameIndex, FrameReg);
  if (rewriteAArch64FrameIndex(MI, FIOperandNum, FrameReg, Offset, TII))
    return;

  assert((!RS || !RS->isScavengingFrameIndex(FrameIndex)) &&
         "Emergency spill slot is out of reach");

  // If we get here, the immediate doesn't fit into the instruction.  We folded
  // as much as possible above.  Handle the rest, providing a register that is
  // SP+LargeImm.
  unsigned ScratchReg =
      MF.getRegInfo().createVirtualRegister(&AArch64::GPR64RegClass);
  emitFrameOffset(MBB, II, MI.getDebugLoc(), ScratchReg, FrameReg, Offset, TII);
  MI.getOperand(FIOperandNum).ChangeToRegister(ScratchReg, false, false, true);
}

namespace llvm {

unsigned AArch64RegisterInfo::getRegPressureLimit(const TargetRegisterClass *RC,
                                                  MachineFunction &MF) const {
  const AArch64FrameLowering *TFI = getFrameLowering(MF);

  switch (RC->getID()) {
  default:
    return 0;
  case AArch64::GPR32RegClassID:
  case AArch64::GPR32spRegClassID:
  case AArch64::GPR32allRegClassID:
  case AArch64::GPR64spRegClassID:
  case AArch64::GPR64allRegClassID:
  case AArch64::GPR64RegClassID:
  case AArch64::GPR32commonRegClassID:
  case AArch64::GPR64commonRegClassID:
    return 32 - 1                                   // XZR/SP
              - (TFI->hasFP(MF) || TT.isOSDarwin()) // FP
              - MF.getSubtarget<AArch64Subtarget>()
                    .isX18Reserved() // X18 reserved as platform register
              - hasBasePointer(MF);  // X19
  case AArch64::FPR8RegClassID:
  case AArch64::FPR16RegClassID:
  case AArch64::FPR32RegClassID:
  case AArch64::FPR64RegClassID:
  case AArch64::FPR128RegClassID:
    return 32;

  case AArch64::DDRegClassID:
  case AArch64::DDDRegClassID:
  case AArch64::DDDDRegClassID:
  case AArch64::QQRegClassID:
  case AArch64::QQQRegClassID:
  case AArch64::QQQQRegClassID:
    return 32;

  case AArch64::FPR128_loRegClassID:
    return 16;
  }
}

} // namespace llvm
