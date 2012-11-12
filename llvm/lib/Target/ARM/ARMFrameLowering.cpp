//===-- ARMFrameLowering.cpp - ARM Frame Information ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "ARMFrameLowering.h"
#include "ARMBaseInstrInfo.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "llvm/CallingConv.h"
#include "llvm/Function.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<bool>
SpillAlignedNEONRegs("align-neon-spills", cl::Hidden, cl::init(true),
                     cl::desc("Align ARM NEON spills in prolog and epilog"));

static MachineBasicBlock::iterator
skipAlignedDPRCS2Spills(MachineBasicBlock::iterator MI,
                        unsigned NumAlignedDPRCS2Regs);

/// hasFP - Return true if the specified function should have a dedicated frame
/// pointer register.  This is true if the function has variable sized allocas
/// or if frame pointer elimination is disabled.
bool ARMFrameLowering::hasFP(const MachineFunction &MF) const {
  const TargetRegisterInfo *RegInfo = MF.getTarget().getRegisterInfo();

  // iOS requires FP not to be clobbered for backtracing purpose.
  if (STI.isTargetIOS())
    return true;

  const MachineFrameInfo *MFI = MF.getFrameInfo();
  // Always eliminate non-leaf frame pointers.
  return ((MF.getTarget().Options.DisableFramePointerElim(MF) &&
           MFI->hasCalls()) ||
          RegInfo->needsStackRealignment(MF) ||
          MFI->hasVarSizedObjects() ||
          MFI->isFrameAddressTaken());
}

/// hasReservedCallFrame - Under normal circumstances, when a frame pointer is
/// not required, we reserve argument space for call sites in the function
/// immediately on entry to the current function.  This eliminates the need for
/// add/sub sp brackets around call sites.  Returns true if the call frame is
/// included as part of the stack frame.
bool ARMFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  const MachineFrameInfo *FFI = MF.getFrameInfo();
  unsigned CFSize = FFI->getMaxCallFrameSize();
  // It's not always a good idea to include the call frame as part of the
  // stack frame. ARM (especially Thumb) has small immediate offset to
  // address the stack frame. So a large call frame can cause poor codegen
  // and may even makes it impossible to scavenge a register.
  if (CFSize >= ((1 << 12) - 1) / 2)  // Half of imm12
    return false;

  return !MF.getFrameInfo()->hasVarSizedObjects();
}

/// canSimplifyCallFramePseudos - If there is a reserved call frame, the
/// call frame pseudos can be simplified.  Unlike most targets, having a FP
/// is not sufficient here since we still may reference some objects via SP
/// even when FP is available in Thumb2 mode.
bool
ARMFrameLowering::canSimplifyCallFramePseudos(const MachineFunction &MF) const {
  return hasReservedCallFrame(MF) || MF.getFrameInfo()->hasVarSizedObjects();
}

static bool isCalleeSavedRegister(unsigned Reg, const uint16_t *CSRegs) {
  for (unsigned i = 0; CSRegs[i]; ++i)
    if (Reg == CSRegs[i])
      return true;
  return false;
}

static bool isCSRestore(MachineInstr *MI,
                        const ARMBaseInstrInfo &TII,
                        const uint16_t *CSRegs) {
  // Integer spill area is handled with "pop".
  if (MI->getOpcode() == ARM::LDMIA_RET ||
      MI->getOpcode() == ARM::t2LDMIA_RET ||
      MI->getOpcode() == ARM::LDMIA_UPD ||
      MI->getOpcode() == ARM::t2LDMIA_UPD ||
      MI->getOpcode() == ARM::VLDMDIA_UPD) {
    // The first two operands are predicates. The last two are
    // imp-def and imp-use of SP. Check everything in between.
    for (int i = 5, e = MI->getNumOperands(); i != e; ++i)
      if (!isCalleeSavedRegister(MI->getOperand(i).getReg(), CSRegs))
        return false;
    return true;
  }
  if ((MI->getOpcode() == ARM::LDR_POST_IMM ||
       MI->getOpcode() == ARM::LDR_POST_REG ||
       MI->getOpcode() == ARM::t2LDR_POST) &&
      isCalleeSavedRegister(MI->getOperand(0).getReg(), CSRegs) &&
      MI->getOperand(1).getReg() == ARM::SP)
    return true;

  return false;
}

static void
emitSPUpdate(bool isARM,
             MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
             DebugLoc dl, const ARMBaseInstrInfo &TII,
             int NumBytes, unsigned MIFlags = MachineInstr::NoFlags) {
  if (isARM)
    emitARMRegPlusImmediate(MBB, MBBI, dl, ARM::SP, ARM::SP, NumBytes,
                            ARMCC::AL, 0, TII, MIFlags);
  else
    emitT2RegPlusImmediate(MBB, MBBI, dl, ARM::SP, ARM::SP, NumBytes,
                           ARMCC::AL, 0, TII, MIFlags);
}

void ARMFrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo  *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  const ARMBaseRegisterInfo *RegInfo =
    static_cast<const ARMBaseRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const ARMBaseInstrInfo &TII =
    *static_cast<const ARMBaseInstrInfo*>(MF.getTarget().getInstrInfo());
  assert(!AFI->isThumb1OnlyFunction() &&
         "This emitPrologue does not support Thumb1!");
  bool isARM = !AFI->isThumbFunction();
  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  unsigned NumBytes = MFI->getStackSize();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  unsigned FramePtr = RegInfo->getFrameRegister(MF);

  // Determine the sizes of each callee-save spill areas and record which frame
  // belongs to which callee-save spill areas.
  unsigned GPRCS1Size = 0, GPRCS2Size = 0, DPRCSSize = 0;
  int FramePtrSpillFI = 0;
  int D8SpillFI = 0;

  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction()->getCallingConv() == CallingConv::GHC)
    return;

  // Allocate the vararg register save area. This is not counted in NumBytes.
  if (VARegSaveSize)
    emitSPUpdate(isARM, MBB, MBBI, dl, TII, -VARegSaveSize,
                 MachineInstr::FrameSetup);

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(isARM, MBB, MBBI, dl, TII, -NumBytes,
                   MachineInstr::FrameSetup);
    return;
  }

  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    int FI = CSI[i].getFrameIdx();
    switch (Reg) {
    case ARM::R4:
    case ARM::R5:
    case ARM::R6:
    case ARM::R7:
    case ARM::LR:
      if (Reg == FramePtr)
        FramePtrSpillFI = FI;
      AFI->addGPRCalleeSavedArea1Frame(FI);
      GPRCS1Size += 4;
      break;
    case ARM::R8:
    case ARM::R9:
    case ARM::R10:
    case ARM::R11:
      if (Reg == FramePtr)
        FramePtrSpillFI = FI;
      if (STI.isTargetIOS()) {
        AFI->addGPRCalleeSavedArea2Frame(FI);
        GPRCS2Size += 4;
      } else {
        AFI->addGPRCalleeSavedArea1Frame(FI);
        GPRCS1Size += 4;
      }
      break;
    default:
      // This is a DPR. Exclude the aligned DPRCS2 spills.
      if (Reg == ARM::D8)
        D8SpillFI = FI;
      if (Reg < ARM::D8 || Reg >= ARM::D8 + AFI->getNumAlignedDPRCS2Regs()) {
        AFI->addDPRCalleeSavedAreaFrame(FI);
        DPRCSSize += 8;
      }
    }
  }

  // Move past area 1.
  if (GPRCS1Size > 0) MBBI++;

  // Set FP to point to the stack slot that contains the previous FP.
  // For iOS, FP is R7, which has now been stored in spill area 1.
  // Otherwise, if this is not iOS, all the callee-saved registers go
  // into spill area 1, including the FP in R11.  In either case, it is
  // now safe to emit this assignment.
  bool HasFP = hasFP(MF);
  if (HasFP) {
    unsigned ADDriOpc = !AFI->isThumbFunction() ? ARM::ADDri : ARM::t2ADDri;
    MachineInstrBuilder MIB =
      BuildMI(MBB, MBBI, dl, TII.get(ADDriOpc), FramePtr)
      .addFrameIndex(FramePtrSpillFI).addImm(0)
      .setMIFlag(MachineInstr::FrameSetup);
    AddDefaultCC(AddDefaultPred(MIB));
  }

  // Move past area 2.
  if (GPRCS2Size > 0) MBBI++;

  // Determine starting offsets of spill areas.
  unsigned DPRCSOffset  = NumBytes - (GPRCS1Size + GPRCS2Size + DPRCSSize);
  unsigned GPRCS2Offset = DPRCSOffset + DPRCSSize;
  unsigned GPRCS1Offset = GPRCS2Offset + GPRCS2Size;
  if (HasFP)
    AFI->setFramePtrSpillOffset(MFI->getObjectOffset(FramePtrSpillFI) +
                                NumBytes);
  AFI->setGPRCalleeSavedArea1Offset(GPRCS1Offset);
  AFI->setGPRCalleeSavedArea2Offset(GPRCS2Offset);
  AFI->setDPRCalleeSavedAreaOffset(DPRCSOffset);

  // Move past area 3.
  if (DPRCSSize > 0) {
    MBBI++;
    // Since vpush register list cannot have gaps, there may be multiple vpush
    // instructions in the prologue.
    while (MBBI->getOpcode() == ARM::VSTMDDB_UPD)
      MBBI++;
  }

  // Move past the aligned DPRCS2 area.
  if (AFI->getNumAlignedDPRCS2Regs() > 0) {
    MBBI = skipAlignedDPRCS2Spills(MBBI, AFI->getNumAlignedDPRCS2Regs());
    // The code inserted by emitAlignedDPRCS2Spills realigns the stack, and
    // leaves the stack pointer pointing to the DPRCS2 area.
    //
    // Adjust NumBytes to represent the stack slots below the DPRCS2 area.
    NumBytes += MFI->getObjectOffset(D8SpillFI);
  } else
    NumBytes = DPRCSOffset;

  if (NumBytes) {
    // Adjust SP after all the callee-save spills.
    emitSPUpdate(isARM, MBB, MBBI, dl, TII, -NumBytes,
                 MachineInstr::FrameSetup);
    if (HasFP && isARM)
      // Restore from fp only in ARM mode: e.g. sub sp, r7, #24
      // Note it's not safe to do this in Thumb2 mode because it would have
      // taken two instructions:
      // mov sp, r7
      // sub sp, #24
      // If an interrupt is taken between the two instructions, then sp is in
      // an inconsistent state (pointing to the middle of callee-saved area).
      // The interrupt handler can end up clobbering the registers.
      AFI->setShouldRestoreSPFromFP(true);
  }

  if (STI.isTargetELF() && hasFP(MF))
    MFI->setOffsetAdjustment(MFI->getOffsetAdjustment() -
                             AFI->getFramePtrSpillOffset());

  AFI->setGPRCalleeSavedArea1Size(GPRCS1Size);
  AFI->setGPRCalleeSavedArea2Size(GPRCS2Size);
  AFI->setDPRCalleeSavedAreaSize(DPRCSSize);

  // If we need dynamic stack realignment, do it here. Be paranoid and make
  // sure if we also have VLAs, we have a base pointer for frame access.
  // If aligned NEON registers were spilled, the stack has already been
  // realigned.
  if (!AFI->getNumAlignedDPRCS2Regs() && RegInfo->needsStackRealignment(MF)) {
    unsigned MaxAlign = MFI->getMaxAlignment();
    assert (!AFI->isThumb1OnlyFunction());
    if (!AFI->isThumbFunction()) {
      // Emit bic sp, sp, MaxAlign
      AddDefaultCC(AddDefaultPred(BuildMI(MBB, MBBI, dl,
                                          TII.get(ARM::BICri), ARM::SP)
                                  .addReg(ARM::SP, RegState::Kill)
                                  .addImm(MaxAlign-1)));
    } else {
      // We cannot use sp as source/dest register here, thus we're emitting the
      // following sequence:
      // mov r4, sp
      // bic r4, r4, MaxAlign
      // mov sp, r4
      // FIXME: It will be better just to find spare register here.
      AddDefaultPred(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVr), ARM::R4)
        .addReg(ARM::SP, RegState::Kill));
      AddDefaultCC(AddDefaultPred(BuildMI(MBB, MBBI, dl,
                                          TII.get(ARM::t2BICri), ARM::R4)
                                  .addReg(ARM::R4, RegState::Kill)
                                  .addImm(MaxAlign-1)));
      AddDefaultPred(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVr), ARM::SP)
        .addReg(ARM::R4, RegState::Kill));
    }

    AFI->setShouldRestoreSPFromFP(true);
  }

  // If we need a base pointer, set it up here. It's whatever the value
  // of the stack pointer is at this point. Any variable size objects
  // will be allocated after this, so we can still use the base pointer
  // to reference locals.
  // FIXME: Clarify FrameSetup flags here.
  if (RegInfo->hasBasePointer(MF)) {
    if (isARM)
      BuildMI(MBB, MBBI, dl,
              TII.get(ARM::MOVr), RegInfo->getBaseRegister())
        .addReg(ARM::SP)
        .addImm((unsigned)ARMCC::AL).addReg(0).addReg(0);
    else
      AddDefaultPred(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVr),
                             RegInfo->getBaseRegister())
        .addReg(ARM::SP));
  }

  // If the frame has variable sized objects then the epilogue must restore
  // the sp from fp. We can assume there's an FP here since hasFP already
  // checks for hasVarSizedObjects.
  if (MFI->hasVarSizedObjects())
    AFI->setShouldRestoreSPFromFP(true);
}

void ARMFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  assert(MBBI->isReturn() && "Can only insert epilog into returning blocks");
  unsigned RetOpcode = MBBI->getOpcode();
  DebugLoc dl = MBBI->getDebugLoc();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  const TargetRegisterInfo *RegInfo = MF.getTarget().getRegisterInfo();
  const ARMBaseInstrInfo &TII =
    *static_cast<const ARMBaseInstrInfo*>(MF.getTarget().getInstrInfo());
  assert(!AFI->isThumb1OnlyFunction() &&
         "This emitEpilogue does not support Thumb1!");
  bool isARM = !AFI->isThumbFunction();

  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  int NumBytes = (int)MFI->getStackSize();
  unsigned FramePtr = RegInfo->getFrameRegister(MF);

  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction()->getCallingConv() == CallingConv::GHC)
    return;

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(isARM, MBB, MBBI, dl, TII, NumBytes);
  } else {
    // Unwind MBBI to point to first LDR / VLDRD.
    const uint16_t *CSRegs = RegInfo->getCalleeSavedRegs();
    if (MBBI != MBB.begin()) {
      do
        --MBBI;
      while (MBBI != MBB.begin() && isCSRestore(MBBI, TII, CSRegs));
      if (!isCSRestore(MBBI, TII, CSRegs))
        ++MBBI;
    }

    // Move SP to start of FP callee save spill area.
    NumBytes -= (AFI->getGPRCalleeSavedArea1Size() +
                 AFI->getGPRCalleeSavedArea2Size() +
                 AFI->getDPRCalleeSavedAreaSize());

    // Reset SP based on frame pointer only if the stack frame extends beyond
    // frame pointer stack slot or target is ELF and the function has FP.
    if (AFI->shouldRestoreSPFromFP()) {
      NumBytes = AFI->getFramePtrSpillOffset() - NumBytes;
      if (NumBytes) {
        if (isARM)
          emitARMRegPlusImmediate(MBB, MBBI, dl, ARM::SP, FramePtr, -NumBytes,
                                  ARMCC::AL, 0, TII);
        else {
          // It's not possible to restore SP from FP in a single instruction.
          // For iOS, this looks like:
          // mov sp, r7
          // sub sp, #24
          // This is bad, if an interrupt is taken after the mov, sp is in an
          // inconsistent state.
          // Use the first callee-saved register as a scratch register.
          assert(MF.getRegInfo().isPhysRegUsed(ARM::R4) &&
                 "No scratch register to restore SP from FP!");
          emitT2RegPlusImmediate(MBB, MBBI, dl, ARM::R4, FramePtr, -NumBytes,
                                 ARMCC::AL, 0, TII);
          AddDefaultPred(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVr),
                                 ARM::SP)
            .addReg(ARM::R4));
        }
      } else {
        // Thumb2 or ARM.
        if (isARM)
          BuildMI(MBB, MBBI, dl, TII.get(ARM::MOVr), ARM::SP)
            .addReg(FramePtr).addImm((unsigned)ARMCC::AL).addReg(0).addReg(0);
        else
          AddDefaultPred(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVr),
                                 ARM::SP)
            .addReg(FramePtr));
      }
    } else if (NumBytes)
      emitSPUpdate(isARM, MBB, MBBI, dl, TII, NumBytes);

    // Increment past our save areas.
    if (AFI->getDPRCalleeSavedAreaSize()) {
      MBBI++;
      // Since vpop register list cannot have gaps, there may be multiple vpop
      // instructions in the epilogue.
      while (MBBI->getOpcode() == ARM::VLDMDIA_UPD)
        MBBI++;
    }
    if (AFI->getGPRCalleeSavedArea2Size()) MBBI++;
    if (AFI->getGPRCalleeSavedArea1Size()) MBBI++;
  }

  if (RetOpcode == ARM::TCRETURNdi || RetOpcode == ARM::TCRETURNri) {
    // Tail call return: adjust the stack pointer and jump to callee.
    MBBI = MBB.getLastNonDebugInstr();
    MachineOperand &JumpTarget = MBBI->getOperand(0);

    // Jump to label or value in register.
    if (RetOpcode == ARM::TCRETURNdi) {
      unsigned TCOpcode = STI.isThumb() ?
               (STI.isTargetIOS() ? ARM::tTAILJMPd : ARM::tTAILJMPdND) :
               ARM::TAILJMPd;
      MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII.get(TCOpcode));
      if (JumpTarget.isGlobal())
        MIB.addGlobalAddress(JumpTarget.getGlobal(), JumpTarget.getOffset(),
                             JumpTarget.getTargetFlags());
      else {
        assert(JumpTarget.isSymbol());
        MIB.addExternalSymbol(JumpTarget.getSymbolName(),
                              JumpTarget.getTargetFlags());
      }

      // Add the default predicate in Thumb mode.
      if (STI.isThumb()) MIB.addImm(ARMCC::AL).addReg(0);
    } else if (RetOpcode == ARM::TCRETURNri) {
      BuildMI(MBB, MBBI, dl,
              TII.get(STI.isThumb() ? ARM::tTAILJMPr : ARM::TAILJMPr)).
        addReg(JumpTarget.getReg(), RegState::Kill);
    }

    MachineInstr *NewMI = prior(MBBI);
    for (unsigned i = 1, e = MBBI->getNumOperands(); i != e; ++i)
      NewMI->addOperand(MBBI->getOperand(i));

    // Delete the pseudo instruction TCRETURN.
    MBB.erase(MBBI);
    MBBI = NewMI;
  }

  if (VARegSaveSize)
    emitSPUpdate(isARM, MBB, MBBI, dl, TII, VARegSaveSize);
}

/// getFrameIndexReference - Provide a base+offset reference to an FI slot for
/// debug info.  It's the same as what we use for resolving the code-gen
/// references for now.  FIXME: This can go wrong when references are
/// SP-relative and simple call frames aren't used.
int
ARMFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                         unsigned &FrameReg) const {
  return ResolveFrameIndexReference(MF, FI, FrameReg, 0);
}

int
ARMFrameLowering::ResolveFrameIndexReference(const MachineFunction &MF,
                                             int FI, unsigned &FrameReg,
                                             int SPAdj) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const ARMBaseRegisterInfo *RegInfo =
    static_cast<const ARMBaseRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  int Offset = MFI->getObjectOffset(FI) + MFI->getStackSize();
  int FPOffset = Offset - AFI->getFramePtrSpillOffset();
  bool isFixed = MFI->isFixedObjectIndex(FI);

  FrameReg = ARM::SP;
  Offset += SPAdj;
  if (AFI->isGPRCalleeSavedArea1Frame(FI))
    return Offset - AFI->getGPRCalleeSavedArea1Offset();
  else if (AFI->isGPRCalleeSavedArea2Frame(FI))
    return Offset - AFI->getGPRCalleeSavedArea2Offset();
  else if (AFI->isDPRCalleeSavedAreaFrame(FI))
    return Offset - AFI->getDPRCalleeSavedAreaOffset();

  // SP can move around if there are allocas.  We may also lose track of SP
  // when emergency spilling inside a non-reserved call frame setup.
  bool hasMovingSP = !hasReservedCallFrame(MF);

  // When dynamically realigning the stack, use the frame pointer for
  // parameters, and the stack/base pointer for locals.
  if (RegInfo->needsStackRealignment(MF)) {
    assert (hasFP(MF) && "dynamic stack realignment without a FP!");
    if (isFixed) {
      FrameReg = RegInfo->getFrameRegister(MF);
      Offset = FPOffset;
    } else if (hasMovingSP) {
      assert(RegInfo->hasBasePointer(MF) &&
             "VLAs and dynamic stack alignment, but missing base pointer!");
      FrameReg = RegInfo->getBaseRegister();
    }
    return Offset;
  }

  // If there is a frame pointer, use it when we can.
  if (hasFP(MF) && AFI->hasStackFrame()) {
    // Use frame pointer to reference fixed objects. Use it for locals if
    // there are VLAs (and thus the SP isn't reliable as a base).
    if (isFixed || (hasMovingSP && !RegInfo->hasBasePointer(MF))) {
      FrameReg = RegInfo->getFrameRegister(MF);
      return FPOffset;
    } else if (hasMovingSP) {
      assert(RegInfo->hasBasePointer(MF) && "missing base pointer!");
      if (AFI->isThumb2Function()) {
        // Try to use the frame pointer if we can, else use the base pointer
        // since it's available. This is handy for the emergency spill slot, in
        // particular.
        if (FPOffset >= -255 && FPOffset < 0) {
          FrameReg = RegInfo->getFrameRegister(MF);
          return FPOffset;
        }
      }
    } else if (AFI->isThumb2Function()) {
      // Use  add <rd>, sp, #<imm8>
      //      ldr <rd>, [sp, #<imm8>]
      // if at all possible to save space.
      if (Offset >= 0 && (Offset & 3) == 0 && Offset <= 1020)
        return Offset;
      // In Thumb2 mode, the negative offset is very limited. Try to avoid
      // out of range references. ldr <rt>,[<rn>, #-<imm8>]
      if (FPOffset >= -255 && FPOffset < 0) {
        FrameReg = RegInfo->getFrameRegister(MF);
        return FPOffset;
      }
    } else if (Offset > (FPOffset < 0 ? -FPOffset : FPOffset)) {
      // Otherwise, use SP or FP, whichever is closer to the stack slot.
      FrameReg = RegInfo->getFrameRegister(MF);
      return FPOffset;
    }
  }
  // Use the base pointer if we have one.
  if (RegInfo->hasBasePointer(MF))
    FrameReg = RegInfo->getBaseRegister();
  return Offset;
}

int ARMFrameLowering::getFrameIndexOffset(const MachineFunction &MF,
                                          int FI) const {
  unsigned FrameReg;
  return getFrameIndexReference(MF, FI, FrameReg);
}

void ARMFrameLowering::emitPushInst(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    const std::vector<CalleeSavedInfo> &CSI,
                                    unsigned StmOpc, unsigned StrOpc,
                                    bool NoGap,
                                    bool(*Func)(unsigned, bool),
                                    unsigned NumAlignedDPRCS2Regs,
                                    unsigned MIFlags) const {
  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();

  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  SmallVector<std::pair<unsigned,bool>, 4> Regs;
  unsigned i = CSI.size();
  while (i != 0) {
    unsigned LastReg = 0;
    for (; i != 0; --i) {
      unsigned Reg = CSI[i-1].getReg();
      if (!(Func)(Reg, STI.isTargetIOS())) continue;

      // D-registers in the aligned area DPRCS2 are NOT spilled here.
      if (Reg >= ARM::D8 && Reg < ARM::D8 + NumAlignedDPRCS2Regs)
        continue;

      // Add the callee-saved register as live-in unless it's LR and
      // @llvm.returnaddress is called. If LR is returned for
      // @llvm.returnaddress then it's already added to the function and
      // entry block live-in sets.
      bool isKill = true;
      if (Reg == ARM::LR) {
        if (MF.getFrameInfo()->isReturnAddressTaken() &&
            MF.getRegInfo().isLiveIn(Reg))
          isKill = false;
      }

      if (isKill)
        MBB.addLiveIn(Reg);

      // If NoGap is true, push consecutive registers and then leave the rest
      // for other instructions. e.g.
      // vpush {d8, d10, d11} -> vpush {d8}, vpush {d10, d11}
      if (NoGap && LastReg && LastReg != Reg-1)
        break;
      LastReg = Reg;
      Regs.push_back(std::make_pair(Reg, isKill));
    }

    if (Regs.empty())
      continue;
    if (Regs.size() > 1 || StrOpc== 0) {
      MachineInstrBuilder MIB =
        AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(StmOpc), ARM::SP)
                       .addReg(ARM::SP).setMIFlags(MIFlags));
      for (unsigned i = 0, e = Regs.size(); i < e; ++i)
        MIB.addReg(Regs[i].first, getKillRegState(Regs[i].second));
    } else if (Regs.size() == 1) {
      MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII.get(StrOpc),
                                        ARM::SP)
        .addReg(Regs[0].first, getKillRegState(Regs[0].second))
        .addReg(ARM::SP).setMIFlags(MIFlags)
        .addImm(-4);
      AddDefaultPred(MIB);
    }
    Regs.clear();
  }
}

void ARMFrameLowering::emitPopInst(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   const std::vector<CalleeSavedInfo> &CSI,
                                   unsigned LdmOpc, unsigned LdrOpc,
                                   bool isVarArg, bool NoGap,
                                   bool(*Func)(unsigned, bool),
                                   unsigned NumAlignedDPRCS2Regs) const {
  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  DebugLoc DL = MI->getDebugLoc();
  unsigned RetOpcode = MI->getOpcode();
  bool isTailCall = (RetOpcode == ARM::TCRETURNdi ||
                     RetOpcode == ARM::TCRETURNri);

  SmallVector<unsigned, 4> Regs;
  unsigned i = CSI.size();
  while (i != 0) {
    unsigned LastReg = 0;
    bool DeleteRet = false;
    for (; i != 0; --i) {
      unsigned Reg = CSI[i-1].getReg();
      if (!(Func)(Reg, STI.isTargetIOS())) continue;

      // The aligned reloads from area DPRCS2 are not inserted here.
      if (Reg >= ARM::D8 && Reg < ARM::D8 + NumAlignedDPRCS2Regs)
        continue;

      if (Reg == ARM::LR && !isTailCall && !isVarArg && STI.hasV5TOps()) {
        Reg = ARM::PC;
        LdmOpc = AFI->isThumbFunction() ? ARM::t2LDMIA_RET : ARM::LDMIA_RET;
        // Fold the return instruction into the LDM.
        DeleteRet = true;
      }

      // If NoGap is true, pop consecutive registers and then leave the rest
      // for other instructions. e.g.
      // vpop {d8, d10, d11} -> vpop {d8}, vpop {d10, d11}
      if (NoGap && LastReg && LastReg != Reg-1)
        break;

      LastReg = Reg;
      Regs.push_back(Reg);
    }

    if (Regs.empty())
      continue;
    if (Regs.size() > 1 || LdrOpc == 0) {
      MachineInstrBuilder MIB =
        AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(LdmOpc), ARM::SP)
                       .addReg(ARM::SP));
      for (unsigned i = 0, e = Regs.size(); i < e; ++i)
        MIB.addReg(Regs[i], getDefRegState(true));
      if (DeleteRet) {
        MIB->copyImplicitOps(&*MI);
        MI->eraseFromParent();
      }
      MI = MIB;
    } else if (Regs.size() == 1) {
      // If we adjusted the reg to PC from LR above, switch it back here. We
      // only do that for LDM.
      if (Regs[0] == ARM::PC)
        Regs[0] = ARM::LR;
      MachineInstrBuilder MIB =
        BuildMI(MBB, MI, DL, TII.get(LdrOpc), Regs[0])
          .addReg(ARM::SP, RegState::Define)
          .addReg(ARM::SP);
      // ARM mode needs an extra reg0 here due to addrmode2. Will go away once
      // that refactoring is complete (eventually).
      if (LdrOpc == ARM::LDR_POST_REG || LdrOpc == ARM::LDR_POST_IMM) {
        MIB.addReg(0);
        MIB.addImm(ARM_AM::getAM2Opc(ARM_AM::add, 4, ARM_AM::no_shift));
      } else
        MIB.addImm(4);
      AddDefaultPred(MIB);
    }
    Regs.clear();
  }
}

/// Emit aligned spill instructions for NumAlignedDPRCS2Regs D-registers
/// starting from d8.  Also insert stack realignment code and leave the stack
/// pointer pointing to the d8 spill slot.
static void emitAlignedDPRCS2Spills(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned NumAlignedDPRCS2Regs,
                                    const std::vector<CalleeSavedInfo> &CSI,
                                    const TargetRegisterInfo *TRI) {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  DebugLoc DL = MI->getDebugLoc();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  MachineFrameInfo &MFI = *MF.getFrameInfo();

  // Mark the D-register spill slots as properly aligned.  Since MFI computes
  // stack slot layout backwards, this can actually mean that the d-reg stack
  // slot offsets can be wrong. The offset for d8 will always be correct.
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned DNum = CSI[i].getReg() - ARM::D8;
    if (DNum >= 8)
      continue;
    int FI = CSI[i].getFrameIdx();
    // The even-numbered registers will be 16-byte aligned, the odd-numbered
    // registers will be 8-byte aligned.
    MFI.setObjectAlignment(FI, DNum % 2 ? 8 : 16);

    // The stack slot for D8 needs to be maximally aligned because this is
    // actually the point where we align the stack pointer.  MachineFrameInfo
    // computes all offsets relative to the incoming stack pointer which is a
    // bit weird when realigning the stack.  Any extra padding for this
    // over-alignment is not realized because the code inserted below adjusts
    // the stack pointer by numregs * 8 before aligning the stack pointer.
    if (DNum == 0)
      MFI.setObjectAlignment(FI, MFI.getMaxAlignment());
  }

  // Move the stack pointer to the d8 spill slot, and align it at the same
  // time. Leave the stack slot address in the scratch register r4.
  //
  //   sub r4, sp, #numregs * 8
  //   bic r4, r4, #align - 1
  //   mov sp, r4
  //
  bool isThumb = AFI->isThumbFunction();
  assert(!AFI->isThumb1OnlyFunction() && "Can't realign stack for thumb1");
  AFI->setShouldRestoreSPFromFP(true);

  // sub r4, sp, #numregs * 8
  // The immediate is <= 64, so it doesn't need any special encoding.
  unsigned Opc = isThumb ? ARM::t2SUBri : ARM::SUBri;
  AddDefaultCC(AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(Opc), ARM::R4)
                              .addReg(ARM::SP)
                              .addImm(8 * NumAlignedDPRCS2Regs)));

  // bic r4, r4, #align-1
  Opc = isThumb ? ARM::t2BICri : ARM::BICri;
  unsigned MaxAlign = MF.getFrameInfo()->getMaxAlignment();
  AddDefaultCC(AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(Opc), ARM::R4)
                              .addReg(ARM::R4, RegState::Kill)
                              .addImm(MaxAlign - 1)));

  // mov sp, r4
  // The stack pointer must be adjusted before spilling anything, otherwise
  // the stack slots could be clobbered by an interrupt handler.
  // Leave r4 live, it is used below.
  Opc = isThumb ? ARM::tMOVr : ARM::MOVr;
  MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII.get(Opc), ARM::SP)
                            .addReg(ARM::R4);
  MIB = AddDefaultPred(MIB);
  if (!isThumb)
    AddDefaultCC(MIB);

  // Now spill NumAlignedDPRCS2Regs registers starting from d8.
  // r4 holds the stack slot address.
  unsigned NextReg = ARM::D8;

  // 16-byte aligned vst1.64 with 4 d-regs and address writeback.
  // The writeback is only needed when emitting two vst1.64 instructions.
  if (NumAlignedDPRCS2Regs >= 6) {
    unsigned SupReg = TRI->getMatchingSuperReg(NextReg, ARM::dsub_0,
                                               &ARM::QQPRRegClass);
    MBB.addLiveIn(SupReg);
    AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(ARM::VST1d64Qwb_fixed),
                           ARM::R4)
                   .addReg(ARM::R4, RegState::Kill).addImm(16)
                   .addReg(NextReg)
                   .addReg(SupReg, RegState::ImplicitKill));
    NextReg += 4;
    NumAlignedDPRCS2Regs -= 4;
  }

  // We won't modify r4 beyond this point.  It currently points to the next
  // register to be spilled.
  unsigned R4BaseReg = NextReg;

  // 16-byte aligned vst1.64 with 4 d-regs, no writeback.
  if (NumAlignedDPRCS2Regs >= 4) {
    unsigned SupReg = TRI->getMatchingSuperReg(NextReg, ARM::dsub_0,
                                               &ARM::QQPRRegClass);
    MBB.addLiveIn(SupReg);
    AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(ARM::VST1d64Q))
                   .addReg(ARM::R4).addImm(16).addReg(NextReg)
                   .addReg(SupReg, RegState::ImplicitKill));
    NextReg += 4;
    NumAlignedDPRCS2Regs -= 4;
  }

  // 16-byte aligned vst1.64 with 2 d-regs.
  if (NumAlignedDPRCS2Regs >= 2) {
    unsigned SupReg = TRI->getMatchingSuperReg(NextReg, ARM::dsub_0,
                                               &ARM::QPRRegClass);
    MBB.addLiveIn(SupReg);
    AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(ARM::VST1q64))
                   .addReg(ARM::R4).addImm(16).addReg(SupReg));
    NextReg += 2;
    NumAlignedDPRCS2Regs -= 2;
  }

  // Finally, use a vanilla vstr.64 for the odd last register.
  if (NumAlignedDPRCS2Regs) {
    MBB.addLiveIn(NextReg);
    // vstr.64 uses addrmode5 which has an offset scale of 4.
    AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(ARM::VSTRD))
                   .addReg(NextReg)
                   .addReg(ARM::R4).addImm((NextReg-R4BaseReg)*2));
  }

  // The last spill instruction inserted should kill the scratch register r4.
  llvm::prior(MI)->addRegisterKilled(ARM::R4, TRI);
}

/// Skip past the code inserted by emitAlignedDPRCS2Spills, and return an
/// iterator to the following instruction.
static MachineBasicBlock::iterator
skipAlignedDPRCS2Spills(MachineBasicBlock::iterator MI,
                        unsigned NumAlignedDPRCS2Regs) {
  //   sub r4, sp, #numregs * 8
  //   bic r4, r4, #align - 1
  //   mov sp, r4
  ++MI; ++MI; ++MI;
  assert(MI->mayStore() && "Expecting spill instruction");

  // These switches all fall through.
  switch(NumAlignedDPRCS2Regs) {
  case 7:
    ++MI;
    assert(MI->mayStore() && "Expecting spill instruction");
  default:
    ++MI;
    assert(MI->mayStore() && "Expecting spill instruction");
  case 1:
  case 2:
  case 4:
    assert(MI->killsRegister(ARM::R4) && "Missed kill flag");
    ++MI;
  }
  return MI;
}

/// Emit aligned reload instructions for NumAlignedDPRCS2Regs D-registers
/// starting from d8.  These instructions are assumed to execute while the
/// stack is still aligned, unlike the code inserted by emitPopInst.
static void emitAlignedDPRCS2Restores(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MI,
                                      unsigned NumAlignedDPRCS2Regs,
                                      const std::vector<CalleeSavedInfo> &CSI,
                                      const TargetRegisterInfo *TRI) {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  DebugLoc DL = MI->getDebugLoc();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();

  // Find the frame index assigned to d8.
  int D8SpillFI = 0;
  for (unsigned i = 0, e = CSI.size(); i != e; ++i)
    if (CSI[i].getReg() == ARM::D8) {
      D8SpillFI = CSI[i].getFrameIdx();
      break;
    }

  // Materialize the address of the d8 spill slot into the scratch register r4.
  // This can be fairly complicated if the stack frame is large, so just use
  // the normal frame index elimination mechanism to do it.  This code runs as
  // the initial part of the epilog where the stack and base pointers haven't
  // been changed yet.
  bool isThumb = AFI->isThumbFunction();
  assert(!AFI->isThumb1OnlyFunction() && "Can't realign stack for thumb1");

  unsigned Opc = isThumb ? ARM::t2ADDri : ARM::ADDri;
  AddDefaultCC(AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(Opc), ARM::R4)
                              .addFrameIndex(D8SpillFI).addImm(0)));

  // Now restore NumAlignedDPRCS2Regs registers starting from d8.
  unsigned NextReg = ARM::D8;

  // 16-byte aligned vld1.64 with 4 d-regs and writeback.
  if (NumAlignedDPRCS2Regs >= 6) {
    unsigned SupReg = TRI->getMatchingSuperReg(NextReg, ARM::dsub_0,
                                               &ARM::QQPRRegClass);
    AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(ARM::VLD1d64Qwb_fixed), NextReg)
                   .addReg(ARM::R4, RegState::Define)
                   .addReg(ARM::R4, RegState::Kill).addImm(16)
                   .addReg(SupReg, RegState::ImplicitDefine));
    NextReg += 4;
    NumAlignedDPRCS2Regs -= 4;
  }

  // We won't modify r4 beyond this point.  It currently points to the next
  // register to be spilled.
  unsigned R4BaseReg = NextReg;

  // 16-byte aligned vld1.64 with 4 d-regs, no writeback.
  if (NumAlignedDPRCS2Regs >= 4) {
    unsigned SupReg = TRI->getMatchingSuperReg(NextReg, ARM::dsub_0,
                                               &ARM::QQPRRegClass);
    AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(ARM::VLD1d64Q), NextReg)
                   .addReg(ARM::R4).addImm(16)
                   .addReg(SupReg, RegState::ImplicitDefine));
    NextReg += 4;
    NumAlignedDPRCS2Regs -= 4;
  }

  // 16-byte aligned vld1.64 with 2 d-regs.
  if (NumAlignedDPRCS2Regs >= 2) {
    unsigned SupReg = TRI->getMatchingSuperReg(NextReg, ARM::dsub_0,
                                               &ARM::QPRRegClass);
    AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(ARM::VLD1q64), SupReg)
                   .addReg(ARM::R4).addImm(16));
    NextReg += 2;
    NumAlignedDPRCS2Regs -= 2;
  }

  // Finally, use a vanilla vldr.64 for the remaining odd register.
  if (NumAlignedDPRCS2Regs)
    AddDefaultPred(BuildMI(MBB, MI, DL, TII.get(ARM::VLDRD), NextReg)
                   .addReg(ARM::R4).addImm(2*(NextReg-R4BaseReg)));

  // Last store kills r4.
  llvm::prior(MI)->addRegisterKilled(ARM::R4, TRI);
}

bool ARMFrameLowering::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                        const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();

  unsigned PushOpc = AFI->isThumbFunction() ? ARM::t2STMDB_UPD : ARM::STMDB_UPD;
  unsigned PushOneOpc = AFI->isThumbFunction() ?
    ARM::t2STR_PRE : ARM::STR_PRE_IMM;
  unsigned FltOpc = ARM::VSTMDDB_UPD;
  unsigned NumAlignedDPRCS2Regs = AFI->getNumAlignedDPRCS2Regs();
  emitPushInst(MBB, MI, CSI, PushOpc, PushOneOpc, false, &isARMArea1Register, 0,
               MachineInstr::FrameSetup);
  emitPushInst(MBB, MI, CSI, PushOpc, PushOneOpc, false, &isARMArea2Register, 0,
               MachineInstr::FrameSetup);
  emitPushInst(MBB, MI, CSI, FltOpc, 0, true, &isARMArea3Register,
               NumAlignedDPRCS2Regs, MachineInstr::FrameSetup);

  // The code above does not insert spill code for the aligned DPRCS2 registers.
  // The stack realignment code will be inserted between the push instructions
  // and these spills.
  if (NumAlignedDPRCS2Regs)
    emitAlignedDPRCS2Spills(MBB, MI, NumAlignedDPRCS2Regs, CSI, TRI);

  return true;
}

bool ARMFrameLowering::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                        const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  bool isVarArg = AFI->getVarArgsRegSaveSize() > 0;
  unsigned NumAlignedDPRCS2Regs = AFI->getNumAlignedDPRCS2Regs();

  // The emitPopInst calls below do not insert reloads for the aligned DPRCS2
  // registers. Do that here instead.
  if (NumAlignedDPRCS2Regs)
    emitAlignedDPRCS2Restores(MBB, MI, NumAlignedDPRCS2Regs, CSI, TRI);

  unsigned PopOpc = AFI->isThumbFunction() ? ARM::t2LDMIA_UPD : ARM::LDMIA_UPD;
  unsigned LdrOpc = AFI->isThumbFunction() ? ARM::t2LDR_POST :ARM::LDR_POST_IMM;
  unsigned FltOpc = ARM::VLDMDIA_UPD;
  emitPopInst(MBB, MI, CSI, FltOpc, 0, isVarArg, true, &isARMArea3Register,
              NumAlignedDPRCS2Regs);
  emitPopInst(MBB, MI, CSI, PopOpc, LdrOpc, isVarArg, false,
              &isARMArea2Register, 0);
  emitPopInst(MBB, MI, CSI, PopOpc, LdrOpc, isVarArg, false,
              &isARMArea1Register, 0);

  return true;
}

// FIXME: Make generic?
static unsigned GetFunctionSizeInBytes(const MachineFunction &MF,
                                       const ARMBaseInstrInfo &TII) {
  unsigned FnSize = 0;
  for (MachineFunction::const_iterator MBBI = MF.begin(), E = MF.end();
       MBBI != E; ++MBBI) {
    const MachineBasicBlock &MBB = *MBBI;
    for (MachineBasicBlock::const_iterator I = MBB.begin(),E = MBB.end();
         I != E; ++I)
      FnSize += TII.GetInstSizeInBytes(I);
  }
  return FnSize;
}

/// estimateStackSize - Estimate and return the size of the frame.
/// FIXME: Make generic?
static unsigned estimateStackSize(MachineFunction &MF) {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  const TargetRegisterInfo *RegInfo = MF.getTarget().getRegisterInfo();
  unsigned MaxAlign = MFI->getMaxAlignment();
  int Offset = 0;

  // This code is very, very similar to PEI::calculateFrameObjectOffsets().
  // It really should be refactored to share code. Until then, changes
  // should keep in mind that there's tight coupling between the two.

  for (int i = MFI->getObjectIndexBegin(); i != 0; ++i) {
    int FixedOff = -MFI->getObjectOffset(i);
    if (FixedOff > Offset) Offset = FixedOff;
  }
  for (unsigned i = 0, e = MFI->getObjectIndexEnd(); i != e; ++i) {
    if (MFI->isDeadObjectIndex(i))
      continue;
    Offset += MFI->getObjectSize(i);
    unsigned Align = MFI->getObjectAlignment(i);
    // Adjust to alignment boundary
    Offset = (Offset+Align-1)/Align*Align;

    MaxAlign = std::max(Align, MaxAlign);
  }

  if (MFI->adjustsStack() && TFI->hasReservedCallFrame(MF))
    Offset += MFI->getMaxCallFrameSize();

  // Round up the size to a multiple of the alignment.  If the function has
  // any calls or alloca's, align to the target's StackAlignment value to
  // ensure that the callee's frame or the alloca data is suitably aligned;
  // otherwise, for leaf functions, align to the TransientStackAlignment
  // value.
  unsigned StackAlign;
  if (MFI->adjustsStack() || MFI->hasVarSizedObjects() ||
      (RegInfo->needsStackRealignment(MF) && MFI->getObjectIndexEnd() != 0))
    StackAlign = TFI->getStackAlignment();
  else
    StackAlign = TFI->getTransientStackAlignment();

  // If the frame pointer is eliminated, all frame offsets will be relative to
  // SP not FP. Align to MaxAlign so this works.
  StackAlign = std::max(StackAlign, MaxAlign);
  unsigned AlignMask = StackAlign - 1;
  Offset = (Offset + AlignMask) & ~uint64_t(AlignMask);

  return (unsigned)Offset;
}

/// estimateRSStackSizeLimit - Look at each instruction that references stack
/// frames and return the stack size limit beyond which some of these
/// instructions will require a scratch register during their expansion later.
// FIXME: Move to TII?
static unsigned estimateRSStackSizeLimit(MachineFunction &MF,
                                         const TargetFrameLowering *TFI) {
  const ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  unsigned Limit = (1 << 12) - 1;
  for (MachineFunction::iterator BB = MF.begin(),E = MF.end(); BB != E; ++BB) {
    for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end();
         I != E; ++I) {
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
        if (!I->getOperand(i).isFI()) continue;

        // When using ADDri to get the address of a stack object, 255 is the
        // largest offset guaranteed to fit in the immediate offset.
        if (I->getOpcode() == ARM::ADDri) {
          Limit = std::min(Limit, (1U << 8) - 1);
          break;
        }

        // Otherwise check the addressing mode.
        switch (I->getDesc().TSFlags & ARMII::AddrModeMask) {
        case ARMII::AddrMode3:
        case ARMII::AddrModeT2_i8:
          Limit = std::min(Limit, (1U << 8) - 1);
          break;
        case ARMII::AddrMode5:
        case ARMII::AddrModeT2_i8s4:
          Limit = std::min(Limit, ((1U << 8) - 1) * 4);
          break;
        case ARMII::AddrModeT2_i12:
          // i12 supports only positive offset so these will be converted to
          // i8 opcodes. See llvm::rewriteT2FrameIndex.
          if (TFI->hasFP(MF) && AFI->hasStackFrame())
            Limit = std::min(Limit, (1U << 8) - 1);
          break;
        case ARMII::AddrMode4:
        case ARMII::AddrMode6:
          // Addressing modes 4 & 6 (load/store) instructions can't encode an
          // immediate offset for stack references.
          return 0;
        default:
          break;
        }
        break; // At most one FI per instruction
      }
    }
  }

  return Limit;
}

// In functions that realign the stack, it can be an advantage to spill the
// callee-saved vector registers after realigning the stack. The vst1 and vld1
// instructions take alignment hints that can improve performance.
//
static void checkNumAlignedDPRCS2Regs(MachineFunction &MF) {
  MF.getInfo<ARMFunctionInfo>()->setNumAlignedDPRCS2Regs(0);
  if (!SpillAlignedNEONRegs)
    return;

  // Naked functions don't spill callee-saved registers.
  if (MF.getFunction()->getFnAttributes().hasAttribute(Attributes::Naked))
    return;

  // We are planning to use NEON instructions vst1 / vld1.
  if (!MF.getTarget().getSubtarget<ARMSubtarget>().hasNEON())
    return;

  // Don't bother if the default stack alignment is sufficiently high.
  if (MF.getTarget().getFrameLowering()->getStackAlignment() >= 8)
    return;

  // Aligned spills require stack realignment.
  const ARMBaseRegisterInfo *RegInfo =
    static_cast<const ARMBaseRegisterInfo*>(MF.getTarget().getRegisterInfo());
  if (!RegInfo->canRealignStack(MF))
    return;

  // We always spill contiguous d-registers starting from d8. Count how many
  // needs spilling.  The register allocator will almost always use the
  // callee-saved registers in order, but it can happen that there are holes in
  // the range.  Registers above the hole will be spilled to the standard DPRCS
  // area.
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned NumSpills = 0;
  for (; NumSpills < 8; ++NumSpills)
    if (!MRI.isPhysRegUsed(ARM::D8 + NumSpills))
      break;

  // Don't do this for just one d-register. It's not worth it.
  if (NumSpills < 2)
    return;

  // Spill the first NumSpills D-registers after realigning the stack.
  MF.getInfo<ARMFunctionInfo>()->setNumAlignedDPRCS2Regs(NumSpills);

  // A scratch register is required for the vst1 / vld1 instructions.
  MF.getRegInfo().setPhysRegUsed(ARM::R4);
}

void
ARMFrameLowering::processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                                       RegScavenger *RS) const {
  // This tells PEI to spill the FP as if it is any other callee-save register
  // to take advantage the eliminateFrameIndex machinery. This also ensures it
  // is spilled in the order specified by getCalleeSavedRegs() to make it easier
  // to combine multiple loads / stores.
  bool CanEliminateFrame = true;
  bool CS1Spilled = false;
  bool LRSpilled = false;
  unsigned NumGPRSpills = 0;
  SmallVector<unsigned, 4> UnspilledCS1GPRs;
  SmallVector<unsigned, 4> UnspilledCS2GPRs;
  const ARMBaseRegisterInfo *RegInfo =
    static_cast<const ARMBaseRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const ARMBaseInstrInfo &TII =
    *static_cast<const ARMBaseInstrInfo*>(MF.getTarget().getInstrInfo());
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned FramePtr = RegInfo->getFrameRegister(MF);

  // Spill R4 if Thumb2 function requires stack realignment - it will be used as
  // scratch register. Also spill R4 if Thumb2 function has varsized objects,
  // since it's not always possible to restore sp from fp in a single
  // instruction.
  // FIXME: It will be better just to find spare register here.
  if (AFI->isThumb2Function() &&
      (MFI->hasVarSizedObjects() || RegInfo->needsStackRealignment(MF)))
    MRI.setPhysRegUsed(ARM::R4);

  if (AFI->isThumb1OnlyFunction()) {
    // Spill LR if Thumb1 function uses variable length argument lists.
    if (AFI->getVarArgsRegSaveSize() > 0)
      MRI.setPhysRegUsed(ARM::LR);

    // Spill R4 if Thumb1 epilogue has to restore SP from FP. We don't know
    // for sure what the stack size will be, but for this, an estimate is good
    // enough. If there anything changes it, it'll be a spill, which implies
    // we've used all the registers and so R4 is already used, so not marking
    // it here will be OK.
    // FIXME: It will be better just to find spare register here.
    unsigned StackSize = estimateStackSize(MF);
    if (MFI->hasVarSizedObjects() || StackSize > 508)
      MRI.setPhysRegUsed(ARM::R4);
  }

  // See if we can spill vector registers to aligned stack.
  checkNumAlignedDPRCS2Regs(MF);

  // Spill the BasePtr if it's used.
  if (RegInfo->hasBasePointer(MF))
    MRI.setPhysRegUsed(RegInfo->getBaseRegister());

  // Don't spill FP if the frame can be eliminated. This is determined
  // by scanning the callee-save registers to see if any is used.
  const uint16_t *CSRegs = RegInfo->getCalleeSavedRegs();
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    bool Spilled = false;
    if (MRI.isPhysRegUsed(Reg)) {
      Spilled = true;
      CanEliminateFrame = false;
    }

    if (!ARM::GPRRegClass.contains(Reg))
      continue;

    if (Spilled) {
      NumGPRSpills++;

      if (!STI.isTargetIOS()) {
        if (Reg == ARM::LR)
          LRSpilled = true;
        CS1Spilled = true;
        continue;
      }

      // Keep track if LR and any of R4, R5, R6, and R7 is spilled.
      switch (Reg) {
      case ARM::LR:
        LRSpilled = true;
        // Fallthrough
      case ARM::R4: case ARM::R5:
      case ARM::R6: case ARM::R7:
        CS1Spilled = true;
        break;
      default:
        break;
      }
    } else {
      if (!STI.isTargetIOS()) {
        UnspilledCS1GPRs.push_back(Reg);
        continue;
      }

      switch (Reg) {
      case ARM::R4: case ARM::R5:
      case ARM::R6: case ARM::R7:
      case ARM::LR:
        UnspilledCS1GPRs.push_back(Reg);
        break;
      default:
        UnspilledCS2GPRs.push_back(Reg);
        break;
      }
    }
  }

  bool ForceLRSpill = false;
  if (!LRSpilled && AFI->isThumb1OnlyFunction()) {
    unsigned FnSize = GetFunctionSizeInBytes(MF, TII);
    // Force LR to be spilled if the Thumb function size is > 2048. This enables
    // use of BL to implement far jump. If it turns out that it's not needed
    // then the branch fix up path will undo it.
    if (FnSize >= (1 << 11)) {
      CanEliminateFrame = false;
      ForceLRSpill = true;
    }
  }

  // If any of the stack slot references may be out of range of an immediate
  // offset, make sure a register (or a spill slot) is available for the
  // register scavenger. Note that if we're indexing off the frame pointer, the
  // effective stack size is 4 bytes larger since the FP points to the stack
  // slot of the previous FP. Also, if we have variable sized objects in the
  // function, stack slot references will often be negative, and some of
  // our instructions are positive-offset only, so conservatively consider
  // that case to want a spill slot (or register) as well. Similarly, if
  // the function adjusts the stack pointer during execution and the
  // adjustments aren't already part of our stack size estimate, our offset
  // calculations may be off, so be conservative.
  // FIXME: We could add logic to be more precise about negative offsets
  //        and which instructions will need a scratch register for them. Is it
  //        worth the effort and added fragility?
  bool BigStack =
    (RS &&
     (estimateStackSize(MF) + ((hasFP(MF) && AFI->hasStackFrame()) ? 4:0) >=
      estimateRSStackSizeLimit(MF, this)))
    || MFI->hasVarSizedObjects()
    || (MFI->adjustsStack() && !canSimplifyCallFramePseudos(MF));

  bool ExtraCSSpill = false;
  if (BigStack || !CanEliminateFrame || RegInfo->cannotEliminateFrame(MF)) {
    AFI->setHasStackFrame(true);

    // If LR is not spilled, but at least one of R4, R5, R6, and R7 is spilled.
    // Spill LR as well so we can fold BX_RET to the registers restore (LDM).
    if (!LRSpilled && CS1Spilled) {
      MRI.setPhysRegUsed(ARM::LR);
      NumGPRSpills++;
      UnspilledCS1GPRs.erase(std::find(UnspilledCS1GPRs.begin(),
                                    UnspilledCS1GPRs.end(), (unsigned)ARM::LR));
      ForceLRSpill = false;
      ExtraCSSpill = true;
    }

    if (hasFP(MF)) {
      MRI.setPhysRegUsed(FramePtr);
      NumGPRSpills++;
    }

    // If stack and double are 8-byte aligned and we are spilling an odd number
    // of GPRs, spill one extra callee save GPR so we won't have to pad between
    // the integer and double callee save areas.
    unsigned TargetAlign = getStackAlignment();
    if (TargetAlign == 8 && (NumGPRSpills & 1)) {
      if (CS1Spilled && !UnspilledCS1GPRs.empty()) {
        for (unsigned i = 0, e = UnspilledCS1GPRs.size(); i != e; ++i) {
          unsigned Reg = UnspilledCS1GPRs[i];
          // Don't spill high register if the function is thumb1
          if (!AFI->isThumb1OnlyFunction() ||
              isARMLowRegister(Reg) || Reg == ARM::LR) {
            MRI.setPhysRegUsed(Reg);
            if (!MRI.isReserved(Reg))
              ExtraCSSpill = true;
            break;
          }
        }
      } else if (!UnspilledCS2GPRs.empty() && !AFI->isThumb1OnlyFunction()) {
        unsigned Reg = UnspilledCS2GPRs.front();
        MRI.setPhysRegUsed(Reg);
        if (!MRI.isReserved(Reg))
          ExtraCSSpill = true;
      }
    }

    // Estimate if we might need to scavenge a register at some point in order
    // to materialize a stack offset. If so, either spill one additional
    // callee-saved register or reserve a special spill slot to facilitate
    // register scavenging. Thumb1 needs a spill slot for stack pointer
    // adjustments also, even when the frame itself is small.
    if (BigStack && !ExtraCSSpill) {
      // If any non-reserved CS register isn't spilled, just spill one or two
      // extra. That should take care of it!
      unsigned NumExtras = TargetAlign / 4;
      SmallVector<unsigned, 2> Extras;
      while (NumExtras && !UnspilledCS1GPRs.empty()) {
        unsigned Reg = UnspilledCS1GPRs.back();
        UnspilledCS1GPRs.pop_back();
        if (!MRI.isReserved(Reg) &&
            (!AFI->isThumb1OnlyFunction() || isARMLowRegister(Reg) ||
             Reg == ARM::LR)) {
          Extras.push_back(Reg);
          NumExtras--;
        }
      }
      // For non-Thumb1 functions, also check for hi-reg CS registers
      if (!AFI->isThumb1OnlyFunction()) {
        while (NumExtras && !UnspilledCS2GPRs.empty()) {
          unsigned Reg = UnspilledCS2GPRs.back();
          UnspilledCS2GPRs.pop_back();
          if (!MRI.isReserved(Reg)) {
            Extras.push_back(Reg);
            NumExtras--;
          }
        }
      }
      if (Extras.size() && NumExtras == 0) {
        for (unsigned i = 0, e = Extras.size(); i != e; ++i) {
          MRI.setPhysRegUsed(Extras[i]);
        }
      } else if (!AFI->isThumb1OnlyFunction()) {
        // note: Thumb1 functions spill to R12, not the stack.  Reserve a slot
        // closest to SP or frame pointer.
        const TargetRegisterClass *RC = &ARM::GPRRegClass;
        RS->setScavengingFrameIndex(MFI->CreateStackObject(RC->getSize(),
                                                           RC->getAlignment(),
                                                           false));
      }
    }
  }

  if (ForceLRSpill) {
    MRI.setPhysRegUsed(ARM::LR);
    AFI->setLRIsSpilledForFarJump(true);
  }
}
