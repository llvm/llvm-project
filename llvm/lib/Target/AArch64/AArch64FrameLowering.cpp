//===- AArch64FrameLowering.cpp - AArch64 Frame Lowering -------*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AArch64 implementation of TargetFrameLowering class.
//
// On AArch64, stack frames are structured as follows:
//
// The stack grows downward.
//
// All of the individual frame areas on the frame below are optional, i.e. it's
// possible to create a function so that the particular area isn't present
// in the frame.
//
// At function entry, the "frame" looks as follows:
//
// |                                   | Higher address
// |-----------------------------------|
// |                                   |
// | arguments passed on the stack     |
// |                                   |
// |-----------------------------------| <- sp
// |                                   | Lower address
//
//
// After the prologue has run, the frame has the following general structure.
// Note that this doesn't depict the case where a red-zone is used. Also,
// technically the last frame area (VLAs) doesn't get created until in the
// main function body, after the prologue is run. However, it's depicted here
// for completeness.
//
// |                                   | Higher address
// |-----------------------------------|
// |                                   |
// | arguments passed on the stack     |
// |                                   |
// |-----------------------------------|
// |                                   |
// | prev_fp, prev_lr                  |
// | (a.k.a. "frame record")           |
// |-----------------------------------| <- fp(=x29)
// |                                   |
// | other callee-saved registers      |
// |                                   |
// |-----------------------------------|
// |.empty.space.to.make.part.below....|
// |.aligned.in.case.it.needs.more.than| (size of this area is unknown at
// |.the.standard.16-byte.alignment....|  compile time; if present)
// |-----------------------------------|
// |                                   |
// | local variables of fixed size     |
// | including spill slots             |
// |-----------------------------------| <- bp(not defined by ABI,
// |.variable-sized.local.variables....|       LLVM chooses X19)
// |.(VLAs)............................| (size of this area is unknown at
// |...................................|  compile time)
// |-----------------------------------| <- sp
// |                                   | Lower address
//
//
// To access the data in a frame, at-compile time, a constant offset must be
// computable from one of the pointers (fp, bp, sp) to access it. The size
// of the areas with a dotted background cannot be computed at compile-time
// if they are present, making it required to have all three of fp, bp and
// sp to be set up to be able to access all contents in the frame areas,
// assuming all of the frame areas are non-empty.
//
// For most functions, some of the frame areas are empty. For those functions,
// it may not be necessary to set up fp or bp:
// * A base pointer is definitly needed when there are both VLAs and local
//   variables with more-than-default alignment requirements.
// * A frame pointer is definitly needed when there are local variables with
//   more-than-default alignment requirements.
//
// In some cases when a base pointer is not strictly needed, it is generated
// anyway when offsets from the frame pointer to access local variables become
// so large that the offset can't be encoded in the immediate fields of loads
// or stores.
//
// FIXME: also explain the redzone concept.
// FIXME: also explain the concept of reserved call frames.
//
//===----------------------------------------------------------------------===//

#include "AArch64FrameLowering.h"
#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "frame-info"

static cl::opt<bool> EnableRedZone("aarch64-redzone",
                                   cl::desc("enable use of redzone on AArch64"),
                                   cl::init(false), cl::Hidden);

STATISTIC(NumRedZoneFunctions, "Number of functions using red zone");

bool AArch64FrameLowering::canUseRedZone(const MachineFunction &MF) const {
  if (!EnableRedZone)
    return false;
  // Don't use the red zone if the function explicitly asks us not to.
  // This is typically used for kernel code.
  if (MF.getFunction()->hasFnAttribute(Attribute::NoRedZone))
    return false;

  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  unsigned NumBytes = AFI->getLocalStackSize();

  // Note: currently hasFP() is always true for hasCalls(), but that's an
  // implementation detail of the current code, not a strict requirement,
  // so stay safe here and check both.
  if (MFI->hasCalls() || hasFP(MF) || NumBytes > 128)
    return false;
  return true;
}

/// hasFP - Return true if the specified function should have a dedicated frame
/// pointer register.
bool AArch64FrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  return (MFI->hasCalls() || MFI->hasVarSizedObjects() ||
          MFI->isFrameAddressTaken() || MFI->hasStackMap() ||
          MFI->hasPatchPoint() || RegInfo->needsStackRealignment(MF));
}

/// hasReservedCallFrame - Under normal circumstances, when a frame pointer is
/// not required, we reserve argument space for call sites in the function
/// immediately on entry to the current function.  This eliminates the need for
/// add/sub sp brackets around call sites.  Returns true if the call frame is
/// included as part of the stack frame.
bool
AArch64FrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  return !MF.getFrameInfo()->hasVarSizedObjects();
}

void AArch64FrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  const AArch64InstrInfo *TII =
      static_cast<const AArch64InstrInfo *>(MF.getSubtarget().getInstrInfo());
  DebugLoc DL = I->getDebugLoc();
  unsigned Opc = I->getOpcode();
  bool IsDestroy = Opc == TII->getCallFrameDestroyOpcode();
  uint64_t CalleePopAmount = IsDestroy ? I->getOperand(1).getImm() : 0;

  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  if (!TFI->hasReservedCallFrame(MF)) {
    unsigned Align = getStackAlignment();

    int64_t Amount = I->getOperand(0).getImm();
    Amount = RoundUpToAlignment(Amount, Align);
    if (!IsDestroy)
      Amount = -Amount;

    // N.b. if CalleePopAmount is valid but zero (i.e. callee would pop, but it
    // doesn't have to pop anything), then the first operand will be zero too so
    // this adjustment is a no-op.
    if (CalleePopAmount == 0) {
      // FIXME: in-function stack adjustment for calls is limited to 24-bits
      // because there's no guaranteed temporary register available.
      //
      // ADD/SUB (immediate) has only LSL #0 and LSL #12 available.
      // 1) For offset <= 12-bit, we use LSL #0
      // 2) For 12-bit <= offset <= 24-bit, we use two instructions. One uses
      // LSL #0, and the other uses LSL #12.
      //
      // Mostly call frames will be allocated at the start of a function so
      // this is OK, but it is a limitation that needs dealing with.
      assert(Amount > -0xffffff && Amount < 0xffffff && "call frame too large");
      emitFrameOffset(MBB, I, DL, AArch64::SP, AArch64::SP, Amount, TII);
    }
  } else if (CalleePopAmount != 0) {
    // If the calling convention demands that the callee pops arguments from the
    // stack, we want to add it back if we have a reserved call frame.
    assert(CalleePopAmount < 0xffffff && "call frame too large");
    emitFrameOffset(MBB, I, DL, AArch64::SP, AArch64::SP, -CalleePopAmount,
                    TII);
  }
  MBB.erase(I);
}

void AArch64FrameLowering::emitCalleeSavedFrameMoves(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    unsigned FramePtr) const {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineModuleInfo &MMI = MF.getMMI();
  const MCRegisterInfo *MRI = MMI.getContext().getRegisterInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  DebugLoc DL = MBB.findDebugLoc(MBBI);

  // Add callee saved registers to move list.
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  if (CSI.empty())
    return;

  const DataLayout *TD = MF.getTarget().getDataLayout();
  bool HasFP = hasFP(MF);

  // Calculate amount of bytes used for return address storing.
  int stackGrowth = -TD->getPointerSize(0);

  // Calculate offsets.
  int64_t saveAreaOffset = (HasFP ? 2 : 1) * stackGrowth;
  unsigned TotalSkipped = 0;
  for (const auto &Info : CSI) {
    unsigned Reg = Info.getReg();
    int64_t Offset = MFI->getObjectOffset(Info.getFrameIdx()) -
                     getOffsetOfLocalArea() + saveAreaOffset;

    // Don't output a new CFI directive if we're re-saving the frame pointer or
    // link register. This happens when the PrologEpilogInserter has inserted an
    // extra "STP" of the frame pointer and link register -- the "emitPrologue"
    // method automatically generates the directives when frame pointers are
    // used. If we generate CFI directives for the extra "STP"s, the linker will
    // lose track of the correct values for the frame pointer and link register.
    if (HasFP && (FramePtr == Reg || Reg == AArch64::LR)) {
      TotalSkipped += stackGrowth;
      continue;
    }

    unsigned DwarfReg = MRI->getDwarfRegNum(Reg, true);
    unsigned CFIIndex = MMI.addFrameInst(MCCFIInstruction::createOffset(
        nullptr, DwarfReg, Offset - TotalSkipped));
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlags(MachineInstr::FrameSetup);
  }
}

/// Get FPOffset by analyzing the first instruction.
static int getFPOffsetInPrologue(MachineInstr *MBBI) {
  // First instruction must a) allocate the stack  and b) have an immediate
  // that is a multiple of -2.
  assert(((MBBI->getOpcode() == AArch64::STPXpre ||
           MBBI->getOpcode() == AArch64::STPDpre) &&
          MBBI->getOperand(3).getReg() == AArch64::SP &&
          MBBI->getOperand(4).getImm() < 0 &&
          (MBBI->getOperand(4).getImm() & 1) == 0));

  // Frame pointer is fp = sp - 16. Since the  STPXpre subtracts the space
  // required for the callee saved register area we get the frame pointer
  // by addding that offset - 16 = -getImm()*8 - 2*8 = -(getImm() + 2) * 8.
  int FPOffset = -(MBBI->getOperand(4).getImm() + 2) * 8;
  assert(FPOffset >= 0 && "Bad Framepointer Offset");
  return FPOffset;
}

static bool isCSSave(MachineInstr *MBBI) {
  return MBBI->getOpcode() == AArch64::STPXi ||
         MBBI->getOpcode() == AArch64::STPDi ||
         MBBI->getOpcode() == AArch64::STPXpre ||
         MBBI->getOpcode() == AArch64::STPDpre;
}

void AArch64FrameLowering::emitPrologue(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.begin();
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const Function *Fn = MF.getFunction();
  const AArch64RegisterInfo *RegInfo = static_cast<const AArch64RegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  MachineModuleInfo &MMI = MF.getMMI();
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  bool needsFrameMoves = MMI.hasDebugInfo() || Fn->needsUnwindTableEntry();
  bool HasFP = hasFP(MF);
  DebugLoc DL = MBB.findDebugLoc(MBBI);

  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction()->getCallingConv() == CallingConv::GHC)
    return;

  int NumBytes = (int)MFI->getStackSize();
  if (!AFI->hasStackFrame()) {
    assert(!HasFP && "unexpected function without stack frame but with FP");

    // All of the stack allocation is for locals.
    AFI->setLocalStackSize(NumBytes);

    // Label used to tie together the PROLOG_LABEL and the MachineMoves.
    MCSymbol *FrameLabel = MMI.getContext().createTempSymbol();

    // REDZONE: If the stack size is less than 128 bytes, we don't need
    // to actually allocate.
    if (NumBytes && !canUseRedZone(MF)) {
      emitFrameOffset(MBB, MBBI, DL, AArch64::SP, AArch64::SP, -NumBytes, TII,
                      MachineInstr::FrameSetup);

      // Encode the stack size of the leaf function.
      unsigned CFIIndex = MMI.addFrameInst(
          MCCFIInstruction::createDefCfaOffset(FrameLabel, -NumBytes));
      BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex)
          .setMIFlags(MachineInstr::FrameSetup);
    } else if (NumBytes) {
      ++NumRedZoneFunctions;
    }

    return;
  }

  // Only set up FP if we actually need to.
  int FPOffset = 0;
  if (HasFP)
    FPOffset = getFPOffsetInPrologue(MBBI);

  // Move past the saves of the callee-saved registers.
  while (isCSSave(MBBI)) {
    ++MBBI;
    NumBytes -= 16;
  }
  assert(NumBytes >= 0 && "Negative stack allocation size!?");
  if (HasFP) {
    // Issue    sub fp, sp, FPOffset or
    //          mov fp,sp          when FPOffset is zero.
    // Note: All stores of callee-saved registers are marked as "FrameSetup".
    // This code marks the instruction(s) that set the FP also.
    emitFrameOffset(MBB, MBBI, DL, AArch64::FP, AArch64::SP, FPOffset, TII,
                    MachineInstr::FrameSetup);
  }

  // All of the remaining stack allocations are for locals.
  AFI->setLocalStackSize(NumBytes);

  // Allocate space for the rest of the frame.

  const unsigned Alignment = MFI->getMaxAlignment();
  const bool NeedsRealignment = RegInfo->needsStackRealignment(MF);
  unsigned scratchSPReg = AArch64::SP;
  if (NumBytes && NeedsRealignment) {
    // Use the first callee-saved register as a scratch register.
    scratchSPReg = AArch64::X9;
    MF.getRegInfo().setPhysRegUsed(scratchSPReg);
  }

  // If we're a leaf function, try using the red zone.
  if (NumBytes && !canUseRedZone(MF))
    // FIXME: in the case of dynamic re-alignment, NumBytes doesn't have
    // the correct value here, as NumBytes also includes padding bytes,
    // which shouldn't be counted here.
    emitFrameOffset(MBB, MBBI, DL, scratchSPReg, AArch64::SP, -NumBytes, TII,
                    MachineInstr::FrameSetup);

  if (NumBytes && NeedsRealignment) {
    const unsigned NrBitsToZero = countTrailingZeros(Alignment);
    assert(NrBitsToZero > 1);
    assert(scratchSPReg != AArch64::SP);

    // SUB X9, SP, NumBytes
    //   -- X9 is temporary register, so shouldn't contain any live data here,
    //   -- free to use. This is already produced by emitFrameOffset above.
    // AND SP, X9, 0b11111...0000
    // The logical immediates have a non-trivial encoding. The following
    // formula computes the encoded immediate with all ones but
    // NrBitsToZero zero bits as least significant bits.
    uint32_t andMaskEncoded =
        (1                   <<12) // = N
      | ((64-NrBitsToZero)   << 6) // immr
      | ((64-NrBitsToZero-1) << 0) // imms
      ;
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ANDXri), AArch64::SP)
      .addReg(scratchSPReg, RegState::Kill)
      .addImm(andMaskEncoded);
  }

  // If we need a base pointer, set it up here. It's whatever the value of the
  // stack pointer is at this point. Any variable size objects will be allocated
  // after this, so we can still use the base pointer to reference locals.
  //
  // FIXME: Clarify FrameSetup flags here.
  // Note: Use emitFrameOffset() like above for FP if the FrameSetup flag is
  // needed.
  if (RegInfo->hasBasePointer(MF)) {
    TII->copyPhysReg(MBB, MBBI, DL, RegInfo->getBaseRegister(), AArch64::SP,
                     false);
  }

  if (needsFrameMoves) {
    const DataLayout *TD = MF.getTarget().getDataLayout();
    const int StackGrowth = -TD->getPointerSize(0);
    unsigned FramePtr = RegInfo->getFrameRegister(MF);
    // An example of the prologue:
    //
    //     .globl __foo
    //     .align 2
    //  __foo:
    // Ltmp0:
    //     .cfi_startproc
    //     .cfi_personality 155, ___gxx_personality_v0
    // Leh_func_begin:
    //     .cfi_lsda 16, Lexception33
    //
    //     stp  xa,bx, [sp, -#offset]!
    //     ...
    //     stp  x28, x27, [sp, #offset-32]
    //     stp  fp, lr, [sp, #offset-16]
    //     add  fp, sp, #offset - 16
    //     sub  sp, sp, #1360
    //
    // The Stack:
    //       +-------------------------------------------+
    // 10000 | ........ | ........ | ........ | ........ |
    // 10004 | ........ | ........ | ........ | ........ |
    //       +-------------------------------------------+
    // 10008 | ........ | ........ | ........ | ........ |
    // 1000c | ........ | ........ | ........ | ........ |
    //       +===========================================+
    // 10010 |                X28 Register               |
    // 10014 |                X28 Register               |
    //       +-------------------------------------------+
    // 10018 |                X27 Register               |
    // 1001c |                X27 Register               |
    //       +===========================================+
    // 10020 |                Frame Pointer              |
    // 10024 |                Frame Pointer              |
    //       +-------------------------------------------+
    // 10028 |                Link Register              |
    // 1002c |                Link Register              |
    //       +===========================================+
    // 10030 | ........ | ........ | ........ | ........ |
    // 10034 | ........ | ........ | ........ | ........ |
    //       +-------------------------------------------+
    // 10038 | ........ | ........ | ........ | ........ |
    // 1003c | ........ | ........ | ........ | ........ |
    //       +-------------------------------------------+
    //
    //     [sp] = 10030        ::    >>initial value<<
    //     sp = 10020          ::  stp fp, lr, [sp, #-16]!
    //     fp = sp == 10020    ::  mov fp, sp
    //     [sp] == 10020       ::  stp x28, x27, [sp, #-16]!
    //     sp == 10010         ::    >>final value<<
    //
    // The frame pointer (w29) points to address 10020. If we use an offset of
    // '16' from 'w29', we get the CFI offsets of -8 for w30, -16 for w29, -24
    // for w27, and -32 for w28:
    //
    //  Ltmp1:
    //     .cfi_def_cfa w29, 16
    //  Ltmp2:
    //     .cfi_offset w30, -8
    //  Ltmp3:
    //     .cfi_offset w29, -16
    //  Ltmp4:
    //     .cfi_offset w27, -24
    //  Ltmp5:
    //     .cfi_offset w28, -32

    if (HasFP) {
      // Define the current CFA rule to use the provided FP.
      unsigned Reg = RegInfo->getDwarfRegNum(FramePtr, true);
      unsigned CFIIndex = MMI.addFrameInst(
          MCCFIInstruction::createDefCfa(nullptr, Reg, 2 * StackGrowth));
      BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex)
          .setMIFlags(MachineInstr::FrameSetup);

      // Record the location of the stored LR
      unsigned LR = RegInfo->getDwarfRegNum(AArch64::LR, true);
      CFIIndex = MMI.addFrameInst(
          MCCFIInstruction::createOffset(nullptr, LR, StackGrowth));
      BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex)
          .setMIFlags(MachineInstr::FrameSetup);

      // Record the location of the stored FP
      CFIIndex = MMI.addFrameInst(
          MCCFIInstruction::createOffset(nullptr, Reg, 2 * StackGrowth));
      BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex)
          .setMIFlags(MachineInstr::FrameSetup);
    } else {
      // Encode the stack size of the leaf function.
      unsigned CFIIndex = MMI.addFrameInst(
          MCCFIInstruction::createDefCfaOffset(nullptr, -MFI->getStackSize()));
      BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex)
          .setMIFlags(MachineInstr::FrameSetup);
    }

    // Now emit the moves for whatever callee saved regs we have.
    emitCalleeSavedFrameMoves(MBB, MBBI, FramePtr);
  }
}

static bool isCalleeSavedRegister(unsigned Reg, const MCPhysReg *CSRegs) {
  for (unsigned i = 0; CSRegs[i]; ++i)
    if (Reg == CSRegs[i])
      return true;
  return false;
}

static bool isCSRestore(MachineInstr *MI, const MCPhysReg *CSRegs) {
  unsigned RtIdx = 0;
  if (MI->getOpcode() == AArch64::LDPXpost ||
      MI->getOpcode() == AArch64::LDPDpost)
    RtIdx = 1;

  if (MI->getOpcode() == AArch64::LDPXpost ||
      MI->getOpcode() == AArch64::LDPDpost ||
      MI->getOpcode() == AArch64::LDPXi || MI->getOpcode() == AArch64::LDPDi) {
    if (!isCalleeSavedRegister(MI->getOperand(RtIdx).getReg(), CSRegs) ||
        !isCalleeSavedRegister(MI->getOperand(RtIdx + 1).getReg(), CSRegs) ||
        MI->getOperand(RtIdx + 2).getReg() != AArch64::SP)
      return false;
    return true;
  }

  return false;
}

void AArch64FrameLowering::emitEpilogue(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const AArch64InstrInfo *TII =
      static_cast<const AArch64InstrInfo *>(MF.getSubtarget().getInstrInfo());
  const AArch64RegisterInfo *RegInfo = static_cast<const AArch64RegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  DebugLoc DL;
  bool IsTailCallReturn = false;
  if (MBB.end() != MBBI) {
    DL = MBBI->getDebugLoc();
    unsigned RetOpcode = MBBI->getOpcode();
    IsTailCallReturn = RetOpcode == AArch64::TCRETURNdi ||
      RetOpcode == AArch64::TCRETURNri;
  }
  int NumBytes = MFI->getStackSize();
  const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();

  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction()->getCallingConv() == CallingConv::GHC)
    return;

  // Initial and residual are named for consistency with the prologue. Note that
  // in the epilogue, the residual adjustment is executed first.
  uint64_t ArgumentPopSize = 0;
  if (IsTailCallReturn) {
    MachineOperand &StackAdjust = MBBI->getOperand(1);

    // For a tail-call in a callee-pops-arguments environment, some or all of
    // the stack may actually be in use for the call's arguments, this is
    // calculated during LowerCall and consumed here...
    ArgumentPopSize = StackAdjust.getImm();
  } else {
    // ... otherwise the amount to pop is *all* of the argument space,
    // conveniently stored in the MachineFunctionInfo by
    // LowerFormalArguments. This will, of course, be zero for the C calling
    // convention.
    ArgumentPopSize = AFI->getArgumentStackToRestore();
  }

  // The stack frame should be like below,
  //
  //      ----------------------                     ---
  //      |                    |                      |
  //      | BytesInStackArgArea|              CalleeArgStackSize
  //      | (NumReusableBytes) |                (of tail call)
  //      |                    |                     ---
  //      |                    |                      |
  //      ---------------------|        ---           |
  //      |                    |         |            |
  //      |   CalleeSavedReg   |         |            |
  //      | (NumRestores * 16) |         |            |
  //      |                    |         |            |
  //      ---------------------|         |         NumBytes
  //      |                    |     StackSize  (StackAdjustUp)
  //      |   LocalStackSize   |         |            |
  //      | (covering callee   |         |            |
  //      |       args)        |         |            |
  //      |                    |         |            |
  //      ----------------------        ---          ---
  //
  // So NumBytes = StackSize + BytesInStackArgArea - CalleeArgStackSize
  //             = StackSize + ArgumentPopSize
  //
  // AArch64TargetLowering::LowerCall figures out ArgumentPopSize and keeps
  // it as the 2nd argument of AArch64ISD::TC_RETURN.
  NumBytes += ArgumentPopSize;

  unsigned NumRestores = 0;
  // Move past the restores of the callee-saved registers.
  MachineBasicBlock::iterator LastPopI = MBB.getFirstTerminator();
  const MCPhysReg *CSRegs = RegInfo->getCalleeSavedRegs(&MF);
  if (LastPopI != MBB.begin()) {
    do {
      ++NumRestores;
      --LastPopI;
    } while (LastPopI != MBB.begin() && isCSRestore(LastPopI, CSRegs));
    if (!isCSRestore(LastPopI, CSRegs)) {
      ++LastPopI;
      --NumRestores;
    }
  }
  NumBytes -= NumRestores * 16;
  assert(NumBytes >= 0 && "Negative stack allocation size!?");

  if (!hasFP(MF)) {
    // If this was a redzone leaf function, we don't need to restore the
    // stack pointer.
    if (!canUseRedZone(MF))
      emitFrameOffset(MBB, LastPopI, DL, AArch64::SP, AArch64::SP, NumBytes,
                      TII);
    return;
  }

  // Restore the original stack pointer.
  // FIXME: Rather than doing the math here, we should instead just use
  // non-post-indexed loads for the restores if we aren't actually going to
  // be able to save any instructions.
  if (NumBytes || MFI->hasVarSizedObjects())
    emitFrameOffset(MBB, LastPopI, DL, AArch64::SP, AArch64::FP,
                    -(NumRestores - 1) * 16, TII, MachineInstr::NoFlags);
}

/// getFrameIndexOffset - Returns the displacement from the frame register to
/// the stack frame of the specified index.
int AArch64FrameLowering::getFrameIndexOffset(const MachineFunction &MF,
                                              int FI) const {
  unsigned FrameReg;
  return getFrameIndexReference(MF, FI, FrameReg);
}

/// getFrameIndexReference - Provide a base+offset reference to an FI slot for
/// debug info.  It's the same as what we use for resolving the code-gen
/// references for now.  FIXME: This can go wrong when references are
/// SP-relative and simple call frames aren't used.
int AArch64FrameLowering::getFrameIndexReference(const MachineFunction &MF,
                                                 int FI,
                                                 unsigned &FrameReg) const {
  return resolveFrameIndexReference(MF, FI, FrameReg);
}

int AArch64FrameLowering::resolveFrameIndexReference(const MachineFunction &MF,
                                                     int FI, unsigned &FrameReg,
                                                     bool PreferFP) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const AArch64RegisterInfo *RegInfo = static_cast<const AArch64RegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  int FPOffset = MFI->getObjectOffset(FI) + 16;
  int Offset = MFI->getObjectOffset(FI) + MFI->getStackSize();
  bool isFixed = MFI->isFixedObjectIndex(FI);

  // Use frame pointer to reference fixed objects. Use it for locals if
  // there are VLAs or a dynamically realigned SP (and thus the SP isn't
  // reliable as a base). Make sure useFPForScavengingIndex() does the
  // right thing for the emergency spill slot.
  bool UseFP = false;
  if (AFI->hasStackFrame()) {
    // Note: Keeping the following as multiple 'if' statements rather than
    // merging to a single expression for readability.
    //
    // Argument access should always use the FP.
    if (isFixed) {
      UseFP = hasFP(MF);
    } else if (hasFP(MF) && !RegInfo->hasBasePointer(MF) &&
               !RegInfo->needsStackRealignment(MF)) {
      // Use SP or FP, whichever gives us the best chance of the offset
      // being in range for direct access. If the FPOffset is positive,
      // that'll always be best, as the SP will be even further away.
      // If the FPOffset is negative, we have to keep in mind that the
      // available offset range for negative offsets is smaller than for
      // positive ones. If we have variable sized objects, we're stuck with
      // using the FP regardless, though, as the SP offset is unknown
      // and we don't have a base pointer available. If an offset is
      // available via the FP and the SP, use whichever is closest.
      if (PreferFP || MFI->hasVarSizedObjects() || FPOffset >= 0 ||
          (FPOffset >= -256 && Offset > -FPOffset))
        UseFP = true;
    }
  }

  assert((isFixed || !RegInfo->needsStackRealignment(MF) || !UseFP) &&
         "In the presence of dynamic stack pointer realignment, "
         "non-argument objects cannot be accessed through the frame pointer");

  if (UseFP) {
    FrameReg = RegInfo->getFrameRegister(MF);
    return FPOffset;
  }

  // Use the base pointer if we have one.
  if (RegInfo->hasBasePointer(MF))
    FrameReg = RegInfo->getBaseRegister();
  else {
    FrameReg = AArch64::SP;
    // If we're using the red zone for this function, the SP won't actually
    // be adjusted, so the offsets will be negative. They're also all
    // within range of the signed 9-bit immediate instructions.
    if (canUseRedZone(MF))
      Offset -= AFI->getLocalStackSize();
  }

  return Offset;
}

static unsigned getPrologueDeath(MachineFunction &MF, unsigned Reg) {
  if (Reg != AArch64::LR)
    return getKillRegState(true);

  // LR maybe referred to later by an @llvm.returnaddress intrinsic.
  bool LRLiveIn = MF.getRegInfo().isLiveIn(AArch64::LR);
  bool LRKill = !(LRLiveIn && MF.getFrameInfo()->isReturnAddressTaken());
  return getKillRegState(LRKill);
}

bool AArch64FrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    const std::vector<CalleeSavedInfo> &CSI,
    const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  unsigned Count = CSI.size();
  DebugLoc DL;
  assert((Count & 1) == 0 && "Odd number of callee-saved regs to spill!");

  if (MI != MBB.end())
    DL = MI->getDebugLoc();

  for (unsigned i = 0; i < Count; i += 2) {
    unsigned idx = Count - i - 2;
    unsigned Reg1 = CSI[idx].getReg();
    unsigned Reg2 = CSI[idx + 1].getReg();
    // GPRs and FPRs are saved in pairs of 64-bit regs. We expect the CSI
    // list to come in sorted by frame index so that we can issue the store
    // pair instructions directly. Assert if we see anything otherwise.
    //
    // The order of the registers in the list is controlled by
    // getCalleeSavedRegs(), so they will always be in-order, as well.
    assert(CSI[idx].getFrameIdx() + 1 == CSI[idx + 1].getFrameIdx() &&
           "Out of order callee saved regs!");
    unsigned StrOpc;
    assert((Count & 1) == 0 && "Odd number of callee-saved regs to spill!");
    assert((i & 1) == 0 && "Odd index for callee-saved reg spill!");
    // Issue sequence of non-sp increment and pi sp spills for cs regs. The
    // first spill is a pre-increment that allocates the stack.
    // For example:
    //    stp     x22, x21, [sp, #-48]!   // addImm(-6)
    //    stp     x20, x19, [sp, #16]    // addImm(+2)
    //    stp     fp, lr, [sp, #32]      // addImm(+4)
    // Rationale: This sequence saves uop updates compared to a sequence of
    // pre-increment spills like stp xi,xj,[sp,#-16]!
    // Note: Similar rational and sequence for restores in epilog.
    if (AArch64::GPR64RegClass.contains(Reg1)) {
      assert(AArch64::GPR64RegClass.contains(Reg2) &&
             "Expected GPR64 callee-saved register pair!");
      // For first spill use pre-increment store.
      if (i == 0)
        StrOpc = AArch64::STPXpre;
      else
        StrOpc = AArch64::STPXi;
    } else if (AArch64::FPR64RegClass.contains(Reg1)) {
      assert(AArch64::FPR64RegClass.contains(Reg2) &&
             "Expected FPR64 callee-saved register pair!");
      // For first spill use pre-increment store.
      if (i == 0)
        StrOpc = AArch64::STPDpre;
      else
        StrOpc = AArch64::STPDi;
    } else
      llvm_unreachable("Unexpected callee saved register!");
    DEBUG(dbgs() << "CSR spill: (" << TRI->getName(Reg1) << ", "
                 << TRI->getName(Reg2) << ") -> fi#(" << CSI[idx].getFrameIdx()
                 << ", " << CSI[idx + 1].getFrameIdx() << ")\n");
    // Compute offset: i = 0 => offset = -Count;
    //                 i = 2 => offset = -(Count - 2) + Count = 2 = i; etc.
    const int Offset = (i == 0) ? -Count : i;
    assert((Offset >= -64 && Offset <= 63) &&
           "Offset out of bounds for STP immediate");
    MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII.get(StrOpc));
    if (StrOpc == AArch64::STPDpre || StrOpc == AArch64::STPXpre)
      MIB.addReg(AArch64::SP, RegState::Define);

    MBB.addLiveIn(Reg1);
    MBB.addLiveIn(Reg2);
    MIB.addReg(Reg2, getPrologueDeath(MF, Reg2))
        .addReg(Reg1, getPrologueDeath(MF, Reg1))
        .addReg(AArch64::SP)
        .addImm(Offset) // [sp, #offset * 8], where factor * 8 is implicit
        .setMIFlag(MachineInstr::FrameSetup);
  }
  return true;
}

bool AArch64FrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    const std::vector<CalleeSavedInfo> &CSI,
    const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  unsigned Count = CSI.size();
  DebugLoc DL;
  assert((Count & 1) == 0 && "Odd number of callee-saved regs to spill!");

  if (MI != MBB.end())
    DL = MI->getDebugLoc();

  for (unsigned i = 0; i < Count; i += 2) {
    unsigned Reg1 = CSI[i].getReg();
    unsigned Reg2 = CSI[i + 1].getReg();
    // GPRs and FPRs are saved in pairs of 64-bit regs. We expect the CSI
    // list to come in sorted by frame index so that we can issue the store
    // pair instructions directly. Assert if we see anything otherwise.
    assert(CSI[i].getFrameIdx() + 1 == CSI[i + 1].getFrameIdx() &&
           "Out of order callee saved regs!");
    // Issue sequence of non-sp increment and sp-pi restores for cs regs. Only
    // the last load is sp-pi post-increment and de-allocates the stack:
    // For example:
    //    ldp     fp, lr, [sp, #32]       // addImm(+4)
    //    ldp     x20, x19, [sp, #16]     // addImm(+2)
    //    ldp     x22, x21, [sp], #48     // addImm(+6)
    // Note: see comment in spillCalleeSavedRegisters()
    unsigned LdrOpc;

    assert((Count & 1) == 0 && "Odd number of callee-saved regs to spill!");
    assert((i & 1) == 0 && "Odd index for callee-saved reg spill!");
    if (AArch64::GPR64RegClass.contains(Reg1)) {
      assert(AArch64::GPR64RegClass.contains(Reg2) &&
             "Expected GPR64 callee-saved register pair!");
      if (i == Count - 2)
        LdrOpc = AArch64::LDPXpost;
      else
        LdrOpc = AArch64::LDPXi;
    } else if (AArch64::FPR64RegClass.contains(Reg1)) {
      assert(AArch64::FPR64RegClass.contains(Reg2) &&
             "Expected FPR64 callee-saved register pair!");
      if (i == Count - 2)
        LdrOpc = AArch64::LDPDpost;
      else
        LdrOpc = AArch64::LDPDi;
    } else
      llvm_unreachable("Unexpected callee saved register!");
    DEBUG(dbgs() << "CSR restore: (" << TRI->getName(Reg1) << ", "
                 << TRI->getName(Reg2) << ") -> fi#(" << CSI[i].getFrameIdx()
                 << ", " << CSI[i + 1].getFrameIdx() << ")\n");

    // Compute offset: i = 0 => offset = Count - 2; i = 2 => offset = Count - 4;
    // etc.
    const int Offset = (i == Count - 2) ? Count : Count - i - 2;
    assert((Offset >= -64 && Offset <= 63) &&
           "Offset out of bounds for LDP immediate");
    MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII.get(LdrOpc));
    if (LdrOpc == AArch64::LDPXpost || LdrOpc == AArch64::LDPDpost)
      MIB.addReg(AArch64::SP, RegState::Define);

    MIB.addReg(Reg2, getDefRegState(true))
        .addReg(Reg1, getDefRegState(true))
        .addReg(AArch64::SP)
        .addImm(Offset); // [sp], #offset * 8  or [sp, #offset * 8]
                         // where the factor * 8 is implicit
  }
  return true;
}

void AArch64FrameLowering::determineCalleeSaves(MachineFunction &MF,
                                                BitVector &SavedRegs,
                                                RegScavenger *RS) const {
  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction()->getCallingConv() == CallingConv::GHC)
    return;

  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  const AArch64RegisterInfo *RegInfo = static_cast<const AArch64RegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  SmallVector<unsigned, 4> UnspilledCSGPRs;
  SmallVector<unsigned, 4> UnspilledCSFPRs;

  // The frame record needs to be created by saving the appropriate registers
  if (hasFP(MF)) {
    SavedRegs.set(AArch64::FP);
    SavedRegs.set(AArch64::LR);
  }

  // Spill the BasePtr if it's used. Do this first thing so that the
  // getCalleeSavedRegs() below will get the right answer.
  if (RegInfo->hasBasePointer(MF))
    SavedRegs.set(RegInfo->getBaseRegister());

  if (RegInfo->needsStackRealignment(MF) && !RegInfo->hasBasePointer(MF))
    SavedRegs.set(AArch64::X9);

  // If any callee-saved registers are used, the frame cannot be eliminated.
  unsigned NumGPRSpilled = 0;
  unsigned NumFPRSpilled = 0;
  bool ExtraCSSpill = false;
  bool CanEliminateFrame = true;
  DEBUG(dbgs() << "*** processFunctionBeforeCalleeSavedScan\nUsed CSRs:");
  const MCPhysReg *CSRegs = RegInfo->getCalleeSavedRegs(&MF);

  // Check pairs of consecutive callee-saved registers.
  for (unsigned i = 0; CSRegs[i]; i += 2) {
    assert(CSRegs[i + 1] && "Odd number of callee-saved registers!");

    const unsigned OddReg = CSRegs[i];
    const unsigned EvenReg = CSRegs[i + 1];
    assert((AArch64::GPR64RegClass.contains(OddReg) &&
            AArch64::GPR64RegClass.contains(EvenReg)) ^
               (AArch64::FPR64RegClass.contains(OddReg) &&
                AArch64::FPR64RegClass.contains(EvenReg)) &&
           "Register class mismatch!");

    const bool OddRegUsed = SavedRegs.test(OddReg);
    const bool EvenRegUsed = SavedRegs.test(EvenReg);

    // Early exit if none of the registers in the register pair is actually
    // used.
    if (!OddRegUsed && !EvenRegUsed) {
      if (AArch64::GPR64RegClass.contains(OddReg)) {
        UnspilledCSGPRs.push_back(OddReg);
        UnspilledCSGPRs.push_back(EvenReg);
      } else {
        UnspilledCSFPRs.push_back(OddReg);
        UnspilledCSFPRs.push_back(EvenReg);
      }
      continue;
    }

    unsigned Reg = AArch64::NoRegister;
    // If only one of the registers of the register pair is used, make sure to
    // mark the other one as used as well.
    if (OddRegUsed ^ EvenRegUsed) {
      // Find out which register is the additional spill.
      Reg = OddRegUsed ? EvenReg : OddReg;
      SavedRegs.set(Reg);
    }

    DEBUG(dbgs() << ' ' << PrintReg(OddReg, RegInfo));
    DEBUG(dbgs() << ' ' << PrintReg(EvenReg, RegInfo));

    assert(((OddReg == AArch64::LR && EvenReg == AArch64::FP) ||
            (RegInfo->getEncodingValue(OddReg) + 1 ==
             RegInfo->getEncodingValue(EvenReg))) &&
           "Register pair of non-adjacent registers!");
    if (AArch64::GPR64RegClass.contains(OddReg)) {
      NumGPRSpilled += 2;
      // If it's not a reserved register, we can use it in lieu of an
      // emergency spill slot for the register scavenger.
      // FIXME: It would be better to instead keep looking and choose another
      // unspilled register that isn't reserved, if there is one.
      if (Reg != AArch64::NoRegister && !RegInfo->isReservedReg(MF, Reg))
        ExtraCSSpill = true;
    } else
      NumFPRSpilled += 2;

    CanEliminateFrame = false;
  }

  // FIXME: Set BigStack if any stack slot references may be out of range.
  // For now, just conservatively guestimate based on unscaled indexing
  // range. We'll end up allocating an unnecessary spill slot a lot, but
  // realistically that's not a big deal at this stage of the game.
  // The CSR spill slots have not been allocated yet, so estimateStackSize
  // won't include them.
  MachineFrameInfo *MFI = MF.getFrameInfo();
  unsigned CFSize =
      MFI->estimateStackSize(MF) + 8 * (NumGPRSpilled + NumFPRSpilled);
  DEBUG(dbgs() << "Estimated stack frame size: " << CFSize << " bytes.\n");
  bool BigStack = (CFSize >= 256);
  if (BigStack || !CanEliminateFrame || RegInfo->cannotEliminateFrame(MF))
    AFI->setHasStackFrame(true);

  // Estimate if we might need to scavenge a register at some point in order
  // to materialize a stack offset. If so, either spill one additional
  // callee-saved register or reserve a special spill slot to facilitate
  // register scavenging. If we already spilled an extra callee-saved register
  // above to keep the number of spills even, we don't need to do anything else
  // here.
  if (BigStack && !ExtraCSSpill) {

    // If we're adding a register to spill here, we have to add two of them
    // to keep the number of regs to spill even.
    assert(((UnspilledCSGPRs.size() & 1) == 0) && "Odd number of registers!");
    unsigned Count = 0;
    while (!UnspilledCSGPRs.empty() && Count < 2) {
      unsigned Reg = UnspilledCSGPRs.back();
      UnspilledCSGPRs.pop_back();
      DEBUG(dbgs() << "Spilling " << PrintReg(Reg, RegInfo)
                   << " to get a scratch register.\n");
      SavedRegs.set(Reg);
      ExtraCSSpill = true;
      ++Count;
    }

    // If we didn't find an extra callee-saved register to spill, create
    // an emergency spill slot.
    if (!ExtraCSSpill) {
      const TargetRegisterClass *RC = &AArch64::GPR64RegClass;
      int FI = MFI->CreateStackObject(RC->getSize(), RC->getAlignment(), false);
      RS->addScavengingFrameIndex(FI);
      DEBUG(dbgs() << "No available CS registers, allocated fi#" << FI
                   << " as the emergency spill slot.\n");
    }
  }
}
