//===- AArch64FrameLowering.cpp - AArch64 Frame Lowering -------*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// | (Win64 only) varargs from reg     |
// |                                   |
// |-----------------------------------|
// |                                   |
// | callee-saved gpr registers        | <--.
// |                                   |    | On Darwin platforms these
// |- - - - - - - - - - - - - - - - - -|    | callee saves are swapped,
// | prev_lr                           |    | (frame record first)
// | prev_fp                           | <--'
// | async context if needed           |
// | (a.k.a. "frame record")           |
// |-----------------------------------| <- fp(=x29)
// |                                   |
// | callee-saved fp/simd/SVE regs     |
// |                                   |
// |-----------------------------------|
// |                                   |
// |        SVE stack objects          |
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
// * A base pointer is definitely needed when there are both VLAs and local
//   variables with more-than-default alignment requirements.
// * A frame pointer is definitely needed when there are local variables with
//   more-than-default alignment requirements.
//
// For Darwin platforms the frame-record (fp, lr) is stored at the top of the
// callee-saved area, since the unwind encoding does not allow for encoding
// this dynamically and existing tools depend on this layout. For other
// platforms, the frame-record is stored at the bottom of the (gpr) callee-saved
// area to allow SVE stack objects (allocated directly below the callee-saves,
// if available) to be accessed directly from the framepointer.
// The SVE spill/fill instructions have VL-scaled addressing modes such
// as:
//    ldr z8, [fp, #-7 mul vl]
// For SVE the size of the vector length (VL) is not known at compile-time, so
// '#-7 mul vl' is an offset that can only be evaluated at runtime. With this
// layout, we don't need to add an unscaled offset to the framepointer before
// accessing the SVE object in the frame.
//
// In some cases when a base pointer is not strictly needed, it is generated
// anyway when offsets from the frame pointer to access local variables become
// so large that the offset can't be encoded in the immediate fields of loads
// or stores.
//
// Outgoing function arguments must be at the bottom of the stack frame when
// calling another function. If we do not have variable-sized stack objects, we
// can allocate a "reserved call frame" area at the bottom of the local
// variable area, large enough for all outgoing calls. If we do have VLAs, then
// the stack pointer must be decremented and incremented around each call to
// make space for the arguments below the VLAs.
//
// FIXME: also explain the redzone concept.
//
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
//
//===----------------------------------------------------------------------===//

#include "AArch64FrameLowering.h"
#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64RegisterInfo.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "frame-info"

static cl::opt<bool> EnableRedZone("aarch64-redzone",
                                   cl::desc("enable use of redzone on AArch64"),
                                   cl::init(false), cl::Hidden);

static cl::opt<bool>
    ReverseCSRRestoreSeq("reverse-csr-restore-seq",
                         cl::desc("reverse the CSR restore sequence"),
                         cl::init(false), cl::Hidden);

static cl::opt<bool> StackTaggingMergeSetTag(
    "stack-tagging-merge-settag",
    cl::desc("merge settag instruction in function epilog"), cl::init(true),
    cl::Hidden);

static cl::opt<bool> OrderFrameObjects("aarch64-order-frame-objects",
                                       cl::desc("sort stack allocations"),
                                       cl::init(true), cl::Hidden);

cl::opt<bool> EnableHomogeneousPrologEpilog(
    "homogeneous-prolog-epilog", cl::init(false), cl::ZeroOrMore, cl::Hidden,
    cl::desc("Emit homogeneous prologue and epilogue for the size "
             "optimization (default = off)"));

STATISTIC(NumRedZoneFunctions, "Number of functions using red zone");

/// Returns how much of the incoming argument stack area (in bytes) we should
/// clean up in an epilogue. For the C calling convention this will be 0, for
/// guaranteed tail call conventions it can be positive (a normal return or a
/// tail call to a function that uses less stack space for arguments) or
/// negative (for a tail call to a function that needs more stack space than us
/// for arguments).
static int64_t getArgumentStackToRestore(MachineFunction &MF,
                                         MachineBasicBlock &MBB) {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  bool IsTailCallReturn = false;
  if (MBB.end() != MBBI) {
    unsigned RetOpcode = MBBI->getOpcode();
    IsTailCallReturn = RetOpcode == AArch64::TCRETURNdi ||
                       RetOpcode == AArch64::TCRETURNri ||
                       RetOpcode == AArch64::TCRETURNriBTI;
  }
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();

  int64_t ArgumentPopSize = 0;
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

  return ArgumentPopSize;
}

static bool produceCompactUnwindFrame(MachineFunction &MF);
static bool needsWinCFI(const MachineFunction &MF);
static StackOffset getSVEStackSize(const MachineFunction &MF);
static bool needsShadowCallStackPrologueEpilogue(MachineFunction &MF);

/// Returns true if a homogeneous prolog or epilog code can be emitted
/// for the size optimization. If possible, a frame helper call is injected.
/// When Exit block is given, this check is for epilog.
bool AArch64FrameLowering::homogeneousPrologEpilog(
    MachineFunction &MF, MachineBasicBlock *Exit) const {
  if (!MF.getFunction().hasMinSize())
    return false;
  if (!EnableHomogeneousPrologEpilog)
    return false;
  if (ReverseCSRRestoreSeq)
    return false;
  if (EnableRedZone)
    return false;

  // TODO: Window is supported yet.
  if (needsWinCFI(MF))
    return false;
  // TODO: SVE is not supported yet.
  if (getSVEStackSize(MF))
    return false;

  // Bail on stack adjustment needed on return for simplicity.
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  if (MFI.hasVarSizedObjects() || RegInfo->hasStackRealignment(MF))
    return false;
  if (Exit && getArgumentStackToRestore(MF, *Exit))
    return false;

  return true;
}

/// Returns true if CSRs should be paired.
bool AArch64FrameLowering::producePairRegisters(MachineFunction &MF) const {
  return produceCompactUnwindFrame(MF) || homogeneousPrologEpilog(MF);
}

/// This is the biggest offset to the stack pointer we can encode in aarch64
/// instructions (without using a separate calculation and a temp register).
/// Note that the exception here are vector stores/loads which cannot encode any
/// displacements (see estimateRSStackSizeLimit(), isAArch64FrameOffsetLegal()).
static const unsigned DefaultSafeSPDisplacement = 255;

/// Look at each instruction that references stack frames and return the stack
/// size limit beyond which some of these instructions will require a scratch
/// register during their expansion later.
static unsigned estimateRSStackSizeLimit(MachineFunction &MF) {
  // FIXME: For now, just conservatively guestimate based on unscaled indexing
  // range. We'll end up allocating an unnecessary spill slot a lot, but
  // realistically that's not a big deal at this stage of the game.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr() || MI.isPseudo() ||
          MI.getOpcode() == AArch64::ADDXri ||
          MI.getOpcode() == AArch64::ADDSXri)
        continue;

      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isFI())
          continue;

        StackOffset Offset;
        if (isAArch64FrameOffsetLegal(MI, Offset, nullptr, nullptr, nullptr) ==
            AArch64FrameOffsetCannotUpdate)
          return 0;
      }
    }
  }
  return DefaultSafeSPDisplacement;
}

TargetStackID::Value
AArch64FrameLowering::getStackIDForScalableVectors() const {
  return TargetStackID::ScalableVector;
}

/// Returns the size of the fixed object area (allocated next to sp on entry)
/// On Win64 this may include a var args area and an UnwindHelp object for EH.
static unsigned getFixedObjectSize(const MachineFunction &MF,
                                   const AArch64FunctionInfo *AFI, bool IsWin64,
                                   bool IsFunclet) {
  if (!IsWin64 || IsFunclet) {
    return AFI->getTailCallReservedStack();
  } else {
    if (AFI->getTailCallReservedStack() != 0)
      report_fatal_error("cannot generate ABI-changing tail call for Win64");
    // Var args are stored here in the primary function.
    const unsigned VarArgsArea = AFI->getVarArgsGPRSize();
    // To support EH funclets we allocate an UnwindHelp object
    const unsigned UnwindHelpObject = (MF.hasEHFunclets() ? 8 : 0);
    return alignTo(VarArgsArea + UnwindHelpObject, 16);
  }
}

/// Returns the size of the entire SVE stackframe (calleesaves + spills).
static StackOffset getSVEStackSize(const MachineFunction &MF) {
  const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  return StackOffset::getScalable((int64_t)AFI->getStackSizeSVE());
}

bool AArch64FrameLowering::canUseRedZone(const MachineFunction &MF) const {
  if (!EnableRedZone)
    return false;

  // Don't use the red zone if the function explicitly asks us not to.
  // This is typically used for kernel code.
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const unsigned RedZoneSize =
      Subtarget.getTargetLowering()->getRedZoneSize(MF.getFunction());
  if (!RedZoneSize)
    return false;

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  uint64_t NumBytes = AFI->getLocalStackSize();

  return !(MFI.hasCalls() || hasFP(MF) || NumBytes > RedZoneSize ||
           getSVEStackSize(MF));
}

/// hasFP - Return true if the specified function should have a dedicated frame
/// pointer register.
bool AArch64FrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  // Win64 EH requires a frame pointer if funclets are present, as the locals
  // are accessed off the frame pointer in both the parent function and the
  // funclets.
  if (MF.hasEHFunclets())
    return true;
  // Retain behavior of always omitting the FP for leaf functions when possible.
  if (MF.getTarget().Options.DisableFramePointerElim(MF))
    return true;
  if (MFI.hasVarSizedObjects() || MFI.isFrameAddressTaken() ||
      MFI.hasStackMap() || MFI.hasPatchPoint() ||
      RegInfo->hasStackRealignment(MF))
    return true;
  // With large callframes around we may need to use FP to access the scavenging
  // emergency spillslot.
  //
  // Unfortunately some calls to hasFP() like machine verifier ->
  // getReservedReg() -> hasFP in the middle of global isel are too early
  // to know the max call frame size. Hopefully conservatively returning "true"
  // in those cases is fine.
  // DefaultSafeSPDisplacement is fine as we only emergency spill GP regs.
  if (!MFI.isMaxCallFrameSizeComputed() ||
      MFI.getMaxCallFrameSize() > DefaultSafeSPDisplacement)
    return true;

  return false;
}

/// hasReservedCallFrame - Under normal circumstances, when a frame pointer is
/// not required, we reserve argument space for call sites in the function
/// immediately on entry to the current function.  This eliminates the need for
/// add/sub sp brackets around call sites.  Returns true if the call frame is
/// included as part of the stack frame.
bool
AArch64FrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  return !MF.getFrameInfo().hasVarSizedObjects();
}

MachineBasicBlock::iterator AArch64FrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  const AArch64InstrInfo *TII =
      static_cast<const AArch64InstrInfo *>(MF.getSubtarget().getInstrInfo());
  DebugLoc DL = I->getDebugLoc();
  unsigned Opc = I->getOpcode();
  bool IsDestroy = Opc == TII->getCallFrameDestroyOpcode();
  uint64_t CalleePopAmount = IsDestroy ? I->getOperand(1).getImm() : 0;

  if (!hasReservedCallFrame(MF)) {
    int64_t Amount = I->getOperand(0).getImm();
    Amount = alignTo(Amount, getStackAlign());
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
      // Most call frames will be allocated at the start of a function so
      // this is OK, but it is a limitation that needs dealing with.
      assert(Amount > -0xffffff && Amount < 0xffffff && "call frame too large");
      emitFrameOffset(MBB, I, DL, AArch64::SP, AArch64::SP,
                      StackOffset::getFixed(Amount), TII);
    }
  } else if (CalleePopAmount != 0) {
    // If the calling convention demands that the callee pops arguments from the
    // stack, we want to add it back if we have a reserved call frame.
    assert(CalleePopAmount < 0xffffff && "call frame too large");
    emitFrameOffset(MBB, I, DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(-(int64_t)CalleePopAmount), TII);
  }
  return MBB.erase(I);
}

void AArch64FrameLowering::emitCalleeSavedGPRLocations(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  if (CSI.empty())
    return;

  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetRegisterInfo &TRI = *STI.getRegisterInfo();
  const TargetInstrInfo &TII = *STI.getInstrInfo();
  DebugLoc DL = MBB.findDebugLoc(MBBI);

  for (const auto &Info : CSI) {
    if (MFI.getStackID(Info.getFrameIdx()) == TargetStackID::ScalableVector)
      continue;

    assert(!Info.isSpilledToReg() && "Spilling to registers not implemented");
    unsigned DwarfReg = TRI.getDwarfRegNum(Info.getReg(), true);

    int64_t Offset =
        MFI.getObjectOffset(Info.getFrameIdx()) - getOffsetOfLocalArea();
    unsigned CFIIndex = MF.addFrameInst(
        MCCFIInstruction::createOffset(nullptr, DwarfReg, Offset));
    BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlags(MachineInstr::FrameSetup);
  }
}

void AArch64FrameLowering::emitCalleeSavedSVELocations(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  // Add callee saved registers to move list.
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  if (CSI.empty())
    return;

  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetRegisterInfo &TRI = *STI.getRegisterInfo();
  const TargetInstrInfo &TII = *STI.getInstrInfo();
  DebugLoc DL = MBB.findDebugLoc(MBBI);
  AArch64FunctionInfo &AFI = *MF.getInfo<AArch64FunctionInfo>();

  for (const auto &Info : CSI) {
    if (!(MFI.getStackID(Info.getFrameIdx()) == TargetStackID::ScalableVector))
      continue;

    // Not all unwinders may know about SVE registers, so assume the lowest
    // common demoninator.
    assert(!Info.isSpilledToReg() && "Spilling to registers not implemented");
    unsigned Reg = Info.getReg();
    if (!static_cast<const AArch64RegisterInfo &>(TRI).regNeedsCFI(Reg, Reg))
      continue;

    StackOffset Offset =
        StackOffset::getScalable(MFI.getObjectOffset(Info.getFrameIdx())) -
        StackOffset::getFixed(AFI.getCalleeSavedStackSize(MFI));

    unsigned CFIIndex = MF.addFrameInst(createCFAOffset(TRI, Reg, Offset));
    BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlags(MachineInstr::FrameSetup);
  }
}

void AArch64FrameLowering::emitCalleeSavedFrameMoves(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  emitCalleeSavedGPRLocations(MBB, MBBI);
  emitCalleeSavedSVELocations(MBB, MBBI);
}

static void insertCFISameValue(const MCInstrDesc &Desc, MachineFunction &MF,
                               MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator InsertPt,
                               unsigned DwarfReg) {
  unsigned CFIIndex =
      MF.addFrameInst(MCCFIInstruction::createSameValue(nullptr, DwarfReg));
  BuildMI(MBB, InsertPt, DebugLoc(), Desc).addCFIIndex(CFIIndex);
}

void AArch64FrameLowering::resetCFIToInitialState(
    MachineBasicBlock &MBB) const {

  MachineFunction &MF = *MBB.getParent();
  const auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  const auto &TRI =
      static_cast<const AArch64RegisterInfo &>(*Subtarget.getRegisterInfo());
  const auto &MFI = *MF.getInfo<AArch64FunctionInfo>();

  const MCInstrDesc &CFIDesc = TII.get(TargetOpcode::CFI_INSTRUCTION);
  DebugLoc DL;

  // Reset the CFA to `SP + 0`.
  MachineBasicBlock::iterator InsertPt = MBB.begin();
  unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::cfiDefCfa(
      nullptr, TRI.getDwarfRegNum(AArch64::SP, true), 0));
  BuildMI(MBB, InsertPt, DL, CFIDesc).addCFIIndex(CFIIndex);

  // Flip the RA sign state.
  if (MFI.shouldSignReturnAddress()) {
    CFIIndex = MF.addFrameInst(MCCFIInstruction::createNegateRAState(nullptr));
    BuildMI(MBB, InsertPt, DL, CFIDesc).addCFIIndex(CFIIndex);
  }

  // Shadow call stack uses X18, reset it.
  if (needsShadowCallStackPrologueEpilogue(MF))
    insertCFISameValue(CFIDesc, MF, MBB, InsertPt,
                       TRI.getDwarfRegNum(AArch64::X18, true));

  // Emit .cfi_same_value for callee-saved registers.
  const std::vector<CalleeSavedInfo> &CSI =
      MF.getFrameInfo().getCalleeSavedInfo();
  for (const auto &Info : CSI) {
    unsigned Reg = Info.getReg();
    if (!TRI.regNeedsCFI(Reg, Reg))
      continue;
    insertCFISameValue(CFIDesc, MF, MBB, InsertPt,
                       TRI.getDwarfRegNum(Reg, true));
  }
}

static void emitCalleeSavedRestores(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    bool SVE) {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  if (CSI.empty())
    return;

  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetRegisterInfo &TRI = *STI.getRegisterInfo();
  const TargetInstrInfo &TII = *STI.getInstrInfo();
  DebugLoc DL = MBB.findDebugLoc(MBBI);

  for (const auto &Info : CSI) {
    if (SVE !=
        (MFI.getStackID(Info.getFrameIdx()) == TargetStackID::ScalableVector))
      continue;

    unsigned Reg = Info.getReg();
    if (SVE &&
        !static_cast<const AArch64RegisterInfo &>(TRI).regNeedsCFI(Reg, Reg))
      continue;

    unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createRestore(
        nullptr, TRI.getDwarfRegNum(Info.getReg(), true)));
    BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlags(MachineInstr::FrameDestroy);
  }
}

void AArch64FrameLowering::emitCalleeSavedGPRRestores(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  emitCalleeSavedRestores(MBB, MBBI, false);
}

void AArch64FrameLowering::emitCalleeSavedSVERestores(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  emitCalleeSavedRestores(MBB, MBBI, true);
}

// Find a scratch register that we can use at the start of the prologue to
// re-align the stack pointer.  We avoid using callee-save registers since they
// may appear to be free when this is called from canUseAsPrologue (during
// shrink wrapping), but then no longer be free when this is called from
// emitPrologue.
//
// FIXME: This is a bit conservative, since in the above case we could use one
// of the callee-save registers as a scratch temp to re-align the stack pointer,
// but we would then have to make sure that we were in fact saving at least one
// callee-save register in the prologue, which is additional complexity that
// doesn't seem worth the benefit.
static unsigned findScratchNonCalleeSaveRegister(MachineBasicBlock *MBB) {
  MachineFunction *MF = MBB->getParent();

  // If MBB is an entry block, use X9 as the scratch register
  if (&MF->front() == MBB)
    return AArch64::X9;

  const AArch64Subtarget &Subtarget = MF->getSubtarget<AArch64Subtarget>();
  const AArch64RegisterInfo &TRI = *Subtarget.getRegisterInfo();
  LivePhysRegs LiveRegs(TRI);
  LiveRegs.addLiveIns(*MBB);

  // Mark callee saved registers as used so we will not choose them.
  const MCPhysReg *CSRegs = MF->getRegInfo().getCalleeSavedRegs();
  for (unsigned i = 0; CSRegs[i]; ++i)
    LiveRegs.addReg(CSRegs[i]);

  // Prefer X9 since it was historically used for the prologue scratch reg.
  const MachineRegisterInfo &MRI = MF->getRegInfo();
  if (LiveRegs.available(MRI, AArch64::X9))
    return AArch64::X9;

  for (unsigned Reg : AArch64::GPR64RegClass) {
    if (LiveRegs.available(MRI, Reg))
      return Reg;
  }
  return AArch64::NoRegister;
}

bool AArch64FrameLowering::canUseAsPrologue(
    const MachineBasicBlock &MBB) const {
  const MachineFunction *MF = MBB.getParent();
  MachineBasicBlock *TmpMBB = const_cast<MachineBasicBlock *>(&MBB);
  const AArch64Subtarget &Subtarget = MF->getSubtarget<AArch64Subtarget>();
  const AArch64RegisterInfo *RegInfo = Subtarget.getRegisterInfo();

  // Don't need a scratch register if we're not going to re-align the stack.
  if (!RegInfo->hasStackRealignment(*MF))
    return true;
  // Otherwise, we can use any block as long as it has a scratch register
  // available.
  return findScratchNonCalleeSaveRegister(TmpMBB) != AArch64::NoRegister;
}

static bool windowsRequiresStackProbe(MachineFunction &MF,
                                      uint64_t StackSizeInBytes) {
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  if (!Subtarget.isTargetWindows())
    return false;
  const Function &F = MF.getFunction();
  // TODO: When implementing stack protectors, take that into account
  // for the probe threshold.
  unsigned StackProbeSize = 4096;
  if (F.hasFnAttribute("stack-probe-size"))
    F.getFnAttribute("stack-probe-size")
        .getValueAsString()
        .getAsInteger(0, StackProbeSize);
  return (StackSizeInBytes >= StackProbeSize) &&
         !F.hasFnAttribute("no-stack-arg-probe");
}

static bool needsWinCFI(const MachineFunction &MF) {
  const Function &F = MF.getFunction();
  return MF.getTarget().getMCAsmInfo()->usesWindowsCFI() &&
         F.needsUnwindTableEntry();
}

bool AArch64FrameLowering::shouldCombineCSRLocalStackBump(
    MachineFunction &MF, uint64_t StackBumpBytes) const {
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const AArch64RegisterInfo *RegInfo = Subtarget.getRegisterInfo();
  if (homogeneousPrologEpilog(MF))
    return false;

  if (AFI->getLocalStackSize() == 0)
    return false;

  // For WinCFI, if optimizing for size, prefer to not combine the stack bump
  // (to force a stp with predecrement) to match the packed unwind format,
  // provided that there actually are any callee saved registers to merge the
  // decrement with.
  // This is potentially marginally slower, but allows using the packed
  // unwind format for functions that both have a local area and callee saved
  // registers. Using the packed unwind format notably reduces the size of
  // the unwind info.
  if (needsWinCFI(MF) && AFI->getCalleeSavedStackSize() > 0 &&
      MF.getFunction().hasOptSize())
    return false;

  // 512 is the maximum immediate for stp/ldp that will be used for
  // callee-save save/restores
  if (StackBumpBytes >= 512 || windowsRequiresStackProbe(MF, StackBumpBytes))
    return false;

  if (MFI.hasVarSizedObjects())
    return false;

  if (RegInfo->hasStackRealignment(MF))
    return false;

  // This isn't strictly necessary, but it simplifies things a bit since the
  // current RedZone handling code assumes the SP is adjusted by the
  // callee-save save/restore code.
  if (canUseRedZone(MF))
    return false;

  // When there is an SVE area on the stack, always allocate the
  // callee-saves and spills/locals separately.
  if (getSVEStackSize(MF))
    return false;

  return true;
}

bool AArch64FrameLowering::shouldCombineCSRLocalStackBumpInEpilogue(
    MachineBasicBlock &MBB, unsigned StackBumpBytes) const {
  if (!shouldCombineCSRLocalStackBump(*MBB.getParent(), StackBumpBytes))
    return false;

  if (MBB.empty())
    return true;

  // Disable combined SP bump if the last instruction is an MTE tag store. It
  // is almost always better to merge SP adjustment into those instructions.
  MachineBasicBlock::iterator LastI = MBB.getFirstTerminator();
  MachineBasicBlock::iterator Begin = MBB.begin();
  while (LastI != Begin) {
    --LastI;
    if (LastI->isTransient())
      continue;
    if (!LastI->getFlag(MachineInstr::FrameDestroy))
      break;
  }
  switch (LastI->getOpcode()) {
  case AArch64::STGloop:
  case AArch64::STZGloop:
  case AArch64::STGOffset:
  case AArch64::STZGOffset:
  case AArch64::ST2GOffset:
  case AArch64::STZ2GOffset:
    return false;
  default:
    return true;
  }
  llvm_unreachable("unreachable");
}

// Given a load or a store instruction, generate an appropriate unwinding SEH
// code on Windows.
static MachineBasicBlock::iterator InsertSEH(MachineBasicBlock::iterator MBBI,
                                             const TargetInstrInfo &TII,
                                             MachineInstr::MIFlag Flag) {
  unsigned Opc = MBBI->getOpcode();
  MachineBasicBlock *MBB = MBBI->getParent();
  MachineFunction &MF = *MBB->getParent();
  DebugLoc DL = MBBI->getDebugLoc();
  unsigned ImmIdx = MBBI->getNumOperands() - 1;
  int Imm = MBBI->getOperand(ImmIdx).getImm();
  MachineInstrBuilder MIB;
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const AArch64RegisterInfo *RegInfo = Subtarget.getRegisterInfo();

  switch (Opc) {
  default:
    llvm_unreachable("No SEH Opcode for this instruction");
  case AArch64::LDPDpost:
    Imm = -Imm;
    LLVM_FALLTHROUGH;
  case AArch64::STPDpre: {
    unsigned Reg0 = RegInfo->getSEHRegNum(MBBI->getOperand(1).getReg());
    unsigned Reg1 = RegInfo->getSEHRegNum(MBBI->getOperand(2).getReg());
    MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveFRegP_X))
              .addImm(Reg0)
              .addImm(Reg1)
              .addImm(Imm * 8)
              .setMIFlag(Flag);
    break;
  }
  case AArch64::LDPXpost:
    Imm = -Imm;
    LLVM_FALLTHROUGH;
  case AArch64::STPXpre: {
    Register Reg0 = MBBI->getOperand(1).getReg();
    Register Reg1 = MBBI->getOperand(2).getReg();
    if (Reg0 == AArch64::FP && Reg1 == AArch64::LR)
      MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveFPLR_X))
                .addImm(Imm * 8)
                .setMIFlag(Flag);
    else
      MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveRegP_X))
                .addImm(RegInfo->getSEHRegNum(Reg0))
                .addImm(RegInfo->getSEHRegNum(Reg1))
                .addImm(Imm * 8)
                .setMIFlag(Flag);
    break;
  }
  case AArch64::LDRDpost:
    Imm = -Imm;
    LLVM_FALLTHROUGH;
  case AArch64::STRDpre: {
    unsigned Reg = RegInfo->getSEHRegNum(MBBI->getOperand(1).getReg());
    MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveFReg_X))
              .addImm(Reg)
              .addImm(Imm)
              .setMIFlag(Flag);
    break;
  }
  case AArch64::LDRXpost:
    Imm = -Imm;
    LLVM_FALLTHROUGH;
  case AArch64::STRXpre: {
    unsigned Reg =  RegInfo->getSEHRegNum(MBBI->getOperand(1).getReg());
    MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveReg_X))
              .addImm(Reg)
              .addImm(Imm)
              .setMIFlag(Flag);
    break;
  }
  case AArch64::STPDi:
  case AArch64::LDPDi: {
    unsigned Reg0 =  RegInfo->getSEHRegNum(MBBI->getOperand(0).getReg());
    unsigned Reg1 =  RegInfo->getSEHRegNum(MBBI->getOperand(1).getReg());
    MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveFRegP))
              .addImm(Reg0)
              .addImm(Reg1)
              .addImm(Imm * 8)
              .setMIFlag(Flag);
    break;
  }
  case AArch64::STPXi:
  case AArch64::LDPXi: {
    Register Reg0 = MBBI->getOperand(0).getReg();
    Register Reg1 = MBBI->getOperand(1).getReg();
    if (Reg0 == AArch64::FP && Reg1 == AArch64::LR)
      MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveFPLR))
                .addImm(Imm * 8)
                .setMIFlag(Flag);
    else
      MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveRegP))
                .addImm(RegInfo->getSEHRegNum(Reg0))
                .addImm(RegInfo->getSEHRegNum(Reg1))
                .addImm(Imm * 8)
                .setMIFlag(Flag);
    break;
  }
  case AArch64::STRXui:
  case AArch64::LDRXui: {
    int Reg = RegInfo->getSEHRegNum(MBBI->getOperand(0).getReg());
    MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveReg))
              .addImm(Reg)
              .addImm(Imm * 8)
              .setMIFlag(Flag);
    break;
  }
  case AArch64::STRDui:
  case AArch64::LDRDui: {
    unsigned Reg = RegInfo->getSEHRegNum(MBBI->getOperand(0).getReg());
    MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveFReg))
              .addImm(Reg)
              .addImm(Imm * 8)
              .setMIFlag(Flag);
    break;
  }
  }
  auto I = MBB->insertAfter(MBBI, MIB);
  return I;
}

// Fix up the SEH opcode associated with the save/restore instruction.
static void fixupSEHOpcode(MachineBasicBlock::iterator MBBI,
                           unsigned LocalStackSize) {
  MachineOperand *ImmOpnd = nullptr;
  unsigned ImmIdx = MBBI->getNumOperands() - 1;
  switch (MBBI->getOpcode()) {
  default:
    llvm_unreachable("Fix the offset in the SEH instruction");
  case AArch64::SEH_SaveFPLR:
  case AArch64::SEH_SaveRegP:
  case AArch64::SEH_SaveReg:
  case AArch64::SEH_SaveFRegP:
  case AArch64::SEH_SaveFReg:
    ImmOpnd = &MBBI->getOperand(ImmIdx);
    break;
  }
  if (ImmOpnd)
    ImmOpnd->setImm(ImmOpnd->getImm() + LocalStackSize);
}

// Convert callee-save register save/restore instruction to do stack pointer
// decrement/increment to allocate/deallocate the callee-save stack area by
// converting store/load to use pre/post increment version.
static MachineBasicBlock::iterator convertCalleeSaveRestoreToSPPrePostIncDec(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    const DebugLoc &DL, const TargetInstrInfo *TII, int CSStackSizeInc,
    bool NeedsWinCFI, bool *HasWinCFI, bool EmitCFI,
    MachineInstr::MIFlag FrameFlag = MachineInstr::FrameSetup,
    int CFAOffset = 0) {
  unsigned NewOpc;
  switch (MBBI->getOpcode()) {
  default:
    llvm_unreachable("Unexpected callee-save save/restore opcode!");
  case AArch64::STPXi:
    NewOpc = AArch64::STPXpre;
    break;
  case AArch64::STPDi:
    NewOpc = AArch64::STPDpre;
    break;
  case AArch64::STPQi:
    NewOpc = AArch64::STPQpre;
    break;
  case AArch64::STRXui:
    NewOpc = AArch64::STRXpre;
    break;
  case AArch64::STRDui:
    NewOpc = AArch64::STRDpre;
    break;
  case AArch64::STRQui:
    NewOpc = AArch64::STRQpre;
    break;
  case AArch64::LDPXi:
    NewOpc = AArch64::LDPXpost;
    break;
  case AArch64::LDPDi:
    NewOpc = AArch64::LDPDpost;
    break;
  case AArch64::LDPQi:
    NewOpc = AArch64::LDPQpost;
    break;
  case AArch64::LDRXui:
    NewOpc = AArch64::LDRXpost;
    break;
  case AArch64::LDRDui:
    NewOpc = AArch64::LDRDpost;
    break;
  case AArch64::LDRQui:
    NewOpc = AArch64::LDRQpost;
    break;
  }
  // Get rid of the SEH code associated with the old instruction.
  if (NeedsWinCFI) {
    auto SEH = std::next(MBBI);
    if (AArch64InstrInfo::isSEHInstruction(*SEH))
      SEH->eraseFromParent();
  }

  TypeSize Scale = TypeSize::Fixed(1);
  unsigned Width;
  int64_t MinOffset, MaxOffset;
  bool Success = static_cast<const AArch64InstrInfo *>(TII)->getMemOpInfo(
      NewOpc, Scale, Width, MinOffset, MaxOffset);
  (void)Success;
  assert(Success && "unknown load/store opcode");

  // If the first store isn't right where we want SP then we can't fold the
  // update in so create a normal arithmetic instruction instead.
  MachineFunction &MF = *MBB.getParent();
  if (MBBI->getOperand(MBBI->getNumOperands() - 1).getImm() != 0 ||
      CSStackSizeInc < MinOffset || CSStackSizeInc > MaxOffset) {
    emitFrameOffset(MBB, MBBI, DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(CSStackSizeInc), TII, FrameFlag,
                    false, false, nullptr, EmitCFI,
                    StackOffset::getFixed(CFAOffset));

    return std::prev(MBBI);
  }

  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(NewOpc));
  MIB.addReg(AArch64::SP, RegState::Define);

  // Copy all operands other than the immediate offset.
  unsigned OpndIdx = 0;
  for (unsigned OpndEnd = MBBI->getNumOperands() - 1; OpndIdx < OpndEnd;
       ++OpndIdx)
    MIB.add(MBBI->getOperand(OpndIdx));

  assert(MBBI->getOperand(OpndIdx).getImm() == 0 &&
         "Unexpected immediate offset in first/last callee-save save/restore "
         "instruction!");
  assert(MBBI->getOperand(OpndIdx - 1).getReg() == AArch64::SP &&
         "Unexpected base register in callee-save save/restore instruction!");
  assert(CSStackSizeInc % Scale == 0);
  MIB.addImm(CSStackSizeInc / (int)Scale);

  MIB.setMIFlags(MBBI->getFlags());
  MIB.setMemRefs(MBBI->memoperands());

  // Generate a new SEH code that corresponds to the new instruction.
  if (NeedsWinCFI) {
    *HasWinCFI = true;
    InsertSEH(*MIB, *TII, FrameFlag);
  }

  if (EmitCFI) {
    unsigned CFIIndex = MF.addFrameInst(
        MCCFIInstruction::cfiDefCfaOffset(nullptr, CFAOffset - CSStackSizeInc));
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlags(FrameFlag);
  }

  return std::prev(MBB.erase(MBBI));
}

// Fixup callee-save register save/restore instructions to take into account
// combined SP bump by adding the local stack size to the stack offsets.
static void fixupCalleeSaveRestoreStackOffset(MachineInstr &MI,
                                              uint64_t LocalStackSize,
                                              bool NeedsWinCFI,
                                              bool *HasWinCFI) {
  if (AArch64InstrInfo::isSEHInstruction(MI))
    return;

  unsigned Opc = MI.getOpcode();
  unsigned Scale;
  switch (Opc) {
  case AArch64::STPXi:
  case AArch64::STRXui:
  case AArch64::STPDi:
  case AArch64::STRDui:
  case AArch64::LDPXi:
  case AArch64::LDRXui:
  case AArch64::LDPDi:
  case AArch64::LDRDui:
    Scale = 8;
    break;
  case AArch64::STPQi:
  case AArch64::STRQui:
  case AArch64::LDPQi:
  case AArch64::LDRQui:
    Scale = 16;
    break;
  default:
    llvm_unreachable("Unexpected callee-save save/restore opcode!");
  }

  unsigned OffsetIdx = MI.getNumExplicitOperands() - 1;
  assert(MI.getOperand(OffsetIdx - 1).getReg() == AArch64::SP &&
         "Unexpected base register in callee-save save/restore instruction!");
  // Last operand is immediate offset that needs fixing.
  MachineOperand &OffsetOpnd = MI.getOperand(OffsetIdx);
  // All generated opcodes have scaled offsets.
  assert(LocalStackSize % Scale == 0);
  OffsetOpnd.setImm(OffsetOpnd.getImm() + LocalStackSize / Scale);

  if (NeedsWinCFI) {
    *HasWinCFI = true;
    auto MBBI = std::next(MachineBasicBlock::iterator(MI));
    assert(MBBI != MI.getParent()->end() && "Expecting a valid instruction");
    assert(AArch64InstrInfo::isSEHInstruction(*MBBI) &&
           "Expecting a SEH instruction");
    fixupSEHOpcode(MBBI, LocalStackSize);
  }
}

static bool isTargetWindows(const MachineFunction &MF) {
  return MF.getSubtarget<AArch64Subtarget>().isTargetWindows();
}

// Convenience function to determine whether I is an SVE callee save.
static bool IsSVECalleeSave(MachineBasicBlock::iterator I) {
  switch (I->getOpcode()) {
  default:
    return false;
  case AArch64::STR_ZXI:
  case AArch64::STR_PXI:
  case AArch64::LDR_ZXI:
  case AArch64::LDR_PXI:
    return I->getFlag(MachineInstr::FrameSetup) ||
           I->getFlag(MachineInstr::FrameDestroy);
  }
}

static bool needsShadowCallStackPrologueEpilogue(MachineFunction &MF) {
  if (!(llvm::any_of(
            MF.getFrameInfo().getCalleeSavedInfo(),
            [](const auto &Info) { return Info.getReg() == AArch64::LR; }) &&
        MF.getFunction().hasFnAttribute(Attribute::ShadowCallStack)))
    return false;

  if (!MF.getSubtarget<AArch64Subtarget>().isXRegisterReserved(18))
    report_fatal_error("Must reserve x18 to use shadow call stack");

  return true;
}

static void emitShadowCallStackPrologue(const TargetInstrInfo &TII,
                                        MachineFunction &MF,
                                        MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        const DebugLoc &DL, bool NeedsWinCFI,
                                        bool NeedsUnwindInfo) {
  // Shadow call stack prolog: str x30, [x18], #8
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::STRXpost))
      .addReg(AArch64::X18, RegState::Define)
      .addReg(AArch64::LR)
      .addReg(AArch64::X18)
      .addImm(8)
      .setMIFlag(MachineInstr::FrameSetup);

  // This instruction also makes x18 live-in to the entry block.
  MBB.addLiveIn(AArch64::X18);

  if (NeedsWinCFI)
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::SEH_Nop))
        .setMIFlag(MachineInstr::FrameSetup);

  if (NeedsUnwindInfo) {
    // Emit a CFI instruction that causes 8 to be subtracted from the value of
    // x18 when unwinding past this frame.
    static const char CFIInst[] = {
        dwarf::DW_CFA_val_expression,
        18, // register
        2,  // length
        static_cast<char>(unsigned(dwarf::DW_OP_breg18)),
        static_cast<char>(-8) & 0x7f, // addend (sleb128)
    };
    unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createEscape(
        nullptr, StringRef(CFIInst, sizeof(CFIInst))));
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlag(MachineInstr::FrameSetup);
  }
}

static void emitShadowCallStackEpilogue(const TargetInstrInfo &TII,
                                        MachineFunction &MF,
                                        MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        const DebugLoc &DL) {
  // Shadow call stack epilog: ldr x30, [x18, #-8]!
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::LDRXpre))
      .addReg(AArch64::X18, RegState::Define)
      .addReg(AArch64::LR, RegState::Define)
      .addReg(AArch64::X18)
      .addImm(-8)
      .setMIFlag(MachineInstr::FrameDestroy);

  if (MF.getInfo<AArch64FunctionInfo>()->needsAsyncDwarfUnwindInfo()) {
    unsigned CFIIndex =
        MF.addFrameInst(MCCFIInstruction::createRestore(nullptr, 18));
    BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlags(MachineInstr::FrameDestroy);
  }
}

void AArch64FrameLowering::emitPrologue(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.begin();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const Function &F = MF.getFunction();
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const AArch64RegisterInfo *RegInfo = Subtarget.getRegisterInfo();
  const TargetInstrInfo *TII = Subtarget.getInstrInfo();
  MachineModuleInfo &MMI = MF.getMMI();
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  bool EmitCFI = AFI->needsDwarfUnwindInfo();
  bool HasFP = hasFP(MF);
  bool NeedsWinCFI = needsWinCFI(MF);
  bool HasWinCFI = false;
  auto Cleanup = make_scope_exit([&]() { MF.setHasWinCFI(HasWinCFI); });

  bool IsFunclet = MBB.isEHFuncletEntry();

  // At this point, we're going to decide whether or not the function uses a
  // redzone. In most cases, the function doesn't have a redzone so let's
  // assume that's false and set it to true in the case that there's a redzone.
  AFI->setHasRedZone(false);

  // Debug location must be unknown since the first debug location is used
  // to determine the end of the prologue.
  DebugLoc DL;

  const auto &MFnI = *MF.getInfo<AArch64FunctionInfo>();
  if (needsShadowCallStackPrologueEpilogue(MF))
    emitShadowCallStackPrologue(*TII, MF, MBB, MBBI, DL, NeedsWinCFI,
                                MFnI.needsDwarfUnwindInfo());

  if (MFnI.shouldSignReturnAddress()) {
    unsigned PACI;
    if (MFnI.shouldSignWithBKey()) {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::EMITBKEY))
          .setMIFlag(MachineInstr::FrameSetup);
      PACI = Subtarget.hasPAuth() ? AArch64::PACIB : AArch64::PACIBSP;
    } else {
      PACI = Subtarget.hasPAuth() ? AArch64::PACIA : AArch64::PACIASP;
    }

    auto MI = BuildMI(MBB, MBBI, DL, TII->get(PACI));
    if (Subtarget.hasPAuth())
      MI.addReg(AArch64::LR, RegState::Define)
          .addReg(AArch64::LR)
          .addReg(AArch64::SP, RegState::InternalRead);
    MI.setMIFlag(MachineInstr::FrameSetup);
    if (EmitCFI) {
      unsigned CFIIndex =
          MF.addFrameInst(MCCFIInstruction::createNegateRAState(nullptr));
      BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex)
          .setMIFlags(MachineInstr::FrameSetup);
    }
  }

  // We signal the presence of a Swift extended frame to external tools by
  // storing FP with 0b0001 in bits 63:60. In normal userland operation a simple
  // ORR is sufficient, it is assumed a Swift kernel would initialize the TBI
  // bits so that is still true.
  if (HasFP && AFI->hasSwiftAsyncContext()) {
    switch (MF.getTarget().Options.SwiftAsyncFramePointer) {
    case SwiftAsyncFramePointerMode::DeploymentBased:
      if (Subtarget.swiftAsyncContextIsDynamicallySet()) {
        // The special symbol below is absolute and has a *value* that can be
        // combined with the frame pointer to signal an extended frame.
        BuildMI(MBB, MBBI, DL, TII->get(AArch64::LOADgot), AArch64::X16)
            .addExternalSymbol("swift_async_extendedFramePointerFlags",
                               AArch64II::MO_GOT);
        BuildMI(MBB, MBBI, DL, TII->get(AArch64::ORRXrs), AArch64::FP)
            .addUse(AArch64::FP)
            .addUse(AArch64::X16)
            .addImm(Subtarget.isTargetILP32() ? 32 : 0);
        break;
      }
      LLVM_FALLTHROUGH;

    case SwiftAsyncFramePointerMode::Always:
      // ORR x29, x29, #0x1000_0000_0000_0000
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::ORRXri), AArch64::FP)
          .addUse(AArch64::FP)
          .addImm(0x1100)
          .setMIFlag(MachineInstr::FrameSetup);
      break;

    case SwiftAsyncFramePointerMode::Never:
      break;
    }
  }

  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction().getCallingConv() == CallingConv::GHC)
    return;

  // Set tagged base pointer to the requested stack slot.
  // Ideally it should match SP value after prologue.
  Optional<int> TBPI = AFI->getTaggedBasePointerIndex();
  if (TBPI)
    AFI->setTaggedBasePointerOffset(-MFI.getObjectOffset(*TBPI));
  else
    AFI->setTaggedBasePointerOffset(MFI.getStackSize());

  const StackOffset &SVEStackSize = getSVEStackSize(MF);

  // getStackSize() includes all the locals in its size calculation. We don't
  // include these locals when computing the stack size of a funclet, as they
  // are allocated in the parent's stack frame and accessed via the frame
  // pointer from the funclet.  We only save the callee saved registers in the
  // funclet, which are really the callee saved registers of the parent
  // function, including the funclet.
  int64_t NumBytes = IsFunclet ? getWinEHFuncletFrameSize(MF)
                               : MFI.getStackSize();
  if (!AFI->hasStackFrame() && !windowsRequiresStackProbe(MF, NumBytes)) {
    assert(!HasFP && "unexpected function without stack frame but with FP");
    assert(!SVEStackSize &&
           "unexpected function without stack frame but with SVE objects");
    // All of the stack allocation is for locals.
    AFI->setLocalStackSize(NumBytes);
    if (!NumBytes)
      return;
    // REDZONE: If the stack size is less than 128 bytes, we don't need
    // to actually allocate.
    if (canUseRedZone(MF)) {
      AFI->setHasRedZone(true);
      ++NumRedZoneFunctions;
    } else {
      emitFrameOffset(MBB, MBBI, DL, AArch64::SP, AArch64::SP,
                      StackOffset::getFixed(-NumBytes), TII,
                      MachineInstr::FrameSetup, false, NeedsWinCFI, &HasWinCFI);
      if (EmitCFI) {
        // Label used to tie together the PROLOG_LABEL and the MachineMoves.
        MCSymbol *FrameLabel = MMI.getContext().createTempSymbol();
          // Encode the stack size of the leaf function.
        unsigned CFIIndex = MF.addFrameInst(
            MCCFIInstruction::cfiDefCfaOffset(FrameLabel, NumBytes));
        BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex)
            .setMIFlags(MachineInstr::FrameSetup);
      }
    }

    if (NeedsWinCFI) {
      HasWinCFI = true;
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_PrologEnd))
          .setMIFlag(MachineInstr::FrameSetup);
    }

    return;
  }

  bool IsWin64 =
      Subtarget.isCallingConvWin64(MF.getFunction().getCallingConv());
  unsigned FixedObject = getFixedObjectSize(MF, AFI, IsWin64, IsFunclet);

  auto PrologueSaveSize = AFI->getCalleeSavedStackSize() + FixedObject;
  // All of the remaining stack allocations are for locals.
  AFI->setLocalStackSize(NumBytes - PrologueSaveSize);
  bool CombineSPBump = shouldCombineCSRLocalStackBump(MF, NumBytes);
  bool HomPrologEpilog = homogeneousPrologEpilog(MF);
  if (CombineSPBump) {
    assert(!SVEStackSize && "Cannot combine SP bump with SVE");
    emitFrameOffset(MBB, MBBI, DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(-NumBytes), TII,
                    MachineInstr::FrameSetup, false, NeedsWinCFI, &HasWinCFI,
                    EmitCFI);
    NumBytes = 0;
  } else if (HomPrologEpilog) {
    // Stack has been already adjusted.
    NumBytes -= PrologueSaveSize;
  } else if (PrologueSaveSize != 0) {
    MBBI = convertCalleeSaveRestoreToSPPrePostIncDec(
        MBB, MBBI, DL, TII, -PrologueSaveSize, NeedsWinCFI, &HasWinCFI,
        EmitCFI);
    NumBytes -= PrologueSaveSize;
  }
  assert(NumBytes >= 0 && "Negative stack allocation size!?");

  // Move past the saves of the callee-saved registers, fixing up the offsets
  // and pre-inc if we decided to combine the callee-save and local stack
  // pointer bump above.
  MachineBasicBlock::iterator End = MBB.end();
  while (MBBI != End && MBBI->getFlag(MachineInstr::FrameSetup) &&
         !IsSVECalleeSave(MBBI)) {
    if (CombineSPBump)
      fixupCalleeSaveRestoreStackOffset(*MBBI, AFI->getLocalStackSize(),
                                        NeedsWinCFI, &HasWinCFI);
    ++MBBI;
  }

  // For funclets the FP belongs to the containing function.
  if (!IsFunclet && HasFP) {
    // Only set up FP if we actually need to.
    int64_t FPOffset = AFI->getCalleeSaveBaseToFrameRecordOffset();

    if (CombineSPBump)
      FPOffset += AFI->getLocalStackSize();

    if (AFI->hasSwiftAsyncContext()) {
      // Before we update the live FP we have to ensure there's a valid (or
      // null) asynchronous context in its slot just before FP in the frame
      // record, so store it now.
      const auto &Attrs = MF.getFunction().getAttributes();
      bool HaveInitialContext = Attrs.hasAttrSomewhere(Attribute::SwiftAsync);
      if (HaveInitialContext)
        MBB.addLiveIn(AArch64::X22);
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::StoreSwiftAsyncContext))
          .addUse(HaveInitialContext ? AArch64::X22 : AArch64::XZR)
          .addUse(AArch64::SP)
          .addImm(FPOffset - 8)
          .setMIFlags(MachineInstr::FrameSetup);
    }

    if (HomPrologEpilog) {
      auto Prolog = MBBI;
      --Prolog;
      assert(Prolog->getOpcode() == AArch64::HOM_Prolog);
      Prolog->addOperand(MachineOperand::CreateImm(FPOffset));
    } else {
      // Issue    sub fp, sp, FPOffset or
      //          mov fp,sp          when FPOffset is zero.
      // Note: All stores of callee-saved registers are marked as "FrameSetup".
      // This code marks the instruction(s) that set the FP also.
      emitFrameOffset(MBB, MBBI, DL, AArch64::FP, AArch64::SP,
                      StackOffset::getFixed(FPOffset), TII,
                      MachineInstr::FrameSetup, false, NeedsWinCFI, &HasWinCFI);
    }
    if (EmitCFI) {
      // Define the current CFA rule to use the provided FP.
      const int OffsetToFirstCalleeSaveFromFP =
          AFI->getCalleeSaveBaseToFrameRecordOffset() -
          AFI->getCalleeSavedStackSize();
      Register FramePtr = RegInfo->getFrameRegister(MF);
      unsigned Reg = RegInfo->getDwarfRegNum(FramePtr, true);
      unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::cfiDefCfa(
          nullptr, Reg, FixedObject - OffsetToFirstCalleeSaveFromFP));
      BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex)
          .setMIFlags(MachineInstr::FrameSetup);
    }
  }

  // Now emit the moves for whatever callee saved regs we have (including FP,
  // LR if those are saved). Frame instructions for SVE register are emitted
  // later, after the instruction which actually save SVE regs.
  if (EmitCFI)
    emitCalleeSavedGPRLocations(MBB, MBBI);

  if (windowsRequiresStackProbe(MF, NumBytes)) {
    uint64_t NumWords = NumBytes >> 4;
    if (NeedsWinCFI) {
      HasWinCFI = true;
      // alloc_l can hold at most 256MB, so assume that NumBytes doesn't
      // exceed this amount.  We need to move at most 2^24 - 1 into x15.
      // This is at most two instructions, MOVZ follwed by MOVK.
      // TODO: Fix to use multiple stack alloc unwind codes for stacks
      // exceeding 256MB in size.
      if (NumBytes >= (1 << 28))
        report_fatal_error("Stack size cannot exceed 256MB for stack "
                            "unwinding purposes");

      uint32_t LowNumWords = NumWords & 0xFFFF;
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::MOVZXi), AArch64::X15)
            .addImm(LowNumWords)
            .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0))
            .setMIFlag(MachineInstr::FrameSetup);
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
            .setMIFlag(MachineInstr::FrameSetup);
      if ((NumWords & 0xFFFF0000) != 0) {
          BuildMI(MBB, MBBI, DL, TII->get(AArch64::MOVKXi), AArch64::X15)
              .addReg(AArch64::X15)
              .addImm((NumWords & 0xFFFF0000) >> 16) // High half
              .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 16))
              .setMIFlag(MachineInstr::FrameSetup);
          BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
            .setMIFlag(MachineInstr::FrameSetup);
      }
    } else {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::MOVi64imm), AArch64::X15)
          .addImm(NumWords)
          .setMIFlags(MachineInstr::FrameSetup);
    }

    switch (MF.getTarget().getCodeModel()) {
    case CodeModel::Tiny:
    case CodeModel::Small:
    case CodeModel::Medium:
    case CodeModel::Kernel:
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::BL))
          .addExternalSymbol("__chkstk")
          .addReg(AArch64::X15, RegState::Implicit)
          .addReg(AArch64::X16, RegState::Implicit | RegState::Define | RegState::Dead)
          .addReg(AArch64::X17, RegState::Implicit | RegState::Define | RegState::Dead)
          .addReg(AArch64::NZCV, RegState::Implicit | RegState::Define | RegState::Dead)
          .setMIFlags(MachineInstr::FrameSetup);
      if (NeedsWinCFI) {
        HasWinCFI = true;
        BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
            .setMIFlag(MachineInstr::FrameSetup);
      }
      break;
    case CodeModel::Large:
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::MOVaddrEXT))
          .addReg(AArch64::X16, RegState::Define)
          .addExternalSymbol("__chkstk")
          .addExternalSymbol("__chkstk")
          .setMIFlags(MachineInstr::FrameSetup);
      if (NeedsWinCFI) {
        HasWinCFI = true;
        BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
            .setMIFlag(MachineInstr::FrameSetup);
      }

      BuildMI(MBB, MBBI, DL, TII->get(getBLRCallOpcode(MF)))
          .addReg(AArch64::X16, RegState::Kill)
          .addReg(AArch64::X15, RegState::Implicit | RegState::Define)
          .addReg(AArch64::X16, RegState::Implicit | RegState::Define | RegState::Dead)
          .addReg(AArch64::X17, RegState::Implicit | RegState::Define | RegState::Dead)
          .addReg(AArch64::NZCV, RegState::Implicit | RegState::Define | RegState::Dead)
          .setMIFlags(MachineInstr::FrameSetup);
      if (NeedsWinCFI) {
        HasWinCFI = true;
        BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
            .setMIFlag(MachineInstr::FrameSetup);
      }
      break;
    }

    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SUBXrx64), AArch64::SP)
        .addReg(AArch64::SP, RegState::Kill)
        .addReg(AArch64::X15, RegState::Kill)
        .addImm(AArch64_AM::getArithExtendImm(AArch64_AM::UXTX, 4))
        .setMIFlags(MachineInstr::FrameSetup);
    if (NeedsWinCFI) {
      HasWinCFI = true;
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_StackAlloc))
          .addImm(NumBytes)
          .setMIFlag(MachineInstr::FrameSetup);
    }
    NumBytes = 0;
  }

  StackOffset AllocateBefore = SVEStackSize, AllocateAfter = {};
  MachineBasicBlock::iterator CalleeSavesBegin = MBBI, CalleeSavesEnd = MBBI;

  // Process the SVE callee-saves to determine what space needs to be
  // allocated.
  if (int64_t CalleeSavedSize = AFI->getSVECalleeSavedStackSize()) {
    // Find callee save instructions in frame.
    CalleeSavesBegin = MBBI;
    assert(IsSVECalleeSave(CalleeSavesBegin) && "Unexpected instruction");
    while (IsSVECalleeSave(MBBI) && MBBI != MBB.getFirstTerminator())
      ++MBBI;
    CalleeSavesEnd = MBBI;

    AllocateBefore = StackOffset::getScalable(CalleeSavedSize);
    AllocateAfter = SVEStackSize - AllocateBefore;
  }

  // Allocate space for the callee saves (if any).
  emitFrameOffset(
      MBB, CalleeSavesBegin, DL, AArch64::SP, AArch64::SP, -AllocateBefore, TII,
      MachineInstr::FrameSetup, false, false, nullptr,
      EmitCFI && !HasFP && AllocateBefore,
      StackOffset::getFixed((int64_t)MFI.getStackSize() - NumBytes));

  if (EmitCFI)
    emitCalleeSavedSVELocations(MBB, CalleeSavesEnd);

  // Finally allocate remaining SVE stack space.
  emitFrameOffset(MBB, CalleeSavesEnd, DL, AArch64::SP, AArch64::SP,
                  -AllocateAfter, TII, MachineInstr::FrameSetup, false, false,
                  nullptr, EmitCFI && !HasFP && AllocateAfter,
                  AllocateBefore + StackOffset::getFixed(
                                       (int64_t)MFI.getStackSize() - NumBytes));

  // Allocate space for the rest of the frame.
  if (NumBytes) {
    // Alignment is required for the parent frame, not the funclet
    const bool NeedsRealignment =
        !IsFunclet && RegInfo->hasStackRealignment(MF);
    unsigned scratchSPReg = AArch64::SP;

    if (NeedsRealignment) {
      scratchSPReg = findScratchNonCalleeSaveRegister(&MBB);
      assert(scratchSPReg != AArch64::NoRegister);
    }

    // If we're a leaf function, try using the red zone.
    if (!canUseRedZone(MF)) {
      // FIXME: in the case of dynamic re-alignment, NumBytes doesn't have
      // the correct value here, as NumBytes also includes padding bytes,
      // which shouldn't be counted here.
      emitFrameOffset(
          MBB, MBBI, DL, scratchSPReg, AArch64::SP,
          StackOffset::getFixed(-NumBytes), TII, MachineInstr::FrameSetup,
          false, NeedsWinCFI, &HasWinCFI, EmitCFI && !HasFP,
          SVEStackSize +
              StackOffset::getFixed((int64_t)MFI.getStackSize() - NumBytes));
    }
    if (NeedsRealignment) {
      const unsigned NrBitsToZero = Log2(MFI.getMaxAlign());
      assert(NrBitsToZero > 1);
      assert(scratchSPReg != AArch64::SP);

      // SUB X9, SP, NumBytes
      //   -- X9 is temporary register, so shouldn't contain any live data here,
      //   -- free to use. This is already produced by emitFrameOffset above.
      // AND SP, X9, 0b11111...0000
      // The logical immediates have a non-trivial encoding. The following
      // formula computes the encoded immediate with all ones but
      // NrBitsToZero zero bits as least significant bits.
      uint32_t andMaskEncoded = (1 << 12)                         // = N
                                | ((64 - NrBitsToZero) << 6)      // immr
                                | ((64 - NrBitsToZero - 1) << 0); // imms

      BuildMI(MBB, MBBI, DL, TII->get(AArch64::ANDXri), AArch64::SP)
          .addReg(scratchSPReg, RegState::Kill)
          .addImm(andMaskEncoded);
      AFI->setStackRealigned(true);
      if (NeedsWinCFI) {
        HasWinCFI = true;
        BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_StackAlloc))
            .addImm(NumBytes & andMaskEncoded)
            .setMIFlag(MachineInstr::FrameSetup);
      }
    }
  }

  // If we need a base pointer, set it up here. It's whatever the value of the
  // stack pointer is at this point. Any variable size objects will be allocated
  // after this, so we can still use the base pointer to reference locals.
  //
  // FIXME: Clarify FrameSetup flags here.
  // Note: Use emitFrameOffset() like above for FP if the FrameSetup flag is
  // needed.
  // For funclets the BP belongs to the containing function.
  if (!IsFunclet && RegInfo->hasBasePointer(MF)) {
    TII->copyPhysReg(MBB, MBBI, DL, RegInfo->getBaseRegister(), AArch64::SP,
                     false);
    if (NeedsWinCFI) {
      HasWinCFI = true;
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
          .setMIFlag(MachineInstr::FrameSetup);
    }
  }

  // The very last FrameSetup instruction indicates the end of prologue. Emit a
  // SEH opcode indicating the prologue end.
  if (NeedsWinCFI && HasWinCFI) {
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_PrologEnd))
        .setMIFlag(MachineInstr::FrameSetup);
  }

  // SEH funclets are passed the frame pointer in X1.  If the parent
  // function uses the base register, then the base register is used
  // directly, and is not retrieved from X1.
  if (IsFunclet && F.hasPersonalityFn()) {
    EHPersonality Per = classifyEHPersonality(F.getPersonalityFn());
    if (isAsynchronousEHPersonality(Per)) {
      BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::COPY), AArch64::FP)
          .addReg(AArch64::X1)
          .setMIFlag(MachineInstr::FrameSetup);
      MBB.addLiveIn(AArch64::X1);
    }
  }
}

static void InsertReturnAddressAuth(MachineFunction &MF,
                                    MachineBasicBlock &MBB) {
  const auto &MFI = *MF.getInfo<AArch64FunctionInfo>();
  if (!MFI.shouldSignReturnAddress())
    return;
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const TargetInstrInfo *TII = Subtarget.getInstrInfo();

  MachineBasicBlock::iterator MBBI = MBB.getFirstTerminator();
  DebugLoc DL;
  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();

  // The AUTIASP instruction assembles to a hint instruction before v8.3a so
  // this instruction can safely used for any v8a architecture.
  // From v8.3a onwards there are optimised authenticate LR and return
  // instructions, namely RETA{A,B}, that can be used instead. In this case the
  // DW_CFA_AARCH64_negate_ra_state can't be emitted.
  if (Subtarget.hasPAuth() && MBBI != MBB.end() &&
      MBBI->getOpcode() == AArch64::RET_ReallyLR) {
    BuildMI(MBB, MBBI, DL,
            TII->get(MFI.shouldSignWithBKey() ? AArch64::RETAB : AArch64::RETAA))
        .copyImplicitOps(*MBBI);
    MBB.erase(MBBI);
  } else {
    BuildMI(
        MBB, MBBI, DL,
        TII->get(MFI.shouldSignWithBKey() ? AArch64::AUTIBSP : AArch64::AUTIASP))
        .setMIFlag(MachineInstr::FrameDestroy);

    unsigned CFIIndex =
        MF.addFrameInst(MCCFIInstruction::createNegateRAState(nullptr));
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlags(MachineInstr::FrameDestroy);
  }
}

static bool isFuncletReturnInstr(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return false;
  case AArch64::CATCHRET:
  case AArch64::CLEANUPRET:
    return true;
  }
}

void AArch64FrameLowering::emitEpilogue(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const TargetInstrInfo *TII = Subtarget.getInstrInfo();
  DebugLoc DL;
  bool NeedsWinCFI = needsWinCFI(MF);
  bool EmitCFI = MF.getInfo<AArch64FunctionInfo>()->needsAsyncDwarfUnwindInfo();
  bool HasWinCFI = false;
  bool IsFunclet = false;
  auto WinCFI = make_scope_exit([&]() { assert(HasWinCFI == MF.hasWinCFI()); });

  if (MBB.end() != MBBI) {
    DL = MBBI->getDebugLoc();
    IsFunclet = isFuncletReturnInstr(*MBBI);
  }

  auto FinishingTouches = make_scope_exit([&]() {
    InsertReturnAddressAuth(MF, MBB);
    if (needsShadowCallStackPrologueEpilogue(MF))
      emitShadowCallStackEpilogue(*TII, MF, MBB, MBB.getFirstTerminator(), DL);
    if (EmitCFI)
      emitCalleeSavedGPRRestores(MBB, MBB.getFirstTerminator());
  });

  int64_t NumBytes = IsFunclet ? getWinEHFuncletFrameSize(MF)
                               : MFI.getStackSize();
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();

  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction().getCallingConv() == CallingConv::GHC)
    return;

  // How much of the stack used by incoming arguments this function is expected
  // to restore in this particular epilogue.
  int64_t ArgumentStackToRestore = getArgumentStackToRestore(MF, MBB);
  bool IsWin64 =
      Subtarget.isCallingConvWin64(MF.getFunction().getCallingConv());
  unsigned FixedObject = getFixedObjectSize(MF, AFI, IsWin64, IsFunclet);

  int64_t AfterCSRPopSize = ArgumentStackToRestore;
  auto PrologueSaveSize = AFI->getCalleeSavedStackSize() + FixedObject;
  // We cannot rely on the local stack size set in emitPrologue if the function
  // has funclets, as funclets have different local stack size requirements, and
  // the current value set in emitPrologue may be that of the containing
  // function.
  if (MF.hasEHFunclets())
    AFI->setLocalStackSize(NumBytes - PrologueSaveSize);
  if (homogeneousPrologEpilog(MF, &MBB)) {
    assert(!NeedsWinCFI);
    auto LastPopI = MBB.getFirstTerminator();
    if (LastPopI != MBB.begin()) {
      auto HomogeneousEpilog = std::prev(LastPopI);
      if (HomogeneousEpilog->getOpcode() == AArch64::HOM_Epilog)
        LastPopI = HomogeneousEpilog;
    }

    // Adjust local stack
    emitFrameOffset(MBB, LastPopI, DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(AFI->getLocalStackSize()), TII,
                    MachineInstr::FrameDestroy, false, NeedsWinCFI);

    // SP has been already adjusted while restoring callee save regs.
    // We've bailed-out the case with adjusting SP for arguments.
    assert(AfterCSRPopSize == 0);
    return;
  }
  bool CombineSPBump = shouldCombineCSRLocalStackBumpInEpilogue(MBB, NumBytes);
  // Assume we can't combine the last pop with the sp restore.

  bool CombineAfterCSRBump = false;
  if (!CombineSPBump && PrologueSaveSize != 0) {
    MachineBasicBlock::iterator Pop = std::prev(MBB.getFirstTerminator());
    while (Pop->getOpcode() == TargetOpcode::CFI_INSTRUCTION ||
           AArch64InstrInfo::isSEHInstruction(*Pop))
      Pop = std::prev(Pop);
    // Converting the last ldp to a post-index ldp is valid only if the last
    // ldp's offset is 0.
    const MachineOperand &OffsetOp = Pop->getOperand(Pop->getNumOperands() - 1);
    // If the offset is 0 and the AfterCSR pop is not actually trying to
    // allocate more stack for arguments (in space that an untimely interrupt
    // may clobber), convert it to a post-index ldp.
    if (OffsetOp.getImm() == 0 && AfterCSRPopSize >= 0) {
      convertCalleeSaveRestoreToSPPrePostIncDec(
          MBB, Pop, DL, TII, PrologueSaveSize, NeedsWinCFI, &HasWinCFI, EmitCFI,
          MachineInstr::FrameDestroy, PrologueSaveSize);
    } else {
      // If not, make sure to emit an add after the last ldp.
      // We're doing this by transfering the size to be restored from the
      // adjustment *before* the CSR pops to the adjustment *after* the CSR
      // pops.
      AfterCSRPopSize += PrologueSaveSize;
      CombineAfterCSRBump = true;
    }
  }

  // Move past the restores of the callee-saved registers.
  // If we plan on combining the sp bump of the local stack size and the callee
  // save stack size, we might need to adjust the CSR save and restore offsets.
  MachineBasicBlock::iterator LastPopI = MBB.getFirstTerminator();
  MachineBasicBlock::iterator Begin = MBB.begin();
  while (LastPopI != Begin) {
    --LastPopI;
    if (!LastPopI->getFlag(MachineInstr::FrameDestroy) ||
        IsSVECalleeSave(LastPopI)) {
      ++LastPopI;
      break;
    } else if (CombineSPBump)
      fixupCalleeSaveRestoreStackOffset(*LastPopI, AFI->getLocalStackSize(),
                                        NeedsWinCFI, &HasWinCFI);
  }

  if (MF.hasWinCFI()) {
    // If the prologue didn't contain any SEH opcodes and didn't set the
    // MF.hasWinCFI() flag, assume the epilogue won't either, and skip the
    // EpilogStart - to avoid generating CFI for functions that don't need it.
    // (And as we didn't generate any prologue at all, it would be asymmetrical
    // to the epilogue.) By the end of the function, we assert that
    // HasWinCFI is equal to MF.hasWinCFI(), to verify this assumption.
    HasWinCFI = true;
    BuildMI(MBB, LastPopI, DL, TII->get(AArch64::SEH_EpilogStart))
        .setMIFlag(MachineInstr::FrameDestroy);
  }

  if (hasFP(MF) && AFI->hasSwiftAsyncContext()) {
    switch (MF.getTarget().Options.SwiftAsyncFramePointer) {
    case SwiftAsyncFramePointerMode::DeploymentBased:
      // Avoid the reload as it is GOT relative, and instead fall back to the
      // hardcoded value below.  This allows a mismatch between the OS and
      // application without immediately terminating on the difference.
      LLVM_FALLTHROUGH;
    case SwiftAsyncFramePointerMode::Always:
      // We need to reset FP to its untagged state on return. Bit 60 is
      // currently used to show the presence of an extended frame.

      // BIC x29, x29, #0x1000_0000_0000_0000
      BuildMI(MBB, MBB.getFirstTerminator(), DL, TII->get(AArch64::ANDXri),
              AArch64::FP)
          .addUse(AArch64::FP)
          .addImm(0x10fe)
          .setMIFlag(MachineInstr::FrameDestroy);
      break;

    case SwiftAsyncFramePointerMode::Never:
      break;
    }
  }

  const StackOffset &SVEStackSize = getSVEStackSize(MF);

  // If there is a single SP update, insert it before the ret and we're done.
  if (CombineSPBump) {
    assert(!SVEStackSize && "Cannot combine SP bump with SVE");

    // When we are about to restore the CSRs, the CFA register is SP again.
    if (EmitCFI && hasFP(MF)) {
      const AArch64RegisterInfo &RegInfo = *Subtarget.getRegisterInfo();
      unsigned Reg = RegInfo.getDwarfRegNum(AArch64::SP, true);
      unsigned CFIIndex =
          MF.addFrameInst(MCCFIInstruction::cfiDefCfa(nullptr, Reg, NumBytes));
      BuildMI(MBB, LastPopI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex)
          .setMIFlags(MachineInstr::FrameDestroy);
    }

    emitFrameOffset(MBB, MBB.getFirstTerminator(), DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(NumBytes + (int64_t)AfterCSRPopSize),
                    TII, MachineInstr::FrameDestroy, false, NeedsWinCFI,
                    &HasWinCFI, EmitCFI, StackOffset::getFixed(NumBytes));
    if (HasWinCFI)
      BuildMI(MBB, MBB.getFirstTerminator(), DL,
              TII->get(AArch64::SEH_EpilogEnd))
          .setMIFlag(MachineInstr::FrameDestroy);
    return;
  }

  NumBytes -= PrologueSaveSize;
  assert(NumBytes >= 0 && "Negative stack allocation size!?");

  // Process the SVE callee-saves to determine what space needs to be
  // deallocated.
  StackOffset DeallocateBefore = {}, DeallocateAfter = SVEStackSize;
  MachineBasicBlock::iterator RestoreBegin = LastPopI, RestoreEnd = LastPopI;
  if (int64_t CalleeSavedSize = AFI->getSVECalleeSavedStackSize()) {
    RestoreBegin = std::prev(RestoreEnd);
    while (RestoreBegin != MBB.begin() &&
           IsSVECalleeSave(std::prev(RestoreBegin)))
      --RestoreBegin;

    assert(IsSVECalleeSave(RestoreBegin) &&
           IsSVECalleeSave(std::prev(RestoreEnd)) && "Unexpected instruction");

    StackOffset CalleeSavedSizeAsOffset =
        StackOffset::getScalable(CalleeSavedSize);
    DeallocateBefore = SVEStackSize - CalleeSavedSizeAsOffset;
    DeallocateAfter = CalleeSavedSizeAsOffset;
  }

  // Deallocate the SVE area.
  if (SVEStackSize) {
    if (AFI->isStackRealigned()) {
      if (int64_t CalleeSavedSize = AFI->getSVECalleeSavedStackSize()) {
        // Set SP to start of SVE callee-save area from which they can
        // be reloaded. The code below will deallocate the stack space
        // space by moving FP -> SP.
        emitFrameOffset(MBB, RestoreBegin, DL, AArch64::SP, AArch64::FP,
                        StackOffset::getScalable(-CalleeSavedSize), TII,
                        MachineInstr::FrameDestroy);
      }
    } else {
      if (AFI->getSVECalleeSavedStackSize()) {
        // Deallocate the non-SVE locals first before we can deallocate (and
        // restore callee saves) from the SVE area.
        emitFrameOffset(
            MBB, RestoreBegin, DL, AArch64::SP, AArch64::SP,
            StackOffset::getFixed(NumBytes), TII, MachineInstr::FrameDestroy,
            false, false, nullptr, EmitCFI && !hasFP(MF),
            SVEStackSize + StackOffset::getFixed(NumBytes + PrologueSaveSize));
        NumBytes = 0;
      }

      emitFrameOffset(MBB, RestoreBegin, DL, AArch64::SP, AArch64::SP,
                      DeallocateBefore, TII, MachineInstr::FrameDestroy, false,
                      false, nullptr, EmitCFI && !hasFP(MF),
                      SVEStackSize +
                          StackOffset::getFixed(NumBytes + PrologueSaveSize));

      emitFrameOffset(MBB, RestoreEnd, DL, AArch64::SP, AArch64::SP,
                      DeallocateAfter, TII, MachineInstr::FrameDestroy, false,
                      false, nullptr, EmitCFI && !hasFP(MF),
                      DeallocateAfter +
                          StackOffset::getFixed(NumBytes + PrologueSaveSize));
    }
    if (EmitCFI)
      emitCalleeSavedSVERestores(MBB, RestoreEnd);
  }

  if (!hasFP(MF)) {
    bool RedZone = canUseRedZone(MF);
    // If this was a redzone leaf function, we don't need to restore the
    // stack pointer (but we may need to pop stack args for fastcc).
    if (RedZone && AfterCSRPopSize == 0)
      return;

    // Pop the local variables off the stack. If there are no callee-saved
    // registers, it means we are actually positioned at the terminator and can
    // combine stack increment for the locals and the stack increment for
    // callee-popped arguments into (possibly) a single instruction and be done.
    bool NoCalleeSaveRestore = PrologueSaveSize == 0;
    int64_t StackRestoreBytes = RedZone ? 0 : NumBytes;
    if (NoCalleeSaveRestore)
      StackRestoreBytes += AfterCSRPopSize;

    emitFrameOffset(
        MBB, LastPopI, DL, AArch64::SP, AArch64::SP,
        StackOffset::getFixed(StackRestoreBytes), TII,
        MachineInstr::FrameDestroy, false, NeedsWinCFI, &HasWinCFI, EmitCFI,
        StackOffset::getFixed((RedZone ? 0 : NumBytes) + PrologueSaveSize));

    // If we were able to combine the local stack pop with the argument pop,
    // then we're done.
    if (NoCalleeSaveRestore || AfterCSRPopSize == 0) {
      if (HasWinCFI) {
        BuildMI(MBB, MBB.getFirstTerminator(), DL,
                TII->get(AArch64::SEH_EpilogEnd))
            .setMIFlag(MachineInstr::FrameDestroy);
      }
      return;
    }

    NumBytes = 0;
  }

  // Restore the original stack pointer.
  // FIXME: Rather than doing the math here, we should instead just use
  // non-post-indexed loads for the restores if we aren't actually going to
  // be able to save any instructions.
  if (!IsFunclet && (MFI.hasVarSizedObjects() || AFI->isStackRealigned())) {
    emitFrameOffset(
        MBB, LastPopI, DL, AArch64::SP, AArch64::FP,
        StackOffset::getFixed(-AFI->getCalleeSaveBaseToFrameRecordOffset()),
        TII, MachineInstr::FrameDestroy, false, NeedsWinCFI);
  } else if (NumBytes)
    emitFrameOffset(MBB, LastPopI, DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(NumBytes), TII,
                    MachineInstr::FrameDestroy, false, NeedsWinCFI);

  // When we are about to restore the CSRs, the CFA register is SP again.
  if (EmitCFI && hasFP(MF)) {
    const AArch64RegisterInfo &RegInfo = *Subtarget.getRegisterInfo();
    unsigned Reg = RegInfo.getDwarfRegNum(AArch64::SP, true);
    unsigned CFIIndex = MF.addFrameInst(
        MCCFIInstruction::cfiDefCfa(nullptr, Reg, PrologueSaveSize));
    BuildMI(MBB, LastPopI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlags(MachineInstr::FrameDestroy);
  }

  // This must be placed after the callee-save restore code because that code
  // assumes the SP is at the same location as it was after the callee-save save
  // code in the prologue.
  if (AfterCSRPopSize) {
    assert(AfterCSRPopSize > 0 && "attempting to reallocate arg stack that an "
                                  "interrupt may have clobbered");

    emitFrameOffset(
        MBB, MBB.getFirstTerminator(), DL, AArch64::SP, AArch64::SP,
        StackOffset::getFixed(AfterCSRPopSize), TII, MachineInstr::FrameDestroy,
        false, NeedsWinCFI, &HasWinCFI, EmitCFI,
        StackOffset::getFixed(CombineAfterCSRBump ? PrologueSaveSize : 0));
  }
  if (HasWinCFI)
    BuildMI(MBB, MBB.getFirstTerminator(), DL, TII->get(AArch64::SEH_EpilogEnd))
        .setMIFlag(MachineInstr::FrameDestroy);
}

/// getFrameIndexReference - Provide a base+offset reference to an FI slot for
/// debug info.  It's the same as what we use for resolving the code-gen
/// references for now.  FIXME: This can go wrong when references are
/// SP-relative and simple call frames aren't used.
StackOffset
AArch64FrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                             Register &FrameReg) const {
  return resolveFrameIndexReference(
      MF, FI, FrameReg,
      /*PreferFP=*/
      MF.getFunction().hasFnAttribute(Attribute::SanitizeHWAddress),
      /*ForSimm=*/false);
}

StackOffset
AArch64FrameLowering::getNonLocalFrameIndexReference(const MachineFunction &MF,
                                                     int FI) const {
  return StackOffset::getFixed(getSEHFrameIndexOffset(MF, FI));
}

static StackOffset getFPOffset(const MachineFunction &MF,
                               int64_t ObjectOffset) {
  const auto *AFI = MF.getInfo<AArch64FunctionInfo>();
  const auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  bool IsWin64 =
      Subtarget.isCallingConvWin64(MF.getFunction().getCallingConv());
  unsigned FixedObject =
      getFixedObjectSize(MF, AFI, IsWin64, /*IsFunclet=*/false);
  int64_t CalleeSaveSize = AFI->getCalleeSavedStackSize(MF.getFrameInfo());
  int64_t FPAdjust =
      CalleeSaveSize - AFI->getCalleeSaveBaseToFrameRecordOffset();
  return StackOffset::getFixed(ObjectOffset + FixedObject + FPAdjust);
}

static StackOffset getStackOffset(const MachineFunction &MF,
                                  int64_t ObjectOffset) {
  const auto &MFI = MF.getFrameInfo();
  return StackOffset::getFixed(ObjectOffset + (int64_t)MFI.getStackSize());
}

  // TODO: This function currently does not work for scalable vectors.
int AArch64FrameLowering::getSEHFrameIndexOffset(const MachineFunction &MF,
                                                 int FI) const {
  const auto *RegInfo = static_cast<const AArch64RegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  int ObjectOffset = MF.getFrameInfo().getObjectOffset(FI);
  return RegInfo->getLocalAddressRegister(MF) == AArch64::FP
             ? getFPOffset(MF, ObjectOffset).getFixed()
             : getStackOffset(MF, ObjectOffset).getFixed();
}

StackOffset AArch64FrameLowering::resolveFrameIndexReference(
    const MachineFunction &MF, int FI, Register &FrameReg, bool PreferFP,
    bool ForSimm) const {
  const auto &MFI = MF.getFrameInfo();
  int64_t ObjectOffset = MFI.getObjectOffset(FI);
  bool isFixed = MFI.isFixedObjectIndex(FI);
  bool isSVE = MFI.getStackID(FI) == TargetStackID::ScalableVector;
  return resolveFrameOffsetReference(MF, ObjectOffset, isFixed, isSVE, FrameReg,
                                     PreferFP, ForSimm);
}

StackOffset AArch64FrameLowering::resolveFrameOffsetReference(
    const MachineFunction &MF, int64_t ObjectOffset, bool isFixed, bool isSVE,
    Register &FrameReg, bool PreferFP, bool ForSimm) const {
  const auto &MFI = MF.getFrameInfo();
  const auto *RegInfo = static_cast<const AArch64RegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  const auto *AFI = MF.getInfo<AArch64FunctionInfo>();
  const auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();

  int64_t FPOffset = getFPOffset(MF, ObjectOffset).getFixed();
  int64_t Offset = getStackOffset(MF, ObjectOffset).getFixed();
  bool isCSR =
      !isFixed && ObjectOffset >= -((int)AFI->getCalleeSavedStackSize(MFI));

  const StackOffset &SVEStackSize = getSVEStackSize(MF);

  // Use frame pointer to reference fixed objects. Use it for locals if
  // there are VLAs or a dynamically realigned SP (and thus the SP isn't
  // reliable as a base). Make sure useFPForScavengingIndex() does the
  // right thing for the emergency spill slot.
  bool UseFP = false;
  if (AFI->hasStackFrame() && !isSVE) {
    // We shouldn't prefer using the FP to access fixed-sized stack objects when
    // there are scalable (SVE) objects in between the FP and the fixed-sized
    // objects.
    PreferFP &= !SVEStackSize;

    // Note: Keeping the following as multiple 'if' statements rather than
    // merging to a single expression for readability.
    //
    // Argument access should always use the FP.
    if (isFixed) {
      UseFP = hasFP(MF);
    } else if (isCSR && RegInfo->hasStackRealignment(MF)) {
      // References to the CSR area must use FP if we're re-aligning the stack
      // since the dynamically-sized alignment padding is between the SP/BP and
      // the CSR area.
      assert(hasFP(MF) && "Re-aligned stack must have frame pointer");
      UseFP = true;
    } else if (hasFP(MF) && !RegInfo->hasStackRealignment(MF)) {
      // If the FPOffset is negative and we're producing a signed immediate, we
      // have to keep in mind that the available offset range for negative
      // offsets is smaller than for positive ones. If an offset is available
      // via the FP and the SP, use whichever is closest.
      bool FPOffsetFits = !ForSimm || FPOffset >= -256;
      PreferFP |= Offset > -FPOffset && !SVEStackSize;

      if (MFI.hasVarSizedObjects()) {
        // If we have variable sized objects, we can use either FP or BP, as the
        // SP offset is unknown. We can use the base pointer if we have one and
        // FP is not preferred. If not, we're stuck with using FP.
        bool CanUseBP = RegInfo->hasBasePointer(MF);
        if (FPOffsetFits && CanUseBP) // Both are ok. Pick the best.
          UseFP = PreferFP;
        else if (!CanUseBP) // Can't use BP. Forced to use FP.
          UseFP = true;
        // else we can use BP and FP, but the offset from FP won't fit.
        // That will make us scavenge registers which we can probably avoid by
        // using BP. If it won't fit for BP either, we'll scavenge anyway.
      } else if (FPOffset >= 0) {
        // Use SP or FP, whichever gives us the best chance of the offset
        // being in range for direct access. If the FPOffset is positive,
        // that'll always be best, as the SP will be even further away.
        UseFP = true;
      } else if (MF.hasEHFunclets() && !RegInfo->hasBasePointer(MF)) {
        // Funclets access the locals contained in the parent's stack frame
        // via the frame pointer, so we have to use the FP in the parent
        // function.
        (void) Subtarget;
        assert(
            Subtarget.isCallingConvWin64(MF.getFunction().getCallingConv()) &&
            "Funclets should only be present on Win64");
        UseFP = true;
      } else {
        // We have the choice between FP and (SP or BP).
        if (FPOffsetFits && PreferFP) // If FP is the best fit, use it.
          UseFP = true;
      }
    }
  }

  assert(
      ((isFixed || isCSR) || !RegInfo->hasStackRealignment(MF) || !UseFP) &&
      "In the presence of dynamic stack pointer realignment, "
      "non-argument/CSR objects cannot be accessed through the frame pointer");

  if (isSVE) {
    StackOffset FPOffset =
        StackOffset::get(-AFI->getCalleeSaveBaseToFrameRecordOffset(), ObjectOffset);
    StackOffset SPOffset =
        SVEStackSize +
        StackOffset::get(MFI.getStackSize() - AFI->getCalleeSavedStackSize(),
                         ObjectOffset);
    // Always use the FP for SVE spills if available and beneficial.
    if (hasFP(MF) && (SPOffset.getFixed() ||
                      FPOffset.getScalable() < SPOffset.getScalable() ||
                      RegInfo->hasStackRealignment(MF))) {
      FrameReg = RegInfo->getFrameRegister(MF);
      return FPOffset;
    }

    FrameReg = RegInfo->hasBasePointer(MF) ? RegInfo->getBaseRegister()
                                           : (unsigned)AArch64::SP;
    return SPOffset;
  }

  StackOffset ScalableOffset = {};
  if (UseFP && !(isFixed || isCSR))
    ScalableOffset = -SVEStackSize;
  if (!UseFP && (isFixed || isCSR))
    ScalableOffset = SVEStackSize;

  if (UseFP) {
    FrameReg = RegInfo->getFrameRegister(MF);
    return StackOffset::getFixed(FPOffset) + ScalableOffset;
  }

  // Use the base pointer if we have one.
  if (RegInfo->hasBasePointer(MF))
    FrameReg = RegInfo->getBaseRegister();
  else {
    assert(!MFI.hasVarSizedObjects() &&
           "Can't use SP when we have var sized objects.");
    FrameReg = AArch64::SP;
    // If we're using the red zone for this function, the SP won't actually
    // be adjusted, so the offsets will be negative. They're also all
    // within range of the signed 9-bit immediate instructions.
    if (canUseRedZone(MF))
      Offset -= AFI->getLocalStackSize();
  }

  return StackOffset::getFixed(Offset) + ScalableOffset;
}

static unsigned getPrologueDeath(MachineFunction &MF, unsigned Reg) {
  // Do not set a kill flag on values that are also marked as live-in. This
  // happens with the @llvm-returnaddress intrinsic and with arguments passed in
  // callee saved registers.
  // Omitting the kill flags is conservatively correct even if the live-in
  // is not used after all.
  bool IsLiveIn = MF.getRegInfo().isLiveIn(Reg);
  return getKillRegState(!IsLiveIn);
}

static bool produceCompactUnwindFrame(MachineFunction &MF) {
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  AttributeList Attrs = MF.getFunction().getAttributes();
  return Subtarget.isTargetMachO() &&
         !(Subtarget.getTargetLowering()->supportSwiftError() &&
           Attrs.hasAttrSomewhere(Attribute::SwiftError)) &&
         MF.getFunction().getCallingConv() != CallingConv::SwiftTail;
}

static bool invalidateWindowsRegisterPairing(unsigned Reg1, unsigned Reg2,
                                             bool NeedsWinCFI, bool IsFirst) {
  // If we are generating register pairs for a Windows function that requires
  // EH support, then pair consecutive registers only.  There are no unwind
  // opcodes for saves/restores of non-consectuve register pairs.
  // The unwind opcodes are save_regp, save_regp_x, save_fregp, save_frepg_x,
  // save_lrpair.
  // https://docs.microsoft.com/en-us/cpp/build/arm64-exception-handling

  if (Reg2 == AArch64::FP)
    return true;
  if (!NeedsWinCFI)
    return false;
  if (Reg2 == Reg1 + 1)
    return false;
  // If pairing a GPR with LR, the pair can be described by the save_lrpair
  // opcode. If this is the first register pair, it would end up with a
  // predecrement, but there's no save_lrpair_x opcode, so we can only do this
  // if LR is paired with something else than the first register.
  // The save_lrpair opcode requires the first register to be an odd one.
  if (Reg1 >= AArch64::X19 && Reg1 <= AArch64::X27 &&
      (Reg1 - AArch64::X19) % 2 == 0 && Reg2 == AArch64::LR && !IsFirst)
    return false;
  return true;
}

/// Returns true if Reg1 and Reg2 cannot be paired using a ldp/stp instruction.
/// WindowsCFI requires that only consecutive registers can be paired.
/// LR and FP need to be allocated together when the frame needs to save
/// the frame-record. This means any other register pairing with LR is invalid.
static bool invalidateRegisterPairing(unsigned Reg1, unsigned Reg2,
                                      bool UsesWinAAPCS, bool NeedsWinCFI,
                                      bool NeedsFrameRecord, bool IsFirst) {
  if (UsesWinAAPCS)
    return invalidateWindowsRegisterPairing(Reg1, Reg2, NeedsWinCFI, IsFirst);

  // If we need to store the frame record, don't pair any register
  // with LR other than FP.
  if (NeedsFrameRecord)
    return Reg2 == AArch64::LR;

  return false;
}

namespace {

struct RegPairInfo {
  unsigned Reg1 = AArch64::NoRegister;
  unsigned Reg2 = AArch64::NoRegister;
  int FrameIdx;
  int Offset;
  enum RegType { GPR, FPR64, FPR128, PPR, ZPR } Type;

  RegPairInfo() = default;

  bool isPaired() const { return Reg2 != AArch64::NoRegister; }

  unsigned getScale() const {
    switch (Type) {
    case PPR:
      return 2;
    case GPR:
    case FPR64:
      return 8;
    case ZPR:
    case FPR128:
      return 16;
    }
    llvm_unreachable("Unsupported type");
  }

  bool isScalable() const { return Type == PPR || Type == ZPR; }
};

} // end anonymous namespace

static void computeCalleeSaveRegisterPairs(
    MachineFunction &MF, ArrayRef<CalleeSavedInfo> CSI,
    const TargetRegisterInfo *TRI, SmallVectorImpl<RegPairInfo> &RegPairs,
    bool NeedsFrameRecord) {

  if (CSI.empty())
    return;

  bool IsWindows = isTargetWindows(MF);
  bool NeedsWinCFI = needsWinCFI(MF);
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  CallingConv::ID CC = MF.getFunction().getCallingConv();
  unsigned Count = CSI.size();
  (void)CC;
  // MachO's compact unwind format relies on all registers being stored in
  // pairs.
  assert((!produceCompactUnwindFrame(MF) ||
          CC == CallingConv::PreserveMost || CC == CallingConv::CXX_FAST_TLS ||
          (Count & 1) == 0) &&
         "Odd number of callee-saved regs to spill!");
  int ByteOffset = AFI->getCalleeSavedStackSize();
  int StackFillDir = -1;
  int RegInc = 1;
  unsigned FirstReg = 0;
  if (NeedsWinCFI) {
    // For WinCFI, fill the stack from the bottom up.
    ByteOffset = 0;
    StackFillDir = 1;
    // As the CSI array is reversed to match PrologEpilogInserter, iterate
    // backwards, to pair up registers starting from lower numbered registers.
    RegInc = -1;
    FirstReg = Count - 1;
  }
  int ScalableByteOffset = AFI->getSVECalleeSavedStackSize();
  bool NeedGapToAlignStack = AFI->hasCalleeSaveStackFreeSpace();

  // When iterating backwards, the loop condition relies on unsigned wraparound.
  for (unsigned i = FirstReg; i < Count; i += RegInc) {
    RegPairInfo RPI;
    RPI.Reg1 = CSI[i].getReg();

    if (AArch64::GPR64RegClass.contains(RPI.Reg1))
      RPI.Type = RegPairInfo::GPR;
    else if (AArch64::FPR64RegClass.contains(RPI.Reg1))
      RPI.Type = RegPairInfo::FPR64;
    else if (AArch64::FPR128RegClass.contains(RPI.Reg1))
      RPI.Type = RegPairInfo::FPR128;
    else if (AArch64::ZPRRegClass.contains(RPI.Reg1))
      RPI.Type = RegPairInfo::ZPR;
    else if (AArch64::PPRRegClass.contains(RPI.Reg1))
      RPI.Type = RegPairInfo::PPR;
    else
      llvm_unreachable("Unsupported register class.");

    // Add the next reg to the pair if it is in the same register class.
    if (unsigned(i + RegInc) < Count) {
      Register NextReg = CSI[i + RegInc].getReg();
      bool IsFirst = i == FirstReg;
      switch (RPI.Type) {
      case RegPairInfo::GPR:
        if (AArch64::GPR64RegClass.contains(NextReg) &&
            !invalidateRegisterPairing(RPI.Reg1, NextReg, IsWindows,
                                       NeedsWinCFI, NeedsFrameRecord, IsFirst))
          RPI.Reg2 = NextReg;
        break;
      case RegPairInfo::FPR64:
        if (AArch64::FPR64RegClass.contains(NextReg) &&
            !invalidateWindowsRegisterPairing(RPI.Reg1, NextReg, NeedsWinCFI,
                                              IsFirst))
          RPI.Reg2 = NextReg;
        break;
      case RegPairInfo::FPR128:
        if (AArch64::FPR128RegClass.contains(NextReg))
          RPI.Reg2 = NextReg;
        break;
      case RegPairInfo::PPR:
      case RegPairInfo::ZPR:
        break;
      }
    }

    // GPRs and FPRs are saved in pairs of 64-bit regs. We expect the CSI
    // list to come in sorted by frame index so that we can issue the store
    // pair instructions directly. Assert if we see anything otherwise.
    //
    // The order of the registers in the list is controlled by
    // getCalleeSavedRegs(), so they will always be in-order, as well.
    assert((!RPI.isPaired() ||
            (CSI[i].getFrameIdx() + RegInc == CSI[i + RegInc].getFrameIdx())) &&
           "Out of order callee saved regs!");

    assert((!RPI.isPaired() || !NeedsFrameRecord || RPI.Reg2 != AArch64::FP ||
            RPI.Reg1 == AArch64::LR) &&
           "FrameRecord must be allocated together with LR");

    // Windows AAPCS has FP and LR reversed.
    assert((!RPI.isPaired() || !NeedsFrameRecord || RPI.Reg1 != AArch64::FP ||
            RPI.Reg2 == AArch64::LR) &&
           "FrameRecord must be allocated together with LR");

    // MachO's compact unwind format relies on all registers being stored in
    // adjacent register pairs.
    assert((!produceCompactUnwindFrame(MF) ||
            CC == CallingConv::PreserveMost || CC == CallingConv::CXX_FAST_TLS ||
            (RPI.isPaired() &&
             ((RPI.Reg1 == AArch64::LR && RPI.Reg2 == AArch64::FP) ||
              RPI.Reg1 + 1 == RPI.Reg2))) &&
           "Callee-save registers not saved as adjacent register pair!");

    RPI.FrameIdx = CSI[i].getFrameIdx();
    if (NeedsWinCFI &&
        RPI.isPaired()) // RPI.FrameIdx must be the lower index of the pair
      RPI.FrameIdx = CSI[i + RegInc].getFrameIdx();

    int Scale = RPI.getScale();

    int OffsetPre = RPI.isScalable() ? ScalableByteOffset : ByteOffset;
    assert(OffsetPre % Scale == 0);

    if (RPI.isScalable())
      ScalableByteOffset += StackFillDir * Scale;
    else
      ByteOffset += StackFillDir * (RPI.isPaired() ? 2 * Scale : Scale);

    // Swift's async context is directly before FP, so allocate an extra
    // 8 bytes for it.
    if (NeedsFrameRecord && AFI->hasSwiftAsyncContext() &&
        RPI.Reg2 == AArch64::FP)
      ByteOffset += StackFillDir * 8;

    assert(!(RPI.isScalable() && RPI.isPaired()) &&
           "Paired spill/fill instructions don't exist for SVE vectors");

    // Round up size of non-pair to pair size if we need to pad the
    // callee-save area to ensure 16-byte alignment.
    if (NeedGapToAlignStack && !NeedsWinCFI &&
        !RPI.isScalable() && RPI.Type != RegPairInfo::FPR128 &&
        !RPI.isPaired() && ByteOffset % 16 != 0) {
      ByteOffset += 8 * StackFillDir;
      assert(MFI.getObjectAlign(RPI.FrameIdx) <= Align(16));
      // A stack frame with a gap looks like this, bottom up:
      // d9, d8. x21, gap, x20, x19.
      // Set extra alignment on the x21 object to create the gap above it.
      MFI.setObjectAlignment(RPI.FrameIdx, Align(16));
      NeedGapToAlignStack = false;
    }

    int OffsetPost = RPI.isScalable() ? ScalableByteOffset : ByteOffset;
    assert(OffsetPost % Scale == 0);
    // If filling top down (default), we want the offset after incrementing it.
    // If fillibg bootom up (WinCFI) we need the original offset.
    int Offset = NeedsWinCFI ? OffsetPre : OffsetPost;

    // The FP, LR pair goes 8 bytes into our expanded 24-byte slot so that the
    // Swift context can directly precede FP.
    if (NeedsFrameRecord && AFI->hasSwiftAsyncContext() &&
        RPI.Reg2 == AArch64::FP)
      Offset += 8;
    RPI.Offset = Offset / Scale;

    assert(((!RPI.isScalable() && RPI.Offset >= -64 && RPI.Offset <= 63) ||
            (RPI.isScalable() && RPI.Offset >= -256 && RPI.Offset <= 255)) &&
           "Offset out of bounds for LDP/STP immediate");

    // Save the offset to frame record so that the FP register can point to the
    // innermost frame record (spilled FP and LR registers).
    if (NeedsFrameRecord && ((!IsWindows && RPI.Reg1 == AArch64::LR &&
                              RPI.Reg2 == AArch64::FP) ||
                             (IsWindows && RPI.Reg1 == AArch64::FP &&
                              RPI.Reg2 == AArch64::LR)))
      AFI->setCalleeSaveBaseToFrameRecordOffset(Offset);

    RegPairs.push_back(RPI);
    if (RPI.isPaired())
      i += RegInc;
  }
  if (NeedsWinCFI) {
    // If we need an alignment gap in the stack, align the topmost stack
    // object. A stack frame with a gap looks like this, bottom up:
    // x19, d8. d9, gap.
    // Set extra alignment on the topmost stack object (the first element in
    // CSI, which goes top down), to create the gap above it.
    if (AFI->hasCalleeSaveStackFreeSpace())
      MFI.setObjectAlignment(CSI[0].getFrameIdx(), Align(16));
    // We iterated bottom up over the registers; flip RegPairs back to top
    // down order.
    std::reverse(RegPairs.begin(), RegPairs.end());
  }
}

bool AArch64FrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  bool NeedsWinCFI = needsWinCFI(MF);
  DebugLoc DL;
  SmallVector<RegPairInfo, 8> RegPairs;

  computeCalleeSaveRegisterPairs(MF, CSI, TRI, RegPairs, hasFP(MF));

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  if (homogeneousPrologEpilog(MF)) {
    auto MIB = BuildMI(MBB, MI, DL, TII.get(AArch64::HOM_Prolog))
                   .setMIFlag(MachineInstr::FrameSetup);

    for (auto &RPI : RegPairs) {
      MIB.addReg(RPI.Reg1);
      MIB.addReg(RPI.Reg2);

      // Update register live in.
      if (!MRI.isReserved(RPI.Reg1))
        MBB.addLiveIn(RPI.Reg1);
      if (!MRI.isReserved(RPI.Reg2))
        MBB.addLiveIn(RPI.Reg2);
    }
    return true;
  }
  for (const RegPairInfo &RPI : llvm::reverse(RegPairs)) {
    unsigned Reg1 = RPI.Reg1;
    unsigned Reg2 = RPI.Reg2;
    unsigned StrOpc;

    // Issue sequence of spills for cs regs.  The first spill may be converted
    // to a pre-decrement store later by emitPrologue if the callee-save stack
    // area allocation can't be combined with the local stack area allocation.
    // For example:
    //    stp     x22, x21, [sp, #0]     // addImm(+0)
    //    stp     x20, x19, [sp, #16]    // addImm(+2)
    //    stp     fp, lr, [sp, #32]      // addImm(+4)
    // Rationale: This sequence saves uop updates compared to a sequence of
    // pre-increment spills like stp xi,xj,[sp,#-16]!
    // Note: Similar rationale and sequence for restores in epilog.
    unsigned Size;
    Align Alignment;
    switch (RPI.Type) {
    case RegPairInfo::GPR:
       StrOpc = RPI.isPaired() ? AArch64::STPXi : AArch64::STRXui;
       Size = 8;
       Alignment = Align(8);
       break;
    case RegPairInfo::FPR64:
       StrOpc = RPI.isPaired() ? AArch64::STPDi : AArch64::STRDui;
       Size = 8;
       Alignment = Align(8);
       break;
    case RegPairInfo::FPR128:
       StrOpc = RPI.isPaired() ? AArch64::STPQi : AArch64::STRQui;
       Size = 16;
       Alignment = Align(16);
       break;
    case RegPairInfo::ZPR:
       StrOpc = AArch64::STR_ZXI;
       Size = 16;
       Alignment = Align(16);
       break;
    case RegPairInfo::PPR:
       StrOpc = AArch64::STR_PXI;
       Size = 2;
       Alignment = Align(2);
       break;
    }
    LLVM_DEBUG(dbgs() << "CSR spill: (" << printReg(Reg1, TRI);
               if (RPI.isPaired()) dbgs() << ", " << printReg(Reg2, TRI);
               dbgs() << ") -> fi#(" << RPI.FrameIdx;
               if (RPI.isPaired()) dbgs() << ", " << RPI.FrameIdx + 1;
               dbgs() << ")\n");

    assert((!NeedsWinCFI || !(Reg1 == AArch64::LR && Reg2 == AArch64::FP)) &&
           "Windows unwdinding requires a consecutive (FP,LR) pair");
    // Windows unwind codes require consecutive registers if registers are
    // paired.  Make the switch here, so that the code below will save (x,x+1)
    // and not (x+1,x).
    unsigned FrameIdxReg1 = RPI.FrameIdx;
    unsigned FrameIdxReg2 = RPI.FrameIdx + 1;
    if (NeedsWinCFI && RPI.isPaired()) {
      std::swap(Reg1, Reg2);
      std::swap(FrameIdxReg1, FrameIdxReg2);
    }
    MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII.get(StrOpc));
    if (!MRI.isReserved(Reg1))
      MBB.addLiveIn(Reg1);
    if (RPI.isPaired()) {
      if (!MRI.isReserved(Reg2))
        MBB.addLiveIn(Reg2);
      MIB.addReg(Reg2, getPrologueDeath(MF, Reg2));
      MIB.addMemOperand(MF.getMachineMemOperand(
          MachinePointerInfo::getFixedStack(MF, FrameIdxReg2),
          MachineMemOperand::MOStore, Size, Alignment));
    }
    MIB.addReg(Reg1, getPrologueDeath(MF, Reg1))
        .addReg(AArch64::SP)
        .addImm(RPI.Offset) // [sp, #offset*scale],
                            // where factor*scale is implicit
        .setMIFlag(MachineInstr::FrameSetup);
    MIB.addMemOperand(MF.getMachineMemOperand(
        MachinePointerInfo::getFixedStack(MF, FrameIdxReg1),
        MachineMemOperand::MOStore, Size, Alignment));
    if (NeedsWinCFI)
      InsertSEH(MIB, TII, MachineInstr::FrameSetup);

    // Update the StackIDs of the SVE stack slots.
    MachineFrameInfo &MFI = MF.getFrameInfo();
    if (RPI.Type == RegPairInfo::ZPR || RPI.Type == RegPairInfo::PPR)
      MFI.setStackID(RPI.FrameIdx, TargetStackID::ScalableVector);

  }
  return true;
}

bool AArch64FrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  DebugLoc DL;
  SmallVector<RegPairInfo, 8> RegPairs;
  bool NeedsWinCFI = needsWinCFI(MF);

  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();

  computeCalleeSaveRegisterPairs(MF, CSI, TRI, RegPairs, hasFP(MF));

  auto EmitMI = [&](const RegPairInfo &RPI) -> MachineBasicBlock::iterator {
    unsigned Reg1 = RPI.Reg1;
    unsigned Reg2 = RPI.Reg2;

    // Issue sequence of restores for cs regs. The last restore may be converted
    // to a post-increment load later by emitEpilogue if the callee-save stack
    // area allocation can't be combined with the local stack area allocation.
    // For example:
    //    ldp     fp, lr, [sp, #32]       // addImm(+4)
    //    ldp     x20, x19, [sp, #16]     // addImm(+2)
    //    ldp     x22, x21, [sp, #0]      // addImm(+0)
    // Note: see comment in spillCalleeSavedRegisters()
    unsigned LdrOpc;
    unsigned Size;
    Align Alignment;
    switch (RPI.Type) {
    case RegPairInfo::GPR:
       LdrOpc = RPI.isPaired() ? AArch64::LDPXi : AArch64::LDRXui;
       Size = 8;
       Alignment = Align(8);
       break;
    case RegPairInfo::FPR64:
       LdrOpc = RPI.isPaired() ? AArch64::LDPDi : AArch64::LDRDui;
       Size = 8;
       Alignment = Align(8);
       break;
    case RegPairInfo::FPR128:
       LdrOpc = RPI.isPaired() ? AArch64::LDPQi : AArch64::LDRQui;
       Size = 16;
       Alignment = Align(16);
       break;
    case RegPairInfo::ZPR:
       LdrOpc = AArch64::LDR_ZXI;
       Size = 16;
       Alignment = Align(16);
       break;
    case RegPairInfo::PPR:
       LdrOpc = AArch64::LDR_PXI;
       Size = 2;
       Alignment = Align(2);
       break;
    }
    LLVM_DEBUG(dbgs() << "CSR restore: (" << printReg(Reg1, TRI);
               if (RPI.isPaired()) dbgs() << ", " << printReg(Reg2, TRI);
               dbgs() << ") -> fi#(" << RPI.FrameIdx;
               if (RPI.isPaired()) dbgs() << ", " << RPI.FrameIdx + 1;
               dbgs() << ")\n");

    // Windows unwind codes require consecutive registers if registers are
    // paired.  Make the switch here, so that the code below will save (x,x+1)
    // and not (x+1,x).
    unsigned FrameIdxReg1 = RPI.FrameIdx;
    unsigned FrameIdxReg2 = RPI.FrameIdx + 1;
    if (NeedsWinCFI && RPI.isPaired()) {
      std::swap(Reg1, Reg2);
      std::swap(FrameIdxReg1, FrameIdxReg2);
    }
    MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII.get(LdrOpc));
    if (RPI.isPaired()) {
      MIB.addReg(Reg2, getDefRegState(true));
      MIB.addMemOperand(MF.getMachineMemOperand(
          MachinePointerInfo::getFixedStack(MF, FrameIdxReg2),
          MachineMemOperand::MOLoad, Size, Alignment));
    }
    MIB.addReg(Reg1, getDefRegState(true))
        .addReg(AArch64::SP)
        .addImm(RPI.Offset) // [sp, #offset*scale]
                            // where factor*scale is implicit
        .setMIFlag(MachineInstr::FrameDestroy);
    MIB.addMemOperand(MF.getMachineMemOperand(
        MachinePointerInfo::getFixedStack(MF, FrameIdxReg1),
        MachineMemOperand::MOLoad, Size, Alignment));
    if (NeedsWinCFI)
      InsertSEH(MIB, TII, MachineInstr::FrameDestroy);

    return MIB->getIterator();
  };

  // SVE objects are always restored in reverse order.
  for (const RegPairInfo &RPI : reverse(RegPairs))
    if (RPI.isScalable())
      EmitMI(RPI);

  if (homogeneousPrologEpilog(MF, &MBB)) {
    auto MIB = BuildMI(MBB, MBBI, DL, TII.get(AArch64::HOM_Epilog))
                   .setMIFlag(MachineInstr::FrameDestroy);
    for (auto &RPI : RegPairs) {
      MIB.addReg(RPI.Reg1, RegState::Define);
      MIB.addReg(RPI.Reg2, RegState::Define);
    }
    return true;
  }

  if (ReverseCSRRestoreSeq) {
    MachineBasicBlock::iterator First = MBB.end();
    for (const RegPairInfo &RPI : reverse(RegPairs)) {
      if (RPI.isScalable())
        continue;
      MachineBasicBlock::iterator It = EmitMI(RPI);
      if (First == MBB.end())
        First = It;
    }
    if (First != MBB.end())
      MBB.splice(MBBI, &MBB, First);
  } else {
    for (const RegPairInfo &RPI : RegPairs) {
      if (RPI.isScalable())
        continue;
      (void)EmitMI(RPI);
    }
  }

  return true;
}

void AArch64FrameLowering::determineCalleeSaves(MachineFunction &MF,
                                                BitVector &SavedRegs,
                                                RegScavenger *RS) const {
  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction().getCallingConv() == CallingConv::GHC)
    return;

  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  const AArch64RegisterInfo *RegInfo = static_cast<const AArch64RegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  unsigned UnspilledCSGPR = AArch64::NoRegister;
  unsigned UnspilledCSGPRPaired = AArch64::NoRegister;

  MachineFrameInfo &MFI = MF.getFrameInfo();
  const MCPhysReg *CSRegs = MF.getRegInfo().getCalleeSavedRegs();

  unsigned BasePointerReg = RegInfo->hasBasePointer(MF)
                                ? RegInfo->getBaseRegister()
                                : (unsigned)AArch64::NoRegister;

  unsigned ExtraCSSpill = 0;
  // Figure out which callee-saved registers to save/restore.
  for (unsigned i = 0; CSRegs[i]; ++i) {
    const unsigned Reg = CSRegs[i];

    // Add the base pointer register to SavedRegs if it is callee-save.
    if (Reg == BasePointerReg)
      SavedRegs.set(Reg);

    bool RegUsed = SavedRegs.test(Reg);
    unsigned PairedReg = AArch64::NoRegister;
    if (AArch64::GPR64RegClass.contains(Reg) ||
        AArch64::FPR64RegClass.contains(Reg) ||
        AArch64::FPR128RegClass.contains(Reg))
      PairedReg = CSRegs[i ^ 1];

    if (!RegUsed) {
      if (AArch64::GPR64RegClass.contains(Reg) &&
          !RegInfo->isReservedReg(MF, Reg)) {
        UnspilledCSGPR = Reg;
        UnspilledCSGPRPaired = PairedReg;
      }
      continue;
    }

    // MachO's compact unwind format relies on all registers being stored in
    // pairs.
    // FIXME: the usual format is actually better if unwinding isn't needed.
    if (producePairRegisters(MF) && PairedReg != AArch64::NoRegister &&
        !SavedRegs.test(PairedReg)) {
      SavedRegs.set(PairedReg);
      if (AArch64::GPR64RegClass.contains(PairedReg) &&
          !RegInfo->isReservedReg(MF, PairedReg))
        ExtraCSSpill = PairedReg;
    }
  }

  if (MF.getFunction().getCallingConv() == CallingConv::Win64 &&
      !Subtarget.isTargetWindows()) {
    // For Windows calling convention on a non-windows OS, where X18 is treated
    // as reserved, back up X18 when entering non-windows code (marked with the
    // Windows calling convention) and restore when returning regardless of
    // whether the individual function uses it - it might call other functions
    // that clobber it.
    SavedRegs.set(AArch64::X18);
  }

  // Calculates the callee saved stack size.
  unsigned CSStackSize = 0;
  unsigned SVECSStackSize = 0;
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (unsigned Reg : SavedRegs.set_bits()) {
    auto RegSize = TRI->getRegSizeInBits(Reg, MRI) / 8;
    if (AArch64::PPRRegClass.contains(Reg) ||
        AArch64::ZPRRegClass.contains(Reg))
      SVECSStackSize += RegSize;
    else
      CSStackSize += RegSize;
  }

  // Save number of saved regs, so we can easily update CSStackSize later.
  unsigned NumSavedRegs = SavedRegs.count();

  // The frame record needs to be created by saving the appropriate registers
  uint64_t EstimatedStackSize = MFI.estimateStackSize(MF);
  if (hasFP(MF) ||
      windowsRequiresStackProbe(MF, EstimatedStackSize + CSStackSize + 16)) {
    SavedRegs.set(AArch64::FP);
    SavedRegs.set(AArch64::LR);
  }

  LLVM_DEBUG(dbgs() << "*** determineCalleeSaves\nSaved CSRs:";
             for (unsigned Reg
                  : SavedRegs.set_bits()) dbgs()
             << ' ' << printReg(Reg, RegInfo);
             dbgs() << "\n";);

  // If any callee-saved registers are used, the frame cannot be eliminated.
  int64_t SVEStackSize =
      alignTo(SVECSStackSize + estimateSVEStackObjectOffsets(MFI), 16);
  bool CanEliminateFrame = (SavedRegs.count() == 0) && !SVEStackSize;

  // The CSR spill slots have not been allocated yet, so estimateStackSize
  // won't include them.
  unsigned EstimatedStackSizeLimit = estimateRSStackSizeLimit(MF);

  // Conservatively always assume BigStack when there are SVE spills.
  bool BigStack = SVEStackSize ||
                  (EstimatedStackSize + CSStackSize) > EstimatedStackSizeLimit;
  if (BigStack || !CanEliminateFrame || RegInfo->cannotEliminateFrame(MF))
    AFI->setHasStackFrame(true);

  // Estimate if we might need to scavenge a register at some point in order
  // to materialize a stack offset. If so, either spill one additional
  // callee-saved register or reserve a special spill slot to facilitate
  // register scavenging. If we already spilled an extra callee-saved register
  // above to keep the number of spills even, we don't need to do anything else
  // here.
  if (BigStack) {
    if (!ExtraCSSpill && UnspilledCSGPR != AArch64::NoRegister) {
      LLVM_DEBUG(dbgs() << "Spilling " << printReg(UnspilledCSGPR, RegInfo)
                        << " to get a scratch register.\n");
      SavedRegs.set(UnspilledCSGPR);
      // MachO's compact unwind format relies on all registers being stored in
      // pairs, so if we need to spill one extra for BigStack, then we need to
      // store the pair.
      if (producePairRegisters(MF))
        SavedRegs.set(UnspilledCSGPRPaired);
      ExtraCSSpill = UnspilledCSGPR;
    }

    // If we didn't find an extra callee-saved register to spill, create
    // an emergency spill slot.
    if (!ExtraCSSpill || MF.getRegInfo().isPhysRegUsed(ExtraCSSpill)) {
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      const TargetRegisterClass &RC = AArch64::GPR64RegClass;
      unsigned Size = TRI->getSpillSize(RC);
      Align Alignment = TRI->getSpillAlign(RC);
      int FI = MFI.CreateStackObject(Size, Alignment, false);
      RS->addScavengingFrameIndex(FI);
      LLVM_DEBUG(dbgs() << "No available CS registers, allocated fi#" << FI
                        << " as the emergency spill slot.\n");
    }
  }

  // Adding the size of additional 64bit GPR saves.
  CSStackSize += 8 * (SavedRegs.count() - NumSavedRegs);

  // A Swift asynchronous context extends the frame record with a pointer
  // directly before FP.
  if (hasFP(MF) && AFI->hasSwiftAsyncContext())
    CSStackSize += 8;

  uint64_t AlignedCSStackSize = alignTo(CSStackSize, 16);
  LLVM_DEBUG(dbgs() << "Estimated stack frame size: "
               << EstimatedStackSize + AlignedCSStackSize
               << " bytes.\n");

  assert((!MFI.isCalleeSavedInfoValid() ||
          AFI->getCalleeSavedStackSize() == AlignedCSStackSize) &&
         "Should not invalidate callee saved info");

  // Round up to register pair alignment to avoid additional SP adjustment
  // instructions.
  AFI->setCalleeSavedStackSize(AlignedCSStackSize);
  AFI->setCalleeSaveStackHasFreeSpace(AlignedCSStackSize != CSStackSize);
  AFI->setSVECalleeSavedStackSize(alignTo(SVECSStackSize, 16));
}

bool AArch64FrameLowering::assignCalleeSavedSpillSlots(
    MachineFunction &MF, const TargetRegisterInfo *RegInfo,
    std::vector<CalleeSavedInfo> &CSI, unsigned &MinCSFrameIndex,
    unsigned &MaxCSFrameIndex) const {
  bool NeedsWinCFI = needsWinCFI(MF);
  // To match the canonical windows frame layout, reverse the list of
  // callee saved registers to get them laid out by PrologEpilogInserter
  // in the right order. (PrologEpilogInserter allocates stack objects top
  // down. Windows canonical prologs store higher numbered registers at
  // the top, thus have the CSI array start from the highest registers.)
  if (NeedsWinCFI)
    std::reverse(CSI.begin(), CSI.end());

  if (CSI.empty())
    return true; // Early exit if no callee saved registers are modified!

  // Now that we know which registers need to be saved and restored, allocate
  // stack slots for them.
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *AFI = MF.getInfo<AArch64FunctionInfo>();
  for (auto &CS : CSI) {
    Register Reg = CS.getReg();
    const TargetRegisterClass *RC = RegInfo->getMinimalPhysRegClass(Reg);

    unsigned Size = RegInfo->getSpillSize(*RC);
    Align Alignment(RegInfo->getSpillAlign(*RC));
    int FrameIdx = MFI.CreateStackObject(Size, Alignment, true);
    CS.setFrameIdx(FrameIdx);

    if ((unsigned)FrameIdx < MinCSFrameIndex) MinCSFrameIndex = FrameIdx;
    if ((unsigned)FrameIdx > MaxCSFrameIndex) MaxCSFrameIndex = FrameIdx;

    // Grab 8 bytes below FP for the extended asynchronous frame info.
    if (hasFP(MF) && AFI->hasSwiftAsyncContext() && Reg == AArch64::FP) {
      FrameIdx = MFI.CreateStackObject(8, Alignment, true);
      AFI->setSwiftAsyncContextFrameIdx(FrameIdx);
      if ((unsigned)FrameIdx < MinCSFrameIndex) MinCSFrameIndex = FrameIdx;
      if ((unsigned)FrameIdx > MaxCSFrameIndex) MaxCSFrameIndex = FrameIdx;
    }
  }
  return true;
}

bool AArch64FrameLowering::enableStackSlotScavenging(
    const MachineFunction &MF) const {
  const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  return AFI->hasCalleeSaveStackFreeSpace();
}

/// returns true if there are any SVE callee saves.
static bool getSVECalleeSaveSlotRange(const MachineFrameInfo &MFI,
                                      int &Min, int &Max) {
  Min = std::numeric_limits<int>::max();
  Max = std::numeric_limits<int>::min();

  if (!MFI.isCalleeSavedInfoValid())
    return false;

  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  for (auto &CS : CSI) {
    if (AArch64::ZPRRegClass.contains(CS.getReg()) ||
        AArch64::PPRRegClass.contains(CS.getReg())) {
      assert((Max == std::numeric_limits<int>::min() ||
              Max + 1 == CS.getFrameIdx()) &&
             "SVE CalleeSaves are not consecutive");

      Min = std::min(Min, CS.getFrameIdx());
      Max = std::max(Max, CS.getFrameIdx());
    }
  }
  return Min != std::numeric_limits<int>::max();
}

// Process all the SVE stack objects and determine offsets for each
// object. If AssignOffsets is true, the offsets get assigned.
// Fills in the first and last callee-saved frame indices into
// Min/MaxCSFrameIndex, respectively.
// Returns the size of the stack.
static int64_t determineSVEStackObjectOffsets(MachineFrameInfo &MFI,
                                              int &MinCSFrameIndex,
                                              int &MaxCSFrameIndex,
                                              bool AssignOffsets) {
#ifndef NDEBUG
  // First process all fixed stack objects.
  for (int I = MFI.getObjectIndexBegin(); I != 0; ++I)
    assert(MFI.getStackID(I) != TargetStackID::ScalableVector &&
           "SVE vectors should never be passed on the stack by value, only by "
           "reference.");
#endif

  auto Assign = [&MFI](int FI, int64_t Offset) {
    LLVM_DEBUG(dbgs() << "alloc FI(" << FI << ") at SP[" << Offset << "]\n");
    MFI.setObjectOffset(FI, Offset);
  };

  int64_t Offset = 0;

  // Then process all callee saved slots.
  if (getSVECalleeSaveSlotRange(MFI, MinCSFrameIndex, MaxCSFrameIndex)) {
    // Assign offsets to the callee save slots.
    for (int I = MinCSFrameIndex; I <= MaxCSFrameIndex; ++I) {
      Offset += MFI.getObjectSize(I);
      Offset = alignTo(Offset, MFI.getObjectAlign(I));
      if (AssignOffsets)
        Assign(I, -Offset);
    }
  }

  // Ensure that the Callee-save area is aligned to 16bytes.
  Offset = alignTo(Offset, Align(16U));

  // Create a buffer of SVE objects to allocate and sort it.
  SmallVector<int, 8> ObjectsToAllocate;
  // If we have a stack protector, and we've previously decided that we have SVE
  // objects on the stack and thus need it to go in the SVE stack area, then it
  // needs to go first.
  int StackProtectorFI = -1;
  if (MFI.hasStackProtectorIndex()) {
    StackProtectorFI = MFI.getStackProtectorIndex();
    if (MFI.getStackID(StackProtectorFI) == TargetStackID::ScalableVector)
      ObjectsToAllocate.push_back(StackProtectorFI);
  }
  for (int I = 0, E = MFI.getObjectIndexEnd(); I != E; ++I) {
    unsigned StackID = MFI.getStackID(I);
    if (StackID != TargetStackID::ScalableVector)
      continue;
    if (I == StackProtectorFI)
      continue;
    if (MaxCSFrameIndex >= I && I >= MinCSFrameIndex)
      continue;
    if (MFI.isDeadObjectIndex(I))
      continue;

    ObjectsToAllocate.push_back(I);
  }

  // Allocate all SVE locals and spills
  for (unsigned FI : ObjectsToAllocate) {
    Align Alignment = MFI.getObjectAlign(FI);
    // FIXME: Given that the length of SVE vectors is not necessarily a power of
    // two, we'd need to align every object dynamically at runtime if the
    // alignment is larger than 16. This is not yet supported.
    if (Alignment > Align(16))
      report_fatal_error(
          "Alignment of scalable vectors > 16 bytes is not yet supported");

    Offset = alignTo(Offset + MFI.getObjectSize(FI), Alignment);
    if (AssignOffsets)
      Assign(FI, -Offset);
  }

  return Offset;
}

int64_t AArch64FrameLowering::estimateSVEStackObjectOffsets(
    MachineFrameInfo &MFI) const {
  int MinCSFrameIndex, MaxCSFrameIndex;
  return determineSVEStackObjectOffsets(MFI, MinCSFrameIndex, MaxCSFrameIndex, false);
}

int64_t AArch64FrameLowering::assignSVEStackObjectOffsets(
    MachineFrameInfo &MFI, int &MinCSFrameIndex, int &MaxCSFrameIndex) const {
  return determineSVEStackObjectOffsets(MFI, MinCSFrameIndex, MaxCSFrameIndex,
                                        true);
}

void AArch64FrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();

  assert(getStackGrowthDirection() == TargetFrameLowering::StackGrowsDown &&
         "Upwards growing stack unsupported");

  int MinCSFrameIndex, MaxCSFrameIndex;
  int64_t SVEStackSize =
      assignSVEStackObjectOffsets(MFI, MinCSFrameIndex, MaxCSFrameIndex);

  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  AFI->setStackSizeSVE(alignTo(SVEStackSize, 16U));
  AFI->setMinMaxSVECSFrameIndex(MinCSFrameIndex, MaxCSFrameIndex);

  // If this function isn't doing Win64-style C++ EH, we don't need to do
  // anything.
  if (!MF.hasEHFunclets())
    return;
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  WinEHFuncInfo &EHInfo = *MF.getWinEHFuncInfo();

  MachineBasicBlock &MBB = MF.front();
  auto MBBI = MBB.begin();
  while (MBBI != MBB.end() && MBBI->getFlag(MachineInstr::FrameSetup))
    ++MBBI;

  // Create an UnwindHelp object.
  // The UnwindHelp object is allocated at the start of the fixed object area
  int64_t FixedObject =
      getFixedObjectSize(MF, AFI, /*IsWin64*/ true, /*IsFunclet*/ false);
  int UnwindHelpFI = MFI.CreateFixedObject(/*Size*/ 8,
                                           /*SPOffset*/ -FixedObject,
                                           /*IsImmutable=*/false);
  EHInfo.UnwindHelpFrameIdx = UnwindHelpFI;

  // We need to store -2 into the UnwindHelp object at the start of the
  // function.
  DebugLoc DL;
  RS->enterBasicBlockEnd(MBB);
  RS->backward(std::prev(MBBI));
  Register DstReg = RS->FindUnusedReg(&AArch64::GPR64commonRegClass);
  assert(DstReg && "There must be a free register after frame setup");
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::MOVi64imm), DstReg).addImm(-2);
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::STURXi))
      .addReg(DstReg, getKillRegState(true))
      .addFrameIndex(UnwindHelpFI)
      .addImm(0);
}

namespace {
struct TagStoreInstr {
  MachineInstr *MI;
  int64_t Offset, Size;
  explicit TagStoreInstr(MachineInstr *MI, int64_t Offset, int64_t Size)
      : MI(MI), Offset(Offset), Size(Size) {}
};

class TagStoreEdit {
  MachineFunction *MF;
  MachineBasicBlock *MBB;
  MachineRegisterInfo *MRI;
  // Tag store instructions that are being replaced.
  SmallVector<TagStoreInstr, 8> TagStores;
  // Combined memref arguments of the above instructions.
  SmallVector<MachineMemOperand *, 8> CombinedMemRefs;

  // Replace allocation tags in [FrameReg + FrameRegOffset, FrameReg +
  // FrameRegOffset + Size) with the address tag of SP.
  Register FrameReg;
  StackOffset FrameRegOffset;
  int64_t Size;
  // If not None, move FrameReg to (FrameReg + FrameRegUpdate) at the end.
  Optional<int64_t> FrameRegUpdate;
  // MIFlags for any FrameReg updating instructions.
  unsigned FrameRegUpdateFlags;

  // Use zeroing instruction variants.
  bool ZeroData;
  DebugLoc DL;

  void emitUnrolled(MachineBasicBlock::iterator InsertI);
  void emitLoop(MachineBasicBlock::iterator InsertI);

public:
  TagStoreEdit(MachineBasicBlock *MBB, bool ZeroData)
      : MBB(MBB), ZeroData(ZeroData) {
    MF = MBB->getParent();
    MRI = &MF->getRegInfo();
  }
  // Add an instruction to be replaced. Instructions must be added in the
  // ascending order of Offset, and have to be adjacent.
  void addInstruction(TagStoreInstr I) {
    assert((TagStores.empty() ||
            TagStores.back().Offset + TagStores.back().Size == I.Offset) &&
           "Non-adjacent tag store instructions.");
    TagStores.push_back(I);
  }
  void clear() { TagStores.clear(); }
  // Emit equivalent code at the given location, and erase the current set of
  // instructions. May skip if the replacement is not profitable. May invalidate
  // the input iterator and replace it with a valid one.
  void emitCode(MachineBasicBlock::iterator &InsertI,
                const AArch64FrameLowering *TFI, bool TryMergeSPUpdate);
};

void TagStoreEdit::emitUnrolled(MachineBasicBlock::iterator InsertI) {
  const AArch64InstrInfo *TII =
      MF->getSubtarget<AArch64Subtarget>().getInstrInfo();

  const int64_t kMinOffset = -256 * 16;
  const int64_t kMaxOffset = 255 * 16;

  Register BaseReg = FrameReg;
  int64_t BaseRegOffsetBytes = FrameRegOffset.getFixed();
  if (BaseRegOffsetBytes < kMinOffset ||
      BaseRegOffsetBytes + (Size - Size % 32) > kMaxOffset) {
    Register ScratchReg = MRI->createVirtualRegister(&AArch64::GPR64RegClass);
    emitFrameOffset(*MBB, InsertI, DL, ScratchReg, BaseReg,
                    StackOffset::getFixed(BaseRegOffsetBytes), TII);
    BaseReg = ScratchReg;
    BaseRegOffsetBytes = 0;
  }

  MachineInstr *LastI = nullptr;
  while (Size) {
    int64_t InstrSize = (Size > 16) ? 32 : 16;
    unsigned Opcode =
        InstrSize == 16
            ? (ZeroData ? AArch64::STZGOffset : AArch64::STGOffset)
            : (ZeroData ? AArch64::STZ2GOffset : AArch64::ST2GOffset);
    MachineInstr *I = BuildMI(*MBB, InsertI, DL, TII->get(Opcode))
                          .addReg(AArch64::SP)
                          .addReg(BaseReg)
                          .addImm(BaseRegOffsetBytes / 16)
                          .setMemRefs(CombinedMemRefs);
    // A store to [BaseReg, #0] should go last for an opportunity to fold the
    // final SP adjustment in the epilogue.
    if (BaseRegOffsetBytes == 0)
      LastI = I;
    BaseRegOffsetBytes += InstrSize;
    Size -= InstrSize;
  }

  if (LastI)
    MBB->splice(InsertI, MBB, LastI);
}

void TagStoreEdit::emitLoop(MachineBasicBlock::iterator InsertI) {
  const AArch64InstrInfo *TII =
      MF->getSubtarget<AArch64Subtarget>().getInstrInfo();

  Register BaseReg = FrameRegUpdate
                         ? FrameReg
                         : MRI->createVirtualRegister(&AArch64::GPR64RegClass);
  Register SizeReg = MRI->createVirtualRegister(&AArch64::GPR64RegClass);

  emitFrameOffset(*MBB, InsertI, DL, BaseReg, FrameReg, FrameRegOffset, TII);

  int64_t LoopSize = Size;
  // If the loop size is not a multiple of 32, split off one 16-byte store at
  // the end to fold BaseReg update into.
  if (FrameRegUpdate && *FrameRegUpdate)
    LoopSize -= LoopSize % 32;
  MachineInstr *LoopI = BuildMI(*MBB, InsertI, DL,
                                TII->get(ZeroData ? AArch64::STZGloop_wback
                                                  : AArch64::STGloop_wback))
                            .addDef(SizeReg)
                            .addDef(BaseReg)
                            .addImm(LoopSize)
                            .addReg(BaseReg)
                            .setMemRefs(CombinedMemRefs);
  if (FrameRegUpdate)
    LoopI->setFlags(FrameRegUpdateFlags);

  int64_t ExtraBaseRegUpdate =
      FrameRegUpdate ? (*FrameRegUpdate - FrameRegOffset.getFixed() - Size) : 0;
  if (LoopSize < Size) {
    assert(FrameRegUpdate);
    assert(Size - LoopSize == 16);
    // Tag 16 more bytes at BaseReg and update BaseReg.
    BuildMI(*MBB, InsertI, DL,
            TII->get(ZeroData ? AArch64::STZGPostIndex : AArch64::STGPostIndex))
        .addDef(BaseReg)
        .addReg(BaseReg)
        .addReg(BaseReg)
        .addImm(1 + ExtraBaseRegUpdate / 16)
        .setMemRefs(CombinedMemRefs)
        .setMIFlags(FrameRegUpdateFlags);
  } else if (ExtraBaseRegUpdate) {
    // Update BaseReg.
    BuildMI(
        *MBB, InsertI, DL,
        TII->get(ExtraBaseRegUpdate > 0 ? AArch64::ADDXri : AArch64::SUBXri))
        .addDef(BaseReg)
        .addReg(BaseReg)
        .addImm(std::abs(ExtraBaseRegUpdate))
        .addImm(0)
        .setMIFlags(FrameRegUpdateFlags);
  }
}

// Check if *II is a register update that can be merged into STGloop that ends
// at (Reg + Size). RemainingOffset is the required adjustment to Reg after the
// end of the loop.
bool canMergeRegUpdate(MachineBasicBlock::iterator II, unsigned Reg,
                       int64_t Size, int64_t *TotalOffset) {
  MachineInstr &MI = *II;
  if ((MI.getOpcode() == AArch64::ADDXri ||
       MI.getOpcode() == AArch64::SUBXri) &&
      MI.getOperand(0).getReg() == Reg && MI.getOperand(1).getReg() == Reg) {
    unsigned Shift = AArch64_AM::getShiftValue(MI.getOperand(3).getImm());
    int64_t Offset = MI.getOperand(2).getImm() << Shift;
    if (MI.getOpcode() == AArch64::SUBXri)
      Offset = -Offset;
    int64_t AbsPostOffset = std::abs(Offset - Size);
    const int64_t kMaxOffset =
        0xFFF; // Max encoding for unshifted ADDXri / SUBXri
    if (AbsPostOffset <= kMaxOffset && AbsPostOffset % 16 == 0) {
      *TotalOffset = Offset;
      return true;
    }
  }
  return false;
}

void mergeMemRefs(const SmallVectorImpl<TagStoreInstr> &TSE,
                  SmallVectorImpl<MachineMemOperand *> &MemRefs) {
  MemRefs.clear();
  for (auto &TS : TSE) {
    MachineInstr *MI = TS.MI;
    // An instruction without memory operands may access anything. Be
    // conservative and return an empty list.
    if (MI->memoperands_empty()) {
      MemRefs.clear();
      return;
    }
    MemRefs.append(MI->memoperands_begin(), MI->memoperands_end());
  }
}

void TagStoreEdit::emitCode(MachineBasicBlock::iterator &InsertI,
                            const AArch64FrameLowering *TFI,
                            bool TryMergeSPUpdate) {
  if (TagStores.empty())
    return;
  TagStoreInstr &FirstTagStore = TagStores[0];
  TagStoreInstr &LastTagStore = TagStores[TagStores.size() - 1];
  Size = LastTagStore.Offset - FirstTagStore.Offset + LastTagStore.Size;
  DL = TagStores[0].MI->getDebugLoc();

  Register Reg;
  FrameRegOffset = TFI->resolveFrameOffsetReference(
      *MF, FirstTagStore.Offset, false /*isFixed*/, false /*isSVE*/, Reg,
      /*PreferFP=*/false, /*ForSimm=*/true);
  FrameReg = Reg;
  FrameRegUpdate = None;

  mergeMemRefs(TagStores, CombinedMemRefs);

  LLVM_DEBUG(dbgs() << "Replacing adjacent STG instructions:\n";
             for (const auto &Instr
                  : TagStores) { dbgs() << "  " << *Instr.MI; });

  // Size threshold where a loop becomes shorter than a linear sequence of
  // tagging instructions.
  const int kSetTagLoopThreshold = 176;
  if (Size < kSetTagLoopThreshold) {
    if (TagStores.size() < 2)
      return;
    emitUnrolled(InsertI);
  } else {
    MachineInstr *UpdateInstr = nullptr;
    int64_t TotalOffset = 0;
    if (TryMergeSPUpdate) {
      // See if we can merge base register update into the STGloop.
      // This is done in AArch64LoadStoreOptimizer for "normal" stores,
      // but STGloop is way too unusual for that, and also it only
      // realistically happens in function epilogue. Also, STGloop is expanded
      // before that pass.
      if (InsertI != MBB->end() &&
          canMergeRegUpdate(InsertI, FrameReg, FrameRegOffset.getFixed() + Size,
                            &TotalOffset)) {
        UpdateInstr = &*InsertI++;
        LLVM_DEBUG(dbgs() << "Folding SP update into loop:\n  "
                          << *UpdateInstr);
      }
    }

    if (!UpdateInstr && TagStores.size() < 2)
      return;

    if (UpdateInstr) {
      FrameRegUpdate = TotalOffset;
      FrameRegUpdateFlags = UpdateInstr->getFlags();
    }
    emitLoop(InsertI);
    if (UpdateInstr)
      UpdateInstr->eraseFromParent();
  }

  for (auto &TS : TagStores)
    TS.MI->eraseFromParent();
}

bool isMergeableStackTaggingInstruction(MachineInstr &MI, int64_t &Offset,
                                        int64_t &Size, bool &ZeroData) {
  MachineFunction &MF = *MI.getParent()->getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  unsigned Opcode = MI.getOpcode();
  ZeroData = (Opcode == AArch64::STZGloop || Opcode == AArch64::STZGOffset ||
              Opcode == AArch64::STZ2GOffset);

  if (Opcode == AArch64::STGloop || Opcode == AArch64::STZGloop) {
    if (!MI.getOperand(0).isDead() || !MI.getOperand(1).isDead())
      return false;
    if (!MI.getOperand(2).isImm() || !MI.getOperand(3).isFI())
      return false;
    Offset = MFI.getObjectOffset(MI.getOperand(3).getIndex());
    Size = MI.getOperand(2).getImm();
    return true;
  }

  if (Opcode == AArch64::STGOffset || Opcode == AArch64::STZGOffset)
    Size = 16;
  else if (Opcode == AArch64::ST2GOffset || Opcode == AArch64::STZ2GOffset)
    Size = 32;
  else
    return false;

  if (MI.getOperand(0).getReg() != AArch64::SP || !MI.getOperand(1).isFI())
    return false;

  Offset = MFI.getObjectOffset(MI.getOperand(1).getIndex()) +
           16 * MI.getOperand(2).getImm();
  return true;
}

// Detect a run of memory tagging instructions for adjacent stack frame slots,
// and replace them with a shorter instruction sequence:
// * replace STG + STG with ST2G
// * replace STGloop + STGloop with STGloop
// This code needs to run when stack slot offsets are already known, but before
// FrameIndex operands in STG instructions are eliminated.
MachineBasicBlock::iterator tryMergeAdjacentSTG(MachineBasicBlock::iterator II,
                                                const AArch64FrameLowering *TFI,
                                                RegScavenger *RS) {
  bool FirstZeroData;
  int64_t Size, Offset;
  MachineInstr &MI = *II;
  MachineBasicBlock *MBB = MI.getParent();
  MachineBasicBlock::iterator NextI = ++II;
  if (&MI == &MBB->instr_back())
    return II;
  if (!isMergeableStackTaggingInstruction(MI, Offset, Size, FirstZeroData))
    return II;

  SmallVector<TagStoreInstr, 4> Instrs;
  Instrs.emplace_back(&MI, Offset, Size);

  constexpr int kScanLimit = 10;
  int Count = 0;
  for (MachineBasicBlock::iterator E = MBB->end();
       NextI != E && Count < kScanLimit; ++NextI) {
    MachineInstr &MI = *NextI;
    bool ZeroData;
    int64_t Size, Offset;
    // Collect instructions that update memory tags with a FrameIndex operand
    // and (when applicable) constant size, and whose output registers are dead
    // (the latter is almost always the case in practice). Since these
    // instructions effectively have no inputs or outputs, we are free to skip
    // any non-aliasing instructions in between without tracking used registers.
    if (isMergeableStackTaggingInstruction(MI, Offset, Size, ZeroData)) {
      if (ZeroData != FirstZeroData)
        break;
      Instrs.emplace_back(&MI, Offset, Size);
      continue;
    }

    // Only count non-transient, non-tagging instructions toward the scan
    // limit.
    if (!MI.isTransient())
      ++Count;

    // Just in case, stop before the epilogue code starts.
    if (MI.getFlag(MachineInstr::FrameSetup) ||
        MI.getFlag(MachineInstr::FrameDestroy))
      break;

    // Reject anything that may alias the collected instructions.
    if (MI.mayLoadOrStore() || MI.hasUnmodeledSideEffects())
      break;
  }

  // New code will be inserted after the last tagging instruction we've found.
  MachineBasicBlock::iterator InsertI = Instrs.back().MI;
  InsertI++;

  llvm::stable_sort(Instrs,
                    [](const TagStoreInstr &Left, const TagStoreInstr &Right) {
                      return Left.Offset < Right.Offset;
                    });

  // Make sure that we don't have any overlapping stores.
  int64_t CurOffset = Instrs[0].Offset;
  for (auto &Instr : Instrs) {
    if (CurOffset > Instr.Offset)
      return NextI;
    CurOffset = Instr.Offset + Instr.Size;
  }

  // Find contiguous runs of tagged memory and emit shorter instruction
  // sequencies for them when possible.
  TagStoreEdit TSE(MBB, FirstZeroData);
  Optional<int64_t> EndOffset;
  for (auto &Instr : Instrs) {
    if (EndOffset && *EndOffset != Instr.Offset) {
      // Found a gap.
      TSE.emitCode(InsertI, TFI, /*TryMergeSPUpdate = */ false);
      TSE.clear();
    }

    TSE.addInstruction(Instr);
    EndOffset = Instr.Offset + Instr.Size;
  }

  // Multiple FP/SP updates in a loop cannot be described by CFI instructions.
  TSE.emitCode(InsertI, TFI, /*TryMergeSPUpdate = */
               !MBB->getParent()
                    ->getInfo<AArch64FunctionInfo>()
                    ->needsAsyncDwarfUnwindInfo());

  return InsertI;
}
} // namespace

void AArch64FrameLowering::processFunctionBeforeFrameIndicesReplaced(
    MachineFunction &MF, RegScavenger *RS = nullptr) const {
  if (StackTaggingMergeSetTag)
    for (auto &BB : MF)
      for (MachineBasicBlock::iterator II = BB.begin(); II != BB.end();)
        II = tryMergeAdjacentSTG(II, this, RS);
}

/// For Win64 AArch64 EH, the offset to the Unwind object is from the SP
/// before the update.  This is easily retrieved as it is exactly the offset
/// that is set in processFunctionBeforeFrameFinalized.
StackOffset AArch64FrameLowering::getFrameIndexReferencePreferSP(
    const MachineFunction &MF, int FI, Register &FrameReg,
    bool IgnoreSPUpdates) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  if (IgnoreSPUpdates) {
    LLVM_DEBUG(dbgs() << "Offset from the SP for " << FI << " is "
                      << MFI.getObjectOffset(FI) << "\n");
    FrameReg = AArch64::SP;
    return StackOffset::getFixed(MFI.getObjectOffset(FI));
  }

  // Go to common code if we cannot provide sp + offset.
  if (MFI.hasVarSizedObjects() ||
      MF.getInfo<AArch64FunctionInfo>()->getStackSizeSVE() ||
      MF.getSubtarget().getRegisterInfo()->hasStackRealignment(MF))
    return getFrameIndexReference(MF, FI, FrameReg);

  FrameReg = AArch64::SP;
  return getStackOffset(MF, MFI.getObjectOffset(FI));
}

/// The parent frame offset (aka dispFrame) is only used on X86_64 to retrieve
/// the parent's frame pointer
unsigned AArch64FrameLowering::getWinEHParentFrameOffset(
    const MachineFunction &MF) const {
  return 0;
}

/// Funclets only need to account for space for the callee saved registers,
/// as the locals are accounted for in the parent's stack frame.
unsigned AArch64FrameLowering::getWinEHFuncletFrameSize(
    const MachineFunction &MF) const {
  // This is the size of the pushed CSRs.
  unsigned CSSize =
      MF.getInfo<AArch64FunctionInfo>()->getCalleeSavedStackSize();
  // This is the amount of stack a funclet needs to allocate.
  return alignTo(CSSize + MF.getFrameInfo().getMaxCallFrameSize(),
                 getStackAlign());
}

namespace {
struct FrameObject {
  bool IsValid = false;
  // Index of the object in MFI.
  int ObjectIndex = 0;
  // Group ID this object belongs to.
  int GroupIndex = -1;
  // This object should be placed first (closest to SP).
  bool ObjectFirst = false;
  // This object's group (which always contains the object with
  // ObjectFirst==true) should be placed first.
  bool GroupFirst = false;
};

class GroupBuilder {
  SmallVector<int, 8> CurrentMembers;
  int NextGroupIndex = 0;
  std::vector<FrameObject> &Objects;

public:
  GroupBuilder(std::vector<FrameObject> &Objects) : Objects(Objects) {}
  void AddMember(int Index) { CurrentMembers.push_back(Index); }
  void EndCurrentGroup() {
    if (CurrentMembers.size() > 1) {
      // Create a new group with the current member list. This might remove them
      // from their pre-existing groups. That's OK, dealing with overlapping
      // groups is too hard and unlikely to make a difference.
      LLVM_DEBUG(dbgs() << "group:");
      for (int Index : CurrentMembers) {
        Objects[Index].GroupIndex = NextGroupIndex;
        LLVM_DEBUG(dbgs() << " " << Index);
      }
      LLVM_DEBUG(dbgs() << "\n");
      NextGroupIndex++;
    }
    CurrentMembers.clear();
  }
};

bool FrameObjectCompare(const FrameObject &A, const FrameObject &B) {
  // Objects at a lower index are closer to FP; objects at a higher index are
  // closer to SP.
  //
  // For consistency in our comparison, all invalid objects are placed
  // at the end. This also allows us to stop walking when we hit the
  // first invalid item after it's all sorted.
  //
  // The "first" object goes first (closest to SP), followed by the members of
  // the "first" group.
  //
  // The rest are sorted by the group index to keep the groups together.
  // Higher numbered groups are more likely to be around longer (i.e. untagged
  // in the function epilogue and not at some earlier point). Place them closer
  // to SP.
  //
  // If all else equal, sort by the object index to keep the objects in the
  // original order.
  return std::make_tuple(!A.IsValid, A.ObjectFirst, A.GroupFirst, A.GroupIndex,
                         A.ObjectIndex) <
         std::make_tuple(!B.IsValid, B.ObjectFirst, B.GroupFirst, B.GroupIndex,
                         B.ObjectIndex);
}
} // namespace

void AArch64FrameLowering::orderFrameObjects(
    const MachineFunction &MF, SmallVectorImpl<int> &ObjectsToAllocate) const {
  if (!OrderFrameObjects || ObjectsToAllocate.empty())
    return;

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  std::vector<FrameObject> FrameObjects(MFI.getObjectIndexEnd());
  for (auto &Obj : ObjectsToAllocate) {
    FrameObjects[Obj].IsValid = true;
    FrameObjects[Obj].ObjectIndex = Obj;
  }

  // Identify stack slots that are tagged at the same time.
  GroupBuilder GB(FrameObjects);
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      int OpIndex;
      switch (MI.getOpcode()) {
      case AArch64::STGloop:
      case AArch64::STZGloop:
        OpIndex = 3;
        break;
      case AArch64::STGOffset:
      case AArch64::STZGOffset:
      case AArch64::ST2GOffset:
      case AArch64::STZ2GOffset:
        OpIndex = 1;
        break;
      default:
        OpIndex = -1;
      }

      int TaggedFI = -1;
      if (OpIndex >= 0) {
        const MachineOperand &MO = MI.getOperand(OpIndex);
        if (MO.isFI()) {
          int FI = MO.getIndex();
          if (FI >= 0 && FI < MFI.getObjectIndexEnd() &&
              FrameObjects[FI].IsValid)
            TaggedFI = FI;
        }
      }

      // If this is a stack tagging instruction for a slot that is not part of a
      // group yet, either start a new group or add it to the current one.
      if (TaggedFI >= 0)
        GB.AddMember(TaggedFI);
      else
        GB.EndCurrentGroup();
    }
    // Groups should never span multiple basic blocks.
    GB.EndCurrentGroup();
  }

  // If the function's tagged base pointer is pinned to a stack slot, we want to
  // put that slot first when possible. This will likely place it at SP + 0,
  // and save one instruction when generating the base pointer because IRG does
  // not allow an immediate offset.
  const AArch64FunctionInfo &AFI = *MF.getInfo<AArch64FunctionInfo>();
  Optional<int> TBPI = AFI.getTaggedBasePointerIndex();
  if (TBPI) {
    FrameObjects[*TBPI].ObjectFirst = true;
    FrameObjects[*TBPI].GroupFirst = true;
    int FirstGroupIndex = FrameObjects[*TBPI].GroupIndex;
    if (FirstGroupIndex >= 0)
      for (FrameObject &Object : FrameObjects)
        if (Object.GroupIndex == FirstGroupIndex)
          Object.GroupFirst = true;
  }

  llvm::stable_sort(FrameObjects, FrameObjectCompare);

  int i = 0;
  for (auto &Obj : FrameObjects) {
    // All invalid items are sorted at the end, so it's safe to stop.
    if (!Obj.IsValid)
      break;
    ObjectsToAllocate[i++] = Obj.ObjectIndex;
  }

  LLVM_DEBUG(dbgs() << "Final frame order:\n"; for (auto &Obj
                                                    : FrameObjects) {
    if (!Obj.IsValid)
      break;
    dbgs() << "  " << Obj.ObjectIndex << ": group " << Obj.GroupIndex;
    if (Obj.ObjectFirst)
      dbgs() << ", first";
    if (Obj.GroupFirst)
      dbgs() << ", group-first";
    dbgs() << "\n";
  });
}
