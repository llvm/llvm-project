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
// | (Win64 only) callee-saved SVE reg |
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
//        Default SVE stack layout                 Split SVE objects
//   (aarch64-split-sve-objects=false)      (aarch64-split-sve-objects=true)
// |-----------------------------------|  |-----------------------------------|
// |         <hazard padding>          |  | callee-saved PPR registers        |
// |-----------------------------------|  |-----------------------------------|
// |                                   |  |         PPR stack objects         |
// | callee-saved fp/simd/SVE regs     |  |-----------------------------------|
// |                                   |  |         <hazard padding>          |
// |-----------------------------------|  |-----------------------------------|
// |                                   |  | callee-saved ZPR/FPR registers    |
// |        SVE stack objects          |  |-----------------------------------|
// |                                   |  |         ZPR stack objects         |
// |-----------------------------------|  |-----------------------------------|
//                                         ^ NB: FPR CSRs are promoted to ZPRs
// |-----------------------------------|
// |.empty.space.to.make.part.below....|
// |.aligned.in.case.it.needs.more.than| (size of this area is unknown at
// |.the.standard.16-byte.alignment....|  compile time; if present)
// |-----------------------------------|
// | local variables of fixed size     |
// | including spill slots             |
// |   <FPR>                           |
// |   <hazard padding>                |
// |   <GPR>                           |
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
// About stack hazards: Under some SME contexts, a coprocessor with its own
// separate cache can used for FP operations. This can create hazards if the CPU
// and the SME unit try to access the same area of memory, including if the
// access is to an area of the stack. To try to alleviate this we attempt to
// introduce extra padding into the stack frame between FP and GPR accesses,
// controlled by the aarch64-stack-hazard-size option. Without changing the
// layout of the stack frame in the diagram above, a stack object of size
// aarch64-stack-hazard-size is added between GPR and FPR CSRs. Another is added
// to the stack objects section, and stack objects are sorted so that FPR >
// Hazard padding slot > GPRs (where possible). Unfortunately some things are
// not handled well (VLA area, arguments on the stack, objects with both GPR and
// FPR accesses), but if those are controlled by the user then the entire stack
// frame becomes GPR at the start/end with FPR in the middle, surrounded by
// Hazard padding.
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
#include "AArch64PrologueEpilogue.h"
#include "AArch64RegisterInfo.h"
#include "AArch64SMEAttributes.h"
#include "AArch64Subtarget.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/CFIInstBuilder.h"
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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "frame-info"

static cl::opt<bool> EnableRedZone("aarch64-redzone",
                                   cl::desc("enable use of redzone on AArch64"),
                                   cl::init(false), cl::Hidden);

static cl::opt<bool> StackTaggingMergeSetTag(
    "stack-tagging-merge-settag",
    cl::desc("merge settag instruction in function epilog"), cl::init(true),
    cl::Hidden);

static cl::opt<bool> OrderFrameObjects("aarch64-order-frame-objects",
                                       cl::desc("sort stack allocations"),
                                       cl::init(true), cl::Hidden);

static cl::opt<bool>
    SplitSVEObjects("aarch64-split-sve-objects",
                    cl::desc("Split allocation of ZPR & PPR objects"),
                    cl::init(true), cl::Hidden);

cl::opt<bool> EnableHomogeneousPrologEpilog(
    "homogeneous-prolog-epilog", cl::Hidden,
    cl::desc("Emit homogeneous prologue and epilogue for the size "
             "optimization (default = off)"));

// Stack hazard size for analysis remarks. StackHazardSize takes precedence.
static cl::opt<unsigned>
    StackHazardRemarkSize("aarch64-stack-hazard-remark-size", cl::init(0),
                          cl::Hidden);
// Whether to insert padding into non-streaming functions (for testing).
static cl::opt<bool>
    StackHazardInNonStreaming("aarch64-stack-hazard-in-non-streaming",
                              cl::init(false), cl::Hidden);

static cl::opt<bool> DisableMultiVectorSpillFill(
    "aarch64-disable-multivector-spill-fill",
    cl::desc("Disable use of LD/ST pairs for SME2 or SVE2p1"), cl::init(false),
    cl::Hidden);

int64_t
AArch64FrameLowering::getArgumentStackToRestore(MachineFunction &MF,
                                                MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  bool IsTailCallReturn = (MBB.end() != MBBI)
                              ? AArch64InstrInfo::isTailCallReturnInst(*MBBI)
                              : false;

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

static bool produceCompactUnwindFrame(const AArch64FrameLowering &,
                                      MachineFunction &MF);

enum class AssignObjectOffsets { No, Yes };
/// Process all the SVE stack objects and the SVE stack size and offsets for
/// each object. If AssignOffsets is "Yes", the offsets get assigned (and SVE
/// stack sizes set). Returns the size of the SVE stack.
static SVEStackSizes determineSVEStackSizes(MachineFunction &MF,
                                            AssignObjectOffsets AssignOffsets);

static unsigned getStackHazardSize(const MachineFunction &MF) {
  return MF.getSubtarget<AArch64Subtarget>().getStreamingHazardSize();
}

StackOffset
AArch64FrameLowering::getZPRStackSize(const MachineFunction &MF) const {
  const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  return StackOffset::getScalable(AFI->getStackSizeZPR());
}

StackOffset
AArch64FrameLowering::getPPRStackSize(const MachineFunction &MF) const {
  // With split SVE objects, the hazard padding is added to the PPR region,
  // which places it between the [GPR, PPR] area and the [ZPR, FPR] area. This
  // avoids hazards between both GPRs and FPRs and ZPRs and PPRs.
  const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  return StackOffset::get(AFI->hasSplitSVEObjects() ? getStackHazardSize(MF)
                                                    : 0,
                          AFI->getStackSizePPR());
}

// Conservatively, returns true if the function is likely to have SVE vectors
// on the stack. This function is safe to be called before callee-saves or
// object offsets have been determined.
static bool isLikelyToHaveSVEStack(const AArch64FrameLowering &AFL,
                                   const MachineFunction &MF) {
  auto *AFI = MF.getInfo<AArch64FunctionInfo>();
  if (AFI->isSVECC())
    return true;

  if (AFI->hasCalculatedStackSizeSVE())
    return bool(AFL.getSVEStackSize(MF));

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  for (int FI = MFI.getObjectIndexBegin(); FI < MFI.getObjectIndexEnd(); FI++) {
    if (MFI.hasScalableStackID(FI))
      return true;
  }

  return false;
}

static bool isTargetWindows(const MachineFunction &MF) {
  return MF.getTarget().getMCAsmInfo()->usesWindowsCFI();
}

bool AArch64FrameLowering::hasSVECalleeSavesAboveFrameRecord(
    const MachineFunction &MF) const {
  auto *AFI = MF.getInfo<AArch64FunctionInfo>();
  return isTargetWindows(MF) && AFI->getSVECalleeSavedStackSize();
}

/// Returns true if a homogeneous prolog or epilog code can be emitted
/// for the size optimization. If possible, a frame helper call is injected.
/// When Exit block is given, this check is for epilog.
bool AArch64FrameLowering::homogeneousPrologEpilog(
    MachineFunction &MF, MachineBasicBlock *Exit) const {
  if (!MF.getFunction().hasMinSize())
    return false;
  if (!EnableHomogeneousPrologEpilog)
    return false;
  if (EnableRedZone)
    return false;

  // TODO: Window is supported yet.
  if (isTargetWindows(MF))
    return false;

  // TODO: SVE is not supported yet.
  if (isLikelyToHaveSVEStack(*this, MF))
    return false;

  // Bail on stack adjustment needed on return for simplicity.
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  if (MFI.hasVarSizedObjects() || RegInfo->hasStackRealignment(MF))
    return false;
  if (Exit && getArgumentStackToRestore(MF, *Exit))
    return false;

  auto *AFI = MF.getInfo<AArch64FunctionInfo>();
  if (AFI->hasSwiftAsyncContext() || AFI->hasStreamingModeChanges())
    return false;

  // If there are an odd number of GPRs before LR and FP in the CSRs list,
  // they will not be paired into one RegPairInfo, which is incompatible with
  // the assumption made by the homogeneous prolog epilog pass.
  const MCPhysReg *CSRegs = MF.getRegInfo().getCalleeSavedRegs();
  unsigned NumGPRs = 0;
  for (unsigned I = 0; CSRegs[I]; ++I) {
    Register Reg = CSRegs[I];
    if (Reg == AArch64::LR) {
      assert(CSRegs[I + 1] == AArch64::FP);
      if (NumGPRs % 2 != 0)
        return false;
      break;
    }
    if (AArch64::GPR64RegClass.contains(Reg))
      ++NumGPRs;
  }

  return true;
}

/// Returns true if CSRs should be paired.
bool AArch64FrameLowering::producePairRegisters(MachineFunction &MF) const {
  return produceCompactUnwindFrame(*this, MF) || homogeneousPrologEpilog(MF);
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
  // FIXME: For now, just conservatively guesstimate based on unscaled indexing
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

unsigned
AArch64FrameLowering::getFixedObjectSize(const MachineFunction &MF,
                                         const AArch64FunctionInfo *AFI,
                                         bool IsWin64, bool IsFunclet) const {
  assert(AFI->getTailCallReservedStack() % 16 == 0 &&
         "Tail call reserved stack must be aligned to 16 bytes");
  if (!IsWin64 || IsFunclet) {
    return AFI->getTailCallReservedStack();
  } else {
    if (AFI->getTailCallReservedStack() != 0 &&
        !MF.getFunction().getAttributes().hasAttrSomewhere(
            Attribute::SwiftAsync))
      report_fatal_error("cannot generate ABI-changing tail call for Win64");
    unsigned FixedObjectSize = AFI->getTailCallReservedStack();

    // Var args are stored here in the primary function.
    FixedObjectSize += AFI->getVarArgsGPRSize();

    if (MF.hasEHFunclets()) {
      // Catch objects are stored here in the primary function.
      const MachineFrameInfo &MFI = MF.getFrameInfo();
      const WinEHFuncInfo &EHInfo = *MF.getWinEHFuncInfo();
      SmallSetVector<int, 8> CatchObjFrameIndices;
      for (const WinEHTryBlockMapEntry &TBME : EHInfo.TryBlockMap) {
        for (const WinEHHandlerType &H : TBME.HandlerArray) {
          int FrameIndex = H.CatchObj.FrameIndex;
          if ((FrameIndex != INT_MAX) &&
              CatchObjFrameIndices.insert(FrameIndex)) {
            FixedObjectSize = alignTo(FixedObjectSize,
                                      MFI.getObjectAlign(FrameIndex).value()) +
                              MFI.getObjectSize(FrameIndex);
          }
        }
      }
      // To support EH funclets we allocate an UnwindHelp object
      FixedObjectSize += 8;
    }
    return alignTo(FixedObjectSize, 16);
  }
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

  // If neither NEON or SVE are available, a COPY from one Q-reg to
  // another requires a spill -> reload sequence. We can do that
  // using a pre-decrementing store/post-decrementing load, but
  // if we do so, we can't use the Red Zone.
  bool LowerQRegCopyThroughMem = Subtarget.hasFPARMv8() &&
                                 !Subtarget.isNeonAvailable() &&
                                 !Subtarget.hasSVE();

  return !(MFI.hasCalls() || hasFP(MF) || NumBytes > RedZoneSize ||
           AFI->hasSVEStackSize() || LowerQRegCopyThroughMem);
}

/// hasFPImpl - Return true if the specified function should have a dedicated
/// frame pointer register.
bool AArch64FrameLowering::hasFPImpl(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  const AArch64FunctionInfo &AFI = *MF.getInfo<AArch64FunctionInfo>();

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

  // If we:
  //
  //   1. Have streaming mode changes
  //     OR:
  //   2. Have a streaming body with SVE stack objects
  //
  // Then the value of VG restored when unwinding to this function may not match
  // the value of VG used to set up the stack.
  //
  // This is a problem as the CFA can be described with an expression of the
  // form: CFA = SP + NumBytes + VG * NumScalableBytes.
  //
  // If the value of VG used in that expression does not match the value used to
  // set up the stack, an incorrect address for the CFA will be computed, and
  // unwinding will fail.
  //
  // We work around this issue by ensuring the frame-pointer can describe the
  // CFA in either of these cases.
  if (AFI.needsDwarfUnwindInfo(MF) &&
      ((requiresSaveVG(MF) || AFI.getSMEFnAttrs().hasStreamingBody()) &&
       (!AFI.hasCalculatedStackSizeSVE() || AFI.hasSVEStackSize())))
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

/// Should the Frame Pointer be reserved for the current function?
bool AArch64FrameLowering::isFPReserved(const MachineFunction &MF) const {
  const TargetMachine &TM = MF.getTarget();
  const Triple &TT = TM.getTargetTriple();

  // These OSes require the frame chain is valid, even if the current frame does
  // not use a frame pointer.
  if (TT.isOSDarwin() || TT.isOSWindows())
    return true;

  // If the function has a frame pointer, it is reserved.
  if (hasFP(MF))
    return true;

  // Frontend has requested to preserve the frame pointer.
  if (TM.Options.FramePointerIsReserved(MF))
    return true;

  return false;
}

/// hasReservedCallFrame - Under normal circumstances, when a frame pointer is
/// not required, we reserve argument space for call sites in the function
/// immediately on entry to the current function.  This eliminates the need for
/// add/sub sp brackets around call sites.  Returns true if the call frame is
/// included as part of the stack frame.
bool AArch64FrameLowering::hasReservedCallFrame(
    const MachineFunction &MF) const {
  // The stack probing code for the dynamically allocated outgoing arguments
  // area assumes that the stack is probed at the top - either by the prologue
  // code, which issues a probe if `hasVarSizedObjects` return true, or by the
  // most recent variable-sized object allocation. Changing the condition here
  // may need to be followed up by changes to the probe issuing logic.
  return !MF.getFrameInfo().hasVarSizedObjects();
}

MachineBasicBlock::iterator AArch64FrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {

  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const AArch64InstrInfo *TII = Subtarget.getInstrInfo();
  const AArch64TargetLowering *TLI = Subtarget.getTargetLowering();
  [[maybe_unused]] MachineFrameInfo &MFI = MF.getFrameInfo();
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

      if (TLI->hasInlineStackProbe(MF) &&
          -Amount >= AArch64::StackProbeMaxUnprobedStack) {
        // When stack probing is enabled, the decrement of SP may need to be
        // probed. We only need to do this if the call site needs 1024 bytes of
        // space or more, because a region smaller than that is allowed to be
        // unprobed at an ABI boundary. We rely on the fact that SP has been
        // probed exactly at this point, either by the prologue or most recent
        // dynamic allocation.
        assert(MFI.hasVarSizedObjects() &&
               "non-reserved call frame without var sized objects?");
        Register ScratchReg =
            MF.getRegInfo().createVirtualRegister(&AArch64::GPR64RegClass);
        inlineStackProbeFixed(I, ScratchReg, -Amount, StackOffset::get(0, 0));
      } else {
        emitFrameOffset(MBB, I, DL, AArch64::SP, AArch64::SP,
                        StackOffset::getFixed(Amount), TII);
      }
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

void AArch64FrameLowering::resetCFIToInitialState(
    MachineBasicBlock &MBB) const {

  MachineFunction &MF = *MBB.getParent();
  const auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const auto &TRI = *Subtarget.getRegisterInfo();
  const auto &MFI = *MF.getInfo<AArch64FunctionInfo>();

  CFIInstBuilder CFIBuilder(MBB, MBB.begin(), MachineInstr::NoFlags);

  // Reset the CFA to `SP + 0`.
  CFIBuilder.buildDefCFA(AArch64::SP, 0);

  // Flip the RA sign state.
  if (MFI.shouldSignReturnAddress(MF))
    MFI.branchProtectionPAuthLR() ? CFIBuilder.buildNegateRAStateWithPC()
                                  : CFIBuilder.buildNegateRAState();

  // Shadow call stack uses X18, reset it.
  if (MFI.needsShadowCallStackPrologueEpilogue(MF))
    CFIBuilder.buildSameValue(AArch64::X18);

  // Emit .cfi_same_value for callee-saved registers.
  const std::vector<CalleeSavedInfo> &CSI =
      MF.getFrameInfo().getCalleeSavedInfo();
  for (const auto &Info : CSI) {
    MCRegister Reg = Info.getReg();
    if (!TRI.regNeedsCFI(Reg, Reg))
      continue;
    CFIBuilder.buildSameValue(Reg);
  }
}

static MCRegister getRegisterOrZero(MCRegister Reg, bool HasSVE) {
  switch (Reg.id()) {
  default:
    // The called routine is expected to preserve r19-r28
    // r29 and r30 are used as frame pointer and link register resp.
    return 0;

    // GPRs
#define CASE(n)                                                                \
  case AArch64::W##n:                                                          \
  case AArch64::X##n:                                                          \
    return AArch64::X##n
  CASE(0);
  CASE(1);
  CASE(2);
  CASE(3);
  CASE(4);
  CASE(5);
  CASE(6);
  CASE(7);
  CASE(8);
  CASE(9);
  CASE(10);
  CASE(11);
  CASE(12);
  CASE(13);
  CASE(14);
  CASE(15);
  CASE(16);
  CASE(17);
  CASE(18);
#undef CASE

    // FPRs
#define CASE(n)                                                                \
  case AArch64::B##n:                                                          \
  case AArch64::H##n:                                                          \
  case AArch64::S##n:                                                          \
  case AArch64::D##n:                                                          \
  case AArch64::Q##n:                                                          \
    return HasSVE ? AArch64::Z##n : AArch64::Q##n
  CASE(0);
  CASE(1);
  CASE(2);
  CASE(3);
  CASE(4);
  CASE(5);
  CASE(6);
  CASE(7);
  CASE(8);
  CASE(9);
  CASE(10);
  CASE(11);
  CASE(12);
  CASE(13);
  CASE(14);
  CASE(15);
  CASE(16);
  CASE(17);
  CASE(18);
  CASE(19);
  CASE(20);
  CASE(21);
  CASE(22);
  CASE(23);
  CASE(24);
  CASE(25);
  CASE(26);
  CASE(27);
  CASE(28);
  CASE(29);
  CASE(30);
  CASE(31);
#undef CASE
  }
}

void AArch64FrameLowering::emitZeroCallUsedRegs(BitVector RegsToZero,
                                                MachineBasicBlock &MBB) const {
  // Insertion point.
  MachineBasicBlock::iterator MBBI = MBB.getFirstTerminator();

  // Fake a debug loc.
  DebugLoc DL;
  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();

  const MachineFunction &MF = *MBB.getParent();
  const AArch64Subtarget &STI = MF.getSubtarget<AArch64Subtarget>();
  const AArch64RegisterInfo &TRI = *STI.getRegisterInfo();

  BitVector GPRsToZero(TRI.getNumRegs());
  BitVector FPRsToZero(TRI.getNumRegs());
  bool HasSVE = STI.isSVEorStreamingSVEAvailable();
  for (MCRegister Reg : RegsToZero.set_bits()) {
    if (TRI.isGeneralPurposeRegister(MF, Reg)) {
      // For GPRs, we only care to clear out the 64-bit register.
      if (MCRegister XReg = getRegisterOrZero(Reg, HasSVE))
        GPRsToZero.set(XReg);
    } else if (AArch64InstrInfo::isFpOrNEON(Reg)) {
      // For FPRs,
      if (MCRegister XReg = getRegisterOrZero(Reg, HasSVE))
        FPRsToZero.set(XReg);
    }
  }

  const AArch64InstrInfo &TII = *STI.getInstrInfo();

  // Zero out GPRs.
  for (MCRegister Reg : GPRsToZero.set_bits())
    TII.buildClearRegister(Reg, MBB, MBBI, DL);

  // Zero out FP/vector registers.
  for (MCRegister Reg : FPRsToZero.set_bits())
    TII.buildClearRegister(Reg, MBB, MBBI, DL);

  if (HasSVE) {
    for (MCRegister PReg :
         {AArch64::P0, AArch64::P1, AArch64::P2, AArch64::P3, AArch64::P4,
          AArch64::P5, AArch64::P6, AArch64::P7, AArch64::P8, AArch64::P9,
          AArch64::P10, AArch64::P11, AArch64::P12, AArch64::P13, AArch64::P14,
          AArch64::P15}) {
      if (RegsToZero[PReg])
        BuildMI(MBB, MBBI, DL, TII.get(AArch64::PFALSE), PReg);
    }
  }
}

bool AArch64FrameLowering::windowsRequiresStackProbe(
    const MachineFunction &MF, uint64_t StackSizeInBytes) const {
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const AArch64FunctionInfo &MFI = *MF.getInfo<AArch64FunctionInfo>();
  // TODO: When implementing stack protectors, take that into account
  // for the probe threshold.
  return Subtarget.isTargetWindows() && MFI.hasStackProbing() &&
         StackSizeInBytes >= uint64_t(MFI.getStackProbeSize());
}

static void getLiveRegsForEntryMBB(LivePhysRegs &LiveRegs,
                                   const MachineBasicBlock &MBB) {
  const MachineFunction *MF = MBB.getParent();
  LiveRegs.addLiveIns(MBB);
  // Mark callee saved registers as used so we will not choose them.
  const MCPhysReg *CSRegs = MF->getRegInfo().getCalleeSavedRegs();
  for (unsigned i = 0; CSRegs[i]; ++i)
    LiveRegs.addReg(CSRegs[i]);
}

Register
AArch64FrameLowering::findScratchNonCalleeSaveRegister(MachineBasicBlock *MBB,
                                                       bool HasCall) const {
  MachineFunction *MF = MBB->getParent();

  // If MBB is an entry block, use X9 as the scratch register
  // preserve_none functions may be using X9 to pass arguments,
  // so prefer to pick an available register below.
  if (&MF->front() == MBB &&
      MF->getFunction().getCallingConv() != CallingConv::PreserveNone)
    return AArch64::X9;

  const AArch64Subtarget &Subtarget = MF->getSubtarget<AArch64Subtarget>();
  const AArch64RegisterInfo &TRI = *Subtarget.getRegisterInfo();
  LivePhysRegs LiveRegs(TRI);
  getLiveRegsForEntryMBB(LiveRegs, *MBB);
  if (HasCall) {
    LiveRegs.addReg(AArch64::X16);
    LiveRegs.addReg(AArch64::X17);
    LiveRegs.addReg(AArch64::X18);
  }

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
  const AArch64TargetLowering *TLI = Subtarget.getTargetLowering();
  const AArch64FunctionInfo *AFI = MF->getInfo<AArch64FunctionInfo>();

  if (AFI->hasSwiftAsyncContext()) {
    const AArch64RegisterInfo &TRI = *Subtarget.getRegisterInfo();
    const MachineRegisterInfo &MRI = MF->getRegInfo();
    LivePhysRegs LiveRegs(TRI);
    getLiveRegsForEntryMBB(LiveRegs, MBB);
    // The StoreSwiftAsyncContext clobbers X16 and X17. Make sure they are
    // available.
    if (!LiveRegs.available(MRI, AArch64::X16) ||
        !LiveRegs.available(MRI, AArch64::X17))
      return false;
  }

  // Certain stack probing sequences might clobber flags, then we can't use
  // the block as a prologue if the flags register is a live-in.
  if (MF->getInfo<AArch64FunctionInfo>()->hasStackProbing() &&
      MBB.isLiveIn(AArch64::NZCV))
    return false;

  if (RegInfo->hasStackRealignment(*MF) || TLI->hasInlineStackProbe(*MF))
    if (findScratchNonCalleeSaveRegister(TmpMBB) == AArch64::NoRegister)
      return false;

  // May need a scratch register (for return value) if require making a special
  // call
  if (requiresSaveVG(*MF) ||
      windowsRequiresStackProbe(*MF, std::numeric_limits<uint64_t>::max()))
    if (findScratchNonCalleeSaveRegister(TmpMBB, true) == AArch64::NoRegister)
      return false;

  return true;
}

bool AArch64FrameLowering::needsWinCFI(const MachineFunction &MF) const {
  const Function &F = MF.getFunction();
  return MF.getTarget().getMCAsmInfo()->usesWindowsCFI() &&
         F.needsUnwindTableEntry();
}

bool AArch64FrameLowering::shouldSignReturnAddressEverywhere(
    const MachineFunction &MF) const {
  // FIXME: With WinCFI, extra care should be taken to place SEH_PACSignLR
  //        and SEH_EpilogEnd instructions in the correct order.
  if (MF.getTarget().getMCAsmInfo()->usesWindowsCFI())
    return false;
  const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  return AFI->getSignReturnAddressCondition() == SignReturnAddress::All;
}

// Given a load or a store instruction, generate an appropriate unwinding SEH
// code on Windows.
MachineBasicBlock::iterator
AArch64FrameLowering::insertSEH(MachineBasicBlock::iterator MBBI,
                                const AArch64InstrInfo &TII,
                                MachineInstr::MIFlag Flag) const {
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
    report_fatal_error("No SEH Opcode for this instruction");
  case AArch64::STR_ZXI:
  case AArch64::LDR_ZXI: {
    unsigned Reg0 = RegInfo->getSEHRegNum(MBBI->getOperand(0).getReg());
    MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveZReg))
              .addImm(Reg0)
              .addImm(Imm)
              .setMIFlag(Flag);
    break;
  }
  case AArch64::STR_PXI:
  case AArch64::LDR_PXI: {
    unsigned Reg0 = RegInfo->getSEHRegNum(MBBI->getOperand(0).getReg());
    MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SavePReg))
              .addImm(Reg0)
              .addImm(Imm)
              .setMIFlag(Flag);
    break;
  }
  case AArch64::LDPDpost:
    Imm = -Imm;
    [[fallthrough]];
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
    [[fallthrough]];
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
    [[fallthrough]];
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
    [[fallthrough]];
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

    int SEHReg0 = RegInfo->getSEHRegNum(Reg0);
    int SEHReg1 = RegInfo->getSEHRegNum(Reg1);

    if (Reg0 == AArch64::FP && Reg1 == AArch64::LR)
      MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveFPLR))
                .addImm(Imm * 8)
                .setMIFlag(Flag);
    else if (SEHReg0 >= 19 && SEHReg1 >= 19)
      MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveRegP))
                .addImm(SEHReg0)
                .addImm(SEHReg1)
                .addImm(Imm * 8)
                .setMIFlag(Flag);
    else
      MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveAnyRegIP))
                .addImm(SEHReg0)
                .addImm(SEHReg1)
                .addImm(Imm * 8)
                .setMIFlag(Flag);
    break;
  }
  case AArch64::STRXui:
  case AArch64::LDRXui: {
    int Reg = RegInfo->getSEHRegNum(MBBI->getOperand(0).getReg());
    if (Reg >= 19)
      MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveReg))
                .addImm(Reg)
                .addImm(Imm * 8)
                .setMIFlag(Flag);
    else
      MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveAnyRegI))
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
  case AArch64::STPQi:
  case AArch64::LDPQi: {
    unsigned Reg0 = RegInfo->getSEHRegNum(MBBI->getOperand(0).getReg());
    unsigned Reg1 = RegInfo->getSEHRegNum(MBBI->getOperand(1).getReg());
    MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveAnyRegQP))
              .addImm(Reg0)
              .addImm(Reg1)
              .addImm(Imm * 16)
              .setMIFlag(Flag);
    break;
  }
  case AArch64::LDPQpost:
    Imm = -Imm;
    [[fallthrough]];
  case AArch64::STPQpre: {
    unsigned Reg0 = RegInfo->getSEHRegNum(MBBI->getOperand(1).getReg());
    unsigned Reg1 = RegInfo->getSEHRegNum(MBBI->getOperand(2).getReg());
    MIB = BuildMI(MF, DL, TII.get(AArch64::SEH_SaveAnyRegQPX))
              .addImm(Reg0)
              .addImm(Reg1)
              .addImm(Imm * 16)
              .setMIFlag(Flag);
    break;
  }
  }
  auto I = MBB->insertAfter(MBBI, MIB);
  return I;
}

bool AArch64FrameLowering::requiresSaveVG(const MachineFunction &MF) const {
  const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  if (!AFI->needsDwarfUnwindInfo(MF) || !AFI->hasStreamingModeChanges())
    return false;
  // For Darwin platforms we don't save VG for non-SVE functions, even if SME
  // is enabled with streaming mode changes.
  auto &ST = MF.getSubtarget<AArch64Subtarget>();
  if (ST.isTargetDarwin())
    return ST.hasSVE();
  return true;
}

void AArch64FrameLowering::emitPacRetPlusLeafHardening(
    MachineFunction &MF) const {
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const AArch64InstrInfo *TII = Subtarget.getInstrInfo();

  auto EmitSignRA = [&](MachineBasicBlock &MBB) {
    DebugLoc DL; // Set debug location to unknown.
    MachineBasicBlock::iterator MBBI = MBB.begin();

    BuildMI(MBB, MBBI, DL, TII->get(AArch64::PAUTH_PROLOGUE))
        .setMIFlag(MachineInstr::FrameSetup);
  };

  auto EmitAuthRA = [&](MachineBasicBlock &MBB) {
    DebugLoc DL;
    MachineBasicBlock::iterator MBBI = MBB.getFirstTerminator();
    if (MBBI != MBB.end())
      DL = MBBI->getDebugLoc();

    BuildMI(MBB, MBBI, DL, TII->get(AArch64::PAUTH_EPILOGUE))
        .setMIFlag(MachineInstr::FrameDestroy);
  };

  // This should be in sync with PEIImpl::calculateSaveRestoreBlocks.
  EmitSignRA(MF.front());
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.isEHFuncletEntry())
      EmitSignRA(MBB);
    if (MBB.isReturnBlock())
      EmitAuthRA(MBB);
  }
}

void AArch64FrameLowering::emitPrologue(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {
  AArch64PrologueEmitter PrologueEmitter(MF, MBB, *this);
  PrologueEmitter.emitPrologue();
}

void AArch64FrameLowering::emitEpilogue(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {
  AArch64EpilogueEmitter EpilogueEmitter(MF, MBB, *this);
  EpilogueEmitter.emitEpilogue();
}

bool AArch64FrameLowering::enableCFIFixup(const MachineFunction &MF) const {
  return TargetFrameLowering::enableCFIFixup(MF) &&
         MF.getInfo<AArch64FunctionInfo>()->needsDwarfUnwindInfo(MF);
}

bool AArch64FrameLowering::enableFullCFIFixup(const MachineFunction &MF) const {
  return enableCFIFixup(MF) &&
         MF.getInfo<AArch64FunctionInfo>()->needsAsyncDwarfUnwindInfo(MF);
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
      MF.getFunction().hasFnAttribute(Attribute::SanitizeHWAddress) ||
          MF.getFunction().hasFnAttribute(Attribute::SanitizeMemTag),
      /*ForSimm=*/false);
}

StackOffset
AArch64FrameLowering::getFrameIndexReferenceFromSP(const MachineFunction &MF,
                                                   int FI) const {
  // This function serves to provide a comparable offset from a single reference
  // point (the value of SP at function entry) that can be used for analysis,
  // e.g. the stack-frame-layout analysis pass. It is not guaranteed to be
  // correct for all objects in the presence of VLA-area objects or dynamic
  // stack re-alignment.

  const auto &MFI = MF.getFrameInfo();

  int64_t ObjectOffset = MFI.getObjectOffset(FI);
  StackOffset ZPRStackSize = getZPRStackSize(MF);
  StackOffset PPRStackSize = getPPRStackSize(MF);
  StackOffset SVEStackSize = ZPRStackSize + PPRStackSize;

  // For VLA-area objects, just emit an offset at the end of the stack frame.
  // Whilst not quite correct, these objects do live at the end of the frame and
  // so it is more useful for analysis for the offset to reflect this.
  if (MFI.isVariableSizedObjectIndex(FI)) {
    return StackOffset::getFixed(-((int64_t)MFI.getStackSize())) - SVEStackSize;
  }

  // This is correct in the absence of any SVE stack objects.
  if (!SVEStackSize)
    return StackOffset::getFixed(ObjectOffset - getOffsetOfLocalArea());

  const auto *AFI = MF.getInfo<AArch64FunctionInfo>();
  bool FPAfterSVECalleeSaves = hasSVECalleeSavesAboveFrameRecord(MF);
  if (MFI.hasScalableStackID(FI)) {
    if (FPAfterSVECalleeSaves &&
        -ObjectOffset <= (int64_t)AFI->getSVECalleeSavedStackSize()) {
      assert(!AFI->hasSplitSVEObjects() &&
             "split-sve-objects not supported with FPAfterSVECalleeSaves");
      return StackOffset::getScalable(ObjectOffset);
    }
    StackOffset AccessOffset{};
    // The scalable vectors are below (lower address) the scalable predicates
    // with split SVE objects, so we must subtract the size of the predicates.
    if (AFI->hasSplitSVEObjects() &&
        MFI.getStackID(FI) == TargetStackID::ScalableVector)
      AccessOffset = -PPRStackSize;
    return AccessOffset +
           StackOffset::get(-((int64_t)AFI->getCalleeSavedStackSize()),
                            ObjectOffset);
  }

  bool IsFixed = MFI.isFixedObjectIndex(FI);
  bool IsCSR =
      !IsFixed && ObjectOffset >= -((int)AFI->getCalleeSavedStackSize(MFI));

  StackOffset ScalableOffset = {};
  if (!IsFixed && !IsCSR) {
    ScalableOffset = -SVEStackSize;
  } else if (FPAfterSVECalleeSaves && IsCSR) {
    ScalableOffset =
        -StackOffset::getScalable(AFI->getSVECalleeSavedStackSize());
  }

  return StackOffset::getFixed(ObjectOffset) + ScalableOffset;
}

StackOffset
AArch64FrameLowering::getNonLocalFrameIndexReference(const MachineFunction &MF,
                                                     int FI) const {
  return StackOffset::getFixed(getSEHFrameIndexOffset(MF, FI));
}

StackOffset AArch64FrameLowering::getFPOffset(const MachineFunction &MF,
                                              int64_t ObjectOffset) const {
  const auto *AFI = MF.getInfo<AArch64FunctionInfo>();
  const auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const Function &F = MF.getFunction();
  bool IsWin64 = Subtarget.isCallingConvWin64(F.getCallingConv(), F.isVarArg());
  unsigned FixedObject =
      getFixedObjectSize(MF, AFI, IsWin64, /*IsFunclet=*/false);
  int64_t CalleeSaveSize = AFI->getCalleeSavedStackSize(MF.getFrameInfo());
  int64_t FPAdjust =
      CalleeSaveSize - AFI->getCalleeSaveBaseToFrameRecordOffset();
  return StackOffset::getFixed(ObjectOffset + FixedObject + FPAdjust);
}

StackOffset AArch64FrameLowering::getStackOffset(const MachineFunction &MF,
                                                 int64_t ObjectOffset) const {
  const auto &MFI = MF.getFrameInfo();
  return StackOffset::getFixed(ObjectOffset + (int64_t)MFI.getStackSize());
}

// TODO: This function currently does not work for scalable vectors.
int AArch64FrameLowering::getSEHFrameIndexOffset(const MachineFunction &MF,
                                                 int FI) const {
  const AArch64RegisterInfo *RegInfo =
      MF.getSubtarget<AArch64Subtarget>().getRegisterInfo();
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
  auto StackID = static_cast<TargetStackID::Value>(MFI.getStackID(FI));
  return resolveFrameOffsetReference(MF, ObjectOffset, isFixed, StackID,
                                     FrameReg, PreferFP, ForSimm);
}

StackOffset AArch64FrameLowering::resolveFrameOffsetReference(
    const MachineFunction &MF, int64_t ObjectOffset, bool isFixed,
    TargetStackID::Value StackID, Register &FrameReg, bool PreferFP,
    bool ForSimm) const {
  const auto &MFI = MF.getFrameInfo();
  const auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const AArch64RegisterInfo *RegInfo = Subtarget.getRegisterInfo();
  const auto *AFI = MF.getInfo<AArch64FunctionInfo>();

  int64_t FPOffset = getFPOffset(MF, ObjectOffset).getFixed();
  int64_t Offset = getStackOffset(MF, ObjectOffset).getFixed();
  bool isCSR =
      !isFixed && ObjectOffset >= -((int)AFI->getCalleeSavedStackSize(MFI));
  bool isSVE = MFI.isScalableStackID(StackID);

  StackOffset ZPRStackSize = getZPRStackSize(MF);
  StackOffset PPRStackSize = getPPRStackSize(MF);
  StackOffset SVEStackSize = ZPRStackSize + PPRStackSize;

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

      if (FPOffset >= 0) {
        // If the FPOffset is positive, that'll always be best, as the SP/BP
        // will be even further away.
        UseFP = true;
      } else if (MFI.hasVarSizedObjects()) {
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
      } else if (MF.hasEHFunclets() && !RegInfo->hasBasePointer(MF)) {
        // Funclets access the locals contained in the parent's stack frame
        // via the frame pointer, so we have to use the FP in the parent
        // function.
        (void) Subtarget;
        assert(Subtarget.isCallingConvWin64(MF.getFunction().getCallingConv(),
                                            MF.getFunction().isVarArg()) &&
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

  bool FPAfterSVECalleeSaves = hasSVECalleeSavesAboveFrameRecord(MF);

  if (isSVE) {
    StackOffset FPOffset = StackOffset::get(
        -AFI->getCalleeSaveBaseToFrameRecordOffset(), ObjectOffset);
    StackOffset SPOffset =
        SVEStackSize +
        StackOffset::get(MFI.getStackSize() - AFI->getCalleeSavedStackSize(),
                         ObjectOffset);

    // With split SVE objects the ObjectOffset is relative to the split area
    // (i.e. the PPR area or ZPR area respectively).
    if (AFI->hasSplitSVEObjects() && StackID == TargetStackID::ScalableVector) {
      // If we're accessing an SVE vector with split SVE objects...
      // - From the FP we need to move down past the PPR area:
      FPOffset -= PPRStackSize;
      // - From the SP we only need to move up to the ZPR area:
      SPOffset -= PPRStackSize;
      // Note: `SPOffset = SVEStackSize + ...`, so `-= PPRStackSize` results in
      // `SPOffset = ZPRStackSize + ...`.
    }

    if (FPAfterSVECalleeSaves) {
      FPOffset += StackOffset::getScalable(AFI->getSVECalleeSavedStackSize());
      if (-ObjectOffset <= (int64_t)AFI->getSVECalleeSavedStackSize()) {
        FPOffset += StackOffset::getFixed(AFI->getCalleeSavedStackSize());
        SPOffset += StackOffset::getFixed(AFI->getCalleeSavedStackSize());
      }
    }

    // Always use the FP for SVE spills if available and beneficial.
    if (hasFP(MF) && (SPOffset.getFixed() ||
                      FPOffset.getScalable() < SPOffset.getScalable() ||
                      RegInfo->hasStackRealignment(MF))) {
      FrameReg = RegInfo->getFrameRegister(MF);
      return FPOffset;
    }
    FrameReg = RegInfo->hasBasePointer(MF) ? RegInfo->getBaseRegister()
                                           : MCRegister(AArch64::SP);

    return SPOffset;
  }

  StackOffset SVEAreaOffset = {};
  if (FPAfterSVECalleeSaves) {
    // In this stack layout, the FP is in between the callee saves and other
    // SVE allocations.
    StackOffset SVECalleeSavedStack =
        StackOffset::getScalable(AFI->getSVECalleeSavedStackSize());
    if (UseFP) {
      if (isFixed)
        SVEAreaOffset = SVECalleeSavedStack;
      else if (!isCSR)
        SVEAreaOffset = SVECalleeSavedStack - SVEStackSize;
    } else {
      if (isFixed)
        SVEAreaOffset = SVEStackSize;
      else if (isCSR)
        SVEAreaOffset = SVEStackSize - SVECalleeSavedStack;
    }
  } else {
    if (UseFP && !(isFixed || isCSR))
      SVEAreaOffset = -SVEStackSize;
    if (!UseFP && (isFixed || isCSR))
      SVEAreaOffset = SVEStackSize;
  }

  if (UseFP) {
    FrameReg = RegInfo->getFrameRegister(MF);
    return StackOffset::getFixed(FPOffset) + SVEAreaOffset;
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

  return StackOffset::getFixed(Offset) + SVEAreaOffset;
}

static RegState getPrologueDeath(MachineFunction &MF, unsigned Reg) {
  // Do not set a kill flag on values that are also marked as live-in. This
  // happens with the @llvm-returnaddress intrinsic and with arguments passed in
  // callee saved registers.
  // Omitting the kill flags is conservatively correct even if the live-in
  // is not used after all.
  bool IsLiveIn = MF.getRegInfo().isLiveIn(Reg);
  return getKillRegState(!IsLiveIn);
}

static bool produceCompactUnwindFrame(const AArch64FrameLowering &AFL,
                                      MachineFunction &MF) {
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  AttributeList Attrs = MF.getFunction().getAttributes();
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  return Subtarget.isTargetMachO() &&
         !(Subtarget.getTargetLowering()->supportSwiftError() &&
           Attrs.hasAttrSomewhere(Attribute::SwiftError)) &&
         MF.getFunction().getCallingConv() != CallingConv::SwiftTail &&
         !AFL.requiresSaveVG(MF) && !AFI->isSVECC();
}

static bool invalidateWindowsRegisterPairing(bool SpillExtendedVolatile,
                                             unsigned SpillCount, unsigned Reg1,
                                             unsigned Reg2, bool NeedsWinCFI,
                                             const TargetRegisterInfo *TRI) {
  // If we are generating register pairs for a Windows function that requires
  // EH support, then pair consecutive registers only.  There are no unwind
  // opcodes for saves/restores of non-consecutive register pairs.
  // The unwind opcodes are save_regp, save_regp_x, save_fregp, save_frepg_x,
  // save_lrpair.
  // https://docs.microsoft.com/en-us/cpp/build/arm64-exception-handling

  if (Reg2 == AArch64::FP)
    return true;
  if (!NeedsWinCFI)
    return false;

  // ARM64EC introduced `save_any_regp`, which expects 16-byte alignment.
  // This is handled by only allowing paired spills for registers spilled at
  // even positions (which should be 16-byte aligned, as other GPRs/FPRs are
  // 8-bytes). We carve out an exception for {FP,LR}, which does not require
  // 16-byte alignment in the uop representation.
  if (TRI->getEncodingValue(Reg2) == TRI->getEncodingValue(Reg1) + 1)
    return SpillExtendedVolatile
               ? !((Reg1 == AArch64::FP && Reg2 == AArch64::LR) ||
                   (SpillCount % 2) == 0)
               : false;

  // If pairing a GPR with LR, the pair can be described by the save_lrpair
  // opcode. The save_lrpair opcode requires the first register to be odd.
  if (Reg1 >= AArch64::X19 && Reg1 <= AArch64::X27 &&
      (Reg1 - AArch64::X19) % 2 == 0 && Reg2 == AArch64::LR)
    return false;
  return true;
}

/// Returns true if Reg1 and Reg2 cannot be paired using a ldp/stp instruction.
/// WindowsCFI requires that only consecutive registers can be paired.
/// LR and FP need to be allocated together when the frame needs to save
/// the frame-record. This means any other register pairing with LR is invalid.
static bool invalidateRegisterPairing(bool SpillExtendedVolatile,
                                      unsigned SpillCount, unsigned Reg1,
                                      unsigned Reg2, bool UsesWinAAPCS,
                                      bool NeedsWinCFI, bool NeedsFrameRecord,
                                      const TargetRegisterInfo *TRI) {
  if (UsesWinAAPCS)
    return invalidateWindowsRegisterPairing(SpillExtendedVolatile, SpillCount,
                                            Reg1, Reg2, NeedsWinCFI, TRI);

  // If we need to store the frame record, don't pair any register
  // with LR other than FP.
  if (NeedsFrameRecord)
    return Reg2 == AArch64::LR;

  return false;
}

namespace {

struct RegPairInfo {
  Register Reg1;
  Register Reg2;
  int FrameIdx;
  int Offset;
  enum RegType { GPR, FPR64, FPR128, PPR, ZPR, VG } Type;
  const TargetRegisterClass *RC;

  RegPairInfo() = default;

  bool isPaired() const { return Reg2.isValid(); }

  bool isScalable() const { return Type == PPR || Type == ZPR; }
};

} // end anonymous namespace

MCRegister findFreePredicateReg(BitVector &SavedRegs) {
  for (unsigned PReg = AArch64::P8; PReg <= AArch64::P15; ++PReg) {
    if (SavedRegs.test(PReg)) {
      unsigned PNReg = PReg - AArch64::P0 + AArch64::PN0;
      return MCRegister(PNReg);
    }
  }
  return MCRegister();
}

// The multivector LD/ST are available only for SME or SVE2p1 targets
bool enableMultiVectorSpillFill(const AArch64Subtarget &Subtarget,
                                MachineFunction &MF) {
  if (DisableMultiVectorSpillFill)
    return false;

  SMEAttrs FuncAttrs = MF.getInfo<AArch64FunctionInfo>()->getSMEFnAttrs();
  bool IsLocallyStreaming =
      FuncAttrs.hasStreamingBody() && !FuncAttrs.hasStreamingInterface();

  // Only when in streaming mode SME2 instructions can be safely used.
  // It is not safe to use SME2 instructions when in streaming compatible or
  // locally streaming mode.
  return Subtarget.hasSVE2p1() ||
         (Subtarget.hasSME2() &&
          (!IsLocallyStreaming && Subtarget.isStreaming()));
}

void computeCalleeSaveRegisterPairs(const AArch64FrameLowering &AFL,
                                    MachineFunction &MF,
                                    ArrayRef<CalleeSavedInfo> CSI,
                                    const TargetRegisterInfo *TRI,
                                    SmallVectorImpl<RegPairInfo> &RegPairs,
                                    bool NeedsFrameRecord) {

  if (CSI.empty())
    return;

  bool IsWindows = isTargetWindows(MF);
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  unsigned StackHazardSize = getStackHazardSize(MF);
  MachineFrameInfo &MFI = MF.getFrameInfo();
  CallingConv::ID CC = MF.getFunction().getCallingConv();
  unsigned Count = CSI.size();
  (void)CC;
  // MachO's compact unwind format relies on all registers being stored in
  // pairs.
  assert((!produceCompactUnwindFrame(AFL, MF) ||
          CC == CallingConv::PreserveMost || CC == CallingConv::PreserveAll ||
          CC == CallingConv::CXX_FAST_TLS || CC == CallingConv::Win64 ||
          (Count & 1) == 0) &&
         "Odd number of callee-saved regs to spill!");
  int ByteOffset = AFI->getCalleeSavedStackSize();
  int StackFillDir = -1;
  int RegInc = 1;
  unsigned FirstReg = 0;
  if (IsWindows) {
    // For WinCFI, fill the stack from the bottom up.
    ByteOffset = 0;
    StackFillDir = 1;
    // As the CSI array is reversed to match PrologEpilogInserter, iterate
    // backwards, to pair up registers starting from lower numbered registers.
    RegInc = -1;
    FirstReg = Count - 1;
  }

  bool FPAfterSVECalleeSaves = AFL.hasSVECalleeSavesAboveFrameRecord(MF);
  // Windows AAPCS has x9-x15 as volatile registers, x16-x17 as intra-procedural
  // scratch, x18 as platform reserved. However, clang has extended calling
  // convensions such as preserve_most and preserve_all which treat these as
  // CSR. As such, the ARM64 unwind uOPs bias registers by 19. We use ARM64EC
  // uOPs which have separate restrictions. We need to check for that.
  //
  // NOTE: we currently do not account for the D registers as LLVM does not
  // support non-ABI compliant D register spills.
  bool SpillExtendedVolatile =
      IsWindows && llvm::any_of(CSI, [](const CalleeSavedInfo &CSI) {
        const auto &Reg = CSI.getReg();
        return Reg >= AArch64::X0 && Reg <= AArch64::X18;
      });

  int ZPRByteOffset = 0;
  int PPRByteOffset = 0;
  bool SplitPPRs = AFI->hasSplitSVEObjects();
  if (SplitPPRs) {
    ZPRByteOffset = AFI->getZPRCalleeSavedStackSize();
    PPRByteOffset = AFI->getPPRCalleeSavedStackSize();
  } else if (!FPAfterSVECalleeSaves) {
    ZPRByteOffset =
        AFI->getZPRCalleeSavedStackSize() + AFI->getPPRCalleeSavedStackSize();
    // Unused: Everything goes in ZPR space.
    PPRByteOffset = 0;
  }

  bool NeedGapToAlignStack = AFI->hasCalleeSaveStackFreeSpace();
  Register LastReg = 0;
  bool HasCSHazardPadding = AFI->hasStackHazardSlotIndex() && !SplitPPRs;

  // When iterating backwards, the loop condition relies on unsigned wraparound.
  for (unsigned i = FirstReg; i < Count; i += RegInc) {
    RegPairInfo RPI;
    RPI.Reg1 = CSI[i].getReg();

    if (AArch64::GPR64RegClass.contains(RPI.Reg1)) {
      RPI.Type = RegPairInfo::GPR;
      RPI.RC = &AArch64::GPR64RegClass;
    } else if (AArch64::FPR64RegClass.contains(RPI.Reg1)) {
      RPI.Type = RegPairInfo::FPR64;
      RPI.RC = &AArch64::FPR64RegClass;
    } else if (AArch64::FPR128RegClass.contains(RPI.Reg1)) {
      RPI.Type = RegPairInfo::FPR128;
      RPI.RC = &AArch64::FPR128RegClass;
    } else if (AArch64::ZPRRegClass.contains(RPI.Reg1)) {
      RPI.Type = RegPairInfo::ZPR;
      RPI.RC = &AArch64::ZPRRegClass;
    } else if (AArch64::PPRRegClass.contains(RPI.Reg1)) {
      RPI.Type = RegPairInfo::PPR;
      RPI.RC = &AArch64::PPRRegClass;
    } else if (RPI.Reg1 == AArch64::VG) {
      RPI.Type = RegPairInfo::VG;
      RPI.RC = &AArch64::FIXED_REGSRegClass;
    } else {
      llvm_unreachable("Unsupported register class.");
    }

    int &ScalableByteOffset = RPI.Type == RegPairInfo::PPR && SplitPPRs
                                  ? PPRByteOffset
                                  : ZPRByteOffset;

    // Add the stack hazard size as we transition from GPR->FPR CSRs.
    if (HasCSHazardPadding &&
        (!LastReg || !AArch64InstrInfo::isFpOrNEON(LastReg)) &&
        AArch64InstrInfo::isFpOrNEON(RPI.Reg1))
      ByteOffset += StackFillDir * StackHazardSize;
    LastReg = RPI.Reg1;

    bool NeedsWinCFI = AFL.needsWinCFI(MF);
    int Scale = TRI->getSpillSize(*RPI.RC);
    // Add the next reg to the pair if it is in the same register class.
    if (unsigned(i + RegInc) < Count && !HasCSHazardPadding) {
      MCRegister NextReg = CSI[i + RegInc].getReg();
      unsigned SpillCount = NeedsWinCFI ? FirstReg - i : i;
      switch (RPI.Type) {
      case RegPairInfo::GPR:
        if (AArch64::GPR64RegClass.contains(NextReg) &&
            !invalidateRegisterPairing(SpillExtendedVolatile, SpillCount,
                                       RPI.Reg1, NextReg, IsWindows,
                                       NeedsWinCFI, NeedsFrameRecord, TRI))
          RPI.Reg2 = NextReg;
        break;
      case RegPairInfo::FPR64:
        if (AArch64::FPR64RegClass.contains(NextReg) &&
            !invalidateRegisterPairing(SpillExtendedVolatile, SpillCount,
                                       RPI.Reg1, NextReg, IsWindows,
                                       NeedsWinCFI, NeedsFrameRecord, TRI))
          RPI.Reg2 = NextReg;
        break;
      case RegPairInfo::FPR128:
        if (AArch64::FPR128RegClass.contains(NextReg))
          RPI.Reg2 = NextReg;
        break;
      case RegPairInfo::PPR:
        break;
      case RegPairInfo::ZPR:
        if (AFI->getPredicateRegForFillSpill() != 0 &&
            ((RPI.Reg1 - AArch64::Z0) & 1) == 0 && (NextReg == RPI.Reg1 + 1)) {
          // Calculate offset of register pair to see if pair instruction can be
          // used.
          int Offset = (ScalableByteOffset + StackFillDir * 2 * Scale) / Scale;
          if ((-16 <= Offset && Offset <= 14) && (Offset % 2 == 0))
            RPI.Reg2 = NextReg;
        }
        break;
      case RegPairInfo::VG:
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
    assert((!produceCompactUnwindFrame(AFL, MF) ||
            CC == CallingConv::PreserveMost || CC == CallingConv::PreserveAll ||
            CC == CallingConv::CXX_FAST_TLS || CC == CallingConv::Win64 ||
            (RPI.isPaired() &&
             ((RPI.Reg1 == AArch64::LR && RPI.Reg2 == AArch64::FP) ||
              RPI.Reg1 + 1 == RPI.Reg2))) &&
           "Callee-save registers not saved as adjacent register pair!");

    RPI.FrameIdx = CSI[i].getFrameIdx();
    if (IsWindows &&
        RPI.isPaired()) // RPI.FrameIdx must be the lower index of the pair
      RPI.FrameIdx = CSI[i + RegInc].getFrameIdx();

    // Realign the scalable offset if necessary.  This is relevant when
    // spilling predicates on Windows.
    if (RPI.isScalable() && ScalableByteOffset % Scale != 0) {
      ScalableByteOffset = alignTo(ScalableByteOffset, Scale);
    }

    int OffsetPre = RPI.isScalable() ? ScalableByteOffset : ByteOffset;
    assert(OffsetPre % Scale == 0);

    if (RPI.isScalable())
      ScalableByteOffset += StackFillDir * (RPI.isPaired() ? 2 * Scale : Scale);
    else
      ByteOffset += StackFillDir * (RPI.isPaired() ? 2 * Scale : Scale);

    // Swift's async context is directly before FP, so allocate an extra
    // 8 bytes for it.
    if (NeedsFrameRecord && AFI->hasSwiftAsyncContext() &&
        ((!IsWindows && RPI.Reg2 == AArch64::FP) ||
         (IsWindows && RPI.Reg2 == AArch64::LR)))
      ByteOffset += StackFillDir * 8;

    // Round up size of non-pair to pair size if we need to pad the
    // callee-save area to ensure 16-byte alignment.
    if (NeedGapToAlignStack && !IsWindows && !RPI.isScalable() &&
        RPI.Type != RegPairInfo::FPR128 && !RPI.isPaired() &&
        ByteOffset % 16 != 0) {
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
    // If filling bottom up (WinCFI) we need the original offset.
    int Offset = IsWindows ? OffsetPre : OffsetPost;

    // The FP, LR pair goes 8 bytes into our expanded 24-byte slot so that the
    // Swift context can directly precede FP.
    if (NeedsFrameRecord && AFI->hasSwiftAsyncContext() &&
        ((!IsWindows && RPI.Reg2 == AArch64::FP) ||
         (IsWindows && RPI.Reg2 == AArch64::LR)))
      Offset += 8;
    RPI.Offset = Offset / Scale;

    assert((!RPI.isPaired() ||
            (!RPI.isScalable() && RPI.Offset >= -64 && RPI.Offset <= 63) ||
            (RPI.isScalable() && RPI.Offset >= -256 && RPI.Offset <= 255)) &&
           "Offset out of bounds for LDP/STP immediate");

    auto isFrameRecord = [&] {
      if (RPI.isPaired())
        return IsWindows ? RPI.Reg1 == AArch64::FP && RPI.Reg2 == AArch64::LR
                         : RPI.Reg1 == AArch64::LR && RPI.Reg2 == AArch64::FP;
      // Otherwise, look for the frame record as two unpaired registers. This is
      // needed for -aarch64-stack-hazard-size=<val>, which disables register
      // pairing (as the padding may be too large for the LDP/STP offset). Note:
      // On Windows, this check works out as current reg == FP, next reg == LR,
      // and on other platforms current reg == FP, previous reg == LR. This
      // works out as the correct pre-increment or post-increment offsets
      // respectively.
      return i > 0 && RPI.Reg1 == AArch64::FP &&
             CSI[i - 1].getReg() == AArch64::LR;
    };

    // Save the offset to frame record so that the FP register can point to the
    // innermost frame record (spilled FP and LR registers).
    if (NeedsFrameRecord && isFrameRecord())
      AFI->setCalleeSaveBaseToFrameRecordOffset(Offset);

    RegPairs.push_back(RPI);
    if (RPI.isPaired())
      i += RegInc;
  }
  if (IsWindows) {
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
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  auto &TLI = *Subtarget.getTargetLowering();
  const AArch64InstrInfo &TII = *Subtarget.getInstrInfo();
  bool NeedsWinCFI = needsWinCFI(MF);
  DebugLoc DL;
  SmallVector<RegPairInfo, 8> RegPairs;

  computeCalleeSaveRegisterPairs(*this, MF, CSI, TRI, RegPairs, hasFP(MF));

  MachineRegisterInfo &MRI = MF.getRegInfo();
  // Refresh the reserved regs in case there are any potential changes since the
  // last freeze.
  MRI.freezeReservedRegs();

  if (homogeneousPrologEpilog(MF)) {
    auto MIB = BuildMI(MBB, MI, DL, TII.get(AArch64::HOM_Prolog))
                   .setMIFlag(MachineInstr::FrameSetup);

    for (auto &RPI : RegPairs) {
      MIB.addReg(RPI.Reg1);
      MIB.addReg(RPI.Reg2);

      // Update register live in.
      if (!MRI.isReserved(RPI.Reg1))
        MBB.addLiveIn(RPI.Reg1);
      if (RPI.isPaired() && !MRI.isReserved(RPI.Reg2))
        MBB.addLiveIn(RPI.Reg2);
    }
    return true;
  }
  bool PTrueCreated = false;
  for (const RegPairInfo &RPI : llvm::reverse(RegPairs)) {
    Register Reg1 = RPI.Reg1;
    Register Reg2 = RPI.Reg2;
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
    unsigned Size = TRI->getSpillSize(*RPI.RC);
    Align Alignment = TRI->getSpillAlign(*RPI.RC);
    switch (RPI.Type) {
    case RegPairInfo::GPR:
      StrOpc = RPI.isPaired() ? AArch64::STPXi : AArch64::STRXui;
      break;
    case RegPairInfo::FPR64:
      StrOpc = RPI.isPaired() ? AArch64::STPDi : AArch64::STRDui;
      break;
    case RegPairInfo::FPR128:
      StrOpc = RPI.isPaired() ? AArch64::STPQi : AArch64::STRQui;
      break;
    case RegPairInfo::ZPR:
      StrOpc = RPI.isPaired() ? AArch64::ST1B_2Z_IMM : AArch64::STR_ZXI;
      break;
    case RegPairInfo::PPR:
      StrOpc = AArch64::STR_PXI;
      break;
    case RegPairInfo::VG:
      StrOpc = AArch64::STRXui;
      break;
    }

    Register X0Scratch;
    llvm::scope_exit RestoreX0([&] {
      if (X0Scratch != AArch64::NoRegister)
        BuildMI(MBB, MI, DL, TII.get(TargetOpcode::COPY), AArch64::X0)
            .addReg(X0Scratch)
            .setMIFlag(MachineInstr::FrameSetup);
    });

    if (Reg1 == AArch64::VG) {
      // Find an available register to store value of VG to.
      Reg1 = findScratchNonCalleeSaveRegister(&MBB, true);
      assert(Reg1 != AArch64::NoRegister);
      if (MF.getSubtarget<AArch64Subtarget>().hasSVE()) {
        BuildMI(MBB, MI, DL, TII.get(AArch64::CNTD_XPiI), Reg1)
            .addImm(31)
            .addImm(1)
            .setMIFlag(MachineInstr::FrameSetup);
      } else {
        const AArch64Subtarget &STI = MF.getSubtarget<AArch64Subtarget>();
        if (any_of(MBB.liveins(),
                   [&STI](const MachineBasicBlock::RegisterMaskPair &LiveIn) {
                     return STI.getRegisterInfo()->isSuperOrSubRegisterEq(
                         AArch64::X0, LiveIn.PhysReg);
                   })) {
          X0Scratch = Reg1;
          BuildMI(MBB, MI, DL, TII.get(TargetOpcode::COPY), X0Scratch)
              .addReg(AArch64::X0)
              .setMIFlag(MachineInstr::FrameSetup);
        }

        RTLIB::Libcall LC = RTLIB::SMEABI_GET_CURRENT_VG;
        const uint32_t *RegMask =
            TRI->getCallPreservedMask(MF, TLI.getLibcallCallingConv(LC));
        BuildMI(MBB, MI, DL, TII.get(AArch64::BL))
            .addExternalSymbol(TLI.getLibcallName(LC))
            .addRegMask(RegMask)
            .addReg(AArch64::X0, RegState::ImplicitDefine)
            .setMIFlag(MachineInstr::FrameSetup);
        Reg1 = AArch64::X0;
      }
    }

    LLVM_DEBUG({
      dbgs() << "CSR spill: (" << printReg(Reg1, TRI);
      if (RPI.isPaired())
        dbgs() << ", " << printReg(Reg2, TRI);
      dbgs() << ") -> fi#(" << RPI.FrameIdx;
      if (RPI.isPaired())
        dbgs() << ", " << RPI.FrameIdx + 1;
      dbgs() << ")\n";
    });

    assert((!isTargetWindows(MF) ||
            !(Reg1 == AArch64::LR && Reg2 == AArch64::FP)) &&
           "Windows unwdinding requires a consecutive (FP,LR) pair");
    // Windows unwind codes require consecutive registers if registers are
    // paired.  Make the switch here, so that the code below will save (x,x+1)
    // and not (x+1,x).
    unsigned FrameIdxReg1 = RPI.FrameIdx;
    unsigned FrameIdxReg2 = RPI.FrameIdx + 1;
    if (isTargetWindows(MF) && RPI.isPaired()) {
      std::swap(Reg1, Reg2);
      std::swap(FrameIdxReg1, FrameIdxReg2);
    }

    if (RPI.isPaired() && RPI.isScalable()) {
      [[maybe_unused]] const AArch64Subtarget &Subtarget =
                              MF.getSubtarget<AArch64Subtarget>();
      AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
      unsigned PnReg = AFI->getPredicateRegForFillSpill();
      assert((PnReg != 0 && enableMultiVectorSpillFill(Subtarget, MF)) &&
             "Expects SVE2.1 or SME2 target and a predicate register");
#ifdef EXPENSIVE_CHECKS
      auto IsPPR = [](const RegPairInfo &c) {
        return c.Reg1 == RegPairInfo::PPR;
      };
      auto PPRBegin = std::find_if(RegPairs.begin(), RegPairs.end(), IsPPR);
      auto IsZPR = [](const RegPairInfo &c) {
        return c.Type == RegPairInfo::ZPR;
      };
      auto ZPRBegin = std::find_if(RegPairs.begin(), RegPairs.end(), IsZPR);
      assert(!(PPRBegin < ZPRBegin) &&
             "Expected callee save predicate to be handled first");
#endif
      if (!PTrueCreated) {
        PTrueCreated = true;
        BuildMI(MBB, MI, DL, TII.get(AArch64::PTRUE_C_B), PnReg)
            .setMIFlags(MachineInstr::FrameSetup);
      }
      MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII.get(StrOpc));
      if (!MRI.isReserved(Reg1))
        MBB.addLiveIn(Reg1);
      if (!MRI.isReserved(Reg2))
        MBB.addLiveIn(Reg2);
      MIB.addReg(/*PairRegs*/ AArch64::Z0_Z1 + (RPI.Reg1 - AArch64::Z0));
      MIB.addMemOperand(MF.getMachineMemOperand(
          MachinePointerInfo::getFixedStack(MF, FrameIdxReg2),
          MachineMemOperand::MOStore, Size, Alignment));
      MIB.addReg(PnReg);
      MIB.addReg(AArch64::SP)
          .addImm(RPI.Offset / 2) // [sp, #imm*2*vscale],
                                  // where 2*vscale is implicit
          .setMIFlag(MachineInstr::FrameSetup);
      MIB.addMemOperand(MF.getMachineMemOperand(
          MachinePointerInfo::getFixedStack(MF, FrameIdxReg1),
          MachineMemOperand::MOStore, Size, Alignment));
      if (NeedsWinCFI)
        insertSEH(MIB, TII, MachineInstr::FrameSetup);
    } else { // The code when the pair of ZReg is not present
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
          .addImm(RPI.Offset) // [sp, #offset*vscale],
                              // where factor*vscale is implicit
          .setMIFlag(MachineInstr::FrameSetup);
      MIB.addMemOperand(MF.getMachineMemOperand(
          MachinePointerInfo::getFixedStack(MF, FrameIdxReg1),
          MachineMemOperand::MOStore, Size, Alignment));
      if (NeedsWinCFI)
        insertSEH(MIB, TII, MachineInstr::FrameSetup);
    }
    // Update the StackIDs of the SVE stack slots.
    MachineFrameInfo &MFI = MF.getFrameInfo();
    if (RPI.Type == RegPairInfo::ZPR) {
      MFI.setStackID(FrameIdxReg1, TargetStackID::ScalableVector);
      if (RPI.isPaired())
        MFI.setStackID(FrameIdxReg2, TargetStackID::ScalableVector);
    } else if (RPI.Type == RegPairInfo::PPR) {
      MFI.setStackID(FrameIdxReg1, TargetStackID::ScalablePredicateVector);
      if (RPI.isPaired())
        MFI.setStackID(FrameIdxReg2, TargetStackID::ScalablePredicateVector);
    }
  }
  return true;
}

bool AArch64FrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();
  const AArch64InstrInfo &TII =
      *MF.getSubtarget<AArch64Subtarget>().getInstrInfo();
  DebugLoc DL;
  SmallVector<RegPairInfo, 8> RegPairs;
  bool NeedsWinCFI = needsWinCFI(MF);

  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();

  computeCalleeSaveRegisterPairs(*this, MF, CSI, TRI, RegPairs, hasFP(MF));
  if (homogeneousPrologEpilog(MF, &MBB)) {
    auto MIB = BuildMI(MBB, MBBI, DL, TII.get(AArch64::HOM_Epilog))
                   .setMIFlag(MachineInstr::FrameDestroy);
    for (auto &RPI : RegPairs) {
      MIB.addReg(RPI.Reg1, RegState::Define);
      MIB.addReg(RPI.Reg2, RegState::Define);
    }
    return true;
  }

  // For performance reasons restore SVE register in increasing order
  auto IsPPR = [](const RegPairInfo &c) { return c.Type == RegPairInfo::PPR; };
  auto PPRBegin = llvm::find_if(RegPairs, IsPPR);
  auto PPREnd = std::find_if_not(PPRBegin, RegPairs.end(), IsPPR);
  std::reverse(PPRBegin, PPREnd);
  auto IsZPR = [](const RegPairInfo &c) { return c.Type == RegPairInfo::ZPR; };
  auto ZPRBegin = llvm::find_if(RegPairs, IsZPR);
  auto ZPREnd = std::find_if_not(ZPRBegin, RegPairs.end(), IsZPR);
  std::reverse(ZPRBegin, ZPREnd);

  bool PTrueCreated = false;
  for (const RegPairInfo &RPI : RegPairs) {
    Register Reg1 = RPI.Reg1;
    Register Reg2 = RPI.Reg2;

    // Issue sequence of restores for cs regs. The last restore may be converted
    // to a post-increment load later by emitEpilogue if the callee-save stack
    // area allocation can't be combined with the local stack area allocation.
    // For example:
    //    ldp     fp, lr, [sp, #32]       // addImm(+4)
    //    ldp     x20, x19, [sp, #16]     // addImm(+2)
    //    ldp     x22, x21, [sp, #0]      // addImm(+0)
    // Note: see comment in spillCalleeSavedRegisters()
    unsigned LdrOpc;
    unsigned Size = TRI->getSpillSize(*RPI.RC);
    Align Alignment = TRI->getSpillAlign(*RPI.RC);
    switch (RPI.Type) {
    case RegPairInfo::GPR:
      LdrOpc = RPI.isPaired() ? AArch64::LDPXi : AArch64::LDRXui;
      break;
    case RegPairInfo::FPR64:
      LdrOpc = RPI.isPaired() ? AArch64::LDPDi : AArch64::LDRDui;
      break;
    case RegPairInfo::FPR128:
      LdrOpc = RPI.isPaired() ? AArch64::LDPQi : AArch64::LDRQui;
      break;
    case RegPairInfo::ZPR:
      LdrOpc = RPI.isPaired() ? AArch64::LD1B_2Z_IMM : AArch64::LDR_ZXI;
      break;
    case RegPairInfo::PPR:
      LdrOpc = AArch64::LDR_PXI;
      break;
    case RegPairInfo::VG:
      continue;
    }
    LLVM_DEBUG({
      dbgs() << "CSR restore: (" << printReg(Reg1, TRI);
      if (RPI.isPaired())
        dbgs() << ", " << printReg(Reg2, TRI);
      dbgs() << ") -> fi#(" << RPI.FrameIdx;
      if (RPI.isPaired())
        dbgs() << ", " << RPI.FrameIdx + 1;
      dbgs() << ")\n";
    });

    // Windows unwind codes require consecutive registers if registers are
    // paired.  Make the switch here, so that the code below will save (x,x+1)
    // and not (x+1,x).
    unsigned FrameIdxReg1 = RPI.FrameIdx;
    unsigned FrameIdxReg2 = RPI.FrameIdx + 1;
    if (isTargetWindows(MF) && RPI.isPaired()) {
      std::swap(Reg1, Reg2);
      std::swap(FrameIdxReg1, FrameIdxReg2);
    }

    AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
    if (RPI.isPaired() && RPI.isScalable()) {
      [[maybe_unused]] const AArch64Subtarget &Subtarget =
                              MF.getSubtarget<AArch64Subtarget>();
      unsigned PnReg = AFI->getPredicateRegForFillSpill();
      assert((PnReg != 0 && enableMultiVectorSpillFill(Subtarget, MF)) &&
             "Expects SVE2.1 or SME2 target and a predicate register");
#ifdef EXPENSIVE_CHECKS
      assert(!(PPRBegin < ZPRBegin) &&
             "Expected callee save predicate to be handled first");
#endif
      if (!PTrueCreated) {
        PTrueCreated = true;
        BuildMI(MBB, MBBI, DL, TII.get(AArch64::PTRUE_C_B), PnReg)
            .setMIFlags(MachineInstr::FrameDestroy);
      }
      MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII.get(LdrOpc));
      MIB.addReg(/*PairRegs*/ AArch64::Z0_Z1 + (RPI.Reg1 - AArch64::Z0),
                 getDefRegState(true));
      MIB.addMemOperand(MF.getMachineMemOperand(
          MachinePointerInfo::getFixedStack(MF, FrameIdxReg2),
          MachineMemOperand::MOLoad, Size, Alignment));
      MIB.addReg(PnReg);
      MIB.addReg(AArch64::SP)
          .addImm(RPI.Offset / 2) // [sp, #imm*2*vscale]
                                  // where 2*vscale is implicit
          .setMIFlag(MachineInstr::FrameDestroy);
      MIB.addMemOperand(MF.getMachineMemOperand(
          MachinePointerInfo::getFixedStack(MF, FrameIdxReg1),
          MachineMemOperand::MOLoad, Size, Alignment));
      if (NeedsWinCFI)
        insertSEH(MIB, TII, MachineInstr::FrameDestroy);
    } else {
      MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII.get(LdrOpc));
      if (RPI.isPaired()) {
        MIB.addReg(Reg2, getDefRegState(true));
        MIB.addMemOperand(MF.getMachineMemOperand(
            MachinePointerInfo::getFixedStack(MF, FrameIdxReg2),
            MachineMemOperand::MOLoad, Size, Alignment));
      }
      MIB.addReg(Reg1, getDefRegState(true));
      MIB.addReg(AArch64::SP)
          .addImm(RPI.Offset) // [sp, #offset*vscale]
                              // where factor*vscale is implicit
          .setMIFlag(MachineInstr::FrameDestroy);
      MIB.addMemOperand(MF.getMachineMemOperand(
          MachinePointerInfo::getFixedStack(MF, FrameIdxReg1),
          MachineMemOperand::MOLoad, Size, Alignment));
      if (NeedsWinCFI)
        insertSEH(MIB, TII, MachineInstr::FrameDestroy);
    }
  }
  return true;
}

// Return the FrameID for a MMO.
static std::optional<int> getMMOFrameID(MachineMemOperand *MMO,
                                        const MachineFrameInfo &MFI) {
  auto *PSV =
      dyn_cast_or_null<FixedStackPseudoSourceValue>(MMO->getPseudoValue());
  if (PSV)
    return std::optional<int>(PSV->getFrameIndex());

  if (MMO->getValue()) {
    if (auto *Al = dyn_cast<AllocaInst>(getUnderlyingObject(MMO->getValue()))) {
      for (int FI = MFI.getObjectIndexBegin(); FI < MFI.getObjectIndexEnd();
           FI++)
        if (MFI.getObjectAllocation(FI) == Al)
          return FI;
    }
  }

  return std::nullopt;
}

// Return the FrameID for a Load/Store instruction by looking at the first MMO.
static std::optional<int> getLdStFrameID(const MachineInstr &MI,
                                         const MachineFrameInfo &MFI) {
  if (!MI.mayLoadOrStore() || MI.getNumMemOperands() < 1)
    return std::nullopt;

  return getMMOFrameID(*MI.memoperands_begin(), MFI);
}

// Returns true if the LDST MachineInstr \p MI is a PPR access.
static bool isPPRAccess(const MachineInstr &MI) {
  return AArch64::PPRRegClass.contains(MI.getOperand(0).getReg());
}

// Check if a Hazard slot is needed for the current function, and if so create
// one for it. The index is stored in AArch64FunctionInfo->StackHazardSlotIndex,
// which can be used to determine if any hazard padding is needed.
void AArch64FrameLowering::determineStackHazardSlot(
    MachineFunction &MF, BitVector &SavedRegs) const {
  unsigned StackHazardSize = getStackHazardSize(MF);
  auto *AFI = MF.getInfo<AArch64FunctionInfo>();
  if (StackHazardSize == 0 || StackHazardSize % 16 != 0 ||
      AFI->hasStackHazardSlotIndex())
    return;

  // Stack hazards are only needed in streaming functions.
  SMEAttrs Attrs = AFI->getSMEFnAttrs();
  if (!StackHazardInNonStreaming && Attrs.hasNonStreamingInterfaceAndBody())
    return;

  MachineFrameInfo &MFI = MF.getFrameInfo();

  // Add a hazard slot if there are any CSR FPR registers, or are any fp-only
  // stack objects.
  bool HasFPRCSRs = any_of(SavedRegs.set_bits(), [](unsigned Reg) {
    return AArch64::FPR64RegClass.contains(Reg) ||
           AArch64::FPR128RegClass.contains(Reg) ||
           AArch64::ZPRRegClass.contains(Reg);
  });
  bool HasPPRCSRs = any_of(SavedRegs.set_bits(), [](unsigned Reg) {
    return AArch64::PPRRegClass.contains(Reg);
  });
  bool HasFPRStackObjects = false;
  bool HasPPRStackObjects = false;
  if (!HasFPRCSRs || SplitSVEObjects) {
    enum SlotType : uint8_t {
      Unknown = 0,
      ZPRorFPR = 1 << 0,
      PPR = 1 << 1,
      GPR = 1 << 2,
      LLVM_MARK_AS_BITMASK_ENUM(GPR)
    };

    // Find stack slots solely used for one kind of register (ZPR, PPR, etc.),
    // based on the kinds of accesses used in the function.
    SmallVector<SlotType> SlotTypes(MFI.getObjectIndexEnd(), SlotType::Unknown);
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        std::optional<int> FI = getLdStFrameID(MI, MFI);
        if (!FI || FI < 0 || FI > int(SlotTypes.size()))
          continue;
        if (MFI.hasScalableStackID(*FI)) {
          SlotTypes[*FI] |=
              isPPRAccess(MI) ? SlotType::PPR : SlotType::ZPRorFPR;
        } else {
          SlotTypes[*FI] |= AArch64InstrInfo::isFpOrNEON(MI)
                                ? SlotType::ZPRorFPR
                                : SlotType::GPR;
        }
      }
    }

    for (int FI = 0; FI < int(SlotTypes.size()); ++FI) {
      HasFPRStackObjects |= SlotTypes[FI] == SlotType::ZPRorFPR;
      // For SplitSVEObjects remember that this stack slot is a predicate, this
      // will be needed later when determining the frame layout.
      if (SlotTypes[FI] == SlotType::PPR) {
        MFI.setStackID(FI, TargetStackID::ScalablePredicateVector);
        HasPPRStackObjects = true;
      }
    }
  }

  if (HasFPRCSRs || HasFPRStackObjects) {
    int ID = MFI.CreateStackObject(StackHazardSize, Align(16), false);
    LLVM_DEBUG(dbgs() << "Created Hazard slot at " << ID << " size "
                      << StackHazardSize << "\n");
    AFI->setStackHazardSlotIndex(ID);
  }

  if (!AFI->hasStackHazardSlotIndex())
    return;

  if (SplitSVEObjects) {
    CallingConv::ID CC = MF.getFunction().getCallingConv();
    if (AFI->isSVECC() || CC == CallingConv::AArch64_SVE_VectorCall) {
      AFI->setSplitSVEObjects(true);
      LLVM_DEBUG(dbgs() << "Using SplitSVEObjects for SVE CC function\n");
      return;
    }

    // We only use SplitSVEObjects in non-SVE CC functions if there's a
    // possibility of a stack hazard between PPRs and ZPRs/FPRs.
    LLVM_DEBUG(dbgs() << "Determining if SplitSVEObjects should be used in "
                         "non-SVE CC function...\n");

    // If another calling convention is explicitly set FPRs can't be promoted to
    // ZPR callee-saves.
    if (!is_contained({CallingConv::C, CallingConv::Fast}, CC)) {
      LLVM_DEBUG(
          dbgs()
          << "Calling convention is not supported with SplitSVEObjects\n");
      return;
    }

    if (!HasPPRCSRs && !HasPPRStackObjects) {
      LLVM_DEBUG(
          dbgs() << "Not using SplitSVEObjects as no PPRs are on the stack\n");
      return;
    }

    if (!HasFPRCSRs && !HasFPRStackObjects) {
      LLVM_DEBUG(
          dbgs()
          << "Not using SplitSVEObjects as no FPRs or ZPRs are on the stack\n");
      return;
    }

    [[maybe_unused]] const AArch64Subtarget &Subtarget =
        MF.getSubtarget<AArch64Subtarget>();
    assert(Subtarget.isSVEorStreamingSVEAvailable() &&
           "Expected SVE to be available for PPRs");

    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    // With SplitSVEObjects the CS hazard padding is placed between the
    // PPRs and ZPRs. If there are any FPR CS there would be a hazard between
    // them and the CS GRPs. Avoid this by promoting all FPR CS to ZPRs.
    BitVector FPRZRegs(SavedRegs.size());
    for (size_t Reg = 0, E = SavedRegs.size(); HasFPRCSRs && Reg < E; ++Reg) {
      BitVector::reference RegBit = SavedRegs[Reg];
      if (!RegBit)
        continue;
      unsigned SubRegIdx = 0;
      if (AArch64::FPR64RegClass.contains(Reg))
        SubRegIdx = AArch64::dsub;
      else if (AArch64::FPR128RegClass.contains(Reg))
        SubRegIdx = AArch64::zsub;
      else
        continue;
      // Clear the bit for the FPR save.
      RegBit = false;
      // Mark that we should save the corresponding ZPR.
      Register ZReg =
          TRI->getMatchingSuperReg(Reg, SubRegIdx, &AArch64::ZPRRegClass);
      FPRZRegs.set(ZReg);
    }
    SavedRegs |= FPRZRegs;

    AFI->setSplitSVEObjects(true);
    LLVM_DEBUG(dbgs() << "SplitSVEObjects enabled!\n");
  }
}

void AArch64FrameLowering::determineCalleeSaves(MachineFunction &MF,
                                                BitVector &SavedRegs,
                                                RegScavenger *RS) const {
  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction().getCallingConv() == CallingConv::GHC)
    return;

  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();

  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  const AArch64RegisterInfo *RegInfo = Subtarget.getRegisterInfo();
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  unsigned UnspilledCSGPR = AArch64::NoRegister;
  unsigned UnspilledCSGPRPaired = AArch64::NoRegister;

  MachineFrameInfo &MFI = MF.getFrameInfo();
  const MCPhysReg *CSRegs = MF.getRegInfo().getCalleeSavedRegs();

  MCRegister BasePointerReg =
      RegInfo->hasBasePointer(MF) ? RegInfo->getBaseRegister() : MCRegister();

  unsigned ExtraCSSpill = 0;
  bool HasUnpairedGPR64 = false;
  bool HasPairZReg = false;
  BitVector UserReservedRegs = RegInfo->getUserReservedRegs(MF);
  BitVector ReservedRegs = RegInfo->getReservedRegs(MF);

  // Figure out which callee-saved registers to save/restore.
  for (unsigned i = 0; CSRegs[i]; ++i) {
    const MCRegister Reg = CSRegs[i];

    // Add the base pointer register to SavedRegs if it is callee-save.
    if (Reg == BasePointerReg)
      SavedRegs.set(Reg);

    // Don't save manually reserved registers set through +reserve-x#i,
    // even for callee-saved registers, as per GCC's behavior.
    if (UserReservedRegs[Reg]) {
      SavedRegs.reset(Reg);
      continue;
    }

    bool RegUsed = SavedRegs.test(Reg);
    MCRegister PairedReg;
    const bool RegIsGPR64 = AArch64::GPR64RegClass.contains(Reg);
    if (RegIsGPR64 || AArch64::FPR64RegClass.contains(Reg) ||
        AArch64::FPR128RegClass.contains(Reg)) {
      // Compensate for odd numbers of GP CSRs.
      // For now, all the known cases of odd number of CSRs are of GPRs.
      if (HasUnpairedGPR64)
        PairedReg = CSRegs[i % 2 == 0 ? i - 1 : i + 1];
      else
        PairedReg = CSRegs[i ^ 1];
    }

    // If the function requires all the GP registers to save (SavedRegs),
    // and there are an odd number of GP CSRs at the same time (CSRegs),
    // PairedReg could be in a different register class from Reg, which would
    // lead to a FPR (usually D8) accidentally being marked saved.
    if (RegIsGPR64 && !AArch64::GPR64RegClass.contains(PairedReg)) {
      PairedReg = AArch64::NoRegister;
      HasUnpairedGPR64 = true;
    }
    assert(PairedReg == AArch64::NoRegister ||
           AArch64::GPR64RegClass.contains(Reg, PairedReg) ||
           AArch64::FPR64RegClass.contains(Reg, PairedReg) ||
           AArch64::FPR128RegClass.contains(Reg, PairedReg));

    if (!RegUsed) {
      if (AArch64::GPR64RegClass.contains(Reg) && !ReservedRegs[Reg]) {
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
          !ReservedRegs[PairedReg])
        ExtraCSSpill = PairedReg;
    }
    // Check if there is a pair of ZRegs, so it can select PReg for spill/fill
    HasPairZReg |= (AArch64::ZPRRegClass.contains(Reg, CSRegs[i ^ 1]) &&
                    SavedRegs.test(CSRegs[i ^ 1]));
  }

  if (HasPairZReg && enableMultiVectorSpillFill(Subtarget, MF)) {
    AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
    // Find a suitable predicate register for the multi-vector spill/fill
    // instructions.
    MCRegister PnReg = findFreePredicateReg(SavedRegs);
    if (PnReg.isValid())
      AFI->setPredicateRegForFillSpill(PnReg);
    // If no free callee-save has been found assign one.
    if (!AFI->getPredicateRegForFillSpill() &&
        MF.getFunction().getCallingConv() ==
            CallingConv::AArch64_SVE_VectorCall) {
      SavedRegs.set(AArch64::P8);
      AFI->setPredicateRegForFillSpill(AArch64::PN8);
    }

    assert(!ReservedRegs[AFI->getPredicateRegForFillSpill()] &&
           "Predicate cannot be a reserved register");
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

  // Determine if a Hazard slot should be used and where it should go.
  // If SplitSVEObjects is used, the hazard padding is placed between the PPRs
  // and ZPRs. Otherwise, it goes in the callee save area.
  determineStackHazardSlot(MF, SavedRegs);

  // Calculates the callee saved stack size.
  unsigned CSStackSize = 0;
  unsigned ZPRCSStackSize = 0;
  unsigned PPRCSStackSize = 0;
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  for (unsigned Reg : SavedRegs.set_bits()) {
    auto *RC = TRI->getMinimalPhysRegClass(MCRegister(Reg));
    assert(RC && "expected register class!");
    auto SpillSize = TRI->getSpillSize(*RC);
    bool IsZPR = AArch64::ZPRRegClass.contains(Reg);
    bool IsPPR = !IsZPR && AArch64::PPRRegClass.contains(Reg);
    if (IsZPR)
      ZPRCSStackSize += SpillSize;
    else if (IsPPR)
      PPRCSStackSize += SpillSize;
    else
      CSStackSize += SpillSize;
  }

  // Save number of saved regs, so we can easily update CSStackSize later to
  // account for any additional 64-bit GPR saves. Note: After this point
  // only 64-bit GPRs can be added to SavedRegs.
  unsigned NumSavedRegs = SavedRegs.count();

  // If we have hazard padding in the CS area add that to the size.
  if (AFI->isStackHazardIncludedInCalleeSaveArea())
    CSStackSize += getStackHazardSize(MF);

  // Increase the callee-saved stack size if the function has streaming mode
  // changes, as we will need to spill the value of the VG register.
  if (requiresSaveVG(MF))
    CSStackSize += 8;

  // If we must call __arm_get_current_vg in the prologue preserve the LR.
  if (requiresSaveVG(MF) && !Subtarget.hasSVE())
    SavedRegs.set(AArch64::LR);

  // The frame record needs to be created by saving the appropriate registers
  uint64_t EstimatedStackSize = MFI.estimateStackSize(MF);
  if (hasFP(MF) ||
      windowsRequiresStackProbe(MF, EstimatedStackSize + CSStackSize + 16)) {
    SavedRegs.set(AArch64::FP);
    SavedRegs.set(AArch64::LR);
  }

  LLVM_DEBUG({
    dbgs() << "*** determineCalleeSaves\nSaved CSRs:";
    for (unsigned Reg : SavedRegs.set_bits())
      dbgs() << ' ' << printReg(MCRegister(Reg), RegInfo);
    dbgs() << "\n";
  });

  // If any callee-saved registers are used, the frame cannot be eliminated.
  auto [ZPRLocalStackSize, PPRLocalStackSize] =
      determineSVEStackSizes(MF, AssignObjectOffsets::No);
  uint64_t SVELocals = ZPRLocalStackSize + PPRLocalStackSize;
  uint64_t SVEStackSize =
      alignTo(ZPRCSStackSize + PPRCSStackSize + SVELocals, 16);
  bool CanEliminateFrame = (SavedRegs.count() == 0) && !SVEStackSize;

  // The CSR spill slots have not been allocated yet, so estimateStackSize
  // won't include them.
  unsigned EstimatedStackSizeLimit = estimateRSStackSizeLimit(MF);

  // We may address some of the stack above the canonical frame address, either
  // for our own arguments or during a call. Include that in calculating whether
  // we have complicated addressing concerns.
  int64_t CalleeStackUsed = 0;
  for (int I = MFI.getObjectIndexBegin(); I != 0; ++I) {
    int64_t FixedOff = MFI.getObjectOffset(I);
    if (FixedOff > CalleeStackUsed)
      CalleeStackUsed = FixedOff;
  }

  // Conservatively always assume BigStack when there are SVE spills.
  bool BigStack = SVEStackSize || (EstimatedStackSize + CSStackSize +
                                   CalleeStackUsed) > EstimatedStackSizeLimit;
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
      ExtraCSSpill = UnspilledCSGPR;

      // MachO's compact unwind format relies on all registers being stored in
      // pairs, so if we need to spill one extra for BigStack, then we need to
      // store the pair.
      if (producePairRegisters(MF)) {
        if (UnspilledCSGPRPaired == AArch64::NoRegister) {
          // Failed to make a pair for compact unwind format, revert spilling.
          if (produceCompactUnwindFrame(*this, MF)) {
            SavedRegs.reset(UnspilledCSGPR);
            ExtraCSSpill = AArch64::NoRegister;
          }
        } else
          SavedRegs.set(UnspilledCSGPRPaired);
      }
    }

    // If we didn't find an extra callee-saved register to spill, create
    // an emergency spill slot.
    if (!ExtraCSSpill || MF.getRegInfo().isPhysRegUsed(ExtraCSSpill)) {
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      const TargetRegisterClass &RC = AArch64::GPR64RegClass;
      unsigned Size = TRI->getSpillSize(RC);
      Align Alignment = TRI->getSpillAlign(RC);
      int FI = MFI.CreateSpillStackObject(Size, Alignment);
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
                    << EstimatedStackSize + AlignedCSStackSize << " bytes.\n");

  assert((!MFI.isCalleeSavedInfoValid() ||
          AFI->getCalleeSavedStackSize() == AlignedCSStackSize) &&
         "Should not invalidate callee saved info");

  // Round up to register pair alignment to avoid additional SP adjustment
  // instructions.
  AFI->setCalleeSavedStackSize(AlignedCSStackSize);
  AFI->setCalleeSaveStackHasFreeSpace(AlignedCSStackSize != CSStackSize);
  AFI->setSVECalleeSavedStackSize(ZPRCSStackSize, alignTo(PPRCSStackSize, 16));
}

bool AArch64FrameLowering::assignCalleeSavedSpillSlots(
    MachineFunction &MF, const TargetRegisterInfo *RegInfo,
    std::vector<CalleeSavedInfo> &CSI) const {
  bool IsWindows = isTargetWindows(MF);
  unsigned StackHazardSize = getStackHazardSize(MF);
  // To match the canonical windows frame layout, reverse the list of
  // callee saved registers to get them laid out by PrologEpilogInserter
  // in the right order. (PrologEpilogInserter allocates stack objects top
  // down. Windows canonical prologs store higher numbered registers at
  // the top, thus have the CSI array start from the highest registers.)
  if (IsWindows)
    std::reverse(CSI.begin(), CSI.end());

  if (CSI.empty())
    return true; // Early exit if no callee saved registers are modified!

  // Now that we know which registers need to be saved and restored, allocate
  // stack slots for them.
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *AFI = MF.getInfo<AArch64FunctionInfo>();

  if (IsWindows && hasFP(MF) && AFI->hasSwiftAsyncContext()) {
    int FrameIdx = MFI.CreateStackObject(8, Align(16), true);
    AFI->setSwiftAsyncContextFrameIdx(FrameIdx);
    MFI.setIsCalleeSavedObjectIndex(FrameIdx, true);
  }

  // Insert VG into the list of CSRs, immediately before LR if saved.
  if (requiresSaveVG(MF)) {
    CalleeSavedInfo VGInfo(AArch64::VG);
    auto It =
        find_if(CSI, [](auto &Info) { return Info.getReg() == AArch64::LR; });
    if (It != CSI.end())
      CSI.insert(It, VGInfo);
    else
      CSI.push_back(VGInfo);
  }

  Register LastReg = 0;
  int HazardSlotIndex = std::numeric_limits<int>::max();
  for (auto &CS : CSI) {
    MCRegister Reg = CS.getReg();
    const TargetRegisterClass *RC = RegInfo->getMinimalPhysRegClass(Reg);

    // Create a hazard slot as we switch between GPR and FPR CSRs.
    if (AFI->isStackHazardIncludedInCalleeSaveArea() &&
        (!LastReg || !AArch64InstrInfo::isFpOrNEON(LastReg)) &&
        AArch64InstrInfo::isFpOrNEON(Reg)) {
      assert(HazardSlotIndex == std::numeric_limits<int>::max() &&
             "Unexpected register order for hazard slot");
      HazardSlotIndex = MFI.CreateStackObject(StackHazardSize, Align(8), true);
      LLVM_DEBUG(dbgs() << "Created CSR Hazard at slot " << HazardSlotIndex
                        << "\n");
      AFI->setStackHazardCSRSlotIndex(HazardSlotIndex);
      MFI.setIsCalleeSavedObjectIndex(HazardSlotIndex, true);
    }

    unsigned Size = RegInfo->getSpillSize(*RC);
    Align Alignment(RegInfo->getSpillAlign(*RC));
    int FrameIdx = MFI.CreateStackObject(Size, Alignment, true);
    CS.setFrameIdx(FrameIdx);
    MFI.setIsCalleeSavedObjectIndex(FrameIdx, true);

    // Grab 8 bytes below FP for the extended asynchronous frame info.
    if (hasFP(MF) && AFI->hasSwiftAsyncContext() && !IsWindows &&
        Reg == AArch64::FP) {
      FrameIdx = MFI.CreateStackObject(8, Alignment, true);
      AFI->setSwiftAsyncContextFrameIdx(FrameIdx);
      MFI.setIsCalleeSavedObjectIndex(FrameIdx, true);
    }
    LastReg = Reg;
  }

  // Add hazard slot in the case where no FPR CSRs are present.
  if (AFI->isStackHazardIncludedInCalleeSaveArea() &&
      HazardSlotIndex == std::numeric_limits<int>::max()) {
    HazardSlotIndex = MFI.CreateStackObject(StackHazardSize, Align(8), true);
    LLVM_DEBUG(dbgs() << "Created CSR Hazard at slot " << HazardSlotIndex
                      << "\n");
    AFI->setStackHazardCSRSlotIndex(HazardSlotIndex);
    MFI.setIsCalleeSavedObjectIndex(HazardSlotIndex, true);
  }

  return true;
}

bool AArch64FrameLowering::enableStackSlotScavenging(
    const MachineFunction &MF) const {
  const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  // If the function has streaming-mode changes, don't scavenge a
  // spillslot in the callee-save area, as that might require an
  // 'addvl' in the streaming-mode-changing call-sequence when the
  // function doesn't use a FP.
  if (AFI->hasStreamingModeChanges() && !hasFP(MF))
    return false;
  // Don't allow register salvaging with hazard slots, in case it moves objects
  // into the wrong place.
  if (AFI->hasStackHazardSlotIndex())
    return false;
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

static SVEStackSizes determineSVEStackSizes(MachineFunction &MF,
                                            AssignObjectOffsets AssignOffsets) {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *AFI = MF.getInfo<AArch64FunctionInfo>();

  SVEStackSizes SVEStack{};

  // With SplitSVEObjects we maintain separate stack offsets for predicates
  // (PPRs) and SVE vectors (ZPRs). When SplitSVEObjects is disabled predicates
  // are included in the SVE vector area.
  uint64_t &ZPRStackTop = SVEStack.ZPRStackSize;
  uint64_t &PPRStackTop =
      AFI->hasSplitSVEObjects() ? SVEStack.PPRStackSize : SVEStack.ZPRStackSize;

#ifndef NDEBUG
  // First process all fixed stack objects.
  for (int I = MFI.getObjectIndexBegin(); I != 0; ++I)
    assert(!MFI.hasScalableStackID(I) &&
           "SVE vectors should never be passed on the stack by value, only by "
           "reference.");
#endif

  auto AllocateObject = [&](int FI) {
    uint64_t &StackTop = MFI.getStackID(FI) == TargetStackID::ScalableVector
                             ? ZPRStackTop
                             : PPRStackTop;

    // FIXME: Given that the length of SVE vectors is not necessarily a power of
    // two, we'd need to align every object dynamically at runtime if the
    // alignment is larger than 16. This is not yet supported.
    Align Alignment = MFI.getObjectAlign(FI);
    if (Alignment > Align(16))
      report_fatal_error(
          "Alignment of scalable vectors > 16 bytes is not yet supported");

    StackTop += MFI.getObjectSize(FI);
    StackTop = alignTo(StackTop, Alignment);

    assert(StackTop < (uint64_t)std::numeric_limits<int64_t>::max() &&
           "SVE StackTop far too large?!");

    int64_t Offset = -int64_t(StackTop);
    if (AssignOffsets == AssignObjectOffsets::Yes)
      MFI.setObjectOffset(FI, Offset);

    LLVM_DEBUG(dbgs() << "alloc FI(" << FI << ") at SP[" << Offset << "]\n");
  };

  // Then process all callee saved slots.
  int MinCSFrameIndex, MaxCSFrameIndex;
  if (getSVECalleeSaveSlotRange(MFI, MinCSFrameIndex, MaxCSFrameIndex)) {
    for (int FI = MinCSFrameIndex; FI <= MaxCSFrameIndex; ++FI)
      AllocateObject(FI);
  }

  // Ensure the CS area is 16-byte aligned.
  PPRStackTop = alignTo(PPRStackTop, Align(16U));
  ZPRStackTop = alignTo(ZPRStackTop, Align(16U));

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

  for (int FI = 0, E = MFI.getObjectIndexEnd(); FI != E; ++FI) {
    if (FI == StackProtectorFI || MFI.isDeadObjectIndex(FI) ||
        MFI.isCalleeSavedObjectIndex(FI))
      continue;

    if (MFI.getStackID(FI) != TargetStackID::ScalableVector &&
        MFI.getStackID(FI) != TargetStackID::ScalablePredicateVector)
      continue;

    ObjectsToAllocate.push_back(FI);
  }

  // Allocate all SVE locals and spills
  for (unsigned FI : ObjectsToAllocate)
    AllocateObject(FI);

  PPRStackTop = alignTo(PPRStackTop, Align(16U));
  ZPRStackTop = alignTo(ZPRStackTop, Align(16U));

  if (AssignOffsets == AssignObjectOffsets::Yes)
    AFI->setStackSizeSVE(SVEStack.ZPRStackSize, SVEStack.PPRStackSize);

  return SVEStack;
}

void AArch64FrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  assert(getStackGrowthDirection() == TargetFrameLowering::StackGrowsDown &&
         "Upwards growing stack unsupported");

  (void)determineSVEStackSizes(MF, AssignObjectOffsets::Yes);

  // If this function isn't doing Win64-style C++ EH, we don't need to do
  // anything.
  if (!MF.hasEHFunclets())
    return;

  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *AFI = MF.getInfo<AArch64FunctionInfo>();

  // Win64 C++ EH needs to allocate space for the catch objects in the fixed
  // object area right next to the UnwindHelp object.
  WinEHFuncInfo &EHInfo = *MF.getWinEHFuncInfo();
  int64_t CurrentOffset =
      AFI->getVarArgsGPRSize() + AFI->getTailCallReservedStack();
  for (WinEHTryBlockMapEntry &TBME : EHInfo.TryBlockMap) {
    for (WinEHHandlerType &H : TBME.HandlerArray) {
      int FrameIndex = H.CatchObj.FrameIndex;
      if ((FrameIndex != INT_MAX) && MFI.getObjectOffset(FrameIndex) == 0) {
        CurrentOffset =
            alignTo(CurrentOffset, MFI.getObjectAlign(FrameIndex).value());
        CurrentOffset += MFI.getObjectSize(FrameIndex);
        MFI.setObjectOffset(FrameIndex, -CurrentOffset);
      }
    }
  }

  // Create an UnwindHelp object.
  // The UnwindHelp object is allocated at the start of the fixed object area
  int64_t UnwindHelpOffset = alignTo(CurrentOffset + 8, Align(16));
  assert(UnwindHelpOffset == getFixedObjectSize(MF, AFI, /*IsWin64*/ true,
                                                /*IsFunclet*/ false) &&
         "UnwindHelpOffset must be at the start of the fixed object area");
  int UnwindHelpFI = MFI.CreateFixedObject(/*Size*/ 8, -UnwindHelpOffset,
                                           /*IsImmutable=*/false);
  EHInfo.UnwindHelpFrameIdx = UnwindHelpFI;

  MachineBasicBlock &MBB = MF.front();
  auto MBBI = MBB.begin();
  while (MBBI != MBB.end() && MBBI->getFlag(MachineInstr::FrameSetup))
    ++MBBI;

  // We need to store -2 into the UnwindHelp object at the start of the
  // function.
  DebugLoc DL;
  RS->enterBasicBlockEnd(MBB);
  RS->backward(MBBI);
  Register DstReg = RS->FindUnusedReg(&AArch64::GPR64commonRegClass);
  assert(DstReg && "There must be a free register after frame setup");
  const AArch64InstrInfo &TII =
      *MF.getSubtarget<AArch64Subtarget>().getInstrInfo();
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
  // If not std::nullopt, move FrameReg to (FrameReg + FrameRegUpdate) at the
  // end.
  std::optional<int64_t> FrameRegUpdate;
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
      BaseRegOffsetBytes + (Size - Size % 32) > kMaxOffset ||
      // BaseReg can be FP, which is not necessarily aligned to 16-bytes. In
      // that case, BaseRegOffsetBytes will not be aligned to 16 bytes, which
      // is required for the offset of ST2G.
      BaseRegOffsetBytes % 16 != 0) {
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
            ? (ZeroData ? AArch64::STZGi : AArch64::STGi)
            : (ZeroData ? AArch64::STZ2Gi : AArch64::ST2Gi);
    assert(BaseRegOffsetBytes % 16 == 0);
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
  LLVM_DEBUG(dbgs() << "TagStoreEdit::emitLoop: LoopSize=" << LoopSize
                    << ", Size=" << Size
                    << ", ExtraBaseRegUpdate=" << ExtraBaseRegUpdate
                    << ", FrameRegUpdate=" << FrameRegUpdate
                    << ", FrameRegOffset.getFixed()="
                    << FrameRegOffset.getFixed() << "\n");
  if (LoopSize < Size) {
    assert(FrameRegUpdate);
    assert(Size - LoopSize == 16);
    // Tag 16 more bytes at BaseReg and update BaseReg.
    int64_t STGOffset = ExtraBaseRegUpdate + 16;
    assert(STGOffset % 16 == 0 && STGOffset >= -4096 && STGOffset <= 4080 &&
           "STG immediate out of range");
    BuildMI(*MBB, InsertI, DL,
            TII->get(ZeroData ? AArch64::STZGPostIndex : AArch64::STGPostIndex))
        .addDef(BaseReg)
        .addReg(BaseReg)
        .addReg(BaseReg)
        .addImm(STGOffset / 16)
        .setMemRefs(CombinedMemRefs)
        .setMIFlags(FrameRegUpdateFlags);
  } else if (ExtraBaseRegUpdate) {
    // Update BaseReg.
    int64_t AddSubOffset = std::abs(ExtraBaseRegUpdate);
    assert(AddSubOffset <= 4095 && "ADD/SUB immediate out of range");
    BuildMI(
        *MBB, InsertI, DL,
        TII->get(ExtraBaseRegUpdate > 0 ? AArch64::ADDXri : AArch64::SUBXri))
        .addDef(BaseReg)
        .addReg(BaseReg)
        .addImm(AddSubOffset)
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
    int64_t PostOffset = Offset - Size;
    // TagStoreEdit::emitLoop might emit either an ADD/SUB after the loop, or
    // an STGPostIndex which does the last 16 bytes of tag write. Which one is
    // chosen depends on the alignment of the loop size, but the difference
    // between the valid ranges for the two instructions is small, so we
    // conservatively assume that it could be either case here.
    //
    // Max offset of STGPostIndex, minus the 16 byte tag write folded into that
    // instruction.
    const int64_t kMaxOffset = 4080 - 16;
    // Max offset of SUBXri.
    const int64_t kMinOffset = -4095;
    if (PostOffset <= kMaxOffset && PostOffset >= kMinOffset &&
        PostOffset % 16 == 0) {
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
      *MF, FirstTagStore.Offset, false /*isFixed*/,
      TargetStackID::Default /*StackID*/, Reg,
      /*PreferFP=*/false, /*ForSimm=*/true);
  FrameReg = Reg;
  FrameRegUpdate = std::nullopt;

  mergeMemRefs(TagStores, CombinedMemRefs);

  LLVM_DEBUG({
    dbgs() << "Replacing adjacent STG instructions:\n";
    for (const auto &Instr : TagStores) {
      dbgs() << "  " << *Instr.MI;
    }
  });

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
  ZeroData = (Opcode == AArch64::STZGloop || Opcode == AArch64::STZGi ||
              Opcode == AArch64::STZ2Gi);

  if (Opcode == AArch64::STGloop || Opcode == AArch64::STZGloop) {
    if (!MI.getOperand(0).isDead() || !MI.getOperand(1).isDead())
      return false;
    if (!MI.getOperand(2).isImm() || !MI.getOperand(3).isFI())
      return false;
    Offset = MFI.getObjectOffset(MI.getOperand(3).getIndex());
    Size = MI.getOperand(2).getImm();
    return true;
  }

  if (Opcode == AArch64::STGi || Opcode == AArch64::STZGi)
    Size = 16;
  else if (Opcode == AArch64::ST2Gi || Opcode == AArch64::STZ2Gi)
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
    if (MI.mayLoadOrStore() || MI.hasUnmodeledSideEffects() || MI.isCall())
      break;
  }

  // New code will be inserted after the last tagging instruction we've found.
  MachineBasicBlock::iterator InsertI = Instrs.back().MI;

  // All the gathered stack tag instructions are merged and placed after
  // last tag store in the list. The check should be made if the nzcv
  // flag is live at the point where we are trying to insert. Otherwise
  // the nzcv flag might get clobbered if any stg loops are present.

  // FIXME : This approach of bailing out from merge is conservative in
  // some ways like even if stg loops are not present after merge the
  // insert list, this liveness check is done (which is not needed).
  LivePhysRegs LiveRegs(*(MBB->getParent()->getSubtarget().getRegisterInfo()));
  LiveRegs.addLiveOuts(*MBB);
  for (auto I = MBB->rbegin();; ++I) {
    MachineInstr &MI = *I;
    if (MI == InsertI)
      break;
    LiveRegs.stepBackward(*I);
  }
  InsertI++;
  if (LiveRegs.contains(AArch64::NZCV))
    return InsertI;

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
  // sequences for them when possible.
  TagStoreEdit TSE(MBB, FirstZeroData);
  std::optional<int64_t> EndOffset;
  for (auto &Instr : Instrs) {
    if (EndOffset && *EndOffset != Instr.Offset) {
      // Found a gap.
      TSE.emitCode(InsertI, TFI, /*TryMergeSPUpdate = */ false);
      TSE.clear();
    }

    TSE.addInstruction(Instr);
    EndOffset = Instr.Offset + Instr.Size;
  }

  const MachineFunction *MF = MBB->getParent();
  // Multiple FP/SP updates in a loop cannot be described by CFI instructions.
  TSE.emitCode(
      InsertI, TFI, /*TryMergeSPUpdate = */
      !MF->getInfo<AArch64FunctionInfo>()->needsAsyncDwarfUnwindInfo(*MF));

  return InsertI;
}
} // namespace

void AArch64FrameLowering::processFunctionBeforeFrameIndicesReplaced(
    MachineFunction &MF, RegScavenger *RS = nullptr) const {
  for (auto &BB : MF)
    for (MachineBasicBlock::iterator II = BB.begin(); II != BB.end();) {
      if (StackTaggingMergeSetTag)
        II = tryMergeAdjacentSTG(II, this, RS);
    }

  // By the time this method is called, most of the prologue/epilogue code is
  // already emitted, whether its location was affected by the shrink-wrapping
  // optimization or not.
  if (!MF.getFunction().hasFnAttribute(Attribute::Naked) &&
      shouldSignReturnAddressEverywhere(MF))
    emitPacRetPlusLeafHardening(MF);
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
      MF.getInfo<AArch64FunctionInfo>()->hasSVEStackSize() ||
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

  // Used to distinguish between FP and GPR accesses. The values are decided so
  // that they sort FPR < Hazard < GPR and they can be or'd together.
  unsigned Accesses = 0;
  enum { AccessFPR = 1, AccessHazard = 2, AccessGPR = 4 };
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
  // If we want to include a stack hazard region, order FPR accesses < the
  // hazard object < GPRs accesses in order to create a separation between the
  // two. For the Accesses field 1 = FPR, 2 = Hazard Object, 4 = GPR.
  //
  // Otherwise the "first" object goes first (closest to SP), followed by the
  // members of the "first" group.
  //
  // The rest are sorted by the group index to keep the groups together.
  // Higher numbered groups are more likely to be around longer (i.e. untagged
  // in the function epilogue and not at some earlier point). Place them closer
  // to SP.
  //
  // If all else equal, sort by the object index to keep the objects in the
  // original order.
  return std::make_tuple(!A.IsValid, A.Accesses, A.ObjectFirst, A.GroupFirst,
                         A.GroupIndex, A.ObjectIndex) <
         std::make_tuple(!B.IsValid, B.Accesses, B.ObjectFirst, B.GroupFirst,
                         B.GroupIndex, B.ObjectIndex);
}
} // namespace

void AArch64FrameLowering::orderFrameObjects(
    const MachineFunction &MF, SmallVectorImpl<int> &ObjectsToAllocate) const {
  const AArch64FunctionInfo &AFI = *MF.getInfo<AArch64FunctionInfo>();

  if ((!OrderFrameObjects && !AFI.hasSplitSVEObjects()) ||
      ObjectsToAllocate.empty())
    return;

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  std::vector<FrameObject> FrameObjects(MFI.getObjectIndexEnd());
  for (auto &Obj : ObjectsToAllocate) {
    FrameObjects[Obj].IsValid = true;
    FrameObjects[Obj].ObjectIndex = Obj;
  }

  // Identify FPR vs GPR slots for hazards, and stack slots that are tagged at
  // the same time.
  GroupBuilder GB(FrameObjects);
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.isDebugInstr())
        continue;

      if (AFI.hasStackHazardSlotIndex()) {
        std::optional<int> FI = getLdStFrameID(MI, MFI);
        if (FI && *FI >= 0 && *FI < (int)FrameObjects.size()) {
          if (MFI.getStackID(*FI) == TargetStackID::ScalableVector ||
              AArch64InstrInfo::isFpOrNEON(MI))
            FrameObjects[*FI].Accesses |= FrameObject::AccessFPR;
          else
            FrameObjects[*FI].Accesses |= FrameObject::AccessGPR;
        }
      }

      int OpIndex;
      switch (MI.getOpcode()) {
      case AArch64::STGloop:
      case AArch64::STZGloop:
        OpIndex = 3;
        break;
      case AArch64::STGi:
      case AArch64::STZGi:
      case AArch64::ST2Gi:
      case AArch64::STZ2Gi:
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

  if (AFI.hasStackHazardSlotIndex()) {
    FrameObjects[AFI.getStackHazardSlotIndex()].Accesses =
        FrameObject::AccessHazard;
    // If a stack object is unknown or both GPR and FPR, sort it into GPR.
    for (auto &Obj : FrameObjects)
      if (!Obj.Accesses ||
          Obj.Accesses == (FrameObject::AccessGPR | FrameObject::AccessFPR))
        Obj.Accesses = FrameObject::AccessGPR;
  }

  // If the function's tagged base pointer is pinned to a stack slot, we want to
  // put that slot first when possible. This will likely place it at SP + 0,
  // and save one instruction when generating the base pointer because IRG does
  // not allow an immediate offset.
  std::optional<int> TBPI = AFI.getTaggedBasePointerIndex();
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

  LLVM_DEBUG({
    dbgs() << "Final frame order:\n";
    for (auto &Obj : FrameObjects) {
      if (!Obj.IsValid)
        break;
      dbgs() << "  " << Obj.ObjectIndex << ": group " << Obj.GroupIndex;
      if (Obj.ObjectFirst)
        dbgs() << ", first";
      if (Obj.GroupFirst)
        dbgs() << ", group-first";
      dbgs() << "\n";
    }
  });
}

/// Emit a loop to decrement SP until it is equal to TargetReg, with probes at
/// least every ProbeSize bytes. Returns an iterator of the first instruction
/// after the loop. The difference between SP and TargetReg must be an exact
/// multiple of ProbeSize.
MachineBasicBlock::iterator
AArch64FrameLowering::inlineStackProbeLoopExactMultiple(
    MachineBasicBlock::iterator MBBI, int64_t ProbeSize,
    Register TargetReg) const {
  MachineBasicBlock &MBB = *MBBI->getParent();
  MachineFunction &MF = *MBB.getParent();
  const AArch64InstrInfo *TII =
      MF.getSubtarget<AArch64Subtarget>().getInstrInfo();
  DebugLoc DL = MBB.findDebugLoc(MBBI);

  MachineFunction::iterator MBBInsertPoint = std::next(MBB.getIterator());
  MachineBasicBlock *LoopMBB = MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MF.insert(MBBInsertPoint, LoopMBB);
  MachineBasicBlock *ExitMBB = MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MF.insert(MBBInsertPoint, ExitMBB);

  // SUB SP, SP, #ProbeSize (or equivalent if ProbeSize is not encodable
  // in SUB).
  emitFrameOffset(*LoopMBB, LoopMBB->end(), DL, AArch64::SP, AArch64::SP,
                  StackOffset::getFixed(-ProbeSize), TII,
                  MachineInstr::FrameSetup);
  // LDR XZR, [SP]
  BuildMI(*LoopMBB, LoopMBB->end(), DL, TII->get(AArch64::LDRXui))
      .addDef(AArch64::XZR)
      .addReg(AArch64::SP)
      .addImm(0)
      .addMemOperand(MF.getMachineMemOperand(
          MachinePointerInfo::getUnknownStack(MF),
          MachineMemOperand::MOLoad | MachineMemOperand::MOVolatile, 8,
          Align(8)))
      .setMIFlags(MachineInstr::FrameSetup);
  // CMP SP, TargetReg
  BuildMI(*LoopMBB, LoopMBB->end(), DL, TII->get(AArch64::SUBSXrx64),
          AArch64::XZR)
      .addReg(AArch64::SP)
      .addReg(TargetReg)
      .addImm(AArch64_AM::getArithExtendImm(AArch64_AM::UXTX, 0))
      .setMIFlags(MachineInstr::FrameSetup);
  // B.CC Loop
  BuildMI(*LoopMBB, LoopMBB->end(), DL, TII->get(AArch64::Bcc))
      .addImm(AArch64CC::NE)
      .addMBB(LoopMBB)
      .setMIFlags(MachineInstr::FrameSetup);

  LoopMBB->addSuccessor(ExitMBB);
  LoopMBB->addSuccessor(LoopMBB);
  // Synthesize the exit MBB.
  ExitMBB->splice(ExitMBB->end(), &MBB, MBBI, MBB.end());
  ExitMBB->transferSuccessorsAndUpdatePHIs(&MBB);
  MBB.addSuccessor(LoopMBB);
  // Update liveins.
  fullyRecomputeLiveIns({ExitMBB, LoopMBB});

  return ExitMBB->begin();
}

void AArch64FrameLowering::inlineStackProbeFixed(
    MachineBasicBlock::iterator MBBI, Register ScratchReg, int64_t FrameSize,
    StackOffset CFAOffset) const {
  MachineBasicBlock *MBB = MBBI->getParent();
  MachineFunction &MF = *MBB->getParent();
  const AArch64InstrInfo *TII =
      MF.getSubtarget<AArch64Subtarget>().getInstrInfo();
  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  bool EmitAsyncCFI = AFI->needsAsyncDwarfUnwindInfo(MF);
  bool HasFP = hasFP(MF);

  DebugLoc DL;
  int64_t ProbeSize = MF.getInfo<AArch64FunctionInfo>()->getStackProbeSize();
  int64_t NumBlocks = FrameSize / ProbeSize;
  int64_t ResidualSize = FrameSize % ProbeSize;

  LLVM_DEBUG(dbgs() << "Stack probing: total " << FrameSize << " bytes, "
                    << NumBlocks << " blocks of " << ProbeSize
                    << " bytes, plus " << ResidualSize << " bytes\n");

  // Decrement SP by NumBlock * ProbeSize bytes, with either unrolled or
  // ordinary loop.
  if (NumBlocks <= AArch64::StackProbeMaxLoopUnroll) {
    for (int i = 0; i < NumBlocks; ++i) {
      // SUB SP, SP, #ProbeSize (or equivalent if ProbeSize is not
      // encodable in a SUB).
      emitFrameOffset(*MBB, MBBI, DL, AArch64::SP, AArch64::SP,
                      StackOffset::getFixed(-ProbeSize), TII,
                      MachineInstr::FrameSetup, false, false, nullptr,
                      EmitAsyncCFI && !HasFP, CFAOffset);
      CFAOffset += StackOffset::getFixed(ProbeSize);
      // LDR XZR, [SP]
      BuildMI(*MBB, MBBI, DL, TII->get(AArch64::LDRXui))
          .addDef(AArch64::XZR)
          .addReg(AArch64::SP)
          .addImm(0)
          .addMemOperand(MF.getMachineMemOperand(
              MachinePointerInfo::getUnknownStack(MF),
              MachineMemOperand::MOLoad | MachineMemOperand::MOVolatile, 8,
              Align(8)))
          .setMIFlags(MachineInstr::FrameSetup);
    }
  } else if (NumBlocks != 0) {
    // SUB ScratchReg, SP, #FrameSize (or equivalent if FrameSize is not
    // encodable in ADD). ScrathReg may temporarily become the CFA register.
    emitFrameOffset(*MBB, MBBI, DL, ScratchReg, AArch64::SP,
                    StackOffset::getFixed(-ProbeSize * NumBlocks), TII,
                    MachineInstr::FrameSetup, false, false, nullptr,
                    EmitAsyncCFI && !HasFP, CFAOffset);
    CFAOffset += StackOffset::getFixed(ProbeSize * NumBlocks);
    MBBI = inlineStackProbeLoopExactMultiple(MBBI, ProbeSize, ScratchReg);
    MBB = MBBI->getParent();
    if (EmitAsyncCFI && !HasFP) {
      // Set the CFA register back to SP.
      CFIInstBuilder(*MBB, MBBI, MachineInstr::FrameSetup)
          .buildDefCFARegister(AArch64::SP);
    }
  }

  if (ResidualSize != 0) {
    // SUB SP, SP, #ResidualSize (or equivalent if ResidualSize is not encodable
    // in SUB).
    emitFrameOffset(*MBB, MBBI, DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(-ResidualSize), TII,
                    MachineInstr::FrameSetup, false, false, nullptr,
                    EmitAsyncCFI && !HasFP, CFAOffset);
    if (ResidualSize > AArch64::StackProbeMaxUnprobedStack) {
      // LDR XZR, [SP]
      BuildMI(*MBB, MBBI, DL, TII->get(AArch64::LDRXui))
          .addDef(AArch64::XZR)
          .addReg(AArch64::SP)
          .addImm(0)
          .addMemOperand(MF.getMachineMemOperand(
              MachinePointerInfo::getUnknownStack(MF),
              MachineMemOperand::MOLoad | MachineMemOperand::MOVolatile, 8,
              Align(8)))
          .setMIFlags(MachineInstr::FrameSetup);
    }
  }
}

void AArch64FrameLowering::inlineStackProbe(MachineFunction &MF,
                                            MachineBasicBlock &MBB) const {
  // Get the instructions that need to be replaced. We emit at most two of
  // these. Remember them in order to avoid complications coming from the need
  // to traverse the block while potentially creating more blocks.
  SmallVector<MachineInstr *, 4> ToReplace;
  for (MachineInstr &MI : MBB)
    if (MI.getOpcode() == AArch64::PROBED_STACKALLOC ||
        MI.getOpcode() == AArch64::PROBED_STACKALLOC_VAR)
      ToReplace.push_back(&MI);

  for (MachineInstr *MI : ToReplace) {
    if (MI->getOpcode() == AArch64::PROBED_STACKALLOC) {
      Register ScratchReg = MI->getOperand(0).getReg();
      int64_t FrameSize = MI->getOperand(1).getImm();
      StackOffset CFAOffset = StackOffset::get(MI->getOperand(2).getImm(),
                                               MI->getOperand(3).getImm());
      inlineStackProbeFixed(MI->getIterator(), ScratchReg, FrameSize,
                            CFAOffset);
    } else {
      assert(MI->getOpcode() == AArch64::PROBED_STACKALLOC_VAR &&
             "Stack probe pseudo-instruction expected");
      const AArch64InstrInfo *TII =
          MI->getMF()->getSubtarget<AArch64Subtarget>().getInstrInfo();
      Register TargetReg = MI->getOperand(0).getReg();
      (void)TII->probedStackAlloc(MI->getIterator(), TargetReg, true);
    }
    MI->eraseFromParent();
  }
}

struct StackAccess {
  enum AccessType {
    NotAccessed = 0, // Stack object not accessed by load/store instructions.
    GPR = 1 << 0,    // A general purpose register.
    PPR = 1 << 1,    // A predicate register.
    FPR = 1 << 2,    // A floating point/Neon/SVE register.
  };

  int Idx;
  StackOffset Offset;
  int64_t Size;
  unsigned AccessTypes;

  StackAccess() : Idx(0), Offset(), Size(0), AccessTypes(NotAccessed) {}

  bool operator<(const StackAccess &Rhs) const {
    return std::make_tuple(start(), Idx) <
           std::make_tuple(Rhs.start(), Rhs.Idx);
  }

  bool isCPU() const {
    // Predicate register load and store instructions execute on the CPU.
    return AccessTypes & (AccessType::GPR | AccessType::PPR);
  }
  bool isSME() const { return AccessTypes & AccessType::FPR; }
  bool isMixed() const { return isCPU() && isSME(); }

  int64_t start() const { return Offset.getFixed() + Offset.getScalable(); }
  int64_t end() const { return start() + Size; }

  std::string getTypeString() const {
    switch (AccessTypes) {
    case AccessType::FPR:
      return "FPR";
    case AccessType::PPR:
      return "PPR";
    case AccessType::GPR:
      return "GPR";
    case AccessType::NotAccessed:
      return "NA";
    default:
      return "Mixed";
    }
  }

  void print(raw_ostream &OS) const {
    OS << getTypeString() << " stack object at [SP"
       << (Offset.getFixed() < 0 ? "" : "+") << Offset.getFixed();
    if (Offset.getScalable())
      OS << (Offset.getScalable() < 0 ? "" : "+") << Offset.getScalable()
         << " * vscale";
    OS << "]";
  }
};

static inline raw_ostream &operator<<(raw_ostream &OS, const StackAccess &SA) {
  SA.print(OS);
  return OS;
}

void AArch64FrameLowering::emitRemarks(
    const MachineFunction &MF, MachineOptimizationRemarkEmitter *ORE) const {

  auto *AFI = MF.getInfo<AArch64FunctionInfo>();
  if (AFI->getSMEFnAttrs().hasNonStreamingInterfaceAndBody())
    return;

  unsigned StackHazardSize = getStackHazardSize(MF);
  const uint64_t HazardSize =
      (StackHazardSize) ? StackHazardSize : StackHazardRemarkSize;

  if (HazardSize == 0)
    return;

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  // Bail if function has no stack objects.
  if (!MFI.hasStackObjects())
    return;

  std::vector<StackAccess> StackAccesses(MFI.getNumObjects());

  size_t NumFPLdSt = 0;
  size_t NumNonFPLdSt = 0;

  // Collect stack accesses via Load/Store instructions.
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (!MI.mayLoadOrStore() || MI.getNumMemOperands() < 1)
        continue;
      for (MachineMemOperand *MMO : MI.memoperands()) {
        std::optional<int> FI = getMMOFrameID(MMO, MFI);
        if (FI && !MFI.isDeadObjectIndex(*FI)) {
          int FrameIdx = *FI;

          size_t ArrIdx = FrameIdx + MFI.getNumFixedObjects();
          if (StackAccesses[ArrIdx].AccessTypes == StackAccess::NotAccessed) {
            StackAccesses[ArrIdx].Idx = FrameIdx;
            StackAccesses[ArrIdx].Offset =
                getFrameIndexReferenceFromSP(MF, FrameIdx);
            StackAccesses[ArrIdx].Size = MFI.getObjectSize(FrameIdx);
          }

          unsigned RegTy = StackAccess::AccessType::GPR;
          if (MFI.hasScalableStackID(FrameIdx))
            RegTy = isPPRAccess(MI) ? StackAccess::PPR : StackAccess::FPR;
          else if (AArch64InstrInfo::isFpOrNEON(MI))
            RegTy = StackAccess::FPR;

          StackAccesses[ArrIdx].AccessTypes |= RegTy;

          if (RegTy == StackAccess::FPR)
            ++NumFPLdSt;
          else
            ++NumNonFPLdSt;
        }
      }
    }
  }

  if (NumFPLdSt == 0 || NumNonFPLdSt == 0)
    return;

  llvm::sort(StackAccesses);
  llvm::erase_if(StackAccesses, [](const StackAccess &S) {
    return S.AccessTypes == StackAccess::NotAccessed;
  });

  SmallVector<const StackAccess *> MixedObjects;
  SmallVector<std::pair<const StackAccess *, const StackAccess *>> HazardPairs;

  if (StackAccesses.front().isMixed())
    MixedObjects.push_back(&StackAccesses.front());

  for (auto It = StackAccesses.begin(), End = std::prev(StackAccesses.end());
       It != End; ++It) {
    const auto &First = *It;
    const auto &Second = *(It + 1);

    if (Second.isMixed())
      MixedObjects.push_back(&Second);

    if ((First.isSME() && Second.isCPU()) ||
        (First.isCPU() && Second.isSME())) {
      uint64_t Distance = static_cast<uint64_t>(Second.start() - First.end());
      if (Distance < HazardSize)
        HazardPairs.emplace_back(&First, &Second);
    }
  }

  auto EmitRemark = [&](llvm::StringRef Str) {
    ORE->emit([&]() {
      auto R = MachineOptimizationRemarkAnalysis(
          "sme", "StackHazard", MF.getFunction().getSubprogram(), &MF.front());
      return R << formatv("stack hazard in '{0}': ", MF.getName()).str() << Str;
    });
  };

  for (const auto &P : HazardPairs)
    EmitRemark(formatv("{0} is too close to {1}", *P.first, *P.second).str());

  for (const auto *Obj : MixedObjects)
    EmitRemark(
        formatv("{0} accessed by both GP and FP instructions", *Obj).str());
}
