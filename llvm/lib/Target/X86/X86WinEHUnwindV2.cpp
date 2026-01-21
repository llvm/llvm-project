//===-- X86WinEHUnwindV2.cpp - Win x64 Unwind v2 ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implements the analysis required to detect if a function can use Unwind v2
/// information, and emits the neccesary pseudo instructions used by MC to
/// generate the unwind info.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "X86.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Module.h"

using namespace llvm;

#define DEBUG_TYPE "x86-wineh-unwindv2"

STATISTIC(MeetsUnwindV2Criteria,
          "Number of functions that meet Unwind v2 criteria");
STATISTIC(FailsUnwindV2Criteria,
          "Number of functions that fail Unwind v2 criteria");

static cl::opt<unsigned>
    UnwindCodeThreshold("x86-wineh-unwindv2-unwind-codes-threshold", cl::Hidden,
                        cl::desc("Maximum number of unwind codes before "
                                 "splitting into a new unwind info."),
                        cl::init(UINT8_MAX));

static cl::opt<unsigned>
    ForceMode("x86-wineh-unwindv2-force-mode", cl::Hidden,
              cl::desc("Overwrites the Unwind v2 mode for testing purposes."));

// This threshold is for the *approximate* number of instructions, see the
// comment in runAnalysisOnFuncOrFunclet for more details.
static cl::opt<unsigned> InstructionCountThreshold(
    "x86-wineh-unwindv2-instruction-count-threshold", cl::Hidden,
    cl::desc("Maximum number of (approximate) instructions before splitting "
             "into a new unwind info."),
    cl::init(600));

namespace {

struct EpilogInfo {
  MachineInstr *UnwindV2StartLocation;
  unsigned ApproximateInstructionPosition;
};

struct FrameInfo {
  unsigned ApproximatePrologCodeCount;
  unsigned ApproximateInstructionCount;
  SmallVector<EpilogInfo> EpilogInfos;
};

class X86WinEHUnwindV2 : public MachineFunctionPass {
public:
  static char ID;

  X86WinEHUnwindV2() : MachineFunctionPass(ID) {
    initializeX86WinEHUnwindV2Pass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "WinEH Unwind V2"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  /// Rejects the current function due to an internal error within LLVM.
  static std::nullopt_t rejectCurrentFunctionInternalError(
      const MachineFunction &MF, WinX64EHUnwindV2Mode Mode, StringRef Reason);

  // Continues running the analysis on the given function or funclet.
  static std::optional<FrameInfo>
  runAnalysisOnFuncOrFunclet(MachineFunction &MF,
                             MachineFunction::iterator &Iter,
                             WinX64EHUnwindV2Mode Mode);
};

enum class FunctionState {
  InProlog,
  HasProlog,
  InEpilog,
  FinishedEpilog,
};

} // end anonymous namespace

char X86WinEHUnwindV2::ID = 0;

INITIALIZE_PASS(X86WinEHUnwindV2, "x86-wineh-unwindv2",
                "Analyze and emit instructions for Win64 Unwind v2", false,
                false)

FunctionPass *llvm::createX86WinEHUnwindV2Pass() {
  return new X86WinEHUnwindV2();
}

DebugLoc findDebugLoc(const MachineBasicBlock &MBB) {
  for (const MachineInstr &MI : MBB)
    if (MI.getDebugLoc())
      return MI.getDebugLoc();

  return DebugLoc::getUnknown();
}

std::optional<FrameInfo>
X86WinEHUnwindV2::runAnalysisOnFuncOrFunclet(MachineFunction &MF,
                                             MachineFunction::iterator &Iter,
                                             WinX64EHUnwindV2Mode Mode) {
  const TargetFrameLowering &TFL = *MF.getSubtarget().getFrameLowering();

  // Current state of processing the function. We'll assume that all functions
  // start with a prolog.
  FunctionState State = FunctionState::InProlog;

  // Prolog information.
  SmallVector<int64_t> PushedRegs;
  bool HasStackAlloc = false;
  bool HasSetFrame = false;
  unsigned ApproximatePrologCodeCount = 0;

  SmallVector<EpilogInfo> EpilogInfos;

  // Unwind v2 requires that the epilog is no more than 4Kb away from the last
  // instruction that the current unwind info covers. If we believe that we are
  // going over that limit then we need to split the unwind info. Ideally we'd
  // do this at the point where we actually know how far away we are from the
  // last instruction, but that's not possible here and splitting unwind infos
  // in MC would be difficult. However, the cost of splitting an unwind info is
  // fairly cheap (in the other of bytes in the xdata section), so we can
  // instead use a heuristic based on the number of MachineInstrs to decide when
  // to split unwind infos, and allow users to tune the threshold if needed.
  // This is not a perfect solution, but 1) it is cheap to calculate, 2) allows
  // the common case for small functions or large functions with multiple
  // returns at the end to have a single unwind info, and 3) allows unwind v2 to
  // be used in large functions (that would otherwise be rejected) for a small
  // binary size cost.
  unsigned ApproximateInstructionCount = 0;

  for (; Iter != MF.end(); ++Iter) {
    MachineBasicBlock &MBB = *Iter;

    // If we're already been processing a function, then come across a funclet
    // then break since the funclet will get a fresh frame info.
    if (MBB.isEHFuncletEntry() && State != FunctionState::InProlog)
      break;

    // Current epilog information. We assume that epilogs cannot cross basic
    // block boundaries.
    unsigned PoppedRegCount = 0;
    bool HasStackDealloc = false;
    bool HasSetFrameBack = false;
    MachineInstr *UnwindV2StartLocation = nullptr;

    for (MachineInstr &MI : MBB) {
      // This is an *approximation* of the number of instructions that will be
      // emitted. It is not the actual number of instructions, but that doesn't
      // matter: see the comment at the declaration of
      // ApproximateInstructionCount.
      if (!MI.isPseudo() && !MI.isMetaInstruction())
        ApproximateInstructionCount++;

      switch (MI.getOpcode()) {
      //
      // Prolog handling.
      //
      case X86::SEH_PushReg:
        if (State != FunctionState::InProlog)
          llvm_unreachable("SEH_PushReg outside of prolog");
        ApproximatePrologCodeCount++;
        PushedRegs.push_back(MI.getOperand(0).getImm());
        break;

      case X86::SEH_StackAlloc:
        if (State != FunctionState::InProlog)
          llvm_unreachable("SEH_StackAlloc outside of prolog");
        // Assume a large alloc...
        ApproximatePrologCodeCount += 3;
        HasStackAlloc = true;
        break;

      case X86::SEH_SetFrame:
        if (State != FunctionState::InProlog)
          llvm_unreachable("SEH_SetFrame outside of prolog");
        ApproximatePrologCodeCount++;
        HasSetFrame = true;
        break;

      case X86::SEH_SaveReg:
      case X86::SEH_SaveXMM:
        if (State != FunctionState::InProlog)
          llvm_unreachable("SEH_SaveXMM or SEH_SaveReg outside of prolog");
        // Assume a big reg...
        ApproximatePrologCodeCount += 3;
        break;

      case X86::SEH_PushFrame:
        if (State != FunctionState::InProlog)
          llvm_unreachable("SEH_PushFrame outside of prolog");
        ApproximatePrologCodeCount++;
        break;

      case X86::SEH_EndPrologue:
        if (State != FunctionState::InProlog)
          llvm_unreachable("SEH_EndPrologue outside of prolog");
        State = FunctionState::HasProlog;
        break;

      //
      // Epilog handling.
      //
      case X86::SEH_BeginEpilogue:
        if (State != FunctionState::HasProlog)
          llvm_unreachable("SEH_BeginEpilogue in prolog or another epilog");
        State = FunctionState::InEpilog;
        break;

      case X86::SEH_EndEpilogue:
        if (State != FunctionState::InEpilog)
          llvm_unreachable("SEH_EndEpilogue outside of epilog");
        if (HasStackAlloc != HasStackDealloc)
          return rejectCurrentFunctionInternalError(
              MF, Mode,
              "The prolog made a stack allocation, "
              "but the epilog did not deallocate it");
        if (PoppedRegCount != PushedRegs.size())
          return rejectCurrentFunctionInternalError(
              MF, Mode,
              "The prolog pushed more registers than "
              "the epilog popped");

        // If we didn't find the start location, then use the end of the
        // epilog.
        if (!UnwindV2StartLocation)
          UnwindV2StartLocation = &MI;
        EpilogInfos.push_back(
            {UnwindV2StartLocation, ApproximateInstructionCount});
        State = FunctionState::FinishedEpilog;
        break;

      case X86::MOV64rr:
        if (State == FunctionState::InEpilog) {
          // If the prolog contains a stack allocation, then the first
          // instruction in the epilog must be to adjust the stack pointer.
          if (!HasSetFrame)
            return rejectCurrentFunctionInternalError(
                MF, Mode,
                "The epilog is setting frame back, but prolog did not set it");
          if (PoppedRegCount > 0)
            return rejectCurrentFunctionInternalError(
                MF, Mode,
                "The epilog is setting the frame back after popping "
                "registers");
          if (HasStackDealloc)
            return rejectCurrentFunctionInternalError(
                MF, Mode,
                "Cannot set the frame back after the stack "
                "allocation has been deallocated");
          HasSetFrameBack = true;
        } else if (State == FunctionState::FinishedEpilog)
          return rejectCurrentFunctionInternalError(
              MF, Mode, "Unexpected mov instruction after the epilog");
        break;

      case X86::LEA64r:
      case X86::ADD64ri32:
        if (State == FunctionState::InEpilog) {
          // If the prolog contains a stack allocation, then the first
          // instruction in the epilog must be to adjust the stack pointer.
          if (!HasStackAlloc)
            return rejectCurrentFunctionInternalError(
                MF, Mode,
                "The epilog is deallocating a stack "
                "allocation, but the prolog did "
                "not allocate one");
          if (PoppedRegCount > 0)
            return rejectCurrentFunctionInternalError(
                MF, Mode,
                "The epilog is deallocating a stack allocation after popping "
                "registers");

          HasStackDealloc = true;
        } else if (State == FunctionState::FinishedEpilog)
          return rejectCurrentFunctionInternalError(
              MF, Mode, "Unexpected lea or add instruction after the epilog");
        break;

      case X86::POP64r:
        if (State == FunctionState::InEpilog) {
          Register Reg = MI.getOperand(0).getReg();
          if (HasStackAlloc && (PoppedRegCount == 0) &&
              !llvm::is_contained(PushedRegs, Reg)) {
            // If this is a pop that doesn't correspond to the set of pushed
            // registers, then assume it was used to adjust the stack pointer.
            HasStackDealloc = true;
          } else {
            // Special case: no explicit stack dealloc is required if SetFrame
            // was used and the function has a frame pointer.
            if (PoppedRegCount == 0 && HasStackAlloc && !HasStackDealloc &&
                HasSetFrameBack && TFL.hasFP(MF))
              HasStackDealloc = true;

            // After the stack pointer has been adjusted, the epilog must
            // POP each register in reverse order of the PUSHes in the prolog.
            PoppedRegCount++;
            if (HasStackAlloc != HasStackDealloc)
              return rejectCurrentFunctionInternalError(
                  MF, Mode,
                  "Cannot pop registers before the stack "
                  "allocation has been deallocated");
            if (PoppedRegCount > PushedRegs.size())
              return rejectCurrentFunctionInternalError(
                  MF, Mode,
                  "The epilog is popping more registers than the prolog "
                  "pushed");
            if (PushedRegs[PushedRegs.size() - PoppedRegCount] != Reg.id())
              return rejectCurrentFunctionInternalError(
                  MF, Mode,
                  "The epilog is popping a registers in "
                  "a different order than the "
                  "prolog pushed them");

            // Unwind v2 records the size of the epilog not from where we place
            // SEH_BeginEpilogue (as that contains the instruction to adjust the
            // stack pointer) but from the first POP instruction (if there is
            // one).
            if (!UnwindV2StartLocation) {
              assert(PoppedRegCount == 1);
              UnwindV2StartLocation = &MI;
            }
          }
        } else if (State == FunctionState::FinishedEpilog)
          // Unexpected instruction after the epilog.
          return rejectCurrentFunctionInternalError(
              MF, Mode, "Registers are being popped after the epilog");
        break;

      default:
        if (MI.isTerminator()) {
          if (State == FunctionState::FinishedEpilog)
            // Found the terminator after the epilog, we're now ready for
            // another epilog.
            State = FunctionState::HasProlog;
          else if (State == FunctionState::InEpilog)
            llvm_unreachable("Terminator in the middle of the epilog");
        } else if (!MI.isDebugOrPseudoInstr()) {
          if ((State == FunctionState::FinishedEpilog) ||
              (State == FunctionState::InEpilog))
            // Unknown instruction in or after the epilog.
            return rejectCurrentFunctionInternalError(
                MF, Mode, "Unexpected instruction in or after the epilog");
        }
      }
    }
  }

  return FrameInfo{ApproximatePrologCodeCount, ApproximateInstructionCount,
                   EpilogInfos};
}

bool X86WinEHUnwindV2::runOnMachineFunction(MachineFunction &MF) {
  WinX64EHUnwindV2Mode Mode =
      ForceMode.getNumOccurrences()
          ? static_cast<WinX64EHUnwindV2Mode>(ForceMode.getValue())
          : MF.getFunction().getParent()->getWinX64EHUnwindV2Mode();

  if (Mode == WinX64EHUnwindV2Mode::Disabled)
    return false;

  // Requested changes.
  SmallVector<FrameInfo> FrameInfos;
  MachineFunction::iterator Iter = MF.begin();
  while (Iter != MF.end()) {
    auto FI = runAnalysisOnFuncOrFunclet(MF, Iter, Mode);
    if (!FI)
      return false;
    if (!FI->EpilogInfos.empty())
      FrameInfos.push_back(std::move(*FI));
  }

  if (FrameInfos.empty())
    return false;

  MeetsUnwindV2Criteria++;

  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  for (auto &FI : FrameInfos) {
    // Walk the list of epilogs backwards and add new SEH pseudo instructions:
    // * SEH_UnwindV2Start at the start of each epilog.
    // * If the current instruction is too far away from where the last unwind
    //   info ended OR there are too many unwind codes in the info, then add
    //   SEH_SplitChainedAtEndOfBlock to finish the current info.
    unsigned LastUnwindInfoEndPosition = FI.ApproximateInstructionCount;
    unsigned UnwindCodeCount = FI.ApproximatePrologCodeCount + 1;
    for (auto &Info : llvm::reverse(FI.EpilogInfos)) {
      MachineBasicBlock &MBB = *Info.UnwindV2StartLocation->getParent();
      const DebugLoc &DL = Info.UnwindV2StartLocation->getDebugLoc();
      BuildMI(MBB, Info.UnwindV2StartLocation, DL,
              TII->get(X86::SEH_UnwindV2Start));

      if ((LastUnwindInfoEndPosition - Info.ApproximateInstructionPosition >=
           InstructionCountThreshold) ||
          (UnwindCodeCount >= UnwindCodeThreshold)) {
        BuildMI(MBB, MBB.begin(), DL,
                TII->get(X86::SEH_SplitChainedAtEndOfBlock));
        LastUnwindInfoEndPosition = Info.ApproximateInstructionPosition;
        // Doesn't reset to 0, as the prolog unwind codes are now in this info.
        UnwindCodeCount = FI.ApproximatePrologCodeCount + 1;
      }

      UnwindCodeCount++;
    }
  }

  // Note that the function is using Unwind v2.
  MachineBasicBlock &FirstMBB = MF.front();
  BuildMI(FirstMBB, FirstMBB.front(), findDebugLoc(FirstMBB),
          TII->get(X86::SEH_UnwindVersion))
      .addImm(2);

  return true;
}

std::nullopt_t X86WinEHUnwindV2::rejectCurrentFunctionInternalError(
    const MachineFunction &MF, WinX64EHUnwindV2Mode Mode, StringRef Reason) {
  if (Mode == WinX64EHUnwindV2Mode::Required)
    reportFatalInternalError("Windows x64 Unwind v2 is required, but LLVM has "
                             "generated incompatible code in function '" +
                             MF.getName() + "': " + Reason);

  FailsUnwindV2Criteria++;
  return std::nullopt;
}
