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

static cl::opt<unsigned> MaximumUnwindCodes(
    "x86-wineh-unwindv2-max-unwind-codes", cl::Hidden,
    cl::desc("Maximum number of unwind codes permitted in each unwind info."),
    cl::init(UINT8_MAX));

static cl::opt<unsigned>
    ForceMode("x86-wineh-unwindv2-force-mode", cl::Hidden,
              cl::desc("Overwrites the Unwind v2 mode for testing purposes."));

namespace {

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
  static bool rejectCurrentFunctionInternalError(const MachineFunction &MF,
                                                 WinX64EHUnwindV2Mode Mode,
                                                 StringRef Reason);
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

bool X86WinEHUnwindV2::runOnMachineFunction(MachineFunction &MF) {
  WinX64EHUnwindV2Mode Mode =
      ForceMode.getNumOccurrences()
          ? static_cast<WinX64EHUnwindV2Mode>(ForceMode.getValue())
          : MF.getFunction().getParent()->getWinX64EHUnwindV2Mode();

  if (Mode == WinX64EHUnwindV2Mode::Disabled)
    return false;

  // Current state of processing the function. We'll assume that all functions
  // start with a prolog.
  FunctionState State = FunctionState::InProlog;

  // Prolog information.
  SmallVector<int64_t> PushedRegs;
  bool HasStackAlloc = false;
  unsigned ApproximatePrologCodeCount = 0;

  // Requested changes.
  SmallVector<MachineInstr *> UnwindV2StartLocations;

  for (MachineBasicBlock &MBB : MF) {
    // Current epilog information. We assume that epilogs cannot cross basic
    // block boundaries.
    unsigned PoppedRegCount = 0;
    bool HasStackDealloc = false;
    MachineInstr *UnwindV2StartLocation = nullptr;

    for (MachineInstr &MI : MBB) {
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
      case X86::SEH_SetFrame:
        if (State != FunctionState::InProlog)
          llvm_unreachable("SEH_StackAlloc or SEH_SetFrame outside of prolog");
        // Assume a large alloc...
        ApproximatePrologCodeCount +=
            (MI.getOpcode() == X86::SEH_StackAlloc) ? 3 : 1;
        HasStackAlloc = true;
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
        UnwindV2StartLocations.push_back(UnwindV2StartLocation);
        State = FunctionState::FinishedEpilog;
        break;

      case X86::LEA64r:
      case X86::MOV64rr:
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
              MF, Mode,
              "Unexpected lea, mov or add instruction after the epilog");
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

  if (UnwindV2StartLocations.empty()) {
    assert(State == FunctionState::InProlog &&
           "If there are no epilogs, then there should be no prolog");
    return false;
  }

  MachineBasicBlock &FirstMBB = MF.front();
  // Assume +1 for the "header" UOP_Epilog that contains the epilog size, and
  // that we won't be able to use the "last epilog at the end of function"
  // optimization.
  if (ApproximatePrologCodeCount + UnwindV2StartLocations.size() + 1 >
      static_cast<unsigned>(MaximumUnwindCodes)) {
    if (Mode == WinX64EHUnwindV2Mode::Required)
      MF.getFunction().getContext().diagnose(DiagnosticInfoGenericWithLoc(
          "Windows x64 Unwind v2 is required, but the function '" +
              MF.getName() +
              "' has too many unwind codes. Try splitting the function or "
              "reducing the number of places where it exits early with a tail "
              "call.",
          MF.getFunction(), findDebugLoc(FirstMBB)));

    FailsUnwindV2Criteria++;
    return false;
  }

  MeetsUnwindV2Criteria++;

  // Emit the pseudo instruction that marks the start of each epilog.
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  for (MachineInstr *MI : UnwindV2StartLocations) {
    BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
            TII->get(X86::SEH_UnwindV2Start));
  }
  // Note that the function is using Unwind v2.
  BuildMI(FirstMBB, FirstMBB.front(), findDebugLoc(FirstMBB),
          TII->get(X86::SEH_UnwindVersion))
      .addImm(2);

  return true;
}

bool X86WinEHUnwindV2::rejectCurrentFunctionInternalError(
    const MachineFunction &MF, WinX64EHUnwindV2Mode Mode, StringRef Reason) {
  if (Mode == WinX64EHUnwindV2Mode::Required)
    reportFatalInternalError("Windows x64 Unwind v2 is required, but LLVM has "
                             "generated incompatible code in function '" +
                             MF.getName() + "': " + Reason);

  FailsUnwindV2Criteria++;
  return false;
}
