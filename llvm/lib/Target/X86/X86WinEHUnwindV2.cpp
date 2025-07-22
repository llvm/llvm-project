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
#include "llvm/IR/Module.h"

using namespace llvm;

#define DEBUG_TYPE "x86-wineh-unwindv2"

STATISTIC(MeetsUnwindV2Criteria,
          "Number of functions that meet Unwind v2 criteria");
STATISTIC(FailsUnwindV2Criteria,
          "Number of functions that fail Unwind v2 criteria");

namespace {

class X86WinEHUnwindV2 : public MachineFunctionPass {
public:
  static char ID;

  X86WinEHUnwindV2() : MachineFunctionPass(ID) {
    initializeX86WinEHUnwindV2Pass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "WinEH Unwind V2"; }

  bool runOnMachineFunction(MachineFunction &MF) override;
  bool rejectCurrentFunction() const {
    FailsUnwindV2Criteria++;
    return false;
  }
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

bool X86WinEHUnwindV2::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().getParent()->getModuleFlag("winx64-eh-unwindv2"))
    return false;

  // Current state of processing the function. We'll assume that all functions
  // start with a prolog.
  FunctionState State = FunctionState::InProlog;

  // Prolog information.
  SmallVector<int64_t> PushedRegs;
  bool HasStackAlloc = false;

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
        PushedRegs.push_back(MI.getOperand(0).getImm());
        break;

      case X86::SEH_StackAlloc:
      case X86::SEH_SetFrame:
        if (State != FunctionState::InProlog)
          llvm_unreachable("SEH_StackAlloc or SEH_SetFrame outside of prolog");
        HasStackAlloc = true;
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
        if ((HasStackAlloc != HasStackDealloc) ||
            (PoppedRegCount != PushedRegs.size()))
          // Non-canonical epilog, reject the function.
          return rejectCurrentFunction();

        // If we didn't find the start location, then use the end of the
        // epilog.
        if (!UnwindV2StartLocation)
          UnwindV2StartLocation = &MI;
        UnwindV2StartLocations.push_back(UnwindV2StartLocation);
        State = FunctionState::FinishedEpilog;
        break;

      case X86::MOV64rr:
      case X86::ADD64ri32:
        if (State == FunctionState::InEpilog) {
          // If the prolog contains a stack allocation, then the first
          // instruction in the epilog must be to adjust the stack pointer.
          if (!HasStackAlloc || HasStackDealloc || (PoppedRegCount > 0)) {
            return rejectCurrentFunction();
          }
          HasStackDealloc = true;
        } else if (State == FunctionState::FinishedEpilog)
          // Unexpected instruction after the epilog.
          return rejectCurrentFunction();
        break;

      case X86::POP64r:
        if (State == FunctionState::InEpilog) {
          // After the stack pointer has been adjusted, the epilog must
          // POP each register in reverse order of the PUSHes in the prolog.
          PoppedRegCount++;
          if ((HasStackAlloc != HasStackDealloc) ||
              (PoppedRegCount > PushedRegs.size()) ||
              (PushedRegs[PushedRegs.size() - PoppedRegCount] !=
               MI.getOperand(0).getReg())) {
            return rejectCurrentFunction();
          }

          // Unwind v2 records the size of the epilog not from where we place
          // SEH_BeginEpilogue (as that contains the instruction to adjust the
          // stack pointer) but from the first POP instruction (if there is
          // one).
          if (!UnwindV2StartLocation) {
            assert(PoppedRegCount == 1);
            UnwindV2StartLocation = &MI;
          }
        } else if (State == FunctionState::FinishedEpilog)
          // Unexpected instruction after the epilog.
          return rejectCurrentFunction();
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
            return rejectCurrentFunction();
        }
      }
    }
  }

  if (UnwindV2StartLocations.empty()) {
    assert(State == FunctionState::InProlog &&
           "If there are no epilogs, then there should be no prolog");
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
  MachineBasicBlock &FirstMBB = MF.front();
  BuildMI(FirstMBB, FirstMBB.front(), FirstMBB.front().getDebugLoc(),
          TII->get(X86::SEH_UnwindVersion))
      .addImm(2);

  return true;
}
