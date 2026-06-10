//===-- X86WinEHUnwindV3.cpp - Win x64 Unwind v3 ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implements the capacity-checking and sub-fragment splitting pass for
/// Unwind v3 information. Unlike the V2 pass, V3 does not need to validate
/// epilog structure (V3 can encode any prolog/epilog pattern). This pass
/// only needs to:
///   1. Count prolog/epilog operations and epilogs.
///   2. Check V3 capacity limits (<=31 prolog/epilog ops, <=7 epilogs).
///   3. Insert sub-fragment split points if limits are exceeded.
///
/// The unwind version is set module-wide, not per-function.
///
/// See https://learn.microsoft.com/en-us/cpp/build/x64-unwind-information-v3
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "X86.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace llvm;

#define DEBUG_TYPE "x86-wineh-unwindv3"

STATISTIC(FunctionsProcessed,
          "Number of functions processed by Unwind v3 pass");
STATISTIC(SubFragmentSplits,
          "Number of sub-fragment splits inserted for Unwind v3");

/// V3 limits from the format specification.
static constexpr unsigned MaxV3PrologOps = 31;
static constexpr unsigned MaxV3Epilogs = 7;
static constexpr unsigned MaxV3EpilogOps = 31;

/// After reporting a recoverable error for `MF`, erase all SEH pseudo-
/// instructions and clear the WinCFI flag so the AsmPrinter doesn't try to
/// emit (potentially malformed) unwind information. The LLVMContext
/// diagnostic recorded by the caller will prevent the object file from
/// actually being written.
static void suppressWinCFI(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      switch (MI.getOpcode()) {
      case X86::SEH_PushReg:
      case X86::SEH_Push2Regs:
      case X86::SEH_SaveReg:
      case X86::SEH_SaveXMM:
      case X86::SEH_StackAlloc:
      case X86::SEH_StackAlign:
      case X86::SEH_SetFrame:
      case X86::SEH_PushFrame:
      case X86::SEH_EndPrologue:
      case X86::SEH_BeginEpilogue:
      case X86::SEH_EndEpilogue:
      case X86::SEH_SplitChained:
      case X86::SEH_SplitChainedAtEndOfBlock:
        MI.eraseFromParent();
        break;
      default:
        break;
      }
    }
  }
  MF.setHasWinCFI(false);
}

namespace {

/// Per-funclet analysis results.
struct FuncletInfo {
  unsigned PrologOpCount = 0;
  unsigned EpilogCount = 0;
  unsigned MaxEpilogOpCount = 0;
  /// SEH_BeginEpilogue instructions, used as insertion points for splitting.
  SmallVector<MachineInstr *, 8> EpilogBegins;
};

class X86WinEHUnwindV3 : public MachineFunctionPass {
public:
  static char ID;

  X86WinEHUnwindV3() : MachineFunctionPass(ID) {
    initializeX86WinEHUnwindV3Pass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "WinEH Unwind V3"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  /// Analyze one funclet (or the main function body) starting at Iter.
  /// Advances Iter past the analyzed region, stopping at the next funclet
  /// entry or the end of the function.
  static FuncletInfo analyzeFunclet(MachineFunction &MF,
                                    MachineFunction::iterator &Iter);
};

} // end anonymous namespace

char X86WinEHUnwindV3::ID = 0;

INITIALIZE_PASS(X86WinEHUnwindV3, "x86-wineh-unwindv3",
                "Capacity check and sub-fragment splitting for Win64 Unwind v3",
                false, false)

FunctionPass *llvm::createX86WinEHUnwindV3Pass() {
  return new X86WinEHUnwindV3();
}

FuncletInfo X86WinEHUnwindV3::analyzeFunclet(MachineFunction &MF,
                                             MachineFunction::iterator &Iter) {
  FuncletInfo Info;
  bool InEpilog = false;
  bool SeenProlog = false;
  unsigned CurrentEpilogOpCount = 0;

  for (; Iter != MF.end(); ++Iter) {
    MachineBasicBlock &MBB = *Iter;

    // If we've already been processing a funclet's prolog/body and encounter
    // another funclet entry, stop - that funclet gets its own analysis.
    if (MBB.isEHFuncletEntry() && SeenProlog)
      break;

    for (MachineInstr &MI : MBB) {
      switch (MI.getOpcode()) {
      case X86::SEH_PushReg:
      case X86::SEH_Push2Regs:
      case X86::SEH_StackAlloc:
      case X86::SEH_SetFrame:
      case X86::SEH_SaveReg:
      case X86::SEH_SaveXMM:
      case X86::SEH_PushFrame:
        if (InEpilog)
          CurrentEpilogOpCount++;
        else
          Info.PrologOpCount++;
        break;
      case X86::SEH_EndPrologue:
        SeenProlog = true;
        break;
      case X86::SEH_BeginEpilogue:
        InEpilog = true;
        CurrentEpilogOpCount = 0;
        Info.EpilogCount++;
        Info.EpilogBegins.push_back(&MI);
        break;
      case X86::SEH_EndEpilogue:
        InEpilog = false;
        Info.MaxEpilogOpCount =
            std::max(Info.MaxEpilogOpCount, CurrentEpilogOpCount);
        break;
      default:
        break;
      }
    }
  }

  return Info;
}

bool X86WinEHUnwindV3::runOnMachineFunction(MachineFunction &MF) {
  WinX64EHUnwindMode Mode =
      MF.getFunction().getParent()->getWinX64EHUnwindMode();

  Function &F = MF.getFunction();
  LLVMContext &Ctx = F.getContext();

  // EGPR (R16-R31) requires V3 unwind info because V1/V2 cannot encode
  // registers beyond R15. Only enforce this for functions that actually
  // emit SEH unwind info — `nounwind` functions and targets that don't
  // require unwind tables (e.g. cross-compilation host defaults) can use
  // EGPR with any unwind mode since no SEH metadata is generated.
  if (Mode != WinX64EHUnwindMode::V3) {
    if (!F.needsUnwindTableEntry())
      return false;
    const auto &STI = MF.getSubtarget<X86Subtarget>();
    if (STI.hasEGPR()) {
      Ctx.diagnose(DiagnosticInfoUnsupported(
          F, "EGPR (R16-R31) requires V3 unwind info on Windows x64"));
      // Stripping the SEH pseudos modifies the function, so report a change.
      suppressWinCFI(MF);
      return true;
    }
    return false;
  }

  bool Changed = false;
  MachineFunction::iterator Iter = MF.begin();

  // Process each funclet (and the main function body) independently.
  // Each funclet gets its own UNWIND_INFO, so V3 limits apply per funclet.
  while (Iter != MF.end()) {
    FuncletInfo Info = analyzeFunclet(MF, Iter);

    if (Info.PrologOpCount > MaxV3PrologOps) {
      Ctx.diagnose(DiagnosticInfoResourceLimit(
          F, "number of unwind v3 prolog operations required",
          Info.PrologOpCount, MaxV3PrologOps, DS_Error, DK_ResourceLimit));
      Ctx.diagnose(DiagnosticInfoGenericWithLoc(
          "sub-fragment splitting for prolog overflow is not yet implemented",
          F, F.getSubprogram(), DS_Note));
      // Stripping the SEH pseudos modifies the function, so report a change.
      suppressWinCFI(MF);
      return true;
    }

    if (Info.MaxEpilogOpCount > MaxV3EpilogOps) {
      Ctx.diagnose(DiagnosticInfoResourceLimit(
          F, "number of unwind v3 epilog operations required",
          Info.MaxEpilogOpCount, MaxV3EpilogOps, DS_Error, DK_ResourceLimit));
      Ctx.diagnose(DiagnosticInfoGenericWithLoc(
          "sub-fragment splitting for epilog overflow is not yet implemented",
          F, F.getSubprogram(), DS_Note));
      // Stripping the SEH pseudos modifies the function, so report a change.
      suppressWinCFI(MF);
      return true;
    }

    if (Info.EpilogCount > MaxV3Epilogs) {
      const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
      unsigned Count = 0;
      for (MachineInstr *BeginEpilog : Info.EpilogBegins) {
        Count++;
        if (Count > MaxV3Epilogs) {
          MachineBasicBlock *MBB = BeginEpilog->getParent();
          BuildMI(*MBB, BeginEpilog, BeginEpilog->getDebugLoc(),
                  TII->get(X86::SEH_SplitChained));
          BuildMI(*MBB, BeginEpilog, BeginEpilog->getDebugLoc(),
                  TII->get(X86::SEH_EndPrologue));
          SubFragmentSplits++;
          Count = 1;
        }
      }
      Changed = true;
    }
  }

  if (Changed)
    FunctionsProcessed++;

  return Changed;
}
