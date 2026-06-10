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
#include "llvm/Support/CommandLine.h"

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

/// Approximate-instruction-count distance between an epilog and its fragment
/// tail beyond which the funclet is split into a new chained sub-fragment.
/// The V3 EpilogOffset field is a signed 16-bit byte offset measured from the
/// fragment tail, so each fragment must span less than 32 KiB of code. The
/// exact byte offsets aren't known until MC layout, so (like the V2 pass) the
/// approximate instruction count is used as a proxy, with margin for the
/// average emitted instruction size.
static cl::opt<unsigned> EpilogDistanceThreshold(
    "x86-wineh-unwindv3-epilog-distance-threshold", cl::Hidden,
    cl::desc("Maximum approximate instruction distance between an epilog and "
             "its fragment tail before splitting into a new chained unwind "
             "info for Unwind v3."),
    cl::init(4000));

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

/// A V3 epilog and the approximate instruction position where it begins, used
/// as a candidate sub-fragment split point.
struct EpilogSplitPoint {
  MachineInstr *BeginEpilog;
  unsigned ApproxInstrPos;
};

/// Per-funclet analysis results.
struct FuncletInfo {
  unsigned PrologOpCount = 0;
  unsigned MaxEpilogOpCount = 0;
  /// Approximate instruction position at the end of the funclet, used as the
  /// initial fragment tail reference for size-based splitting.
  unsigned EndInstrPos = 0;
  /// SEH_BeginEpilogue instructions (with approximate positions), used as
  /// candidate insertion points for sub-fragment splitting.
  SmallVector<EpilogSplitPoint, 8> Epilogs;
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
  /// entry or the end of the function. ApproxInstrPos is a running count of
  /// emitted instructions across the whole function, used to estimate the
  /// byte distance between epilogs and their fragment tail.
  static FuncletInfo analyzeFunclet(MachineFunction &MF,
                                    MachineFunction::iterator &Iter,
                                    unsigned &ApproxInstrPos);
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
                                             MachineFunction::iterator &Iter,
                                             unsigned &ApproxInstrPos) {
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
      // Approximate the number of emitted instructions, mirroring the V2 pass.
      // This estimates how far each epilog sits from its fragment tail; the
      // exact byte offsets aren't available until MC layout.
      if (!MI.isPseudo() && !MI.isMetaInstruction())
        ApproxInstrPos++;

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
        Info.Epilogs.push_back({&MI, ApproxInstrPos});
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

  Info.EndInstrPos = ApproxInstrPos;
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
  unsigned ApproxInstrPos = 0;
  MachineFunction::iterator Iter = MF.begin();

  // Process each funclet (and the main function body) independently.
  // Each funclet gets its own UNWIND_INFO, so V3 limits apply per funclet.
  while (Iter != MF.end()) {
    FuncletInfo Info = analyzeFunclet(MF, Iter, ApproxInstrPos);

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

    // Split the funclet into chained sub-fragments so that each fragment's
    // UNWIND_INFO stays within the V3 capacity limits:
    //   * at most 7 epilogs per fragment, and
    //   * every epilog close enough to its fragment tail that the tail-relative
    //     EpilogOffset fits in the signed 16-bit field.
    // The exact byte offsets aren't known until MC layout, so (like the V2
    // pass) the distance bound uses an approximate instruction count as a
    // proxy. A SEH_SplitChainedAtEndOfBlock is inserted at the start of an
    // epilog's block; the AsmPrinter emits the actual .seh_splitchained at the
    // *end* of that block, so the epilog becomes the last epilog of the
    // earlier fragment, immediately followed by the new chained fragment. This
    // keeps every epilog close to its tail even when a single epilog is
    // followed by a large amount of code (a long "tail" after the last
    // epilog is pushed into its own epilog-free chained fragment).
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    auto SplitAfter = [&](const EpilogSplitPoint &Epilog) {
      MachineBasicBlock *MBB = Epilog.BeginEpilog->getParent();
      BuildMI(*MBB, MBB->begin(), Epilog.BeginEpilog->getDebugLoc(),
              TII->get(X86::SEH_SplitChainedAtEndOfBlock));
      SubFragmentSplits++;
      Changed = true;
    };

    unsigned FragmentFirstPos = 0;
    unsigned EpilogsInFragment = 0;
    const EpilogSplitPoint *LastEpilog = nullptr;
    for (const EpilogSplitPoint &Epilog : Info.Epilogs) {
      // If adding this epilog would exceed a fragment limit, end the current
      // fragment after the previous epilog and start a new one here.
      if (EpilogsInFragment > 0 && (EpilogsInFragment >= MaxV3Epilogs ||
                                    Epilog.ApproxInstrPos - FragmentFirstPos >=
                                        EpilogDistanceThreshold)) {
        SplitAfter(*LastEpilog);
        EpilogsInFragment = 0;
      }
      if (EpilogsInFragment == 0)
        FragmentFirstPos = Epilog.ApproxInstrPos;
      EpilogsInFragment++;
      LastEpilog = &Epilog;
    }

    // If the last fragment's first epilog is too far from the funclet end,
    // split after the last epilog so the trailing code becomes its own
    // epilog-free chained fragment, keeping the last fragment's epilogs close
    // to their tail.
    if (LastEpilog &&
        Info.EndInstrPos - FragmentFirstPos >= EpilogDistanceThreshold)
      SplitAfter(*LastEpilog);
  }

  if (Changed)
    FunctionsProcessed++;

  return Changed;
}
