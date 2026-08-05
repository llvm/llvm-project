//===-- X86WinEHUnwindV3.cpp - Win x64 Unwind v3 ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implements the capacity-checking and sub-fragment splitting pass for
/// Unwind v3 information. V3 can encode any prolog/epilog pattern, so this
/// pass does not validate epilog structure; it only needs to:
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
#include "llvm/Support/Debug.h"

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
static constexpr unsigned EpilogDistanceThreshold = 32767;

/// Approximate byte distance between an epilog and its fragment tail beyond
/// which the funclet is split into a new chained sub-fragment. The V3
/// EpilogOffset field is a signed 16-bit byte offset measured from the
/// fragment tail, so each fragment must span less than 32 KiB of code. The
/// exact byte offsets aren't known until MC layout, so (like the V2 pass) an
/// approximate byte count is used as a proxy — instructions are charged
/// ApproxBytesPerInstr each and alignment padding is added.
static cl::opt<unsigned> ApproxBytesPerInstr(
    "x86-wineh-unwindv3-instr-avg-size", cl::Hidden,
    cl::desc(
        "Average size of an instruction. This value is used in determining "
        "split points for chained unwinder info"),
    cl::init(7));

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

/// A V3 epilog and the approximate byte position where it begins, used
/// as a candidate sub-fragment split point.
struct EpilogSplitPoint {
  MachineInstr *BeginEpilog;
  unsigned ApproxBytePos;
};

/// Per-funclet analysis results.
struct FuncletInfo {
  unsigned PrologOpCount = 0;
  unsigned MaxEpilogOpCount = 0;
  /// Approximate byte position at the end of the funclet, used as the
  /// initial fragment tail reference for size-based splitting.
  unsigned EndBytePos = 0;
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
  /// entry or the end of the function. ApproxBytePos is a running estimate of
  /// the byte position across the whole function, used to estimate the byte
  /// distance between epilogs and their fragment tail.
  static FuncletInfo analyzeFunclet(MachineFunction &MF,
                                    MachineFunction::iterator &Iter,
                                    unsigned &ApproxBytePos);
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
                                             unsigned &ApproxBytePos) {
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

    // Account for worst-case scenario of padding inserted to align this block.
    Align A = MBB.getAlignment();
    unsigned MaxPadding = A.value() - 1;
    if (unsigned MaxBytes = MBB.getMaxBytesForAlignment())
      MaxPadding = std::min(MaxPadding, MaxBytes);
    ApproxBytePos += MaxPadding;

    for (MachineInstr &MI : MBB) {
      // Approximate the emitted byte size, mirroring the V2 pass. This
      // estimates how far each epilog sits from its fragment tail; the exact
      // byte offsets aren't available until MC layout, so each real
      // instruction is charged ApproxBytesPerInstr bytes.
      if (!MI.isPseudo() && !MI.isMetaInstruction())
        ApproxBytePos += ApproxBytesPerInstr;

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
        LLVM_DEBUG(dbgs() << "  epilog " << Info.Epilogs.size()
                          << " begins at approx byte position " << ApproxBytePos
                          << "\n");
        Info.Epilogs.push_back({&MI, ApproxBytePos});
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

  Info.EndBytePos = ApproxBytePos;
  LLVM_DEBUG(dbgs() << "  funclet has " << Info.Epilogs.size()
                    << " epilog(s); ends at approx byte position "
                    << ApproxBytePos << "\n");
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
  unsigned ApproxBytePos = 0;
  MachineFunction::iterator Iter = MF.begin();

  LLVM_DEBUG(dbgs() << "X86WinEHUnwindV3: processing " << MF.getName() << "\n");

  // Process each funclet (and the main function body) independently.
  // Each funclet gets its own UNWIND_INFO, so V3 limits apply per funclet.
  while (Iter != MF.end()) {
    FuncletInfo Info = analyzeFunclet(MF, Iter, ApproxBytePos);

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
    // UNWIND_INFO stays within the V3 capacity limits: at most 7 epilogs per
    // fragment, and each adjacent-epilog gap (plus the gap from the last epilog
    // to the fragment tail) small enough that the corresponding signed-16-bit
    // EpilogOffset delta fits.
    //
    // A SEH_SplitChainedAtEndOfBlock inserted at the start of an epilog's
    // block makes the AsmPrinter emit the actual .seh_splitchained at the
    // *end* of that block, so the epilog becomes the last epilog of the
    // earlier fragment, immediately followed by the new chained fragment. A
    // long tail after the last epilog is pushed into its own epilog-free
    // chained fragment.
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    auto SplitAfter = [&](const EpilogSplitPoint &Epilog) {
      MachineBasicBlock *MBB = Epilog.BeginEpilog->getParent();
      BuildMI(*MBB, MBB->begin(), Epilog.BeginEpilog->getDebugLoc(),
              TII->get(X86::SEH_SplitChainedAtEndOfBlock));
      SubFragmentSplits++;
      Changed = true;
    };

    unsigned EpilogsInFragment = 0;
    const EpilogSplitPoint *LastEpilog = nullptr;
    [[maybe_unused]] unsigned LastEpilogIdx = 0;
    for (unsigned Idx = 0; Idx < Info.Epilogs.size(); ++Idx) {
      const EpilogSplitPoint &Epilog = Info.Epilogs[Idx];
      // If adding this epilog would exceed a fragment limit or is too far, end
      // the current fragment after the previous epilog and start a new one.
      if (EpilogsInFragment > 0) {
        bool ExceedsEpilogCount = EpilogsInFragment >= MaxV3Epilogs;
        bool ExceedsDistance =
            Epilog.ApproxBytePos - LastEpilog->ApproxBytePos >=
            EpilogDistanceThreshold;
        if (ExceedsEpilogCount || ExceedsDistance) {
          LLVM_DEBUG({
            dbgs() << "  splitting after epilog " << LastEpilogIdx
                   << " because adding epilog " << Idx << " would exceed the ";
            if (ExceedsEpilogCount)
              dbgs() << "7-epilog-per-fragment limit\n";
            else
              dbgs() << "epilog distance threshold (gap from previous epilog "
                        "at "
                     << LastEpilog->ApproxBytePos << " to epilog at "
                     << Epilog.ApproxBytePos << ")\n";
          });
          SplitAfter(*LastEpilog);
          EpilogsInFragment = 0;
        }
      }
      EpilogsInFragment++;
      LastEpilog = &Epilog;
      LastEpilogIdx = Idx;
    }

    // If the last epilog is too far from the funclet end, split after it so the
    // trailing code becomes its own epilog-free chained fragment.
    if (LastEpilog && Info.EndBytePos - LastEpilog->ApproxBytePos >=
                          EpilogDistanceThreshold) {
      LLVM_DEBUG(dbgs() << "  splitting after last epilog " << LastEpilogIdx
                        << " to isolate the trailing tail (gap from epilog at "
                        << LastEpilog->ApproxBytePos << " to funclet end "
                        << Info.EndBytePos << ")\n");
      SplitAfter(*LastEpilog);
    }
  }

  if (Changed)
    FunctionsProcessed++;

  return Changed;
}
