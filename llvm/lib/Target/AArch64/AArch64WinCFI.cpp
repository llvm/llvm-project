//===- AArch64WinCFI.cpp - Windows ARM64 unwind range insertion
//-------------===//
//
// This pass runs after machine block layout and branch relaxation to describe
// shrink-wrapped, handler-free functions as contiguous Windows unwind ranges.
// It tracks where the real prologue's effects remain in force, then inserts
// explicit range-boundary pseudos. Ranges entered with those effects already
// in force receive a copy of its unwind recipe and an end_c marker, describing
// a phantom prologue without emitting additional frame instructions.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-win-cfi"

namespace {
class AArch64WinCFILegacy : public MachineFunctionPass {
public:
  static char ID;
  AArch64WinCFILegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  StringRef getPassName() const override {
    return "AArch64 Windows unwind range insertion";
  }
};
} // namespace

char AArch64WinCFILegacy::ID = 0;

INITIALIZE_PASS(AArch64WinCFILegacy, DEBUG_TYPE,
                "AArch64 Windows unwind range insertion", false, false)

static void endWinCFIRange(MachineBasicBlock &MBB,
                           const AArch64InstrInfo &TII) {
  MachineInstr *End =
      BuildMI(MBB, MBB.end(), DebugLoc(), TII.get(AArch64::SEH_EndProc));
  // Cover the final terminator too. Bundling the zero-size end marker with
  // it preserves the terminator's CFG properties and keeps the MIR valid.
  if (End->getPrevNode() && End->getPrevNode()->isTerminator())
    End->bundleWithPred();
}

static bool insertWinCFIRanges(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<AArch64Subtarget>();

  // These range pseudos describe Windows SEH, not other unwind formats.
  if (!ST.isTargetWindows())
    return false;

  // ARM64EC's hybrid unwinding model is not supported by range insertion yet.
  if (ST.isWindowsArm64EC())
    return false;

  // Without Windows CFI, there is no SEH frame recipe to split into ranges.
  if (!MF.hasWinCFI())
    return false;

  // Leave handler-bearing functions on WinException's existing path until
  // range splitting preserves their language-specific handler data.
  if (MF.getFunction().hasPersonalityFn())
    return false;

  // Funclets have independent frames; this pass describes a single frame.
  if (MF.hasEHFunclets())
    return false;

  // Propagating the prologue's effects through the CFG needs an entry block.
  if (MF.empty())
    return false;

  // Find the physical prologue's SEH marker. Only a prologue outside the entry
  // block needs separate unwind ranges.
  MachineBasicBlock *PrologueBlock = nullptr;
  SmallPtrSet<const MachineBasicBlock *, 4> EpilogueBlocks;
  for (MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.getOpcode() == AArch64::SEH_StartProc)
        reportFatalInternalError(
            "AArch64 Windows unwind range insertion cannot be run twice");
      if (MI.getOpcode() == AArch64::SEH_PrologEnd)
        PrologueBlock = &MBB;
      if (MI.getOpcode() == AArch64::SEH_EpilogEnd)
        EpilogueBlocks.insert(&MBB);
    }
  }
  // No prologue found, or the prologue is in the entry block (so there is no
  // shrink wrapping), so nothing to do.
  if (!PrologueBlock || PrologueBlock == &MF.front())
    return false;

  // Collect the real prologue's SEH unwind pseudos to copy into phantom
  // prologues later; do not collect the instructions that establish the frame.
  // The first frame-setup instruction also locates the real range's start,
  // excluding any frameless prefix within this block.
  SmallVector<const MachineInstr *, 8> Prologue;
  auto PrologueStart = PrologueBlock->end();
  for (MachineInstr &MI : *PrologueBlock) {
    if (MI.getOpcode() == AArch64::SEH_PrologEnd)
      break;
    if (!MI.getFlag(MachineInstr::FrameSetup))
      continue;
    if (PrologueStart == PrologueBlock->end())
      PrologueStart = MI.getIterator();
    if (AArch64InstrInfo::isSEHInstruction(MI))
      Prologue.push_back(&MI);
  }
  // No prologue instructions, so nothing to do.
  if (PrologueStart == PrologueBlock->end())
    return false;

  // The framed region starts at the prologue and includes epilogue blocks,
  // but not their successors.
  SmallPtrSet<const MachineBasicBlock *, 16> FramedBlocks;
  SmallVector<const MachineBasicBlock *, 16> PendingBlocks{PrologueBlock};
  while (!PendingBlocks.empty()) {
    const MachineBasicBlock *Block = PendingBlocks.pop_back_val();
    if (!FramedBlocks.insert(Block).second || EpilogueBlocks.contains(Block))
      continue;
    append_range(PendingBlocks, Block->successors());
  }

  // Region membership must agree with the frame state on every reachable edge,
  // including joins with frameless paths and loop backedges.
  for (const MachineBasicBlock *Block : depth_first(&MF.front())) {
    bool FrameOnExit =
        FramedBlocks.contains(Block) && !EpilogueBlocks.contains(Block);
    for (const MachineBasicBlock *Successor : Block->successors()) {
      bool FrameOnEntry =
          Successor != PrologueBlock && FramedBlocks.contains(Successor);
      if (FrameOnExit != FrameOnEntry)
        reportFatalInternalError(
            "inconsistent Windows ARM64 frame state at CFG join");
    }
  }

  // Group framed blocks into contiguous records in final layout. Other gaps
  // retain leaf-style unwinding.
  const AArch64InstrInfo &TII = *ST.getInstrInfo();
  MachineBasicBlock *PreviousLayoutBlock = nullptr;
  bool RangeStarted = false;
  for (MachineBasicBlock &MBB : MF) {
    bool IsFramed = FramedBlocks.contains(&MBB);
    bool IsPhysicalPrologue = &MBB == PrologueBlock;
    // The physical prologue starts a new record. Otherwise only section
    // boundaries and frameless gaps separate adjacent framed blocks.
    if (RangeStarted &&
        (MBB.isBeginSection() || !IsFramed || IsPhysicalPrologue)) {
      endWinCFIRange(*PreviousLayoutBlock, TII);
      RangeStarted = false;
    }

    if (IsFramed && !RangeStarted) {
      if (IsPhysicalPrologue) {
        BuildMI(MBB, PrologueStart, DebugLoc(),
                TII.get(AArch64::SEH_StartProc));
      } else {
        // This range does not execute the prologue. Copy its recipe and use
        // end_c to prevent partial-prologue unwinding near the range start.
        auto InsertPt = MBB.begin();
        BuildMI(MBB, InsertPt, DebugLoc(), TII.get(AArch64::SEH_StartProc));
        for (const MachineInstr *MI : Prologue)
          MBB.insert(InsertPt, MF.CloneMachineInstr(MI));
        BuildMI(MBB, InsertPt, DebugLoc(), TII.get(AArch64::SEH_EndC));
        BuildMI(MBB, InsertPt, DebugLoc(), TII.get(AArch64::SEH_PrologEnd));
      }
      RangeStarted = true;
    }

    // Close the record while emission is still in its text section, before a
    // section switch or the end-of-function machinery runs.
    if (RangeStarted && (MBB.isEndSection() || &MBB == &MF.back())) {
      endWinCFIRange(MBB, TII);
      RangeStarted = false;
    }
    PreviousLayoutBlock = &MBB;
  }
  return true;
}

bool AArch64WinCFILegacy::runOnMachineFunction(MachineFunction &MF) {
  return insertWinCFIRanges(MF);
}

PreservedAnalyses AArch64WinCFIPass::run(MachineFunction &MF,
                                         MachineFunctionAnalysisManager &MFAM) {
  if (!insertWinCFIRanges(MF))
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

FunctionPass *llvm::createAArch64WinCFIPass() {
  return new AArch64WinCFILegacy();
}