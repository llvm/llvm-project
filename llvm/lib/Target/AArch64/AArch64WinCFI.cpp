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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

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
  for (MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      // Resuming from transformed MIR must not insert a second set of ranges.
      if (MI.getOpcode() == AArch64::SEH_StartProc)
        return false;
      if (MI.getOpcode() == AArch64::SEH_PrologEnd)
        PrologueBlock = &MBB;
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

  // Track whether the real prologue has completed without a completed epilogue
  // undoing its effects. On entry, a block inherits this state from its CFG
  // predecessors; this does not say whether the block contains a prologue.
  // Function entry starts with no prologue in effect. Analyze before inserting
  // phantom markers so only real prologue and epilogue transitions count.
  DenseMap<const MachineBasicBlock *, bool> PrologueInEffectOnEntry;
  DenseMap<const MachineBasicBlock *, bool> PrologueInEffectOnExit;
  SmallVector<const MachineBasicBlock *, 16> PendingBlocks{&MF.front()};
  PrologueInEffectOnEntry[&MF.front()] = false;
  while (!PendingBlocks.empty()) {
    const MachineBasicBlock *Block = PendingBlocks.pop_back_val();
    bool PrologueInEffect = PrologueInEffectOnEntry.lookup(Block);
    for (const MachineInstr &MI : *Block) {
      if (MI.getOpcode() == AArch64::SEH_PrologEnd)
        PrologueInEffect = true;
      else if (MI.getOpcode() == AArch64::SEH_EpilogEnd)
        PrologueInEffect = false;
    }
    PrologueInEffectOnExit[Block] = PrologueInEffect;
    // Propagate this predecessor's exit state to each successor. Its other
    // predecessors, including loop backedges, must agree.
    for (const MachineBasicBlock *Successor : Block->successors()) {
      auto [EntryState, FirstVisit] =
          PrologueInEffectOnEntry.try_emplace(Successor, PrologueInEffect);
      if (!FirstVisit && EntryState->second != PrologueInEffect)
        reportFatalInternalError(
            "inconsistent Windows ARM64 frame state at CFG join");
      if (FirstVisit)
        PendingBlocks.push_back(Successor);
    }
  }

  // Walk final layout to group blocks where the prologue remains in effect
  // into contiguous records. Other gaps retain leaf-style unwinding.
  const AArch64InstrInfo &TII = *ST.getInstrInfo();
  MachineBasicBlock *PreviousLayoutBlock = nullptr;
  bool RangeStarted = false;
  for (MachineBasicBlock &MBB : MF) {
    bool PrologueInEffect = PrologueInEffectOnEntry.lookup(&MBB);
    // A record cannot span sections or bridge blocks where the prologue is not
    // in effect, even when they are adjacent in layout.
    if (RangeStarted && (MBB.isBeginSection() ||
                         !PrologueInEffectOnExit.lookup(PreviousLayoutBlock) ||
                         !PrologueInEffect)) {
      endWinCFIRange(*PreviousLayoutBlock, TII);
      RangeStarted = false;
    }

    // The real prologue is already in effect on entry to this range; this block
    // does not execute it. Copy its unwind recipe and use end_c so the unwinder
    // does not treat PCs near the range start as a partially executed prologue.
    if (PrologueInEffect && !RangeStarted) {
      assert(&MBB != PrologueBlock &&
             "Physical prologue cannot start a phantom unwind range");
      auto InsertPt = MBB.begin();
      BuildMI(MBB, InsertPt, DebugLoc(), TII.get(AArch64::SEH_StartProc));
      for (const MachineInstr *MI : Prologue)
        MBB.insert(InsertPt, MF.CloneMachineInstr(MI));
      BuildMI(MBB, InsertPt, DebugLoc(), TII.get(AArch64::SEH_EndC));
      BuildMI(MBB, InsertPt, DebugLoc(), TII.get(AArch64::SEH_PrologEnd));
      RangeStarted = true;
    }

    // Mark the real prologue - since we are manually managing the unwind
    // ranges, AsmPrinter won't insert the start/end for us.
    if (&MBB == PrologueBlock) {
      assert(!PrologueInEffect &&
             "Prologue is already in effect on entry to its physical block");
      assert(!RangeStarted &&
             "Physical prologue must start a new unwind range");
      BuildMI(MBB, PrologueStart, DebugLoc(), TII.get(AArch64::SEH_StartProc));
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