//===-- WebAssemblyCFGStackify.cpp - CFG Stackification -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a CFG stacking pass.
///
/// This pass inserts BLOCK, LOOP, and TRY markers to mark the start of scopes,
/// since scope boundaries serve as the labels for WebAssembly's control
/// transfers.
///
/// This is sufficient to convert arbitrary CFGs into a form that works on
/// WebAssembly, provided that all loops are single-entry.
///
/// In case we use exceptions, this pass also fixes mismatches in unwind
/// destinations created during transforming CFG into wasm structured format.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "WebAssemblyExceptionInfo.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySortRegion.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;
using WebAssembly::SortRegionInfo;

#define DEBUG_TYPE "wasm-cfg-stackify"

STATISTIC(NumUnwindMismatches, "Number of EH pad unwind mismatches found");

namespace {
class WebAssemblyCFGStackify final : public MachineFunctionPass {
  StringRef getPassName() const override { return "WebAssembly CFG Stackify"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachineLoopInfo>();
    AU.addRequired<WebAssemblyExceptionInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  // For each block whose label represents the end of a scope, record the block
  // which holds the beginning of the scope. This will allow us to quickly skip
  // over scoped regions when walking blocks.
  SmallVector<MachineBasicBlock *, 8> ScopeTops;
  void updateScopeTops(MachineBasicBlock *Begin, MachineBasicBlock *End) {
    int EndNo = End->getNumber();
    if (!ScopeTops[EndNo] || ScopeTops[EndNo]->getNumber() > Begin->getNumber())
      ScopeTops[EndNo] = Begin;
  }

  // Placing markers.
  void placeMarkers(MachineFunction &MF);
  void placeBlockMarker(MachineBasicBlock &MBB);
  void placeLoopMarker(MachineBasicBlock &MBB);
  void placeTryMarker(MachineBasicBlock &MBB);
  void removeUnnecessaryInstrs(MachineFunction &MF);
  bool fixUnwindMismatches(MachineFunction &MF);
  void rewriteDepthImmediates(MachineFunction &MF);
  void fixEndsAtEndOfFunction(MachineFunction &MF);

  // For each BLOCK|LOOP|TRY, the corresponding END_(BLOCK|LOOP|TRY).
  DenseMap<const MachineInstr *, MachineInstr *> BeginToEnd;
  // For each END_(BLOCK|LOOP|TRY), the corresponding BLOCK|LOOP|TRY.
  DenseMap<const MachineInstr *, MachineInstr *> EndToBegin;
  // <TRY marker, EH pad> map
  DenseMap<const MachineInstr *, MachineBasicBlock *> TryToEHPad;
  // <EH pad, TRY marker> map
  DenseMap<const MachineBasicBlock *, MachineInstr *> EHPadToTry;

  // There can be an appendix block at the end of each function, shared for:
  // - creating a correct signature for fallthrough returns
  // - target for rethrows that need to unwind to the caller, but are trapped
  //   inside another try/catch
  MachineBasicBlock *AppendixBB = nullptr;
  MachineBasicBlock *getAppendixBlock(MachineFunction &MF) {
    if (!AppendixBB) {
      AppendixBB = MF.CreateMachineBasicBlock();
      // Give it a fake predecessor so that AsmPrinter prints its label.
      AppendixBB->addSuccessor(AppendixBB);
      MF.push_back(AppendixBB);
    }
    return AppendixBB;
  }

  // Helper functions to register / unregister scope information created by
  // marker instructions.
  void registerScope(MachineInstr *Begin, MachineInstr *End);
  void registerTryScope(MachineInstr *Begin, MachineInstr *End,
                        MachineBasicBlock *EHPad);
  void unregisterScope(MachineInstr *Begin);

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyCFGStackify() : MachineFunctionPass(ID) {}
  ~WebAssemblyCFGStackify() override { releaseMemory(); }
  void releaseMemory() override;
};
} // end anonymous namespace

char WebAssemblyCFGStackify::ID = 0;
INITIALIZE_PASS(WebAssemblyCFGStackify, DEBUG_TYPE,
                "Insert BLOCK/LOOP/TRY markers for WebAssembly scopes", false,
                false)

FunctionPass *llvm::createWebAssemblyCFGStackify() {
  return new WebAssemblyCFGStackify();
}

/// Test whether Pred has any terminators explicitly branching to MBB, as
/// opposed to falling through. Note that it's possible (eg. in unoptimized
/// code) for a branch instruction to both branch to a block and fallthrough
/// to it, so we check the actual branch operands to see if there are any
/// explicit mentions.
static bool explicitlyBranchesTo(MachineBasicBlock *Pred,
                                 MachineBasicBlock *MBB) {
  for (MachineInstr &MI : Pred->terminators())
    for (MachineOperand &MO : MI.explicit_operands())
      if (MO.isMBB() && MO.getMBB() == MBB)
        return true;
  return false;
}

// Returns an iterator to the earliest position possible within the MBB,
// satisfying the restrictions given by BeforeSet and AfterSet. BeforeSet
// contains instructions that should go before the marker, and AfterSet contains
// ones that should go after the marker. In this function, AfterSet is only
// used for sanity checking.
template <typename Container>
static MachineBasicBlock::iterator
getEarliestInsertPos(MachineBasicBlock *MBB, const Container &BeforeSet,
                     const Container &AfterSet) {
  auto InsertPos = MBB->end();
  while (InsertPos != MBB->begin()) {
    if (BeforeSet.count(&*std::prev(InsertPos))) {
#ifndef NDEBUG
      // Sanity check
      for (auto Pos = InsertPos, E = MBB->begin(); Pos != E; --Pos)
        assert(!AfterSet.count(&*std::prev(Pos)));
#endif
      break;
    }
    --InsertPos;
  }
  return InsertPos;
}

// Returns an iterator to the latest position possible within the MBB,
// satisfying the restrictions given by BeforeSet and AfterSet. BeforeSet
// contains instructions that should go before the marker, and AfterSet contains
// ones that should go after the marker. In this function, BeforeSet is only
// used for sanity checking.
template <typename Container>
static MachineBasicBlock::iterator
getLatestInsertPos(MachineBasicBlock *MBB, const Container &BeforeSet,
                   const Container &AfterSet) {
  auto InsertPos = MBB->begin();
  while (InsertPos != MBB->end()) {
    if (AfterSet.count(&*InsertPos)) {
#ifndef NDEBUG
      // Sanity check
      for (auto Pos = InsertPos, E = MBB->end(); Pos != E; ++Pos)
        assert(!BeforeSet.count(&*Pos));
#endif
      break;
    }
    ++InsertPos;
  }
  return InsertPos;
}

void WebAssemblyCFGStackify::registerScope(MachineInstr *Begin,
                                           MachineInstr *End) {
  BeginToEnd[Begin] = End;
  EndToBegin[End] = Begin;
}

void WebAssemblyCFGStackify::registerTryScope(MachineInstr *Begin,
                                              MachineInstr *End,
                                              MachineBasicBlock *EHPad) {
  registerScope(Begin, End);
  TryToEHPad[Begin] = EHPad;
  EHPadToTry[EHPad] = Begin;
}

void WebAssemblyCFGStackify::unregisterScope(MachineInstr *Begin) {
  assert(BeginToEnd.count(Begin));
  MachineInstr *End = BeginToEnd[Begin];
  assert(EndToBegin.count(End));
  BeginToEnd.erase(Begin);
  EndToBegin.erase(End);
  MachineBasicBlock *EHPad = TryToEHPad.lookup(Begin);
  if (EHPad) {
    assert(EHPadToTry.count(EHPad));
    TryToEHPad.erase(Begin);
    EHPadToTry.erase(EHPad);
  }
}

/// Insert a BLOCK marker for branches to MBB (if needed).
// TODO Consider a more generalized way of handling block (and also loop and
// try) signatures when we implement the multi-value proposal later.
void WebAssemblyCFGStackify::placeBlockMarker(MachineBasicBlock &MBB) {
  assert(!MBB.isEHPad());
  MachineFunction &MF = *MBB.getParent();
  auto &MDT = getAnalysis<MachineDominatorTree>();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  const auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();

  // First compute the nearest common dominator of all forward non-fallthrough
  // predecessors so that we minimize the time that the BLOCK is on the stack,
  // which reduces overall stack height.
  MachineBasicBlock *Header = nullptr;
  bool IsBranchedTo = false;
  int MBBNumber = MBB.getNumber();
  for (MachineBasicBlock *Pred : MBB.predecessors()) {
    if (Pred->getNumber() < MBBNumber) {
      Header = Header ? MDT.findNearestCommonDominator(Header, Pred) : Pred;
      if (explicitlyBranchesTo(Pred, &MBB))
        IsBranchedTo = true;
    }
  }
  if (!Header)
    return;
  if (!IsBranchedTo)
    return;

  assert(&MBB != &MF.front() && "Header blocks shouldn't have predecessors");
  MachineBasicBlock *LayoutPred = MBB.getPrevNode();

  // If the nearest common dominator is inside a more deeply nested context,
  // walk out to the nearest scope which isn't more deeply nested.
  for (MachineFunction::iterator I(LayoutPred), E(Header); I != E; --I) {
    if (MachineBasicBlock *ScopeTop = ScopeTops[I->getNumber()]) {
      if (ScopeTop->getNumber() > Header->getNumber()) {
        // Skip over an intervening scope.
        I = std::next(ScopeTop->getIterator());
      } else {
        // We found a scope level at an appropriate depth.
        Header = ScopeTop;
        break;
      }
    }
  }

  // Decide where in Header to put the BLOCK.

  // Instructions that should go before the BLOCK.
  SmallPtrSet<const MachineInstr *, 4> BeforeSet;
  // Instructions that should go after the BLOCK.
  SmallPtrSet<const MachineInstr *, 4> AfterSet;
  for (const auto &MI : *Header) {
    // If there is a previously placed LOOP marker and the bottom block of the
    // loop is above MBB, it should be after the BLOCK, because the loop is
    // nested in this BLOCK. Otherwise it should be before the BLOCK.
    if (MI.getOpcode() == WebAssembly::LOOP) {
      auto *LoopBottom = BeginToEnd[&MI]->getParent()->getPrevNode();
      if (MBB.getNumber() > LoopBottom->getNumber())
        AfterSet.insert(&MI);
#ifndef NDEBUG
      else
        BeforeSet.insert(&MI);
#endif
    }

    // If there is a previously placed BLOCK/TRY marker and its corresponding
    // END marker is before the current BLOCK's END marker, that should be
    // placed after this BLOCK. Otherwise it should be placed before this BLOCK
    // marker.
    if (MI.getOpcode() == WebAssembly::BLOCK ||
        MI.getOpcode() == WebAssembly::TRY) {
      if (BeginToEnd[&MI]->getParent()->getNumber() <= MBB.getNumber())
        AfterSet.insert(&MI);
#ifndef NDEBUG
      else
        BeforeSet.insert(&MI);
#endif
    }

#ifndef NDEBUG
    // All END_(BLOCK|LOOP|TRY) markers should be before the BLOCK.
    if (MI.getOpcode() == WebAssembly::END_BLOCK ||
        MI.getOpcode() == WebAssembly::END_LOOP ||
        MI.getOpcode() == WebAssembly::END_TRY)
      BeforeSet.insert(&MI);
#endif

    // Terminators should go after the BLOCK.
    if (MI.isTerminator())
      AfterSet.insert(&MI);
  }

  // Local expression tree should go after the BLOCK.
  for (auto I = Header->getFirstTerminator(), E = Header->begin(); I != E;
       --I) {
    if (std::prev(I)->isDebugInstr() || std::prev(I)->isPosition())
      continue;
    if (WebAssembly::isChild(*std::prev(I), MFI))
      AfterSet.insert(&*std::prev(I));
    else
      break;
  }

  // Add the BLOCK.
  WebAssembly::BlockType ReturnType = WebAssembly::BlockType::Void;
  auto InsertPos = getLatestInsertPos(Header, BeforeSet, AfterSet);
  MachineInstr *Begin =
      BuildMI(*Header, InsertPos, Header->findDebugLoc(InsertPos),
              TII.get(WebAssembly::BLOCK))
          .addImm(int64_t(ReturnType));

  // Decide where in Header to put the END_BLOCK.
  BeforeSet.clear();
  AfterSet.clear();
  for (auto &MI : MBB) {
#ifndef NDEBUG
    // END_BLOCK should precede existing LOOP and TRY markers.
    if (MI.getOpcode() == WebAssembly::LOOP ||
        MI.getOpcode() == WebAssembly::TRY)
      AfterSet.insert(&MI);
#endif

    // If there is a previously placed END_LOOP marker and the header of the
    // loop is above this block's header, the END_LOOP should be placed after
    // the BLOCK, because the loop contains this block. Otherwise the END_LOOP
    // should be placed before the BLOCK. The same for END_TRY.
    if (MI.getOpcode() == WebAssembly::END_LOOP ||
        MI.getOpcode() == WebAssembly::END_TRY) {
      if (EndToBegin[&MI]->getParent()->getNumber() >= Header->getNumber())
        BeforeSet.insert(&MI);
#ifndef NDEBUG
      else
        AfterSet.insert(&MI);
#endif
    }
  }

  // Mark the end of the block.
  InsertPos = getEarliestInsertPos(&MBB, BeforeSet, AfterSet);
  MachineInstr *End = BuildMI(MBB, InsertPos, MBB.findPrevDebugLoc(InsertPos),
                              TII.get(WebAssembly::END_BLOCK));
  registerScope(Begin, End);

  // Track the farthest-spanning scope that ends at this point.
  updateScopeTops(Header, &MBB);
}

/// Insert a LOOP marker for a loop starting at MBB (if it's a loop header).
void WebAssemblyCFGStackify::placeLoopMarker(MachineBasicBlock &MBB) {
  MachineFunction &MF = *MBB.getParent();
  const auto &MLI = getAnalysis<MachineLoopInfo>();
  const auto &WEI = getAnalysis<WebAssemblyExceptionInfo>();
  SortRegionInfo SRI(MLI, WEI);
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  MachineLoop *Loop = MLI.getLoopFor(&MBB);
  if (!Loop || Loop->getHeader() != &MBB)
    return;

  // The operand of a LOOP is the first block after the loop. If the loop is the
  // bottom of the function, insert a dummy block at the end.
  MachineBasicBlock *Bottom = SRI.getBottom(Loop);
  auto Iter = std::next(Bottom->getIterator());
  if (Iter == MF.end()) {
    getAppendixBlock(MF);
    Iter = std::next(Bottom->getIterator());
  }
  MachineBasicBlock *AfterLoop = &*Iter;

  // Decide where in Header to put the LOOP.
  SmallPtrSet<const MachineInstr *, 4> BeforeSet;
  SmallPtrSet<const MachineInstr *, 4> AfterSet;
  for (const auto &MI : MBB) {
    // LOOP marker should be after any existing loop that ends here. Otherwise
    // we assume the instruction belongs to the loop.
    if (MI.getOpcode() == WebAssembly::END_LOOP)
      BeforeSet.insert(&MI);
#ifndef NDEBUG
    else
      AfterSet.insert(&MI);
#endif
  }

  // Mark the beginning of the loop.
  auto InsertPos = getEarliestInsertPos(&MBB, BeforeSet, AfterSet);
  MachineInstr *Begin = BuildMI(MBB, InsertPos, MBB.findDebugLoc(InsertPos),
                                TII.get(WebAssembly::LOOP))
                            .addImm(int64_t(WebAssembly::BlockType::Void));

  // Decide where in Header to put the END_LOOP.
  BeforeSet.clear();
  AfterSet.clear();
#ifndef NDEBUG
  for (const auto &MI : MBB)
    // Existing END_LOOP markers belong to parent loops of this loop
    if (MI.getOpcode() == WebAssembly::END_LOOP)
      AfterSet.insert(&MI);
#endif

  // Mark the end of the loop (using arbitrary debug location that branched to
  // the loop end as its location).
  InsertPos = getEarliestInsertPos(AfterLoop, BeforeSet, AfterSet);
  DebugLoc EndDL = AfterLoop->pred_empty()
                       ? DebugLoc()
                       : (*AfterLoop->pred_rbegin())->findBranchDebugLoc();
  MachineInstr *End =
      BuildMI(*AfterLoop, InsertPos, EndDL, TII.get(WebAssembly::END_LOOP));
  registerScope(Begin, End);

  assert((!ScopeTops[AfterLoop->getNumber()] ||
          ScopeTops[AfterLoop->getNumber()]->getNumber() < MBB.getNumber()) &&
         "With block sorting the outermost loop for a block should be first.");
  updateScopeTops(&MBB, AfterLoop);
}

void WebAssemblyCFGStackify::placeTryMarker(MachineBasicBlock &MBB) {
  assert(MBB.isEHPad());
  MachineFunction &MF = *MBB.getParent();
  auto &MDT = getAnalysis<MachineDominatorTree>();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  const auto &MLI = getAnalysis<MachineLoopInfo>();
  const auto &WEI = getAnalysis<WebAssemblyExceptionInfo>();
  SortRegionInfo SRI(MLI, WEI);
  const auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();

  // Compute the nearest common dominator of all unwind predecessors
  MachineBasicBlock *Header = nullptr;
  int MBBNumber = MBB.getNumber();
  for (auto *Pred : MBB.predecessors()) {
    if (Pred->getNumber() < MBBNumber) {
      Header = Header ? MDT.findNearestCommonDominator(Header, Pred) : Pred;
      assert(!explicitlyBranchesTo(Pred, &MBB) &&
             "Explicit branch to an EH pad!");
    }
  }
  if (!Header)
    return;

  // If this try is at the bottom of the function, insert a dummy block at the
  // end.
  WebAssemblyException *WE = WEI.getExceptionFor(&MBB);
  assert(WE);
  MachineBasicBlock *Bottom = SRI.getBottom(WE);

  auto Iter = std::next(Bottom->getIterator());
  if (Iter == MF.end()) {
    getAppendixBlock(MF);
    Iter = std::next(Bottom->getIterator());
  }
  MachineBasicBlock *Cont = &*Iter;

  assert(Cont != &MF.front());
  MachineBasicBlock *LayoutPred = Cont->getPrevNode();

  // If the nearest common dominator is inside a more deeply nested context,
  // walk out to the nearest scope which isn't more deeply nested.
  for (MachineFunction::iterator I(LayoutPred), E(Header); I != E; --I) {
    if (MachineBasicBlock *ScopeTop = ScopeTops[I->getNumber()]) {
      if (ScopeTop->getNumber() > Header->getNumber()) {
        // Skip over an intervening scope.
        I = std::next(ScopeTop->getIterator());
      } else {
        // We found a scope level at an appropriate depth.
        Header = ScopeTop;
        break;
      }
    }
  }

  // Decide where in Header to put the TRY.

  // Instructions that should go before the TRY.
  SmallPtrSet<const MachineInstr *, 4> BeforeSet;
  // Instructions that should go after the TRY.
  SmallPtrSet<const MachineInstr *, 4> AfterSet;
  for (const auto &MI : *Header) {
    // If there is a previously placed LOOP marker and the bottom block of the
    // loop is above MBB, it should be after the TRY, because the loop is nested
    // in this TRY. Otherwise it should be before the TRY.
    if (MI.getOpcode() == WebAssembly::LOOP) {
      auto *LoopBottom = BeginToEnd[&MI]->getParent()->getPrevNode();
      if (MBB.getNumber() > LoopBottom->getNumber())
        AfterSet.insert(&MI);
#ifndef NDEBUG
      else
        BeforeSet.insert(&MI);
#endif
    }

    // All previously inserted BLOCK/TRY markers should be after the TRY because
    // they are all nested trys.
    if (MI.getOpcode() == WebAssembly::BLOCK ||
        MI.getOpcode() == WebAssembly::TRY)
      AfterSet.insert(&MI);

#ifndef NDEBUG
    // All END_(BLOCK/LOOP/TRY) markers should be before the TRY.
    if (MI.getOpcode() == WebAssembly::END_BLOCK ||
        MI.getOpcode() == WebAssembly::END_LOOP ||
        MI.getOpcode() == WebAssembly::END_TRY)
      BeforeSet.insert(&MI);
#endif

    // Terminators should go after the TRY.
    if (MI.isTerminator())
      AfterSet.insert(&MI);
  }

  // If Header unwinds to MBB (= Header contains 'invoke'), the try block should
  // contain the call within it. So the call should go after the TRY. The
  // exception is when the header's terminator is a rethrow instruction, in
  // which case that instruction, not a call instruction before it, is gonna
  // throw.
  MachineInstr *ThrowingCall = nullptr;
  if (MBB.isPredecessor(Header)) {
    auto TermPos = Header->getFirstTerminator();
    if (TermPos == Header->end() ||
        TermPos->getOpcode() != WebAssembly::RETHROW) {
      for (auto &MI : reverse(*Header)) {
        if (MI.isCall()) {
          AfterSet.insert(&MI);
          ThrowingCall = &MI;
          // Possibly throwing calls are usually wrapped by EH_LABEL
          // instructions. We don't want to split them and the call.
          if (MI.getIterator() != Header->begin() &&
              std::prev(MI.getIterator())->isEHLabel()) {
            AfterSet.insert(&*std::prev(MI.getIterator()));
            ThrowingCall = &*std::prev(MI.getIterator());
          }
          break;
        }
      }
    }
  }

  // Local expression tree should go after the TRY.
  // For BLOCK placement, we start the search from the previous instruction of a
  // BB's terminator, but in TRY's case, we should start from the previous
  // instruction of a call that can throw, or a EH_LABEL that precedes the call,
  // because the return values of the call's previous instructions can be
  // stackified and consumed by the throwing call.
  auto SearchStartPt = ThrowingCall ? MachineBasicBlock::iterator(ThrowingCall)
                                    : Header->getFirstTerminator();
  for (auto I = SearchStartPt, E = Header->begin(); I != E; --I) {
    if (std::prev(I)->isDebugInstr() || std::prev(I)->isPosition())
      continue;
    if (WebAssembly::isChild(*std::prev(I), MFI))
      AfterSet.insert(&*std::prev(I));
    else
      break;
  }

  // Add the TRY.
  auto InsertPos = getLatestInsertPos(Header, BeforeSet, AfterSet);
  MachineInstr *Begin =
      BuildMI(*Header, InsertPos, Header->findDebugLoc(InsertPos),
              TII.get(WebAssembly::TRY))
          .addImm(int64_t(WebAssembly::BlockType::Void));

  // Decide where in Header to put the END_TRY.
  BeforeSet.clear();
  AfterSet.clear();
  for (const auto &MI : *Cont) {
#ifndef NDEBUG
    // END_TRY should precede existing LOOP and BLOCK markers.
    if (MI.getOpcode() == WebAssembly::LOOP ||
        MI.getOpcode() == WebAssembly::BLOCK)
      AfterSet.insert(&MI);

    // All END_TRY markers placed earlier belong to exceptions that contains
    // this one.
    if (MI.getOpcode() == WebAssembly::END_TRY)
      AfterSet.insert(&MI);
#endif

    // If there is a previously placed END_LOOP marker and its header is after
    // where TRY marker is, this loop is contained within the 'catch' part, so
    // the END_TRY marker should go after that. Otherwise, the whole try-catch
    // is contained within this loop, so the END_TRY should go before that.
    if (MI.getOpcode() == WebAssembly::END_LOOP) {
      // For a LOOP to be after TRY, LOOP's BB should be after TRY's BB; if they
      // are in the same BB, LOOP is always before TRY.
      if (EndToBegin[&MI]->getParent()->getNumber() > Header->getNumber())
        BeforeSet.insert(&MI);
#ifndef NDEBUG
      else
        AfterSet.insert(&MI);
#endif
    }

    // It is not possible for an END_BLOCK to be already in this block.
  }

  // Mark the end of the TRY.
  InsertPos = getEarliestInsertPos(Cont, BeforeSet, AfterSet);
  MachineInstr *End =
      BuildMI(*Cont, InsertPos, Bottom->findBranchDebugLoc(),
              TII.get(WebAssembly::END_TRY));
  registerTryScope(Begin, End, &MBB);

  // Track the farthest-spanning scope that ends at this point. We create two
  // mappings: (BB with 'end_try' -> BB with 'try') and (BB with 'catch' -> BB
  // with 'try'). We need to create 'catch' -> 'try' mapping here too because
  // markers should not span across 'catch'. For example, this should not
  // happen:
  //
  // try
  //   block     --|  (X)
  // catch         |
  //   end_block --|
  // end_try
  for (auto *End : {&MBB, Cont})
    updateScopeTops(Header, End);
}

void WebAssemblyCFGStackify::removeUnnecessaryInstrs(MachineFunction &MF) {
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  // When there is an unconditional branch right before a catch instruction and
  // it branches to the end of end_try marker, we don't need the branch, because
  // it there is no exception, the control flow transfers to that point anyway.
  // bb0:
  //   try
  //     ...
  //     br bb2      <- Not necessary
  // bb1 (ehpad):
  //   catch
  //     ...
  // bb2:            <- Continuation BB
  //   end
  //
  // A more involved case: When the BB where 'end' is located is an another EH
  // pad, the Cont (= continuation) BB is that EH pad's 'end' BB. For example,
  // bb0:
  //   try
  //     try
  //       ...
  //       br bb3      <- Not necessary
  // bb1 (ehpad):
  //     catch
  // bb2 (ehpad):
  //     end
  //   catch
  //     ...
  // bb3:            <- Continuation BB
  //   end
  //
  // When the EH pad at hand is bb1, its matching end_try is in bb2. But it is
  // another EH pad, so bb0's continuation BB becomes bb3. So 'br bb3' in the
  // code can be deleted. This is why we run 'while' until 'Cont' is not an EH
  // pad.
  for (auto &MBB : MF) {
    if (!MBB.isEHPad())
      continue;

    MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
    SmallVector<MachineOperand, 4> Cond;
    MachineBasicBlock *EHPadLayoutPred = MBB.getPrevNode();

    MachineBasicBlock *Cont = &MBB;
    while (Cont->isEHPad()) {
      MachineInstr *Try = EHPadToTry[Cont];
      MachineInstr *EndTry = BeginToEnd[Try];
      Cont = EndTry->getParent();
    }

    bool Analyzable = !TII.analyzeBranch(*EHPadLayoutPred, TBB, FBB, Cond);
    // This condition means either
    // 1. This BB ends with a single unconditional branch whose destinaion is
    //    Cont.
    // 2. This BB ends with a conditional branch followed by an unconditional
    //    branch, and the unconditional branch's destination is Cont.
    // In both cases, we want to remove the last (= unconditional) branch.
    if (Analyzable && ((Cond.empty() && TBB && TBB == Cont) ||
                       (!Cond.empty() && FBB && FBB == Cont))) {
      bool ErasedUncondBr = false;
      (void)ErasedUncondBr;
      for (auto I = EHPadLayoutPred->end(), E = EHPadLayoutPred->begin();
           I != E; --I) {
        auto PrevI = std::prev(I);
        if (PrevI->isTerminator()) {
          assert(PrevI->getOpcode() == WebAssembly::BR);
          PrevI->eraseFromParent();
          ErasedUncondBr = true;
          break;
        }
      }
      assert(ErasedUncondBr && "Unconditional branch not erased!");
    }
  }

  // When there are block / end_block markers that overlap with try / end_try
  // markers, and the block and try markers' return types are the same, the
  // block /end_block markers are not necessary, because try / end_try markers
  // also can serve as boundaries for branches.
  // block         <- Not necessary
  //   try
  //     ...
  //   catch
  //     ...
  //   end
  // end           <- Not necessary
  SmallVector<MachineInstr *, 32> ToDelete;
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.getOpcode() != WebAssembly::TRY)
        continue;

      MachineInstr *Try = &MI, *EndTry = BeginToEnd[Try];
      MachineBasicBlock *TryBB = Try->getParent();
      MachineBasicBlock *Cont = EndTry->getParent();
      int64_t RetType = Try->getOperand(0).getImm();
      for (auto B = Try->getIterator(), E = std::next(EndTry->getIterator());
           B != TryBB->begin() && E != Cont->end() &&
           std::prev(B)->getOpcode() == WebAssembly::BLOCK &&
           E->getOpcode() == WebAssembly::END_BLOCK &&
           std::prev(B)->getOperand(0).getImm() == RetType;
           --B, ++E) {
        ToDelete.push_back(&*std::prev(B));
        ToDelete.push_back(&*E);
      }
    }
  }
  for (auto *MI : ToDelete) {
    if (MI->getOpcode() == WebAssembly::BLOCK)
      unregisterScope(MI);
    MI->eraseFromParent();
  }
}

// Get the appropriate copy opcode for the given register class.
static unsigned getCopyOpcode(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return WebAssembly::COPY_I32;
  if (RC == &WebAssembly::I64RegClass)
    return WebAssembly::COPY_I64;
  if (RC == &WebAssembly::F32RegClass)
    return WebAssembly::COPY_F32;
  if (RC == &WebAssembly::F64RegClass)
    return WebAssembly::COPY_F64;
  if (RC == &WebAssembly::V128RegClass)
    return WebAssembly::COPY_V128;
  if (RC == &WebAssembly::FUNCREFRegClass)
    return WebAssembly::COPY_FUNCREF;
  if (RC == &WebAssembly::EXTERNREFRegClass)
    return WebAssembly::COPY_EXTERNREF;
  llvm_unreachable("Unexpected register class");
}

// When MBB is split into MBB and Split, we should unstackify defs in MBB that
// have their uses in Split.
// FIXME This function will be used when fixing unwind mismatches, but the old
// version of that function was removed for the moment and the new version has
// not yet been added. So 'LLVM_ATTRIBUTE_UNUSED' is added to suppress the
// warning. Remove the attribute after the new functionality is added.
LLVM_ATTRIBUTE_UNUSED static void
unstackifyVRegsUsedInSplitBB(MachineBasicBlock &MBB, MachineBasicBlock &Split) {
  MachineFunction &MF = *MBB.getParent();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  auto &MRI = MF.getRegInfo();

  for (auto &MI : Split) {
    for (auto &MO : MI.explicit_uses()) {
      if (!MO.isReg() || Register::isPhysicalRegister(MO.getReg()))
        continue;
      if (MachineInstr *Def = MRI.getUniqueVRegDef(MO.getReg()))
        if (Def->getParent() == &MBB)
          MFI.unstackifyVReg(MO.getReg());
    }
  }

  // In RegStackify, when a register definition is used multiple times,
  //    Reg = INST ...
  //    INST ..., Reg, ...
  //    INST ..., Reg, ...
  //    INST ..., Reg, ...
  //
  // we introduce a TEE, which has the following form:
  //    DefReg = INST ...
  //    TeeReg, Reg = TEE_... DefReg
  //    INST ..., TeeReg, ...
  //    INST ..., Reg, ...
  //    INST ..., Reg, ...
  // with DefReg and TeeReg stackified but Reg not stackified.
  //
  // But the invariant that TeeReg should be stackified can be violated while we
  // unstackify registers in the split BB above. In this case, we convert TEEs
  // into two COPYs. This COPY will be eventually eliminated in ExplicitLocals.
  //    DefReg = INST ...
  //    TeeReg = COPY DefReg
  //    Reg = COPY DefReg
  //    INST ..., TeeReg, ...
  //    INST ..., Reg, ...
  //    INST ..., Reg, ...
  for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
    MachineInstr &MI = *I++;
    if (!WebAssembly::isTee(MI.getOpcode()))
      continue;
    Register TeeReg = MI.getOperand(0).getReg();
    Register Reg = MI.getOperand(1).getReg();
    Register DefReg = MI.getOperand(2).getReg();
    if (!MFI.isVRegStackified(TeeReg)) {
      // Now we are not using TEE anymore, so unstackify DefReg too
      MFI.unstackifyVReg(DefReg);
      unsigned CopyOpc = getCopyOpcode(MRI.getRegClass(DefReg));
      BuildMI(MBB, &MI, MI.getDebugLoc(), TII.get(CopyOpc), TeeReg)
          .addReg(DefReg);
      BuildMI(MBB, &MI, MI.getDebugLoc(), TII.get(CopyOpc), Reg).addReg(DefReg);
      MI.eraseFromParent();
    }
  }
}

bool WebAssemblyCFGStackify::fixUnwindMismatches(MachineFunction &MF) {
  // TODO Implement this
  return false;
}

static unsigned
getDepth(const SmallVectorImpl<const MachineBasicBlock *> &Stack,
         const MachineBasicBlock *MBB) {
  unsigned Depth = 0;
  for (auto X : reverse(Stack)) {
    if (X == MBB)
      break;
    ++Depth;
  }
  assert(Depth < Stack.size() && "Branch destination should be in scope");
  return Depth;
}

/// In normal assembly languages, when the end of a function is unreachable,
/// because the function ends in an infinite loop or a noreturn call or similar,
/// it isn't necessary to worry about the function return type at the end of
/// the function, because it's never reached. However, in WebAssembly, blocks
/// that end at the function end need to have a return type signature that
/// matches the function signature, even though it's unreachable. This function
/// checks for such cases and fixes up the signatures.
void WebAssemblyCFGStackify::fixEndsAtEndOfFunction(MachineFunction &MF) {
  const auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();

  if (MFI.getResults().empty())
    return;

  // MCInstLower will add the proper types to multivalue signatures based on the
  // function return type
  WebAssembly::BlockType RetType =
      MFI.getResults().size() > 1
          ? WebAssembly::BlockType::Multivalue
          : WebAssembly::BlockType(
                WebAssembly::toValType(MFI.getResults().front()));

  SmallVector<MachineBasicBlock::reverse_iterator, 4> Worklist;
  Worklist.push_back(MF.rbegin()->rbegin());

  auto Process = [&](MachineBasicBlock::reverse_iterator It) {
    auto *MBB = It->getParent();
    while (It != MBB->rend()) {
      MachineInstr &MI = *It++;
      if (MI.isPosition() || MI.isDebugInstr())
        continue;
      switch (MI.getOpcode()) {
      case WebAssembly::END_TRY: {
        // If a 'try''s return type is fixed, both its try body and catch body
        // should satisfy the return type, so we need to search 'end'
        // instructions before its corresponding 'catch' too.
        auto *EHPad = TryToEHPad.lookup(EndToBegin[&MI]);
        assert(EHPad);
        auto NextIt =
            std::next(WebAssembly::findCatch(EHPad)->getReverseIterator());
        if (NextIt != EHPad->rend())
          Worklist.push_back(NextIt);
        LLVM_FALLTHROUGH;
      }
      case WebAssembly::END_BLOCK:
      case WebAssembly::END_LOOP:
        EndToBegin[&MI]->getOperand(0).setImm(int32_t(RetType));
        continue;
      default:
        // Something other than an `end`. We're done for this BB.
        return;
      }
    }
    // We've reached the beginning of a BB. Continue the search in the previous
    // BB.
    Worklist.push_back(MBB->getPrevNode()->rbegin());
  };

  while (!Worklist.empty())
    Process(Worklist.pop_back_val());
}

// WebAssembly functions end with an end instruction, as if the function body
// were a block.
static void appendEndToFunction(MachineFunction &MF,
                                const WebAssemblyInstrInfo &TII) {
  BuildMI(MF.back(), MF.back().end(),
          MF.back().findPrevDebugLoc(MF.back().end()),
          TII.get(WebAssembly::END_FUNCTION));
}

/// Insert LOOP/TRY/BLOCK markers at appropriate places.
void WebAssemblyCFGStackify::placeMarkers(MachineFunction &MF) {
  // We allocate one more than the number of blocks in the function to
  // accommodate for the possible fake block we may insert at the end.
  ScopeTops.resize(MF.getNumBlockIDs() + 1);
  // Place the LOOP for MBB if MBB is the header of a loop.
  for (auto &MBB : MF)
    placeLoopMarker(MBB);

  const MCAsmInfo *MCAI = MF.getTarget().getMCAsmInfo();
  for (auto &MBB : MF) {
    if (MBB.isEHPad()) {
      // Place the TRY for MBB if MBB is the EH pad of an exception.
      if (MCAI->getExceptionHandlingType() == ExceptionHandling::Wasm &&
          MF.getFunction().hasPersonalityFn())
        placeTryMarker(MBB);
    } else {
      // Place the BLOCK for MBB if MBB is branched to from above.
      placeBlockMarker(MBB);
    }
  }
  // Fix mismatches in unwind destinations induced by linearizing the code.
  if (MCAI->getExceptionHandlingType() == ExceptionHandling::Wasm &&
      MF.getFunction().hasPersonalityFn())
    fixUnwindMismatches(MF);
}

void WebAssemblyCFGStackify::rewriteDepthImmediates(MachineFunction &MF) {
  // Now rewrite references to basic blocks to be depth immediates.
  SmallVector<const MachineBasicBlock *, 8> Stack;
  for (auto &MBB : reverse(MF)) {
    for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
      MachineInstr &MI = *I;
      switch (MI.getOpcode()) {
      case WebAssembly::BLOCK:
      case WebAssembly::TRY:
        assert(ScopeTops[Stack.back()->getNumber()]->getNumber() <=
                   MBB.getNumber() &&
               "Block/try marker should be balanced");
        Stack.pop_back();
        break;

      case WebAssembly::LOOP:
        assert(Stack.back() == &MBB && "Loop top should be balanced");
        Stack.pop_back();
        break;

      case WebAssembly::END_BLOCK:
      case WebAssembly::END_TRY:
        Stack.push_back(&MBB);
        break;

      case WebAssembly::END_LOOP:
        Stack.push_back(EndToBegin[&MI]->getParent());
        break;

      default:
        if (MI.isTerminator()) {
          // Rewrite MBB operands to be depth immediates.
          SmallVector<MachineOperand, 4> Ops(MI.operands());
          while (MI.getNumOperands() > 0)
            MI.RemoveOperand(MI.getNumOperands() - 1);
          for (auto MO : Ops) {
            if (MO.isMBB())
              MO = MachineOperand::CreateImm(getDepth(Stack, MO.getMBB()));
            MI.addOperand(MF, MO);
          }
        }
        break;
      }
    }
  }
  assert(Stack.empty() && "Control flow should be balanced");
}

void WebAssemblyCFGStackify::releaseMemory() {
  ScopeTops.clear();
  BeginToEnd.clear();
  EndToBegin.clear();
  TryToEHPad.clear();
  EHPadToTry.clear();
  AppendixBB = nullptr;
}

bool WebAssemblyCFGStackify::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** CFG Stackifying **********\n"
                       "********** Function: "
                    << MF.getName() << '\n');
  const MCAsmInfo *MCAI = MF.getTarget().getMCAsmInfo();

  releaseMemory();

  // Liveness is not tracked for VALUE_STACK physreg.
  MF.getRegInfo().invalidateLiveness();

  // Place the BLOCK/LOOP/TRY markers to indicate the beginnings of scopes.
  placeMarkers(MF);

  // Remove unnecessary instructions possibly introduced by try/end_trys.
  if (MCAI->getExceptionHandlingType() == ExceptionHandling::Wasm &&
      MF.getFunction().hasPersonalityFn())
    removeUnnecessaryInstrs(MF);

  // Convert MBB operands in terminators to relative depth immediates.
  rewriteDepthImmediates(MF);

  // Fix up block/loop/try signatures at the end of the function to conform to
  // WebAssembly's rules.
  fixEndsAtEndOfFunction(MF);

  // Add an end instruction at the end of the function body.
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  if (!MF.getSubtarget<WebAssemblySubtarget>()
           .getTargetTriple()
           .isOSBinFormatELF())
    appendEndToFunction(MF, TII);

  MF.getInfo<WebAssemblyFunctionInfo>()->setCFGStackified();
  return true;
}
