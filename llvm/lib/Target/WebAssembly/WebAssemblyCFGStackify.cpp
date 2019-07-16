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
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/MC/MCAsmInfo.h"
using namespace llvm;

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
static MachineBasicBlock::iterator
getEarliestInsertPos(MachineBasicBlock *MBB,
                     const SmallPtrSet<const MachineInstr *, 4> &BeforeSet,
                     const SmallPtrSet<const MachineInstr *, 4> &AfterSet) {
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
static MachineBasicBlock::iterator
getLatestInsertPos(MachineBasicBlock *MBB,
                   const SmallPtrSet<const MachineInstr *, 4> &BeforeSet,
                   const SmallPtrSet<const MachineInstr *, 4> &AfterSet) {
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
  bool IsBrOnExn = false;
  MachineInstr *BrOnExn = nullptr;
  int MBBNumber = MBB.getNumber();
  for (MachineBasicBlock *Pred : MBB.predecessors()) {
    if (Pred->getNumber() < MBBNumber) {
      Header = Header ? MDT.findNearestCommonDominator(Header, Pred) : Pred;
      if (explicitlyBranchesTo(Pred, &MBB)) {
        IsBranchedTo = true;
        if (Pred->getFirstTerminator()->getOpcode() == WebAssembly::BR_ON_EXN) {
          IsBrOnExn = true;
          assert(!BrOnExn && "There should be only one br_on_exn per block");
          BrOnExn = &*Pred->getFirstTerminator();
        }
      }
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

    // All previously inserted BLOCK/TRY markers should be after the BLOCK
    // because they are all nested blocks.
    if (MI.getOpcode() == WebAssembly::BLOCK ||
        MI.getOpcode() == WebAssembly::TRY)
      AfterSet.insert(&MI);

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

  // 'br_on_exn' extracts exnref object and pushes variable number of values
  // depending on its tag. For C++ exception, its a single i32 value, and the
  // generated code will be in the form of:
  // block i32
  //   br_on_exn 0, $__cpp_exception
  //   rethrow
  // end_block
  WebAssembly::ExprType ReturnType = WebAssembly::ExprType::Void;
  if (IsBrOnExn) {
    const char *TagName = BrOnExn->getOperand(1).getSymbolName();
    if (std::strcmp(TagName, "__cpp_exception") != 0)
      llvm_unreachable("Only C++ exception is supported");
    ReturnType = WebAssembly::ExprType::I32;
  }

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
  int Number = MBB.getNumber();
  if (!ScopeTops[Number] ||
      ScopeTops[Number]->getNumber() > Header->getNumber())
    ScopeTops[Number] = Header;
}

/// Insert a LOOP marker for a loop starting at MBB (if it's a loop header).
void WebAssemblyCFGStackify::placeLoopMarker(MachineBasicBlock &MBB) {
  MachineFunction &MF = *MBB.getParent();
  const auto &MLI = getAnalysis<MachineLoopInfo>();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  MachineLoop *Loop = MLI.getLoopFor(&MBB);
  if (!Loop || Loop->getHeader() != &MBB)
    return;

  // The operand of a LOOP is the first block after the loop. If the loop is the
  // bottom of the function, insert a dummy block at the end.
  MachineBasicBlock *Bottom = WebAssembly::getBottom(Loop);
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
                            .addImm(int64_t(WebAssembly::ExprType::Void));

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
  if (!ScopeTops[AfterLoop->getNumber()])
    ScopeTops[AfterLoop->getNumber()] = &MBB;
}

void WebAssemblyCFGStackify::placeTryMarker(MachineBasicBlock &MBB) {
  assert(MBB.isEHPad());
  MachineFunction &MF = *MBB.getParent();
  auto &MDT = getAnalysis<MachineDominatorTree>();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  const auto &WEI = getAnalysis<WebAssemblyExceptionInfo>();
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
  MachineBasicBlock *Bottom = WebAssembly::getBottom(WE);

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

  // Local expression tree should go after the TRY.
  for (auto I = Header->getFirstTerminator(), E = Header->begin(); I != E;
       --I) {
    if (std::prev(I)->isDebugInstr() || std::prev(I)->isPosition())
      continue;
    if (WebAssembly::isChild(*std::prev(I), MFI))
      AfterSet.insert(&*std::prev(I));
    else
      break;
  }

  // If Header unwinds to MBB (= Header contains 'invoke'), the try block should
  // contain the call within it. So the call should go after the TRY. The
  // exception is when the header's terminator is a rethrow instruction, in
  // which case that instruction, not a call instruction before it, is gonna
  // throw.
  if (MBB.isPredecessor(Header)) {
    auto TermPos = Header->getFirstTerminator();
    if (TermPos == Header->end() ||
        TermPos->getOpcode() != WebAssembly::RETHROW) {
      for (const auto &MI : reverse(*Header)) {
        if (MI.isCall()) {
          AfterSet.insert(&MI);
          // Possibly throwing calls are usually wrapped by EH_LABEL
          // instructions. We don't want to split them and the call.
          if (MI.getIterator() != Header->begin() &&
              std::prev(MI.getIterator())->isEHLabel())
            AfterSet.insert(&*std::prev(MI.getIterator()));
          break;
        }
      }
    }
  }

  // Add the TRY.
  auto InsertPos = getLatestInsertPos(Header, BeforeSet, AfterSet);
  MachineInstr *Begin =
      BuildMI(*Header, InsertPos, Header->findDebugLoc(InsertPos),
              TII.get(WebAssembly::TRY))
          .addImm(int64_t(WebAssembly::ExprType::Void));

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
  for (int Number : {Cont->getNumber(), MBB.getNumber()}) {
    if (!ScopeTops[Number] ||
        ScopeTops[Number]->getNumber() > Header->getNumber())
      ScopeTops[Number] = Header;
  }
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
  // bb1:
  //   catch
  //     ...
  // bb2:
  //   end
  for (auto &MBB : MF) {
    if (!MBB.isEHPad())
      continue;

    MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
    SmallVector<MachineOperand, 4> Cond;
    MachineBasicBlock *EHPadLayoutPred = MBB.getPrevNode();
    MachineBasicBlock *Cont = BeginToEnd[EHPadToTry[&MBB]]->getParent();
    bool Analyzable = !TII.analyzeBranch(*EHPadLayoutPred, TBB, FBB, Cond);
    if (Analyzable && ((Cond.empty() && TBB && TBB == Cont) ||
                       (!Cond.empty() && FBB && FBB == Cont)))
      TII.removeBranch(*EHPadLayoutPred);
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

bool WebAssemblyCFGStackify::fixUnwindMismatches(MachineFunction &MF) {
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Linearizing the control flow by placing TRY / END_TRY markers can create
  // mismatches in unwind destinations. There are two kinds of mismatches we
  // try to solve here.

  // 1. When an instruction may throw, but the EH pad it will unwind to can be
  //    different from the original CFG.
  //
  // Example: we have the following CFG:
  // bb0:
  //   call @foo (if it throws, unwind to bb2)
  // bb1:
  //   call @bar (if it throws, unwind to bb3)
  // bb2 (ehpad):
  //   catch
  //   ...
  // bb3 (ehpad)
  //   catch
  //   handler body
  //
  // And the CFG is sorted in this order. Then after placing TRY markers, it
  // will look like: (BB markers are omitted)
  // try $label1
  //   try
  //     call @foo
  //     call @bar   (if it throws, unwind to bb3)
  //   catch         <- ehpad (bb2)
  //     ...
  //   end_try
  // catch           <- ehpad (bb3)
  //   handler body
  // end_try
  //
  // Now if bar() throws, it is going to end up ip in bb2, not bb3, where it
  // is supposed to end up. We solve this problem by
  // a. Split the target unwind EH pad (here bb3) so that the handler body is
  //    right after 'end_try', which means we extract the handler body out of
  //    the catch block. We do this because this handler body should be
  //    somewhere branch-eable from the inner scope.
  // b. Wrap the call that has an incorrect unwind destination ('call @bar'
  //    here) with a nested try/catch/end_try scope, and within the new catch
  //    block, branches to the handler body.
  // c. Place a branch after the newly inserted nested end_try so it can bypass
  //    the handler body, which is now outside of a catch block.
  //
  // The result will like as follows. (new: a) means this instruction is newly
  // created in the process of doing 'a' above.
  //
  // block $label0                 (new: placeBlockMarker)
  //   try $label1
  //     try
  //       call @foo
  //       try                     (new: b)
  //         call @bar
  //       catch                   (new: b)
  //         local.set n / drop    (new: b)
  //         br $label1            (new: b)
  //       end_try                 (new: b)
  //     catch                     <- ehpad (bb2)
  //     end_try
  //     br $label0                (new: c)
  //   catch                       <- ehpad (bb3)
  //   end_try                     (hoisted: a)
  //   handler body
  // end_block                     (new: placeBlockMarker)
  //
  // Note that the new wrapping block/end_block will be generated later in
  // placeBlockMarker.
  //
  // TODO Currently local.set and local.gets are generated to move exnref value
  // created by catches. That's because we don't support yielding values from a
  // block in LLVM machine IR yet, even though it is supported by wasm. Delete
  // unnecessary local.get/local.sets once yielding values from a block is
  // supported. The full EH spec requires multi-value support to do this, but
  // for C++ we don't yet need it because we only throw a single i32.
  //
  // ---
  // 2. The same as 1, but in this case an instruction unwinds to a caller
  //    function and not another EH pad.
  //
  // Example: we have the following CFG:
  // bb0:
  //   call @foo (if it throws, unwind to bb2)
  // bb1:
  //   call @bar (if it throws, unwind to caller)
  // bb2 (ehpad):
  //   catch
  //   ...
  //
  // And the CFG is sorted in this order. Then after placing TRY markers, it
  // will look like:
  // try
  //   call @foo
  //   call @bar   (if it throws, unwind to caller)
  // catch         <- ehpad (bb2)
  //   ...
  // end_try
  //
  // Now if bar() throws, it is going to end up ip in bb2, when it is supposed
  // throw up to the caller.
  // We solve this problem by
  // a. Create a new 'appendix' BB at the end of the function and put a single
  //    'rethrow' instruction (+ local.get) in there.
  // b. Wrap the call that has an incorrect unwind destination ('call @bar'
  //    here) with a nested try/catch/end_try scope, and within the new catch
  //    block, branches to the new appendix block.
  //
  // block $label0          (new: placeBlockMarker)
  //   try
  //     call @foo
  //     try                (new: b)
  //       call @bar
  //     catch              (new: b)
  //       local.set n      (new: b)
  //       br $label0       (new: b)
  //     end_try            (new: b)
  //   catch                <- ehpad (bb2)
  //     ...
  //   end_try
  // ...
  // end_block              (new: placeBlockMarker)
  // local.get n            (new: a)  <- appendix block
  // rethrow                (new: a)
  //
  // In case there are multiple calls in a BB that may throw to the caller, they
  // can be wrapped together in one nested try scope. (In 1, this couldn't
  // happen, because may-throwing instruction there had an unwind destination,
  // i.e., it was an invoke before, and there could be only one invoke within a
  // BB.)

  SmallVector<const MachineBasicBlock *, 8> EHPadStack;
  // Range of intructions to be wrapped in a new nested try/catch
  using TryRange = std::pair<MachineInstr *, MachineInstr *>;
  // In original CFG, <unwind destionation BB, a vector of try ranges>
  DenseMap<MachineBasicBlock *, SmallVector<TryRange, 4>> UnwindDestToTryRanges;
  // In new CFG, <destination to branch to, a vector of try ranges>
  DenseMap<MachineBasicBlock *, SmallVector<TryRange, 4>> BrDestToTryRanges;
  // In new CFG, <destination to branch to, register containing exnref>
  DenseMap<MachineBasicBlock *, unsigned> BrDestToExnReg;

  // Gather possibly throwing calls (i.e., previously invokes) whose current
  // unwind destination is not the same as the original CFG.
  for (auto &MBB : reverse(MF)) {
    bool SeenThrowableInstInBB = false;
    for (auto &MI : reverse(MBB)) {
      if (MI.getOpcode() == WebAssembly::TRY)
        EHPadStack.pop_back();
      else if (MI.getOpcode() == WebAssembly::CATCH)
        EHPadStack.push_back(MI.getParent());

      // In this loop we only gather calls that have an EH pad to unwind. So
      // there will be at most 1 such call (= invoke) in a BB, so after we've
      // seen one, we can skip the rest of BB. Also if MBB has no EH pad
      // successor or MI does not throw, this is not an invoke.
      if (SeenThrowableInstInBB || !MBB.hasEHPadSuccessor() ||
          !WebAssembly::mayThrow(MI))
        continue;
      SeenThrowableInstInBB = true;

      // If the EH pad on the stack top is where this instruction should unwind
      // next, we're good.
      MachineBasicBlock *UnwindDest = nullptr;
      for (auto *Succ : MBB.successors()) {
        if (Succ->isEHPad()) {
          UnwindDest = Succ;
          break;
        }
      }
      if (EHPadStack.back() == UnwindDest)
        continue;

      // If not, record the range.
      UnwindDestToTryRanges[UnwindDest].push_back(TryRange(&MI, &MI));
    }
  }

  assert(EHPadStack.empty());

  // Gather possibly throwing calls that are supposed to unwind up to the caller
  // if they throw, but currently unwind to an incorrect destination. Unlike the
  // loop above, there can be multiple calls within a BB that unwind to the
  // caller, which we should group together in a range.
  bool NeedAppendixBlock = false;
  for (auto &MBB : reverse(MF)) {
    MachineInstr *RangeBegin = nullptr, *RangeEnd = nullptr; // inclusive
    for (auto &MI : reverse(MBB)) {
      if (MI.getOpcode() == WebAssembly::TRY)
        EHPadStack.pop_back();
      else if (MI.getOpcode() == WebAssembly::CATCH)
        EHPadStack.push_back(MI.getParent());

      // If MBB has an EH pad successor, this inst does not unwind to caller.
      if (MBB.hasEHPadSuccessor())
        continue;

      // We wrap up the current range when we see a marker even if we haven't
      // finished a BB.
      if (RangeEnd && WebAssembly::isMarker(MI.getOpcode())) {
        NeedAppendixBlock = true;
        // Record the range. nullptr here means the unwind destination is the
        // caller.
        UnwindDestToTryRanges[nullptr].push_back(
            TryRange(RangeBegin, RangeEnd));
        RangeBegin = RangeEnd = nullptr; // Reset range pointers
      }

      // If EHPadStack is empty, that means it is correctly unwind to caller if
      // it throws, so we're good. If MI does not throw, we're good too.
      if (EHPadStack.empty() || !WebAssembly::mayThrow(MI))
        continue;

      // We found an instruction that unwinds to the caller but currently has an
      // incorrect unwind destination. Create a new range or increment the
      // currently existing range.
      if (!RangeEnd)
        RangeBegin = RangeEnd = &MI;
      else
        RangeBegin = &MI;
    }

    if (RangeEnd) {
      NeedAppendixBlock = true;
      // Record the range. nullptr here means the unwind destination is the
      // caller.
      UnwindDestToTryRanges[nullptr].push_back(TryRange(RangeBegin, RangeEnd));
      RangeBegin = RangeEnd = nullptr; // Reset range pointers
    }
  }

  assert(EHPadStack.empty());
  // We don't have any unwind destination mismatches to resolve.
  if (UnwindDestToTryRanges.empty())
    return false;

  // If we found instructions that should unwind to the caller but currently
  // have incorrect unwind destination, we create an appendix block at the end
  // of the function with a local.get and a rethrow instruction.
  if (NeedAppendixBlock) {
    auto *AppendixBB = getAppendixBlock(MF);
    unsigned ExnReg = MRI.createVirtualRegister(&WebAssembly::EXNREFRegClass);
    BuildMI(AppendixBB, DebugLoc(), TII.get(WebAssembly::RETHROW))
        .addReg(ExnReg);
    // These instruction ranges should branch to this appendix BB.
    for (auto Range : UnwindDestToTryRanges[nullptr])
      BrDestToTryRanges[AppendixBB].push_back(Range);
    BrDestToExnReg[AppendixBB] = ExnReg;
  }

  // We loop through unwind destination EH pads that are targeted from some
  // inner scopes. Because these EH pads are destination of more than one scope
  // now, we split them so that the handler body is after 'end_try'.
  // - Before
  // ehpad:
  //   catch
  //   local.set n / drop
  //   handler body
  // ...
  // cont:
  //   end_try
  //
  // - After
  // ehpad:
  //   catch
  //   local.set n / drop
  // brdest:               (new)
  //   end_try             (hoisted from 'cont' BB)
  //   handler body        (taken from 'ehpad')
  // ...
  // cont:
  for (auto &P : UnwindDestToTryRanges) {
    NumUnwindMismatches++;

    // This means the destination is the appendix BB, which was separately
    // handled above.
    if (!P.first)
      continue;

    MachineBasicBlock *EHPad = P.first;

    // Find 'catch' and 'local.set' or 'drop' instruction that follows the
    // 'catch'. If -wasm-disable-explicit-locals is not set, 'catch' should be
    // always followed by either 'local.set' or a 'drop', because 'br_on_exn' is
    // generated after 'catch' in LateEHPrepare and we don't support blocks
    // taking values yet.
    MachineInstr *Catch = nullptr;
    unsigned ExnReg = 0;
    for (auto &MI : *EHPad) {
      switch (MI.getOpcode()) {
      case WebAssembly::CATCH:
        Catch = &MI;
        ExnReg = Catch->getOperand(0).getReg();
        break;
      }
    }
    assert(Catch && "EH pad does not have a catch");
    assert(ExnReg != 0 && "Invalid register");

    auto SplitPos = std::next(Catch->getIterator());

    // Create a new BB that's gonna be the destination for branches from the
    // inner mismatched scope.
    MachineInstr *BeginTry = EHPadToTry[EHPad];
    MachineInstr *EndTry = BeginToEnd[BeginTry];
    MachineBasicBlock *Cont = EndTry->getParent();
    auto *BrDest = MF.CreateMachineBasicBlock();
    MF.insert(std::next(EHPad->getIterator()), BrDest);
    // Hoist up the existing 'end_try'.
    BrDest->insert(BrDest->end(), EndTry->removeFromParent());
    // Take out the handler body from EH pad to the new branch destination BB.
    BrDest->splice(BrDest->end(), EHPad, SplitPos, EHPad->end());
    // Fix predecessor-successor relationship.
    BrDest->transferSuccessors(EHPad);
    EHPad->addSuccessor(BrDest);

    // All try ranges that were supposed to unwind to this EH pad now have to
    // branch to this new branch dest BB.
    for (auto Range : UnwindDestToTryRanges[EHPad])
      BrDestToTryRanges[BrDest].push_back(Range);
    BrDestToExnReg[BrDest] = ExnReg;

    // In case we fall through to the continuation BB after the catch block, we
    // now have to add a branch to it.
    // - Before
    // try
    //   ...
    //   (falls through to 'cont')
    // catch
    //   handler body
    // end
    //               <-- cont
    //
    // - After
    // try
    //   ...
    //   br %cont    (new)
    // catch
    // end
    // handler body
    //               <-- cont
    MachineBasicBlock *EHPadLayoutPred = &*std::prev(EHPad->getIterator());
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
    SmallVector<MachineOperand, 4> Cond;
    bool Analyzable = !TII.analyzeBranch(*EHPadLayoutPred, TBB, FBB, Cond);
    if (Analyzable && !TBB && !FBB) {
      DebugLoc DL = EHPadLayoutPred->empty()
                        ? DebugLoc()
                        : EHPadLayoutPred->rbegin()->getDebugLoc();
      BuildMI(EHPadLayoutPred, DL, TII.get(WebAssembly::BR)).addMBB(Cont);
    }
  }

  // For possibly throwing calls whose unwind destinations are currently
  // incorrect because of CFG linearization, we wrap them with a nested
  // try/catch/end_try, and within the new catch block, we branch to the correct
  // handler.
  // - Before
  // mbb:
  //   call @foo       <- Unwind destination mismatch!
  // ehpad:
  //   ...
  //
  // - After
  // mbb:
  //   try                (new)
  //   call @foo
  // nested-ehpad:        (new)
  //   catch              (new)
  //   local.set n / drop (new)
  //   br %brdest         (new)
  // nested-end:          (new)
  //   end_try            (new)
  // ehpad:
  //   ...
  for (auto &P : BrDestToTryRanges) {
    MachineBasicBlock *BrDest = P.first;
    auto &TryRanges = P.second;
    unsigned ExnReg = BrDestToExnReg[BrDest];

    for (auto Range : TryRanges) {
      MachineInstr *RangeBegin = nullptr, *RangeEnd = nullptr;
      std::tie(RangeBegin, RangeEnd) = Range;
      auto *MBB = RangeBegin->getParent();

      // Include possible EH_LABELs in the range
      if (RangeBegin->getIterator() != MBB->begin() &&
          std::prev(RangeBegin->getIterator())->isEHLabel())
        RangeBegin = &*std::prev(RangeBegin->getIterator());
      if (std::next(RangeEnd->getIterator()) != MBB->end() &&
          std::next(RangeEnd->getIterator())->isEHLabel())
        RangeEnd = &*std::next(RangeEnd->getIterator());

      MachineBasicBlock *EHPad = nullptr;
      for (auto *Succ : MBB->successors()) {
        if (Succ->isEHPad()) {
          EHPad = Succ;
          break;
        }
      }

      // Create the nested try instruction.
      MachineInstr *NestedTry =
          BuildMI(*MBB, *RangeBegin, RangeBegin->getDebugLoc(),
                  TII.get(WebAssembly::TRY))
              .addImm(int64_t(WebAssembly::ExprType::Void));

      // Create the nested EH pad and fill instructions in.
      MachineBasicBlock *NestedEHPad = MF.CreateMachineBasicBlock();
      MF.insert(std::next(MBB->getIterator()), NestedEHPad);
      NestedEHPad->setIsEHPad();
      NestedEHPad->setIsEHScopeEntry();
      BuildMI(NestedEHPad, RangeEnd->getDebugLoc(), TII.get(WebAssembly::CATCH),
              ExnReg);
      BuildMI(NestedEHPad, RangeEnd->getDebugLoc(), TII.get(WebAssembly::BR))
          .addMBB(BrDest);

      // Create the nested continuation BB and end_try instruction.
      MachineBasicBlock *NestedCont = MF.CreateMachineBasicBlock();
      MF.insert(std::next(NestedEHPad->getIterator()), NestedCont);
      MachineInstr *NestedEndTry =
          BuildMI(*NestedCont, NestedCont->begin(), RangeEnd->getDebugLoc(),
                  TII.get(WebAssembly::END_TRY));
      // In case MBB has more instructions after the try range, move them to the
      // new nested continuation BB.
      NestedCont->splice(NestedCont->end(), MBB,
                         std::next(RangeEnd->getIterator()), MBB->end());
      registerTryScope(NestedTry, NestedEndTry, NestedEHPad);

      // Fix predecessor-successor relationship.
      NestedCont->transferSuccessors(MBB);
      if (EHPad)
        NestedCont->removeSuccessor(EHPad);
      MBB->addSuccessor(NestedEHPad);
      MBB->addSuccessor(NestedCont);
      NestedEHPad->addSuccessor(BrDest);
    }
  }

  // Renumber BBs and recalculate ScopeTop info because new BBs might have been
  // created and inserted above.
  MF.RenumberBlocks();
  ScopeTops.clear();
  ScopeTops.resize(MF.getNumBlockIDs());
  for (auto &MBB : reverse(MF)) {
    for (auto &MI : reverse(MBB)) {
      if (ScopeTops[MBB.getNumber()])
        break;
      switch (MI.getOpcode()) {
      case WebAssembly::END_BLOCK:
      case WebAssembly::END_LOOP:
      case WebAssembly::END_TRY:
        ScopeTops[MBB.getNumber()] = EndToBegin[&MI]->getParent();
        break;
      case WebAssembly::CATCH:
        ScopeTops[MBB.getNumber()] = EHPadToTry[&MBB]->getParent();
        break;
      }
    }
  }

  // Recompute the dominator tree.
  getAnalysis<MachineDominatorTree>().runOnMachineFunction(MF);

  // Place block markers for newly added branches.
  SmallVector <MachineBasicBlock *, 8> BrDests;
  for (auto &P : BrDestToTryRanges)
    BrDests.push_back(P.first);
  llvm::sort(BrDests,
             [&](const MachineBasicBlock *A, const MachineBasicBlock *B) {
               auto ANum = A->getNumber();
               auto BNum = B->getNumber();
               return ANum < BNum;
             });
  for (auto *Dest : BrDests)
    placeBlockMarker(*Dest);

  return true;
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
  assert(MFI.getResults().size() <= 1);

  if (MFI.getResults().empty())
    return;

  WebAssembly::ExprType RetType;
  switch (MFI.getResults().front().SimpleTy) {
  case MVT::i32:
    RetType = WebAssembly::ExprType::I32;
    break;
  case MVT::i64:
    RetType = WebAssembly::ExprType::I64;
    break;
  case MVT::f32:
    RetType = WebAssembly::ExprType::F32;
    break;
  case MVT::f64:
    RetType = WebAssembly::ExprType::F64;
    break;
  case MVT::v16i8:
  case MVT::v8i16:
  case MVT::v4i32:
  case MVT::v2i64:
  case MVT::v4f32:
  case MVT::v2f64:
    RetType = WebAssembly::ExprType::V128;
    break;
  case MVT::exnref:
    RetType = WebAssembly::ExprType::Exnref;
    break;
  default:
    llvm_unreachable("unexpected return type");
  }

  for (MachineBasicBlock &MBB : reverse(MF)) {
    for (MachineInstr &MI : reverse(MBB)) {
      if (MI.isPosition() || MI.isDebugInstr())
        continue;
      if (MI.getOpcode() == WebAssembly::END_BLOCK) {
        EndToBegin[&MI]->getOperand(0).setImm(int32_t(RetType));
        continue;
      }
      if (MI.getOpcode() == WebAssembly::END_LOOP) {
        EndToBegin[&MI]->getOperand(0).setImm(int32_t(RetType));
        continue;
      }
      // Something other than an `end`. We're done.
      return;
    }
  }
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
