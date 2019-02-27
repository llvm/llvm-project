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

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblyExceptionInfo.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/WasmEHFuncInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
using namespace llvm;

#define DEBUG_TYPE "wasm-cfg-stackify"

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

  void placeMarkers(MachineFunction &MF);
  void placeBlockMarker(MachineBasicBlock &MBB);
  void placeLoopMarker(MachineBasicBlock &MBB);
  void placeTryMarker(MachineBasicBlock &MBB);
  void removeUnnecessaryInstrs(MachineFunction &MF);
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
                "Insert BLOCK and LOOP markers for WebAssembly scopes", false,
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
void WebAssemblyCFGStackify::placeBlockMarker(MachineBasicBlock &MBB) {
  // This should have been handled in placeTryMarker.
  if (MBB.isEHPad())
    return;

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
  MachineBasicBlock *LayoutPred = &*std::prev(MachineFunction::iterator(&MBB));

  // If the nearest common dominator is inside a more deeply nested context,
  // walk out to the nearest scope which isn't more deeply nested.
  for (MachineFunction::iterator I(LayoutPred), E(Header); I != E; --I) {
    if (MachineBasicBlock *ScopeTop = ScopeTops[I->getNumber()]) {
      if (ScopeTop->getNumber() > Header->getNumber()) {
        // Skip over an intervening scope.
        I = std::next(MachineFunction::iterator(ScopeTop));
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
    // If there is a previously placed LOOP/TRY marker and the bottom block of
    // the loop/exception is above MBB, it should be after the BLOCK, because
    // the loop/exception is nested in this block. Otherwise it should be before
    // the BLOCK.
    if (MI.getOpcode() == WebAssembly::LOOP ||
        MI.getOpcode() == WebAssembly::TRY) {
      auto *BottomBB =
          &*std::prev(MachineFunction::iterator(BeginToEnd[&MI]->getParent()));
      if (MBB.getNumber() > BottomBB->getNumber())
        AfterSet.insert(&MI);
#ifndef NDEBUG
      else
        BeforeSet.insert(&MI);
#endif
    }

    // All previously inserted BLOCK markers should be after the BLOCK because
    // they are all nested blocks.
    if (MI.getOpcode() == WebAssembly::BLOCK)
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

  // 'br_on_exn' extracts except_ref object and pushes variable number of values
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
  auto Iter = std::next(MachineFunction::iterator(Bottom));
  if (Iter == MF.end()) {
    MachineBasicBlock *Label = MF.CreateMachineBasicBlock();
    // Give it a fake predecessor so that AsmPrinter prints its label.
    Label->addSuccessor(Label);
    MF.push_back(Label);
    Iter = std::next(MachineFunction::iterator(Bottom));
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
  DebugLoc EndDL = (*AfterLoop->pred_rbegin())->findBranchDebugLoc();
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
  if (!MBB.isEHPad())
    return;

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

  auto Iter = std::next(MachineFunction::iterator(Bottom));
  if (Iter == MF.end()) {
    MachineBasicBlock *Label = MF.CreateMachineBasicBlock();
    // Give it a fake predecessor so that AsmPrinter prints its label.
    Label->addSuccessor(Label);
    MF.push_back(Label);
    Iter = std::next(MachineFunction::iterator(Bottom));
  }
  MachineBasicBlock *Cont = &*Iter;

  assert(Cont != &MF.front());
  MachineBasicBlock *LayoutPred =
      &*std::prev(MachineFunction::iterator(Cont));

  // If the nearest common dominator is inside a more deeply nested context,
  // walk out to the nearest scope which isn't more deeply nested.
  for (MachineFunction::iterator I(LayoutPred), E(Header); I != E; --I) {
    if (MachineBasicBlock *ScopeTop = ScopeTops[I->getNumber()]) {
      if (ScopeTop->getNumber() > Header->getNumber()) {
        // Skip over an intervening scope.
        I = std::next(MachineFunction::iterator(ScopeTop));
      } else {
        // We found a scope level at an appropriate depth.
        Header = ScopeTop;
        break;
      }
    }
  }

  // Decide where in Header to put the TRY.

  // Instructions that should go before the BLOCK.
  SmallPtrSet<const MachineInstr *, 4> BeforeSet;
  // Instructions that should go after the BLOCK.
  SmallPtrSet<const MachineInstr *, 4> AfterSet;
  for (const auto &MI : *Header) {
    // If there is a previously placed LOOP marker and the bottom block of
    // the loop is above MBB, the LOOP should be after the TRY, because the
    // loop is nested in this try. Otherwise it should be before the TRY.
    if (MI.getOpcode() == WebAssembly::LOOP) {
      if (MBB.getNumber() > Bottom->getNumber())
        AfterSet.insert(&MI);
#ifndef NDEBUG
      else
        BeforeSet.insert(&MI);
#endif
    }

    // All previously inserted TRY markers should be after the TRY because they
    // are all nested trys.
    if (MI.getOpcode() == WebAssembly::TRY)
      AfterSet.insert(&MI);

#ifndef NDEBUG
    // All END_(LOOP/TRY) markers should be before the TRY.
    if (MI.getOpcode() == WebAssembly::END_LOOP ||
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
    // END_TRY should precede existing LOOP markers.
    if (MI.getOpcode() == WebAssembly::LOOP)
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
      if (EndToBegin[&MI]->getParent()->getNumber() >= Header->getNumber())
        BeforeSet.insert(&MI);
#ifndef NDEBUG
      else
        AfterSet.insert(&MI);
#endif
    }
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
    MachineBasicBlock *EHPadLayoutPred =
        &*std::prev(MachineFunction::iterator(&MBB));
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
      for (auto B = MachineBasicBlock::iterator(Try),
                E = std::next(MachineBasicBlock::iterator(EndTry));
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
  case MVT::ExceptRef:
    RetType = WebAssembly::ExprType::ExceptRef;
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
  // Place the TRY for MBB if MBB is the EH pad of an exception.
  const MCAsmInfo *MCAI = MF.getTarget().getMCAsmInfo();
  if (MCAI->getExceptionHandlingType() == ExceptionHandling::Wasm &&
      MF.getFunction().hasPersonalityFn())
    for (auto &MBB : MF)
      placeTryMarker(MBB);
  // Place the BLOCK for MBB if MBB is branched to from above.
  for (auto &MBB : MF)
    placeBlockMarker(MBB);
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

  if (MCAI->getExceptionHandlingType() == ExceptionHandling::Wasm &&
      MF.getFunction().hasPersonalityFn())
    // Remove unnecessary instructions.
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

  return true;
}
