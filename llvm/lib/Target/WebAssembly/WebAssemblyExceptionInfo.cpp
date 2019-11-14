//===--- WebAssemblyExceptionInfo.cpp - Exception Infomation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements WebAssemblyException information analysis.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyExceptionInfo.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyUtilities.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineDominanceFrontier.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-exception-info"

char WebAssemblyExceptionInfo::ID = 0;

INITIALIZE_PASS_BEGIN(WebAssemblyExceptionInfo, DEBUG_TYPE,
                      "WebAssembly Exception Information", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineDominanceFrontier)
INITIALIZE_PASS_END(WebAssemblyExceptionInfo, DEBUG_TYPE,
                    "WebAssembly Exception Information", true, true)

bool WebAssemblyExceptionInfo::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** Exception Info Calculation **********\n"
                       "********** Function: "
                    << MF.getName() << '\n');
  releaseMemory();
  auto &MDT = getAnalysis<MachineDominatorTree>();
  auto &MDF = getAnalysis<MachineDominanceFrontier>();
  recalculate(MDT, MDF);
  return false;
}

void WebAssemblyExceptionInfo::recalculate(
    MachineDominatorTree &MDT, const MachineDominanceFrontier &MDF) {
  // Postorder traversal of the dominator tree.
  SmallVector<WebAssemblyException *, 8> Exceptions;
  for (auto DomNode : post_order(&MDT)) {
    MachineBasicBlock *EHPad = DomNode->getBlock();
    if (!EHPad->isEHPad())
      continue;
    auto *WE = new WebAssemblyException(EHPad);
    discoverAndMapException(WE, MDT, MDF);
    Exceptions.push_back(WE);
  }

  // Add BBs to exceptions
  for (auto DomNode : post_order(&MDT)) {
    MachineBasicBlock *MBB = DomNode->getBlock();
    WebAssemblyException *WE = getExceptionFor(MBB);
    for (; WE; WE = WE->getParentException())
      WE->addBlock(MBB);
  }

  // Add subexceptions to exceptions
  for (auto *WE : Exceptions) {
    if (WE->getParentException())
      WE->getParentException()->getSubExceptions().push_back(WE);
    else
      addTopLevelException(WE);
  }

  // For convenience, Blocks and SubExceptions are inserted in postorder.
  // Reverse the lists.
  for (auto *WE : Exceptions) {
    WE->reverseBlock();
    std::reverse(WE->getSubExceptions().begin(), WE->getSubExceptions().end());
  }
}

void WebAssemblyExceptionInfo::releaseMemory() {
  BBMap.clear();
  DeleteContainerPointers(TopLevelExceptions);
  TopLevelExceptions.clear();
}

void WebAssemblyExceptionInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineDominatorTree>();
  AU.addRequired<MachineDominanceFrontier>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void WebAssemblyExceptionInfo::discoverAndMapException(
    WebAssemblyException *WE, const MachineDominatorTree &MDT,
    const MachineDominanceFrontier &MDF) {
  unsigned NumBlocks = 0;
  unsigned NumSubExceptions = 0;

  // Map blocks that belong to a catchpad / cleanuppad
  MachineBasicBlock *EHPad = WE->getEHPad();
  SmallVector<MachineBasicBlock *, 8> WL;
  WL.push_back(EHPad);
  while (!WL.empty()) {
    MachineBasicBlock *MBB = WL.pop_back_val();

    // Find its outermost discovered exception. If this is a discovered block,
    // check if it is already discovered to be a subexception of this exception.
    WebAssemblyException *SubE = getOutermostException(MBB);
    if (SubE) {
      if (SubE != WE) {
        // Discover a subexception of this exception.
        SubE->setParentException(WE);
        ++NumSubExceptions;
        NumBlocks += SubE->getBlocksVector().capacity();
        // All blocks that belong to this subexception have been already
        // discovered. Skip all of them. Add the subexception's landing pad's
        // dominance frontier to the worklist.
        for (auto &Frontier : MDF.find(SubE->getEHPad())->second)
          if (MDT.dominates(EHPad, Frontier))
            WL.push_back(Frontier);
      }
      continue;
    }

    // This is an undiscovered block. Map it to the current exception.
    changeExceptionFor(MBB, WE);
    ++NumBlocks;

    // Add successors dominated by the current BB to the worklist.
    for (auto *Succ : MBB->successors())
      if (MDT.dominates(EHPad, Succ))
        WL.push_back(Succ);
  }

  WE->getSubExceptions().reserve(NumSubExceptions);
  WE->reserveBlocks(NumBlocks);
}

WebAssemblyException *
WebAssemblyExceptionInfo::getOutermostException(MachineBasicBlock *MBB) const {
  WebAssemblyException *WE = getExceptionFor(MBB);
  if (WE) {
    while (WebAssemblyException *Parent = WE->getParentException())
      WE = Parent;
  }
  return WE;
}

void WebAssemblyException::print(raw_ostream &OS, unsigned Depth) const {
  OS.indent(Depth * 2) << "Exception at depth " << getExceptionDepth()
                       << " containing: ";

  for (unsigned I = 0; I < getBlocks().size(); ++I) {
    MachineBasicBlock *MBB = getBlocks()[I];
    if (I)
      OS << ", ";
    OS << "%bb." << MBB->getNumber();
    if (const auto *BB = MBB->getBasicBlock())
      if (BB->hasName())
        OS << "." << BB->getName();

    if (getEHPad() == MBB)
      OS << " (landing-pad)";
  }
  OS << "\n";

  for (auto &SubE : SubExceptions)
    SubE->print(OS, Depth + 2);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void WebAssemblyException::dump() const { print(dbgs()); }
#endif

raw_ostream &operator<<(raw_ostream &OS, const WebAssemblyException &WE) {
  WE.print(OS);
  return OS;
}

void WebAssemblyExceptionInfo::print(raw_ostream &OS, const Module *) const {
  for (auto *WE : TopLevelExceptions)
    WE->print(OS);
}
