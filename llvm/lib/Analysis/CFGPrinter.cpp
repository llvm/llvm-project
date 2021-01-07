//===- CFGPrinter.cpp - DOT printer for the control flow graph ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a `-dot-cfg` analysis pass, which emits the
// `<prefix>.<fnname>.dot` file for each function in the program, with a graph
// of the CFG for that function. The default value for `<prefix>` is `cfg` but
// can be customized as needed.
//
// The other main feature of this file is that it implements the
// Function::viewCFG method, which is useful for debugging passes which operate
// on the CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include <algorithm>

using namespace llvm;

static cl::opt<std::string>
    CFGFuncName("cfg-func-name", cl::Hidden,
                cl::desc("The name of a function (or its substring)"
                         " whose CFG is viewed/printed."));

static cl::opt<std::string> CFGDotFilenamePrefix(
    "cfg-dot-filename-prefix", cl::Hidden,
    cl::desc("The prefix used for the CFG dot file names."));

static cl::opt<bool> HideUnreachablePaths("cfg-hide-unreachable-paths",
                                          cl::init(false));

static cl::opt<bool> HideDeoptimizePaths("cfg-hide-deoptimize-paths",
                                         cl::init(false));

static cl::opt<bool> ShowHeatColors("cfg-heat-colors", cl::init(true),
                                    cl::Hidden,
                                    cl::desc("Show heat colors in CFG"));

static cl::opt<bool> UseRawEdgeWeight("cfg-raw-weights", cl::init(false),
                                      cl::Hidden,
                                      cl::desc("Use raw weights for labels. "
                                               "Use percentages as default."));

static cl::opt<bool>
    ShowEdgeWeight("cfg-weights", cl::init(false), cl::Hidden,
                   cl::desc("Show edges labeled with weights"));

static void writeCFGToDotFile(Function &F, BlockFrequencyInfo *BFI,
                              BranchProbabilityInfo *BPI, uint64_t MaxFreq,
                              bool CFGOnly = false) {
  std::string Filename =
      (CFGDotFilenamePrefix + "." + F.getName() + ".dot").str();
  errs() << "Writing '" << Filename << "'...";

  std::error_code EC;
  raw_fd_ostream File(Filename, EC, sys::fs::F_Text);

  DOTFuncInfo CFGInfo(&F, BFI, BPI, MaxFreq);
  CFGInfo.setHeatColors(ShowHeatColors);
  CFGInfo.setEdgeWeights(ShowEdgeWeight);
  CFGInfo.setRawEdgeWeights(UseRawEdgeWeight);

  if (!EC)
    WriteGraph(File, &CFGInfo, CFGOnly);
  else
    errs() << "  error opening file for writing!";
  errs() << "\n";
}

static void viewCFG(Function &F, const BlockFrequencyInfo *BFI,
                    const BranchProbabilityInfo *BPI, uint64_t MaxFreq,
                    bool CFGOnly = false) {
  DOTFuncInfo CFGInfo(&F, BFI, BPI, MaxFreq);
  CFGInfo.setHeatColors(ShowHeatColors);
  CFGInfo.setEdgeWeights(ShowEdgeWeight);
  CFGInfo.setRawEdgeWeights(UseRawEdgeWeight);

  ViewGraph(&CFGInfo, "cfg." + F.getName(), CFGOnly);
}

namespace {
struct CFGViewerLegacyPass : public FunctionPass {
  static char ID; // Pass identifcation, replacement for typeid
  CFGViewerLegacyPass() : FunctionPass(ID) {
    initializeCFGViewerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto *BPI = &getAnalysis<BranchProbabilityInfoWrapperPass>().getBPI();
    auto *BFI = &getAnalysis<BlockFrequencyInfoWrapperPass>().getBFI();
    viewCFG(F, BFI, BPI, getMaxFreq(F, BFI));
    return false;
  }

  void print(raw_ostream &OS, const Module * = nullptr) const override {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<BranchProbabilityInfoWrapperPass>();
    AU.setPreservesAll();
  }
};
}

char CFGViewerLegacyPass::ID = 0;
INITIALIZE_PASS(CFGViewerLegacyPass, "view-cfg", "View CFG of function", false,
                true)

PreservedAnalyses CFGViewerPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto *BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  auto *BPI = &AM.getResult<BranchProbabilityAnalysis>(F);
  viewCFG(F, BFI, BPI, getMaxFreq(F, BFI));
  return PreservedAnalyses::all();
}

namespace {
struct CFGOnlyViewerLegacyPass : public FunctionPass {
  static char ID; // Pass identifcation, replacement for typeid
  CFGOnlyViewerLegacyPass() : FunctionPass(ID) {
    initializeCFGOnlyViewerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto *BPI = &getAnalysis<BranchProbabilityInfoWrapperPass>().getBPI();
    auto *BFI = &getAnalysis<BlockFrequencyInfoWrapperPass>().getBFI();
    viewCFG(F, BFI, BPI, getMaxFreq(F, BFI), /*CFGOnly=*/true);
    return false;
  }

  void print(raw_ostream &OS, const Module * = nullptr) const override {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<BranchProbabilityInfoWrapperPass>();
    AU.setPreservesAll();
  }
};
}

char CFGOnlyViewerLegacyPass::ID = 0;
INITIALIZE_PASS(CFGOnlyViewerLegacyPass, "view-cfg-only",
                "View CFG of function (with no function bodies)", false, true)

PreservedAnalyses CFGOnlyViewerPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  auto *BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  auto *BPI = &AM.getResult<BranchProbabilityAnalysis>(F);
  viewCFG(F, BFI, BPI, getMaxFreq(F, BFI), /*CFGOnly=*/true);
  return PreservedAnalyses::all();
}

namespace {
struct CFGPrinterLegacyPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  CFGPrinterLegacyPass() : FunctionPass(ID) {
    initializeCFGPrinterLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto *BPI = &getAnalysis<BranchProbabilityInfoWrapperPass>().getBPI();
    auto *BFI = &getAnalysis<BlockFrequencyInfoWrapperPass>().getBFI();
    writeCFGToDotFile(F, BFI, BPI, getMaxFreq(F, BFI));
    return false;
  }

  void print(raw_ostream &OS, const Module * = nullptr) const override {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<BranchProbabilityInfoWrapperPass>();
    AU.setPreservesAll();
  }
};
}

char CFGPrinterLegacyPass::ID = 0;
INITIALIZE_PASS(CFGPrinterLegacyPass, "dot-cfg",
                "Print CFG of function to 'dot' file", false, true)

PreservedAnalyses CFGPrinterPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  auto *BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  auto *BPI = &AM.getResult<BranchProbabilityAnalysis>(F);
  writeCFGToDotFile(F, BFI, BPI, getMaxFreq(F, BFI));
  return PreservedAnalyses::all();
}

namespace {
struct CFGOnlyPrinterLegacyPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  CFGOnlyPrinterLegacyPass() : FunctionPass(ID) {
    initializeCFGOnlyPrinterLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto *BPI = &getAnalysis<BranchProbabilityInfoWrapperPass>().getBPI();
    auto *BFI = &getAnalysis<BlockFrequencyInfoWrapperPass>().getBFI();
    writeCFGToDotFile(F, BFI, BPI, getMaxFreq(F, BFI), /*CFGOnly=*/true);
    return false;
  }
  void print(raw_ostream &OS, const Module * = nullptr) const override {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<BranchProbabilityInfoWrapperPass>();
    AU.setPreservesAll();
  }
};
}

char CFGOnlyPrinterLegacyPass::ID = 0;
INITIALIZE_PASS(CFGOnlyPrinterLegacyPass, "dot-cfg-only",
                "Print CFG of function to 'dot' file (with no function bodies)",
                false, true)

PreservedAnalyses CFGOnlyPrinterPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  auto *BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  auto *BPI = &AM.getResult<BranchProbabilityAnalysis>(F);
  writeCFGToDotFile(F, BFI, BPI, getMaxFreq(F, BFI), /*CFGOnly=*/true);
  return PreservedAnalyses::all();
}

/// viewCFG - This function is meant for use from the debugger.  You can just
/// say 'call F->viewCFG()' and a ghostview window should pop up from the
/// program, displaying the CFG of the current function.  This depends on there
/// being a 'dot' and 'gv' program in your path.
///
void Function::viewCFG() const { viewCFG(false, nullptr, nullptr); }

void Function::viewCFG(bool ViewCFGOnly, const BlockFrequencyInfo *BFI,
                       const BranchProbabilityInfo *BPI) const {
  if (!CFGFuncName.empty() && !getName().contains(CFGFuncName))
    return;
  DOTFuncInfo CFGInfo(this, BFI, BPI, BFI ? getMaxFreq(*this, BFI) : 0);
  ViewGraph(&CFGInfo, "cfg" + getName(), ViewCFGOnly);
}

/// viewCFGOnly - This function is meant for use from the debugger.  It works
/// just like viewCFG, but it does not include the contents of basic blocks
/// into the nodes, just the label.  If you are only interested in the CFG
/// this can make the graph smaller.
///
void Function::viewCFGOnly() const { viewCFGOnly(nullptr, nullptr); }

void Function::viewCFGOnly(const BlockFrequencyInfo *BFI,
                           const BranchProbabilityInfo *BPI) const {
  viewCFG(true, BFI, BPI);
}

FunctionPass *llvm::createCFGPrinterLegacyPassPass() {
  return new CFGPrinterLegacyPass();
}

FunctionPass *llvm::createCFGOnlyPrinterLegacyPassPass() {
  return new CFGOnlyPrinterLegacyPass();
}

void DOTGraphTraits<DOTFuncInfo *>::computeHiddenNodes(const Function *F) {
  auto evaluateBB = [&](const BasicBlock *Node) {
    if (succ_empty(Node)) {
      const Instruction *TI = Node->getTerminator();
      isHiddenBasicBlock[Node] =
          (HideUnreachablePaths && isa<UnreachableInst>(TI)) ||
          (HideDeoptimizePaths && Node->getTerminatingDeoptimizeCall());
      return;
    }
    isHiddenBasicBlock[Node] =
        llvm::all_of(successors(Node), [this](const BasicBlock *BB) {
          return isHiddenBasicBlock[BB];
        });
  };
  /// The post order traversal iteration is done to know the status of
  /// isHiddenBasicBlock for all the successors on the current BB.
  for_each(po_begin(&F->getEntryBlock()), po_end(&F->getEntryBlock()),
           evaluateBB);
}

bool DOTGraphTraits<DOTFuncInfo *>::isNodeHidden(const BasicBlock *Node,
                                                 const DOTFuncInfo *CFGInfo) {
  // If both restricting flags are false, all nodes are displayed.
  if (!HideUnreachablePaths && !HideDeoptimizePaths)
    return false;
  if (isHiddenBasicBlock.find(Node) == isHiddenBasicBlock.end())
    computeHiddenNodes(Node->getParent());
  return isHiddenBasicBlock[Node];
}
