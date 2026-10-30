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
#include "AnalysisOptions.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GraphWriter.h"

using namespace llvm;

static void writeCFGToDotFile(const AnalysisOptions &Opts, Function &F,
                              BlockFrequencyInfo *BFI,
                              BranchProbabilityInfo *BPI, uint64_t MaxFreq,
                              bool CFGOnly = false) {
  std::string Filename =
      (Opts.cfg_dot_filename_prefix + "." + F.getName() + ".dot").str();
  errs() << "Writing '" << Filename << "'...";

  std::error_code EC;
  raw_fd_ostream File(Filename, EC, sys::fs::OF_Text);

  DOTFuncInfo CFGInfo(&F, BFI, BPI, MaxFreq);
  CFGInfo.setHeatColors(Opts.cfg_heat_colors);
  CFGInfo.setEdgeWeights(Opts.cfg_weights);
  CFGInfo.setRawEdgeWeights(Opts.cfg_raw_weights);

  if (!EC)
    WriteGraph(File, &CFGInfo, CFGOnly);
  else
    errs() << "  error opening file for writing!";
  errs() << "\n";
}

static void viewCFG(const AnalysisOptions &Opts, Function &F,
                    const BlockFrequencyInfo *BFI,
                    const BranchProbabilityInfo *BPI, uint64_t MaxFreq,
                    bool CFGOnly = false) {
  DOTFuncInfo CFGInfo(&F, BFI, BPI, MaxFreq);
  CFGInfo.setHeatColors(Opts.cfg_heat_colors);
  CFGInfo.setEdgeWeights(Opts.cfg_weights);
  CFGInfo.setRawEdgeWeights(Opts.cfg_raw_weights);

  ViewGraph(&CFGInfo, "cfg." + F.getName(), CFGOnly);
}

DOTFuncInfo::DOTFuncInfo(const Function *F, const BlockFrequencyInfo *BFI,
                         const BranchProbabilityInfo *BPI, uint64_t MaxFreq,
                         std::optional<NodeIdFormatterTy> NodeIdFormatter)
    : F(F), BFI(BFI), BPI(BPI), MaxFreq(MaxFreq),
      NodeIdFormatter(NodeIdFormatter) {
  ShowHeat = false;
  EdgeWeights = !!BPI; // Print EdgeWeights when BPI is available.
  RawWeights = !!BFI;  // Print RawWeights when BFI is available.
}

DOTFuncInfo::~DOTFuncInfo() = default;

ModuleSlotTracker *DOTFuncInfo::getModuleSlotTracker() {
  if (!MSTStorage)
    MSTStorage = std::make_unique<ModuleSlotTracker>(F->getParent());
  return &*MSTStorage;
}

PreservedAnalyses CFGViewerPass::run(Function &F, FunctionAnalysisManager &AM) {
  const AnalysisOptions &Opts = getAnalysisOptions(F);
  if (!Opts.cfg_func_name.empty() && !F.getName().contains(Opts.cfg_func_name))
    return PreservedAnalyses::all();
  auto *BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  auto *BPI = &AM.getResult<BranchProbabilityAnalysis>(F);
  viewCFG(Opts, F, BFI, BPI, getMaxFreq(F, BFI));
  return PreservedAnalyses::all();
}

PreservedAnalyses CFGOnlyViewerPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  const AnalysisOptions &Opts = getAnalysisOptions(F);
  if (!Opts.cfg_func_name.empty() && !F.getName().contains(Opts.cfg_func_name))
    return PreservedAnalyses::all();
  auto *BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  auto *BPI = &AM.getResult<BranchProbabilityAnalysis>(F);
  viewCFG(Opts, F, BFI, BPI, getMaxFreq(F, BFI), /*CFGOnly=*/true);
  return PreservedAnalyses::all();
}

PreservedAnalyses CFGPrinterPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  const AnalysisOptions &Opts = getAnalysisOptions(F);
  if (!Opts.cfg_func_name.empty() && !F.getName().contains(Opts.cfg_func_name))
    return PreservedAnalyses::all();
  auto *BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  auto *BPI = &AM.getResult<BranchProbabilityAnalysis>(F);
  writeCFGToDotFile(Opts, F, BFI, BPI, getMaxFreq(F, BFI));
  return PreservedAnalyses::all();
}

PreservedAnalyses CFGOnlyPrinterPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  const AnalysisOptions &Opts = getAnalysisOptions(F);
  if (!Opts.cfg_func_name.empty() && !F.getName().contains(Opts.cfg_func_name))
    return PreservedAnalyses::all();
  auto *BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  auto *BPI = &AM.getResult<BranchProbabilityAnalysis>(F);
  writeCFGToDotFile(Opts, F, BFI, BPI, getMaxFreq(F, BFI), /*CFGOnly=*/true);
  return PreservedAnalyses::all();
}

/// viewCFG - This function is meant for use from the debugger.  You can just
/// say 'call F->viewCFG()' and a ghostview window should pop up from the
/// program, displaying the CFG of the current function.  This depends on there
/// being a 'dot' and 'gv' program in your path.
///
void Function::viewCFG() const { viewCFG(false, nullptr, nullptr); }

void Function::viewCFG(const char *OutputFileName) const {
  viewCFG(false, nullptr, nullptr, OutputFileName);
}

void Function::viewCFG(bool ViewCFGOnly, const BlockFrequencyInfo *BFI,
                       const BranchProbabilityInfo *BPI,
                       const char *OutputFileName) const {
  StringRef CFGFuncName = getAnalysisOptions(*this).cfg_func_name;
  if (!CFGFuncName.empty() && !getName().contains(CFGFuncName))
    return;
  DOTFuncInfo CFGInfo(this, BFI, BPI, BFI ? getMaxFreq(*this, BFI) : 0);
  ViewGraph(&CFGInfo, OutputFileName ? OutputFileName : "cfg" + getName(),
            ViewCFGOnly);
}

/// viewCFGOnly - This function is meant for use from the debugger.  It works
/// just like viewCFG, but it does not include the contents of basic blocks
/// into the nodes, just the label.  If you are only interested in the CFG
/// this can make the graph smaller.
///
void Function::viewCFGOnly() const { viewCFGOnly(nullptr, nullptr); }

void Function::viewCFGOnly(const char *OutputFileName) const {
  viewCFG(true, nullptr, nullptr, OutputFileName);
}

void Function::viewCFGOnly(const BlockFrequencyInfo *BFI,
                           const BranchProbabilityInfo *BPI) const {
  viewCFG(true, BFI, BPI);
}

/// Find all blocks on the paths which terminate with a deoptimize or 
/// unreachable (i.e. all blocks which are post-dominated by a deoptimize 
/// or unreachable). These paths are hidden if the corresponding cl::opts
/// are enabled.
void DOTGraphTraits<DOTFuncInfo *>::computeDeoptOrUnreachablePaths(
    const Function *F) {
  const AnalysisOptions &Opts = getAnalysisOptions(*F);
  auto evaluateBB = [&](const BasicBlock *Node) {
    if (succ_empty(Node)) {
      const Instruction *TI = Node->getTerminator();
      isOnDeoptOrUnreachablePath[Node] =
          (Opts.cfg_hide_unreachable_paths && isa<UnreachableInst>(TI)) ||
          (Opts.cfg_hide_deoptimize_paths &&
           Node->getTerminatingDeoptimizeCall());
      return;
    }
    isOnDeoptOrUnreachablePath[Node] =
        llvm::all_of(successors(Node), [this](const BasicBlock *BB) {
          return isOnDeoptOrUnreachablePath[BB];
        });
  };
  /// The post order traversal iteration is done to know the status of
  /// isOnDeoptOrUnreachablePath for all the successors on the current BB.
  llvm::for_each(post_order(&F->getEntryBlock()), evaluateBB);
}

bool DOTGraphTraits<DOTFuncInfo *>::isNodeHidden(const BasicBlock *Node,
                                                 const DOTFuncInfo *CFGInfo) {
  const AnalysisOptions &Opts = getAnalysisOptions(*CFGInfo->getFunction());
  if (Opts.cfg_hide_cold_paths)
    if (auto *BFI = CFGInfo->getBFI()) {
      BlockFrequency NodeFreq = BFI->getBlockFreq(Node);
      BlockFrequency EntryFreq = BFI->getEntryFreq();
      // Hide blocks with relative frequency below -cfg-hide-cold-paths.
      if ((double)NodeFreq.getFrequency() / EntryFreq.getFrequency() <
          *Opts.cfg_hide_cold_paths)
        return true;
    }
  if (Opts.cfg_hide_unreachable_paths || Opts.cfg_hide_deoptimize_paths) {
    if (!isOnDeoptOrUnreachablePath.contains(Node))
      computeDeoptOrUnreachablePaths(Node->getParent());
    return isOnDeoptOrUnreachablePath[Node];
  }
  return false;
}

std::string DOTGraphTraits<DOTFuncInfo *>::getCompleteNodeLabel(
    const BasicBlock *Node, DOTFuncInfo *CFGInfo,
    function_ref<void(raw_string_ostream &, const BasicBlock &)>
        HandleBasicBlock,
    function_ref<void(std::string &, unsigned &, unsigned)> HandleComment) {
  if (HandleBasicBlock)
    return CompleteNodeLabelString(Node, HandleBasicBlock, HandleComment);

  // Default basic block printing
  std::optional<ModuleSlotTracker> MSTStorage;
  ModuleSlotTracker *MST = nullptr;

  if (CFGInfo) {
    MST = CFGInfo->getModuleSlotTracker();
  } else {
    MSTStorage.emplace(Node->getModule());
    MST = &*MSTStorage;
  }

  return CompleteNodeLabelString(
      Node,
      function_ref<void(raw_string_ostream &, const BasicBlock &)>(
          [MST](raw_string_ostream &OS, const BasicBlock &Node) -> void {
            // Prepend label name
            Node.printAsOperand(OS, false, *MST);
            OS << ":\n";

            for (const Instruction &Inst : Node) {
              Inst.print(OS, *MST, /* IsForDebug */ false);
              OS << '\n';
            }
          }),
      HandleComment);
}
