//===-- CFGPrinter.h - CFG printer external interface -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a 'dot-cfg' analysis pass, which emits the
// cfg.<fnname>.dot file for each function in the program, with a graph of the
// CFG for that function.
//
// This file defines external functions that can be called to explicitly
// instantiate the CFG printer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CFGPRINTER_H
#define LLVM_ANALYSIS_CFGPRINTER_H

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/HeatUtils.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/GraphWriter.h"

namespace llvm {
class CFGViewerPass : public PassInfoMixin<CFGViewerPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class CFGOnlyViewerPass : public PassInfoMixin<CFGOnlyViewerPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class CFGPrinterPass : public PassInfoMixin<CFGPrinterPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class CFGOnlyPrinterPass : public PassInfoMixin<CFGOnlyPrinterPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class DOTFuncInfo {
private:
  const Function *F;
  const BlockFrequencyInfo *BFI;
  const BranchProbabilityInfo *BPI;
  uint64_t MaxFreq;
  bool ShowHeat;
  bool EdgeWeights;
  bool RawWeights;

public:
  DOTFuncInfo(const Function *F) : DOTFuncInfo(F, nullptr, nullptr, 0) {}

  DOTFuncInfo(const Function *F, const BlockFrequencyInfo *BFI,
              const BranchProbabilityInfo *BPI, uint64_t MaxFreq)
      : F(F), BFI(BFI), BPI(BPI), MaxFreq(MaxFreq) {
    ShowHeat = false;
    EdgeWeights = !!BPI; // Print EdgeWeights when BPI is available.
    RawWeights = !!BFI;  // Print RawWeights when BFI is available.
  }

  const BlockFrequencyInfo *getBFI() { return BFI; }

  const BranchProbabilityInfo *getBPI() { return BPI; }

  const Function *getFunction() { return this->F; }

  uint64_t getMaxFreq() { return MaxFreq; }

  uint64_t getFreq(const BasicBlock *BB) {
    return BFI->getBlockFreq(BB).getFrequency();
  }

  void setHeatColors(bool ShowHeat) { this->ShowHeat = ShowHeat; }

  bool showHeatColors() { return ShowHeat; }

  void setRawEdgeWeights(bool RawWeights) { this->RawWeights = RawWeights; }

  bool useRawEdgeWeights() { return RawWeights; }

  void setEdgeWeights(bool EdgeWeights) { this->EdgeWeights = EdgeWeights; }

  bool showEdgeWeights() { return EdgeWeights; }
};

template <>
struct GraphTraits<DOTFuncInfo *> : public GraphTraits<const BasicBlock *> {
  static NodeRef getEntryNode(DOTFuncInfo *CFGInfo) {
    return &(CFGInfo->getFunction()->getEntryBlock());
  }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  using nodes_iterator = pointer_iterator<Function::const_iterator>;

  static nodes_iterator nodes_begin(DOTFuncInfo *CFGInfo) {
    return nodes_iterator(CFGInfo->getFunction()->begin());
  }

  static nodes_iterator nodes_end(DOTFuncInfo *CFGInfo) {
    return nodes_iterator(CFGInfo->getFunction()->end());
  }

  static size_t size(DOTFuncInfo *CFGInfo) {
    return CFGInfo->getFunction()->size();
  }
};

template <>
struct DOTGraphTraits<DOTFuncInfo *> : public DefaultDOTGraphTraits {

  // Cache for is hidden property
  llvm::DenseMap<const BasicBlock *, bool> isHiddenBasicBlock;

  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  static std::string getGraphName(DOTFuncInfo *CFGInfo) {
    return "CFG for '" + CFGInfo->getFunction()->getName().str() + "' function";
  }

  static std::string getSimpleNodeLabel(const BasicBlock *Node, DOTFuncInfo *) {
    if (!Node->getName().empty())
      return Node->getName().str();

    std::string Str;
    raw_string_ostream OS(Str);

    Node->printAsOperand(OS, false);
    return OS.str();
  }

  static std::string getCompleteNodeLabel(const BasicBlock *Node,
                                          DOTFuncInfo *) {
    enum { MaxColumns = 80 };
    std::string Str;
    raw_string_ostream OS(Str);

    if (Node->getName().empty()) {
      Node->printAsOperand(OS, false);
      OS << ":";
    }

    OS << *Node;
    std::string OutStr = OS.str();
    if (OutStr[0] == '\n')
      OutStr.erase(OutStr.begin());

    // Process string output to make it nicer...
    unsigned ColNum = 0;
    unsigned LastSpace = 0;
    for (unsigned i = 0; i != OutStr.length(); ++i) {
      if (OutStr[i] == '\n') { // Left justify
        OutStr[i] = '\\';
        OutStr.insert(OutStr.begin() + i + 1, 'l');
        ColNum = 0;
        LastSpace = 0;
      } else if (OutStr[i] == ';') {             // Delete comments!
        unsigned Idx = OutStr.find('\n', i + 1); // Find end of line
        OutStr.erase(OutStr.begin() + i, OutStr.begin() + Idx);
        --i;
      } else if (ColNum == MaxColumns) { // Wrap lines.
        // Wrap very long names even though we can't find a space.
        if (!LastSpace)
          LastSpace = i;
        OutStr.insert(LastSpace, "\\l...");
        ColNum = i - LastSpace;
        LastSpace = 0;
        i += 3; // The loop will advance 'i' again.
      } else
        ++ColNum;
      if (OutStr[i] == ' ')
        LastSpace = i;
    }
    return OutStr;
  }

  std::string getNodeLabel(const BasicBlock *Node, DOTFuncInfo *CFGInfo) {

    if (isSimple())
      return getSimpleNodeLabel(Node, CFGInfo);
    else
      return getCompleteNodeLabel(Node, CFGInfo);
  }

  static std::string getEdgeSourceLabel(const BasicBlock *Node,
                                        const_succ_iterator I) {
    // Label source of conditional branches with "T" or "F"
    if (const BranchInst *BI = dyn_cast<BranchInst>(Node->getTerminator()))
      if (BI->isConditional())
        return (I == succ_begin(Node)) ? "T" : "F";

    // Label source of switch edges with the associated value.
    if (const SwitchInst *SI = dyn_cast<SwitchInst>(Node->getTerminator())) {
      unsigned SuccNo = I.getSuccessorIndex();

      if (SuccNo == 0)
        return "def";

      std::string Str;
      raw_string_ostream OS(Str);
      auto Case = *SwitchInst::ConstCaseIt::fromSuccessorIndex(SI, SuccNo);
      OS << Case.getCaseValue()->getValue();
      return OS.str();
    }
    return "";
  }

  /// Display the raw branch weights from PGO.
  std::string getEdgeAttributes(const BasicBlock *Node, const_succ_iterator I,
                                DOTFuncInfo *CFGInfo) {
    if (!CFGInfo->showEdgeWeights())
      return "";

    const Instruction *TI = Node->getTerminator();
    if (TI->getNumSuccessors() == 1)
      return "penwidth=2";

    unsigned OpNo = I.getSuccessorIndex();

    if (OpNo >= TI->getNumSuccessors())
      return "";

    BasicBlock *SuccBB = TI->getSuccessor(OpNo);
    auto BranchProb = CFGInfo->getBPI()->getEdgeProbability(Node, SuccBB);
    double WeightPercent = ((double)BranchProb.getNumerator()) /
                           ((double)BranchProb.getDenominator());
    double Width = 1 + WeightPercent;

    if (!CFGInfo->useRawEdgeWeights())
      return formatv("label=\"{0:P}\" penwidth={1}", WeightPercent, Width)
          .str();

    // Prepend a 'W' to indicate that this is a weight rather than the actual
    // profile count (due to scaling).

    uint64_t Freq = CFGInfo->getFreq(Node);
    std::string Attrs = formatv("label=\"W:{0}\" penwidth={1}",
                                (uint64_t)(Freq * WeightPercent), Width);
    if (Attrs.size())
      return Attrs;

    MDNode *WeightsNode = TI->getMetadata(LLVMContext::MD_prof);
    if (!WeightsNode)
      return "";

    MDString *MDName = cast<MDString>(WeightsNode->getOperand(0));
    if (MDName->getString() != "branch_weights")
      return "";

    OpNo = I.getSuccessorIndex() + 1;
    if (OpNo >= WeightsNode->getNumOperands())
      return "";
    ConstantInt *Weight =
        mdconst::dyn_extract<ConstantInt>(WeightsNode->getOperand(OpNo));
    if (!Weight)
      return "";
    return ("label=\"W:" + std::to_string(Weight->getZExtValue()) +
            "\" penwidth=" + std::to_string(Width));
  }

  std::string getNodeAttributes(const BasicBlock *Node, DOTFuncInfo *CFGInfo) {

    if (!CFGInfo->showHeatColors())
      return "";

    uint64_t Freq = CFGInfo->getFreq(Node);
    std::string Color = getHeatColor(Freq, CFGInfo->getMaxFreq());
    std::string EdgeColor = (Freq <= (CFGInfo->getMaxFreq() / 2))
                                ? (getHeatColor(0))
                                : (getHeatColor(1));

    std::string Attrs = "color=\"" + EdgeColor + "ff\", style=filled," +
                        " fillcolor=\"" + Color + "70\"";
    return Attrs;
  }
  bool isNodeHidden(const BasicBlock *Node);
  void computeHiddenNodes(const Function *F);
};
} // End llvm namespace

namespace llvm {
class FunctionPass;
FunctionPass *createCFGPrinterLegacyPassPass();
FunctionPass *createCFGOnlyPrinterLegacyPassPass();
} // End llvm namespace

#endif
