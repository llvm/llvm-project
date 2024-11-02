//===- GraphPrinter.cpp - Create a DOT output describing the Scop. --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Create a DOT output describing the Scop.
//
// For each function a dot file is created that shows the control flow graph of
// the function and highlights the detected Scops.
//
//===----------------------------------------------------------------------===//

#include "polly/ScopGraphPrinter.h"
#include "polly/LinkAllPasses.h"
#include "polly/ScopDetection.h"
#include "llvm/Support/CommandLine.h"

using namespace polly;
using namespace llvm;
static cl::opt<std::string>
    ViewFilter("polly-view-only",
               cl::desc("Only view functions that match this pattern"),
               cl::Hidden, cl::init(""), cl::ZeroOrMore);

static cl::opt<bool> ViewAll("polly-view-all",
                             cl::desc("Also show functions without any scops"),
                             cl::Hidden, cl::init(false), cl::ZeroOrMore);

namespace llvm {

std::string DOTGraphTraits<ScopDetection *>::getEdgeAttributes(
    RegionNode *srcNode, GraphTraits<RegionInfo *>::ChildIteratorType CI,
    ScopDetection *SD) {
  RegionNode *destNode = *CI;

  if (srcNode->isSubRegion() || destNode->isSubRegion())
    return "";

  // In case of a backedge, do not use it to define the layout of the nodes.
  BasicBlock *srcBB = srcNode->getNodeAs<BasicBlock>();
  BasicBlock *destBB = destNode->getNodeAs<BasicBlock>();

  RegionInfo *RI = SD->getRI();
  Region *R = RI->getRegionFor(destBB);

  while (R && R->getParent())
    if (R->getParent()->getEntry() == destBB)
      R = R->getParent();
    else
      break;

  if (R && R->getEntry() == destBB && R->contains(srcBB))
    return "constraint=false";

  return "";
}

std::string
DOTGraphTraits<ScopDetection *>::escapeString(llvm::StringRef String) {
  std::string Escaped;

  for (const auto &C : String) {
    if (C == '"')
      Escaped += '\\';

    Escaped += C;
  }
  return Escaped;
}

void DOTGraphTraits<ScopDetection *>::printRegionCluster(ScopDetection *SD,
                                                         const Region *R,
                                                         raw_ostream &O,
                                                         unsigned depth) {
  O.indent(2 * depth) << "subgraph cluster_" << static_cast<const void *>(R)
                      << " {\n";
  unsigned LineBegin, LineEnd;
  std::string FileName;

  getDebugLocation(R, LineBegin, LineEnd, FileName);

  std::string Location;
  if (LineBegin != (unsigned)-1) {
    Location = escapeString(FileName + ":" + std::to_string(LineBegin) + "-" +
                            std::to_string(LineEnd) + "\n");
  }

  std::string ErrorMessage = SD->regionIsInvalidBecause(R);
  ErrorMessage = escapeString(ErrorMessage);
  O.indent(2 * (depth + 1))
      << "label = \"" << Location << ErrorMessage << "\";\n";

  if (SD->isMaxRegionInScop(*R)) {
    O.indent(2 * (depth + 1)) << "style = filled;\n";

    // Set color to green.
    O.indent(2 * (depth + 1)) << "color = 3";
  } else {
    O.indent(2 * (depth + 1)) << "style = solid;\n";

    int color = (R->getDepth() * 2 % 12) + 1;

    // We do not want green again.
    if (color == 3)
      color = 6;

    O.indent(2 * (depth + 1)) << "color = " << color << "\n";
  }

  for (const auto &SubRegion : *R)
    printRegionCluster(SD, SubRegion.get(), O, depth + 1);

  RegionInfo *RI = R->getRegionInfo();

  for (BasicBlock *BB : R->blocks())
    if (RI->getRegionFor(BB) == R)
      O.indent(2 * (depth + 1))
          << "Node"
          << static_cast<void *>(RI->getTopLevelRegion()->getBBNode(BB))
          << ";\n";

  O.indent(2 * depth) << "}\n";
}

void DOTGraphTraits<ScopDetection *>::addCustomGraphFeatures(
    ScopDetection *SD, GraphWriter<ScopDetection *> &GW) {
  raw_ostream &O = GW.getOStream();
  O << "\tcolorscheme = \"paired12\"\n";
  printRegionCluster(SD, SD->getRI()->getTopLevelRegion(), O, 4);
}

} // namespace llvm

struct ScopDetectionAnalysisGraphTraits {
  static ScopDetection *getGraph(ScopDetectionWrapperPass *Analysis) {
    return &Analysis->getSD();
  }
};

struct ScopViewerWrapperPass
    : DOTGraphTraitsViewerWrapperPass<ScopDetectionWrapperPass, false,
                                      ScopDetection *,
                                      ScopDetectionAnalysisGraphTraits> {
  static char ID;
  ScopViewerWrapperPass()
      : DOTGraphTraitsViewerWrapperPass<ScopDetectionWrapperPass, false,
                                        ScopDetection *,
                                        ScopDetectionAnalysisGraphTraits>(
            "scops", ID) {}
  bool processFunction(Function &F, ScopDetectionWrapperPass &SD) override {
    if (ViewFilter != "" && !F.getName().count(ViewFilter))
      return false;

    if (ViewAll)
      return true;

    // Check that at least one scop was detected.
    return std::distance(SD.getSD().begin(), SD.getSD().end()) > 0;
  }
};
char ScopViewerWrapperPass::ID = 0;

struct ScopOnlyViewerWrapperPass
    : DOTGraphTraitsViewerWrapperPass<ScopDetectionWrapperPass, false,
                                      ScopDetection *,
                                      ScopDetectionAnalysisGraphTraits> {
  static char ID;
  ScopOnlyViewerWrapperPass()
      : DOTGraphTraitsViewerWrapperPass<ScopDetectionWrapperPass, false,
                                        ScopDetection *,
                                        ScopDetectionAnalysisGraphTraits>(
            "scopsonly", ID) {}
};
char ScopOnlyViewerWrapperPass::ID = 0;

struct ScopPrinterWrapperPass
    : DOTGraphTraitsPrinterWrapperPass<ScopDetectionWrapperPass, false,
                                       ScopDetection *,
                                       ScopDetectionAnalysisGraphTraits> {
  static char ID;
  ScopPrinterWrapperPass()
      : DOTGraphTraitsPrinterWrapperPass<ScopDetectionWrapperPass, false,
                                         ScopDetection *,
                                         ScopDetectionAnalysisGraphTraits>(
            "scops", ID) {}
};
char ScopPrinterWrapperPass::ID = 0;

struct ScopOnlyPrinterWrapperPass
    : DOTGraphTraitsPrinterWrapperPass<ScopDetectionWrapperPass, true,
                                       ScopDetection *,
                                       ScopDetectionAnalysisGraphTraits> {
  static char ID;
  ScopOnlyPrinterWrapperPass()
      : DOTGraphTraitsPrinterWrapperPass<ScopDetectionWrapperPass, true,
                                         ScopDetection *,
                                         ScopDetectionAnalysisGraphTraits>(
            "scopsonly", ID) {}
};
char ScopOnlyPrinterWrapperPass::ID = 0;

static RegisterPass<ScopViewerWrapperPass> X("view-scops",
                                             "Polly - View Scops of function");

static RegisterPass<ScopOnlyViewerWrapperPass>
    Y("view-scops-only",
      "Polly - View Scops of function (with no function bodies)");

static RegisterPass<ScopPrinterWrapperPass>
    M("dot-scops", "Polly - Print Scops of function");

static RegisterPass<ScopOnlyPrinterWrapperPass>
    N("dot-scops-only",
      "Polly - Print Scops of function (with no function bodies)");

Pass *polly::createDOTViewerWrapperPass() {
  return new ScopViewerWrapperPass();
}

Pass *polly::createDOTOnlyViewerWrapperPass() {
  return new ScopOnlyViewerWrapperPass();
}

Pass *polly::createDOTPrinterWrapperPass() {
  return new ScopPrinterWrapperPass();
}

Pass *polly::createDOTOnlyPrinterWrapperPass() {
  return new ScopOnlyPrinterWrapperPass();
}

bool ScopViewer::processFunction(Function &F, const ScopDetection &SD) {
  if (ViewFilter != "" && !F.getName().count(ViewFilter))
    return false;

  if (ViewAll)
    return true;

  // Check that at least one scop was detected.
  return std::distance(SD.begin(), SD.end()) > 0;
}
