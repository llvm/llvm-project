//===- GraphPrinter.cpp - Create a DOT output describing the Scop. --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Create a DOT output describing the Scop.
//
// For each function a dot file is created that shows the control flow graph of
// the function and highlights the detected Scops.
//
//===----------------------------------------------------------------------===//

#include "polly/LinkAllPasses.h"
#include "polly/ScopDetection.h"
#include "llvm/Analysis/DOTGraphTraitsPass.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"

using namespace polly;
using namespace llvm;

namespace llvm {
template <>
struct GraphTraits<ScopDetection *> : public GraphTraits<RegionInfo *> {

  static NodeType *getEntryNode(ScopDetection *SD) {
    return GraphTraits<RegionInfo *>::getEntryNode(SD->getRI());
  }
  static nodes_iterator nodes_begin(ScopDetection *SD) {
    return nodes_iterator::begin(getEntryNode(SD));
  }
  static nodes_iterator nodes_end(ScopDetection *SD) {
    return nodes_iterator::end(getEntryNode(SD));
  }
};

template <> struct DOTGraphTraits<RegionNode *> : public DefaultDOTGraphTraits {

  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  std::string getNodeLabel(RegionNode *Node, RegionNode *Graph) {

    if (!Node->isSubRegion()) {
      BasicBlock *BB = Node->getNodeAs<BasicBlock>();

      if (isSimple())
        return DOTGraphTraits<const Function *>::getSimpleNodeLabel(
            BB, BB->getParent());
      else
        return DOTGraphTraits<const Function *>::getCompleteNodeLabel(
            BB, BB->getParent());
    }

    return "Not implemented";
  }
};

template <>
struct DOTGraphTraits<ScopDetection *> : public DOTGraphTraits<RegionNode *> {
  DOTGraphTraits(bool isSimple = false)
      : DOTGraphTraits<RegionNode *>(isSimple) {}
  static std::string getGraphName(ScopDetection *SD) { return "Scop Graph"; }

  std::string getEdgeAttributes(RegionNode *srcNode,
                                GraphTraits<RegionInfo *>::ChildIteratorType CI,
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

    if (R->getEntry() == destBB && R->contains(srcBB))
      return "constraint=false";

    return "";
  }

  std::string getNodeLabel(RegionNode *Node, ScopDetection *SD) {
    return DOTGraphTraits<RegionNode *>::getNodeLabel(
        Node, SD->getRI()->getTopLevelRegion());
  }

  static std::string escapeString(std::string String) {
    std::string Escaped;

    for (std::string::iterator SI = String.begin(), SE = String.end(); SI != SE;
         ++SI) {

      if (*SI == '"')
        Escaped += '\\';

      Escaped += *SI;
    }
    return Escaped;
  }

  // Print the cluster of the subregions. This groups the single basic blocks
  // and adds a different background color for each group.
  static void printRegionCluster(const ScopDetection *SD, const Region *R,
                                 raw_ostream &O, unsigned depth = 0) {
    O.indent(2 * depth) << "subgraph cluster_" << static_cast<const void *>(R)
                        << " {\n";
    std::string ErrorMessage = SD->regionIsInvalidBecause(R);
    ErrorMessage = escapeString(ErrorMessage);
    O.indent(2 * (depth + 1)) << "label = \"" << ErrorMessage << "\";\n";

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

    for (Region::const_iterator RI = R->begin(), RE = R->end(); RI != RE; ++RI)
      printRegionCluster(SD, *RI, O, depth + 1);

    RegionInfo *RI = R->getRegionInfo();

    for (Region::const_block_iterator BI = R->block_begin(),
                                      BE = R->block_end();
         BI != BE; ++BI)
      if (RI->getRegionFor(*BI) == R)
        O.indent(2 * (depth + 1))
            << "Node" << static_cast<const void *>(
                             RI->getTopLevelRegion()->getBBNode(*BI)) << ";\n";

    O.indent(2 * depth) << "}\n";
  }
  static void addCustomGraphFeatures(const ScopDetection *SD,
                                     GraphWriter<ScopDetection *> &GW) {
    raw_ostream &O = GW.getOStream();
    O << "\tcolorscheme = \"paired12\"\n";
    printRegionCluster(SD, SD->getRI()->getTopLevelRegion(), O, 4);
  }
};

} //end namespace llvm

struct ScopViewer : public DOTGraphTraitsViewer<ScopDetection, false> {
  static char ID;
  ScopViewer() : DOTGraphTraitsViewer<ScopDetection, false>("scops", ID) {}
};
char ScopViewer::ID = 0;

struct ScopOnlyViewer : public DOTGraphTraitsViewer<ScopDetection, true> {
  static char ID;
  ScopOnlyViewer()
      : DOTGraphTraitsViewer<ScopDetection, true>("scopsonly", ID) {}
};
char ScopOnlyViewer::ID = 0;

struct ScopPrinter : public DOTGraphTraitsPrinter<ScopDetection, false> {
  static char ID;
  ScopPrinter() : DOTGraphTraitsPrinter<ScopDetection, false>("scops", ID) {}
};
char ScopPrinter::ID = 0;

struct ScopOnlyPrinter : public DOTGraphTraitsPrinter<ScopDetection, true> {
  static char ID;
  ScopOnlyPrinter()
      : DOTGraphTraitsPrinter<ScopDetection, true>("scopsonly", ID) {}
};
char ScopOnlyPrinter::ID = 0;

static RegisterPass<ScopViewer> X("view-scops",
                                  "Polly - View Scops of function");

static RegisterPass<ScopOnlyViewer>
Y("view-scops-only",
  "Polly - View Scops of function (with no function bodies)");

static RegisterPass<ScopPrinter> M("dot-scops",
                                   "Polly - Print Scops of function");

static RegisterPass<ScopOnlyPrinter>
N("dot-scops-only",
  "Polly - Print Scops of function (with no function bodies)");

Pass *polly::createDOTViewerPass() { return new ScopViewer(); }

Pass *polly::createDOTOnlyViewerPass() { return new ScopOnlyViewer(); }

Pass *polly::createDOTPrinterPass() { return new ScopPrinter(); }

Pass *polly::createDOTOnlyPrinterPass() { return new ScopOnlyPrinter(); }
