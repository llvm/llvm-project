//===- GraphPrinter.h - Create a DOT output describing the Scop. ----------===//
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

#ifndef POLLY_SCOP_GRAPH_PRINTER_H
#define POLLY_SCOP_GRAPH_PRINTER_H

#include "polly/ScopDetection.h"
#include "polly/Support/ScopLocation.h"
#include "llvm/Analysis/DOTGraphTraitsPass.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/RegionPrinter.h"
#include "llvm/IR/PassManager.h"

using namespace polly;
using namespace llvm;

namespace llvm {

template <>
struct GraphTraits<ScopDetection *> : public GraphTraits<RegionInfo *> {
  static NodeRef getEntryNode(ScopDetection *SD) {
    return GraphTraits<RegionInfo *>::getEntryNode(SD->getRI());
  }
  static nodes_iterator nodes_begin(ScopDetection *SD) {
    return nodes_iterator::begin(getEntryNode(SD));
  }
  static nodes_iterator nodes_end(ScopDetection *SD) {
    return nodes_iterator::end(getEntryNode(SD));
  }
};

template <>
struct DOTGraphTraits<ScopDetection *> : public DOTGraphTraits<RegionNode *> {
  DOTGraphTraits(bool isSimple = false)
      : DOTGraphTraits<RegionNode *>(isSimple) {}
  static std::string getGraphName(ScopDetection *SD) { return "Scop Graph"; }

  std::string getEdgeAttributes(RegionNode *srcNode,
                                GraphTraits<RegionInfo *>::ChildIteratorType CI,
                                ScopDetection *SD);

  std::string getNodeLabel(RegionNode *Node, ScopDetection *SD) {
    return DOTGraphTraits<RegionNode *>::getNodeLabel(
        Node, reinterpret_cast<RegionNode *>(SD->getRI()->getTopLevelRegion()));
  }

  static std::string escapeString(llvm::StringRef String);

  /// Print the cluster of the subregions. This groups the single basic blocks
  /// and adds a different background color for each group.
  static void printRegionCluster(ScopDetection *SD, const Region *R,
                                 raw_ostream &O, unsigned depth = 0);

  static void addCustomGraphFeatures(ScopDetection *SD,
                                     GraphWriter<ScopDetection *> &GW);
};
} // end namespace llvm

namespace polly {

struct ScopViewer : public DOTGraphTraitsViewer<ScopAnalysis, false> {
  ScopViewer() : DOTGraphTraitsViewer<ScopAnalysis, false>("scops") {}

  bool processFunction(Function &F, const ScopDetection &SD) override;
};

struct ScopOnlyViewer : public DOTGraphTraitsViewer<ScopAnalysis, false> {
  ScopOnlyViewer() : DOTGraphTraitsViewer<ScopAnalysis, false>("scops-only") {}
};

struct ScopPrinter : public DOTGraphTraitsPrinter<ScopAnalysis, false> {
  ScopPrinter() : DOTGraphTraitsPrinter<ScopAnalysis, false>("scops") {}
};

struct ScopOnlyPrinter : public DOTGraphTraitsPrinter<ScopAnalysis, true> {
  ScopOnlyPrinter() : DOTGraphTraitsPrinter<ScopAnalysis, true>("scopsonly") {}
};

} // end namespace polly

#endif /* POLLY_SCOP_GRAPH_PRINTER_H */
