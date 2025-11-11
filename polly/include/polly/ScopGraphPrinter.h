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

namespace llvm {

template <>
struct GraphTraits<polly::ScopDetection *> : GraphTraits<RegionInfo *> {
  static NodeRef getEntryNode(polly::ScopDetection *SD) {
    return GraphTraits<RegionInfo *>::getEntryNode(SD->getRI());
  }
  static nodes_iterator nodes_begin(polly::ScopDetection *SD) {
    return nodes_iterator::begin(getEntryNode(SD));
  }
  static nodes_iterator nodes_end(polly::ScopDetection *SD) {
    return nodes_iterator::end(getEntryNode(SD));
  }
};

template <>
struct DOTGraphTraits<polly::ScopDetection *> : DOTGraphTraits<RegionNode *> {
  DOTGraphTraits(bool isSimple = false)
      : DOTGraphTraits<RegionNode *>(isSimple) {}
  static std::string getGraphName(polly::ScopDetection *SD) {
    return "Scop Graph";
  }

  std::string getEdgeAttributes(RegionNode *srcNode,
                                GraphTraits<RegionInfo *>::ChildIteratorType CI,
                                polly::ScopDetection *SD);

  std::string getNodeLabel(RegionNode *Node, polly::ScopDetection *SD) {
    return DOTGraphTraits<RegionNode *>::getNodeLabel(
        Node, reinterpret_cast<RegionNode *>(SD->getRI()->getTopLevelRegion()));
  }

  static std::string escapeString(llvm::StringRef String);

  /// Print the cluster of the subregions. This groups the single basic blocks
  /// and adds a different background color for each group.
  static void printRegionCluster(polly::ScopDetection *SD, const Region *R,
                                 raw_ostream &O, unsigned depth = 0);

  static void addCustomGraphFeatures(polly::ScopDetection *SD,
                                     GraphWriter<polly::ScopDetection *> &GW);
};
} // end namespace llvm

namespace polly {

struct ScopViewer final : llvm::DOTGraphTraitsViewer<ScopAnalysis, false> {
  ScopViewer() : llvm::DOTGraphTraitsViewer<ScopAnalysis, false>("scops") {}

  bool processFunction(Function &F, const ScopDetection &SD) override;
};

struct ScopOnlyViewer final : llvm::DOTGraphTraitsViewer<ScopAnalysis, false> {
  ScopOnlyViewer()
      : llvm::DOTGraphTraitsViewer<ScopAnalysis, false>("scops-only") {}
};

struct ScopPrinter final : llvm::DOTGraphTraitsPrinter<ScopAnalysis, false> {
  ScopPrinter() : llvm::DOTGraphTraitsPrinter<ScopAnalysis, false>("scops") {}
};

struct ScopOnlyPrinter final : llvm::DOTGraphTraitsPrinter<ScopAnalysis, true> {
  ScopOnlyPrinter()
      : llvm::DOTGraphTraitsPrinter<ScopAnalysis, true>("scopsonly") {}
};

} // end namespace polly

#endif /* POLLY_SCOP_GRAPH_PRINTER_H */
