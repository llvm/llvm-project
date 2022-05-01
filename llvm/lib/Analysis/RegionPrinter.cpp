//===- RegionPrinter.cpp - Print regions tree pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Print out the region tree of a function using dotty/graphviz.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/RegionPrinter.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/DOTGraphTraitsPass.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#ifndef NDEBUG
#include "llvm/IR/LegacyPassManager.h"
#endif

using namespace llvm;

//===----------------------------------------------------------------------===//
/// onlySimpleRegion - Show only the simple regions in the RegionViewer.
static cl::opt<bool>
onlySimpleRegions("only-simple-regions",
                  cl::desc("Show only simple regions in the graphviz viewer"),
                  cl::Hidden,
                  cl::init(false));

namespace {
struct HighlightingRegionInfo {
  RegionInfo *RI;
  const Function *F;
  const BasicBlock *HighlightBB;
  const Instruction *HighlightInst;

  HighlightingRegionInfo() = delete;

  HighlightingRegionInfo(RegionInfo *RI, const Function *F,
                         const BasicBlock *HighlightBB = nullptr,
                         const Instruction *HighlightInst = nullptr)
      : RI(RI), F(F), HighlightBB(HighlightBB), HighlightInst(HighlightInst) {}

public:
  RegionInfo *getRegionInfo() const { return RI; }
  const Function *getFunction() const { return F; }
};

#if 0
    struct HighlightingRegionInfoPassGraphTraits {
        static HighlightingRegionInfo* getGraph(RegionInfoPass* RIP) {
            // ...
            return nullptr;
        }
    };
#endif
} // namespace

namespace llvm {
template <>
struct GraphTraits<HighlightingRegionInfo *>
    : public GraphTraits<FlatIt<RegionNode *>> {
  using nodes_iterator = df_iterator<NodeRef, df_iterator_default_set<NodeRef>,
                                     false, GraphTraits<FlatIt<NodeRef>>>;

  static NodeRef getEntryNode(HighlightingRegionInfo *G) {
    return GraphTraits<FlatIt<Region *>>::getEntryNode(
        G->RI->getTopLevelRegion());
  }

  static nodes_iterator nodes_begin(HighlightingRegionInfo *G) {
    return nodes_iterator::begin(getEntryNode(G));
  }

  static nodes_iterator nodes_end(HighlightingRegionInfo *G) {
    return nodes_iterator::end(getEntryNode(G));
  }
};

template<>
struct DOTGraphTraits<RegionNode*> : public DefaultDOTGraphTraits {

  DOTGraphTraits (bool isSimple=false)
    : DefaultDOTGraphTraits(isSimple) {}

  std::string getNodeLabel(RegionNode *Node, RegionNode *Graph) {

    if (!Node->isSubRegion()) {
      BasicBlock *BB = Node->getNodeAs<BasicBlock>();

      if (isSimple())
        return DOTGraphTraits<DOTFuncInfo *>
          ::getSimpleNodeLabel(BB, nullptr);
      else
        return DOTGraphTraits<DOTFuncInfo *>
          ::getCompleteNodeLabel(BB, nullptr);
    }

    return "Not implemented";
  }
};

template <>
struct DOTGraphTraits<RegionInfo *> : public DOTGraphTraits<RegionNode *> {

  DOTGraphTraits (bool isSimple = false)
    : DOTGraphTraits<RegionNode*>(isSimple) {}

  static std::string getGraphName(const RegionInfo *) { return "Region Graph"; }

  std::string getNodeLabel(RegionNode *Node, RegionInfo *G) {
    return DOTGraphTraits<RegionNode *>::getNodeLabel(
        Node, reinterpret_cast<RegionNode *>(G->getTopLevelRegion()));
  }

  std::string getEdgeAttributes(RegionNode *srcNode,
                                GraphTraits<RegionInfo *>::ChildIteratorType CI,
                                RegionInfo *G) {
    RegionNode *destNode = *CI;

    if (srcNode->isSubRegion() || destNode->isSubRegion())
      return "";

    // In case of a backedge, do not use it to define the layout of the nodes.
    BasicBlock *srcBB = srcNode->getNodeAs<BasicBlock>();
    BasicBlock *destBB = destNode->getNodeAs<BasicBlock>();

    Region *R = G->getRegionFor(destBB);

    while (R && R->getParent())
      if (R->getParent()->getEntry() == destBB)
        R = R->getParent();
      else
        break;

    if (R && R->getEntry() == destBB && R->contains(srcBB))
      return "constraint=false";

    return "";
  }

  // Print the cluster of the subregions. This groups the single basic blocks
  // and adds a different background color for each group.
  static void printRegionCluster(const Region &R, GraphWriter<RegionInfo *> &GW,
                                 unsigned depth = 0) {
    raw_ostream &O = GW.getOStream();
    O.indent(2 * depth) << "subgraph cluster_" << static_cast<const void*>(&R)
      << " {\n";
    O.indent(2 * (depth + 1)) << "label = \"\";\n";

    if (!onlySimpleRegions || R.isSimple()) {
      O.indent(2 * (depth + 1)) << "style = filled;\n";
      O.indent(2 * (depth + 1)) << "color = "
        << ((R.getDepth() * 2 % 12) + 1) << "\n";

    } else {
      O.indent(2 * (depth + 1)) << "style = solid;\n";
      O.indent(2 * (depth + 1)) << "color = "
        << ((R.getDepth() * 2 % 12) + 2) << "\n";
    }

    for (const auto &RI : R)
      printRegionCluster(*RI, GW, depth + 1);

    const RegionInfo &RI = *static_cast<const RegionInfo*>(R.getRegionInfo());

    for (auto *BB : R.blocks())
      if (RI.getRegionFor(BB) == &R)
        O.indent(2 * (depth + 1)) << "Node"
          << static_cast<const void*>(RI.getTopLevelRegion()->getBBNode(BB))
          << ";\n";

    O.indent(2 * depth) << "}\n";
  }

  static void addCustomGraphFeatures(const RegionInfo *G,
                                     GraphWriter<RegionInfo *> &GW) {
    raw_ostream &O = GW.getOStream();
    O << "\tcolorscheme = \"paired12\"\n";
    printRegionCluster(*G->getTopLevelRegion(), GW, 4);
  }
};

} //end namespace llvm

namespace {
struct RegionInfoPassGraphTraits {
  static RegionInfo *getGraph(RegionInfoPass *RIP) {
    return &RIP->getRegionInfo();
  }
};

} // namespace

namespace llvm {

#if 0
    template <>
    struct GraphTraits<HighlightingRegionInfo*> : public GraphTraits<Region*> {
        using Base = GraphTraits<Region*>;
        //  using Base::NodeRef;
    };
#endif

template <>
struct DOTGraphTraits<HighlightingRegionInfo *>
    : public DOTGraphTraits<RegionNode *> {
  using Base = DOTGraphTraits<RegionNode *>;
  using Traits = GraphTraits<HighlightingRegionInfo *>;

  DOTGraphTraits(bool IsSimple = false) : Base(IsSimple) {}

#if 0
        static std::string getGraphName(const HighlightingRegionInfo *G) { 
            return Base::getGraphName(G->RI);
        }

        std::string getNodeLabel(RegionNode *Node, HighlightingRegionInfo *G) {
            return Base::getNodeLabel(Node, G->RI);
        }

        std::string getEdgeAttributes(
            RegionNode *SrcNode,
            Traits::ChildIteratorType CI,
            HighlightingRegionInfo *G) {
            return Base::getEdgeAttributes(SrcNode, CI, G->RI);
        }
#endif

  static std::string getGraphName(const HighlightingRegionInfo *) {
    return "Region Graph";
  }

  std::string getNodeLabel(RegionNode *Node, HighlightingRegionInfo *G) {
    // return Base::getNodeLabel(Node, reinterpret_cast<RegionNode
    // *>(G->RI->getTopLevelRegion()));

    if (!Node->isSubRegion()) {
      BasicBlock *BB = Node->getNodeAs<BasicBlock>();

      DOTFuncInfo CFGInfo(G->F, G->HighlightBB, G->HighlightInst);
      if (isSimple())
        return DOTGraphTraits<DOTFuncInfo *>::getSimpleNodeLabel(BB, &CFGInfo);
      else
        return DOTGraphTraits<DOTFuncInfo *>::getCompleteNodeLabel(BB,
                                                                   &CFGInfo);
    }

    return "Not implemented";
  }

  static std::string getNodeAttributes(RegionNode *R,
                                       HighlightingRegionInfo *G) {
    auto HighlightBB = G->HighlightBB;
    if (!R->isSubRegion() && R->getNodeAs<BasicBlock>() == HighlightBB) {
      return "penwidth=5.0,style=filled";
    }

    return "";
  }

  std::string getEdgeAttributes(RegionNode *srcNode,
                                Traits::ChildIteratorType CI,
                                HighlightingRegionInfo *G) {
    RegionNode *destNode = *CI;

    if (srcNode->isSubRegion() || destNode->isSubRegion())
      return "";

    // In case of a backedge, do not use it to define the layout of the nodes.
    BasicBlock *srcBB = srcNode->getNodeAs<BasicBlock>();
    BasicBlock *destBB = destNode->getNodeAs<BasicBlock>();

    Region *R = G->RI->getRegionFor(destBB);

    while (R && R->getParent())
      if (R->getParent()->getEntry() == destBB)
        R = R->getParent();
      else
        break;

    if (R && R->getEntry() == destBB && R->contains(srcBB))
      return "constraint=false";

    return "";
  }

  static void printRegionCluster(const Region &R,
                                 GraphWriter<HighlightingRegionInfo *> &GW,
                                 unsigned depth = 0,
                                 const BasicBlock *HighlightBB = nullptr,
                                 const Instruction *HighlightInst = nullptr) {
    raw_ostream &O = GW.getOStream();
    O.indent(2 * depth) << "subgraph cluster_" << static_cast<const void *>(&R)
                        << " {\n";
    O.indent(2 * (depth + 1)) << "label = \"\";\n";

    if (!onlySimpleRegions || R.isSimple()) {
      O.indent(2 * (depth + 1)) << "style = filled;\n";
      O.indent(2 * (depth + 1))
          << "color = " << ((R.getDepth() * 2 % 12) + 1) << "\n";

    } else {
      O.indent(2 * (depth + 1)) << "style = solid;\n";
      O.indent(2 * (depth + 1))
          << "color = " << ((R.getDepth() * 2 % 12) + 2) << "\n";
    }

    for (const auto &RI : R)
      printRegionCluster(*RI, GW, depth + 1, HighlightBB, HighlightInst);

    const RegionInfo &RI = *static_cast<const RegionInfo *>(R.getRegionInfo());

    for (auto *BB : R.blocks())
      if (RI.getRegionFor(BB) == &R)
        O.indent(2 * (depth + 1))
            << "Node"
            << static_cast<const void *>(RI.getTopLevelRegion()->getBBNode(BB))
            << ";\n";

    O.indent(2 * depth) << "}\n";
  }

  static void
  addCustomGraphFeatures(const HighlightingRegionInfo *G,
                         GraphWriter<HighlightingRegionInfo *> &GW) {
    raw_ostream &O = GW.getOStream();
    O << "\tcolorscheme = \"paired12\"\n";
    printRegionCluster(*G->RI->getTopLevelRegion(), GW, 4, G->HighlightBB,
                       G->HighlightInst);
  }
};
} // namespace llvm

namespace {
struct RegionPrinter
    : public DOTGraphTraitsPrinter<RegionInfoPass, false, RegionInfo *,
                                   RegionInfoPassGraphTraits> {
  static char ID;
  RegionPrinter()
      : DOTGraphTraitsPrinter<RegionInfoPass, false, RegionInfo *,
                              RegionInfoPassGraphTraits>("reg", ID) {
    initializeRegionPrinterPass(*PassRegistry::getPassRegistry());
  }
};
char RegionPrinter::ID = 0;

struct RegionOnlyPrinter
    : public DOTGraphTraitsPrinter<RegionInfoPass, true, RegionInfo *,
                                   RegionInfoPassGraphTraits> {
  static char ID;
  RegionOnlyPrinter()
      : DOTGraphTraitsPrinter<RegionInfoPass, true, RegionInfo *,
                              RegionInfoPassGraphTraits>("reg", ID) {
    initializeRegionOnlyPrinterPass(*PassRegistry::getPassRegistry());
  }
};
char RegionOnlyPrinter::ID = 0;

struct RegionViewer : public FunctionPass {
  using Base = FunctionPass;

  static char ID;
  RegionViewer() : RegionViewer(nullptr, nullptr) {}

  RegionViewer(const BasicBlock *HighlightBB, const Instruction *HighlightInst)
      : FunctionPass(ID), HighlightBB(HighlightBB),
        HighlightInst(HighlightInst) {
    initializeRegionViewerPass(*PassRegistry::getPassRegistry());
  }

  virtual bool processFunction(Function &F, RegionInfo &Analysis) {
    return true;
  }

  bool runOnFunction(Function &F) override {
    auto &Analysis = getAnalysis<RegionInfoPass>().getRegionInfo();

    if (!processFunction(F, Analysis))
      return false;

    HighlightingRegionInfo Graph(&Analysis, &F, HighlightBB, HighlightInst);
    ViewGraph(&Graph, "reg", false,
              Twine("Region Graph for '") + F.getName().str() + "' function");

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<RegionInfoPass>();
  }

private:
  const BasicBlock *HighlightBB;
  const Instruction *HighlightInst;
};
char RegionViewer::ID = 0;

struct RegionOnlyViewer
    : public DOTGraphTraitsViewer<RegionInfoPass, true, RegionInfo *,
                                  RegionInfoPassGraphTraits> {
  static char ID;
  RegionOnlyViewer()
      : DOTGraphTraitsViewer<RegionInfoPass, true, RegionInfo *,
                             RegionInfoPassGraphTraits>("regonly", ID) {
    initializeRegionOnlyViewerPass(*PassRegistry::getPassRegistry());
  }
};
char RegionOnlyViewer::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(RegionPrinter, "dot-regions",
                "Print regions of function to 'dot' file", true, true)

INITIALIZE_PASS(
    RegionOnlyPrinter, "dot-regions-only",
    "Print regions of function to 'dot' file (with no function bodies)", true,
    true)

INITIALIZE_PASS(RegionViewer, "view-regions", "View regions of function",
                true, true)

INITIALIZE_PASS(RegionOnlyViewer, "view-regions-only",
                "View regions of function (with no function bodies)",
                true, true)

FunctionPass *llvm::createRegionPrinterPass() { return new RegionPrinter(); }

FunctionPass *llvm::createRegionOnlyPrinterPass() {
  return new RegionOnlyPrinter();
}

FunctionPass* llvm::createRegionViewerPass() {
  return new RegionViewer();
}

FunctionPass *llvm::createRegionViewerPass(const BasicBlock *BB,
                                           const Instruction *Inst) {
  return new RegionViewer(BB, Inst);
}

FunctionPass* llvm::createRegionOnlyViewerPass() {
  return new RegionOnlyViewer();
}

#ifndef NDEBUG
static void viewRegionInfo(RegionInfo *RI, bool ShortNames) {
  assert(RI && "Argument must be non-null");

  llvm::Function *F = RI->getTopLevelRegion()->getEntry()->getParent();
  std::string GraphName = DOTGraphTraits<RegionInfo *>::getGraphName(RI);

  llvm::ViewGraph(RI, "reg", ShortNames,
                  Twine(GraphName) + " for '" + F->getName() + "' function");
}

static void invokeFunctionPass(const Function *F, FunctionPass *ViewerPass) {
  assert(F && "Argument must be non-null");
  assert(!F->isDeclaration() && "Function must have an implementation");

  // The viewer and analysis passes do not modify anything, so we can safely
  // remove the const qualifier
  auto NonConstF = const_cast<Function *>(F);

  llvm::legacy::FunctionPassManager FPM(NonConstF->getParent());
  FPM.add(ViewerPass);
  FPM.doInitialization();
  FPM.run(*NonConstF);
  FPM.doFinalization();
}

void llvm::viewRegion(RegionInfo *RI) {
  if (!RI)
    return;
  viewRegionInfo(RI, false);
}
void llvm::viewRegion(RegionInfo &RI) { return viewRegion(&RI); }

void llvm::viewRegion(const Function *F) {
  if (!F)
    return;
  invokeFunctionPass(F, createRegionViewerPass());
}
void llvm::viewRegion(const Function &F) { viewRegion(&F); }

void llvm::viewRegionOnly(RegionInfo *RI) { viewRegionInfo(RI, true); }

void llvm::viewRegionOnly(const Function *F) {
  if (!F)
    return;
  invokeFunctionPass(F, createRegionOnlyViewerPass());
}

void llvm::viewRegion(const llvm::BasicBlock *BB) {
  if (!BB)
    return;

  auto F = BB->getParent();

  invokeFunctionPass(F, createRegionViewerPass(BB, nullptr));
}
void llvm::viewRegion(const llvm::BasicBlock &BB) { return viewRegion(&BB); }

void llvm::viewRegion(const llvm::Instruction *Inst) {
  if (!Inst)
    return;

  auto Block = Inst->getParent();
  auto F = Inst->getFunction();

  invokeFunctionPass(F, createRegionViewerPass(Block, Inst));
}

void llvm::viewRegion(const llvm::Instruction &I) { return viewRegion(&I); }

#endif
