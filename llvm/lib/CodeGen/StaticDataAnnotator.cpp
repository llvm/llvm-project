//===- StaticDataAnnotator - Annotate static data's section prefix --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// To reason about module-wide data hotness in a module granularity, this file
// implements a module pass StaticDataAnnotator to work coordinately with the
// StaticDataSplitter pass.
//
// The StaticDataSplitter pass is a machine function pass. It analyzes data
// hotness based on code and adds counters in StaticDataProfileInfo via its
// wrapper pass StaticDataProfileInfoWrapper.
// The StaticDataProfileInfoWrapper sits in the middle between the
// StaticDataSplitter and StaticDataAnnotator passes.
// The StaticDataAnnotator pass is a module pass. It iterates global variables
// in the module, looks up counters from StaticDataProfileInfo and sets the
// section prefix based on profiles.
//
// The three-pass structure is implemented for practical reasons, to work around
// the limitation that a module pass based on legacy pass manager cannot make
// use of MachineBlockFrequencyInfo analysis. In the future, we can consider
// porting the StaticDataSplitter pass to a module-pass using the new pass
// manager framework. That way, analysis are lazily computed as opposed to
// eagerly scheduled, and a module pass can use MachineBlockFrequencyInfo.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/StaticDataProfileInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

#define DEBUG_TYPE "static-data-annotator"

using namespace llvm;

static cl::opt<std::string> GlobalVariableRefGraphDotFile(
    "global-var-ref-graph-dot-file", cl::init(""),
    cl::desc("Global variable reference graph to dot file"));

static cl::opt<bool>
    EnableSCCHotnessPropagation("enable-scc-hotness-propagation",
                                cl::init(true),
                                cl::desc("Enable SCC hotness propagation"));

namespace {

struct RefGraphNode;

struct RefGraphEdge {
  RefGraphEdge(RefGraphNode *Src, RefGraphNode *Dst) : Src(Src), Dst(Dst) {}

  RefGraphNode *Src = nullptr;
  RefGraphNode *Dst = nullptr;

  operator RefGraphNode *() const { return Dst; }
};

struct RefGraphNode {
  RefGraphNode(GlobalVariable *Var, int ID) : Var(Var), VarID(ID) {}

  struct EdgeComparator {
    bool operator()(const RefGraphEdge &L, const RefGraphEdge &R) const {
      return L.Dst->VarID < R.Dst->VarID;
    }
  };

  using Edge = RefGraphEdge;
  using RefEdges = std::set<RefGraphEdge, EdgeComparator>;
  using iterator = RefEdges::iterator;
  using const_iterator = RefEdges::const_iterator;

  GlobalVariable *Var;

  int VarID;

  RefEdges Edges;
};

class RefGraph {
public:
  using iterator = RefGraphNode::iterator;
  RefGraph(Module &M) : M(M) {}

  void buildGraph();

  iterator begin() { return Root->Edges.begin(); }
  iterator end() { return Root->Edges.end(); }

  RefGraphNode *getEntryNode() { return Root.get(); }

  void exportToDot(raw_ostream &OS);

private:
  void addEdge(GlobalVariable *Src, GlobalVariable *Dst);

  RefGraphNode *getOrCreateNode(GlobalVariable *GV);

  void computeGVDependencies(Value *V, SmallPtrSetImpl<GlobalVariable *> &Deps);

  void updateGVDependencies(GlobalVariable *GV);

  DenseMap<GlobalVariable *, std::unique_ptr<RefGraphNode>> GVToNode;

  Module &M;
  std::unique_ptr<RefGraphNode> Root; // A dummy root

  std::unordered_map<Constant *, SmallPtrSet<GlobalVariable *, 4>>
      ConstantDepsCache;
};

void RefGraph::addEdge(GlobalVariable *Src, GlobalVariable *Dst) {
  auto SrcNode = getOrCreateNode(Src);
  auto DstNode = getOrCreateNode(Dst);
  SrcNode->Edges.insert(RefGraphEdge(SrcNode, DstNode));
}

RefGraphNode *RefGraph::getOrCreateNode(GlobalVariable *GV) {
  if (GV == nullptr)
    return Root.get();
  auto &Node = GVToNode[GV];
  if (Node == nullptr)
    Node = std::make_unique<RefGraphNode>(GV, GVToNode.size());
  return Node.get();
}

void RefGraph::computeGVDependencies(Value *V,
                                     SmallPtrSetImpl<GlobalVariable *> &Deps) {
  if (auto *GV = dyn_cast<GlobalVariable>(V)) {
    Deps.insert(GV);
  } else if (auto *CE = dyn_cast<Constant>(V)) {
    auto [Where, Inserted] = ConstantDepsCache.try_emplace(CE);
    auto &LocalDeps = Where->second;
    if (Inserted)
      for (User *CEUser : CE->users())
        computeGVDependencies(CEUser, LocalDeps);
    Deps.insert_range(LocalDeps);
  }
}

void RefGraph::updateGVDependencies(GlobalVariable *GV) {
  SmallPtrSet<GlobalVariable *, 4> Deps;
  for (auto *User : GV->users())
    computeGVDependencies(User, Deps);
  Deps.erase(GV);
  for (GlobalVariable *Dep : Deps)
    addEdge(GV, Dep);
}

void RefGraph::buildGraph() {
  Root = std::make_unique<RefGraphNode>(nullptr, -1);
  for (GlobalVariable &GV : M.globals()) {
    if (GV.isDeclarationForLinker())
      continue;

    updateGVDependencies(&GV);
    addEdge(Root->Var, &GV);
  }
}

void RefGraph::exportToDot(raw_ostream &OS) {
  OS << "digraph {\n";
  SmallVector<RefGraphNode *, 4> Nodes; // Non-owned pointers.
  for (auto Iter = GVToNode.begin(); Iter != GVToNode.end(); Iter++)
    Nodes.push_back(Iter->second.get());
  // Sort nodes for deterministic output in tests.
  std::sort(Nodes.begin(), Nodes.end(),
            [](const RefGraphNode *L, const RefGraphNode *R) {
              return L->VarID < R->VarID;
            });
  SmallVector<std::pair<int, int>> Edges;
  for (auto Iter = GVToNode.begin(); Iter != GVToNode.end(); Iter++) {
    RefGraphNode *Node = Iter->second.get();
    for (const RefGraphEdge &Edge : Node->Edges) {
      // OS << "\t" << Node->VarID << " -> " << Edge.Dst->VarID << "\n";
      Edges.emplace_back(Node->VarID, Edge.Dst->VarID);
    }
  }
  // Sort edges for deterministic output in tests.
  std::sort(Edges.begin(), Edges.end(),
            [](const std::pair<int, int> &L, const std::pair<int, int> &R) {
              return L.first < R.first ||
                     (L.first == R.first && L.second < R.second);
            });
  for (RefGraphNode *Node : Nodes) {
    const GlobalVariable *Var = Node->Var;
    OS << "\t" << Node->VarID << " [label=\"" << (Var ? Var->getName() : "Root")
       << "\", style=filled, fillcolor=\"";
    std::optional<StringRef> SectionPrefix = Var->getSectionPrefix();
    if (SectionPrefix) {
      if (*SectionPrefix == "hot")
        OS << "darkgoldenrod1";
      else {
        assert(*SectionPrefix == "unlikely" && "Unknown section prefix");
        OS << "cadetblue1";
      }
    } else {
      OS << "lightgrey";
    }
    OS << "\", shape=\"";
    if (Var && Var->hasLocalLinkage())
      OS << "ellipse";
    else
      OS << "box";
    OS << "\"]\n";
  }

  for (const auto &Edge : Edges)
    OS << "\t" << Edge.first << " -> " << Edge.second << "\n";
  OS << "}\n";
}

} // namespace

namespace llvm {
template <> struct GraphTraits<RefGraphNode *> {
  using NodeType = RefGraphNode;
  using NodeRef = RefGraphNode *;
  using EdgeType = NodeType::Edge;
  using ChildIteratorType = NodeType::const_iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) { return N->Edges.begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->Edges.end(); }
};

template <>
struct GraphTraits<RefGraph *> : public GraphTraits<RefGraphNode *> {
  static NodeRef getEntryNode(RefGraph *G) { return G->getEntryNode(); }

  static ChildIteratorType nodes_begin(RefGraph *G) { return G->begin(); }

  static ChildIteratorType nodes_end(RefGraph *G) { return G->end(); }
};

} // end namespace llvm

/// A module pass which iterates global variables in the module and
/// annotates their section prefixes based on profile-driven analysis.
class StaticDataAnnotator : public ModulePass {
  void propagateHotness(Module &M);

  DenseMap<GlobalVariable *, DataHotness> GVHotness;

public:
  static char ID;

  StaticDataProfileInfo *SDPI = nullptr;
  const ProfileSummaryInfo *PSI = nullptr;

  StaticDataAnnotator() : ModulePass(ID) {
    initializeStaticDataAnnotatorPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<StaticDataProfileInfoWrapperPass>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    AU.setPreservesAll();
    ModulePass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "Static Data Annotator"; }

  bool runOnModule(Module &M) override;
};

void StaticDataAnnotator::propagateHotness(Module &M) {
  RefGraph G(M);
  G.buildGraph();

  scc_iterator<RefGraph *> RefGraphI = scc_begin(&G);
  // Iterate SCCs and propagate hotness.
  int SccNum = 0;
  while (!RefGraphI.isAtEnd()) {
    ++SccNum;
    auto Range = *RefGraphI;

    DataHotness LocalHotness = DataHotness::kCold;
    for (auto *Node : Range) {
      if (Node != nullptr && Node->Var != nullptr) {
        LocalHotness = std::max(LocalHotness, GVHotness[Node->Var]);
      }
    }

    for (auto *Node : Range) {
      DataHotness NodeVarHotness = LocalHotness;
      if (Node != nullptr && Node->Var != nullptr) {
        for (auto &Edge : Node->Edges) {
          NodeVarHotness = std::max(NodeVarHotness, GVHotness[Edge.Dst->Var]);
        }
        if (NodeVarHotness != DataHotness::kUnknown) {

          Node->Var->setSectionPrefix(
              StaticDataProfileInfo::hotnessToStr(NodeVarHotness));
        }
      }
    }
    ++RefGraphI;
  }

  if (!GlobalVariableRefGraphDotFile.empty()) {
    std::error_code EC;
    raw_fd_ostream OSDot(GlobalVariableRefGraphDotFile, EC,
                         sys::fs::OpenFlags::OF_Text);
    if (EC)
      report_fatal_error(Twine("Failed to open dot file ") +
                         GlobalVariableRefGraphDotFile + ": " + EC.message() +
                         "\n");
    G.exportToDot(OSDot);
  }
}

bool StaticDataAnnotator::runOnModule(Module &M) {
  SDPI = &getAnalysis<StaticDataProfileInfoWrapperPass>()
              .getStaticDataProfileInfo();
  PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
  GVHotness.clear();

  if (!PSI->hasProfileSummary())
    return false;

  bool Changed = false;
  for (auto &GV : M.globals()) {
    if (GV.isDeclarationForLinker())
      continue;

    // The implementation below assumes prior passes don't set section prefixes,
    // and specifically do 'assign' rather than 'update'. So report error if a
    // section prefix is already set.
    if (auto maybeSectionPrefix = GV.getSectionPrefix();
        maybeSectionPrefix && !maybeSectionPrefix->empty())
      llvm::report_fatal_error("Global variable " + GV.getName() +
                               " already has a section prefix " +
                               *maybeSectionPrefix);
    auto Hotness = SDPI->getConstantHotness(&GV, PSI);
    GVHotness[&GV] = Hotness;
    if (Hotness != DataHotness::kUnknown)
      Changed = true;
  }

  if (EnableSCCHotnessPropagation) {
    propagateHotness(M);
  } else {
    for (auto &[GV, Hotness] : GVHotness)
      if (Hotness != DataHotness::kUnknown)
        GV->setSectionPrefix(StaticDataProfileInfo::hotnessToStr(Hotness));
  }

  return Changed;
}

char StaticDataAnnotator::ID = 0;

INITIALIZE_PASS(StaticDataAnnotator, DEBUG_TYPE, "Static Data Annotator", false,
                false)

ModulePass *llvm::createStaticDataAnnotatorPass() {
  return new StaticDataAnnotator();
}
