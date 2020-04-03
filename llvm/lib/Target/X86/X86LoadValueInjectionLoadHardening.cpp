//==-- X86LoadValueInjectionLoadHardening.cpp - LVI load hardening for x86 --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Description: This pass finds Load Value Injection (LVI) gadgets consisting
/// of a load from memory (i.e., SOURCE), and any operation that may transmit
/// the value loaded from memory over a covert channel, or use the value loaded
/// from memory to determine a branch/call target (i.e., SINK). After finding
/// all such gadgets in a given function, the pass minimally inserts LFENCE
/// instructions in such a manner that the following property is satisfied: for
/// all SOURCE+SINK pairs, all paths in the CFG from SOURCE to SINK contain at
/// least one LFENCE instruction. The algorithm that implements this minimal
/// insertion is influenced by an academic paper that minimally inserts memory
/// fences for high-performance concurrent programs:
///         http://www.cs.ucr.edu/~lesani/companion/oopsla15/OOPSLA15.pdf
/// The algorithm implemented in this pass is as follows:
/// 1. Build a condensed CFG (i.e., a GadgetGraph) consisting only of the
/// following components:
///    - SOURCE instructions (also includes function arguments)
///    - SINK instructions
///    - Basic block entry points
///    - Basic block terminators
///    - LFENCE instructions
/// 2. Analyze the GadgetGraph to determine which SOURCE+SINK pairs (i.e.,
/// gadgets) are already mitigated by existing LFENCEs. If all gadgets have been
/// mitigated, go to step 6.
/// 3. Use a heuristic or plugin to approximate minimal LFENCE insertion.
/// 4. Insert one LFENCE along each CFG edge that was cut in step 3.
/// 5. Go to step 2.
/// 6. If any LFENCEs were inserted, return `true` from runOnFunction() to tell
/// LLVM that the function was modified.
///
//===----------------------------------------------------------------------===//

#include "ImmutableGraph.h"
#include "X86.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominanceFrontier.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RDFGraph.h"
#include "llvm/CodeGen/RDFLiveness.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define PASS_KEY "x86-lvi-load"
#define DEBUG_TYPE PASS_KEY

STATISTIC(NumFences, "Number of LFENCEs inserted for LVI mitigation");
STATISTIC(NumFunctionsConsidered, "Number of functions analyzed");
STATISTIC(NumFunctionsMitigated, "Number of functions for which mitigations "
                                 "were deployed");
STATISTIC(NumGadgets, "Number of LVI gadgets detected during analysis");

static cl::opt<std::string> OptimizePluginPath(
    PASS_KEY "-opt-plugin",
    cl::desc("Specify a plugin to optimize LFENCE insertion"), cl::Hidden);

static cl::opt<bool> NoConditionalBranches(
    PASS_KEY "-no-cbranch",
    cl::desc("Don't treat conditional branches as disclosure gadgets. This "
             "may improve performance, at the cost of security."),
    cl::init(false), cl::Hidden);

static cl::opt<bool> EmitDot(
    PASS_KEY "-dot",
    cl::desc(
        "For each function, emit a dot graph depicting potential LVI gadgets"),
    cl::init(false), cl::Hidden);

static cl::opt<bool> EmitDotOnly(
    PASS_KEY "-dot-only",
    cl::desc("For each function, emit a dot graph depicting potential LVI "
             "gadgets, and do not insert any fences"),
    cl::init(false), cl::Hidden);

static cl::opt<bool> EmitDotVerify(
    PASS_KEY "-dot-verify",
    cl::desc("For each function, emit a dot graph to stdout depicting "
             "potential LVI gadgets, used for testing purposes only"),
    cl::init(false), cl::Hidden);

static cl::opt<bool> NoFixedLoads(
    PASS_KEY "-no-fixed",
    cl::desc("Don't mitigate RIP-relative or RSP-relative loads. This "
             "may improve performance, at the cost of security."),
    cl::init(false), cl::Hidden);

static llvm::sys::DynamicLibrary OptimizeDL{};
typedef int (*OptimizeCutT)(unsigned int *nodes, unsigned int nodes_size,
                            unsigned int *edges, int *edge_values,
                            int *cut_edges /* out */, unsigned int edges_size);
static OptimizeCutT OptimizeCut = nullptr;

#define ARG_NODE nullptr
#define GADGET_EDGE ((int)(-1))
#define WEIGHT(EdgeValue) ((double)(2 * (EdgeValue) + 1))

namespace {

class X86LoadValueInjectionLoadHardeningPass : public MachineFunctionPass {
public:
  X86LoadValueInjectionLoadHardeningPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "X86 Load Value Injection (LVI) Load Hardening";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;

  static char ID;

private:
  struct MachineGadgetGraph : ImmutableGraph<MachineInstr *, int> {
    using GraphT = ImmutableGraph<MachineInstr *, int>;
    using Node = typename GraphT::Node;
    using Edge = typename GraphT::Edge;
    using size_type = typename GraphT::size_type;
    MachineGadgetGraph(Node *Nodes, size_type NodesSize, Edge *Edges,
                       size_type EdgesSize, int NumFences = 0,
                       int NumGadgets = 0)
        : GraphT{Nodes, NodesSize, Edges, EdgesSize}, NumFences{NumFences},
          NumGadgets{NumGadgets} {}
    MachineFunction &getMF() { // FIXME: This function should be cleaner
      for (Node *NI = nodes_begin(), *const NE = nodes_end(); NI != NE; ++NI) {
        if (NI->value()) {
          return *NI->value()->getMF();
        }
      }
      llvm_unreachable("Could not find a valid node");
    }
    static inline bool isCFGEdge(Edge &E) { return E.value() != GADGET_EDGE; }
    static inline bool isGadgetEdge(Edge &E) {
      return E.value() == GADGET_EDGE;
    }
    int NumFences;
    int NumGadgets;
  };
  friend struct llvm::DOTGraphTraits<MachineGadgetGraph *>;
  using GTraits = llvm::GraphTraits<MachineGadgetGraph *>;
  using GraphBuilder = ImmutableGraphBuilder<MachineGadgetGraph>;
  using EdgeSet = MachineGadgetGraph::EdgeSet;
  using Gadget = std::pair<MachineInstr *, MachineInstr *>;

  const X86Subtarget *STI;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;

  int hardenLoads(MachineFunction &MF, bool Fixed) const;
  std::unique_ptr<MachineGadgetGraph>
  getGadgetGraph(MachineFunction &MF, const MachineLoopInfo &MLI,
                 const MachineDominatorTree &MDT,
                 const MachineDominanceFrontier &MDF, bool FixedLoads) const;
  std::unique_ptr<MachineGadgetGraph>
  elimEdges(std::unique_ptr<MachineGadgetGraph> Graph) const;
  void cutEdges(MachineGadgetGraph &G, EdgeSet &CutEdges /* out */) const;
  int insertFences(MachineGadgetGraph &G,
                   EdgeSet &CutEdges /* in, out */) const;

  bool instrUsesRegToAccessMemory(const MachineInstr &I, unsigned Reg) const;
  bool instrUsesRegToBranch(const MachineInstr &I, unsigned Reg) const;
  template <unsigned K> bool hasLoadFrom(const MachineInstr &MI) const;
  bool instrAccessesStackSlot(const MachineInstr &MI) const;
  bool instrAccessesConstantPool(const MachineInstr &MI) const;
  bool instrAccessesGOT(const MachineInstr &MI) const;
  inline bool instrIsFixedAccess(const MachineInstr &MI) const {
    return instrAccessesConstantPool(MI) || instrAccessesStackSlot(MI) ||
           instrAccessesGOT(MI);
  }
  inline bool isFence(const MachineInstr *MI) const {
    return MI && (MI->getOpcode() == X86::LFENCE ||
                  (STI->useLVIControlFlowIntegrity() && MI->isCall()));
  }
};

} // end anonymous namespace

namespace llvm {

template <>
struct GraphTraits<X86LoadValueInjectionLoadHardeningPass::MachineGadgetGraph *>
    : GraphTraits<ImmutableGraph<MachineInstr *, int> *> {};

template <>
struct DOTGraphTraits<
    X86LoadValueInjectionLoadHardeningPass::MachineGadgetGraph *>
    : DefaultDOTGraphTraits {
  using GraphType = X86LoadValueInjectionLoadHardeningPass::MachineGadgetGraph;
  using Traits = X86LoadValueInjectionLoadHardeningPass::GTraits;
  using NodeRef = typename Traits::NodeRef;
  using EdgeRef = typename Traits::EdgeRef;
  using ChildIteratorType = typename Traits::ChildIteratorType;
  using ChildEdgeIteratorType = typename Traits::ChildEdgeIteratorType;

  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  static std::string getGraphName(GraphType *G) {
    std::string GraphName{"Speculative gadgets for \""};
    GraphName += G->getMF().getName();
    GraphName += "\" function";
    return GraphName;
  }

  std::string getNodeLabel(NodeRef Node, GraphType *) {
    std::string str;
    raw_string_ostream str_stream{str};
    if (Node->value() == ARG_NODE)
      return "ARGS";
    str_stream << *Node->value();
    return str_stream.str();
  }

  static std::string getNodeAttributes(NodeRef Node, GraphType *) {
    MachineInstr *MI = Node->value();
    if (MI == ARG_NODE)
      return "color = blue";
    else if (MI->getOpcode() == X86::LFENCE)
      return "color = green";
    else
      return "";
  }

  static std::string getEdgeAttributes(NodeRef, ChildIteratorType E,
                                       GraphType *) {
    int EdgeVal = (*E.getCurrent()).value();
    return EdgeVal >= 0 ? "label = " + std::to_string(EdgeVal)
                        : "color = red, style = \"dashed\"";
  }
};

} // end namespace llvm

char X86LoadValueInjectionLoadHardeningPass::ID = 0;

void X86LoadValueInjectionLoadHardeningPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<MachineLoopInfo>();
  AU.addRequired<MachineDominatorTree>();
  AU.addRequired<MachineDominanceFrontier>();
  AU.setPreservesCFG();
}

bool X86LoadValueInjectionLoadHardeningPass::runOnMachineFunction(
    MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "***** " << getPassName() << " : " << MF.getName()
                    << " *****\n");
  STI = &MF.getSubtarget<X86Subtarget>();
  if (!STI->useLVILoadHardening() || !STI->is64Bit())
    return false; // FIXME: support 32-bit

  // Don't skip functions with the "optnone" attr but participate in opt-bisect.
  const Function &F = MF.getFunction();
  if (!F.hasOptNone() && skipFunction(F))
    return false;

  ++NumFunctionsConsidered;
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  LLVM_DEBUG(dbgs() << "Hardening data-dependent loads...\n");
  int FencesInserted = hardenLoads(MF, false);
  LLVM_DEBUG(dbgs() << "Hardening data-dependent loads... Done\n");
  if (!NoFixedLoads) {
    LLVM_DEBUG(dbgs() << "Hardening fixed loads...\n");
    FencesInserted += hardenLoads(MF, true);
    LLVM_DEBUG(dbgs() << "Hardening fixed loads... Done\n");
  }
  if (FencesInserted > 0)
    ++NumFunctionsMitigated;
  NumFences += FencesInserted;
  return (FencesInserted > 0);
}

// Apply the mitigation to `MF`, return the number of fences inserted.
// If `FixedLoads` is `true`, then the mitigation will be applied to fixed
// loads; otherwise, mitigation will be applied to non-fixed loads.
int X86LoadValueInjectionLoadHardeningPass::hardenLoads(MachineFunction &MF,
                                                        bool FixedLoads) const {
  int FencesInserted = 0;

  LLVM_DEBUG(dbgs() << "Building gadget graph...\n");
  const auto &MLI = getAnalysis<MachineLoopInfo>();
  const auto &MDT = getAnalysis<MachineDominatorTree>();
  const auto &MDF = getAnalysis<MachineDominanceFrontier>();
  std::unique_ptr<MachineGadgetGraph> Graph =
      getGadgetGraph(MF, MLI, MDT, MDF, FixedLoads);
  LLVM_DEBUG(dbgs() << "Building gadget graph... Done\n");
  if (Graph == nullptr)
    return 0; // didn't find any gadgets

  if (EmitDotVerify) {
    WriteGraph(outs(), Graph.get());
    return 0;
  }

  if (EmitDot || EmitDotOnly) {
    LLVM_DEBUG(dbgs() << "Emitting gadget graph...\n");
    std::error_code FileError;
    std::string FileName = "lvi.";
    if (FixedLoads)
      FileName += "fixed.";
    FileName += Graph->getMF().getName();
    FileName += ".dot";
    raw_fd_ostream FileOut(FileName, FileError);
    if (FileError)
      errs() << FileError.message();
    WriteGraph(FileOut, Graph.get());
    FileOut.close();
    LLVM_DEBUG(dbgs() << "Emitting gadget graph... Done\n");
    if (EmitDotOnly)
      return 0;
  }

  do {
    LLVM_DEBUG(dbgs() << "Eliminating mitigated paths...\n");
    std::unique_ptr<MachineGadgetGraph> ElimGraph = elimEdges(std::move(Graph));
    LLVM_DEBUG(dbgs() << "Eliminating mitigated paths... Done\n");
    if (ElimGraph->NumGadgets == 0)
      break;

    EdgeSet CutEdges{*ElimGraph};
    LLVM_DEBUG(dbgs() << "Cutting edges...\n");
    cutEdges(*ElimGraph, CutEdges);
    LLVM_DEBUG(dbgs() << "Cutting edges... Done\n");

    LLVM_DEBUG(dbgs() << "Inserting LFENCEs...\n");
    FencesInserted += insertFences(*ElimGraph, CutEdges);
    LLVM_DEBUG(dbgs() << "Inserting LFENCEs... Done\n");

    Graph.reset(GraphBuilder::trim(
        *ElimGraph, MachineGadgetGraph::NodeSet{*ElimGraph}, CutEdges));
  } while (true);

  return FencesInserted;
}

std::unique_ptr<X86LoadValueInjectionLoadHardeningPass::MachineGadgetGraph>
X86LoadValueInjectionLoadHardeningPass::getGadgetGraph(
    MachineFunction &MF, const MachineLoopInfo &MLI,
    const MachineDominatorTree &MDT, const MachineDominanceFrontier &MDF,
    bool FixedLoads) const {
  using namespace rdf;

  // Build the Register Dataflow Graph using the RDF framework
  TargetOperandInfo TOI{*TII};
  DataFlowGraph DFG{MF, *TII, *TRI, MDT, MDF, TOI};
  DFG.build();
  Liveness L{MF.getRegInfo(), DFG};
  L.computePhiInfo();

  GraphBuilder Builder;
  using GraphIter = typename GraphBuilder::NodeRef;
  DenseMap<MachineInstr *, GraphIter> NodeMap;
  int FenceCount = 0;
  auto MaybeAddNode = [&NodeMap, &Builder](MachineInstr *MI) {
    auto Ref = NodeMap.find(MI);
    if (Ref == NodeMap.end()) {
      auto I = Builder.addVertex(MI);
      NodeMap[MI] = I;
      return std::pair<GraphIter, bool>{I, true};
    } else {
      return std::pair<GraphIter, bool>{Ref->getSecond(), false};
    }
  };

  // Analyze all machine instructions to find gadgets and LFENCEs, adding
  // each interesting value to `Nodes`
  DenseSet<std::pair<GraphIter, GraphIter>> GadgetEdgeSet;
  auto AnalyzeDef = [&](NodeAddr<DefNode *> Def) {
    MachineInstr *MI = Def.Addr->getFlags() & NodeAttrs::PhiRef
                           ? ARG_NODE
                           : Def.Addr->getOp().getParent();
    auto AnalyzeUse = [&](NodeAddr<UseNode *> Use) {
      assert(!(Use.Addr->getFlags() & NodeAttrs::PhiRef));
      MachineOperand &UseMO = Use.Addr->getOp();
      MachineInstr &UseMI = *UseMO.getParent();
      assert(UseMO.isReg());
      // We naively assume that an instruction propagates any loaded Uses
      // to all Defs, unless the instruction is a call
      if (UseMI.isCall())
        return false;
      if (instrUsesRegToAccessMemory(UseMI, UseMO.getReg()) ||
          (!NoConditionalBranches &&
           instrUsesRegToBranch(UseMI, UseMO.getReg()))) { // found a gadget!
        // add the root of this chain
        auto GadgetBegin = MaybeAddNode(MI);
        // and the instruction that (transitively) discloses the root
        auto GadgetEnd = MaybeAddNode(&UseMI);
        if (GadgetEdgeSet.insert({GadgetBegin.first, GadgetEnd.first}).second)
          Builder.addEdge(GADGET_EDGE, GadgetBegin.first, GadgetEnd.first);
        if (UseMI.mayLoad()) // FIXME: This should be more precise
          return false;      // stop traversing further uses of `Reg`
      }
      return true;
    };
    SmallSet<NodeId, 8> NodesVisited;
    std::function<void(NodeAddr<DefNode *>)> AnalyzeDefUseChain =
        [&](NodeAddr<DefNode *> Def) {
          if (Def.Addr->getAttrs() & NodeAttrs::Dead)
            return;
          RegisterRef DefReg = DFG.getPRI().normalize(Def.Addr->getRegRef(DFG));
          NodeList Uses;
          for (auto UseID : L.getAllReachedUses(DefReg, Def)) {
            auto Use = DFG.addr<UseNode *>(UseID);
            if (Use.Addr->getFlags() & NodeAttrs::PhiRef) { // phi node
              NodeAddr<PhiNode *> Phi = Use.Addr->getOwner(DFG);
              for (auto I : L.getRealUses(Phi.Id)) {
                if (DFG.getPRI().alias(RegisterRef(I.first), DefReg)) {
                  for (auto UA : I.second) {
                    auto PhiUse = DFG.addr<UseNode *>(UA.first);
                    Uses.push_back(PhiUse);
                  }
                }
              }
            } else { // not a phi node
              Uses.push_back(Use);
            }
          }
          for (auto N : Uses) {
            NodeAddr<UseNode *> Use{N};
            if (NodesVisited.insert(Use.Id).second && AnalyzeUse(Use)) {
              NodeAddr<InstrNode *> Owner{Use.Addr->getOwner(DFG)};
              NodeList Defs = Owner.Addr->members_if(DataFlowGraph::IsDef, DFG);
              std::for_each(Defs.begin(), Defs.end(), AnalyzeDefUseChain);
            }
          }
        };
    AnalyzeDefUseChain(Def);
  };

  LLVM_DEBUG(dbgs() << "Analyzing def-use chains to find gadgets\n");
  // Analyze function arguments
  if (!FixedLoads) { // only need to analyze function args once
    NodeAddr<BlockNode *> EntryBlock = DFG.getFunc().Addr->getEntryBlock(DFG);
    for (NodeAddr<PhiNode *> ArgPhi :
         EntryBlock.Addr->members_if(DataFlowGraph::IsPhi, DFG)) {
      NodeList Defs = ArgPhi.Addr->members_if(DataFlowGraph::IsDef, DFG);
      std::for_each(Defs.begin(), Defs.end(), AnalyzeDef);
    }
  }
  // Analyze every instruction in MF
  for (NodeAddr<BlockNode *> BA : DFG.getFunc().Addr->members(DFG)) {
    for (NodeAddr<StmtNode *> SA :
         BA.Addr->members_if(DataFlowGraph::IsCode<NodeAttrs::Stmt>, DFG)) {
      MachineInstr *MI = SA.Addr->getCode();
      if (isFence(MI)) {
        MaybeAddNode(MI);
        ++FenceCount;
      } else if (MI->mayLoad() && ((FixedLoads && instrIsFixedAccess(*MI)) ||
                                   (!FixedLoads && !instrIsFixedAccess(*MI)))) {
        NodeList Defs = SA.Addr->members_if(DataFlowGraph::IsDef, DFG);
        std::for_each(Defs.begin(), Defs.end(), AnalyzeDef);
      }
    }
  }
  int GadgetCount = static_cast<int>(GadgetEdgeSet.size());
  LLVM_DEBUG(dbgs() << "Found " << FenceCount << " fences\n");
  LLVM_DEBUG(dbgs() << "Found " << GadgetCount << " gadgets\n");
  if (GadgetCount == 0)
    return nullptr;
  NumGadgets += GadgetCount;

  // Traverse CFG to build the rest of the graph
  SmallSet<MachineBasicBlock *, 8> BlocksVisited;
  std::function<void(MachineBasicBlock *, GraphIter, unsigned)> TraverseCFG =
      [&](MachineBasicBlock *MBB, GraphIter GI, unsigned ParentDepth) {
        unsigned LoopDepth = MLI.getLoopDepth(MBB);
        if (!MBB->empty()) {
          // Always add the first instruction in each block
          auto NI = MBB->begin();
          auto BeginBB = MaybeAddNode(&*NI);
          Builder.addEdge(ParentDepth, GI, BeginBB.first);
          if (!BlocksVisited.insert(MBB).second)
            return;

          // Add any instructions within the block that are gadget components
          GI = BeginBB.first;
          while (++NI != MBB->end()) {
            auto Ref = NodeMap.find(&*NI);
            if (Ref != NodeMap.end()) {
              Builder.addEdge(LoopDepth, GI, Ref->getSecond());
              GI = Ref->getSecond();
            }
          }

          // Always add the terminator instruction, if one exists
          auto T = MBB->getFirstTerminator();
          if (T != MBB->end()) {
            auto EndBB = MaybeAddNode(&*T);
            if (EndBB.second)
              Builder.addEdge(LoopDepth, GI, EndBB.first);
            GI = EndBB.first;
          }
        }
        for (MachineBasicBlock *Succ : MBB->successors())
          TraverseCFG(Succ, GI, LoopDepth);
      };
  // ARG_NODE is a pseudo-instruction that represents MF args in the GadgetGraph
  GraphIter ArgNode = MaybeAddNode(ARG_NODE).first;
  TraverseCFG(&MF.front(), ArgNode, 0);
  std::unique_ptr<MachineGadgetGraph> G{Builder.get(FenceCount, GadgetCount)};
  LLVM_DEBUG(dbgs() << "Found " << GTraits::size(G.get()) << " nodes\n");
  return G;
}

std::unique_ptr<X86LoadValueInjectionLoadHardeningPass::MachineGadgetGraph>
X86LoadValueInjectionLoadHardeningPass::elimEdges(
    std::unique_ptr<MachineGadgetGraph> Graph) const {
  MachineGadgetGraph::NodeSet ElimNodes{*Graph};
  MachineGadgetGraph::EdgeSet ElimEdges{*Graph};

  if (Graph->NumFences > 0) { // eliminate fences
    for (auto EI = Graph->edges_begin(), EE = Graph->edges_end(); EI != EE;
         ++EI) {
      GTraits::NodeRef Dest = GTraits::edge_dest(*EI);
      if (isFence(Dest->value())) {
        ElimNodes.insert(Dest);
        ElimEdges.insert(EI);
        std::for_each(
            GTraits::child_edge_begin(Dest), GTraits::child_edge_end(Dest),
            [&ElimEdges](GTraits::EdgeRef E) { ElimEdges.insert(&E); });
      }
    }
    LLVM_DEBUG(dbgs() << "Eliminated " << ElimNodes.count()
                      << " fence nodes\n");
  }

  // eliminate gadget edges that are mitigated
  int NumGadgets = 0;
  MachineGadgetGraph::NodeSet Visited{*Graph}, GadgetSinks{*Graph};
  MachineGadgetGraph::EdgeSet ElimGadgets{*Graph};
  for (auto NI = GTraits::nodes_begin(Graph.get()),
            NE = GTraits::nodes_end(Graph.get());
       NI != NE; ++NI) {
    // collect the gadgets for this node
    for (auto EI = GTraits::child_edge_begin(*NI),
              EE = GTraits::child_edge_end(*NI);
         EI != EE; ++EI) {
      if (MachineGadgetGraph::isGadgetEdge(*EI)) {
        ++NumGadgets;
        ElimGadgets.insert(EI);
        GadgetSinks.insert(GTraits::edge_dest(*EI));
      }
    }
    if (GadgetSinks.empty())
      continue;
    std::function<void(GTraits::NodeRef, bool)> TraverseDFS =
        [&](GTraits::NodeRef N, bool FirstNode) {
          if (!FirstNode) {
            Visited.insert(N);
            if (GadgetSinks.contains(N)) {
              for (auto CEI = GTraits::child_edge_begin(*NI),
                        CEE = GTraits::child_edge_end(*NI);
                   CEI != CEE; ++CEI) {
                if (MachineGadgetGraph::isGadgetEdge(*CEI) &&
                    GTraits::edge_dest(*CEI) == N)
                  ElimGadgets.erase(CEI);
              }
            }
          }
          for (auto CEI = GTraits::child_edge_begin(N),
                    CEE = GTraits::child_edge_end(N);
               CEI != CEE; ++CEI) {
            GTraits::NodeRef Dest = GTraits::edge_dest(*CEI);
            if (MachineGadgetGraph::isCFGEdge(*CEI) &&
                !Visited.contains(Dest) && !ElimEdges.contains(CEI))
              TraverseDFS(Dest, false);
          }
        };
    TraverseDFS(*NI, true);
    Visited.clear();
    GadgetSinks.clear();
  }
  LLVM_DEBUG(dbgs() << "Eliminated " << ElimGadgets.count()
                    << " gadget edges\n");
  ElimEdges |= ElimGadgets;

  if (!(ElimEdges.empty() && ElimNodes.empty())) {
    int NumRemainingGadgets = NumGadgets - ElimGadgets.count();
    Graph.reset(GraphBuilder::trim(*Graph, ElimNodes, ElimEdges,
                                   0 /* NumFences */, NumRemainingGadgets));
  } else {
    Graph->NumFences = 0;
    Graph->NumGadgets = NumGadgets;
  }
  return Graph;
}

void X86LoadValueInjectionLoadHardeningPass::cutEdges(
    MachineGadgetGraph &G,
    MachineGadgetGraph::EdgeSet &CutEdges /* out */) const {
  if (!OptimizePluginPath.empty()) {
    if (!OptimizeDL.isValid()) {
      std::string ErrorMsg{};
      OptimizeDL = llvm::sys::DynamicLibrary::getPermanentLibrary(
          OptimizePluginPath.c_str(), &ErrorMsg);
      if (!ErrorMsg.empty())
        report_fatal_error("Failed to load opt plugin: \"" + ErrorMsg + '\"');
      OptimizeCut = (OptimizeCutT)OptimizeDL.getAddressOfSymbol("optimize_cut");
      if (!OptimizeCut)
        report_fatal_error("Invalid optimization plugin");
    }
    auto *Nodes = new unsigned int[G.nodes_size() + 1 /* terminator node */];
    auto *Edges = new unsigned int[G.edges_size()];
    auto *EdgeCuts = new int[G.edges_size()];
    auto *EdgeValues = new int[G.edges_size()];
    for (auto *NI = G.nodes_begin(), *NE = G.nodes_end(); NI != NE; ++NI) {
      Nodes[std::distance(G.nodes_begin(), NI)] =
          std::distance(G.edges_begin(), GTraits::child_edge_begin(NI));
    }
    Nodes[G.nodes_size()] = G.edges_size(); // terminator node
    for (auto *EI = G.edges_begin(), *EE = G.edges_end(); EI != EE; ++EI) {
      Edges[std::distance(G.edges_begin(), EI)] =
          std::distance(G.nodes_begin(), GTraits::edge_dest(*EI));
      EdgeValues[std::distance(G.edges_begin(), EI)] = EI->value();
    }
    OptimizeCut(Nodes, G.nodes_size(), Edges, EdgeValues, EdgeCuts,
                G.edges_size());
    for (int I = 0; I < G.edges_size(); ++I) {
      if (EdgeCuts[I])
        CutEdges.set(I);
    }
    delete[] Nodes;
    delete[] Edges;
    delete[] EdgeCuts;
    delete[] EdgeValues;
  } else { // Use the default greedy heuristic
    // Find the cheapest CFG edge that will eliminate a gadget (by being egress
    // from a SOURCE node or ingress to a SINK node), and cut it.
    MachineGadgetGraph::NodeSet GadgetSinks{G};
    MachineGadgetGraph::Edge *CheapestSoFar = nullptr;
    for (auto NI = GTraits::nodes_begin(&G), NE = GTraits::nodes_end(&G);
         NI != NE; ++NI) {
      for (auto EI = GTraits::child_edge_begin(*NI),
                EE = GTraits::child_edge_end(*NI);
           EI != EE; ++EI) {
        if (MachineGadgetGraph::isGadgetEdge(*EI)) {
          // NI is a SOURCE node. Look for a cheap egress edge
          for (auto EEI = GTraits::child_edge_begin(*NI); EEI != EE; ++EEI) {
            if (MachineGadgetGraph::isCFGEdge(*EEI)) {
              if (!CheapestSoFar || EEI->value() < CheapestSoFar->value())
                CheapestSoFar = EEI;
            }
          }
          GadgetSinks.insert(GTraits::edge_dest(*EI));
        } else { // EI is a CFG edge
          if (GadgetSinks.contains(GTraits::edge_dest(*EI))) {
            // The dest is a SINK node. Hence EI is an ingress edge
            if (!CheapestSoFar || EI->value() < CheapestSoFar->value())
              CheapestSoFar = EI;
          }
        }
      }
    }
    assert(CheapestSoFar && "Failed to cut an edge");
    CutEdges.insert(CheapestSoFar);
  }
  LLVM_DEBUG(dbgs() << "Cut " << CutEdges.count() << " edges\n");
}

int X86LoadValueInjectionLoadHardeningPass::insertFences(
    MachineGadgetGraph &G, EdgeSet &CutEdges /* in, out */) const {
  int FencesInserted = 0, AdditionalEdgesCut = 0;
  auto CutAllCFGEdges = [&CutEdges, &AdditionalEdgesCut](GTraits::NodeRef N) {
    for (auto CEI = GTraits::child_edge_begin(N),
              CEE = GTraits::child_edge_end(N);
         CEI != CEE; ++CEI) {
      if (MachineGadgetGraph::isCFGEdge(*CEI) && !CutEdges.contains(CEI)) {
        CutEdges.insert(CEI);
        ++AdditionalEdgesCut;
      }
    }
  };
  for (auto NI = GTraits::nodes_begin(&G), NE = GTraits::nodes_end(&G);
       NI != NE; ++NI) {
    for (auto CEI = GTraits::child_edge_begin(*NI),
              CEE = GTraits::child_edge_end(*NI);
         CEI != CEE; ++CEI) {
      if (CutEdges.contains(CEI)) {
        MachineInstr *MI = (*NI)->value(), *Prev;
        MachineBasicBlock *MBB;
        MachineBasicBlock::iterator InsertionPt;
        if (MI == ARG_NODE) { // insert LFENCE at beginning of entry block
          MBB = &G.getMF().front();
          InsertionPt = MBB->begin();
          Prev = nullptr;
        } else if (MI->isBranch()) { // insert the LFENCE before the branch
          MBB = MI->getParent();
          InsertionPt = MI;
          Prev = MI->getPrevNode();
          CutAllCFGEdges(*NI);
        } else { // insert the LFENCE after the instruction
          MBB = MI->getParent();
          InsertionPt = MI->getNextNode() ? MI->getNextNode() : MBB->end();
          Prev = InsertionPt == MBB->end()
                     ? (MBB->empty() ? nullptr : &MBB->back())
                     : InsertionPt->getPrevNode();
        }
        if ((InsertionPt == MBB->end() || !isFence(&*InsertionPt)) &&
            (!Prev || !isFence(Prev))) {
          BuildMI(*MBB, InsertionPt, DebugLoc(), TII->get(X86::LFENCE));
          ++FencesInserted;
        }
      }
    }
  }
  LLVM_DEBUG(dbgs() << "Inserted " << FencesInserted << " fences\n");
  LLVM_DEBUG(dbgs() << "Cut an additional " << AdditionalEdgesCut
                    << " edges during fence insertion\n");
  return FencesInserted;
}

bool X86LoadValueInjectionLoadHardeningPass::instrUsesRegToAccessMemory(
    const MachineInstr &MI, unsigned Reg) const {
  if (!MI.mayLoadOrStore() || MI.getOpcode() == X86::MFENCE ||
      MI.getOpcode() == X86::SFENCE || MI.getOpcode() == X86::LFENCE)
    return false;

  // FIXME: This does not handle pseudo loading instruction like TCRETURN*
  const MCInstrDesc &Desc = MI.getDesc();
  int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);
  if (MemRefBeginIdx < 0) {
    LLVM_DEBUG(dbgs() << "Warning: unable to obtain memory operand for loading "
                         "instruction:\n";
               MI.print(dbgs()); dbgs() << '\n';);
    return false;
  }
  MemRefBeginIdx += X86II::getOperandBias(Desc);

  const MachineOperand &BaseMO =
      MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg);
  const MachineOperand &IndexMO =
      MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg);
  return (BaseMO.isReg() && BaseMO.getReg() != X86::NoRegister &&
          TRI->regsOverlap(BaseMO.getReg(), Reg)) ||
         (IndexMO.isReg() && IndexMO.getReg() != X86::NoRegister &&
          TRI->regsOverlap(IndexMO.getReg(), Reg));
}

bool X86LoadValueInjectionLoadHardeningPass::instrUsesRegToBranch(
    const MachineInstr &MI, unsigned Reg) const {
  if (!MI.isConditionalBranch())
    return false;
  for (const MachineOperand &Use : MI.uses())
    if (Use.isReg() && Use.getReg() == Reg)
      return true;
  return false;
}

template <unsigned K>
bool X86LoadValueInjectionLoadHardeningPass::hasLoadFrom(
    const MachineInstr &MI) const {
  for (auto &MMO : MI.memoperands()) {
    const PseudoSourceValue *PSV = MMO->getPseudoValue();
    if (PSV && PSV->kind() == K && MMO->isLoad())
      return true;
  }
  return false;
}

bool X86LoadValueInjectionLoadHardeningPass::instrAccessesStackSlot(
    const MachineInstr &MI) const {
  // Check the PSV first
  if (hasLoadFrom<PseudoSourceValue::PSVKind::FixedStack>(MI))
    return true;
  // Some loads are not marked with a PSV, so we always need to double check
  const MCInstrDesc &Desc = MI.getDesc();
  int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);
  if (MemRefBeginIdx < 0)
    return false;
  MemRefBeginIdx += X86II::getOperandBias(Desc);
  return MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg).isFI() &&
         MI.getOperand(MemRefBeginIdx + X86::AddrScaleAmt).isImm() &&
         MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg).isReg() &&
         MI.getOperand(MemRefBeginIdx + X86::AddrDisp).isImm() &&
         MI.getOperand(MemRefBeginIdx + X86::AddrScaleAmt).getImm() == 1 &&
         MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg).getReg() ==
             X86::NoRegister &&
         MI.getOperand(MemRefBeginIdx + X86::AddrDisp).getImm() == 0;
}

bool X86LoadValueInjectionLoadHardeningPass::instrAccessesConstantPool(
    const MachineInstr &MI) const {
  if (hasLoadFrom<PseudoSourceValue::PSVKind::ConstantPool>(MI))
    return true;
  const MCInstrDesc &Desc = MI.getDesc();
  int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);
  if (MemRefBeginIdx < 0)
    return false;
  MemRefBeginIdx += X86II::getOperandBias(Desc);
  return MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg).isReg() &&
         MI.getOperand(MemRefBeginIdx + X86::AddrScaleAmt).isImm() &&
         MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg).isReg() &&
         MI.getOperand(MemRefBeginIdx + X86::AddrDisp).isCPI() &&
         (MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg).getReg() ==
              X86::RIP ||
          MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg).getReg() ==
              X86::NoRegister) &&
         MI.getOperand(MemRefBeginIdx + X86::AddrScaleAmt).getImm() == 1 &&
         MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg).getReg() ==
             X86::NoRegister;
}

bool X86LoadValueInjectionLoadHardeningPass::instrAccessesGOT(
    const MachineInstr &MI) const {
  if (hasLoadFrom<PseudoSourceValue::PSVKind::GOT>(MI))
    return true;
  const MCInstrDesc &Desc = MI.getDesc();
  int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);
  if (MemRefBeginIdx < 0)
    return false;
  MemRefBeginIdx += X86II::getOperandBias(Desc);
  return MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg).isReg() &&
         MI.getOperand(MemRefBeginIdx + X86::AddrScaleAmt).isImm() &&
         MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg).isReg() &&
         MI.getOperand(MemRefBeginIdx + X86::AddrDisp).getTargetFlags() ==
             X86II::MO_GOTPCREL &&
         MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg).getReg() ==
             X86::RIP &&
         MI.getOperand(MemRefBeginIdx + X86::AddrScaleAmt).getImm() == 1 &&
         MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg).getReg() ==
             X86::NoRegister;
}

INITIALIZE_PASS_BEGIN(X86LoadValueInjectionLoadHardeningPass, PASS_KEY,
                      "X86 LVI load hardening", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineDominanceFrontier)
INITIALIZE_PASS_END(X86LoadValueInjectionLoadHardeningPass, PASS_KEY,
                    "X86 LVI load hardening", false, false)

FunctionPass *llvm::createX86LoadValueInjectionLoadHardeningPass() {
  return new X86LoadValueInjectionLoadHardeningPass();
}
