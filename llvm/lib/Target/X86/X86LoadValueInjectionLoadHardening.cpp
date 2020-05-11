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
/// from memory to determine a branch/call target (i.e., SINK).
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
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define PASS_KEY "x86-lvi-load"
#define DEBUG_TYPE PASS_KEY

STATISTIC(NumFunctionsConsidered, "Number of functions analyzed");
STATISTIC(NumFunctionsMitigated, "Number of functions for which mitigations "
                                 "were deployed");
STATISTIC(NumGadgets, "Number of LVI gadgets detected during analysis");

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

namespace {

struct MachineGadgetGraph : ImmutableGraph<MachineInstr *, int> {
  static constexpr int GadgetEdgeSentinel = -1;
  static constexpr MachineInstr *const ArgNodeSentinel = nullptr;

  using GraphT = ImmutableGraph<MachineInstr *, int>;
  using Node = typename GraphT::Node;
  using Edge = typename GraphT::Edge;
  using size_type = typename GraphT::size_type;
  MachineGadgetGraph(std::unique_ptr<Node[]> Nodes,
                     std::unique_ptr<Edge[]> Edges, size_type NodesSize,
                     size_type EdgesSize, int NumFences = 0, int NumGadgets = 0)
      : GraphT(std::move(Nodes), std::move(Edges), NodesSize, EdgesSize),
        NumFences(NumFences), NumGadgets(NumGadgets) {}
  static inline bool isCFGEdge(const Edge &E) {
    return E.getValue() != GadgetEdgeSentinel;
  }
  static inline bool isGadgetEdge(const Edge &E) {
    return E.getValue() == GadgetEdgeSentinel;
  }
  int NumFences;
  int NumGadgets;
};

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
  using GraphBuilder = ImmutableGraphBuilder<MachineGadgetGraph>;
  using EdgeSet = MachineGadgetGraph::EdgeSet;
  using NodeSet = MachineGadgetGraph::NodeSet;
  using Gadget = std::pair<MachineInstr *, MachineInstr *>;

  const X86Subtarget *STI;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;

  std::unique_ptr<MachineGadgetGraph>
  getGadgetGraph(MachineFunction &MF, const MachineLoopInfo &MLI,
                 const MachineDominatorTree &MDT,
                 const MachineDominanceFrontier &MDF) const;

  bool instrUsesRegToAccessMemory(const MachineInstr &I, unsigned Reg) const;
  bool instrUsesRegToBranch(const MachineInstr &I, unsigned Reg) const;
  inline bool isFence(const MachineInstr *MI) const {
    return MI && (MI->getOpcode() == X86::LFENCE ||
                  (STI->useLVIControlFlowIntegrity() && MI->isCall()));
  }
};

} // end anonymous namespace

namespace llvm {

template <>
struct GraphTraits<MachineGadgetGraph *>
    : GraphTraits<ImmutableGraph<MachineInstr *, int> *> {};

template <>
struct DOTGraphTraits<MachineGadgetGraph *> : DefaultDOTGraphTraits {
  using GraphType = MachineGadgetGraph;
  using Traits = llvm::GraphTraits<GraphType *>;
  using NodeRef = typename Traits::NodeRef;
  using EdgeRef = typename Traits::EdgeRef;
  using ChildIteratorType = typename Traits::ChildIteratorType;
  using ChildEdgeIteratorType = typename Traits::ChildEdgeIteratorType;

  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  std::string getNodeLabel(NodeRef Node, GraphType *) {
    if (Node->getValue() == MachineGadgetGraph::ArgNodeSentinel)
      return "ARGS";

    std::string Str;
    raw_string_ostream OS(Str);
    OS << *Node->getValue();
    return OS.str();
  }

  static std::string getNodeAttributes(NodeRef Node, GraphType *) {
    MachineInstr *MI = Node->getValue();
    if (MI == MachineGadgetGraph::ArgNodeSentinel)
      return "color = blue";
    if (MI->getOpcode() == X86::LFENCE)
      return "color = green";
    return "";
  }

  static std::string getEdgeAttributes(NodeRef, ChildIteratorType E,
                                       GraphType *) {
    int EdgeVal = (*E.getCurrent()).getValue();
    return EdgeVal >= 0 ? "label = " + std::to_string(EdgeVal)
                        : "color = red, style = \"dashed\"";
  }
};

} // end namespace llvm

constexpr MachineInstr *MachineGadgetGraph::ArgNodeSentinel;
constexpr int MachineGadgetGraph::GadgetEdgeSentinel;

char X86LoadValueInjectionLoadHardeningPass::ID = 0;

void X86LoadValueInjectionLoadHardeningPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<MachineLoopInfo>();
  AU.addRequired<MachineDominatorTree>();
  AU.addRequired<MachineDominanceFrontier>();
  AU.setPreservesCFG();
}

static void WriteGadgetGraph(raw_ostream &OS, MachineFunction &MF,
                             MachineGadgetGraph *G) {
  WriteGraph(OS, G, /*ShortNames*/ false,
             "Speculative gadgets for \"" + MF.getName() + "\" function");
}

bool X86LoadValueInjectionLoadHardeningPass::runOnMachineFunction(
    MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "***** " << getPassName() << " : " << MF.getName()
                    << " *****\n");
  STI = &MF.getSubtarget<X86Subtarget>();
  if (!STI->useLVILoadHardening())
    return false;

  // FIXME: support 32-bit
  if (!STI->is64Bit())
    report_fatal_error("LVI load hardening is only supported on 64-bit", false);

  // Don't skip functions with the "optnone" attr but participate in opt-bisect.
  const Function &F = MF.getFunction();
  if (!F.hasOptNone() && skipFunction(F))
    return false;

  ++NumFunctionsConsidered;
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  LLVM_DEBUG(dbgs() << "Building gadget graph...\n");
  const auto &MLI = getAnalysis<MachineLoopInfo>();
  const auto &MDT = getAnalysis<MachineDominatorTree>();
  const auto &MDF = getAnalysis<MachineDominanceFrontier>();
  std::unique_ptr<MachineGadgetGraph> Graph = getGadgetGraph(MF, MLI, MDT, MDF);
  LLVM_DEBUG(dbgs() << "Building gadget graph... Done\n");
  if (Graph == nullptr)
    return false; // didn't find any gadgets

  if (EmitDotVerify) {
    WriteGadgetGraph(outs(), MF, Graph.get());
    return false;
  }

  if (EmitDot || EmitDotOnly) {
    LLVM_DEBUG(dbgs() << "Emitting gadget graph...\n");
    std::error_code FileError;
    std::string FileName = "lvi.";
    FileName += MF.getName();
    FileName += ".dot";
    raw_fd_ostream FileOut(FileName, FileError);
    if (FileError)
      errs() << FileError.message();
    WriteGadgetGraph(FileOut, MF, Graph.get());
    FileOut.close();
    LLVM_DEBUG(dbgs() << "Emitting gadget graph... Done\n");
    if (EmitDotOnly)
      return false;
  }

  return 0;
}

std::unique_ptr<MachineGadgetGraph>
X86LoadValueInjectionLoadHardeningPass::getGadgetGraph(
    MachineFunction &MF, const MachineLoopInfo &MLI,
    const MachineDominatorTree &MDT,
    const MachineDominanceFrontier &MDF) const {
  using namespace rdf;

  // Build the Register Dataflow Graph using the RDF framework
  TargetOperandInfo TOI{*TII};
  DataFlowGraph DFG{MF, *TII, *TRI, MDT, MDF, TOI};
  DFG.build();
  Liveness L{MF.getRegInfo(), DFG};
  L.computePhiInfo();

  GraphBuilder Builder;
  using GraphIter = typename GraphBuilder::BuilderNodeRef;
  DenseMap<MachineInstr *, GraphIter> NodeMap;
  int FenceCount = 0, GadgetCount = 0;
  auto MaybeAddNode = [&NodeMap, &Builder](MachineInstr *MI) {
    auto Ref = NodeMap.find(MI);
    if (Ref == NodeMap.end()) {
      auto I = Builder.addVertex(MI);
      NodeMap[MI] = I;
      return std::pair<GraphIter, bool>{I, true};
    }
    return std::pair<GraphIter, bool>{Ref->getSecond(), false};
  };

  // The `Transmitters` map memoizes transmitters found for each def. If a def
  // has not yet been analyzed, then it will not appear in the map. If a def
  // has been analyzed and was determined not to have any transmitters, then
  // its list of transmitters will be empty.
  DenseMap<NodeId, std::vector<NodeId>> Transmitters;

  // Analyze all machine instructions to find gadgets and LFENCEs, adding
  // each interesting value to `Nodes`
  auto AnalyzeDef = [&](NodeAddr<DefNode *> SourceDef) {
    SmallSet<NodeId, 8> UsesVisited, DefsVisited;
    std::function<void(NodeAddr<DefNode *>)> AnalyzeDefUseChain =
        [&](NodeAddr<DefNode *> Def) {
          if (Transmitters.find(Def.Id) != Transmitters.end())
            return; // Already analyzed `Def`

          // Use RDF to find all the uses of `Def`
          rdf::NodeSet Uses;
          RegisterRef DefReg = DFG.getPRI().normalize(Def.Addr->getRegRef(DFG));
          for (auto UseID : L.getAllReachedUses(DefReg, Def)) {
            auto Use = DFG.addr<UseNode *>(UseID);
            if (Use.Addr->getFlags() & NodeAttrs::PhiRef) { // phi node
              NodeAddr<PhiNode *> Phi = Use.Addr->getOwner(DFG);
              for (auto I : L.getRealUses(Phi.Id)) {
                if (DFG.getPRI().alias(RegisterRef(I.first), DefReg)) {
                  for (auto UA : I.second)
                    Uses.emplace(UA.first);
                }
              }
            } else { // not a phi node
              Uses.emplace(UseID);
            }
          }

          // For each use of `Def`, we want to know whether:
          // (1) The use can leak the Def'ed value,
          // (2) The use can further propagate the Def'ed value to more defs
          for (auto UseID : Uses) {
            if (!UsesVisited.insert(UseID).second)
              continue; // Already visited this use of `Def`

            auto Use = DFG.addr<UseNode *>(UseID);
            assert(!(Use.Addr->getFlags() & NodeAttrs::PhiRef));
            MachineOperand &UseMO = Use.Addr->getOp();
            MachineInstr &UseMI = *UseMO.getParent();
            assert(UseMO.isReg());

            // We naively assume that an instruction propagates any loaded
            // uses to all defs unless the instruction is a call, in which
            // case all arguments will be treated as gadget sources during
            // analysis of the callee function.
            if (UseMI.isCall())
              continue;

            // Check whether this use can transmit (leak) its value.
            if (instrUsesRegToAccessMemory(UseMI, UseMO.getReg()) ||
                (!NoConditionalBranches &&
                 instrUsesRegToBranch(UseMI, UseMO.getReg()))) {
              Transmitters[Def.Id].push_back(Use.Addr->getOwner(DFG).Id);
              if (UseMI.mayLoad())
                continue; // Found a transmitting load -- no need to continue
                          // traversing its defs (i.e., this load will become
                          // a new gadget source anyways).
            }

            // Check whether the use propagates to more defs.
            NodeAddr<InstrNode *> Owner{Use.Addr->getOwner(DFG)};
            rdf::NodeList AnalyzedChildDefs;
            for (auto &ChildDef :
                 Owner.Addr->members_if(DataFlowGraph::IsDef, DFG)) {
              if (!DefsVisited.insert(ChildDef.Id).second)
                continue; // Already visited this def
              if (Def.Addr->getAttrs() & NodeAttrs::Dead)
                continue;
              if (Def.Id == ChildDef.Id)
                continue; // `Def` uses itself (e.g., increment loop counter)

              AnalyzeDefUseChain(ChildDef);

              // `Def` inherits all of its child defs' transmitters.
              for (auto TransmitterId : Transmitters[ChildDef.Id])
                Transmitters[Def.Id].push_back(TransmitterId);
            }
          }

          // Note that this statement adds `Def.Id` to the map if no
          // transmitters were found for `Def`.
          auto &DefTransmitters = Transmitters[Def.Id];

          // Remove duplicate transmitters
          llvm::sort(DefTransmitters);
          DefTransmitters.erase(
              std::unique(DefTransmitters.begin(), DefTransmitters.end()),
              DefTransmitters.end());
        };

    // Find all of the transmitters
    AnalyzeDefUseChain(SourceDef);
    auto &SourceDefTransmitters = Transmitters[SourceDef.Id];
    if (SourceDefTransmitters.empty())
      return; // No transmitters for `SourceDef`

    MachineInstr *Source = SourceDef.Addr->getFlags() & NodeAttrs::PhiRef
                               ? MachineGadgetGraph::ArgNodeSentinel
                               : SourceDef.Addr->getOp().getParent();
    auto GadgetSource = MaybeAddNode(Source);
    // Each transmitter is a sink for `SourceDef`.
    for (auto TransmitterId : SourceDefTransmitters) {
      MachineInstr *Sink = DFG.addr<StmtNode *>(TransmitterId).Addr->getCode();
      auto GadgetSink = MaybeAddNode(Sink);
      // Add the gadget edge to the graph.
      Builder.addEdge(MachineGadgetGraph::GadgetEdgeSentinel,
                      GadgetSource.first, GadgetSink.first);
      ++GadgetCount;
    }
  };

  LLVM_DEBUG(dbgs() << "Analyzing def-use chains to find gadgets\n");
  // Analyze function arguments
  NodeAddr<BlockNode *> EntryBlock = DFG.getFunc().Addr->getEntryBlock(DFG);
  for (NodeAddr<PhiNode *> ArgPhi :
       EntryBlock.Addr->members_if(DataFlowGraph::IsPhi, DFG)) {
    NodeList Defs = ArgPhi.Addr->members_if(DataFlowGraph::IsDef, DFG);
    llvm::for_each(Defs, AnalyzeDef);
  }
  // Analyze every instruction in MF
  for (NodeAddr<BlockNode *> BA : DFG.getFunc().Addr->members(DFG)) {
    for (NodeAddr<StmtNode *> SA :
         BA.Addr->members_if(DataFlowGraph::IsCode<NodeAttrs::Stmt>, DFG)) {
      MachineInstr *MI = SA.Addr->getCode();
      if (isFence(MI)) {
        MaybeAddNode(MI);
        ++FenceCount;
      } else if (MI->mayLoad()) {
        NodeList Defs = SA.Addr->members_if(DataFlowGraph::IsDef, DFG);
        llvm::for_each(Defs, AnalyzeDef);
      }
    }
  }
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
  // ArgNodeSentinel is a pseudo-instruction that represents MF args in the
  // GadgetGraph
  GraphIter ArgNode = MaybeAddNode(MachineGadgetGraph::ArgNodeSentinel).first;
  TraverseCFG(&MF.front(), ArgNode, 0);
  std::unique_ptr<MachineGadgetGraph> G{Builder.get(FenceCount, GadgetCount)};
  LLVM_DEBUG(dbgs() << "Found " << G->nodes_size() << " nodes\n");
  return G;
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
