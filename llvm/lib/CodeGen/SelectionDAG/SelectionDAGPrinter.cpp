//===-- SelectionDAGPrinter.cpp - Implement SelectionDAG::viewGraph() -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAG::viewGraph method.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/ScheduleDAGSDNodes.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/config.h"
#include <fstream>
using namespace llvm;

namespace llvm {
  template<>
  struct DOTGraphTraits<SelectionDAG*> : public DefaultDOTGraphTraits {
    static bool hasEdgeDestLabels() {
      return true;
    }

    static unsigned numEdgeDestLabels(const void *Node) {
      return ((const SDNode *) Node)->getNumValues();
    }

    static std::string getEdgeDestLabel(const void *Node, unsigned i) {
      return ((const SDNode *) Node)->getValueType(i).getMVTString();
    }

    /// edgeTargetsEdgeSource - This method returns true if this outgoing edge
    /// should actually target another edge source, not a node.  If this method is
    /// implemented, getEdgeTarget should be implemented.
    template<typename EdgeIter>
    static bool edgeTargetsEdgeSource(const void *Node, EdgeIter I) {
      return true;
    }

    /// getEdgeTarget - If edgeTargetsEdgeSource returns true, this method is
    /// called to determine which outgoing edge of Node is the target of this
    /// edge.
    template<typename EdgeIter>
    static EdgeIter getEdgeTarget(const void *Node, EdgeIter I) {
      SDNode *TargetNode = *I;
      SDNodeIterator NI = SDNodeIterator::begin(TargetNode);
      std::advance(NI, I.getNode()->getOperand(I.getOperand()).getResNo());
      return NI;
    }

    static std::string getGraphName(const SelectionDAG *G) {
      return G->getMachineFunction().getFunction()->getName();
    }

    static bool renderGraphFromBottomUp() {
      return true;
    }
    
    static bool hasNodeAddressLabel(const SDNode *Node,
                                    const SelectionDAG *Graph) {
      return true;
    }
    
    /// If you want to override the dot attributes printed for a particular
    /// edge, override this method.
    template<typename EdgeIter>
    static std::string getEdgeAttributes(const void *Node, EdgeIter EI) {
      SDValue Op = EI.getNode()->getOperand(EI.getOperand());
      MVT VT = Op.getValueType();
      if (VT == MVT::Flag)
        return "color=red,style=bold";
      else if (VT == MVT::Other)
        return "color=blue,style=dashed";
      return "";
    }
    

    static std::string getNodeLabel(const SDNode *Node,
                                    const SelectionDAG *Graph);
    static std::string getNodeAttributes(const SDNode *N,
                                         const SelectionDAG *Graph) {
#ifndef NDEBUG
      const std::string &Attrs = Graph->getGraphAttrs(N);
      if (!Attrs.empty()) {
        if (Attrs.find("shape=") == std::string::npos)
          return std::string("shape=Mrecord,") + Attrs;
        else
          return Attrs;
      }
#endif
      return "shape=Mrecord";
    }

    static void addCustomGraphFeatures(SelectionDAG *G,
                                       GraphWriter<SelectionDAG*> &GW) {
      GW.emitSimpleNode(0, "plaintext=circle", "GraphRoot");
      if (G->getRoot().getNode())
        GW.emitEdge(0, -1, G->getRoot().getNode(), G->getRoot().getResNo(),
                    "color=blue,style=dashed");
    }
  };
}

std::string DOTGraphTraits<SelectionDAG*>::getNodeLabel(const SDNode *Node,
                                                        const SelectionDAG *G) {
  std::string Op = Node->getOperationName(G);

  if (const ConstantSDNode *CSDN = dyn_cast<ConstantSDNode>(Node)) {
    Op += ": " + utostr(CSDN->getZExtValue());
  } else if (const ConstantFPSDNode *CSDN = dyn_cast<ConstantFPSDNode>(Node)) {
    Op += ": " + ftostr(CSDN->getValueAPF());
  } else if (const GlobalAddressSDNode *GADN =
             dyn_cast<GlobalAddressSDNode>(Node)) {
    Op += ": " + GADN->getGlobal()->getName();
    if (int64_t Offset = GADN->getOffset()) {
      if (Offset > 0)
        Op += "+" + itostr(Offset);
      else
        Op += itostr(Offset);
    }
  } else if (const FrameIndexSDNode *FIDN = dyn_cast<FrameIndexSDNode>(Node)) {
    Op += " " + itostr(FIDN->getIndex());
  } else if (const JumpTableSDNode *JTDN = dyn_cast<JumpTableSDNode>(Node)) {
    Op += " " + itostr(JTDN->getIndex());
  } else if (const ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Node)){
    if (CP->isMachineConstantPoolEntry()) {
      Op += '<';
      {
        raw_string_ostream OSS(Op);
        OSS << *CP->getMachineCPVal();
      }
      Op += '>';
    } else {
      if (ConstantFP *CFP = dyn_cast<ConstantFP>(CP->getConstVal()))
        Op += "<" + ftostr(CFP->getValueAPF()) + ">";
      else if (ConstantInt *CI = dyn_cast<ConstantInt>(CP->getConstVal()))
        Op += "<" + utostr(CI->getZExtValue()) + ">";
      else {
        Op += '<';
        {
          raw_string_ostream OSS(Op);
          WriteAsOperand(OSS, CP->getConstVal(), false);
        }
        Op += '>';
      }
    }
    Op += " A=" + itostr(1 << CP->getAlignment());
  } else if (const BasicBlockSDNode *BBDN = dyn_cast<BasicBlockSDNode>(Node)) {
    Op = "BB: ";
    const Value *LBB = (const Value*)BBDN->getBasicBlock()->getBasicBlock();
    if (LBB)
      Op += LBB->getName();
    //Op += " " + (const void*)BBDN->getBasicBlock();
  } else if (const RegisterSDNode *R = dyn_cast<RegisterSDNode>(Node)) {
    if (G && R->getReg() != 0 &&
        TargetRegisterInfo::isPhysicalRegister(R->getReg())) {
      Op = Op + " " +
        G->getTarget().getRegisterInfo()->getName(R->getReg());
    } else {
      Op += " #" + utostr(R->getReg());
    }
  } else if (const DbgStopPointSDNode *D = dyn_cast<DbgStopPointSDNode>(Node)) {
    DICompileUnit CU(cast<GlobalVariable>(D->getCompileUnit()));
    Op += ": " + CU.getFilename();
    Op += ":" + utostr(D->getLine());
    if (D->getColumn() != 0)
      Op += ":" + utostr(D->getColumn());
  } else if (const LabelSDNode *L = dyn_cast<LabelSDNode>(Node)) {
    Op += ": LabelID=" + utostr(L->getLabelID());
  } else if (const CallSDNode *C = dyn_cast<CallSDNode>(Node)) {
    Op += ": CallingConv=" + utostr(C->getCallingConv());
    if (C->isVarArg())
      Op += ", isVarArg";
    if (C->isTailCall())
      Op += ", isTailCall";
  } else if (const ExternalSymbolSDNode *ES =
             dyn_cast<ExternalSymbolSDNode>(Node)) {
    Op += "'" + std::string(ES->getSymbol()) + "'";
  } else if (const SrcValueSDNode *M = dyn_cast<SrcValueSDNode>(Node)) {
    if (M->getValue())
      Op += "<" + M->getValue()->getName() + ">";
    else
      Op += "<null>";
  } else if (const MemOperandSDNode *M = dyn_cast<MemOperandSDNode>(Node)) {
    const Value *V = M->MO.getValue();
    Op += '<';
    if (!V) {
      Op += "(unknown)";
    } else if (const PseudoSourceValue *PSV = dyn_cast<PseudoSourceValue>(V)) {
      // PseudoSourceValues don't have names, so use their print method.
      raw_string_ostream OSS(Op);
      PSV->print(OSS);
    } else {
      Op += V->getName();
    }
    Op += '+' + itostr(M->MO.getOffset()) + '>';
  } else if (const ARG_FLAGSSDNode *N = dyn_cast<ARG_FLAGSSDNode>(Node)) {
    Op = Op + " AF=" + N->getArgFlags().getArgFlagsString();
  } else if (const VTSDNode *N = dyn_cast<VTSDNode>(Node)) {
    Op = Op + " VT=" + N->getVT().getMVTString();
  } else if (const LoadSDNode *LD = dyn_cast<LoadSDNode>(Node)) {
    bool doExt = true;
    switch (LD->getExtensionType()) {
    default: doExt = false; break;
    case ISD::EXTLOAD:
      Op = Op + "<anyext ";
      break;
    case ISD::SEXTLOAD:
      Op = Op + " <sext ";
      break;
    case ISD::ZEXTLOAD:
      Op = Op + " <zext ";
      break;
    }
    if (doExt)
      Op += LD->getMemoryVT().getMVTString() + ">";
    if (LD->isVolatile())
      Op += "<V>";
    Op += LD->getIndexedModeName(LD->getAddressingMode());
    if (LD->getAlignment() > 1)
      Op += " A=" + utostr(LD->getAlignment());
  } else if (const StoreSDNode *ST = dyn_cast<StoreSDNode>(Node)) {
    if (ST->isTruncatingStore())
      Op += "<trunc " + ST->getMemoryVT().getMVTString() + ">";
    if (ST->isVolatile())
      Op += "<V>";
    Op += ST->getIndexedModeName(ST->getAddressingMode());
    if (ST->getAlignment() > 1)
      Op += " A=" + utostr(ST->getAlignment());
  }

#if 0
  Op += " Id=" + itostr(Node->getNodeId());
#endif
  
  return Op;
}


/// viewGraph - Pop up a ghostview window with the reachable parts of the DAG
/// rendered using 'dot'.
///
void SelectionDAG::viewGraph(const std::string &Title) {
// This code is only for debugging!
#ifndef NDEBUG
  ViewGraph(this, "dag." + getMachineFunction().getFunction()->getName(),
            Title);
#else
  cerr << "SelectionDAG::viewGraph is only available in debug builds on "
       << "systems with Graphviz or gv!\n";
#endif  // NDEBUG
}

// This overload is defined out-of-line here instead of just using a
// default parameter because this is easiest for gdb to call.
void SelectionDAG::viewGraph() {
  viewGraph("");
}

/// clearGraphAttrs - Clear all previously defined node graph attributes.
/// Intended to be used from a debugging tool (eg. gdb).
void SelectionDAG::clearGraphAttrs() {
#ifndef NDEBUG
  NodeGraphAttrs.clear();
#else
  cerr << "SelectionDAG::clearGraphAttrs is only available in debug builds"
       << " on systems with Graphviz or gv!\n";
#endif
}


/// setGraphAttrs - Set graph attributes for a node. (eg. "color=red".)
///
void SelectionDAG::setGraphAttrs(const SDNode *N, const char *Attrs) {
#ifndef NDEBUG
  NodeGraphAttrs[N] = Attrs;
#else
  cerr << "SelectionDAG::setGraphAttrs is only available in debug builds"
       << " on systems with Graphviz or gv!\n";
#endif
}


/// getGraphAttrs - Get graph attributes for a node. (eg. "color=red".)
/// Used from getNodeAttributes.
const std::string SelectionDAG::getGraphAttrs(const SDNode *N) const {
#ifndef NDEBUG
  std::map<const SDNode *, std::string>::const_iterator I =
    NodeGraphAttrs.find(N);
    
  if (I != NodeGraphAttrs.end())
    return I->second;
  else
    return "";
#else
  cerr << "SelectionDAG::getGraphAttrs is only available in debug builds"
       << " on systems with Graphviz or gv!\n";
  return std::string("");
#endif
}

/// setGraphColor - Convenience for setting node color attribute.
///
void SelectionDAG::setGraphColor(const SDNode *N, const char *Color) {
#ifndef NDEBUG
  NodeGraphAttrs[N] = std::string("color=") + Color;
#else
  cerr << "SelectionDAG::setGraphColor is only available in debug builds"
       << " on systems with Graphviz or gv!\n";
#endif
}

/// setSubgraphColorHelper - Implement setSubgraphColor.  Return
/// whether we truncated the search.
///
bool SelectionDAG::setSubgraphColorHelper(SDNode *N, const char *Color, DenseSet<SDNode *> &visited,
                                          int level, bool &printed) {
  bool hit_limit = false;

#ifndef NDEBUG
  if (level >= 20) {
    if (!printed) {
      printed = true;
      DOUT << "setSubgraphColor hit max level\n";
    }
    return true;
  }

  unsigned oldSize = visited.size();
  visited.insert(N);
  if (visited.size() != oldSize) {
    setGraphColor(N, Color);
    for(SDNodeIterator i = SDNodeIterator::begin(N), iend = SDNodeIterator::end(N);
        i != iend;
        ++i) {
      hit_limit = setSubgraphColorHelper(*i, Color, visited, level+1, printed) || hit_limit;
    }
  }
#else
  cerr << "SelectionDAG::setSubgraphColor is only available in debug builds"
       << " on systems with Graphviz or gv!\n";
#endif
  return hit_limit;
}

/// setSubgraphColor - Convenience for setting subgraph color attribute.
///
void SelectionDAG::setSubgraphColor(SDNode *N, const char *Color) {
#ifndef NDEBUG
  DenseSet<SDNode *> visited;
  bool printed = false;
  if (setSubgraphColorHelper(N, Color, visited, 0, printed)) {
    // Visually mark that we hit the limit
    if (strcmp(Color, "red") == 0) {
      setSubgraphColorHelper(N, "blue", visited, 0, printed);
    }
    else if (strcmp(Color, "yellow") == 0) {
      setSubgraphColorHelper(N, "green", visited, 0, printed);
    }
  }

#else
  cerr << "SelectionDAG::setSubgraphColor is only available in debug builds"
       << " on systems with Graphviz or gv!\n";
#endif
}

std::string ScheduleDAGSDNodes::getGraphNodeLabel(const SUnit *SU) const {
  std::string s;
  raw_string_ostream O(s);
  O << "SU(" << SU->NodeNum << "): ";
  if (SU->getNode()) {
    SmallVector<SDNode *, 4> FlaggedNodes;
    for (SDNode *N = SU->getNode(); N; N = N->getFlaggedNode())
      FlaggedNodes.push_back(N);
    while (!FlaggedNodes.empty()) {
      O << DOTGraphTraits<SelectionDAG*>::getNodeLabel(FlaggedNodes.back(), DAG);
      FlaggedNodes.pop_back();
      if (!FlaggedNodes.empty())
        O << "\n    ";
    }
  } else {
    O << "CROSS RC COPY";
  }
  return O.str();
}

void ScheduleDAGSDNodes::getCustomGraphFeatures(GraphWriter<ScheduleDAG*> &GW) const {
  if (DAG) {
    // Draw a special "GraphRoot" node to indicate the root of the graph.
    GW.emitSimpleNode(0, "plaintext=circle", "GraphRoot");
    const SDNode *N = DAG->getRoot().getNode();
    if (N && N->getNodeId() != -1)
      GW.emitEdge(0, -1, &SUnits[N->getNodeId()], -1,
                  "color=blue,style=dashed");
  }
}
