//===--- ScheduleDAGSDNodes.cpp - Implement the ScheduleDAGSDNodes class --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the ScheduleDAG class, which is a base class used by
// scheduling implementation classes.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "llvm/CodeGen/ScheduleDAGSDNodes.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

ScheduleDAGSDNodes::ScheduleDAGSDNodes(MachineFunction &mf)
  : ScheduleDAG(mf) {
}

SUnit *ScheduleDAGSDNodes::Clone(SUnit *Old) {
  SUnit *SU = NewSUnit(Old->getNode());
  SU->OrigNode = Old->OrigNode;
  SU->Latency = Old->Latency;
  SU->isTwoAddress = Old->isTwoAddress;
  SU->isCommutable = Old->isCommutable;
  SU->hasPhysRegDefs = Old->hasPhysRegDefs;
  Old->isCloned = true;
  return SU;
}

/// CheckForPhysRegDependency - Check if the dependency between def and use of
/// a specified operand is a physical register dependency. If so, returns the
/// register and the cost of copying the register.
static void CheckForPhysRegDependency(SDNode *Def, SDNode *User, unsigned Op,
                                      const TargetRegisterInfo *TRI, 
                                      const TargetInstrInfo *TII,
                                      unsigned &PhysReg, int &Cost) {
  if (Op != 2 || User->getOpcode() != ISD::CopyToReg)
    return;

  unsigned Reg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
  if (TargetRegisterInfo::isVirtualRegister(Reg))
    return;

  unsigned ResNo = User->getOperand(2).getResNo();
  if (Def->isMachineOpcode()) {
    const TargetInstrDesc &II = TII->get(Def->getMachineOpcode());
    if (ResNo >= II.getNumDefs() &&
        II.ImplicitDefs[ResNo - II.getNumDefs()] == Reg) {
      PhysReg = Reg;
      const TargetRegisterClass *RC =
        TRI->getPhysicalRegisterRegClass(Reg, Def->getValueType(ResNo));
      Cost = RC->getCopyCost();
    }
  }
}

void ScheduleDAGSDNodes::BuildSchedUnits() {
  // During scheduling, the NodeId field of SDNode is used to map SDNodes
  // to their associated SUnits by holding SUnits table indices. A value
  // of -1 means the SDNode does not yet have an associated SUnit.
  unsigned NumNodes = 0;
  for (SelectionDAG::allnodes_iterator NI = DAG->allnodes_begin(),
       E = DAG->allnodes_end(); NI != E; ++NI) {
    NI->setNodeId(-1);
    ++NumNodes;
  }

  // Reserve entries in the vector for each of the SUnits we are creating.  This
  // ensure that reallocation of the vector won't happen, so SUnit*'s won't get
  // invalidated.
  // FIXME: Multiply by 2 because we may clone nodes during scheduling.
  // This is a temporary workaround.
  SUnits.reserve(NumNodes * 2);
  
  // Check to see if the scheduler cares about latencies.
  bool UnitLatencies = ForceUnitLatencies();

  for (SelectionDAG::allnodes_iterator NI = DAG->allnodes_begin(),
       E = DAG->allnodes_end(); NI != E; ++NI) {
    if (isPassiveNode(NI))  // Leaf node, e.g. a TargetImmediate.
      continue;
    
    // If this node has already been processed, stop now.
    if (NI->getNodeId() != -1) continue;
    
    SUnit *NodeSUnit = NewSUnit(NI);
    
    // See if anything is flagged to this node, if so, add them to flagged
    // nodes.  Nodes can have at most one flag input and one flag output.  Flags
    // are required the be the last operand and result of a node.
    
    // Scan up to find flagged preds.
    SDNode *N = NI;
    if (N->getNumOperands() &&
        N->getOperand(N->getNumOperands()-1).getValueType() == MVT::Flag) {
      do {
        N = N->getOperand(N->getNumOperands()-1).getNode();
        assert(N->getNodeId() == -1 && "Node already inserted!");
        N->setNodeId(NodeSUnit->NodeNum);
      } while (N->getNumOperands() &&
               N->getOperand(N->getNumOperands()-1).getValueType()== MVT::Flag);
    }
    
    // Scan down to find any flagged succs.
    N = NI;
    while (N->getValueType(N->getNumValues()-1) == MVT::Flag) {
      SDValue FlagVal(N, N->getNumValues()-1);
      
      // There are either zero or one users of the Flag result.
      bool HasFlagUse = false;
      for (SDNode::use_iterator UI = N->use_begin(), E = N->use_end(); 
           UI != E; ++UI)
        if (FlagVal.isOperandOf(*UI)) {
          HasFlagUse = true;
          assert(N->getNodeId() == -1 && "Node already inserted!");
          N->setNodeId(NodeSUnit->NodeNum);
          N = *UI;
          break;
        }
      if (!HasFlagUse) break;
    }
    
    // If there are flag operands involved, N is now the bottom-most node
    // of the sequence of nodes that are flagged together.
    // Update the SUnit.
    NodeSUnit->setNode(N);
    assert(N->getNodeId() == -1 && "Node already inserted!");
    N->setNodeId(NodeSUnit->NodeNum);

    // Assign the Latency field of NodeSUnit using target-provided information.
    if (UnitLatencies)
      NodeSUnit->Latency = 1;
    else
      ComputeLatency(NodeSUnit);
  }
}

void ScheduleDAGSDNodes::AddSchedEdges() {
  // Pass 2: add the preds, succs, etc.
  for (unsigned su = 0, e = SUnits.size(); su != e; ++su) {
    SUnit *SU = &SUnits[su];
    SDNode *MainNode = SU->getNode();
    
    if (MainNode->isMachineOpcode()) {
      unsigned Opc = MainNode->getMachineOpcode();
      const TargetInstrDesc &TID = TII->get(Opc);
      for (unsigned i = 0; i != TID.getNumOperands(); ++i) {
        if (TID.getOperandConstraint(i, TOI::TIED_TO) != -1) {
          SU->isTwoAddress = true;
          break;
        }
      }
      if (TID.isCommutable())
        SU->isCommutable = true;
    }
    
    // Find all predecessors and successors of the group.
    for (SDNode *N = SU->getNode(); N; N = N->getFlaggedNode()) {
      if (N->isMachineOpcode() &&
          TII->get(N->getMachineOpcode()).getImplicitDefs() &&
          CountResults(N) > TII->get(N->getMachineOpcode()).getNumDefs())
        SU->hasPhysRegDefs = true;
      
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
        SDNode *OpN = N->getOperand(i).getNode();
        if (isPassiveNode(OpN)) continue;   // Not scheduled.
        SUnit *OpSU = &SUnits[OpN->getNodeId()];
        assert(OpSU && "Node has no SUnit!");
        if (OpSU == SU) continue;           // In the same group.

        MVT OpVT = N->getOperand(i).getValueType();
        assert(OpVT != MVT::Flag && "Flagged nodes should be in same sunit!");
        bool isChain = OpVT == MVT::Other;

        unsigned PhysReg = 0;
        int Cost = 1;
        // Determine if this is a physical register dependency.
        CheckForPhysRegDependency(OpN, N, i, TRI, TII, PhysReg, Cost);
        assert((PhysReg == 0 || !isChain) &&
               "Chain dependence via physreg data?");
        // FIXME: See ScheduleDAGSDNodes::EmitCopyFromReg. For now, scheduler
        // emits a copy from the physical register to a virtual register unless
        // it requires a cross class copy (cost < 0). That means we are only
        // treating "expensive to copy" register dependency as physical register
        // dependency. This may change in the future though.
        if (Cost >= 0)
          PhysReg = 0;
        SU->addPred(SDep(OpSU, isChain ? SDep::Order : SDep::Data,
                         OpSU->Latency, PhysReg));
      }
    }
  }
}

/// BuildSchedGraph - Build the SUnit graph from the selection dag that we
/// are input.  This SUnit graph is similar to the SelectionDAG, but
/// excludes nodes that aren't interesting to scheduling, and represents
/// flagged together nodes with a single SUnit.
void ScheduleDAGSDNodes::BuildSchedGraph() {
  // Populate the SUnits array.
  BuildSchedUnits();
  // Compute all the scheduling dependencies between nodes.
  AddSchedEdges();
}

void ScheduleDAGSDNodes::ComputeLatency(SUnit *SU) {
  const InstrItineraryData &InstrItins = TM.getInstrItineraryData();
  
  // Compute the latency for the node.  We use the sum of the latencies for
  // all nodes flagged together into this SUnit.
  SU->Latency = 0;
  bool SawMachineOpcode = false;
  for (SDNode *N = SU->getNode(); N; N = N->getFlaggedNode())
    if (N->isMachineOpcode()) {
      SawMachineOpcode = true;
      SU->Latency +=
        InstrItins.getLatency(TII->get(N->getMachineOpcode()).getSchedClass());
    }
}

/// CountResults - The results of target nodes have register or immediate
/// operands first, then an optional chain, and optional flag operands (which do
/// not go into the resulting MachineInstr).
unsigned ScheduleDAGSDNodes::CountResults(SDNode *Node) {
  unsigned N = Node->getNumValues();
  while (N && Node->getValueType(N - 1) == MVT::Flag)
    --N;
  if (N && Node->getValueType(N - 1) == MVT::Other)
    --N;    // Skip over chain result.
  return N;
}

/// CountOperands - The inputs to target nodes have any actual inputs first,
/// followed by special operands that describe memory references, then an
/// optional chain operand, then an optional flag operand.  Compute the number
/// of actual operands that will go into the resulting MachineInstr.
unsigned ScheduleDAGSDNodes::CountOperands(SDNode *Node) {
  unsigned N = ComputeMemOperandsEnd(Node);
  while (N && isa<MemOperandSDNode>(Node->getOperand(N - 1).getNode()))
    --N; // Ignore MEMOPERAND nodes
  return N;
}

/// ComputeMemOperandsEnd - Find the index one past the last MemOperandSDNode
/// operand
unsigned ScheduleDAGSDNodes::ComputeMemOperandsEnd(SDNode *Node) {
  unsigned N = Node->getNumOperands();
  while (N && Node->getOperand(N - 1).getValueType() == MVT::Flag)
    --N;
  if (N && Node->getOperand(N - 1).getValueType() == MVT::Other)
    --N; // Ignore chain if it exists.
  return N;
}


void ScheduleDAGSDNodes::dumpNode(const SUnit *SU) const {
  if (!SU->getNode()) {
    cerr << "PHYS REG COPY\n";
    return;
  }

  SU->getNode()->dump(DAG);
  cerr << "\n";
  SmallVector<SDNode *, 4> FlaggedNodes;
  for (SDNode *N = SU->getNode()->getFlaggedNode(); N; N = N->getFlaggedNode())
    FlaggedNodes.push_back(N);
  while (!FlaggedNodes.empty()) {
    cerr << "    ";
    FlaggedNodes.back()->dump(DAG);
    cerr << "\n";
    FlaggedNodes.pop_back();
  }
}
