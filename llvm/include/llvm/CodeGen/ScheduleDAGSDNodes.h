//===---- llvm/CodeGen/ScheduleDAGSDNodes.h - SDNode Scheduling -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleDAGSDNodes class, which implements
// scheduling for an SDNode-based dependency graph.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDAGSDNODES_H
#define LLVM_CODEGEN_SCHEDULEDAGSDNODES_H

#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SelectionDAG.h"

namespace llvm {
  /// ScheduleDAGSDNodes - A ScheduleDAG for scheduling SDNode-based DAGs.
  /// 
  /// Edges between SUnits are initially based on edges in the SelectionDAG,
  /// and additional edges can be added by the schedulers as heuristics.
  /// SDNodes such as Constants, Registers, and a few others that are not
  /// interesting to schedulers are not allocated SUnits.
  ///
  /// SDNodes with MVT::Flag operands are grouped along with the flagged
  /// nodes into a single SUnit so that they are scheduled together.
  ///
  /// SDNode-based scheduling graphs do not use SDep::Anti or SDep::Output
  /// edges.  Physical register dependence information is not carried in
  /// the DAG and must be handled explicitly by schedulers.
  ///
  class ScheduleDAGSDNodes : public ScheduleDAG {
  public:
    explicit ScheduleDAGSDNodes(MachineFunction &mf);

    virtual ~ScheduleDAGSDNodes() {}

    /// isPassiveNode - Return true if the node is a non-scheduled leaf.
    ///
    static bool isPassiveNode(SDNode *Node) {
      if (isa<ConstantSDNode>(Node))       return true;
      if (isa<ConstantFPSDNode>(Node))     return true;
      if (isa<RegisterSDNode>(Node))       return true;
      if (isa<GlobalAddressSDNode>(Node))  return true;
      if (isa<BasicBlockSDNode>(Node))     return true;
      if (isa<FrameIndexSDNode>(Node))     return true;
      if (isa<ConstantPoolSDNode>(Node))   return true;
      if (isa<JumpTableSDNode>(Node))      return true;
      if (isa<ExternalSymbolSDNode>(Node)) return true;
      if (isa<MemOperandSDNode>(Node))     return true;
      if (Node->getOpcode() == ISD::EntryToken) return true;
      return false;
    }

    /// NewSUnit - Creates a new SUnit and return a ptr to it.
    ///
    SUnit *NewSUnit(SDNode *N) {
#ifndef NDEBUG
      const SUnit *Addr = 0;
      if (!SUnits.empty())
        Addr = &SUnits[0];
#endif
      SUnits.push_back(SUnit(N, (unsigned)SUnits.size()));
      assert((Addr == 0 || Addr == &SUnits[0]) &&
             "SUnits std::vector reallocated on the fly!");
      SUnits.back().OrigNode = &SUnits.back();
      return &SUnits.back();
    }

    /// Clone - Creates a clone of the specified SUnit. It does not copy the
    /// predecessors / successors info nor the temporary scheduling states.
    ///
    SUnit *Clone(SUnit *N);
    
    virtual SelectionDAG *getDAG() { return DAG; }

    /// BuildSchedGraph - Build the SUnit graph from the selection dag that we
    /// are input.  This SUnit graph is similar to the SelectionDAG, but
    /// excludes nodes that aren't interesting to scheduling, and represents
    /// flagged together nodes with a single SUnit.
    virtual void BuildSchedGraph();

    /// ComputeLatency - Compute node latency.
    ///
    virtual void ComputeLatency(SUnit *SU);

    /// CountResults - The results of target nodes have register or immediate
    /// operands first, then an optional chain, and optional flag operands
    /// (which do not go into the machine instrs.)
    static unsigned CountResults(SDNode *Node);

    /// CountOperands - The inputs to target nodes have any actual inputs first,
    /// followed by special operands that describe memory references, then an
    /// optional chain operand, then flag operands.  Compute the number of
    /// actual operands that will go into the resulting MachineInstr.
    static unsigned CountOperands(SDNode *Node);

    /// ComputeMemOperandsEnd - Find the index one past the last
    /// MemOperandSDNode operand
    static unsigned ComputeMemOperandsEnd(SDNode *Node);

    /// EmitNode - Generate machine code for an node and needed dependencies.
    /// VRBaseMap contains, for each already emitted node, the first virtual
    /// register number for the results of the node.
    ///
    void EmitNode(SDNode *Node, bool IsClone, bool HasClone,
                  DenseMap<SDValue, unsigned> &VRBaseMap);
    
    virtual MachineBasicBlock *EmitSchedule();

    /// Schedule - Order nodes according to selected style, filling
    /// in the Sequence member.
    ///
    virtual void Schedule() = 0;

    virtual void dumpNode(const SUnit *SU) const;

    virtual std::string getGraphNodeLabel(const SUnit *SU) const;

    virtual void getCustomGraphFeatures(GraphWriter<ScheduleDAG*> &GW) const;

  private:
    /// EmitSubregNode - Generate machine code for subreg nodes.
    ///
    void EmitSubregNode(SDNode *Node, 
                        DenseMap<SDValue, unsigned> &VRBaseMap);

    /// getVR - Return the virtual register corresponding to the specified result
    /// of the specified node.
    unsigned getVR(SDValue Op, DenseMap<SDValue, unsigned> &VRBaseMap);
  
    /// getDstOfCopyToRegUse - If the only use of the specified result number of
    /// node is a CopyToReg, return its destination register. Return 0 otherwise.
    unsigned getDstOfOnlyCopyToRegUse(SDNode *Node, unsigned ResNo) const;

    void AddOperand(MachineInstr *MI, SDValue Op, unsigned IIOpNum,
                    const TargetInstrDesc *II,
                    DenseMap<SDValue, unsigned> &VRBaseMap);

    /// EmitCopyFromReg - Generate machine code for an CopyFromReg node or an
    /// implicit physical register output.
    void EmitCopyFromReg(SDNode *Node, unsigned ResNo, bool IsClone,
                         bool IsCloned, unsigned SrcReg,
                         DenseMap<SDValue, unsigned> &VRBaseMap);
    
    void CreateVirtualRegisters(SDNode *Node, MachineInstr *MI,
                                const TargetInstrDesc &II, bool IsClone,
                                bool IsCloned,
                                DenseMap<SDValue, unsigned> &VRBaseMap);

    /// BuildSchedUnits, AddSchedEdges - Helper functions for BuildSchedGraph.
    void BuildSchedUnits();
    void AddSchedEdges();
  };
}

#endif
