//===--- RDFCopy.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_RDFCOPY_H
#define LLVM_LIB_TARGET_HEXAGON_RDFCOPY_H

#include "RDFGraph.h"
#include <map>
#include <vector>

namespace llvm {

  class MachineBasicBlock;
  class MachineDominatorTree;
  class MachineInstr;

namespace rdf {

  struct CopyPropagation {
    CopyPropagation(DataFlowGraph &dfg) : MDT(dfg.getDT()), DFG(dfg),
        Trace(false) {}

    virtual ~CopyPropagation() = default;

    bool run();
    void trace(bool On) { Trace = On; }
    bool trace() const { return Trace; }
    DataFlowGraph &getDFG() { return DFG; }

    typedef std::map<RegisterRef, RegisterRef> EqualityMap;
    virtual bool interpretAsCopy(const MachineInstr *MI, EqualityMap &EM);

  private:
    const MachineDominatorTree &MDT;
    DataFlowGraph &DFG;
    DataFlowGraph::DefStackMap DefM;
    bool Trace;

    // map: register -> (map: stmt -> reaching def)
    std::map<RegisterRef,std::map<NodeId,NodeId>> RDefMap;
    // map: statement -> (map: dst reg -> src reg)
    std::map<NodeId, EqualityMap> CopyMap;
    std::vector<NodeId> Copies;

    void recordCopy(NodeAddr<StmtNode*> SA, EqualityMap &EM);
    void updateMap(NodeAddr<InstrNode*> IA);
    bool scanBlock(MachineBasicBlock *B);
  };

} // end namespace rdf

} // end namespace llvm

#endif // LLVM_LIB_TARGET_HEXAGON_RDFCOPY_H
