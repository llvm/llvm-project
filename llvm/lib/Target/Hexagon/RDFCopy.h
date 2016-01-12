//===--- RDFCopy.h --------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef RDF_COPY_H
#define RDF_COPY_H

#include "RDFGraph.h"
#include <map>
#include <vector>

namespace llvm {
  class MachineBasicBlock;
  class MachineDominatorTree;
  class MachineInstr;
}

namespace rdf {
  struct CopyPropagation {
    CopyPropagation(DataFlowGraph &dfg) : MDT(dfg.getDT()), DFG(dfg),
        Trace(false) {}

    bool run();
    void trace(bool On) { Trace = On; }
    bool trace() const { return Trace; }

  private:
    const MachineDominatorTree &MDT;
    DataFlowGraph &DFG;
    DataFlowGraph::DefStackMap DefM;
    bool Trace;

    // map: register -> (map: stmt -> reaching def)
    std::map<RegisterRef,std::map<NodeId,NodeId>> RDefMap;
    std::vector<NodeId> Copies;

    void recordCopy(NodeAddr<StmtNode*> SA, MachineInstr *MI);
    void updateMap(NodeAddr<InstrNode*> IA);
    bool scanBlock(MachineBasicBlock *B);
  };
}

#endif
