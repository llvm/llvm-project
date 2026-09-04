//===--- RDFCopy.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_RDFCOPY_H
#define LLVM_LIB_TARGET_HEXAGON_RDFCOPY_H

#include "RDFCopyBase.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/RDFGraph.h"
#include "llvm/CodeGen/RDFLiveness.h"
#include "llvm/CodeGen/RDFRegisters.h"
#include <map>
#include <vector>

namespace llvm {

namespace rdf {

struct CopyPropagation : CopyPropagationBase {
  CopyPropagation(DataFlowGraph &dfg) : CopyPropagationBase(dfg) {}

  virtual ~CopyPropagation() = default;

  bool run();
  virtual bool interpretAsCopy(const MachineInstr *MI, EqualityMap &EM);

private:
  std::vector<NodeId> Copies;

  void recordCopy(NodeAddr<StmtNode *> SA, EqualityMap &EM);
  void updateMap(NodeAddr<InstrNode *> IA);
  bool scanBlock(MachineBasicBlock *B);
};

} // end namespace rdf
} // end namespace llvm

#endif // LLVM_LIB_TARGET_HEXAGON_RDFCOPY_H
