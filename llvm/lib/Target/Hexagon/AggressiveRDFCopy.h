//===--- AggressiveRDFCopy.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_AGGRESSIVE_RDFCOPY_H
#define LLVM_LIB_TARGET_HEXAGON_AGGRESSIVE_RDFCOPY_H

#include "HexagonSubtarget.h"
#include "RDFCopyBase.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/RDFGraph.h"
#include "llvm/CodeGen/RDFLiveness.h"
#include "llvm/CodeGen/RDFRegisters.h"
#include <vector>

namespace llvm {
class HexagonRegisterInfo;
namespace rdf {

struct AggressiveCopyPropagation : CopyPropagationBase {
  AggressiveCopyPropagation(DataFlowGraph &dfg)
      : CopyPropagationBase(dfg), PRI(dfg.getPRI()), TRI(dfg.getTRI()),
        HRI(*dfg.getMF().getSubtarget<HexagonSubtarget>().getRegisterInfo()) {}

  virtual ~AggressiveCopyPropagation() = default;

  bool run();
  virtual bool interpretAsCopy(const MachineInstr *MI, EqualityMap &EM);

private:
  const PhysicalRegisterInfo &PRI;
  const TargetRegisterInfo &TRI;
  const HexagonRegisterInfo &HRI;

  // map: reached use node id -> (copy def node, source reg used in copy)
  // Maps each use reached by a copy to the copy's def node and source register.
  // We might only want to propagate a sub register of the source register.
  using CopyPair = std::pair<NodeAddr<DefNode *>, RegisterRef>;
  DenseMap<NodeId, CopyPair> ReachedUseToCopyMap;

  // Copy propagation can be applied to uses found in this vector
  // Vector of pairs, where each pair:
  // Use Node -> vector of refs that are to be added as Use Nodes
  // in the updated instruction
  using RegRefList = SmallVector<RegisterRef, 4>;
  using ReplacableUse = std::pair<NodeAddr<UseNode *>, RegRefList>;
  SmallVector<ReplacableUse, 16> ReplacableUses;

  void recordCopy(NodeAddr<StmtNode *> SA, EqualityMap &EM);
  void recordReplacableUses(NodeAddr<InstrNode *> IA);
  void scanBlock(MachineBasicBlock *B);
};

} // end namespace rdf
} // end namespace llvm

#endif // LLVM_LIB_TARGET_HEXAGON_AGGRESSIVE_RDFCOPY_H
