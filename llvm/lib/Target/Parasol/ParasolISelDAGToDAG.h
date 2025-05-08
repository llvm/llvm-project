//===---- ParasolISelDAGToDAG.h - A Dag to Dag Inst Selector for Parasol --===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the Parasol target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PARASOL_PARASOLISELDAGTODAG_H
#define LLVM_LIB_TARGET_PARASOL_PARASOLISELDAGTODAG_H

#include "ParasolSubtarget.h"
#include "ParasolTargetMachine.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {
class ParasolDAGToDAGISel : public SelectionDAGISel {
public:
  static char ID;

  explicit ParasolDAGToDAGISel(ParasolTargetMachine &TM)
      : SelectionDAGISel(ID, TM), Subtarget(nullptr) {}

  // Pass Name
  StringRef getPassName() const override {
    return "CPU0 DAG->DAG Pattern Instruction Selection";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void Select(SDNode *Node) override;

#include "ParasolGenDAGISel.inc"

private:
  const ParasolSubtarget *Subtarget;
};

} // namespace llvm

#endif // end LLVM_LIB_TARGET_PARASOL_PARASOLISELDAGTODAG_H
