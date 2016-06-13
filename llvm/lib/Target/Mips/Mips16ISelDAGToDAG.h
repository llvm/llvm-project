//===---- Mips16ISelDAGToDAG.h - A Dag to Dag Inst Selector for Mips ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Subclass of MipsDAGToDAGISel specialized for mips16.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MIPS16ISELDAGTODAG_H
#define LLVM_LIB_TARGET_MIPS_MIPS16ISELDAGTODAG_H

#include "MipsISelDAGToDAG.h"

namespace llvm {

class Mips16DAGToDAGISel : public MipsDAGToDAGISel {
public:
  explicit Mips16DAGToDAGISel(MipsTargetMachine &TM) : MipsDAGToDAGISel(TM) {}

private:
  std::pair<SDNode *, SDNode *> selectMULT(SDNode *N, unsigned Opc,
                                           const SDLoc &DL, EVT Ty, bool HasLo,
                                           bool HasHi);

  SDValue getMips16SPAliasReg();

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getMips16SPRefReg(SDNode *Parent, SDValue &AliasReg);

  bool selectAddr16(SDNode *Parent, SDValue N, SDValue &Base,
                    SDValue &Offset, SDValue &Alias) override;

  bool trySelect(SDNode *Node) override;

  void processFunctionAfterISel(MachineFunction &MF) override;

  // Insert instructions to initialize the global base register in the
  // first MBB of the function.
  void initGlobalBaseReg(MachineFunction &MF);

  void initMips16SPAliasReg(MachineFunction &MF);
};

FunctionPass *createMips16ISelDag(MipsTargetMachine &TM);

}

#endif
