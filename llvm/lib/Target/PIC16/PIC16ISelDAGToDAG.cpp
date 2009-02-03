//===-- PIC16ISelDAGToDAG.cpp - A dag to dag inst selector for PIC16 ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the PIC16 target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pic16-isel"

#include "PIC16ISelDAGToDAG.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

/// createPIC16ISelDag - This pass converts a legalized DAG into a
/// PIC16-specific DAG, ready for instruction scheduling.
FunctionPass *llvm::createPIC16ISelDag(PIC16TargetMachine &TM) {
  return new PIC16DAGToDAGISel(TM);
}


/// InstructionSelect - This callback is invoked by
/// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
void PIC16DAGToDAGISel::InstructionSelect() {
  DEBUG(BB->dump());
  SelectRoot(*CurDAG);
  CurDAG->RemoveDeadNodes();
}

/// Select - Select instructions not customized! Used for
/// expanded, promoted and normal instructions.
SDNode* PIC16DAGToDAGISel::Select(SDValue N) {

  // Select the default instruction.
  SDNode *ResNode = SelectCode(N);

  return ResNode;
}


// SelectDirectAddr - Match a direct address for DAG. 
// A direct address could be a globaladdress or externalsymbol.
bool PIC16DAGToDAGISel::SelectDirectAddr(SDValue Op, SDValue N, 
                                      SDValue &Address) {
  // Return true if TGA or ES.
  if (N.getOpcode() == ISD::TargetGlobalAddress
      || N.getOpcode() == ISD::TargetExternalSymbol) {
    Address = N;
    return true;
  }

  return false;
}
