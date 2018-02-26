//===-- Nios2ISelLowering.h - Nios2 DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Nios2 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NIOS2_NIOS2ISELLOWERING_H
#define LLVM_LIB_TARGET_NIOS2_NIOS2ISELLOWERING_H

#include "Nios2.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
class Nios2Subtarget;

namespace Nios2ISD {
enum NodeType {
  // Start the numbering from where ISD NodeType finishes.
  FIRST_NUMBER = ISD::BUILTIN_OP_END,

  // Get the Higher 16 bits from a 32-bit immediate
  // No relation with Nios2 Hi register
  Hi,
  // Get the Lower 16 bits from a 32-bit immediate
  // No relation with Nios2 Lo register
  Lo,
  // Return
  Ret
};
}

class Nios2TargetLowering : public TargetLowering {
  const Nios2Subtarget *Subtarget;

public:
  Nios2TargetLowering(const TargetMachine &TM, const Nios2Subtarget &STI);

  /// getTargetNodeName - This method returns the name of a target specific
  //  DAG node.
  const char *getTargetNodeName(unsigned Opcode) const override;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &dl, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &dl,
                      SelectionDAG &DAG) const override;
};
} // end namespace llvm

#endif // NIOS2_ISELLOWERING_H
