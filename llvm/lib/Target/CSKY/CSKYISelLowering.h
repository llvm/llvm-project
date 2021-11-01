//===-- CSKYISelLowering.cpp - CSKY DAG Lowering Implementation  ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that CSKY uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CSKY_CSKYISELLOWERING_H
#define LLVM_LIB_TARGET_CSKY_CSKYISELLOWERING_H

#include "MCTargetDesc/CSKYBaseInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
class CSKYSubtarget;

namespace CSKYISD {
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  NIE,
  NIR,
  RET,
  BITCAST_TO_LOHI
};
}

class CSKYTargetLowering : public TargetLowering {
  const CSKYSubtarget &Subtarget;

public:
  explicit CSKYTargetLowering(const TargetMachine &TM,
                              const CSKYSubtarget &STI);

  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                         EVT VT) const override;

private:
  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                      bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      LLVMContext &Context) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;

  const char *getTargetNodeName(unsigned Opcode) const override;

  CCAssignFn *CCAssignFnForCall(CallingConv::ID CC, bool IsVarArg) const;
  CCAssignFn *CCAssignFnForReturn(CallingConv::ID CC, bool IsVarArg) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_CSKY_CSKYISELLOWERING_H
