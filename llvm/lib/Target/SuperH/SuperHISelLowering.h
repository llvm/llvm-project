//===-- SuperHISelLowering.h - SH DAG Lowering Interface --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===//
//
// This file defines the interfaces that SuperH uses to lower LLVM code into a
// selection DAG.
//
//===-----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_SUPERHISELLOWERING_H
#define LLVM_LIB_TARGET_SUPERH_SUPERHISELLOWERING_H

#include "SuperH.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
class SuperHSubtarget;

class SuperHTargetLowering : public TargetLowering  {
  const SuperHSubtarget *Subtarget;

  SDValue LowerFormalArguments(SDValue Chain,
                         CallingConv::ID CallConv, bool IsVarArg,
                         const SmallVectorImpl<ISD::InputArg> &Ins,
                         const SDLoc &dl, SelectionDAG &DAG,
                         SmallVectorImpl<SDValue> &InVals) const override;
  SDValue LowerReturn(SDValue Chain,
                      CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals,
                      const SDLoc &dl, SelectionDAG &DAG) const override;
  bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF, bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context,
                      const Type *RetTy) const override;
  SDValue LowerCall(CallLoweringInfo &/*CLI*/,
              SmallVectorImpl<SDValue> &/*InVals*/) const override;

  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

  // Custom Lowerings
  SDValue LowerDiv(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerCallResult(
    SDValue Chain, SDValue InGlue, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const;
public:
  SuperHTargetLowering(const TargetMachine &TM, const SuperHSubtarget &STI);
};

} // namespace llvm

#endif