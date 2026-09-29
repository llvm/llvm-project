//===-- LanaiISelLowering.h - Lanai DAG Lowering Interface -....-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Lanai uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LANAI_LANAIISELLOWERING_H
#define LLVM_LIB_TARGET_LANAI_LANAIISELLOWERING_H

#include "Lanai.h"
#include "LanaiRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class LanaiSubtarget;

class LanaiTargetLowering : public TargetLowering {
public:
  LanaiTargetLowering(const TargetMachine &TM, const LanaiSubtarget &STI);

  // LowerOperation - Provide custom lowering hooks for some operations.
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

  SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerMUL(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSHL_PARTS(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSRL_PARTS(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;

  bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                      bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      LLVMContext &Context, const Type *RetTy) const override;

  Register getRegisterByName(const char *RegName, LLT VT,
                             const MachineFunction &MF) const override;
  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;
  ConstraintWeight
  getSingleConstraintMatchWeight(AsmOperandInfo &Info,
                                 const char *Constraint) const override;
  void LowerAsmOperandForConstraint(SDValue Op, StringRef Constraint,
                                    std::vector<SDValue> &Ops,
                                    SelectionDAG &DAG) const override;

  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

  void computeKnownBitsForTargetNode(const SDValue Op, KnownBits &Known,
                                     const APInt &DemandedElts,
                                     const SelectionDAG &DAG,
                                     unsigned Depth = 0) const override;

private:
  SDValue LowerCCCCallTo(SDValue Chain, SDValue Callee,
                         CallingConv::ID CallConv, bool IsVarArg,
                         bool IsTailCall,
                         const SmallVectorImpl<ISD::OutputArg> &Outs,
                         const SmallVectorImpl<SDValue> &OutVals,
                         const SmallVectorImpl<ISD::InputArg> &Ins,
                         const SDLoc &dl, SelectionDAG &DAG,
                         SmallVectorImpl<SDValue> &InVals) const;

  SDValue LowerCCCArguments(SDValue Chain, CallingConv::ID CallConv,
                            bool IsVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            const SDLoc &DL, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals) const;

  SDValue LowerCallResult(SDValue Chain, SDValue InGlue,
                          CallingConv::ID CallConv, bool IsVarArg,
                          const SmallVectorImpl<ISD::InputArg> &Ins,
                          const SDLoc &DL, SelectionDAG &DAG,
                          SmallVectorImpl<SDValue> &InVals) const;

  SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;

  const LanaiRegisterInfo *TRI;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_LANAI_LANAIISELLOWERING_H
