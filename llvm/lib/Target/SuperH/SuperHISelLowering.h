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
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/CodeGen/MachineConstantPool.h"

namespace llvm {
class SuperHSubtarget;
class MachineConstantPool;

class SuperHTargetLowering : public TargetLowering  {
  const SuperHSubtarget *Subtarget;

public:
  SuperHTargetLowering(const TargetMachine &TM, const SuperHSubtarget &STI);

  // LowerToConstantPool - SuperH's compressed instruction set means that 
  // immediates and displacements can not be larger than 8 bits. 
  // As such we need to store said immediates and displacements within 
  // constants that are within range of the program counter.
  //
  // As such this function is a helper that:
  //  1. Allocates a constant pool slot for the address
  //  2. Inserts the target address into said slot.
  //  3. Returns the neccesary, but non-legalized instruction sequence to fetch
  //     the address from that constant pool slot.
  template<class NodeTy>
  SDValue LowerToConstantPool(NodeTy* N, SelectionDAG &DAG) const;

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
  
  // Lowerings
  SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerExternalSymbol(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerDiv(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue getSHCmp(SDValue LHS, SDValue RHS, ISD::CondCode CC, SDValue &OutCC,
                   SelectionDAG &DAG, SDLoc DL) const;
private:

  SDValue getPICJumpTableRelocBase(SDValue Table, SelectionDAG &DAG) const override;

  // Lowerings
  SDValue LowerCallResult(
    SDValue Chain, SDValue InGlue, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const;
};

} // namespace llvm

#endif