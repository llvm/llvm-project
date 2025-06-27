/* --- PEISelLowering.h --- */

/* ------------------------------------------
Author: 高宇翔
Date: 4/3/2025
------------------------------------------ */

#ifndef PEISELLOWERING_H
#define PEISELLOWERING_H

#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class PESubtarget;

namespace PEISD {
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  RET_GLUE,
  Call,
  HI,
  LO,
  DIV,
  DIVR
  // RET_GLUE
};

} // namespace PEISD

class PETargetLowering : public TargetLowering {
  const PESubtarget &Subtarget;

public:
  explicit PETargetLowering(const TargetMachine &TM, const PESubtarget &STI);
  const PESubtarget &getSubtarget() const { return Subtarget; }

  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;
  SDValue LowerCall(CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;
  const char *getTargetNodeName(unsigned Opcode) const override;

  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                         EVT VT) const override;
};

} // namespace llvm

#endif // PEISELLOWERING_H