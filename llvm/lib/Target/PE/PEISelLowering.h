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
        RET_GLUE
    };

} // namespace PEISD

class PETargetLowering : public TargetLowering {
  const PESubtarget &Subtarget;

public:
  explicit PETargetLowering(const TargetMachine &TM, const PESubtarget &STI);
  const PESubtarget &getSubtarget() const { return Subtarget; }

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;
  const char *getTargetNodeName(unsigned Opcode) const override;
};

} // namespace llvm

#endif // PEISELLOWERING_H