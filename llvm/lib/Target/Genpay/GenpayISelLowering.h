#ifndef LLVM_LIB_TARGET_GENPAY_GENPAYISELLOWERING_H
#define LLVM_LIB_TARGET_GENPAY_GENPAYISELLOWERING_H

#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class GenpaySubtarget;

namespace GenpayISD {
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  RET, ///< Return from function.
};
} // namespace GenpayISD

class GenpayTargetLowering : public TargetLowering {
public:
  GenpayTargetLowering(const TargetMachine &TM, const GenpaySubtarget &STI);

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals,
                      const SDLoc &DL, SelectionDAG &DAG) const override;
};

} // namespace llvm

#endif
