#ifndef LLVM_LIB_TARGET_SC32_SC32TARGETLOWERING_H
#define LLVM_LIB_TARGET_SC32_SC32TARGETLOWERING_H

#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class SC32TargetLowering : public TargetLowering {
public:
  SC32TargetLowering(const TargetMachine &TM, const TargetSubtargetInfo &STI);

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;

  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
};

} // namespace llvm

#endif
