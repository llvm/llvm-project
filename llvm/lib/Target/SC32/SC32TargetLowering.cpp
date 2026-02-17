#include "SC32TargetLowering.h"
#include "MCTargetDesc/SC32MCTargetDesc.h"
#include "SC32RegisterInfo.h"
#include "SC32SelectionDAGInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"

using namespace llvm;

#include "SC32GenCallingConv.inc"

SC32TargetLowering::SC32TargetLowering(const TargetMachine &TM,
                                       const TargetSubtargetInfo &STI)
    : TargetLowering(TM, STI) {
  addRegisterClass(MVT::i32, &SC32::GPRegClass);

  computeRegisterProperties(STI.getRegisterInfo());
}

SDValue SC32TargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  return Chain;
}

SDValue
SC32TargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                bool IsVarArg,
                                const SmallVectorImpl<ISD::OutputArg> &Outs,
                                const SmallVectorImpl<SDValue> &OutVals,
                                const SDLoc &DL, SelectionDAG &DAG) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC_SC32);

  SDValue Glue;

  for (size_t I = 0; I < RVLocs.size(); I++) {
    Chain =
        DAG.getCopyToReg(Chain, DL, RVLocs[I].getLocReg(), OutVals[I], Glue);
    Glue = Chain.getValue(1);
  }

  SmallVector<SDValue, 2> RetOps = {Chain};

  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(SC32ISD::RET_GLUE, DL, MVT::Other, RetOps);
}
