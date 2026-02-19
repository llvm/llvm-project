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

  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);

  setOperationAction(ISD::BR_CC, MVT::i32, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Expand);

  setLoadExtAction(ISD::SEXTLOAD, MVT::i32, MVT::i8, Expand);

  computeRegisterProperties(STI.getRegisterInfo());
}

SDValue SC32TargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RI = MF.getRegInfo();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_SC32);

  for (size_t I = 0; I < ArgLocs.size(); I++) {
    Register VReg = RI.createVirtualRegister(&SC32::GPRegClass);
    RI.addLiveIn(ArgLocs[I].getLocReg(), VReg);
    InVals.push_back(DAG.getCopyFromReg(Chain, DL, VReg, MVT::i32));
  }

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

  SmallVector<SDValue, 2> RetOps = {SDValue()};
  SDValue Glue;

  for (size_t I = 0; I < RVLocs.size(); I++) {
    Register Reg = RVLocs[I].getLocReg();
    Chain = DAG.getCopyToReg(Chain, DL, Reg, OutVals[I], Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(Reg, MVT::i32));
  }

  RetOps[0] = Chain;

  if (Glue.getNode()) {
    RetOps.push_back(Glue);
  }

  return DAG.getNode(SC32ISD::RET_GLUE, DL, MVT::Other, RetOps);
}

static SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) {
  SDValue Chain = Op.getOperand(0);
  SDValue CC = Op.getOperand(1);
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);

  SDLoc DL(Op);

  SDValue CompareFlag = DAG.getNode(SC32ISD::CMP, DL, MVT::Glue, LHS, RHS);
  return DAG.getNode(SC32ISD::JMP, DL, MVT::Other, Chain, CC, Dest,
                     CompareFlag);
}

SDValue SC32TargetLowering::LowerOperation(SDValue Op,
                                           SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("Should not custom lower this!");
  case ISD::BR_CC:
    return LowerBR_CC(Op, DAG);
  }
}
