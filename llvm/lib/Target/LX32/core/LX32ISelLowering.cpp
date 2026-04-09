//===-- LX32ISelLowering.cpp - LX32 SelectionDAG Lowering ----------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "LX32ISelLowering.h"

#include "LX32RegisterInfo.h"
#include "LX32Subtarget.h"

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "lx32-lower"

using namespace llvm;

#include "../TableGen/LX32GenCallingConv.inc"

static SDValue lowerCCValue(SDValue Val, CCValAssign::LocInfo LocInfo,
                            EVT ValVT, SelectionDAG &DAG,
                            const SDLoc &DL) {
  switch (LocInfo) {
  case CCValAssign::Full:
    return Val;
  case CCValAssign::BCvt:
    return DAG.getNode(ISD::BITCAST, DL, ValVT, Val);
  case CCValAssign::SExt:
    if (Val.getValueType() == ValVT)
      return Val;
    return DAG.getNode(ISD::AssertSext, DL, Val.getValueType(), Val,
                       DAG.getValueType(ValVT));
  case CCValAssign::ZExt:
    if (Val.getValueType() == ValVT)
      return Val;
    return DAG.getNode(ISD::AssertZext, DL, Val.getValueType(), Val,
                       DAG.getValueType(ValVT));
  case CCValAssign::AExt:
    if (Val.getValueType() == ValVT)
      return Val;
    return DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
  default:
    report_fatal_error("lx32: unsupported CC value location info");
  }
}

LX32TargetLowering::LX32TargetLowering(const TargetMachine &TM,
                                       const LX32Subtarget &STI)
    : TargetLowering(TM, STI), STI(STI) {
  addRegisterClass(MVT::i32, &LX32::GPRRegClass);
  setStackPointerRegisterToSaveRestore(LX32::X2);

  setOperationAction(ISD::SDIV, MVT::i32, Expand);
  setOperationAction(ISD::UDIV, MVT::i32, Expand);
  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);

  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTR, MVT::i32, Expand);
  setOperationAction(ISD::CTLZ, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);

  // First functional slice: keep select lowering on generic expansion.
  setOperationAction(ISD::SELECT, MVT::i32, Expand);

  setOperationAction(ISD::GlobalAddress, MVT::i32, Expand);
  setOperationAction(ISD::BlockAddress, MVT::i32, Expand);
  setOperationAction(ISD::ConstantPool, MVT::i32, Expand);

  setMaxAtomicSizeInBitsSupported(0);

  computeRegisterProperties(STI.getRegisterInfo());
}

const char *LX32TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  case LX32ISD::RET:
    return "LX32ISD::RET";
  case LX32ISD::CALL:
    return "LX32ISD::CALL";
  case LX32ISD::SELECT_CC:
    return "LX32ISD::SELECT_CC";
  default:
    return nullptr;
  }
}

SDValue LX32TargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  if (IsVarArg)
    report_fatal_error("lx32: varargs lowering is not implemented yet");

  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_LX32);

  for (unsigned I = 0, E = Ins.size(); I != E; ++I) {
    const CCValAssign &VA = ArgLocs[I];

    SDValue Val;
    if (VA.isRegLoc()) {
      Register VReg = RegInfo.createVirtualRegister(&LX32::GPRRegClass);
      RegInfo.addLiveIn(VA.getLocReg(), VReg);

      SDValue Arg = DAG.getCopyFromReg(Chain, DL, VReg, VA.getLocVT());
      Chain = Arg.getValue(1);
      Val = Arg;
    } else {
      int FI = MFI.CreateFixedObject(VA.getLocVT().getStoreSize(),
                                     VA.getLocMemOffset(), true);
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue Ld = DAG.getLoad(VA.getLocVT(), DL, Chain, FIN,
                               MachinePointerInfo::getFixedStack(MF, FI));
      Chain = Ld.getValue(1);
      Val = Ld;
    }

    EVT ValVT = Ins[I].VT;
    Val = lowerCCValue(Val, VA.getLocInfo(), ValVT, DAG, DL);
    if (Val.getValueType() != ValVT)
      Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
    InVals.push_back(Val);
  }

  return Chain;
}

SDValue LX32TargetLowering::LowerCall(
    TargetLowering::CallLoweringInfo &CLI,
    SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc DL = CLI.DL;
  MachineFunction &MF = DAG.getMachineFunction();

  if (CLI.IsVarArg)
    report_fatal_error("lx32: varargs call lowering is not implemented yet");

  EVT PtrVT = getPointerTy(DAG.getDataLayout());

  SDValue Chain = CLI.Chain;
  SDValue Glue;

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CLI.CallConv, CLI.IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeCallOperands(CLI.Outs, CC_LX32);

  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    const CCValAssign &VA = ArgLocs[I];
    SDValue Val = CLI.OutVals[I];

    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::SExt:
      Val = DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Val);
      break;
    case CCValAssign::ZExt:
      Val = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Val);
      break;
    case CCValAssign::AExt:
      Val = DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Val);
      break;
    case CCValAssign::BCvt:
      Val = DAG.getNode(ISD::BITCAST, DL, VA.getLocVT(), Val);
      break;
    default:
      report_fatal_error("lx32: unsupported call argument location info");
    }

    if (!VA.isRegLoc())
      report_fatal_error("lx32: stack-passed call arguments are not implemented yet");

    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Val, Glue);
    Glue = Chain.getValue(1);
  }

  SDValue Callee = CLI.Callee;
  if (const auto *ES = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    Callee = DAG.getTargetExternalSymbol(ES->getSymbol(), PtrVT);
  } else if (const auto *GA = dyn_cast<GlobalAddressSDNode>(Callee)) {
    Callee = DAG.getTargetGlobalAddress(GA->getGlobal(), DL, PtrVT, GA->getOffset());
  } else {
    report_fatal_error("lx32: only direct global/external calls are supported");
  }

  SmallVector<SDValue, 4> CallOps;
  CallOps.push_back(Chain);
  CallOps.push_back(Callee);
  if (Glue)
    CallOps.push_back(Glue);

  SDValue Call = DAG.getNode(LX32ISD::CALL, DL, DAG.getVTList(MVT::Other, MVT::Glue),
                             CallOps);
  Chain = Call.getValue(0);
  Glue = Call.getValue(1);

  SmallVector<CCValAssign, 4> RetLocs;
  CCState RetCC(CLI.CallConv, CLI.IsVarArg, MF, RetLocs, *DAG.getContext());
  RetCC.AnalyzeCallResult(CLI.Ins, RetCC_LX32);

  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    const CCValAssign &VA = RetLocs[I];
    SDValue Ret = DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT(), Glue);
    Chain = Ret.getValue(1);
    Glue = Ret.getValue(2);
    InVals.push_back(lowerCCValue(Ret, VA.getLocInfo(), CLI.Ins[I].VT, DAG, DL));
  }

  return Chain;
}

SDValue LX32TargetLowering::LowerReturn(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
    SelectionDAG &DAG) const {
  if (IsVarArg)
    report_fatal_error("lx32: varargs return lowering is not implemented yet");

  MachineFunction &MF = DAG.getMachineFunction();
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCC(CallConv, IsVarArg, MF, RetLocs, *DAG.getContext());
  RetCC.AnalyzeReturn(Outs, RetCC_LX32);

  SDValue Flag;
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    const CCValAssign &VA = RetLocs[I];
    SDValue Val = OutVals[I];

    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::SExt:
      Val = DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Val);
      break;
    case CCValAssign::ZExt:
      Val = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Val);
      break;
    case CCValAssign::AExt:
      Val = DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Val);
      break;
    case CCValAssign::BCvt:
      Val = DAG.getNode(ISD::BITCAST, DL, VA.getLocVT(), Val);
      break;
    default:
      report_fatal_error("lx32: unsupported return value location info");
    }

    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Val, Flag);
    Flag = Chain.getValue(1);
  }

  SmallVector<SDValue, 4> RetOps;
  RetOps.push_back(Chain);
  if (Flag)
    RetOps.push_back(Flag);
  return DAG.getNode(LX32ISD::RET, DL, MVT::Other, RetOps);
}

SDValue LX32TargetLowering::LowerOperation(SDValue Op,
                                           SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("lx32: unexpected custom-lowered operation");
  }
}


