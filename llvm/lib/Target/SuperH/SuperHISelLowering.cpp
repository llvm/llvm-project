//===-- SuperHISelLowering.cpp - SH DAG Lowering Interface ------*- C++ -*-===//
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

#include "SuperHISelLowering.h"
#include "SuperHSelectionDAGInfo.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "SuperHRegisterInfo.h"
#include "SuperHTargetMachine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Support/DebugLog.h"

using namespace llvm;

#define DEBUG_TYPE "sh-lower"

static bool RetCC_SuperH_SRet(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                               CCValAssign::LocInfo &LocInfo,
                               ISD::ArgFlagsTy &ArgFlags, CCState &State) {
  assert (ArgFlags.isSRet());

  // Assign SRet argument.
  State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT,
                                         0,
                                         LocVT, LocInfo));
  return true;
}


#include "SuperHGenCallingConv.inc"

SuperHTargetLowering::SuperHTargetLowering(const TargetMachine &TM,
                                           const SuperHSubtarget &STI)
    : TargetLowering(TM, STI), Subtarget(&STI) {

  // GPR Registers are always 32 bit on SuperH.
  addRegisterClass(MVT::i32, &SH::GPRRegClass);
  computeRegisterProperties(Subtarget->getRegisterInfo());


  setBooleanContents(ZeroOrOneBooleanContent);
  setBooleanVectorContents(ZeroOrOneBooleanContent);
  setStackPointerRegisterToSaveRestore(SH::GBR);
  setJumpIsExpensive(true);
  setMinFunctionAlignment(Align(4));
}

SDValue SuperHTargetLowering::LowerFormalArguments(SDValue Chain,
                       CallingConv::ID CallConv, bool IsVarArg,
                       const SmallVectorImpl<ISD::InputArg> &Ins,
                       const SDLoc &dl, SelectionDAG &DAG,
                       SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  DataLayout DL = DAG.getDataLayout();

  EVT PtrVT = getPointerTy(DAG.getDataLayout());

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_SH);

  unsigned InIdx = 0;
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i, ++InIdx) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg;
    
    if (VA.isRegLoc()) {
      Register VReg = RegInfo.createVirtualRegister(&SH::GPRRegClass);
      MF.getRegInfo().addLiveIn(VA.getLocReg(), VReg);
      Arg = DAG.getCopyFromReg(Chain, dl, VReg, MVT::i32);
      if (VA.getLocInfo() != CCValAssign::Indirect) {
        if (VA.getLocVT() == MVT::f32)
          Arg = DAG.getNode(ISD::BITCAST, dl, MVT::f32, Arg);
        else if (VA.getLocVT() != MVT::i32) {
          Arg = DAG.getNode(ISD::AssertSext, dl, MVT::i32, Arg,
                            DAG.getValueType(VA.getLocVT()));
          Arg = DAG.getNode(ISD::TRUNCATE, dl, VA.getLocVT(), Arg);
        }
        InVals.push_back(Arg);
        continue;
      }
    } else {
      // Try matching frame index.
      assert(VA.isMemLoc());

      EVT LocVT = VA.getLocVT();

      // Create the frame index object for this incoming parameter.
      int FI = MFI.CreateFixedObject(LocVT.getSizeInBits() / 8,
                                     VA.getLocMemOffset(), true);

      // Create the SelectionDAG nodes corresponding to a load
      // from this parameter.
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy(DL));
      InVals.push_back(DAG.getLoad(LocVT, dl, Chain, FIN,
                                   MachinePointerInfo::getFixedStack(MF, FI)));

    }

    SDValue ArgValue =
        DAG.getLoad(VA.getValVT(), dl, Chain, Arg, MachinePointerInfo());
    InVals.push_back(ArgValue);

    unsigned ArgIndex = Ins[InIdx].OrigArgIndex;
    assert(Ins[InIdx].PartOffset == 0);
    while (i + 1 != e && Ins[InIdx + 1].OrigArgIndex == ArgIndex) {
      CCValAssign &PartVA = ArgLocs[i + 1];
      unsigned PartOffset = Ins[InIdx + 1].PartOffset;
      SDValue Address = DAG.getMemBasePlusOffset(
          ArgValue, TypeSize::getFixed(PartOffset), dl);
      InVals.push_back(DAG.getLoad(PartVA.getValVT(), dl, Chain, Address,
                                   MachinePointerInfo()));
      ++i;
      ++InIdx;
    }
  }

  return Chain;
}

//===----------------------------------------------------------------------===//
//                              RETURN LOWERING
//===----------------------------------------------------------------------===//

bool SuperHTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context,
    const Type *RetTy) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);
  return CCInfo.CheckReturn(Outs, RetCC_SH);
}

SDValue SuperHTargetLowering::LowerReturn(SDValue Chain,
                    CallingConv::ID CallConv, bool IsVarArg,
                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                    const SmallVectorImpl<SDValue> &OutVals,
                    const SDLoc &dl, SelectionDAG &DAG) const {
  
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                *DAG.getContext());

  // Analyze return values.
  MachineFunction &MF = DAG.getMachineFunction();
  CCInfo.AnalyzeReturn(Outs, RetCC_SH);

  // Fill out values into registers.
  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);
  for (unsigned i = 0, e = RVLocs.size(); i != e; ++i) {
    CCValAssign &VA = RVLocs[i];

    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), OutVals[i], Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  // If function is naked, don't emit rts.
  if (MF.getFunction().getAttributes().hasFnAttr(Attribute::Naked)) {
    return Chain;
  }

  // Update chain.
  RetOps[0] = Chain; 
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(SHISD::RET, dl, MVT::Other, RetOps);
}

SDValue SuperHTargetLowering::LowerCall(CallLoweringInfo &CLI, SmallVectorImpl<SDValue> &InVals) const {
  for(unsigned i = 0; i < CLI.Ins.size(); i++) {
    auto VArg = CLI.getArgs()[i];
    InVals.push_back(VArg.Node);
  }

  return CLI.Chain;
}