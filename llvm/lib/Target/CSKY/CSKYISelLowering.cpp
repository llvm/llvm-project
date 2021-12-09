//===-- CSKYISelLowering.cpp - CSKY DAG Lowering Implementation  ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that CSKY uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "CSKYISelLowering.h"
#include "CSKYCallingConv.h"
#include "CSKYMachineFunctionInfo.h"
#include "CSKYRegisterInfo.h"
#include "CSKYSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "csky-isel-lowering"

STATISTIC(NumTailCalls, "Number of tail calls");

#include "CSKYGenCallingConv.inc"

static const MCPhysReg GPRArgRegs[] = {CSKY::R0, CSKY::R1, CSKY::R2, CSKY::R3};

CSKYTargetLowering::CSKYTargetLowering(const TargetMachine &TM,
                                       const CSKYSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {
  // Register Class
  addRegisterClass(MVT::i32, &CSKY::GPRRegClass);

  setOperationAction(ISD::ADDCARRY, MVT::i32, Legal);
  setOperationAction(ISD::SUBCARRY, MVT::i32, Legal);
  setOperationAction(ISD::BITREVERSE, MVT::i32, Legal);

  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  setOperationAction(ISD::ROTR, MVT::i32, Expand);
  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Expand);
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::MULHS, MVT::i32, Expand);
  setOperationAction(ISD::MULHU, MVT::i32, Expand);

  setLoadExtAction(ISD::EXTLOAD, MVT::i32, MVT::i1, Promote);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i32, MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i32, MVT::i1, Promote);

  if (!Subtarget.hasE2()) {
    setLoadExtAction(ISD::SEXTLOAD, MVT::i32, MVT::i8, Expand);
    setLoadExtAction(ISD::SEXTLOAD, MVT::i32, MVT::i16, Expand);
    setOperationAction(ISD::CTLZ, MVT::i32, Expand);
    setOperationAction(ISD::BSWAP, MVT::i32, Expand);
  }

  if (!Subtarget.has2E3()) {
    setOperationAction(ISD::ABS, MVT::i32, Expand);
    setOperationAction(ISD::BITREVERSE, MVT::i32, Expand);
    setOperationAction(ISD::SDIV, MVT::i32, Expand);
    setOperationAction(ISD::UDIV, MVT::i32, Expand);
  }

  // Compute derived properties from the register classes.
  computeRegisterProperties(STI.getRegisterInfo());

  setBooleanContents(UndefinedBooleanContent);
  setBooleanVectorContents(ZeroOrNegativeOneBooleanContent);

  // TODO: Add atomic support fully.
  setMaxAtomicSizeInBitsSupported(0);

  setStackPointerRegisterToSaveRestore(CSKY::R14);
  const Align FunctionAlignment(2);
  setMinFunctionAlignment(FunctionAlignment);
  setSchedulingPreference(Sched::Source);
}

EVT CSKYTargetLowering::getSetCCResultType(const DataLayout &DL,
                                           LLVMContext &Context, EVT VT) const {
  if (!VT.isVector())
    return MVT::i32;

  return VT.changeVectorElementTypeToInteger();
}

static SDValue convertValVTToLocVT(SelectionDAG &DAG, SDValue Val,
                                   const CCValAssign &VA, const SDLoc &DL) {
  EVT LocVT = VA.getLocVT();

  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
    break;
  case CCValAssign::BCvt:
    Val = DAG.getNode(ISD::BITCAST, DL, LocVT, Val);
    break;
  }
  return Val;
}

static SDValue convertLocVTToValVT(SelectionDAG &DAG, SDValue Val,
                                   const CCValAssign &VA, const SDLoc &DL) {
  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
    break;
  case CCValAssign::BCvt:
    Val = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Val);
    break;
  }
  return Val;
}

static SDValue unpackFromRegLoc(const CSKYSubtarget &Subtarget,
                                SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  EVT LocVT = VA.getLocVT();
  SDValue Val;
  const TargetRegisterClass *RC;

  switch (LocVT.getSimpleVT().SimpleTy) {
  default:
    llvm_unreachable("Unexpected register type");
  case MVT::i32:
    RC = &CSKY::GPRRegClass;
    break;
  }

  Register VReg = RegInfo.createVirtualRegister(RC);
  RegInfo.addLiveIn(VA.getLocReg(), VReg);
  Val = DAG.getCopyFromReg(Chain, DL, VReg, LocVT);

  return convertLocVTToValVT(DAG, Val, VA, DL);
}

static SDValue unpackFromMemLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  EVT LocVT = VA.getLocVT();
  EVT ValVT = VA.getValVT();
  EVT PtrVT = MVT::getIntegerVT(DAG.getDataLayout().getPointerSizeInBits(0));
  int FI = MFI.CreateFixedObject(ValVT.getSizeInBits() / 8,
                                 VA.getLocMemOffset(), /*Immutable=*/true);
  SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
  SDValue Val;

  ISD::LoadExtType ExtType;
  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
  case CCValAssign::BCvt:
    ExtType = ISD::NON_EXTLOAD;
    break;
  }
  Val = DAG.getExtLoad(
      ExtType, DL, LocVT, Chain, FIN,
      MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI), ValVT);
  return Val;
}

// Transform physical registers into virtual registers.
SDValue CSKYTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {

  switch (CallConv) {
  default:
    report_fatal_error("Unsupported calling convention");
  case CallingConv::C:
  case CallingConv::Fast:
    break;
  }

  MachineFunction &MF = DAG.getMachineFunction();

  // Used with vargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  CCInfo.AnalyzeFormalArguments(Ins, CCAssignFnForCall(CallConv, IsVarArg));

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue;

    if (VA.isRegLoc())
      ArgValue = unpackFromRegLoc(Subtarget, DAG, Chain, VA, DL);
    else
      ArgValue = unpackFromMemLoc(DAG, Chain, VA, DL);

    InVals.push_back(ArgValue);
  }

  if (IsVarArg) {
    const unsigned XLenInBytes = 4;
    const MVT XLenVT = MVT::i32;

    ArrayRef<MCPhysReg> ArgRegs = makeArrayRef(GPRArgRegs);
    unsigned Idx = CCInfo.getFirstUnallocated(ArgRegs);
    const TargetRegisterClass *RC = &CSKY::GPRRegClass;
    MachineFrameInfo &MFI = MF.getFrameInfo();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    CSKYMachineFunctionInfo *CSKYFI = MF.getInfo<CSKYMachineFunctionInfo>();

    // Offset of the first variable argument from stack pointer, and size of
    // the vararg save area. For now, the varargs save area is either zero or
    // large enough to hold a0-a4.
    int VaArgOffset, VarArgsSaveSize;

    // If all registers are allocated, then all varargs must be passed on the
    // stack and we don't need to save any argregs.
    if (ArgRegs.size() == Idx) {
      VaArgOffset = CCInfo.getNextStackOffset();
      VarArgsSaveSize = 0;
    } else {
      VarArgsSaveSize = XLenInBytes * (ArgRegs.size() - Idx);
      VaArgOffset = -VarArgsSaveSize;
    }

    // Record the frame index of the first variable argument
    // which is a value necessary to VASTART.
    int FI = MFI.CreateFixedObject(XLenInBytes, VaArgOffset, true);
    CSKYFI->setVarArgsFrameIndex(FI);

    // Copy the integer registers that may have been used for passing varargs
    // to the vararg save area.
    for (unsigned I = Idx; I < ArgRegs.size();
         ++I, VaArgOffset += XLenInBytes) {
      const Register Reg = RegInfo.createVirtualRegister(RC);
      RegInfo.addLiveIn(ArgRegs[I], Reg);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, XLenVT);
      FI = MFI.CreateFixedObject(XLenInBytes, VaArgOffset, true);
      SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue Store = DAG.getStore(Chain, DL, ArgValue, PtrOff,
                                   MachinePointerInfo::getFixedStack(MF, FI));
      cast<StoreSDNode>(Store.getNode())
          ->getMemOperand()
          ->setValue((Value *)nullptr);
      OutChains.push_back(Store);
    }
    CSKYFI->setVarArgsSaveSize(VarArgsSaveSize);
  }

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens for vararg functions.
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  }

  return Chain;
}

bool CSKYTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> CSKYLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, CSKYLocs, Context);
  return CCInfo.CheckReturn(Outs, CCAssignFnForReturn(CallConv, IsVarArg));
}

SDValue
CSKYTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                bool IsVarArg,
                                const SmallVectorImpl<ISD::OutputArg> &Outs,
                                const SmallVectorImpl<SDValue> &OutVals,
                                const SDLoc &DL, SelectionDAG &DAG) const {
  // Stores the assignment of the return value to a location.
  SmallVector<CCValAssign, 16> CSKYLocs;

  // Info about the registers and stack slot.
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), CSKYLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, CCAssignFnForReturn(CallConv, IsVarArg));

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0, e = CSKYLocs.size(); i < e; ++i) {
    SDValue Val = OutVals[i];
    CCValAssign &VA = CSKYLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    bool IsF64OnCSKY = VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64;

    if (IsF64OnCSKY) {

      assert(VA.isRegLoc() && "Expected return via registers");
      SDValue Split64 = DAG.getNode(CSKYISD::BITCAST_TO_LOHI, DL,
                                    DAG.getVTList(MVT::i32, MVT::i32), Val);
      SDValue Lo = Split64.getValue(0);
      SDValue Hi = Split64.getValue(1);

      Register RegLo = VA.getLocReg();
      assert(RegLo < CSKY::R31 && "Invalid register pair");
      Register RegHi = RegLo + 1;

      Chain = DAG.getCopyToReg(Chain, DL, RegLo, Lo, Glue);
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(RegLo, MVT::i32));
      Chain = DAG.getCopyToReg(Chain, DL, RegHi, Hi, Glue);
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(RegHi, MVT::i32));
    } else {
      // Handle a 'normal' return.
      Val = convertValVTToLocVT(DAG, Val, VA, DL);
      Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Val, Glue);

      // Guarantee that all emitted copies are stuck together.
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
    }
  }

  RetOps[0] = Chain; // Update chain.

  // Add the glue node if we have it.
  if (Glue.getNode()) {
    RetOps.push_back(Glue);
  }

  // Interrupt service routines use different return instructions.
  if (DAG.getMachineFunction().getFunction().hasFnAttribute("interrupt"))
    return DAG.getNode(CSKYISD::NIR, DL, MVT::Other, RetOps);

  return DAG.getNode(CSKYISD::RET, DL, MVT::Other, RetOps);
}

CCAssignFn *CSKYTargetLowering::CCAssignFnForReturn(CallingConv::ID CC,
                                                    bool IsVarArg) const {
  if (IsVarArg || !Subtarget.useHardFloatABI())
    return RetCC_CSKY_ABIV2_SOFT;
  else
    return RetCC_CSKY_ABIV2_FP;
}

CCAssignFn *CSKYTargetLowering::CCAssignFnForCall(CallingConv::ID CC,
                                                  bool IsVarArg) const {
  if (IsVarArg || !Subtarget.useHardFloatABI())
    return CC_CSKY_ABIV2_SOFT;
  else
    return CC_CSKY_ABIV2_FP;
}

const char *CSKYTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default:
    llvm_unreachable("unknown CSKYISD node");
  case CSKYISD::NIE:
    return "CSKYISD::NIE";
  case CSKYISD::NIR:
    return "CSKYISD::NIR";
  case CSKYISD::RET:
    return "CSKYISD::RET";
  case CSKYISD::BITCAST_TO_LOHI:
    return "CSKYISD::BITCAST_TO_LOHI";
  }
}
