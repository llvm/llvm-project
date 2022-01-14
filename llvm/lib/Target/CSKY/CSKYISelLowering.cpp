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
#include "CSKYConstantPoolValue.h"
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
  setOperationAction(ISD::SELECT_CC, MVT::i32, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Expand);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Expand);
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::MULHS, MVT::i32, Expand);
  setOperationAction(ISD::MULHU, MVT::i32, Expand);
  setOperationAction(ISD::VAARG, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  setLoadExtAction(ISD::EXTLOAD, MVT::i32, MVT::i1, Promote);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i32, MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i32, MVT::i1, Promote);

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::ExternalSymbol, MVT::i32, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32, Custom);
  setOperationAction(ISD::BlockAddress, MVT::i32, Custom);
  setOperationAction(ISD::JumpTable, MVT::i32, Custom);
  setOperationAction(ISD::VASTART, MVT::Other, Custom);

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

  if (!Subtarget.has3r2E3r3()) {
    setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Expand);
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

SDValue CSKYTargetLowering::LowerOperation(SDValue Op,
                                           SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("unimplemented op");
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::ExternalSymbol:
    return LowerExternalSymbol(Op, DAG);
  case ISD::JumpTable:
    return LowerJumpTable(Op, DAG);
  case ISD::BlockAddress:
    return LowerBlockAddress(Op, DAG);
  case ISD::VASTART:
    return LowerVASTART(Op, DAG);
  case ISD::FRAMEADDR:
    return LowerFRAMEADDR(Op, DAG);
  case ISD::RETURNADDR:
    return LowerRETURNADDR(Op, DAG);
  }
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
  case MVT::f32:
    RC = Subtarget.hasFPUv2SingleFloat() ? &CSKY::sFPR32RegClass
                                         : &CSKY::FPR32RegClass;
    break;
  case MVT::f64:
    RC = Subtarget.hasFPUv2DoubleFloat() ? &CSKY::sFPR64RegClass
                                         : &CSKY::FPR64RegClass;
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

static SDValue unpack64(SelectionDAG &DAG, SDValue Chain, const CCValAssign &VA,
                        const SDLoc &DL) {
  assert(VA.getLocVT() == MVT::i32 &&
         (VA.getValVT() == MVT::f64 || VA.getValVT() == MVT::i64) &&
         "Unexpected VA");
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  if (VA.isMemLoc()) {
    // f64/i64 is passed on the stack.
    int FI = MFI.CreateFixedObject(8, VA.getLocMemOffset(), /*Immutable=*/true);
    SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
    return DAG.getLoad(VA.getValVT(), DL, Chain, FIN,
                       MachinePointerInfo::getFixedStack(MF, FI));
  }

  assert(VA.isRegLoc() && "Expected register VA assignment");

  Register LoVReg = RegInfo.createVirtualRegister(&CSKY::GPRRegClass);
  RegInfo.addLiveIn(VA.getLocReg(), LoVReg);
  SDValue Lo = DAG.getCopyFromReg(Chain, DL, LoVReg, MVT::i32);
  SDValue Hi;
  if (VA.getLocReg() == CSKY::R3) {
    // Second half of f64/i64 is passed on the stack.
    int FI = MFI.CreateFixedObject(4, 0, /*Immutable=*/true);
    SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
    Hi = DAG.getLoad(MVT::i32, DL, Chain, FIN,
                     MachinePointerInfo::getFixedStack(MF, FI));
  } else {
    // Second half of f64/i64 is passed in another GPR.
    Register HiVReg = RegInfo.createVirtualRegister(&CSKY::GPRRegClass);
    RegInfo.addLiveIn(VA.getLocReg() + 1, HiVReg);
    Hi = DAG.getCopyFromReg(Chain, DL, HiVReg, MVT::i32);
  }
  return DAG.getNode(CSKYISD::BITCAST_FROM_LOHI, DL, VA.getValVT(), Lo, Hi);
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

    bool IsF64OnCSKY = VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64;

    if (IsF64OnCSKY)
      ArgValue = unpack64(DAG, Chain, VA, DL);
    else if (VA.isRegLoc())
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

// Lower a call to a callseq_start + CALL + callseq_end chain, and add input
// and output parameter nodes.
SDValue CSKYTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                      SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  MVT XLenVT = MVT::i32;

  MachineFunction &MF = DAG.getMachineFunction();

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState ArgCCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  ArgCCInfo.AnalyzeCallOperands(Outs, CCAssignFnForCall(CallConv, IsVarArg));

  // Check if it's really possible to do a tail call.
  if (IsTailCall)
    IsTailCall = false; // TODO: TailCallOptimization;

  if (IsTailCall)
    ++NumTailCalls;
  else if (CLI.CB && CLI.CB->isMustTailCall())
    report_fatal_error("failed to perform tail call elimination on a call "
                       "site marked musttail");

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = ArgCCInfo.getNextStackOffset();

  // Create local copies for byval args
  SmallVector<SDValue, 8> ByValArgs;
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    ISD::ArgFlagsTy Flags = Outs[i].Flags;
    if (!Flags.isByVal())
      continue;

    SDValue Arg = OutVals[i];
    unsigned Size = Flags.getByValSize();
    Align Alignment = Flags.getNonZeroByValAlign();

    int FI =
        MF.getFrameInfo().CreateStackObject(Size, Alignment, /*isSS=*/false);
    SDValue FIPtr = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
    SDValue SizeNode = DAG.getConstant(Size, DL, XLenVT);

    Chain = DAG.getMemcpy(Chain, DL, FIPtr, Arg, SizeNode, Alignment,
                          /*IsVolatile=*/false,
                          /*AlwaysInline=*/false, IsTailCall,
                          MachinePointerInfo(), MachinePointerInfo());
    ByValArgs.push_back(FIPtr);
  }

  if (!IsTailCall)
    Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, CLI.DL);

  // Copy argument values to their designated locations.
  SmallVector<std::pair<Register, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;
  for (unsigned i = 0, j = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue = OutVals[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;

    bool IsF64OnCSKY = VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64;

    if (IsF64OnCSKY && VA.isRegLoc()) {
      SDValue Split64 =
          DAG.getNode(CSKYISD::BITCAST_TO_LOHI, DL,
                      DAG.getVTList(MVT::i32, MVT::i32), ArgValue);
      SDValue Lo = Split64.getValue(0);
      SDValue Hi = Split64.getValue(1);

      Register RegLo = VA.getLocReg();
      RegsToPass.push_back(std::make_pair(RegLo, Lo));

      if (RegLo == CSKY::R3) {
        // Second half of f64/i64 is passed on the stack.
        // Work out the address of the stack slot.
        if (!StackPtr.getNode())
          StackPtr = DAG.getCopyFromReg(Chain, DL, CSKY::R14, PtrVT);
        // Emit the store.
        MemOpChains.push_back(
            DAG.getStore(Chain, DL, Hi, StackPtr, MachinePointerInfo()));
      } else {
        // Second half of f64/i64 is passed in another GPR.
        assert(RegLo < CSKY::R31 && "Invalid register pair");
        Register RegHigh = RegLo + 1;
        RegsToPass.push_back(std::make_pair(RegHigh, Hi));
      }
      continue;
    }

    ArgValue = convertValVTToLocVT(DAG, ArgValue, VA, DL);

    // Use local copy if it is a byval arg.
    if (Flags.isByVal())
      ArgValue = ByValArgs[j++];

    if (VA.isRegLoc()) {
      // Queue up the argument copies and emit them at the end.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ArgValue));
    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");
      assert(!IsTailCall && "Tail call not allowed if stack is used "
                            "for passing parameters");

      // Work out the address of the stack slot.
      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, CSKY::R14, PtrVT);
      SDValue Address =
          DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr,
                      DAG.getIntPtrConstant(VA.getLocMemOffset(), DL));

      // Emit the store.
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, Address, MachinePointerInfo()));
    }
  }

  // Join the stores, which are independent of one another.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  SDValue Glue;

  // Build a sequence of copy-to-reg nodes, chained and glued together.
  for (auto &Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, Reg.first, Reg.second, Glue);
    Glue = Chain.getValue(1);
  }

  SmallVector<SDValue, 8> Ops;
  EVT Ty = getPointerTy(DAG.getDataLayout());
  bool IsRegCall = false;

  Ops.push_back(Chain);

  if (GlobalAddressSDNode *S = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = S->getGlobal();
    bool IsLocal =
        getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV);

    if (isPositionIndependent() || !Subtarget.has2E3()) {
      IsRegCall = true;
      Ops.push_back(getAddr<GlobalAddressSDNode, true>(S, DAG, IsLocal));
    } else {
      Ops.push_back(getTargetNode(cast<GlobalAddressSDNode>(Callee), DL, Ty,
                                  DAG, CSKYII::MO_None));
      Ops.push_back(getTargetConstantPoolValue(
          cast<GlobalAddressSDNode>(Callee), Ty, DAG, CSKYII::MO_None));
    }
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    bool IsLocal = getTargetMachine().shouldAssumeDSOLocal(
        *MF.getFunction().getParent(), nullptr);

    if (isPositionIndependent() || !Subtarget.has2E3()) {
      IsRegCall = true;
      Ops.push_back(getAddr<ExternalSymbolSDNode, true>(S, DAG, IsLocal));
    } else {
      Ops.push_back(getTargetNode(cast<ExternalSymbolSDNode>(Callee), DL, Ty,
                                  DAG, CSKYII::MO_None));
      Ops.push_back(getTargetConstantPoolValue(
          cast<ExternalSymbolSDNode>(Callee), Ty, DAG, CSKYII::MO_None));
    }
  } else {
    IsRegCall = true;
    Ops.push_back(Callee);
  }

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (auto &Reg : RegsToPass)
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));

  if (!IsTailCall) {
    // Add a register mask operand representing the call-preserved registers.
    const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
    const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
    assert(Mask && "Missing call preserved mask for calling convention");
    Ops.push_back(DAG.getRegisterMask(Mask));
  }

  // Glue the call to the argument copies, if any.
  if (Glue.getNode())
    Ops.push_back(Glue);

  // Emit the call.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  if (IsTailCall) {
    MF.getFrameInfo().setHasTailCall();
    return DAG.getNode(IsRegCall ? CSKYISD::TAILReg : CSKYISD::TAIL, DL,
                       NodeTys, Ops);
  }

  Chain = DAG.getNode(IsRegCall ? CSKYISD::CALLReg : CSKYISD::CALL, DL, NodeTys,
                      Ops);
  DAG.addNoMergeSiteInfo(Chain.getNode(), CLI.NoMerge);
  Glue = Chain.getValue(1);

  // Mark the end of the call, which is glued to the call itself.
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getConstant(NumBytes, DL, PtrVT, true),
                             DAG.getConstant(0, DL, PtrVT, true), Glue, DL);
  Glue = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> CSKYLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, CSKYLocs, *DAG.getContext());
  RetCCInfo.AnalyzeCallResult(Ins, CCAssignFnForReturn(CallConv, IsVarArg));

  // Copy all of the result registers out of their specified physreg.
  for (auto &VA : CSKYLocs) {
    // Copy the value out
    SDValue RetValue =
        DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT(), Glue);
    // Glue the RetValue to the end of the call sequence
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    bool IsF64OnCSKY = VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64;

    if (IsF64OnCSKY) {
      assert(VA.getLocReg() == GPRArgRegs[0] && "Unexpected reg assignment");
      SDValue RetValue2 =
          DAG.getCopyFromReg(Chain, DL, GPRArgRegs[1], MVT::i32, Glue);
      Chain = RetValue2.getValue(1);
      Glue = RetValue2.getValue(2);
      RetValue = DAG.getNode(CSKYISD::BITCAST_FROM_LOHI, DL, VA.getValVT(),
                             RetValue, RetValue2);
    }

    RetValue = convertLocVTToValVT(DAG, RetValue, VA, DL);

    InVals.push_back(RetValue);
  }

  return Chain;
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

static CSKYCP::CSKYCPModifier getModifier(unsigned Flags) {

  if (Flags == CSKYII::MO_ADDR32)
    return CSKYCP::ADDR;
  else if (Flags == CSKYII::MO_GOT32)
    return CSKYCP::GOT;
  else if (Flags == CSKYII::MO_GOTOFF)
    return CSKYCP::GOTOFF;
  else if (Flags == CSKYII::MO_PLT32)
    return CSKYCP::PLT;
  else if (Flags == CSKYII::MO_None)
    return CSKYCP::NO_MOD;
  else
    assert(0 && "unknown CSKYII Modifier");
  return CSKYCP::NO_MOD;
}

SDValue CSKYTargetLowering::getTargetConstantPoolValue(GlobalAddressSDNode *N,
                                                       EVT Ty,
                                                       SelectionDAG &DAG,
                                                       unsigned Flags) const {
  CSKYConstantPoolValue *CPV = CSKYConstantPoolConstant::Create(
      N->getGlobal(), CSKYCP::CPValue, 0, getModifier(Flags), false);

  return DAG.getTargetConstantPool(CPV, Ty);
}

static MachineBasicBlock *
emitSelectPseudo(MachineInstr &MI, MachineBasicBlock *BB, unsigned Opcode) {

  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  // To "insert" a SELECT instruction, we actually have to insert the
  // diamond control-flow pattern.  The incoming instruction knows the
  // destination vreg to set, the condition code register to branch on, the
  // true/false values to select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  //  thisMBB:
  //  ...
  //   TrueVal = ...
  //   bt32 c, sinkMBB
  //   fallthrough --> copyMBB
  MachineBasicBlock *thisMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *copyMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(It, copyMBB);
  F->insert(It, sinkMBB);

  // Transfer the remainder of BB and its successor edges to sinkMBB.
  sinkMBB->splice(sinkMBB->begin(), BB,
                  std::next(MachineBasicBlock::iterator(MI)), BB->end());
  sinkMBB->transferSuccessorsAndUpdatePHIs(BB);

  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(copyMBB);
  BB->addSuccessor(sinkMBB);

  // bt32 condition, sinkMBB
  BuildMI(BB, DL, TII.get(Opcode))
      .addReg(MI.getOperand(1).getReg())
      .addMBB(sinkMBB);

  //  copyMBB:
  //   %FalseValue = ...
  //   # fallthrough to sinkMBB
  BB = copyMBB;

  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  //  sinkMBB:
  //   %Result = phi [ %TrueValue, thisMBB ], [ %FalseValue, copyMBB ]
  //  ...
  BB = sinkMBB;

  BuildMI(*BB, BB->begin(), DL, TII.get(CSKY::PHI), MI.getOperand(0).getReg())
      .addReg(MI.getOperand(2).getReg())
      .addMBB(thisMBB)
      .addReg(MI.getOperand(3).getReg())
      .addMBB(copyMBB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.

  return BB;
}

MachineBasicBlock *
CSKYTargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                                MachineBasicBlock *BB) const {
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instr type to insert");
  case CSKY::ISEL32:
    return emitSelectPseudo(MI, BB, CSKY::BT32);
  case CSKY::ISEL16:
    return emitSelectPseudo(MI, BB, CSKY::BT16);
  }
}

SDValue CSKYTargetLowering::getTargetConstantPoolValue(ExternalSymbolSDNode *N,
                                                       EVT Ty,
                                                       SelectionDAG &DAG,
                                                       unsigned Flags) const {
  CSKYConstantPoolValue *CPV =
      CSKYConstantPoolSymbol::Create(Type::getInt32Ty(*DAG.getContext()),
                                     N->getSymbol(), 0, getModifier(Flags));

  return DAG.getTargetConstantPool(CPV, Ty);
}

SDValue CSKYTargetLowering::getTargetConstantPoolValue(JumpTableSDNode *N,
                                                       EVT Ty,
                                                       SelectionDAG &DAG,
                                                       unsigned Flags) const {
  CSKYConstantPoolValue *CPV =
      CSKYConstantPoolJT::Create(Type::getInt32Ty(*DAG.getContext()),
                                 N->getIndex(), 0, getModifier(Flags));
  return DAG.getTargetConstantPool(CPV, Ty);
}

SDValue CSKYTargetLowering::getTargetConstantPoolValue(BlockAddressSDNode *N,
                                                       EVT Ty,
                                                       SelectionDAG &DAG,
                                                       unsigned Flags) const {
  CSKYConstantPoolValue *CPV = CSKYConstantPoolConstant::Create(
      N->getBlockAddress(), CSKYCP::CPBlockAddress, 0, getModifier(Flags),
      false);
  return DAG.getTargetConstantPool(CPV, Ty);
}

SDValue CSKYTargetLowering::getTargetNode(GlobalAddressSDNode *N, SDLoc DL,
                                          EVT Ty, SelectionDAG &DAG,
                                          unsigned Flags) const {
  return DAG.getTargetGlobalAddress(N->getGlobal(), DL, Ty, 0, Flags);
}

SDValue CSKYTargetLowering::getTargetNode(ExternalSymbolSDNode *N, SDLoc DL,
                                          EVT Ty, SelectionDAG &DAG,
                                          unsigned Flags) const {
  return DAG.getTargetExternalSymbol(N->getSymbol(), Ty, Flags);
}

SDValue CSKYTargetLowering::getTargetNode(JumpTableSDNode *N, SDLoc DL, EVT Ty,
                                          SelectionDAG &DAG,
                                          unsigned Flags) const {
  return DAG.getTargetJumpTable(N->getIndex(), Ty, Flags);
}

SDValue CSKYTargetLowering::getTargetNode(BlockAddressSDNode *N, SDLoc DL,
                                          EVT Ty, SelectionDAG &DAG,
                                          unsigned Flags) const {
  return DAG.getTargetBlockAddress(N->getBlockAddress(), Ty, N->getOffset(),
                                   Flags);
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
  case CSKYISD::CALL:
    return "CSKYISD::CALL";
  case CSKYISD::CALLReg:
    return "CSKYISD::CALLReg";
  case CSKYISD::TAIL:
    return "CSKYISD::TAIL";
  case CSKYISD::TAILReg:
    return "CSKYISD::TAILReg";
  case CSKYISD::LOAD_ADDR:
    return "CSKYISD::LOAD_ADDR";
  case CSKYISD::BITCAST_TO_LOHI:
    return "CSKYISD::BITCAST_TO_LOHI";
  case CSKYISD::BITCAST_FROM_LOHI:
    return "CSKYISD::BITCAST_FROM_LOHI";
  }
}

SDValue CSKYTargetLowering::LowerGlobalAddress(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  int64_t Offset = N->getOffset();

  const GlobalValue *GV = N->getGlobal();
  bool IsLocal = getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV);
  SDValue Addr = getAddr<GlobalAddressSDNode, false>(N, DAG, IsLocal);

  // In order to maximise the opportunity for common subexpression elimination,
  // emit a separate ADD node for the global address offset instead of folding
  // it in the global address node. Later peephole optimisations may choose to
  // fold it back in when profitable.
  if (Offset != 0)
    return DAG.getNode(ISD::ADD, DL, Ty, Addr,
                       DAG.getConstant(Offset, DL, MVT::i32));
  return Addr;
}

SDValue CSKYTargetLowering::LowerExternalSymbol(SDValue Op,
                                                SelectionDAG &DAG) const {
  ExternalSymbolSDNode *N = cast<ExternalSymbolSDNode>(Op);

  return getAddr(N, DAG, false);
}

SDValue CSKYTargetLowering::LowerJumpTable(SDValue Op,
                                           SelectionDAG &DAG) const {
  JumpTableSDNode *N = cast<JumpTableSDNode>(Op);

  return getAddr<JumpTableSDNode, false>(N, DAG);
}

SDValue CSKYTargetLowering::LowerBlockAddress(SDValue Op,
                                              SelectionDAG &DAG) const {
  BlockAddressSDNode *N = cast<BlockAddressSDNode>(Op);

  return getAddr(N, DAG);
}

SDValue CSKYTargetLowering::LowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  CSKYMachineFunctionInfo *FuncInfo = MF.getInfo<CSKYMachineFunctionInfo>();

  SDLoc DL(Op);
  SDValue FI = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                 getPointerTy(MF.getDataLayout()));

  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), DL, FI, Op.getOperand(1),
                      MachinePointerInfo(SV));
}

SDValue CSKYTargetLowering::LowerFRAMEADDR(SDValue Op,
                                           SelectionDAG &DAG) const {
  const CSKYRegisterInfo &RI = *Subtarget.getRegisterInfo();
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setFrameAddressIsTaken(true);

  EVT VT = Op.getValueType();
  SDLoc dl(Op);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  Register FrameReg = RI.getFrameRegister(MF);
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), dl, FrameReg, VT);
  while (Depth--)
    FrameAddr = DAG.getLoad(VT, dl, DAG.getEntryNode(), FrameAddr,
                            MachinePointerInfo());
  return FrameAddr;
}

SDValue CSKYTargetLowering::LowerRETURNADDR(SDValue Op,
                                            SelectionDAG &DAG) const {
  const CSKYRegisterInfo &RI = *Subtarget.getRegisterInfo();
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setReturnAddressIsTaken(true);

  if (verifyReturnAddressArgumentIsConstant(Op, DAG))
    return SDValue();

  EVT VT = Op.getValueType();
  SDLoc dl(Op);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  if (Depth) {
    SDValue FrameAddr = LowerFRAMEADDR(Op, DAG);
    SDValue Offset = DAG.getConstant(4, dl, MVT::i32);
    return DAG.getLoad(VT, dl, DAG.getEntryNode(),
                       DAG.getNode(ISD::ADD, dl, VT, FrameAddr, Offset),
                       MachinePointerInfo());
  }
  // Return the value of the return address register, marking it an implicit
  // live-in.
  unsigned Reg = MF.addLiveIn(RI.getRARegister(), getRegClassFor(MVT::i32));
  return DAG.getCopyFromReg(DAG.getEntryNode(), dl, Reg, VT);
}

Register CSKYTargetLowering::getExceptionPointerRegister(
    const Constant *PersonalityFn) const {
  return CSKY::R0;
}

Register CSKYTargetLowering::getExceptionSelectorRegister(
    const Constant *PersonalityFn) const {
  return CSKY::R1;
}
