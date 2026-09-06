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
#include "SuperHConstantPoolValue.h"
#include "SuperHInstrInfo.h"
#include "SuperHMachineFunctionInfo.h"
#include "SuperHSelectionDAGInfo.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "MCTargetDesc/SuperHBaseInfo.h"
#include "SuperHRegisterInfo.h"
#include "SuperHSubtarget.h"
#include "SuperHTargetMachine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "sh-lower"

#define DEBUG_FN_PRINT() LLVM_DEBUG(dbgs() << " - " << __PRETTY_FUNCTION__ << "\n");

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
  auto *RegInfo = Subtarget->getRegisterInfo();

  // GPR Registers are always 32 bit on SuperH.
  addRegisterClass(MVT::i32, &SH::GPRRegClass);
  computeRegisterProperties(Subtarget->getRegisterInfo());

  setSchedulingPreference(Sched::RegPressure);
  setSupportsUnalignedAtomics(false);
  // setStackPointerRegisterToSaveRestore(RegInfo->getStackRegister());

  // Loads and stores are legal
  for (MVT VT : MVT::integer_valuetypes()) {
    for (auto N : {ISD::EXTLOAD, ISD::SEXTLOAD, ISD::ZEXTLOAD}) {
      setLoadExtAction(N, VT, MVT::i1, Promote);
      setLoadExtAction(N, VT, MVT::i64, Expand);
    }

    setTruncStoreAction(VT, MVT::i1, Promote);
  }

  // Division and remainders are multi-instruction sequences
  // on SuperH. Use a custom pass to lower those.
  for (MVT VT : {MVT::i8, MVT::i16, MVT::i32}) {
    setOperationAction(ISD::UDIV, VT, Custom);
    setOperationAction(ISD::UREM, VT, Custom);
    setOperationAction(ISD::SDIV, VT, Custom);
    setOperationAction(ISD::SREM, VT, Custom);
  }

  // Multiplies have multiple lowerings depending on size.
  for (MVT VT : {MVT::i8, MVT::i16, MVT::i32, MVT::i64}) {
    setOperationAction(ISD::MUL, VT, Custom);
  }

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool, MVT::i32, Custom);
  setOperationAction(ISD::ExternalSymbol, MVT::i32, Custom);
  setOperationAction(ISD::BlockAddress, MVT::i32, Custom);
  setOperationAction(ISD::Constant, MVT::i32, Custom);

  setOperationAction(ISD::BR_CC, MVT::i8, Custom);
  setOperationAction(ISD::BR_CC, MVT::i16, Custom);
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::i64, Custom);
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);

  setOperationAction(ISD::SELECT_CC, MVT::i8, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i16, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Expand);
  setOperationAction(ISD::SETCC, MVT::i8, Custom);
  setOperationAction(ISD::SETCC, MVT::i16, Custom);
  setOperationAction(ISD::SETCC, MVT::i32, Custom);
  setOperationAction(ISD::SETCC, MVT::i64, Custom);

  // for (MVT VT : MVT::integer_valuetypes()) {
  //   setOperationAction(ISD::SIGN_EXTEND_INREG, VT, Expand);
  // }

  setBooleanContents(ZeroOrOneBooleanContent);
  setBooleanVectorContents(ZeroOrOneBooleanContent);
  setJumpIsExpensive(false);
  setMinFunctionAlignment(Align(4));
  setMinStackArgumentAlignment(Align(4));
}

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
//                                LOWERING
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//





//===----------------------------------------------------------------------===//
//                        CONDITIONAL BRANCH LOWERING
//===----------------------------------------------------------------------===//
SDValue SuperHTargetLowering::getSHCmp(SDValue LHS, SDValue RHS, ISD::CondCode CC,
                                       SelectionDAG &DAG, SDLoc DL) const {
  return DAG.getNode(SHISD::CMP, DL, MVT::Glue, LHS, RHS, DAG.getCondCode(CC));
}

SDValue SuperHTargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue TrueV = Op.getOperand(2);
  SDValue FalseV = Op.getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDLoc DL(Op);

  SDValue Cmp = getSHCmp(LHS, RHS, CC, DAG, DL);
  SDValue TargetCC = DAG.getConstant(CC, DL, MVT::i8);

  SDValue Ops[] = {TrueV, FalseV, TargetCC, Cmp};
  return DAG.getNode(SHISD::SELECT_CC, DL, Op.getValueType(), Ops);
}

SDValue SuperHTargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  SDLoc DL(Op);

  SDValue Cmp = getSHCmp(LHS, RHS, CC, DAG, DL);
  SDValue TargetCC = DAG.getConstant(CC, DL, MVT::i8);

  SDValue TrueV = DAG.getConstant(1, DL, Op.getValueType());
  SDValue FalseV = DAG.getConstant(0, DL, Op.getValueType());
  SDValue Ops[] = {TrueV, FalseV, TargetCC, Cmp};
  return DAG.getNode(SHISD::SELECT_CC, DL, Op.getValueType(), Ops);
}

SDValue SuperHTargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  DEBUG_FN_PRINT()

  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);
  SDLoc DL(Op);

  SDValue Cmp = getSHCmp(LHS, RHS, CC, DAG, DL);
  SDValue TargetCC = DAG.getCondCode(CC);

  return DAG.getNode(SHISD::BRCOND, DL, MVT::Other, Chain, Dest, TargetCC, Cmp);
}




//===----------------------------------------------------------------------===//
//                            CONSTANT LOWERING
//===----------------------------------------------------------------------===//

SDValue SuperHTargetLowering::LowerConstant(SDValue Op, SelectionDAG &DAG) const {
  DEBUG_FN_PRINT()

  // Get the address of the target into a register
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
    auto DL = SDLoc(C);

    // Constant would fit in immediate
    if (isInt<8>(C->getSExtValue())) {
      return Op;
    }

    // Constant would NOT fit in immediate, lower to constpool.
    SDValue Const = DAG.getSignedTargetConstant(C->getSExtValue(), DL, C->getValueType(0));
    return DAG.getNode(SHISD::WRAPPER, DL, C->getValueType(0), Const);
  }
  return SDValue();
}




//===----------------------------------------------------------------------===//
//                             ADDRESS LOWERING
//===----------------------------------------------------------------------===//

SDValue SuperHTargetLowering::LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const {
  DEBUG_FN_PRINT()

  // Get the address of the target into a register
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Op)) {
    auto PtrVT = getPointerTy(DAG.getDataLayout());
    auto DL = SDLoc(G);

    SHRefClass OpFlags = Subtarget->classifyGlobalReference(G->getGlobal());
    SDValue Addr = DAG.getTargetGlobalAddress(G->getGlobal(), DL, PtrVT, 0, OpFlags);
    return DAG.getNode(SHISD::WRAPPER, DL, MVT::i32, Addr);
  }
  return SDValue();
}

SDValue SuperHTargetLowering::LowerExternalSymbol(SDValue Op, SelectionDAG &DAG) const {
  DEBUG_FN_PRINT()

  // Get the address of the target into a register
  if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Op)) {
    auto PtrVT = getPointerTy(DAG.getDataLayout());
    auto DL = SDLoc(S);

    const Module *Mod = DAG.getMachineFunction().getFunction().getParent();
    SHRefClass OpFlags = Subtarget->classifyGlobalFunctionReference(nullptr, *Mod);

    SDValue Addr = DAG.getTargetExternalSymbol(S->getSymbol(), PtrVT, OpFlags);
    return DAG.getNode(SHISD::WRAPPER, DL, MVT::i32, Addr);
  }
  return SDValue();
}

SDValue SuperHTargetLowering::LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const {
  DEBUG_FN_PRINT()

  // Get the address of the target into a register
  if (BlockAddressSDNode *BA = dyn_cast<BlockAddressSDNode>(Op)) {
    auto PtrVT = getPointerTy(DAG.getDataLayout());
    auto DL = SDLoc(BA);

    const Module *Mod = DAG.getMachineFunction().getFunction().getParent();
    SHRefClass OpFlags = Subtarget->classifyGlobalFunctionReference(nullptr, *Mod);

    SDValue Addr = DAG.getTargetBlockAddress(BA->getBlockAddress(), PtrVT, 0, OpFlags);
    return DAG.getNode(SHISD::WRAPPER, DL, MVT::i32, Addr);
  }
  return SDValue();
}




//===----------------------------------------------------------------------===//
//                             ARGUMENT LOWERING
//===----------------------------------------------------------------------===//

SDValue SuperHTargetLowering::LowerFormalArguments(SDValue Chain,
                       CallingConv::ID CallConv, bool IsVarArg,
                       const SmallVectorImpl<ISD::InputArg> &Ins,
                       const SDLoc &dl, SelectionDAG &DAG,
                       SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  DataLayout DL = DAG.getDataLayout();

  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_SH);

  SDValue ArgValue;
  for (CCValAssign &VA : ArgLocs) {

    // Arguments stored on registers.
    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();
      const TargetRegisterClass *RC;
      if (RegVT == MVT::i8) {
        RC = &SH::GPRRegClass;
      } else if (RegVT == MVT::i16) {
        RC = &SH::GPRRegClass;
      } else if (RegVT == MVT::i32) {
        RC = &SH::GPRRegClass;
      } else {
        llvm_unreachable("Unknown argument type!");
      }

      Register Reg = MF.addLiveIn(VA.getLocReg(), RC);
      ArgValue = DAG.getCopyFromReg(Chain, dl, Reg, RegVT);

      // NOTE:  Clang should not promote any i8 or i16 to i32 but for safety the
      //        following code will handle zexts or sexts generated by other
      //        front ends. Otherwise:
      switch (VA.getLocInfo()) {
      default:
        llvm_unreachable("Unknown loc info!");
      case CCValAssign::Full:
        break;
      case CCValAssign::BCvt:
        ArgValue = DAG.getNode(ISD::BITCAST, dl, VA.getValVT(), ArgValue);
        break;
      case CCValAssign::SExt:
        ArgValue = DAG.getNode(ISD::AssertSext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
        ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);
        break;
      case CCValAssign::ZExt:
        ArgValue = DAG.getNode(ISD::AssertZext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
        ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);
        break;
      }

      InVals.push_back(ArgValue);
    } else {

      // Only arguments passed on the stack should make it here.
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
  }

  // TODO: Handle varargs.
  if (IsVarArg) {
    llvm_unreachable("varargs are not yet supported!");
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

  // If function is naked, don't emit return glue.
  if (MF.getFunction().getAttributes().hasFnAttr(Attribute::Naked)) {
    return Chain;
  }

  // Update chain.
  RetOps[0] = Chain; 
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(SHISD::RET_GLUE, dl, MVT::Other, RetOps);
}

SDValue SuperHTargetLowering::getPICJumpTableRelocBase(SDValue Table, SelectionDAG &DAG) const {
  return DAG.getRegister(Subtarget->getRegisterInfo()->getGOTRegister(),
                         getPointerTy(DAG.getDataLayout()));
}





//===----------------------------------------------------------------------===//
//                              CALL LOWERING
//===----------------------------------------------------------------------===//

SDValue SuperHTargetLowering::LowerCall(CallLoweringInfo &CLI, SmallVectorImpl<SDValue> &InVals) const {
  const SuperHRegisterInfo &RI = *Subtarget->getRegisterInfo();
  SelectionDAG &DAG = CLI.DAG;
  MachineFunction &MF = DAG.getMachineFunction();
  SDLoc &DL = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;

  // TODO: This was all yoinked from AVR, it likely needs to be modified to fit the calling
  // convention of SuperH.

  // Tail Call Optimisation not supported yet.
  IsTailCall = false;
  IsVarArg = false;

  if (IsVarArg) {
    return Chain;
  }

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  // Resolve the global value to jump to.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    Callee = LowerGlobalAddress(SDValue(G, 0), DAG);
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    Callee = LowerExternalSymbol(SDValue(S, 0), DAG);
  }
  CCInfo.AnalyzeCallOperands(Outs, CC_SH);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getStackSize();
  Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, DL);
  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;

  // First, walk the register assignments, inserting copies.
  unsigned AI, AE;
  bool HasStackArgs = false;
  for (AI = 0, AE = ArgLocs.size(); AI != AE; ++AI) {
    CCValAssign &VA = ArgLocs[AI];
    EVT RegVT = VA.getLocVT();
    SDValue Arg = OutVals[AI];

    // Promote the value if needed. With Clang this should not happen.
    switch (VA.getLocInfo()) {
    default:
      llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full:
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, DL, RegVT, Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, RegVT, Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, DL, RegVT, Arg);
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, DL, RegVT, Arg);
      break;
    }

    // Stop when we encounter a stack argument, we need to process them
    // in reverse order in the loop below.
    if (VA.isMemLoc()) {
      HasStackArgs = true;
      break;
    }

    // Arguments that can be passed on registers must be kept in the RegsToPass
    // vector.
    RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
  }

  // Second, stack arguments have to walked.
  // Previously this code created chained stores but those chained stores appear
  // to be unchained in the legalization phase. Therefore, do not attempt to
  // chain them here. In fact, chaining them here somehow causes the first and
  // second store to be reversed which is the exact opposite of the intended
  // effect.
  if (HasStackArgs) {
    SmallVector<SDValue, 8> MemOpChains;
    for (; AI != AE; AI++) {
      CCValAssign &VA = ArgLocs[AI];
      SDValue Arg = OutVals[AI];

      assert(VA.isMemLoc());

      // SP points to one stack slot further so add one to adjust it.
      SDValue PtrOff = DAG.getNode(
          ISD::ADD, DL, getPointerTy(DAG.getDataLayout()),
          DAG.getRegister(SH::R15, getPointerTy(DAG.getDataLayout())),
          DAG.getIntPtrConstant(VA.getLocMemOffset() + 1, DL));

      MemOpChains.push_back(
          DAG.getStore(Chain, DL, Arg, PtrOff,
                       MachinePointerInfo::getStack(MF, VA.getLocMemOffset())));
    }

    if (!MemOpChains.empty())
      Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);
  }

  // Build a sequence of copy-to-reg nodes chained together with token chain and
  // flag operands which copy the outgoing args into registers.  The InGlue in
  // necessary since all emited instructions must be stuck together.
  SDValue InGlue;
  for (auto Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, Reg.first, Reg.second, InGlue);
    InGlue = Chain.getValue(1);
  }

  // Returns a chain & a flag for retval copy to use.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (auto Reg : RegsToPass) {
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));
  }

  // Add a register mask operand representing the call-preserved registers.
  const TargetRegisterInfo *TRI = Subtarget->getRegisterInfo();
  const uint32_t *Mask =
      TRI->getCallPreservedMask(DAG.getMachineFunction(), CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  if (InGlue.getNode()) {
    Ops.push_back(InGlue);
  }

  Chain = DAG.getNode(SHISD::CALL, DL, {MVT::Other, MVT::Glue}, Ops);
  InGlue = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  Chain = DAG.getCALLSEQ_END(Chain, NumBytes, 0, InGlue, DL);
  InGlue = Chain.getValue(1);

  return LowerCallResult(Chain, InGlue, CallConv, IsVarArg, Ins, DL, DAG, InVals);
}

SDValue SuperHTargetLowering::LowerCallResult(
    SDValue Chain, SDValue InGlue, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  // Handle runtime calling convs.
  CCInfo.AnalyzeCallResult(Ins, RetCC_SH);

  // Copy all of the result registers out of their specified physreg.
  for (CCValAssign const &RVLoc : RVLocs) {
    Chain = DAG.getCopyFromReg(Chain, dl, RVLoc.getLocReg(), RVLoc.getValVT(), InGlue)
                .getValue(1);

    InGlue = Chain.getValue(2);
    InVals.push_back(Chain.getValue(0));
  }

  return Chain;
}




//===----------------------------------------------------------------------===//
//                              DIVISION LOWERING
//===----------------------------------------------------------------------===//

SDValue SuperHTargetLowering::LowerDIV(SDValue Op, SelectionDAG &DAG) const {
  unsigned Opcode = Op->getOpcode();
  assert((Opcode == ISD::SDIV || Opcode == ISD::UDIV) &&
         "Invalid opcode for Div lowering");
  bool IsSigned = (Opcode == ISD::SDIV);
  EVT VT = Op->getValueType(0);
  Type *Ty = VT.getTypeForEVT(*DAG.getContext());

  RTLIB::Libcall LC;
  switch (VT.getSimpleVT().SimpleTy) {
  default:
    llvm_unreachable("Unexpected request for libcall!");
  case MVT::i8:
    LC = IsSigned ? RTLIB::SDIV_I8 : RTLIB::UDIV_I8;
    break;
  case MVT::i16:
    LC = IsSigned ? RTLIB::SDIV_I16 : RTLIB::UDIV_I16;
    break;
  case MVT::i32:
    LC = IsSigned ? RTLIB::SDIV_I32 : RTLIB::UDIV_I32;
    break;
  }

  SDValue InChain = DAG.getEntryNode();

  TargetLowering::ArgListTy Args;
  for (SDValue const &Value : Op->op_values()) {
    TargetLowering::ArgListEntry Entry(
        Value, Value.getValueType().getTypeForEVT(*DAG.getContext()));
    Entry.IsSExt = IsSigned;
    Entry.IsZExt = !IsSigned;
    Args.push_back(Entry);
  }

  RTLIB::LibcallImpl LCImpl = DAG.getLibcalls().getLibcallImpl(LC);
  if (LCImpl == RTLIB::Unsupported)
    return SDValue();

  SDValue Callee =
      DAG.getExternalSymbol(LCImpl, getPointerTy(DAG.getDataLayout()));

  Type *RetTy = (Type *)StructType::get(Ty, Ty);


  SDLoc dl(Op);
  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(dl)
      .setChain(InChain)
      .setLibCallee(DAG.getLibcalls().getLibcallImplCallingConv(LCImpl), RetTy,
                    Callee, std::move(Args))
      .setInRegister()
      .setSExtResult(IsSigned)
      .setZExtResult(!IsSigned);

  std::pair<SDValue, SDValue> CallInfo = LowerCallTo(CLI);
  return CallInfo.first;
}




//===----------------------------------------------------------------------===//
//                          MULTIPLICATION LOWERING
//===----------------------------------------------------------------------===//

SDValue SuperHTargetLowering::LowerMUL(SDValue Op, SelectionDAG &DAG) const {
  unsigned Opcode = Op->getOpcode();
  assert((Opcode == ISD::MUL) && "Invalid opcode for MUL lowering");
  SDLoc DL(Op);

  EVT VT = Op->getValueType(0);
  Type *Ty = VT.getTypeForEVT(*DAG.getContext());

  switch (VT.getSimpleVT().SimpleTy) {
  default:
    llvm_unreachable("Unexpected type for ");

  case MVT::i8: {
    MachineSDNode *MUL = DAG.getMachineNode(SH::MULL, DL, VT, Op.getOperand(0), Op.getOperand(1));
    return DAG.getCopyFromReg(SDValue(MUL, 0), DL, SH::MACLO, VT);
  }

  case MVT::i16: {
    MachineSDNode *MUL = DAG.getMachineNode(SH::MULL, DL, VT, Op.getOperand(0), Op.getOperand(1));
    return DAG.getCopyFromReg(SDValue(MUL, 0), DL, SH::MACLO, VT);
  }

  case MVT::i32: {
    MachineSDNode *MUL = DAG.getMachineNode(SH::MULL, DL, VT, Op.getOperand(0), Op.getOperand(1));
    return DAG.getCopyFromReg(SDValue(MUL, 0), DL, SH::MACLO, VT);
  }

  case MVT::i64: {
    // 64-bit multiplication requires lowering to __mulsi3
    SDValue InChain = DAG.getEntryNode();

    TargetLowering::ArgListTy Args;
    for (SDValue const &Value : Op->op_values()) {
      TargetLowering::ArgListEntry Entry(
          Value, Value.getValueType().getTypeForEVT(*DAG.getContext()));
      Args.push_back(Entry);
    }

    RTLIB::LibcallImpl LCImpl = DAG.getLibcalls().getLibcallImpl(RTLIB::MUL_I32);
    if (LCImpl == RTLIB::Unsupported)
      return SDValue();

    SDValue Callee =
        DAG.getExternalSymbol(LCImpl, getPointerTy(DAG.getDataLayout()));

    Type *RetTy = (Type *)StructType::get(Ty, Ty);


    TargetLowering::CallLoweringInfo CLI(DAG);
    CLI.setDebugLoc(DL)
        .setChain(InChain)
        .setLibCallee(DAG.getLibcalls().getLibcallImplCallingConv(LCImpl), RetTy,
                      Callee, std::move(Args))
        .setInRegister();

    std::pair<SDValue, SDValue> CallInfo = LowerCallTo(CLI);
    return CallInfo.first;
  }
  }
}




//===----------------------------------------------------------------------===//
//                              CUSTOM LOWERING
//===----------------------------------------------------------------------===//

SDValue SuperHTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch(Op->getOpcode()) {
  case ISD::UDIV:
  case ISD::SDIV:
    return LowerDIV(Op, DAG);
  case ISD::MUL:
    return LowerMUL(Op, DAG);
  case ISD::Constant:
    return LowerConstant(Op, DAG);
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::ExternalSymbol:
    return LowerExternalSymbol(Op, DAG);
  case ISD::BlockAddress:
    return LowerBlockAddress(Op, DAG);
  case ISD::BR_CC:
    return LowerBR_CC(Op, DAG);
  case ISD::SETCC:
    return LowerSETCC(Op, DAG);
  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG);
  }
  return SDValue();
}

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
//                                INSERTERS
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//




//===----------------------------------------------------------------------===//
//                          SelectCC/SetCC Lowering
//===----------------------------------------------------------------------===//

MachineBasicBlock *SuperHTargetLowering::insertSELECTCC(MachineInstr &MI, 
                                                        MachineBasicBlock *MBB) const {
  DEBUG_FN_PRINT()

  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
  const SuperHInstrInfo &TII = *Subtarget->getInstrInfo();
  MachineBasicBlock::iterator BBI(MI);
  DebugLoc DL = MI.getDebugLoc();

  MachineOperand TrueV = MI.getOperand(0);
  MachineOperand FalseV = MI.getOperand(1);
  ISD::CondCode CC = (ISD::CondCode)MI.getOperand(3).getImm();

  // Lower True-False selection
  if (TrueV.isImm() && FalseV.isImm()) {
    if (TrueV.getImm() == 1 && FalseV.getImm() == 0) {
      BuildMI(*MBB, BBI, DL, TII.get(SH::MOVT), SH::R0);
      MI.eraseFromParent();
      return MBB;
    }
  }

  // To "insert" a SELECT instruction, we insert the diamond
  // control-flow pattern. The incoming instruction knows the
  // destination vreg to set, the condition code register to branch
  // on, the true/false values to select between, and a branch opcode
  // to use.

  MachineFunction *MF = MBB->getParent();
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineBasicBlock *FallThrough = MBB->getFallThrough();

  // If the current basic block falls through to another basic block,
  // we must insert an unconditional branch to the fallthrough destination
  // if we are to insert basic blocks at the prior fallthrough point.
  if (FallThrough != nullptr) {
    BuildMI(MBB, DL, TII.get(SH::BRA)).addMBB(FallThrough);
  }

  MachineBasicBlock *trueMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *falseMBB = MF->CreateMachineBasicBlock(LLVM_BB);

  MachineFunction::iterator I;
  for (I = MF->begin(); I != MF->end() && &(*I) != MBB; ++I)
    ;
  if (I != MF->end())
    ++I;
  MF->insert(I, trueMBB);
  MF->insert(I, falseMBB);

  // Set the call frame size on entry to the new basic blocks.
  unsigned CallFrameSize = TII.getCallFrameSizeAt(MI);
  trueMBB->setCallFrameSize(CallFrameSize);
  falseMBB->setCallFrameSize(CallFrameSize);

  // Transfer remaining instructions and all successors of the current
  // block to the block which will contain the Phi node for the
  // select.
  trueMBB->splice(trueMBB->begin(), MBB,
                  std::next(MachineBasicBlock::iterator(MI)), MBB->end());
  trueMBB->transferSuccessorsAndUpdatePHIs(MBB);

  BuildMI(MBB, DL, TII.getBrCond(CC)).addMBB(trueMBB);
  BuildMI(MBB, DL, TII.get(SH::BRA)).addMBB(falseMBB);
  MBB->addSuccessor(falseMBB);
  MBB->addSuccessor(trueMBB);

  // Unconditionally flow back to the true block
  BuildMI(falseMBB, DL, TII.get(SH::BRA)).addMBB(trueMBB);
  falseMBB->addSuccessor(trueMBB);

  // Set up the Phi node to determine where we came from
  BuildMI(*trueMBB, trueMBB->begin(), DL, TII.get(SH::PHI),
          MI.getOperand(0).getReg())
      .addReg(MI.getOperand(1).getReg())
      .addMBB(MBB)
      .addReg(MI.getOperand(2).getReg())
      .addMBB(falseMBB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return trueMBB;
}




//===----------------------------------------------------------------------===//
//                              CUSTOM LOWERING
//===----------------------------------------------------------------------===//

MachineBasicBlock *
SuperHTargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                                  MachineBasicBlock *MBB) const {
  int Op = MI.getOpcode();

  switch (Op) {
  default: break;
  case SH::Select8:
  case SH::Select16:
  case SH::Select32:
    return insertSELECTCC(MI, MBB);
  }

  return MBB;
}