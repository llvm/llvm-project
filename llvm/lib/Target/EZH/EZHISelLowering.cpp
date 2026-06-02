//===-- EZHISelLowering.cpp - EZH DAG Lowering Implementation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EZHISelLowering.h"
#include "EZHCondCode.h"
#include "EZHConstantPoolValue.h"
#include "EZHMachineFunctionInfo.h"
#include "EZHSubtarget.h"
#include "EZHTargetObjectFile.h"
#include "MCTargetDesc/EZHBaseInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetCallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_SDNODE_DESC
#include "EZHGenSDNodeInfo.inc"
#undef GET_SDNODE_DESC

using namespace llvm;

#define DEBUG_TYPE "ezh-lower"

EZHTargetLowering::EZHTargetLowering(const TargetMachine &TM,
                                     const EZHSubtarget &STI)
    : TargetLowering(TM, STI) {
  addRegisterClass(MVT::i32, &EZH::GPRRegClass);
  computeRegisterProperties(STI.getRegisterInfo());

  setOperationAction(ISD::LOAD, MVT::i64, Expand);
  setOperationAction(ISD::STORE, MVT::i64, Expand);

  setOperationAction(ISD::CTLZ, MVT::i32, Custom);
  setOperationAction(ISD::CTLZ, MVT::i16, Custom);
  setOperationAction(ISD::CTLZ, MVT::i8, Custom);
  setOperationAction(ISD::CTLZ_ZERO_POISON, MVT::i32, Custom);
  setOperationAction(ISD::CTLZ_ZERO_POISON, MVT::i16, Custom);
  setOperationAction(ISD::CTLZ_ZERO_POISON, MVT::i8, Custom);

  setStackPointerRegisterToSaveRestore(EZH::SP);

  setOperationAction(ISD::Constant, MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32, Custom);
  setOperationAction(ISD::ExternalSymbol, MVT::i32, Custom);
  setOperationAction(ISD::BlockAddress, MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);

  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::FRAMEADDR, MVT::i32, Custom);
  setOperationAction(ISD::EH_SJLJ_SETJMP, MVT::i32, Custom);
  setOperationAction(ISD::EH_SJLJ_LONGJMP, MVT::Other, Custom);
  setOperationAction(ISD::EH_SJLJ_SETUP_DISPATCH, MVT::Other, Custom);
  setOperationAction(ISD::JumpTable, MVT::i32, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  setOperationAction(ISD::BR_JT, MVT::Other, Custom);
  setOperationAction(ISD::BlockAddress, MVT::i32, Custom);

  setOperationAction(ISD::BR_CC, MVT::Other, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);

  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::BSWAP, MVT::i32, Legal);
  setOperationAction(ISD::BITREVERSE, MVT::i32, Legal);

  // Atomics
  setOperationAction(ISD::ATOMIC_LOAD, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_LOAD, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_LOAD, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_STORE, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_STORE, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_STORE, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_SWAP, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_SWAP, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_SWAP, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_ADD, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_ADD, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_ADD, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_AND, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_AND, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_AND, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_OR, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_OR, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_OR, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_XOR, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_XOR, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_XOR, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_NAND, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_NAND, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_NAND, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_MIN, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_MIN, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_MIN, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_MAX, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_MAX, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_MAX, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_UMIN, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_UMIN, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_UMIN, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_UMAX, MVT::i8, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_UMAX, MVT::i16, Expand);
  setOperationAction(ISD::ATOMIC_LOAD_UMAX, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Expand);

  setOperationAction(ISD::SETCC, MVT::i32, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);

  setOperationAction(ISD::MUL, MVT::i32, LibCall);
  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::MULHS, MVT::i32, Expand);
  setOperationAction(ISD::MULHU, MVT::i32, Expand);
  setOperationAction(ISD::SDIV, MVT::i32, LibCall);
  setOperationAction(ISD::UDIV, MVT::i32, LibCall);
  setOperationAction(ISD::SREM, MVT::i32, LibCall);
  setOperationAction(ISD::UREM, MVT::i32, LibCall);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);

  setOperationAction(ISD::SHL, MVT::i32, Custom);
  setOperationAction(ISD::SRL, MVT::i32, Custom);
  setOperationAction(ISD::SRA, MVT::i32, Custom);
  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTR, MVT::i32, Legal);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);

  setOperationAction(ISD::CTLZ, MVT::i64, Expand);
  setOperationAction(ISD::CTLZ_ZERO_POISON, MVT::i64, Expand);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ_ZERO_POISON, MVT::i32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, LibCall);

  // Mirror ARM soft-float architecture exactly!
  // We recursively register all basic float and double mathematical and casting
  // operations as strictly 'Expand' for both MVT::f32 and MVT::f64 to ensure
  // 100% EABI soft-float parity.
  for (auto VT : {MVT::f32, MVT::f64}) {
    setOperationAction(ISD::SINT_TO_FP, VT, Expand);
    setOperationAction(ISD::UINT_TO_FP, VT, Expand);
    setOperationAction(ISD::FP_TO_SINT, VT, Expand);
    setOperationAction(ISD::FP_TO_UINT, VT, Expand);

    setOperationAction(ISD::FADD, VT, Expand);
    setOperationAction(ISD::FSUB, VT, Expand);
    setOperationAction(ISD::FMUL, VT, Expand);
    setOperationAction(ISD::FDIV, VT, Expand);
    setOperationAction(ISD::FREM, VT, Expand);
    setOperationAction(ISD::FCOPYSIGN, VT, Expand);
    setOperationAction(ISD::FNEG, VT, Expand);
    setOperationAction(ISD::FMA, VT, Expand);
    setOperationAction(ISD::FSQRT, VT, Expand);
    setOperationAction(ISD::FPOW, VT, Expand);
    setOperationAction(ISD::FPOWI, VT, Expand);
    setOperationAction(ISD::FSIN, VT, Expand);
    setOperationAction(ISD::FCOS, VT, Expand);
    setOperationAction(ISD::FLOG, VT, Expand);
    setOperationAction(ISD::FLOG2, VT, Expand);
    setOperationAction(ISD::FLOG10, VT, Expand);
    setOperationAction(ISD::FEXP, VT, Expand);
    setOperationAction(ISD::FEXP2, VT, Expand);
  }

  setLibcallImpl(RTLIB::SDIV_I32, RTLIB::impl___divsi3);
  setLibcallImpl(RTLIB::UDIV_I32, RTLIB::impl___udivsi3);
  setLibcallImpl(RTLIB::SREM_I32, RTLIB::impl___modsi3);
  setLibcallImpl(RTLIB::UREM_I32, RTLIB::impl___umodsi3);
  setLibcallImpl(RTLIB::SDIVREM_I32, RTLIB::impl___divmodsi4);
  setLibcallImpl(RTLIB::UDIVREM_I32, RTLIB::impl___udivmodsi4);
  setLibcallImpl(RTLIB::MUL_I32, RTLIB::impl___mulsi3);
  setLibcallImpl(RTLIB::SHL_I32, RTLIB::impl___ashlsi3);
  setLibcallImpl(RTLIB::SRL_I32, RTLIB::impl___lshrsi3);
  setLibcallImpl(RTLIB::SRA_I32, RTLIB::impl___ashrsi3);

  setLibcallImpl(RTLIB::SDIV_I64, RTLIB::impl___divdi3);
  setLibcallImpl(RTLIB::UDIV_I64, RTLIB::impl___udivdi3);
  setLibcallImpl(RTLIB::SREM_I64, RTLIB::impl___moddi3);
  setLibcallImpl(RTLIB::UREM_I64, RTLIB::impl___umoddi3);
  setLibcallImpl(RTLIB::MUL_I64, RTLIB::impl___muldi3);
  setLibcallImpl(RTLIB::SHL_I64, RTLIB::impl___ashldi3);
  setLibcallImpl(RTLIB::SRL_I64, RTLIB::impl___lshrdi3);
  setLibcallImpl(RTLIB::SRA_I64, RTLIB::impl___ashrdi3);

  for (auto VT : {MVT::i8, MVT::i16}) {
    setOperationAction(ISD::MUL, VT, Promote);
    setOperationAction(ISD::SDIV, VT, Promote);
    setOperationAction(ISD::UDIV, VT, Promote);
    setOperationAction(ISD::SREM, VT, Promote);
    setOperationAction(ISD::UREM, VT, Promote);
    setOperationAction(ISD::SHL, VT, Promote);
    setOperationAction(ISD::SRL, VT, Promote);
    setOperationAction(ISD::SRA, VT, Promote);
    setOperationAction(ISD::ROTL, VT, Promote);
    setOperationAction(ISD::ROTR, VT, Promote);
    setOperationAction(ISD::SETCC, VT, Promote);
    setOperationAction(ISD::SELECT_CC, VT, Promote);
    setOperationAction(ISD::ADD, VT, Promote);
    setOperationAction(ISD::SUB, VT, Promote);
    setOperationAction(ISD::AND, VT, Promote);
    setOperationAction(ISD::OR, VT, Promote);
    setOperationAction(ISD::XOR, VT, Promote);
  }

  setOperationAction(ISD::MUL, MVT::i32, LibCall);
  setOperationAction(ISD::SDIV, MVT::i32, LibCall);
  setOperationAction(ISD::UDIV, MVT::i32, LibCall);
  setOperationAction(ISD::SREM, MVT::i32, LibCall);
  setOperationAction(ISD::UREM, MVT::i32, LibCall);

  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Custom);

  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAARG, MVT::Other, Custom);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  setLoadExtAction(ISD::EXTLOAD, MVT::i32, MVT::i16, Custom);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i32, MVT::i16, Custom);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i32, MVT::i16, Custom);
  setLoadExtAction(ISD::EXTLOAD, MVT::i32, MVT::i8, Legal);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i32, MVT::i8, Legal);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i32, MVT::i8, Legal);
  setTruncStoreAction(MVT::i32, MVT::i16, Custom);
  setTruncStoreAction(MVT::i32, MVT::i8, Legal);

  setOperationAction(ISD::LOAD, MVT::i16, Custom);
  setOperationAction(ISD::STORE, MVT::i16, Custom);

  setOperationAction(ISD::BUILD_PAIR, MVT::i64, Expand);
  setOperationAction(ISD::EXTRACT_ELEMENT, MVT::i32, Expand);

  for (MVT VT : {MVT::i32, MVT::i8}) {
    setOperationAction(ISD::LOAD, VT, Legal);
    setOperationAction(ISD::STORE, VT, Legal);
    setIndexedLoadAction(ISD::POST_INC, VT, Legal);
    setIndexedLoadAction(ISD::PRE_INC, VT, Legal);
    setIndexedLoadAction(ISD::POST_DEC, VT, Legal);
    setIndexedLoadAction(ISD::PRE_DEC, VT, Legal);
    setIndexedStoreAction(ISD::POST_INC, VT, Legal);
    setIndexedStoreAction(ISD::PRE_INC, VT, Legal);
    setIndexedStoreAction(ISD::POST_DEC, VT, Legal);
    setIndexedStoreAction(ISD::PRE_DEC, VT, Legal);
  }

  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);
}

bool EZHTargetLowering::getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                                   SDValue &Base,
                                                   SDValue &Offset,
                                                   ISD::MemIndexedMode &AM,
                                                   SelectionDAG &DAG) const {
  if (Op->getOpcode() != ISD::ADD && Op->getOpcode() != ISD::SUB)
    return false;

  if (auto *C = dyn_cast<ConstantSDNode>(Op->getOperand(1))) {
    int64_t Val = C->getSExtValue();
    if (Op->getOpcode() == ISD::SUB)
      Val = -Val;

    EVT VT = cast<MemSDNode>(N)->getMemoryVT();

    // For 32-bit loads/stores, hardware uses scaled offset (Value * 4)
    if (VT == MVT::i32) {
      if (Val >= -512 && Val <= 508 && (Val & 3) == 0) {
        Base = Op->getOperand(0);
        Offset = DAG.getConstant(std::abs(Val), SDLoc(Op), MVT::i32);
        AM = (Val < 0) ? ISD::POST_DEC : ISD::POST_INC;
        return true;
      }
    } else if (VT == MVT::i8) {
      if (Val >= -128 && Val <= 127) {
        Base = Op->getOperand(0);
        Offset = DAG.getConstant(std::abs(Val), SDLoc(Op), MVT::i32);
        AM = (Val < 0) ? ISD::POST_DEC : ISD::POST_INC;
        return true;
      }
    }
  }
  return false;
}

bool EZHTargetLowering::getPreIndexedAddressParts(SDNode *N, SDValue &Base,
                                                  SDValue &Offset,
                                                  ISD::MemIndexedMode &AM,
                                                  SelectionDAG &DAG) const {
  SDValue Ptr = cast<MemSDNode>(N)->getBasePtr();
  if (Ptr.getOpcode() != ISD::ADD && Ptr.getOpcode() != ISD::SUB)
    return false;

  if (auto *C = dyn_cast<ConstantSDNode>(Ptr.getOperand(1))) {
    int64_t Val = C->getSExtValue();
    if (Ptr.getOpcode() == ISD::SUB)
      Val = -Val;

    EVT VT = cast<MemSDNode>(N)->getMemoryVT();

    if (VT == MVT::i32) {
      if (Val >= -512 && Val <= 508 && (Val & 3) == 0) {
        Base = Ptr.getOperand(0);
        Offset = DAG.getConstant(std::abs(Val), SDLoc(Ptr), MVT::i32);
        AM = (Val < 0) ? ISD::PRE_DEC : ISD::PRE_INC;
        return true;
      }
    }

    else if (VT == MVT::i8) {
      if (Val >= -128 && Val <= 127) {
        Base = Ptr.getOperand(0);
        Offset = DAG.getConstant(std::abs(Val), SDLoc(Ptr), MVT::i32);
        AM = (Val < 0) ? ISD::PRE_DEC : ISD::PRE_INC;
        return true;
      }
    }
  }
  return false;
}

bool EZHTargetLowering::shouldReduceLoadWidth(
    SDNode *Load, ISD::LoadExtType ExtTy, EVT NewVT,
    std::optional<unsigned> ByteOffset) const {
  // EZH has no 16-bit load in hardware (it is emulated using two 8-bit loads).
  // Reducing load width to 16-bit is therefore never profitable.
  if (NewVT == MVT::i16)
    return false;
  return TargetLowering::shouldReduceLoadWidth(Load, ExtTy, NewVT, ByteOffset);
}

SDValue EZHTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::CTLZ:
  case ISD::CTLZ_ZERO_POISON: {
    SDLoc dl(Op);
    SDValue Src = Op.getOperand(0);
    EVT VT = Src.getValueType();

    if (VT == MVT::i64)
      return SDValue();

    TargetLowering::ArgListTy Args;
    Type *ArgTy = Type::getInt32Ty(*DAG.getContext());
    Args.push_back(TargetLowering::ArgListEntry(Src, ArgTy));

    const char *LibcallName = "__clzsi2";

    TargetLowering::CallLoweringInfo CLI(DAG);
    CLI.setDebugLoc(dl)
        .setChain(DAG.getEntryNode())
        .setCallee(CallingConv::C, Type::getInt32Ty(*DAG.getContext()),
                   DAG.getExternalSymbol(LibcallName,
                                         getPointerTy(DAG.getDataLayout())),
                   std::move(Args));

    std::pair<SDValue, SDValue> CallResult = LowerCallTo(CLI);
    return CallResult.first;
  }
  case ISD::STORE: {
    StoreSDNode *ST = cast<StoreSDNode>(Op);
    EVT MemVT = ST->getMemoryVT();
    if (MemVT == MVT::i16) {
      SDLoc DL(Op);
      SDValue Chain = ST->getChain();
      SDValue Ptr = ST->getBasePtr();
      SDValue Val = ST->getValue();

      // Store Low Byte
      SDValue Lo = DAG.getTruncStore(
          Chain, DL, Val, Ptr, ST->getPointerInfo(), MVT::i8, ST->getAlign(),
          ST->getMemOperand()->getFlags(), ST->getAAInfo());

      // Store High Byte (at Ptr + 1)
      SDValue PtrHi = DAG.getNode(ISD::ADD, DL, Ptr.getValueType(), Ptr,
                                  DAG.getConstant(1, DL, Ptr.getValueType()));

      // Ensure Val is 32-bit for shift
      SDValue Val32 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i32, Val);
      SDValue HiVal = DAG.getNode(ISD::SRL, DL, MVT::i32, Val32,
                                  DAG.getConstant(8, DL, MVT::i32));

      SDValue Hi = DAG.getTruncStore(
          Chain, DL, HiVal, PtrHi, ST->getPointerInfo().getWithOffset(1),
          MVT::i8, ST->getAlign(), ST->getMemOperand()->getFlags(),
          ST->getAAInfo());

      return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Lo, Hi);
    }
    return SDValue();
  }
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Op);
    EVT MemVT = LD->getMemoryVT();
    if (MemVT == MVT::i16) {
      SDLoc DL(Op);
      SDValue Chain = LD->getChain();
      SDValue Ptr = LD->getBasePtr();

      // Load Low Byte (ZExt to 32-bit)
      SDValue Lo =
          DAG.getExtLoad(ISD::ZEXTLOAD, DL, MVT::i32, Chain, Ptr,
                         LD->getPointerInfo(), MVT::i8, LD->getAlign(),
                         LD->getMemOperand()->getFlags(), LD->getAAInfo());

      // Load High Byte (ZExt to 32-bit, at Ptr + 1)
      SDValue PtrHi = DAG.getNode(ISD::ADD, DL, Ptr.getValueType(), Ptr,
                                  DAG.getConstant(1, DL, Ptr.getValueType()));
      SDValue Hi = DAG.getExtLoad(
          ISD::ZEXTLOAD, DL, MVT::i32, Chain, PtrHi,
          LD->getPointerInfo().getWithOffset(1), MVT::i8, LD->getAlign(),
          LD->getMemOperand()->getFlags(), LD->getAAInfo());

      // Combine: (Hi << 8) | Lo
      SDValue HiShifted = DAG.getNode(ISD::SHL, DL, MVT::i32, Hi,
                                      DAG.getConstant(8, DL, MVT::i32));
      SDValue Res = DAG.getNode(ISD::OR, DL, MVT::i32, HiShifted, Lo);

      if (LD->getExtensionType() == ISD::SEXTLOAD) {
        SDValue ResShifted = DAG.getNode(ISD::SHL, DL, MVT::i32, Res,
                                         DAG.getConstant(16, DL, MVT::i32));
        Res = DAG.getNode(ISD::SRA, DL, MVT::i32, ResShifted,
                          DAG.getConstant(16, DL, MVT::i32));
      }

      // Return merged value AND merged chain
      SDValue NewChain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other,
                                     Lo.getValue(1), Hi.getValue(1));
      return DAG.getMergeValues({Res, NewChain}, DL);
    }
    return SDValue();
  }
  case ISD::JumpTable:
    return LowerJumpTable(Op, DAG);
  case ISD::BR_JT:
    return LowerBR_JT(Op, DAG);
  case ISD::BlockAddress:
    return LowerBlockAddress(Op, DAG);
  case ISD::ConstantPool:
    return LowerConstantPool(Op, DAG);
  case ISD::Constant:
    return LowerConstant(Op, DAG);
  case ISD::GlobalAddress:
  case ISD::GlobalTLSAddress:
  case ISD::ExternalSymbol:
    return LowerGlobalAddress(Op, DAG);
  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG);
  case ISD::VASTART:
    return LowerVASTART(Op, DAG);
  case ISD::VAARG:
    return LowerVAARG(Op, DAG);
  case ISD::BR_CC:
    return LowerBR_CC(Op, DAG);
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    return Op;
  case ISD::EH_SJLJ_SETJMP: {
    SDLoc dl(Op);
    return DAG.getNode(EZHISD::EH_SJLJ_SETJMP, dl,
                       DAG.getVTList(MVT::i32, MVT::Other), Op.getOperand(0),
                       Op.getOperand(1));
  }
  case ISD::EH_SJLJ_LONGJMP: {
    SDLoc dl(Op);
    return DAG.getNode(EZHISD::EH_SJLJ_LONGJMP, dl, MVT::Other,
                       Op.getOperand(0), Op.getOperand(1));
  }
  case ISD::EH_SJLJ_SETUP_DISPATCH: {
    SDLoc dl(Op);
    return DAG.getNode(EZHISD::EH_SJLJ_SETUP_DISPATCH, dl, MVT::Other,
                       Op.getOperand(0));
  }
  case ISD::FRAMEADDR:
    return LowerFRAMEADDR(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC:
    return LowerDYNAMIC_STACKALLOC(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN:
    return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  default:
    return SDValue();
  }
}

void EZHTargetLowering::ReplaceNodeResults(SDNode *N,
                                           SmallVectorImpl<SDValue> &Results,
                                           SelectionDAG &DAG) const {}
static unsigned IntCCToEZHCC(ISD::CondCode CC);

SDValue EZHTargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue TrueV = Op.getOperand(2);
  SDValue FalseV = Op.getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDLoc DL(Op);

  if (LHS.getValueType() != MVT::i32)
    return SDValue();

  if (CC == ISD::SETGT || CC == ISD::SETGE || CC == ISD::SETLT ||
      CC == ISD::SETLE) {
    bool LHSExt = (LHS.getOpcode() == ISD::SIGN_EXTEND_INREG ||
                   LHS.getOpcode() == ISD::SIGN_EXTEND ||
                   LHS.getOpcode() == ISD::ZERO_EXTEND ||
                   LHS.getOpcode() == ISD::ANY_EXTEND);
    bool RHSExt = (RHS.getOpcode() == ISD::SIGN_EXTEND_INREG ||
                   RHS.getOpcode() == ISD::SIGN_EXTEND ||
                   RHS.getOpcode() == ISD::ZERO_EXTEND ||
                   RHS.getOpcode() == ISD::ANY_EXTEND);
    bool LHSConst = false;
    if (auto *C = dyn_cast<ConstantSDNode>(LHS)) {
      int64_t Val = C->getSExtValue();
      if (Val >= -32768 && Val <= 32767)
        LHSConst = true;
    }
    bool RHSConst = false;
    if (auto *C = dyn_cast<ConstantSDNode>(RHS)) {
      int64_t Val = C->getSExtValue();
      if (Val >= -32768 && Val <= 32767)
        RHSConst = true;
    }
    bool IsI8OrI16 = (LHSExt && RHSExt) || (LHSExt && RHSConst) ||
                     (RHSExt && LHSConst) || (LHSConst && RHSConst);

    if (!IsI8OrI16) {
      LHS = DAG.getNode(EZHISD::BTOG, DL, MVT::i32, LHS,
                        DAG.getConstant(31, DL, MVT::i32));
      RHS = DAG.getNode(EZHISD::BTOG, DL, MVT::i32, RHS,
                        DAG.getConstant(31, DL, MVT::i32));

      switch (CC) {
      case ISD::SETGT:
        CC = ISD::SETUGT;
        break;
      case ISD::SETGE:
        CC = ISD::SETUGE;
        break;
      case ISD::SETLT:
        CC = ISD::SETULT;
        break;
      case ISD::SETLE:
        CC = ISD::SETULE;
        break;
      default:
        llvm_unreachable("Invalid signed condition code");
      }
    }
  }

  if (CC == ISD::SETUGT) {
    CC = ISD::SETULT;
    std::swap(LHS, RHS);
  } else if (CC == ISD::SETULE) {
    CC = ISD::SETUGE;
    std::swap(LHS, RHS);
  }

  SDValue TargetCC = DAG.getTargetConstant(CC, DL, MVT::i32);
  SDValue Cmp = DAG.getNode(EZHISD::CMP, DL, MVT::Glue, LHS, RHS);
  SDValue Ops[] = {TrueV, FalseV, TargetCC, Cmp};
  return DAG.getNode(EZHISD::SELECT_CC, DL, Op.getValueType(), Ops);
}

SDValue EZHTargetLowering::LowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  EZHMachineFunctionInfo *FuncInfo = MF.getInfo<EZHMachineFunctionInfo>();

  SDValue Ptr = Op.getOperand(1);
  EVT PtrVT = Ptr.getValueType();

  SDLoc dl(Op);
  SDValue FrameIndex =
      DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(), PtrVT);

  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();

  return DAG.getStore(Op.getOperand(0), dl, FrameIndex, Ptr,
                      MachinePointerInfo(SV));
}

SDValue EZHTargetLowering::LowerVAARG(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  EVT VT = Node->getValueType(0);
  SDValue InChain = Node->getOperand(0);
  SDValue VAListPtr = Node->getOperand(1);
  EVT PtrVT = VAListPtr.getValueType();
  const Value *SV = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
  SDLoc dl(Node);

  // Load the current VAList pointer (which points to the next arg on the stack)
  SDValue VAList =
      DAG.getLoad(PtrVT, dl, InChain, VAListPtr, MachinePointerInfo(SV));
  SDValue VAListInChain = VAList.getValue(1);

  // Align VAList pointer if the argument type alignment requires it (> 4-byte
  // alignment).
  Type *Ty = VT.getTypeForEVT(*DAG.getContext());
  auto &TD = DAG.getDataLayout();
  Align ArgAlignment = TD.getABITypeAlign(Ty);
  unsigned ArgAlignInBytes = ArgAlignment.value();

  SDValue ArgPtr = VAList;
  if (ArgAlignInBytes > 4) {
    unsigned AlignMask = ArgAlignInBytes - 1;
    SDValue AddOffset = DAG.getNode(ISD::ADD, dl, PtrVT, VAList,
                                    DAG.getConstant(AlignMask, dl, PtrVT));
    ArgPtr = DAG.getNode(ISD::AND, dl, PtrVT, AddOffset,
                         DAG.getConstant(~AlignMask, dl, PtrVT));
  }

  // Increment the VAList pointer past the current argument.
  // Round up the argument size to the nearest multiple of 4 bytes (32 bits)
  // as all varargs arguments on EZH stack are aligned to at least 4 bytes.
  unsigned ArgSize = (VT.getSizeInBits() + 31) / 32 * 4;
  SDValue NextVAList = DAG.getNode(ISD::ADD, dl, PtrVT, ArgPtr,
                                   DAG.getConstant(ArgSize, dl, PtrVT));

  // Store the incremented VAList pointer back.
  SDValue StoreChain = DAG.getStore(VAListInChain, dl, NextVAList, VAListPtr,
                                    MachinePointerInfo(SV));

  // Load the actual argument from ArgPtr.
  SDValue ArgVal =
      DAG.getLoad(VT, dl, StoreChain, ArgPtr, MachinePointerInfo());

  // Return the loaded value and the new chain.
  return DAG.getMergeValues({ArgVal, ArgVal.getValue(1)}, dl);
}

SDValue EZHTargetLowering::LowerJumpTable(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  auto *JT = cast<JumpTableSDNode>(Op);

  // Mark target blocks as address-taken to prevent deletion!
  MachineFunction &MF = DAG.getMachineFunction();
  if (MachineJumpTableInfo *MJTI = MF.getJumpTableInfo()) {
    const std::vector<MachineJumpTableEntry> &JTEntries = MJTI->getJumpTables();
    int Index = JT->getIndex();
    if (Index >= 0 && static_cast<size_t>(Index) < JTEntries.size())
      for (MachineBasicBlock *MBB : JTEntries[Index].MBBs)
        MBB->setMachineBlockAddressTaken();
  }

  EZHConstantPoolValue *CPV = new EZHConstantPoolValue(
      JT->getIndex(), Type::getInt32Ty(*DAG.getContext()));
  SDValue CPIdx =
      DAG.getTargetConstantPool(CPV, getPointerTy(DAG.getDataLayout()));
  SDValue Ops[] = {CPIdx, DAG.getEntryNode()};
  return SDValue(
      DAG.getMachineNode(EZH::LOAD_CONSTANT, DL, MVT::i32, MVT::Other, Ops), 0);
}

SDValue EZHTargetLowering::LowerBR_JT(SDValue Op, SelectionDAG &DAG) const {
  SDLoc dl(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Table = Op.getOperand(1);
  SDValue Index = Op.getOperand(2);

  int JTI = -1;
  if (auto *JT = dyn_cast<JumpTableSDNode>(Table)) {
    JTI = JT->getIndex();
  }

  assert(JTI != -1 && "Failed to extract JTI for BR_JT!");

  SDValue JTIVal =
      DAG.getTargetJumpTable(JTI, getPointerTy(DAG.getDataLayout()));
  return DAG.getNode(EZHISD::BR_JT, dl, MVT::Other, Chain, Table, Index,
                     JTIVal);
}

SDValue EZHTargetLowering::LowerBlockAddress(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc DL(Op);
  auto *BA = cast<BlockAddressSDNode>(Op);

  EZHConstantPoolValue *CPV = new EZHConstantPoolValue(
      BA->getBlockAddress(), Type::getInt32Ty(*DAG.getContext()));
  SDValue CPIdx =
      DAG.getTargetConstantPool(CPV, getPointerTy(DAG.getDataLayout()));
  SDValue Ops[] = {CPIdx, DAG.getEntryNode()};
  return SDValue(
      DAG.getMachineNode(EZH::LOAD_CONSTANT, DL, MVT::i32, MVT::Other, Ops), 0);
}

Register EZHTargetLowering::getRegisterByName(const char *RegName, LLT Ty,
                                              const MachineFunction &MF) const {
  StringRef Name(RegName);
  if (Name == "r0")
    return EZH::R0;
  if (Name == "r1")
    return EZH::R1;
  if (Name == "r2")
    return EZH::R2;
  if (Name == "r3")
    return EZH::R3;
  if (Name == "r4")
    return EZH::R4;
  if (Name == "r5")
    return EZH::R5;
  if (Name == "r6")
    return EZH::R6;
  if (Name == "r7")
    return EZH::R7;
  if (Name == "sp")
    return EZH::SP;
  if (Name == "pc")
    return EZH::PC;
  if (Name == "gpo")
    return EZH::GPO;
  if (Name == "gpd")
    return EZH::GPD;
  if (Name == "cfs")
    return EZH::CFS;
  if (Name == "cfm")
    return EZH::CFM;
  if (Name == "gpi")
    return EZH::GPI;
  if (Name == "ra")
    return EZH::RA;
  report_fatal_error("invalid register name");
}

TargetLowering::ConstraintType
EZHTargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      return C_RegisterClass;
    default:
      break;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass *>
EZHTargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                                StringRef Constraint,
                                                MVT VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      return std::make_pair(0U, &EZH::GPRRegClass);
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

void EZHTargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                     StringRef Constraint,
                                                     std::vector<SDValue> &Ops,
                                                     SelectionDAG &DAG) const {
  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

SDValue EZHTargetLowering::LowerConstantPool(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc DL(Op);
  auto *CP = cast<ConstantPoolSDNode>(Op);

  // 1. Retrieve the index of the original constant in the constant pool
  MachineConstantPool *MCP = DAG.getMachineFunction().getConstantPool();
  int OrigIndex = -1;
  if (CP->isMachineConstantPoolEntry())
    OrigIndex =
        MCP->getConstantPoolIndex(CP->getMachineCPVal(), CP->getAlign());
  else
    OrigIndex = MCP->getConstantPoolIndex(CP->getConstVal(), CP->getAlign());

  // 2. Wrap the CPI in EZHConstantPoolValue to force loading its absolute
  // address!
  EZHConstantPoolValue *CPV =
      new EZHConstantPoolValue(OrigIndex, CP->getType(), true);
  SDValue TargetCP = DAG.getTargetConstantPool(
      CPV, getPointerTy(DAG.getDataLayout()), CP->getAlign(), CP->getOffset(),
      CP->getTargetFlags());

  SDValue Ops[] = {TargetCP, DAG.getEntryNode()};
  return SDValue(
      DAG.getMachineNode(EZH::LOAD_CONSTANT, DL, MVT::i32, MVT::Other, Ops), 0);
}

SDValue EZHTargetLowering::LowerConstant(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  auto *CN = cast<ConstantSDNode>(Op);
  int64_t Val = CN->getSExtValue();

  if (isInt<11>(Val)) {
    return Op; // Natively matched
  }

  uint32_t UVal = static_cast<uint32_t>(CN->getZExtValue());

  uint32_t HiVal = (UVal >> 20);
  if ((UVal & 0x000FFFFF) == 0 && isInt<11>(HiVal)) {
    SDValue Hi = DAG.getTargetConstant(HiVal, DL, MVT::i32);
    SDValue Lo = DAG.getTargetConstant(UVal & 0xFFF, DL, MVT::i32);
    SDValue Shift = DAG.getTargetConstant(20, DL, MVT::i32);

    SDValue HiNode =
        SDValue(DAG.getMachineNode(EZH::MOVSri__, DL, MVT::i32, Hi, Shift), 0);
    SDValue LoNode =
        SDValue(DAG.getMachineNode(EZH::ORri__, DL, MVT::i32, HiNode, Lo), 0);
    return LoNode;
  }

  // For large constants, use an inline literal pool (LOAD_CONSTANT)
  SDValue CPIdx = DAG.getTargetConstantPool(
      ConstantInt::get(Type::getInt32Ty(*DAG.getContext()), UVal),
      getPointerTy(DAG.getDataLayout()));

  SDValue Ops[] = {CPIdx, DAG.getEntryNode()};
  return SDValue(
      DAG.getMachineNode(EZH::LOAD_CONSTANT, DL, MVT::i32, MVT::Other, Ops), 0);
}

SDValue EZHTargetLowering::LowerGlobalAddress(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);

  if (auto *GV = dyn_cast<GlobalAddressSDNode>(Op)) {
    SDValue CPIdx;
    if (GV->getOffset() != 0) {
      EZHConstantPoolValue *CPV =
          new EZHConstantPoolValue(GV->getGlobal(), GV->getOffset(),
                                   Type::getInt32Ty(*DAG.getContext()));
      CPIdx = DAG.getTargetConstantPool(CPV, getPointerTy(DAG.getDataLayout()));
    } else {
      CPIdx = DAG.getTargetConstantPool(GV->getGlobal(),
                                        getPointerTy(DAG.getDataLayout()));
    }
    SDValue Ops[] = {CPIdx, DAG.getEntryNode()};
    return SDValue(
        DAG.getMachineNode(EZH::LOAD_CONSTANT, DL, MVT::i32, MVT::Other, Ops),
        0);
  }

  if (auto *S = dyn_cast<ExternalSymbolSDNode>(Op)) {
    EZHConstantPoolValue *CPV = new EZHConstantPoolValue(
        S->getSymbol(), Type::getInt32Ty(*DAG.getContext()));
    SDValue CPIdx =
        DAG.getTargetConstantPool(CPV, getPointerTy(DAG.getDataLayout()));
    SDValue Ops[] = {CPIdx, DAG.getEntryNode()};
    return SDValue(
        DAG.getMachineNode(EZH::LOAD_CONSTANT, DL, MVT::i32, MVT::Other, Ops),
        0);
  }

  llvm_unreachable("Unhandled global address type");
}

#include "EZHGenCallingConv.inc"

SDValue EZHTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_EZH);

  DenseMap<unsigned, int> SplitArgFIs;

  assert(ArgLocs.size() == Ins.size() && "ArgLocs and Ins size mismatch!");
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    ISD::ArgFlagsTy Flags = Ins[i].Flags;
    if (Flags.isByVal()) {
      unsigned Size = Flags.getByValSize();
      if (Size == 0) {
        InVals.push_back(DAG.getUNDEF(getPointerTy(DAG.getDataLayout())));
        continue;
      }
      int FI =
          MF.getFrameInfo().CreateFixedObject(Size, VA.getLocMemOffset(), true);
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      InVals.push_back(FIN);
    } else if (VA.isRegLoc()) {
      Register VReg = RegInfo.createVirtualRegister(&EZH::GPRRegClass);
      RegInfo.addLiveIn(VA.getLocReg(), VReg);
      InVals.push_back(DAG.getCopyFromReg(Chain, DL, VReg, VA.getLocVT()));
    } else {
      int FI;
      bool IsSplit = false;
      unsigned OrigArgIdx = 0;

      if (Ins[i].isOrigArg() && (Ins[i].ArgVT.getSizeInBits() / 8 >
                                 VA.getLocVT().getSizeInBits() / 8)) {
        IsSplit = true;
        OrigArgIdx = Ins[i].getOrigArgIndex();
        unsigned ValVTSize = Ins[i].ArgVT.getSizeInBits() / 8;
        auto It = SplitArgFIs.find(OrigArgIdx);
        if (It == SplitArgFIs.end()) {
          FI = MF.getFrameInfo().CreateFixedObject(ValVTSize,
                                                   VA.getLocMemOffset(), true);
          SplitArgFIs[OrigArgIdx] = FI;
        } else {
          FI = It->second;
        }
      }

      if (!IsSplit)
        FI = MF.getFrameInfo().CreateFixedObject(
            VA.getLocVT().getSizeInBits() / 8, VA.getLocMemOffset(), true);

      int FIOffset = MF.getFrameInfo().getObjectOffset(FI);
      unsigned Offset = VA.getLocMemOffset() - FIOffset;

      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      if (Offset != 0)
        FIN = DAG.getObjectPtrOffset(DL, FIN, TypeSize::getFixed(Offset));

      InVals.push_back(
          DAG.getLoad(VA.getValVT(), DL, Chain, FIN,
                      MachinePointerInfo::getFixedStack(MF, FI, Offset)));
    }
  }

  if (IsVarArg) {
    EZHMachineFunctionInfo *FuncInfo = MF.getInfo<EZHMachineFunctionInfo>();
    static const MCPhysReg ArgRegs[] = {EZH::R0, EZH::R1, EZH::R2, EZH::R3};
    unsigned NumArgRegs = std::size(ArgRegs);
    unsigned Idx = CCInfo.getFirstUnallocated(ArgRegs);

    unsigned NumSpillBytes = (NumArgRegs - Idx) * 4;
    unsigned NumStackBytes = CCInfo.getStackSize();
    FuncInfo->setVarArgsRegIdx(Idx);

    int VarArgsFI;
    SmallVector<SDValue, 4> MemOps;

    if (NumSpillBytes > 0) {
      FuncInfo->setVarArgsSaveSize(NumSpillBytes);
      int VaArgOffset = -static_cast<int>(NumSpillBytes);
      VarArgsFI =
          MF.getFrameInfo().CreateFixedObject(NumSpillBytes, VaArgOffset, true);
      FuncInfo->setVarArgsFrameIndex(VarArgsFI);

      SDValue FIN =
          DAG.getFrameIndex(VarArgsFI, getPointerTy(DAG.getDataLayout()));

      for (unsigned i = Idx; i < NumArgRegs; ++i) {
        Register VReg = RegInfo.createVirtualRegister(&EZH::GPRRegClass);
        RegInfo.addLiveIn(ArgRegs[i], VReg);
        SDValue Arg = DAG.getCopyFromReg(Chain, DL, VReg, MVT::i32);

        unsigned StoreOffset = (i - Idx) * 4;
        SDValue Addr = DAG.getNode(ISD::ADD, DL, MVT::i32, FIN,
                                   DAG.getConstant(StoreOffset, DL, MVT::i32));
        MemOps.push_back(DAG.getStore(
            Chain, DL, Arg, Addr,
            MachinePointerInfo::getFixedStack(MF, VarArgsFI, StoreOffset)));
      }
    } else {
      VarArgsFI = MF.getFrameInfo().CreateFixedObject(4, NumStackBytes, true);
      FuncInfo->setVarArgsFrameIndex(VarArgsFI);
    }

    if (!MemOps.empty())
      Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOps);
  }

  return Chain;
}

SDValue EZHTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                                     SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;

  CLI.IsTailCall = false;

  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), DL,
                                        getPointerTy(DAG.getDataLayout()));
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(),
                                         getPointerTy(DAG.getDataLayout()));

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeCallOperands(Outs, CC_EZH);

  unsigned NumBytes = CCInfo.getStackSize();
  Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, DL);

  SmallVector<std::pair<unsigned, SDValue>, 4> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;

  SDValue StackPtr =
      DAG.getCopyFromReg(Chain, DL, EZH::SP, getPointerTy(DAG.getDataLayout()));

  assert(ArgLocs.size() == Outs.size() && "ArgLocs and Outs size mismatch!");
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;

    if (Flags.isByVal()) {
      unsigned Size = Flags.getByValSize();
      if (Size == 0)
        continue;
      Align Alignment = Flags.getNonZeroByValAlign();
      SDValue DstAddr =
          DAG.getNode(ISD::ADD, DL, getPointerTy(DAG.getDataLayout()), StackPtr,
                      DAG.getIntPtrConstant(VA.getLocMemOffset(), DL));
      SDValue MemCpy = DAG.getMemcpy(
          Chain, DL, DstAddr, Arg,
          DAG.getConstant(Size, DL, getPointerTy(DAG.getDataLayout())),
          Alignment, /*isVol=*/false, /*AlwaysInline=*/false,
          /*CI=*/nullptr, /*OverrideTailCall=*/std::nullopt,
          MachinePointerInfo(), MachinePointerInfo());
      MemOpChains.push_back(MemCpy);
      continue;
    }

    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      // Store to stack
      int32_t Offset = VA.getLocMemOffset();
      SDValue PtrOff = DAG.getConstant(static_cast<uint32_t>(Offset), DL,
                                       getPointerTy(DAG.getDataLayout()));
      SDValue DstAddr = DAG.getNode(
          ISD::ADD, DL, getPointerTy(DAG.getDataLayout()), StackPtr, PtrOff);
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, Arg, DstAddr, MachinePointerInfo()));
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  SDValue InGlue;
  for (auto &Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, Reg.first, Reg.second, InGlue);
    InGlue = Chain.getValue(1);
  }

  SDValue CalleeVal = Callee;

  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(CalleeVal);

  for (auto &Reg : RegsToPass)
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));

  if (InGlue.getNode())
    Ops.push_back(InGlue);

  Chain =
      DAG.getNode(EZHISD::CALL, DL, DAG.getVTList(MVT::Other, MVT::Glue), Ops);
  InGlue = Chain.getValue(1);

  Chain = DAG.getCALLSEQ_END(Chain, NumBytes, 0, InGlue, DL);
  InGlue = Chain.getValue(1);

  return LowerCallResult(Chain, InGlue, CallConv, IsVarArg, Ins, DL, DAG,
                         InVals);
}

SDValue EZHTargetLowering::LowerCallResult(
    SDValue Chain, SDValue InGlue, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, RetCC_EZH);

  unsigned VaIdx = 0;
  for (auto &VA : RVLocs) {
    if (!VA.isRegLoc()) {
      llvm::errs() << "=== EZH LowerCallResult VA #" << VaIdx << " ===\n";
      llvm::errs() << "  isReg: " << VA.isRegLoc() << "\n";
      llvm::errs() << "  isMem: " << VA.isMemLoc() << "\n";
      llvm::errs() << "  LocVT: " << EVT(VA.getLocVT()).getEVTString() << "\n";
      llvm::errs() << "  MemOffset: " << VA.getLocMemOffset() << "\n";
    }
    ++VaIdx;
    assert(VA.isRegLoc() && "Only register returns supported!");
    Chain = DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getValVT(), InGlue)
                .getValue(1);
    InGlue = Chain.getValue(2);
    InVals.push_back(Chain.getValue(0));
  }
  return Chain;
}

SDValue
EZHTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<SDValue> &OutVals,
                               const SDLoc &DL, SelectionDAG &DAG) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC_EZH);

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  for (unsigned i = 0, e = RVLocs.size(); i != e; ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Only register returns supported!");
    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), OutVals[i], Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain;
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(EZHISD::RET_GLUE_INTERNAL, DL, MVT::Other, RetOps);
}

EVT EZHTargetLowering::getSetCCResultType(const DataLayout &DL,
                                          LLVMContext &Context, EVT VT) const {
  if (!VT.isVector())
    return getPointerTy(DL);
  return VT.changeVectorElementTypeToInteger();
}

bool EZHTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context,
    const Type *RetTy) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);
  return CCInfo.CheckReturn(Outs, RetCC_EZH);
}

const char *EZHTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (static_cast<EZHISD::NodeType>(Opcode)) {
  case EZHISD::FIRST_NUMBER:
    break;
  case EZHISD::RET_GLUE_INTERNAL:
    return "EZHISD::RET_GLUE_INTERNAL";
  case EZHISD::CALL:
    return "EZHISD::CALL";
  case EZHISD::CMP:
    return "EZHISD::CMP";
  case EZHISD::BR_CC:
    return "EZHISD::BR_CC";
  case EZHISD::SELECT_CC:
    return "EZHISD::SELECT_CC";
  case EZHISD::BTOG:
    return "EZHISD::BTOG";
  case EZHISD::BR_JT:
    return "EZHISD::BR_JT";
  case EZHISD::EH_SJLJ_SETJMP:
    return "EZHISD::EH_SJLJ_SETJMP";
  case EZHISD::EH_SJLJ_LONGJMP:
    return "EZHISD::EH_SJLJ_LONGJMP";
  case EZHISD::EH_SJLJ_SETUP_DISPATCH:
    return "EZHISD::EH_SJLJ_SETUP_DISPATCH";
  }
  return nullptr;
}

static unsigned IntCCToEZHCC(ISD::CondCode CC) {
  switch (CC) {
  default:
    llvm_unreachable("Unknown condition code!");
  case ISD::SETEQ:
    return 1; // ZE: Zero
  case ISD::SETNE:
    return 2; // NZ: Not Zero
  case ISD::SETGE:
    return 3; // PO: Positive (GE / Signed Greater or Equal)
  case ISD::SETLT:
    return 4; // NE: Negative (LT / Signed Less Than)
  case ISD::SETGT:
    return 5; // AZ: Above Zero (GT)
  case ISD::SETLE:
    return 6; // ZB: Zero or Below (LE)
  case ISD::SETULT:
    return 7; // CA: Carry Set (HS/LO: Unsigned Less Than / Borrow)
  case ISD::SETUGE:
    return 8; // NC: Carry Not Set (LO/HS: Unsigned Greater or Same / No Borrow)
  }
}

SDValue EZHTargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);
  SDLoc dl(Op);

  if (CC == ISD::SETGT || CC == ISD::SETGE || CC == ISD::SETLT ||
      CC == ISD::SETLE) {
    bool LHSExt = (LHS.getOpcode() == ISD::SIGN_EXTEND_INREG ||
                   LHS.getOpcode() == ISD::SIGN_EXTEND ||
                   LHS.getOpcode() == ISD::ZERO_EXTEND ||
                   LHS.getOpcode() == ISD::ANY_EXTEND);
    bool RHSExt = (RHS.getOpcode() == ISD::SIGN_EXTEND_INREG ||
                   RHS.getOpcode() == ISD::SIGN_EXTEND ||
                   RHS.getOpcode() == ISD::ZERO_EXTEND ||
                   RHS.getOpcode() == ISD::ANY_EXTEND);
    bool LHSConst = false;
    if (auto *C = dyn_cast<ConstantSDNode>(LHS)) {
      int64_t Val = C->getSExtValue();
      if (Val >= -32768 && Val <= 32767)
        LHSConst = true;
    }
    bool RHSConst = false;
    if (auto *C = dyn_cast<ConstantSDNode>(RHS)) {
      int64_t Val = C->getSExtValue();
      if (Val >= -32768 && Val <= 32767)
        RHSConst = true;
    }
    bool IsI8OrI16 = (LHSExt && RHSExt) || (LHSExt && RHSConst) ||
                     (RHSExt && LHSConst) || (LHSConst && RHSConst);

    if (!IsI8OrI16) {
      LHS = DAG.getNode(EZHISD::BTOG, dl, MVT::i32, LHS,
                        DAG.getConstant(31, dl, MVT::i32));
      RHS = DAG.getNode(EZHISD::BTOG, dl, MVT::i32, RHS,
                        DAG.getConstant(31, dl, MVT::i32));

      switch (CC) {
      case ISD::SETGT:
        CC = ISD::SETUGT;
        break;
      case ISD::SETGE:
        CC = ISD::SETUGE;
        break;
      case ISD::SETLT:
        CC = ISD::SETULT;
        break;
      case ISD::SETLE:
        CC = ISD::SETULE;
        break;
      default:
        llvm_unreachable("Invalid signed condition code");
      }
    }
  }

  if (CC == ISD::SETUGT) {
    CC = ISD::SETULT;
    std::swap(LHS, RHS);
  } else if (CC == ISD::SETULE) {
    CC = ISD::SETUGE;
    std::swap(LHS, RHS);
  }

  unsigned EzhCC = IntCCToEZHCC(CC);
  SDValue TargetCC = DAG.getTargetConstant(EzhCC, dl, MVT::i32);
  SDValue Cmp = DAG.getNode(EZHISD::CMP, dl, MVT::Glue, LHS, RHS);
  return DAG.getNode(EZHISD::BR_CC, dl, MVT::Other, Chain, Dest, TargetCC, Cmp);
}

#include "EZHInstrInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

MachineBasicBlock *
EZHTargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                               MachineBasicBlock *BB) const {
  const EZHSubtarget &STI =
      MI.getParent()->getParent()->getSubtarget<EZHSubtarget>();
  const TargetInstrInfo *TII = STI.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();
  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  switch (MI.getOpcode()) {
  case EZH::PseudoCMP: {
    Register DestReg = MRI.createVirtualRegister(&EZH::GPRRegClass);
    BuildMI(*BB, MI, DL, TII->get(EZH::SUBrr_s_), DestReg)
        .addReg(MI.getOperand(0).getReg())
        .addReg(MI.getOperand(1).getReg());
    MI.eraseFromParent();
    return BB;
  }
  case EZH::PseudoCMPi: {
    Register DestReg = MRI.createVirtualRegister(&EZH::GPRRegClass);
    BuildMI(*BB, MI, DL, TII->get(EZH::SUBri_s_), DestReg)
        .addReg(MI.getOperand(0).getReg())
        .addImm(MI.getOperand(1).getImm());
    MI.eraseFromParent();
    return BB;
  }
  case EZH::PseudoBR_CC: {
    unsigned CC = MI.getOperand(1).getImm();
    MachineBasicBlock *TrueBB = MI.getOperand(0).getMBB();
    MachineBasicBlock *FalseBB = nullptr;
    for (auto *Succ : BB->successors()) {
      if (Succ != TrueBB) {
        FalseBB = Succ;
        break;
      }
    }

    unsigned Opc = EZH::GOTO; // fallback
    switch (CC) {
    case 1:
      Opc = EZH::GOTO_ze;
      break;
    case 2:
      Opc = EZH::GOTO_nz;
      break;
    case 3:
      Opc = EZH::GOTO_po;
      break;
    case 4:
      Opc = EZH::GOTO_ne;
      break;
    case 5:
      Opc = EZH::GOTO_az;
      break;
    case 6:
      Opc = EZH::GOTO_zb;
      break;
    case 7:
      Opc = EZH::GOTO_ca;
      break;
    case 8:
      Opc = EZH::GOTO_nc;
      break;
    case 9:
      Opc = EZH::GOTO_cz;
      break;
    }
    BuildMI(*BB, MI, DL, TII->get(Opc)).addMBB(TrueBB);
    auto NextIT = next_nodbg(MachineBasicBlock::iterator(MI), BB->end());
    if (FalseBB && (NextIT == BB->end() || !NextIT->isBranch()))
      BuildMI(*BB, MI, DL, TII->get(EZH::GOTO)).addMBB(FalseBB);
    MI.eraseFromParent();
    return BB;
  }
  case EZH::PseudoSELECT_CC: {
    unsigned DestReg = MI.getOperand(0).getReg();
    unsigned TrueReg = MI.getOperand(1).getReg();
    unsigned FalseReg = MI.getOperand(2).getReg();
    ISD::CondCode CC = static_cast<ISD::CondCode>(MI.getOperand(3).getImm());

    const BasicBlock *LLVM_BB = BB->getBasicBlock();
    MachineFunction::iterator It = ++BB->getIterator();
    MachineFunction *F = BB->getParent();

    MachineBasicBlock *TrueBB = F->CreateMachineBasicBlock(LLVM_BB);
    MachineBasicBlock *DoneBB = F->CreateMachineBasicBlock(LLVM_BB);
    F->insert(It, TrueBB);
    F->insert(It, DoneBB);

    DoneBB->splice(DoneBB->end(), BB,
                   std::next(MachineBasicBlock::iterator(MI)), BB->end());
    DoneBB->transferSuccessorsAndUpdatePHIs(BB);

    BB->addSuccessor(TrueBB);
    BB->addSuccessor(DoneBB);
    TrueBB->addSuccessor(DoneBB);

    unsigned Opc = EZH::GOTO_ze;
    switch (CC) {
    case ISD::SETEQ:
      Opc = EZH::GOTO_ze;
      break;
    case ISD::SETNE:
      Opc = EZH::GOTO_nz;
      break;
    case ISD::SETLT:
      Opc = EZH::GOTO_ne;
      break;
    case ISD::SETLE:
      Opc = EZH::GOTO_zb;
      break;
    case ISD::SETGT:
      Opc = EZH::GOTO_az;
      break;
    case ISD::SETGE:
      Opc = EZH::GOTO_po;
      break;
    case ISD::SETULT:
      Opc = EZH::GOTO_ca;
      break;
    case ISD::SETUGE:
      Opc = EZH::GOTO_nc;
      break;
    case ISD::SETUGT:
      std::swap(TrueReg, FalseReg);
      Opc = EZH::GOTO_ca;
      break;
    case ISD::SETULE:
      std::swap(TrueReg, FalseReg);
      Opc = EZH::GOTO_nc;
      break;
    default:
      Opc = EZH::GOTO_ze;
      break;
    }

    BuildMI(*BB, MI, DL, TII->get(Opc)).addMBB(TrueBB);
    BuildMI(*BB, MI, DL, TII->get(EZH::GOTO)).addMBB(DoneBB);

    BuildMI(*DoneBB, DoneBB->begin(), DL, TII->get(EZH::PHI), DestReg)
        .addReg(TrueReg)
        .addMBB(TrueBB)
        .addReg(FalseReg)
        .addMBB(BB);

    MI.eraseFromParent();
    return DoneBB;
  }
  case EZH::EH_SjLj_SetJmp:
    return emitEHSjLjSetJmp(MI, BB);
  case EZH::EH_SjLj_LongJmp:
    return emitEHSjLjLongJmp(MI, BB);
  case EZH::EH_SjLj_Setup_Dispatch:
    return emitSjLjDispatchBlock(MI, BB);
  }

  llvm_unreachable("Unexpected instr type to insert");
}

Register EZHTargetLowering::getExceptionPointerRegister(
    const Constant *PersonalityFn) const {
  return Register();
}

Register EZHTargetLowering::getExceptionSelectorRegister(
    const Constant *PersonalityFn) const {
  return Register();
}

MachineBasicBlock *
EZHTargetLowering::emitEHSjLjSetJmp(MachineInstr &MI,
                                    MachineBasicBlock *MBB) const {
  DebugLoc DL = MI.getDebugLoc();
  MachineFunction *MF = MBB->getParent();
  const EZHSubtarget &STI = MF->getSubtarget<EZHSubtarget>();
  const TargetInstrInfo *TII = STI.getInstrInfo();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  Register DstReg = MI.getOperand(0).getReg();
  Register BufReg = MI.getOperand(1).getReg();

  // 1. EZH SjLj Block Splitting Flow:
  // Split the basic block to materialize SjLj return targets:
  //         [ThisMBB (BB)]
  //               |
  //       -------- --------
  //      |                 |
  //  [MainMBB]        [RestoreMBB] (Target PC label address taken)
  //    v = 0             v = 1
  //      |                 |
  //       -------- --------
  //               |
  //           [SinkMBB]
  //       v = phi(v_main, v_restore)

  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineFunction::iterator It = ++MBB->getIterator();

  MachineBasicBlock *ThisMBB = MBB;
  MachineBasicBlock *MainMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *RestoreMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *SinkMBB = MF->CreateMachineBasicBlock(LLVM_BB);

  MF->insert(It, MainMBB);
  MF->insert(It, RestoreMBB);
  MF->insert(It, SinkMBB);

  RestoreMBB->setMachineBlockAddressTaken();

  // Split the block at MI
  SinkMBB->splice(SinkMBB->begin(), MBB,
                  std::next(MachineBasicBlock::iterator(MI)), MBB->end());
  SinkMBB->transferSuccessorsAndUpdatePHIs(MBB);

  // Succ edges
  ThisMBB->addSuccessor(MainMBB);
  ThisMBB->addSuccessor(RestoreMBB);

  // 2. Save the PC target address (taking address of RestoreMBB) at Offset 4 in
  // BufReg
  Register TargetPCReg = MRI.createVirtualRegister(&EZH::GPRRegClass);
  // Programmatically allocate RestoreMBB as a custom EZHConstantPoolValue
  Type *PtrTy = Type::getInt32Ty(MF->getFunction().getContext());
  auto *CPV = new EZHConstantPoolValue(RestoreMBB, PtrTy);
  unsigned CPI = MF->getConstantPool()->getConstantPoolIndex(CPV, Align(4));

  BuildMI(*ThisMBB, MI, DL, TII->get(EZH::LOAD_CONSTANT), TargetPCReg)
      .addConstantPoolIndex(CPI);

  BuildMI(*ThisMBB, MI, DL, TII->get(EZH::STR))
      .addReg(TargetPCReg)
      .addReg(BufReg)
      .addImm(4);

  // 3. Save Frame Pointer R7 at Offset 0 in BufReg
  BuildMI(*ThisMBB, MI, DL, TII->get(EZH::STR))
      .addReg(EZH::R7)
      .addReg(BufReg)
      .addImm(0);

  // 4. Save Stack Pointer SP at Offset 8 in BufReg
  BuildMI(*ThisMBB, MI, DL, TII->get(EZH::STR))
      .addReg(EZH::SP)
      .addReg(BufReg)
      .addImm(8);

  // 5. Save Callee-Saved Register R6 (Base Pointer) at Offset 12 in BufReg
  BuildMI(*ThisMBB, MI, DL, TII->get(EZH::STR))
      .addReg(EZH::R6)
      .addReg(BufReg)
      .addImm(12);

  // Safety Goto from ThisMBB directly into MainMBB
  BuildMI(*ThisMBB, MI, DL, TII->get(EZH::GOTO)).addMBB(MainMBB);

  // 5. MainMBB returns 0 on initialization
  Register MainValReg = MRI.createVirtualRegister(&EZH::GPRRegClass);
  BuildMI(MainMBB, DL, TII->get(EZH::MOVri__), MainValReg).addImm(0);
  BuildMI(MainMBB, DL, TII->get(EZH::GOTO)).addMBB(SinkMBB);
  MainMBB->addSuccessor(SinkMBB);

  // 6. RestoreMBB returns 1 on builtin longjmp return
  Register RestoreValReg = MRI.createVirtualRegister(&EZH::GPRRegClass);
  BuildMI(RestoreMBB, DL, TII->get(EZH::MOVri__), RestoreValReg).addImm(1);
  BuildMI(RestoreMBB, DL, TII->get(EZH::GOTO)).addMBB(SinkMBB);
  RestoreMBB->addSuccessor(SinkMBB);

  // 7. SinkMBB merges return statuses via PHI node
  BuildMI(*SinkMBB, SinkMBB->begin(), DL, TII->get(EZH::PHI), DstReg)
      .addReg(MainValReg)
      .addMBB(MainMBB)
      .addReg(RestoreValReg)
      .addMBB(RestoreMBB);

  MI.eraseFromParent();
  return SinkMBB;
}

MachineBasicBlock *
EZHTargetLowering::emitEHSjLjLongJmp(MachineInstr &MI,
                                     MachineBasicBlock *MBB) const {
  DebugLoc DL = MI.getDebugLoc();
  MachineFunction *MF = MBB->getParent();
  const EZHSubtarget &STI = MF->getSubtarget<EZHSubtarget>();
  const TargetInstrInfo *TII = STI.getInstrInfo();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  Register BufReg = MI.getOperand(0).getReg();

  Register TargetPCReg = MRI.createVirtualRegister(&EZH::GPRRegClass);

  // EZH SjLj Restore:
  // 1. Load Target PC from Offset 4 (jbuf[1])
  BuildMI(*MBB, MI, DL, TII->get(EZH::LDR), TargetPCReg)
      .addReg(BufReg)
      .addImm(4);

  // 2. Load Frame Pointer R7 from Offset 0 (jbuf[0])
  BuildMI(*MBB, MI, DL, TII->get(EZH::LDR), EZH::R7).addReg(BufReg).addImm(0);

  // 3. Load Stack Pointer SP from Offset 8
  BuildMI(*MBB, MI, DL, TII->get(EZH::LDR), EZH::SP).addReg(BufReg).addImm(8);

  // 4. Load Callee-Saved Register R6 (Base Pointer) from Offset 12
  BuildMI(*MBB, MI, DL, TII->get(EZH::LDR), EZH::R6).addReg(BufReg).addImm(12);

  // 5. Jump to restored PC target natively!
  BuildMI(*MBB, MI, DL, TII->get(EZH::GOTO_REG)).addReg(TargetPCReg);

  MI.eraseFromParent();
  return MBB;
}

SDValue EZHTargetLowering::LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const {
  const EZHSubtarget &ASTI =
      DAG.getMachineFunction().getSubtarget<EZHSubtarget>();
  const EZHRegisterInfo &RI = *ASTI.getRegisterInfo();
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setFrameAddressIsTaken(true);

  EVT VT = Op.getValueType();
  SDLoc dl(Op);
  unsigned Depth = Op.getConstantOperandVal(0);
  Register FrameReg = RI.getFrameRegister(MF);
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), dl, FrameReg, VT);
  while (Depth--)
    FrameAddr = DAG.getLoad(VT, dl, DAG.getEntryNode(), FrameAddr,
                            MachinePointerInfo());
  return FrameAddr;
}

SDValue EZHTargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc dl(Op);

  // Inputs: Chain (0), Size (1), Alignment (2)
  SDValue Chain = Op.getOperand(0);
  SDValue Size = Op.getOperand(1);
  Align Alignment = cast<ConstantSDNode>(Op.getOperand(2))->getAlignValue();

  // 1. Get current stack pointer SP
  SDValue SP = DAG.getCopyFromReg(Chain, dl, EZH::SP, MVT::i32);
  Chain = SP.getValue(1);

  // 2. Subtract size from SP to allocate space down the stack
  SDValue NewSP = DAG.getNode(ISD::SUB, dl, MVT::i32, SP, Size);

  // 3. Enforce strict EZH stack alignment boundary
  // Align dynamic stack allocations to at least 16 bytes to satisfy max
  // structural alignments
  uint64_t AlignVal = std::max<uint64_t>(16, Alignment.value());
  NewSP = DAG.getNode(ISD::AND, dl, MVT::i32, NewSP,
                      DAG.getSignedConstant(-AlignVal, dl, MVT::i32));

  // 4. Copy new aligned pointer back to EZH SP
  Chain = DAG.getCopyToReg(Chain, dl, EZH::SP, NewSP);

  // Return the new SP and updated Chain
  SDValue Ops[2] = {NewSP, Chain};
  return DAG.getMergeValues(Ops, dl);
}

SDValue EZHTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc dl(Op);
  unsigned IntNo = Op.getConstantOperandVal(0);
  switch (IntNo) {
  default:
    return SDValue(); // Don't custom lower most intrinsics.
  case Intrinsic::eh_sjlj_lsda: {
    MachineFunction &MF = DAG.getMachineFunction();
    MVT VT = Op.getSimpleValueType();

    // 1. Generate symbol name "GCC_except_tableXX" and persist it in the
    // MCContext string pool
    MCSymbol *LSDASym = MF.getContext().getOrCreateSymbol(
        Twine("GCC_except_table") + Twine(MF.getFunctionNumber()));
    const char *Name = LSDASym->getName().data();

    // 2. Wrap the symbol name in EZHConstantPoolValue to force loading its
    // absolute address!
    auto *CPV =
        new EZHConstantPoolValue(Name, Type::getInt32Ty(*DAG.getContext()));
    SDValue CPAddr = DAG.getTargetConstantPool(CPV, VT, Align(4));

    // 3. Load the address of the exception table from the constant pool
    SDValue Ops[] = {CPAddr, DAG.getEntryNode()};
    return SDValue(
        DAG.getMachineNode(EZH::LOAD_CONSTANT, dl, MVT::i32, MVT::Other, Ops),
        0);
  }
  }
}

MachineBasicBlock *
EZHTargetLowering::emitSjLjDispatchBlock(MachineInstr &MI,
                                         MachineBasicBlock *BB) const {
  DebugLoc DL = MI.getDebugLoc();
  MachineFunction *MF = BB->getParent();
  MachineFrameInfo &MFI = MF->getFrameInfo();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const EZHSubtarget &STI = MF->getSubtarget<EZHSubtarget>();
  const TargetInstrInfo *TII = STI.getInstrInfo();
  int FI = MFI.getFunctionContextIndex();

  // 1. Find all landing pads and their associated call site numbers.
  DenseMap<unsigned, SmallVector<MachineBasicBlock *, 2>> CallSiteNumToLPad;
  unsigned MaxCSNum = 0;
  for (auto &MBB : *MF) {
    if (!MBB.isEHPad())
      continue;

    auto FirstI = MBB.getFirstNonDebugInstr();
    if (FirstI == MBB.end())
      continue;
    assert(FirstI->isEHLabel() && "expected EH_LABEL");
    MCSymbol *Sym = FirstI->getOperand(0).getMCSymbol();

    if (!MF->hasCallSiteLandingPad(Sym))
      continue;

    for (unsigned CSI : MF->getCallSiteLandingPad(Sym)) {
      CallSiteNumToLPad[CSI].push_back(&MBB);
      MaxCSNum = std::max(MaxCSNum, CSI);
    }
  }

  assert(MaxCSNum > 0 &&
         "No landing pad destinations for the dispatch jump table!");

  // 2. Create the dispatch, continuation, and trap basic blocks.
  MachineBasicBlock *DispatchBB = MF->CreateMachineBasicBlock();
  DispatchBB->setIsEHPad(true);
  DispatchBB->setMachineBlockAddressTaken();

  MachineBasicBlock *DispContBB = MF->CreateMachineBasicBlock();

  MachineBasicBlock *TrapBB = MF->CreateMachineBasicBlock();

  // Insert them at the end of the function in layout order.
  MF->insert(MF->end(), TrapBB);
  MF->insert(MF->end(), DispatchBB);
  MF->insert(MF->end(), DispContBB);

  // 2.5 Update CFG: Replace landing pad successors in invoke blocks with
  // DispatchBB
  SmallPtrSet<MachineBasicBlock *, 16> InvokeBBs;
  for (unsigned CSI = 1; CSI <= MaxCSNum; ++CSI) {
    for (auto *LPad : CallSiteNumToLPad[CSI]) {
      for (auto *Pred : LPad->predecessors()) {
        InvokeBBs.insert(Pred);
      }
    }
  }

  for (MachineBasicBlock *InvokeBB : InvokeBBs) {
    SmallVector<MachineBasicBlock *, 4> Successors(InvokeBB->successors());
    while (!Successors.empty()) {
      MachineBasicBlock *SMBB = Successors.pop_back_val();
      if (SMBB->isEHPad()) {
        InvokeBB->removeSuccessor(SMBB);
      }
    }
    InvokeBB->addSuccessor(DispatchBB, BranchProbability::getZero());
    InvokeBB->normalizeSuccProbs();
  }

  // TrapBB just loops forever.
  BuildMI(TrapBB, DL, TII->get(EZH::GOTO)).addMBB(TrapBB);
  TrapBB->addSuccessor(TrapBB);

  // 3. Load the address of DispatchBB and store it to jbuf[0] (offset 32 of
  // context) in the entry block.
  Register DispatchPCReg = MRI.createVirtualRegister(&EZH::GPRRegClass);
  Type *PtrTy = Type::getInt32Ty(MF->getFunction().getContext());
  auto *CPV = new EZHConstantPoolValue(DispatchBB, PtrTy);
  unsigned CPI = MF->getConstantPool()->getConstantPoolIndex(CPV, Align(4));

  BuildMI(*BB, MI, DL, TII->get(EZH::LOAD_CONSTANT), DispatchPCReg)
      .addConstantPoolIndex(CPI);

  BuildMI(*BB, MI, DL, TII->get(EZH::STR))
      .addReg(DispatchPCReg)
      .addFrameIndex(FI)
      .addImm(36); // Offset 36 is jbuf[1]

  // Save Callee-Saved Register R6 (Base Pointer) at Offset 44 in function
  // context (jbuf[3])
  BuildMI(*BB, MI, DL, TII->get(EZH::STR))
      .addReg(EZH::R6)
      .addFrameIndex(FI)
      .addImm(44); // Offset 44 is jbuf[3]

  // 4. In DispatchBB, load the call_site value.
  Register CSReg = MRI.createVirtualRegister(&EZH::GPRRegClass);
  BuildMI(DispatchBB, DL, TII->get(EZH::LDR), CSReg)
      .addFrameIndex(FI)
      .addImm(4); // offset of call_site in context

  // 5. Build LPadList with placeholders for holes
  std::vector<MachineBasicBlock *> LPadList;
  LPadList.reserve(MaxCSNum);
  for (unsigned I = 1; I <= MaxCSNum; ++I) {
    SmallVectorImpl<MachineBasicBlock *> &MBBList = CallSiteNumToLPad[I];
    if (!MBBList.empty())
      LPadList.push_back(MBBList[0]); // Take first one
    else
      LPadList.push_back(TrapBB); // Placeholder for inactive CSI
  }

  // 6. Create Jump Table
  MachineJumpTableInfo *MJTI =
      MF->getOrCreateJumpTableInfo(MachineJumpTableInfo::EK_BlockAddress);
  unsigned JTI = MJTI->createJumpTableIndex(LPadList);

  // 7. Bounds check in DispatchBB:
  // IndexReg = CSReg
  // If IndexReg >= MaxCSNum (unsigned) -> TrapBB (using GOTO_nc)
  // Else -> DispContBB (fallthrough)
  Register IndexReg = CSReg;

  Register TempReg = MRI.createVirtualRegister(&EZH::GPRRegClass);
  BuildMI(DispatchBB, DL, TII->get(EZH::SUBri_s_), TempReg)
      .addReg(IndexReg)
      .addImm(MaxCSNum);

  BuildMI(DispatchBB, DL, TII->get(EZH::GOTO_nc)).addMBB(TrapBB);
  DispatchBB->addSuccessor(TrapBB);
  DispatchBB->addSuccessor(DispContBB);

  // 8. In DispContBB: Load JT address and do indirect branch
  auto *JT_CPV = new EZHConstantPoolValue(
      JTI, Type::getInt32Ty(MF->getFunction().getContext()));
  unsigned JT_CPI =
      MF->getConstantPool()->getConstantPoolIndex(JT_CPV, Align(4));
  Register JTReg = MRI.createVirtualRegister(&EZH::GPRRegClass);
  BuildMI(DispContBB, DL, TII->get(EZH::LOAD_CONSTANT), JTReg)
      .addConstantPoolIndex(JT_CPI);

  BuildMI(DispContBB, DL, TII->get(EZH::PseudoBR_JT))
      .addReg(JTReg)
      .addReg(IndexReg)
      .addJumpTableIndex(JTI);

  // Update successors for DispContBB (unique only)
  SmallPtrSet<MachineBasicBlock *, 8> UniqueSuccs;
  for (auto *LPad : LPadList) {
    if (UniqueSuccs.insert(LPad).second) {
      DispContBB->addSuccessor(LPad);
    }
  }

  // Mark all former landing pads as non-landing pads. The dispatch is the only
  // landing pad now.
  for (unsigned CSI = 1; CSI <= MaxCSNum; ++CSI)
    for (auto *LPad : CallSiteNumToLPad[CSI])
      LPad->setIsEHPad(false);

  MI.eraseFromParent();
  return BB;
}
