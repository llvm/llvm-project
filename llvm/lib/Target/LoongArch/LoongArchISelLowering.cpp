//=- LoongArchISelLowering.cpp - LoongArch DAG Lowering Implementation  ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that LoongArch uses to lower LLVM code into
// a selection DAG.
//
//===----------------------------------------------------------------------===//

#include "LoongArchISelLowering.h"
#include "LoongArch.h"
#include "LoongArchMachineFunctionInfo.h"
#include "LoongArchRegisterInfo.h"
#include "LoongArchSubtarget.h"
#include "LoongArchTargetMachine.h"
#include "MCTargetDesc/LoongArchMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"

using namespace llvm;

#define DEBUG_TYPE "loongarch-isel-lowering"

static cl::opt<bool> ZeroDivCheck(
    "loongarch-check-zero-division", cl::Hidden,
    cl::desc("Trap on integer division by zero."),
    cl::init(false));

LoongArchTargetLowering::LoongArchTargetLowering(const TargetMachine &TM,
                                                 const LoongArchSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {

  MVT GRLenVT = Subtarget.getGRLenVT();
  // Set up the register classes.
  addRegisterClass(GRLenVT, &LoongArch::GPRRegClass);
  if (Subtarget.hasBasicF())
    addRegisterClass(MVT::f32, &LoongArch::FPR32RegClass);
  if (Subtarget.hasBasicD())
    addRegisterClass(MVT::f64, &LoongArch::FPR64RegClass);

  setLoadExtAction({ISD::EXTLOAD, ISD::SEXTLOAD, ISD::ZEXTLOAD}, GRLenVT,
                   MVT::i1, Promote);

  // TODO: add necessary setOperationAction calls later.
  setOperationAction(ISD::SHL_PARTS, GRLenVT, Custom);
  setOperationAction(ISD::SRA_PARTS, GRLenVT, Custom);
  setOperationAction(ISD::SRL_PARTS, GRLenVT, Custom);
  setOperationAction(ISD::FP_TO_SINT, GRLenVT, Custom);

  setOperationAction({ISD::GlobalAddress, ISD::ConstantPool}, GRLenVT, Custom);

  if (Subtarget.is64Bit()) {
    setOperationAction(ISD::SHL, MVT::i32, Custom);
    setOperationAction(ISD::SRA, MVT::i32, Custom);
    setOperationAction(ISD::SRL, MVT::i32, Custom);
    setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
    setOperationAction(ISD::BITCAST, MVT::i32, Custom);
    if (Subtarget.hasBasicF() && !Subtarget.hasBasicD())
      setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
  }

  static const ISD::CondCode FPCCToExpand[] = {ISD::SETOGT, ISD::SETOGE,
                                               ISD::SETUGT, ISD::SETUGE};

  if (Subtarget.hasBasicF()) {
    setCondCodeAction(FPCCToExpand, MVT::f32, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
  }
  if (Subtarget.hasBasicD()) {
    setCondCodeAction(FPCCToExpand, MVT::f64, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
  }

  setOperationAction(ISD::BR_CC, GRLenVT, Expand);
  setOperationAction(ISD::SELECT_CC, GRLenVT, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction({ISD::SMUL_LOHI, ISD::UMUL_LOHI}, GRLenVT, Expand);
  if (!Subtarget.is64Bit())
    setLibcallName(RTLIB::MUL_I128, nullptr);

  setOperationAction(ISD::FP_TO_UINT, GRLenVT, Custom);
  setOperationAction(ISD::UINT_TO_FP, GRLenVT, Custom);

  // Compute derived properties from the register classes.
  computeRegisterProperties(STI.getRegisterInfo());

  setStackPointerRegisterToSaveRestore(LoongArch::R3);

  setBooleanContents(ZeroOrOneBooleanContent);

  setMaxAtomicSizeInBitsSupported(Subtarget.getGRLen());

  // Function alignments.
  const Align FunctionAlignment(4);
  setMinFunctionAlignment(FunctionAlignment);

  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::SRL);
}

SDValue LoongArchTargetLowering::LowerOperation(SDValue Op,
                                                SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    report_fatal_error("unimplemented operand");
  case ISD::GlobalAddress:
    return lowerGlobalAddress(Op, DAG);
  case ISD::SHL_PARTS:
    return lowerShiftLeftParts(Op, DAG);
  case ISD::SRA_PARTS:
    return lowerShiftRightParts(Op, DAG, true);
  case ISD::SRL_PARTS:
    return lowerShiftRightParts(Op, DAG, false);
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
    // This can be called for an i32 shift amount that needs to be promoted.
    assert(Op.getOperand(1).getValueType() == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    return SDValue();
  case ISD::ConstantPool:
    return lowerConstantPool(Op, DAG);
  case ISD::FP_TO_SINT:
    return lowerFP_TO_SINT(Op, DAG);
  case ISD::BITCAST:
    return lowerBITCAST(Op, DAG);
  case ISD::FP_TO_UINT:
    return SDValue();
  case ISD::UINT_TO_FP:
    return lowerUINT_TO_FP(Op, DAG);
  }
}

SDValue LoongArchTargetLowering::lowerUINT_TO_FP(SDValue Op,
                                                 SelectionDAG &DAG) const {

  SDLoc DL(Op);
  auto &TLI = DAG.getTargetLoweringInfo();
  SDValue Tmp1, Tmp2;
  SDValue Op1 = Op.getOperand(0);
  if (Op1->getOpcode() == ISD::AssertZext ||
      Op1->getOpcode() == ISD::AssertSext)
    return Op;
  SDValue Trunc = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Op.getOperand(0));
  SDValue Res = DAG.getNode(ISD::UINT_TO_FP, DL, MVT::f64, Trunc);
  SDNode *N = Res.getNode();
  TLI.expandUINT_TO_FP(N, Tmp1, Tmp2, DAG);
  return Tmp1;
}

SDValue LoongArchTargetLowering::lowerBITCAST(SDValue Op,
                                              SelectionDAG &DAG) const {

  SDLoc DL(Op);
  SDValue Op0 = Op.getOperand(0);

  if (Op.getValueType() == MVT::f32 && Op0.getValueType() == MVT::i32 &&
      Subtarget.is64Bit() && Subtarget.hasBasicF()) {
    SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op0);
    return DAG.getNode(LoongArchISD::MOVGR2FR_W_LA64, DL, MVT::f32, NewOp0);
  }
  return Op;
}

SDValue LoongArchTargetLowering::lowerFP_TO_SINT(SDValue Op,
                                                 SelectionDAG &DAG) const {

  SDLoc DL(Op);

  if (Op.getValueSizeInBits() > 32 && Subtarget.hasBasicF() &&
      !Subtarget.hasBasicD()) {
    SDValue Dst =
        DAG.getNode(LoongArchISD::FTINT, DL, MVT::f32, Op.getOperand(0));
    return DAG.getNode(LoongArchISD::MOVFR2GR_S_LA64, DL, MVT::i64, Dst);
  }

  EVT FPTy = EVT::getFloatingPointVT(Op.getValueSizeInBits());
  SDValue Trunc = DAG.getNode(LoongArchISD::FTINT, DL, FPTy, Op.getOperand(0));
  return DAG.getNode(ISD::BITCAST, DL, Op.getValueType(), Trunc);
}

SDValue LoongArchTargetLowering::lowerConstantPool(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  ConstantPoolSDNode *N = cast<ConstantPoolSDNode>(Op);

  // FIXME: Only support PC-relative addressing to access the symbol.
  // Target flags will be added later.
  if (!isPositionIndependent()) {
    SDValue ConstantN = DAG.getTargetConstantPool(
        N->getConstVal(), Ty, N->getAlign(), N->getOffset());
    SDValue AddrHi(DAG.getMachineNode(LoongArch::PCALAU12I, DL, Ty, ConstantN),
                   0);
    SDValue Addr(DAG.getMachineNode(Subtarget.is64Bit() ? LoongArch::ADDI_D
                                                        : LoongArch::ADDI_W,
                                    DL, Ty, AddrHi, ConstantN),
                 0);
    return Addr;
  }
  report_fatal_error("Unable to lower ConstantPool");
}

SDValue LoongArchTargetLowering::lowerGlobalAddress(SDValue Op,
                                                    SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  unsigned ADDIOp = Subtarget.is64Bit() ? LoongArch::ADDI_D : LoongArch::ADDI_W;

  // TODO: Support dso_preemptable and target flags.
  if (GV->isDSOLocal()) {
    SDValue GA = DAG.getTargetGlobalAddress(GV, DL, Ty);
    SDValue AddrHi(DAG.getMachineNode(LoongArch::PCALAU12I, DL, Ty, GA), 0);
    SDValue Addr(DAG.getMachineNode(ADDIOp, DL, Ty, AddrHi, GA), 0);
    return Addr;
  }
  report_fatal_error("Unable to lowerGlobalAddress");
}

SDValue LoongArchTargetLowering::lowerShiftLeftParts(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  EVT VT = Lo.getValueType();

  // if Shamt-GRLen < 0: // Shamt < GRLen
  //   Lo = Lo << Shamt
  //   Hi = (Hi << Shamt) | ((Lo >>u 1) >>u (GRLen-1 ^ Shamt))
  // else:
  //   Lo = 0
  //   Hi = Lo << (Shamt-GRLen)

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One = DAG.getConstant(1, DL, VT);
  SDValue MinusGRLen = DAG.getConstant(-(int)Subtarget.getGRLen(), DL, VT);
  SDValue GRLenMinus1 = DAG.getConstant(Subtarget.getGRLen() - 1, DL, VT);
  SDValue ShamtMinusGRLen = DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusGRLen);
  SDValue GRLenMinus1Shamt = DAG.getNode(ISD::XOR, DL, VT, Shamt, GRLenMinus1);

  SDValue LoTrue = DAG.getNode(ISD::SHL, DL, VT, Lo, Shamt);
  SDValue ShiftRight1Lo = DAG.getNode(ISD::SRL, DL, VT, Lo, One);
  SDValue ShiftRightLo =
      DAG.getNode(ISD::SRL, DL, VT, ShiftRight1Lo, GRLenMinus1Shamt);
  SDValue ShiftLeftHi = DAG.getNode(ISD::SHL, DL, VT, Hi, Shamt);
  SDValue HiTrue = DAG.getNode(ISD::OR, DL, VT, ShiftLeftHi, ShiftRightLo);
  SDValue HiFalse = DAG.getNode(ISD::SHL, DL, VT, Lo, ShamtMinusGRLen);

  SDValue CC = DAG.getSetCC(DL, VT, ShamtMinusGRLen, Zero, ISD::SETLT);

  Lo = DAG.getNode(ISD::SELECT, DL, VT, CC, LoTrue, Zero);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, CC, HiTrue, HiFalse);

  SDValue Parts[2] = {Lo, Hi};
  return DAG.getMergeValues(Parts, DL);
}

SDValue LoongArchTargetLowering::lowerShiftRightParts(SDValue Op,
                                                      SelectionDAG &DAG,
                                                      bool IsSRA) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  EVT VT = Lo.getValueType();

  // SRA expansion:
  //   if Shamt-GRLen < 0: // Shamt < GRLen
  //     Lo = (Lo >>u Shamt) | ((Hi << 1) << (ShAmt ^ GRLen-1))
  //     Hi = Hi >>s Shamt
  //   else:
  //     Lo = Hi >>s (Shamt-GRLen);
  //     Hi = Hi >>s (GRLen-1)
  //
  // SRL expansion:
  //   if Shamt-GRLen < 0: // Shamt < GRLen
  //     Lo = (Lo >>u Shamt) | ((Hi << 1) << (ShAmt ^ GRLen-1))
  //     Hi = Hi >>u Shamt
  //   else:
  //     Lo = Hi >>u (Shamt-GRLen);
  //     Hi = 0;

  unsigned ShiftRightOp = IsSRA ? ISD::SRA : ISD::SRL;

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One = DAG.getConstant(1, DL, VT);
  SDValue MinusGRLen = DAG.getConstant(-(int)Subtarget.getGRLen(), DL, VT);
  SDValue GRLenMinus1 = DAG.getConstant(Subtarget.getGRLen() - 1, DL, VT);
  SDValue ShamtMinusGRLen = DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusGRLen);
  SDValue GRLenMinus1Shamt = DAG.getNode(ISD::XOR, DL, VT, Shamt, GRLenMinus1);

  SDValue ShiftRightLo = DAG.getNode(ISD::SRL, DL, VT, Lo, Shamt);
  SDValue ShiftLeftHi1 = DAG.getNode(ISD::SHL, DL, VT, Hi, One);
  SDValue ShiftLeftHi =
      DAG.getNode(ISD::SHL, DL, VT, ShiftLeftHi1, GRLenMinus1Shamt);
  SDValue LoTrue = DAG.getNode(ISD::OR, DL, VT, ShiftRightLo, ShiftLeftHi);
  SDValue HiTrue = DAG.getNode(ShiftRightOp, DL, VT, Hi, Shamt);
  SDValue LoFalse = DAG.getNode(ShiftRightOp, DL, VT, Hi, ShamtMinusGRLen);
  SDValue HiFalse =
      IsSRA ? DAG.getNode(ISD::SRA, DL, VT, Hi, GRLenMinus1) : Zero;

  SDValue CC = DAG.getSetCC(DL, VT, ShamtMinusGRLen, Zero, ISD::SETLT);

  Lo = DAG.getNode(ISD::SELECT, DL, VT, CC, LoTrue, LoFalse);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, CC, HiTrue, HiFalse);

  SDValue Parts[2] = {Lo, Hi};
  return DAG.getMergeValues(Parts, DL);
}

// Returns the opcode of the target-specific SDNode that implements the 32-bit
// form of the given Opcode.
static LoongArchISD::NodeType getLoongArchWOpcode(unsigned Opcode) {
  switch (Opcode) {
  default:
    llvm_unreachable("Unexpected opcode");
  case ISD::SHL:
    return LoongArchISD::SLL_W;
  case ISD::SRA:
    return LoongArchISD::SRA_W;
  case ISD::SRL:
    return LoongArchISD::SRL_W;
  }
}

// Converts the given i8/i16/i32 operation to a target-specific SelectionDAG
// node. Because i8/i16/i32 isn't a legal type for LA64, these operations would
// otherwise be promoted to i64, making it difficult to select the
// SLL_W/.../*W later one because the fact the operation was originally of
// type i8/i16/i32 is lost.
static SDValue customLegalizeToWOp(SDNode *N, SelectionDAG &DAG,
                                   unsigned ExtOpc = ISD::ANY_EXTEND) {
  SDLoc DL(N);
  LoongArchISD::NodeType WOpcode = getLoongArchWOpcode(N->getOpcode());
  SDValue NewOp0 = DAG.getNode(ExtOpc, DL, MVT::i64, N->getOperand(0));
  SDValue NewOp1 = DAG.getNode(ExtOpc, DL, MVT::i64, N->getOperand(1));
  SDValue NewRes = DAG.getNode(WOpcode, DL, MVT::i64, NewOp0, NewOp1);
  // ReplaceNodeResults requires we maintain the same type for the return value.
  return DAG.getNode(ISD::TRUNCATE, DL, N->getValueType(0), NewRes);
}

void LoongArchTargetLowering::ReplaceNodeResults(
    SDNode *N, SmallVectorImpl<SDValue> &Results, SelectionDAG &DAG) const {
  SDLoc DL(N);
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Don't know how to legalize this operation");
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    if (N->getOperand(1).getOpcode() != ISD::Constant) {
      Results.push_back(customLegalizeToWOp(N, DAG));
      break;
    }
    break;
  case ISD::FP_TO_SINT: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    SDValue Src = N->getOperand(0);
    EVT VT = EVT::getFloatingPointVT(N->getValueSizeInBits(0));
    SDValue Dst = DAG.getNode(LoongArchISD::FTINT, DL, VT, Src);
    Results.push_back(DAG.getNode(ISD::BITCAST, DL, N->getValueType(0), Dst));
    break;
  }
  case ISD::BITCAST: {
    EVT VT = N->getValueType(0);
    SDValue Src = N->getOperand(0);
    EVT SrcVT = Src.getValueType();
    if (VT == MVT::i32 && SrcVT == MVT::f32 && Subtarget.is64Bit() &&
        Subtarget.hasBasicF()) {
      SDValue Dst =
          DAG.getNode(LoongArchISD::MOVFR2GR_S_LA64, DL, MVT::i64, Src);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Dst));
    }
    break;
  }
  case ISD::FP_TO_UINT: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    auto &TLI = DAG.getTargetLoweringInfo();
    SDValue Tmp1, Tmp2;
    TLI.expandFP_TO_UINT(N, Tmp1, Tmp2, DAG);
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Tmp1));
    break;
  }
  }
}

static SDValue performANDCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const LoongArchSubtarget &Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  SDValue FirstOperand = N->getOperand(0);
  SDValue SecondOperand = N->getOperand(1);
  unsigned FirstOperandOpc = FirstOperand.getOpcode();
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);
  uint64_t lsb, msb;
  unsigned SMIdx, SMLen;
  ConstantSDNode *CN;
  SDValue NewOperand;
  MVT GRLenVT = Subtarget.getGRLenVT();

  // Op's second operand must be a shifted mask.
  if (!(CN = dyn_cast<ConstantSDNode>(SecondOperand)) ||
      !isShiftedMask_64(CN->getZExtValue(), SMIdx, SMLen))
    return SDValue();

  if (FirstOperandOpc == ISD::SRA || FirstOperandOpc == ISD::SRL) {
    // Pattern match BSTRPICK.
    //  $dst = and ((sra or srl) $src , lsb), (2**len - 1)
    //  => BSTRPICK $dst, $src, msb, lsb
    //  where msb = lsb + len - 1

    // The second operand of the shift must be an immediate.
    if (!(CN = dyn_cast<ConstantSDNode>(FirstOperand.getOperand(1))))
      return SDValue();

    lsb = CN->getZExtValue();

    // Return if the shifted mask does not start at bit 0 or the sum of its
    // length and lsb exceeds the word's size.
    if (SMIdx != 0 || lsb + SMLen > ValTy.getSizeInBits())
      return SDValue();

    NewOperand = FirstOperand.getOperand(0);
  } else {
    // Pattern match BSTRPICK.
    //  $dst = and $src, (2**len- 1) , if len > 12
    //  => BSTRPICK $dst, $src, msb, lsb
    //  where lsb = 0 and msb = len - 1

    // If the mask is <= 0xfff, andi can be used instead.
    if (CN->getZExtValue() <= 0xfff)
      return SDValue();

    // Return if the mask doesn't start at position 0.
    if (SMIdx)
      return SDValue();

    lsb = 0;
    NewOperand = FirstOperand;
  }
  msb = lsb + SMLen - 1;
  return DAG.getNode(LoongArchISD::BSTRPICK, DL, ValTy, NewOperand,
                     DAG.getConstant(msb, DL, GRLenVT),
                     DAG.getConstant(lsb, DL, GRLenVT));
}

static SDValue performSRLCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const LoongArchSubtarget &Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  // $dst = srl (and $src, Mask), Shamt
  // =>
  // BSTRPICK $dst, $src, MaskIdx+MaskLen-1, Shamt
  // when Mask is a shifted mask, and MaskIdx <= Shamt <= MaskIdx+MaskLen-1
  //

  SDValue FirstOperand = N->getOperand(0);
  ConstantSDNode *CN;
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);
  MVT GRLenVT = Subtarget.getGRLenVT();
  unsigned MaskIdx, MaskLen;
  uint64_t Shamt;

  // The first operand must be an AND and the second operand of the AND must be
  // a shifted mask.
  if (FirstOperand.getOpcode() != ISD::AND ||
      !(CN = dyn_cast<ConstantSDNode>(FirstOperand.getOperand(1))) ||
      !isShiftedMask_64(CN->getZExtValue(), MaskIdx, MaskLen))
    return SDValue();

  // The second operand (shift amount) must be an immediate.
  if (!(CN = dyn_cast<ConstantSDNode>(N->getOperand(1))))
    return SDValue();

  Shamt = CN->getZExtValue();
  if (MaskIdx <= Shamt && Shamt <= MaskIdx + MaskLen - 1)
    return DAG.getNode(LoongArchISD::BSTRPICK, DL, ValTy,
                       FirstOperand->getOperand(0),
                       DAG.getConstant(MaskIdx + MaskLen - 1, DL, GRLenVT),
                       DAG.getConstant(Shamt, DL, GRLenVT));

  return SDValue();
}

static SDValue performORCombine(SDNode *N, SelectionDAG &DAG,
                                TargetLowering::DAGCombinerInfo &DCI,
                                const LoongArchSubtarget &Subtarget) {
  MVT GRLenVT = Subtarget.getGRLenVT();
  EVT ValTy = N->getValueType(0);
  SDValue N0 = N->getOperand(0), N1 = N->getOperand(1);
  ConstantSDNode *CN0, *CN1;
  SDLoc DL(N);
  unsigned ValBits = ValTy.getSizeInBits();
  unsigned MaskIdx0, MaskLen0, MaskIdx1, MaskLen1;
  unsigned Shamt;
  bool SwapAndRetried = false;

  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  if (ValBits != 32 && ValBits != 64)
    return SDValue();

Retry:
  // 1st pattern to match BSTRINS:
  //  R = or (and X, mask0), (and (shl Y, lsb), mask1)
  //  where mask1 = (2**size - 1) << lsb, mask0 = ~mask1
  //  =>
  //  R = BSTRINS X, Y, msb, lsb (where msb = lsb + size - 1)
  if (N0.getOpcode() == ISD::AND &&
      (CN0 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) &&
      isShiftedMask_64(~CN0->getSExtValue(), MaskIdx0, MaskLen0) &&
      N1.getOpcode() == ISD::AND && N1.getOperand(0).getOpcode() == ISD::SHL &&
      (CN1 = dyn_cast<ConstantSDNode>(N1.getOperand(1))) &&
      isShiftedMask_64(CN1->getZExtValue(), MaskIdx1, MaskLen1) &&
      MaskIdx0 == MaskIdx1 && MaskLen0 == MaskLen1 &&
      (CN1 = dyn_cast<ConstantSDNode>(N1.getOperand(0).getOperand(1))) &&
      (Shamt = CN1->getZExtValue()) == MaskIdx0 &&
      (MaskIdx0 + MaskLen0 <= ValBits)) {
    LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 1\n");
    return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0.getOperand(0),
                       N1.getOperand(0).getOperand(0),
                       DAG.getConstant((MaskIdx0 + MaskLen0 - 1), DL, GRLenVT),
                       DAG.getConstant(MaskIdx0, DL, GRLenVT));
  }

  // 2nd pattern to match BSTRINS:
  //  R = or (and X, mask0), (shl (and Y, mask1), lsb)
  //  where mask1 = (2**size - 1), mask0 = ~(mask1 << lsb)
  //  =>
  //  R = BSTRINS X, Y, msb, lsb (where msb = lsb + size - 1)
  if (N0.getOpcode() == ISD::AND &&
      (CN0 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) &&
      isShiftedMask_64(~CN0->getSExtValue(), MaskIdx0, MaskLen0) &&
      N1.getOpcode() == ISD::SHL && N1.getOperand(0).getOpcode() == ISD::AND &&
      (CN1 = dyn_cast<ConstantSDNode>(N1.getOperand(1))) &&
      (Shamt = CN1->getZExtValue()) == MaskIdx0 &&
      (CN1 = dyn_cast<ConstantSDNode>(N1.getOperand(0).getOperand(1))) &&
      isShiftedMask_64(CN1->getZExtValue(), MaskIdx1, MaskLen1) &&
      MaskLen0 == MaskLen1 && MaskIdx1 == 0 &&
      (MaskIdx0 + MaskLen0 <= ValBits)) {
    LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 2\n");
    return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0.getOperand(0),
                       N1.getOperand(0).getOperand(0),
                       DAG.getConstant((MaskIdx0 + MaskLen0 - 1), DL, GRLenVT),
                       DAG.getConstant(MaskIdx0, DL, GRLenVT));
  }

  // 3rd pattern to match BSTRINS:
  //  R = or (and X, mask0), (and Y, mask1)
  //  where ~mask0 = (2**size - 1) << lsb, mask0 & mask1 = 0
  //  =>
  //  R = BSTRINS X, (shr (and Y, mask1), lsb), msb, lsb
  //  where msb = lsb + size - 1
  if (N0.getOpcode() == ISD::AND && N1.getOpcode() == ISD::AND &&
      (CN0 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) &&
      isShiftedMask_64(~CN0->getSExtValue(), MaskIdx0, MaskLen0) &&
      (MaskIdx0 + MaskLen0 <= 64) &&
      (CN1 = dyn_cast<ConstantSDNode>(N1->getOperand(1))) &&
      (CN1->getSExtValue() & CN0->getSExtValue()) == 0) {
    LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 3\n");
    return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0.getOperand(0),
                       DAG.getNode(ISD::SRL, DL, N1->getValueType(0), N1,
                                   DAG.getConstant(MaskIdx0, DL, GRLenVT)),
                       DAG.getConstant(ValBits == 32
                                           ? (MaskIdx0 + (MaskLen0 & 31) - 1)
                                           : (MaskIdx0 + MaskLen0 - 1),
                                       DL, GRLenVT),
                       DAG.getConstant(MaskIdx0, DL, GRLenVT));
  }

  // 4th pattern to match BSTRINS:
  //  R = or (and X, mask), (shl Y, shamt)
  //  where mask = (2**shamt - 1)
  //  =>
  //  R = BSTRINS X, Y, ValBits - 1, shamt
  //  where ValBits = 32 or 64
  if (N0.getOpcode() == ISD::AND && N1.getOpcode() == ISD::SHL &&
      (CN0 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) &&
      isShiftedMask_64(CN0->getZExtValue(), MaskIdx0, MaskLen0) &&
      MaskIdx0 == 0 && (CN1 = dyn_cast<ConstantSDNode>(N1.getOperand(1))) &&
      (Shamt = CN1->getZExtValue()) == MaskLen0 &&
      (MaskIdx0 + MaskLen0 <= ValBits)) {
    LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 4\n");
    return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0.getOperand(0),
                       N1.getOperand(0),
                       DAG.getConstant((ValBits - 1), DL, GRLenVT),
                       DAG.getConstant(Shamt, DL, GRLenVT));
  }

  // 5th pattern to match BSTRINS:
  //  R = or (and X, mask), const
  //  where ~mask = (2**size - 1) << lsb, mask & const = 0
  //  =>
  //  R = BSTRINS X, (const >> lsb), msb, lsb
  //  where msb = lsb + size - 1
  if (N0.getOpcode() == ISD::AND &&
      (CN0 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) &&
      isShiftedMask_64(~CN0->getSExtValue(), MaskIdx0, MaskLen0) &&
      (CN1 = dyn_cast<ConstantSDNode>(N1)) &&
      (CN1->getSExtValue() & CN0->getSExtValue()) == 0) {
    LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 5\n");
    return DAG.getNode(
        LoongArchISD::BSTRINS, DL, ValTy, N0.getOperand(0),
        DAG.getConstant(CN1->getSExtValue() >> MaskIdx0, DL, ValTy),
        DAG.getConstant((MaskIdx0 + MaskLen0 - 1), DL, GRLenVT),
        DAG.getConstant(MaskIdx0, DL, GRLenVT));
  }

  // 6th pattern.
  // a = b | ((c & mask) << shamt), where all positions in b to be overwritten
  // by the incoming bits are known to be zero.
  // =>
  // a = BSTRINS b, c, shamt + MaskLen - 1, shamt
  //
  // Note that the 1st pattern is a special situation of the 6th, i.e. the 6th
  // pattern is more common than the 1st. So we put the 1st before the 6th in
  // order to match as many nodes as possible.
  ConstantSDNode *CNMask, *CNShamt;
  unsigned MaskIdx, MaskLen;
  if (N1.getOpcode() == ISD::SHL && N1.getOperand(0).getOpcode() == ISD::AND &&
      (CNMask = dyn_cast<ConstantSDNode>(N1.getOperand(0).getOperand(1))) &&
      isShiftedMask_64(CNMask->getZExtValue(), MaskIdx, MaskLen) &&
      MaskIdx == 0 && (CNShamt = dyn_cast<ConstantSDNode>(N1.getOperand(1))) &&
      CNShamt->getZExtValue() + MaskLen <= ValBits) {
    Shamt = CNShamt->getZExtValue();
    APInt ShMask(ValBits, CNMask->getZExtValue() << Shamt);
    if (ShMask.isSubsetOf(DAG.computeKnownBits(N0).Zero)) {
      LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 6\n");
      return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0,
                         N1.getOperand(0).getOperand(0),
                         DAG.getConstant(Shamt + MaskLen - 1, DL, GRLenVT),
                         DAG.getConstant(Shamt, DL, GRLenVT));
    }
  }

  // 7th pattern.
  // a = b | ((c << shamt) & shifted_mask), where all positions in b to be
  // overwritten by the incoming bits are known to be zero.
  // =>
  // a = BSTRINS b, c, MaskIdx + MaskLen - 1, MaskIdx
  //
  // Similarly, the 7th pattern is more common than the 2nd. So we put the 2nd
  // before the 7th in order to match as many nodes as possible.
  if (N1.getOpcode() == ISD::AND &&
      (CNMask = dyn_cast<ConstantSDNode>(N1.getOperand(1))) &&
      isShiftedMask_64(CNMask->getZExtValue(), MaskIdx, MaskLen) &&
      N1.getOperand(0).getOpcode() == ISD::SHL &&
      (CNShamt = dyn_cast<ConstantSDNode>(N1.getOperand(0).getOperand(1))) &&
      CNShamt->getZExtValue() == MaskIdx) {
    APInt ShMask(ValBits, CNMask->getZExtValue());
    if (ShMask.isSubsetOf(DAG.computeKnownBits(N0).Zero)) {
      LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 7\n");
      return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0,
                         N1.getOperand(0).getOperand(0),
                         DAG.getConstant(MaskIdx + MaskLen - 1, DL, GRLenVT),
                         DAG.getConstant(MaskIdx, DL, GRLenVT));
    }
  }

  // (or a, b) and (or b, a) are equivalent, so swap the operands and retry.
  if (!SwapAndRetried) {
    std::swap(N0, N1);
    SwapAndRetried = true;
    goto Retry;
  }

  SwapAndRetried = false;
Retry2:
  // 8th pattern.
  // a = b | (c & shifted_mask), where all positions in b to be overwritten by
  // the incoming bits are known to be zero.
  // =>
  // a = BSTRINS b, c >> MaskIdx, MaskIdx + MaskLen - 1, MaskIdx
  //
  // Similarly, the 8th pattern is more common than the 4th and 5th patterns. So
  // we put it here in order to match as many nodes as possible or generate less
  // instructions.
  if (N1.getOpcode() == ISD::AND &&
      (CNMask = dyn_cast<ConstantSDNode>(N1.getOperand(1))) &&
      isShiftedMask_64(CNMask->getZExtValue(), MaskIdx, MaskLen)) {
    APInt ShMask(ValBits, CNMask->getZExtValue());
    if (ShMask.isSubsetOf(DAG.computeKnownBits(N0).Zero)) {
      LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 8\n");
      return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0,
                         DAG.getNode(ISD::SRL, DL, N1->getValueType(0),
                                     N1->getOperand(0),
                                     DAG.getConstant(MaskIdx, DL, GRLenVT)),
                         DAG.getConstant(MaskIdx + MaskLen - 1, DL, GRLenVT),
                         DAG.getConstant(MaskIdx, DL, GRLenVT));
    }
  }
  // Swap N0/N1 and retry.
  if (!SwapAndRetried) {
    std::swap(N0, N1);
    SwapAndRetried = true;
    goto Retry2;
  }

  return SDValue();
}

SDValue LoongArchTargetLowering::PerformDAGCombine(SDNode *N,
                                                   DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  switch (N->getOpcode()) {
  default:
    break;
  case ISD::AND:
    return performANDCombine(N, DAG, DCI, Subtarget);
  case ISD::OR:
    return performORCombine(N, DAG, DCI, Subtarget);
  case ISD::SRL:
    return performSRLCombine(N, DAG, DCI, Subtarget);
  }
  return SDValue();
}

static MachineBasicBlock *insertDivByZeroTrap(MachineInstr &MI,
                                              MachineBasicBlock &MBB,
                                              const TargetInstrInfo &TII) {
  if (!ZeroDivCheck)
    return &MBB;

  // Build instructions:
  //   div(or mod)   $dst, $dividend, $divisor
  //   bnez          $divisor, 8
  //   break         7
  //   fallthrough
  MachineOperand &Divisor = MI.getOperand(2);
  auto FallThrough = std::next(MI.getIterator());

  BuildMI(MBB, FallThrough, MI.getDebugLoc(), TII.get(LoongArch::BNEZ))
      .addReg(Divisor.getReg(), getKillRegState(Divisor.isKill()))
      .addImm(8);

  // See linux header file arch/loongarch/include/uapi/asm/break.h for the
  // definition of BRK_DIVZERO.
  BuildMI(MBB, FallThrough, MI.getDebugLoc(), TII.get(LoongArch::BREAK))
      .addImm(7/*BRK_DIVZERO*/);

  // Clear Divisor's kill flag.
  Divisor.setIsKill(false);

  return &MBB;
}

MachineBasicBlock *LoongArchTargetLowering::EmitInstrWithCustomInserter(
    MachineInstr &MI, MachineBasicBlock *BB) const {

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instr type to insert");
  case LoongArch::DIV_W:
  case LoongArch::DIV_WU:
  case LoongArch::MOD_W:
  case LoongArch::MOD_WU:
  case LoongArch::DIV_D:
  case LoongArch::DIV_DU:
  case LoongArch::MOD_D:
  case LoongArch::MOD_DU:
    return insertDivByZeroTrap(MI, *BB, *Subtarget.getInstrInfo());
    break;
  }
}

const char *LoongArchTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((LoongArchISD::NodeType)Opcode) {
  case LoongArchISD::FIRST_NUMBER:
    break;

#define NODE_NAME_CASE(node)                                                   \
  case LoongArchISD::node:                                                     \
    return "LoongArchISD::" #node;

    // TODO: Add more target-dependent nodes later.
    NODE_NAME_CASE(CALL)
    NODE_NAME_CASE(RET)
    NODE_NAME_CASE(SLL_W)
    NODE_NAME_CASE(SRA_W)
    NODE_NAME_CASE(SRL_W)
    NODE_NAME_CASE(BSTRINS)
    NODE_NAME_CASE(BSTRPICK)
    NODE_NAME_CASE(MOVGR2FR_W_LA64)
    NODE_NAME_CASE(MOVFR2GR_S_LA64)
    NODE_NAME_CASE(FTINT)
  }
#undef NODE_NAME_CASE
  return nullptr;
}

//===----------------------------------------------------------------------===//
//                     Calling Convention Implementation
//===----------------------------------------------------------------------===//
// FIXME: Now, we only support CallingConv::C with fixed arguments which are
// passed with integer or floating-point registers.
const MCPhysReg ArgGPRs[] = {LoongArch::R4,  LoongArch::R5, LoongArch::R6,
                             LoongArch::R7,  LoongArch::R8, LoongArch::R9,
                             LoongArch::R10, LoongArch::R11};
const MCPhysReg ArgFPR32s[] = {LoongArch::F0, LoongArch::F1, LoongArch::F2,
                               LoongArch::F3, LoongArch::F4, LoongArch::F5,
                               LoongArch::F6, LoongArch::F7};
const MCPhysReg ArgFPR64s[] = {
    LoongArch::F0_64, LoongArch::F1_64, LoongArch::F2_64, LoongArch::F3_64,
    LoongArch::F4_64, LoongArch::F5_64, LoongArch::F6_64, LoongArch::F7_64};

// Implements the LoongArch calling convention. Returns true upon failure.
static bool CC_LoongArch(unsigned ValNo, MVT ValVT,
                         CCValAssign::LocInfo LocInfo, CCState &State) {
  // Allocate to a register if possible.
  Register Reg;

  if (ValVT == MVT::f32)
    Reg = State.AllocateReg(ArgFPR32s);
  else if (ValVT == MVT::f64)
    Reg = State.AllocateReg(ArgFPR64s);
  else
    Reg = State.AllocateReg(ArgGPRs);
  if (Reg) {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, ValVT, LocInfo));
    return false;
  }

  // TODO: Handle arguments passed without register.
  return true;
}

void LoongArchTargetLowering::analyzeInputArgs(
    CCState &CCInfo, const SmallVectorImpl<ISD::InputArg> &Ins,
    LoongArchCCAssignFn Fn) const {
  for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
    MVT ArgVT = Ins[i].VT;

    if (Fn(i, ArgVT, CCValAssign::Full, CCInfo)) {
      LLVM_DEBUG(dbgs() << "InputArg #" << i << " has unhandled type "
                        << EVT(ArgVT).getEVTString() << '\n');
      llvm_unreachable("");
    }
  }
}

void LoongArchTargetLowering::analyzeOutputArgs(
    CCState &CCInfo, const SmallVectorImpl<ISD::OutputArg> &Outs,
    LoongArchCCAssignFn Fn) const {
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    MVT ArgVT = Outs[i].VT;

    if (Fn(i, ArgVT, CCValAssign::Full, CCInfo)) {
      LLVM_DEBUG(dbgs() << "OutputArg #" << i << " has unhandled type "
                        << EVT(ArgVT).getEVTString() << "\n");
      llvm_unreachable("");
    }
  }
}

static SDValue unpackFromRegLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL,
                                const LoongArchTargetLowering &TLI) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  EVT LocVT = VA.getLocVT();
  const TargetRegisterClass *RC = TLI.getRegClassFor(LocVT.getSimpleVT());
  Register VReg = RegInfo.createVirtualRegister(RC);
  RegInfo.addLiveIn(VA.getLocReg(), VReg);

  return DAG.getCopyFromReg(Chain, DL, VReg, LocVT);
}

// Transform physical registers into virtual registers.
SDValue LoongArchTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {

  MachineFunction &MF = DAG.getMachineFunction();

  switch (CallConv) {
  default:
    llvm_unreachable("Unsupported calling convention");
  case CallingConv::C:
    break;
  }

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  analyzeInputArgs(CCInfo, Ins, CC_LoongArch);

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i)
    InVals.push_back(unpackFromRegLoc(DAG, Chain, ArgLocs[i], DL, *this));

  return Chain;
}

// Lower a call to a callseq_start + CALL + callseq_end chain, and add input
// and output parameter nodes.
SDValue
LoongArchTargetLowering::LowerCall(CallLoweringInfo &CLI,
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
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  CLI.IsTailCall = false;

  if (IsVarArg)
    report_fatal_error("LowerCall with varargs not implemented");

  MachineFunction &MF = DAG.getMachineFunction();

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign> ArgLocs;
  CCState ArgCCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  analyzeOutputArgs(ArgCCInfo, Outs, CC_LoongArch);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = ArgCCInfo.getNextStackOffset();

  for (auto &Arg : Outs) {
    if (!Arg.Flags.isByVal())
      continue;
    report_fatal_error("Passing arguments byval not implemented");
  }

  Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, CLI.DL);

  // Copy argument values to their designated locations.
  SmallVector<std::pair<Register, SDValue>> RegsToPass;
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue = OutVals[i];

    // Promote the value if needed.
    // For now, only handle fully promoted arguments.
    if (VA.getLocInfo() != CCValAssign::Full)
      report_fatal_error("Unknown loc info");

    if (VA.isRegLoc()) {
      // Queue up the argument copies and emit them at the end.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ArgValue));
    } else {
      report_fatal_error("Passing arguments via the stack not implemented");
    }
  }

  SDValue Glue;

  // Build a sequence of copy-to-reg nodes, chained and glued together.
  for (auto &Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, Reg.first, Reg.second, Glue);
    Glue = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress/ExternalSymbol node, turn it into a
  // TargetGlobalAddress/TargetExternalSymbol node so that legalize won't
  // split it and then direct call can be matched by PseudoCALL.
  // FIXME: Add target flags for relocation.
  if (GlobalAddressSDNode *S = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(S->getGlobal(), DL, PtrVT);
  else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), PtrVT);

  // The first call operand is the chain and the second is the target address.
  SmallVector<SDValue> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (auto &Reg : RegsToPass)
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));

  // Add a register mask operand representing the call-preserved registers.
  const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  // Glue the call to the argument copies, if any.
  if (Glue.getNode())
    Ops.push_back(Glue);

  // Emit the call.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  Chain = DAG.getNode(LoongArchISD::CALL, DL, NodeTys, Ops);
  DAG.addNoMergeSiteInfo(Chain.getNode(), CLI.NoMerge);
  Glue = Chain.getValue(1);

  // Mark the end of the call, which is glued to the call itself.
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getConstant(NumBytes, DL, PtrVT, true),
                             DAG.getConstant(0, DL, PtrVT, true), Glue, DL);
  Glue = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign> RVLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RVLocs, *DAG.getContext());
  analyzeInputArgs(RetCCInfo, Ins, CC_LoongArch);

  // Copy all of the result registers out of their specified physreg.
  for (auto &VA : RVLocs) {
    // Copy the value out.
    SDValue RetValue =
        DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT(), Glue);
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    InVals.push_back(Chain.getValue(0));
  }

  return Chain;
}

bool LoongArchTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  // Any return value split in to more than two values can't be returned
  // directly.
  return Outs.size() <= 2;
}

SDValue LoongArchTargetLowering::LowerReturn(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
    SelectionDAG &DAG) const {
  // Stores the assignment of the return value to a location.
  SmallVector<CCValAssign> RVLocs;

  // Info about the registers and stack slot.
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  analyzeOutputArgs(CCInfo, Outs, CC_LoongArch);

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0, e = RVLocs.size(); i < e; ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    // Handle a 'normal' return.
    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), OutVals[i], Glue);

    // Guarantee that all emitted copies are stuck together.
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain; // Update chain.

  // Add the glue node if we have it.
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(LoongArchISD::RET, DL, MVT::Other, RetOps);
}

bool LoongArchTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT,
                                           bool ForCodeSize) const {
  assert((VT == MVT::f32 || VT == MVT::f64) && "Unexpected VT");

  if (VT == MVT::f32 && !Subtarget.hasBasicF())
    return false;
  if (VT == MVT::f64 && !Subtarget.hasBasicD())
    return false;
  return (Imm.isZero() || Imm.isExactlyValue(+1.0));
}
