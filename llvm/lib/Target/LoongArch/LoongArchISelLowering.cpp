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

using namespace llvm;

#define DEBUG_TYPE "loongarch-isel-lowering"

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

  setOperationAction(ISD::GlobalAddress, GRLenVT, Custom);

  if (Subtarget.is64Bit()) {
    setOperationAction(ISD::SHL, MVT::i32, Custom);
    setOperationAction(ISD::SRA, MVT::i32, Custom);
    setOperationAction(ISD::SRL, MVT::i32, Custom);
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
  }

  setOperationAction(ISD::SELECT_CC, GRLenVT, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  // Compute derived properties from the register classes.
  computeRegisterProperties(STI.getRegisterInfo());

  setStackPointerRegisterToSaveRestore(LoongArch::R3);

  setBooleanContents(ZeroOrOneBooleanContent);

  // Function alignments.
  const Align FunctionAlignment(4);
  setMinFunctionAlignment(FunctionAlignment);

  setTargetDAGCombine(ISD::AND);
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
  }
}

SDValue LoongArchTargetLowering::lowerGlobalAddress(SDValue Op,
                                                    SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  unsigned ADDIOp = Subtarget.is64Bit() ? LoongArch::ADDI_D : LoongArch::ADDI_W;

  // FIXME: Only support PC-relative addressing to access the symbol.
  // TODO: Add target flags.
  if (!isPositionIndependent()) {
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

SDValue LoongArchTargetLowering::PerformDAGCombine(SDNode *N,
                                                   DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  switch (N->getOpcode()) {
  default:
    break;
  case ISD::AND:
    return performANDCombine(N, DAG, DCI, Subtarget);
  case ISD::SRL:
    return performSRLCombine(N, DAG, DCI, Subtarget);
  }
  return SDValue();
}

const char *LoongArchTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((LoongArchISD::NodeType)Opcode) {
  case LoongArchISD::FIRST_NUMBER:
    break;

#define NODE_NAME_CASE(node)                                                   \
  case LoongArchISD::node:                                                     \
    return "LoongArchISD::" #node;

    // TODO: Add more target-dependent nodes later.
    NODE_NAME_CASE(RET)
    NODE_NAME_CASE(SLL_W)
    NODE_NAME_CASE(SRA_W)
    NODE_NAME_CASE(SRL_W)
    NODE_NAME_CASE(BSTRPICK)
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
