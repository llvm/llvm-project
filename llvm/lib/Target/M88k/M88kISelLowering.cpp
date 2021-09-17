//===-- M88kISelLowering.cpp - M88k DAG lowering implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the M88kTargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "M88kISelLowering.h"
#include "M88kCondCode.h"
//#include "M88kCallingConv.h"
//#include "M88kConstantPoolValue.h"
//#include "M88kMachineFunctionInfo.h"
#include "M88kTargetMachine.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
//#include "llvm/IR/IntrinsicsM88k.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/KnownBits.h"
#include <cctype>

using namespace llvm;

#define DEBUG_TYPE "m88k-lower"

// If I is a shifted mask, set the size (Width) and the first bit of the
// mask (Offset), and return true.
// For example, if I is 0x003e, (Width, Offset) = (5, 1).
static bool isShiftedMask(uint64_t I, uint64_t &Width, uint64_t &Offset) {
  if (!isShiftedMask_64(I))
    return false;

  Width = countPopulation(I);
  Offset = countTrailingZeros(I);
  return true;
}

static M88kCC::CondCode ISDCCtoM88kCC(ISD::CondCode isdCC) {
  switch (isdCC) {
  case ISD::SETUEQ:
    return M88kCC::EQ;
  case ISD::SETUGT:
    return M88kCC::HI;
  case ISD::SETUGE:
    return M88kCC::HS;
  case ISD::SETULT:
    return M88kCC::LO;
  case ISD::SETULE:
    return M88kCC::LS;
  case ISD::SETUNE:
    return M88kCC::NE;
  case ISD::SETEQ:
    return M88kCC::EQ;
  case ISD::SETGT:
    return M88kCC::GT;
  case ISD::SETGE:
    return M88kCC::GE;
  case ISD::SETLT:
    return M88kCC::LT;
  case ISD::SETLE:
    return M88kCC::LE;
  case ISD::SETNE:
    return M88kCC::NE;
  default:
    llvm_unreachable("Unhandled ISDCC code.");
  }
}

M88kTargetLowering::M88kTargetLowering(const TargetMachine &TM,
                                       const M88kSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {
  addRegisterClass(MVT::i32, &M88k::GPRRegClass);

  // Compute derived properties from the register classes
  computeRegisterProperties(Subtarget.getRegisterInfo());

  // Set up special registers.
  setStackPointerRegisterToSaveRestore(M88k::R31);

  // How we extend i1 boolean values.
  setBooleanContents(ZeroOrOneBooleanContent);

  setMinFunctionAlignment(Align(8));
  setPrefFunctionAlignment(Align(8));

  // setOperationAction(ISD::CTLZ, MVT::i32, Custom);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);

  // Special DAG combiner for bit-field operations.
  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::SHL);
}

SDValue M88kTargetLowering::LowerOperation(SDValue Op,
                                           SelectionDAG &DAG) const {
  // TODO Implement for ops not covered by patterns in .td files.
  /*
    switch (Op.getOpcode())
    {
    case ISD::SHL:          return lowerShiftLeft(Op, DAG);
    }
  */
  return SDValue();
}

namespace {
SDValue performANDCombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  SelectionDAG &DAG = DCI.DAG;
  SDValue FirstOperand = N->getOperand(0);
  unsigned FirstOperandOpc = FirstOperand.getOpcode();
  // Second operand of and must be a constant.
  ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!Mask)
    return SDValue();
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);

  SDValue NewOperand;
  unsigned Opc;

  uint64_t Offset;
  uint64_t MaskWidth, MaskOffset;
  if (isShiftedMask(Mask->getZExtValue(), MaskWidth, MaskOffset)) {
    if (FirstOperandOpc == ISD::SRL || FirstOperandOpc == ISD::SRA) {
      // Pattern match:
      // $dst = and (srl/sra $src, offset), (2**width - 1)
      // => EXTU $dst, $src, width<offset>

      // The second operand of the shift must be an immediate.
      ConstantSDNode *ShiftAmt =
          dyn_cast<ConstantSDNode>(FirstOperand.getOperand(1));
      if (!(ShiftAmt))
        return SDValue();

      Offset = ShiftAmt->getZExtValue();

      // Return if the shifted mask does not start at bit 0 or the sum of its
      // width and offset exceeds the word's size.
      if (MaskOffset != 0 || Offset + MaskWidth > ValTy.getSizeInBits())
        return SDValue();

      Opc = M88kISD::EXTU;
      NewOperand = FirstOperand.getOperand(0);
    } else
      return SDValue();
  } else if (isShiftedMask(~Mask->getZExtValue() &
                               ((0x1ULL << ValTy.getSizeInBits()) - 1),
                           MaskWidth, MaskOffset)) {
    // Pattern match:
    // $dst = and $src, ~((2**width - 1) << offset)
    // => CLR $dst, $src, width<offset>
    Opc = M88kISD::CLR;
    NewOperand = FirstOperand;
    Offset = MaskOffset;
  } else
    return SDValue();
  return DAG.getNode(Opc, DL, ValTy, NewOperand,
                     DAG.getConstant(MaskWidth << 5 | Offset, DL, MVT::i32));
}

SDValue performORCombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  SelectionDAG &DAG = DCI.DAG;
  uint64_t Width, Offset;

  // Second operand of or must be a constant shifted mask.
  ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!Mask || !isShiftedMask(Mask->getZExtValue(), Width, Offset))
    return SDValue();

  // Pattern match:
  // $dst = or $src, ((2**width - 1) << offset
  // => SET $dst, $src, width<offset>
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);
  return DAG.getNode(M88kISD::SET, DL, ValTy, N->getOperand(0),
                     DAG.getConstant(Width << 5 | Offset, DL, MVT::i32));
}

SDValue performSHLCombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  // Pattern match:
  // $dst = shl (and $src, (2**width - 1)), offset
  // => MAK $dst, $src, width<offset>
  SelectionDAG &DAG = DCI.DAG;
  SDValue FirstOperand = N->getOperand(0);
  unsigned FirstOperandOpc = FirstOperand.getOpcode();
  // First operdns shl must be and, second operand must a constant.
  ConstantSDNode *ShiftAmt = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!ShiftAmt || FirstOperandOpc != ISD::AND)
    return SDValue();
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);

  uint64_t Offset;
  uint64_t MaskWidth, MaskOffset;
  ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(FirstOperand->getOperand(1));
  if (!Mask || !isShiftedMask(Mask->getZExtValue(), MaskWidth, MaskOffset))
    return SDValue();

  // The second operand of the shift must be an immediate.
  Offset = ShiftAmt->getZExtValue();

  // Return if the shifted mask does not start at bit 0 or the sum of its
  // width and offset exceeds the word's size.
  if (MaskOffset != 0 || Offset + MaskWidth > ValTy.getSizeInBits())
    return SDValue();

  return DAG.getNode(M88kISD::MAK, DL, ValTy, FirstOperand.getOperand(0),
                     DAG.getConstant(MaskWidth << 5 | Offset, DL, MVT::i32));
}
} // namespace

SDValue M88kTargetLowering::PerformDAGCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();
  LLVM_DEBUG(dbgs() << "In PerformDAGCombine\n");

  // TODO: Match certain and/or/shift ops to ext/mak.
  unsigned Opc = N->getOpcode();

  switch (Opc) {
  default:
    break;
  case ISD::AND:
    return performANDCombine(N, DCI);
  case ISD::OR:
    return performORCombine(N, DCI);
  case ISD::SHL:
    return performSHLCombine(N, DCI);
  }

  return SDValue();
}

//===----------------------------------------------------------------------===//
// Calling conventions
//===----------------------------------------------------------------------===//

#include "M88kGenCallingConv.inc"

SDValue M88kTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {

  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_M88k);

  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    SDValue ArgValue;
    CCValAssign &VA = ArgLocs[I];
    EVT LocVT = VA.getLocVT();
    if (VA.isRegLoc()) {
      // Arguments passed in registers
      const TargetRegisterClass *RC;
      switch (LocVT.getSimpleVT().SimpleTy) {
      default:
        // Integers smaller than i64 should be promoted to i32.
        llvm_unreachable("Unexpected argument type");
      case MVT::i32:
        RC = &M88k::GPRRegClass;
        break;
      }

      Register VReg = MRI.createVirtualRegister(RC);
      MRI.addLiveIn(VA.getLocReg(), VReg);
      ArgValue = DAG.getCopyFromReg(Chain, DL, VReg, LocVT);

      // If this is an 8/16-bit value, it is really passed promoted to 32
      // bits. Insert an assert[sz]ext to capture this, then truncate to the
      // right size.
      if (VA.getLocInfo() == CCValAssign::SExt)
        ArgValue = DAG.getNode(ISD::AssertSext, DL, LocVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      else if (VA.getLocInfo() == CCValAssign::ZExt)
        ArgValue = DAG.getNode(ISD::AssertZext, DL, LocVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));

      if (VA.getLocInfo() != CCValAssign::Full)
        ArgValue = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), ArgValue);

      InVals.push_back(ArgValue);
    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");
      llvm_unreachable(
          "M88k - LowerFormalArguments - Memory argument not implemented");
    }
  }

  if (IsVarArg) {
    llvm_unreachable("M88k - LowerFormalArguments - VarArgs not Implemented");
  }

  return Chain;
}

SDValue
M88kTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                bool IsVarArg,
                                const SmallVectorImpl<ISD::OutputArg> &Outs,
                                const SmallVectorImpl<SDValue> &OutVals,
                                const SDLoc &DL, SelectionDAG &DAG) const {

  MachineFunction &MF = DAG.getMachineFunction();

  // Assign locations to each returned value.
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RetLocs, *DAG.getContext());
  RetCCInfo.AnalyzeReturn(Outs, RetCC_M88k);

  // Quick exit for void returns
  if (RetLocs.empty())
    return DAG.getNode(M88kISD::RET_FLAG, DL, MVT::Other, Chain);

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps;
  RetOps.push_back(Chain);
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    CCValAssign &VA = RetLocs[I];
    SDValue RetValue = OutVals[I];

    // Make the return register live on exit.
    assert(VA.isRegLoc() && "Can only return in registers!");

    // Promote the value as required.
    // TODO: Refactor into own method?
    switch (VA.getLocInfo()) {
    case CCValAssign::SExt:
      RetValue = DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), RetValue);
      break;
    case CCValAssign::ZExt:
      RetValue = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), RetValue);
      break;
    case CCValAssign::AExt:
      RetValue = DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), RetValue);
      break;
    case CCValAssign::Full:
      break;
    default:
      llvm_unreachable("Unhandled VA.getLocInfo()");
    }

    // Chain and glue the copies together.
    Register Reg = VA.getLocReg();
    Chain = DAG.getCopyToReg(Chain, DL, Reg, RetValue, Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(Reg, VA.getLocVT()));
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(M88kISD::RET_FLAG, DL, MVT::Other, RetOps);
}

SDValue M88kTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                      SmallVectorImpl<SDValue> &InVals) const {
  llvm_unreachable("M88k - LowerCall - Not Implemented");
}

const char *M88kTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
#define OPCODE(Opc)                                                            \
  case Opc:                                                                    \
    return #Opc
    OPCODE(M88kISD::RET_FLAG);
    OPCODE(M88kISD::CALL);
    OPCODE(M88kISD::CLR);
    OPCODE(M88kISD::SET);
    OPCODE(M88kISD::EXT);
    OPCODE(M88kISD::EXTU);
    OPCODE(M88kISD::MAK);
    OPCODE(M88kISD::ROT);
    OPCODE(M88kISD::FF1);
    OPCODE(M88kISD::FF0);
#undef OPCODE
  default:
    return nullptr;
  }
}
