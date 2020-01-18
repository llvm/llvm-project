//===- XtensaISelLowering.cpp - Xtensa DAG Lowering Implementation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Xtensa uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "XtensaISelLowering.h"
#include "MCTargetDesc/XtensaBaseInfo.h"
//#include "XtensaCallingConv.h"
#include "XtensaConstantPoolValue.h"
#include "XtensaMachineFunctionInfo.h"
#include "XtensaSubtarget.h"
#include "XtensaTargetMachine.h"
#include "XtensaTargetObjectFile.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>

using namespace llvm;

#define DEBUG_TYPE "xtensa-lower"

static const MCPhysReg XtensaArgRegs[6] = {Xtensa::A2, Xtensa::A3, Xtensa::A4,
                                           Xtensa::A5, Xtensa::A6, Xtensa::A7};

// Return true if we must use long (in fact, indirect) function call.
// It's simplified version, production implimentation must
// resolve a functions in ROM (usually glibc functions)
static bool isLongCall(const char *str) {
  // Currently always use long calls
  return true;
}

// The calling conventions in XtensaCallingConv.td are described in terms of the
// callee's register window. This function translates registers to the
// corresponding caller window %o register.
static unsigned toCallerWindow(unsigned Reg) {
  if (Reg >= Xtensa::A2 && Reg <= Xtensa::A7)
    return Reg - Xtensa::A2 + Xtensa::A10;
  return Reg;
}

XtensaTargetLowering::XtensaTargetLowering(const TargetMachine &tm,
                                           const XtensaSubtarget &STI)
    : TargetLowering(tm), Subtarget(STI) {
  MVT PtrVT = MVT::i32;
  // Set up the register classes.
  addRegisterClass(MVT::i32, &Xtensa::ARRegClass);
  if (Subtarget.hasSingleFloat()) {
    addRegisterClass(MVT::f32, &Xtensa::FPRRegClass);
  }
  // addRegisterClass(MVT::i32, &Xtensa::URRegClass);

  // Set up special registers.
  setStackPointerRegisterToSaveRestore(Xtensa::SP);

  setSchedulingPreference(Sched::RegPressure);

  // For i1 types all bits are zero except bit 0
  setBooleanContents(ZeroOrOneBooleanContent);
  setBooleanVectorContents(
      ZeroOrOneBooleanContent); // vectors of i1s are the same

  // Used by legalize types to correctly generate the setcc result.
  // AddPromotedToType(ISD::SETCC, MVT::i1, MVT::i32);
  setOperationPromotedToType(ISD::SETCC, MVT::i1, MVT::i32);
  setOperationPromotedToType(ISD::BR_CC, MVT::i1, MVT::i32);

  setMinFunctionAlignment(2);

  setOperationAction(ISD::BR_CC, MVT::i32, Legal);
  setOperationAction(ISD::BR_CC, MVT::i64, Expand);
  if (Subtarget.hasSingleFloat())
    setOperationAction(ISD::BR_CC, MVT::f32, Custom);
  else
    setOperationAction(ISD::BR_CC, MVT::f32, Expand);

  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT, MVT::i64, Expand);
  setOperationAction(ISD::SELECT, MVT::f32, Expand);

  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Expand);
  if (Subtarget.hasSingleFloat())
    setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  else
    setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);

  setOperationAction(ISD::SETCC, MVT::i32,
                     Custom /* Legal */); // folds into brcond
  setOperationAction(ISD::SETCC, MVT::i64, Expand);
  if (Subtarget.hasSingleFloat())
    setOperationAction(ISD::SETCC, MVT::f32, Custom);
  else
    setOperationAction(ISD::SETCC, MVT::f32, Expand);

  setOperationAction(ISD::Constant, MVT::i32, Custom);
  setOperationAction(ISD::Constant, MVT::i64, Expand /*Custom */);
  setOperationAction(ISD::ConstantFP, MVT::f32, Custom);

  // Expand jump table branches as address arithmetic followed by an
  // indirect jump.
  setOperationAction(ISD::BR_JT, MVT::Other, Custom);
  // Xtensa also does not have indirect branch so expand them
  setOperationAction(ISD::BRIND, MVT::Other, Expand);

  // make BRCOND legal, its actually only legal for a subset of conds
  setOperationAction(ISD::BRCOND, MVT::Other, Legal);

  // Custom Lower Overflow operators

  // Handle integer types.
  for (unsigned I = MVT::FIRST_INTEGER_VALUETYPE;
       I <= MVT::LAST_INTEGER_VALUETYPE; ++I) {
    MVT VT = MVT::SimpleValueType(I);
    if (isTypeLegal(VT)) {
      // No support at all
      setOperationAction(ISD::SDIVREM, VT, Expand);
      setOperationAction(ISD::UDIVREM, VT, Expand);

      setOperationAction(ISD::ATOMIC_SWAP, VT, Expand);
    }
  }

  if (Subtarget.hasMul32())
    setOperationAction(ISD::MUL, MVT::i32, Legal);
  else
    setOperationAction(ISD::MUL, MVT::i32, Expand);

  if (Subtarget.hasMul32High()) {
    setOperationAction(ISD::MULHU, MVT::i32, Legal);
    setOperationAction(ISD::MULHS, MVT::i32, Legal);
  } else {
    setOperationAction(ISD::MULHU, MVT::i32, Expand);
    setOperationAction(ISD::MULHS, MVT::i32, Expand);
  }
  setOperationAction(ISD::MUL, MVT::i64, Expand);
  setOperationAction(ISD::MULHS, MVT::i64, Expand);
  setOperationAction(ISD::MULHU, MVT::i64, Expand);

  if (Subtarget.hasDiv32()) {
    setOperationAction(ISD::SDIV, MVT::i32, Legal);
    setOperationAction(ISD::UDIV, MVT::i32, Legal);
    setOperationAction(ISD::SREM, MVT::i32, Legal);
    setOperationAction(ISD::UREM, MVT::i32, Legal);
  } else {
    setOperationAction(ISD::SDIV, MVT::i32, Expand);
    setOperationAction(ISD::UDIV, MVT::i32, Expand);
    setOperationAction(ISD::SREM, MVT::i32, Expand);
    setOperationAction(ISD::UREM, MVT::i32, Expand);
  }
  setOperationAction(ISD::SDIV, MVT::i64, Expand);
  setOperationAction(ISD::UDIV, MVT::i64, Expand);
  setOperationAction(ISD::SREM, MVT::i64, Expand);
  setOperationAction(ISD::UREM, MVT::i64, Expand);

  // Xtensa doesn't support  [ADD,SUB][E,C]
  setOperationAction(ISD::ADDC, MVT::i32, Expand);
  setOperationAction(ISD::ADDE, MVT::i32, Expand);
  setOperationAction(ISD::SUBC, MVT::i32, Expand);
  setOperationAction(ISD::SUBE, MVT::i32, Expand);

  setOperationAction(ISD::ADD, MVT::i64, Expand);
  setOperationAction(ISD::SUB, MVT::i64, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);

  // Xtensa doesn't support s[hl,rl,ra]_parts
  // TODO
  setOperationAction(ISD::SHL_PARTS, MVT::i32, Custom);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Custom);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Custom);

  // Bit Manipulation
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  // Xtensa doesn't support s[hl,rl,ra]_parts
  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTR, MVT::i32, Expand);
  // No special instructions for these.
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Expand);

  setOperationAction(ISD::TRAP, MVT::Other, Legal);

  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i64, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i64, Expand);

  // No sign extend instructions for i1
  for (MVT VT : MVT::integer_valuetypes()) {
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i1, Promote);
  }
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Expand);

  setOperationAction(ISD::BSWAP, MVT::i32, Expand);
  setOperationAction(ISD::BSWAP, MVT::i64, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);
  setOperationAction(ISD::CTLZ, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Expand);

  // Handle the various types of symbolic address.
  setOperationAction(ISD::ConstantPool, PtrVT, Custom);
  setOperationAction(ISD::GlobalAddress, PtrVT, Custom);
  setOperationAction(ISD::GlobalTLSAddress, PtrVT, Custom);
  setOperationAction(ISD::BlockAddress, PtrVT, Custom);
  setOperationAction(ISD::JumpTable, PtrVT, Custom);

  // Expand stack allocations
  setOperationAction(ISD::DYNAMIC_STACKALLOC, PtrVT, Custom /* Expand */);

  // Use custom expanders so that we can force the function to use
  // a frame pointer.
  // TODO: real comment
  setOperationAction(ISD::STACKSAVE, MVT::Other, Custom);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Custom);
  setOperationAction(ISD::FRAMEADDR, MVT::Other, Custom);

  // Handle floating-point types.
  // TODO
  for (unsigned I = MVT::FIRST_FP_VALUETYPE; I <= MVT::LAST_FP_VALUETYPE; ++I) {
    MVT VT = MVT::SimpleValueType(I);
    if (isTypeLegal(VT)) {
      // We can use FI for FRINT.
      // setOperationAction(ISD::FRINT, VT, Legal);
      if (VT.getSizeInBits() == 32 && Subtarget.hasSingleFloat()) {
        setOperationAction(ISD::FADD, VT, Legal);
        setOperationAction(ISD::FSUB, VT, Legal);
        setOperationAction(ISD::FMUL, VT, Legal);
        setOperationAction(ISD::FDIV, VT, Expand);
      } else {
        setOperationAction(ISD::FADD, VT, Expand);
        setOperationAction(ISD::FSUB, VT, Expand);
        setOperationAction(ISD::FMUL, VT, Expand);
        setOperationAction(ISD::FDIV, VT, Expand);
      }

      // TODO: once implemented in InstrInfo uncomment
      setOperationAction(ISD::FSQRT, VT, Expand);

      // No special instructions for these.
      setOperationAction(ISD::FSIN, VT, Expand);
      setOperationAction(ISD::FCOS, VT, Expand);
      setOperationAction(ISD::FREM, VT, Expand);
      setOperationAction(ISD::FABS, VT, Expand);
    }
  }

  // Handle floating-point types.
  if (Subtarget.hasSingleFloat()) {
    setOperationAction(ISD::FMA, MVT::f32, Legal);
    setOperationAction(ISD::BITCAST, MVT::i32, Legal);
    setOperationAction(ISD::BITCAST, MVT::f32, Legal);
    setOperationAction(ISD::UINT_TO_FP, MVT::i32, Legal);
    setOperationAction(ISD::SINT_TO_FP, MVT::i32, Legal);
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Legal);
    setOperationAction(ISD::FP_TO_SINT, MVT::i32, Legal);
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);
  } else {
    setOperationAction(ISD::FMA, MVT::f32, Expand);
    setOperationAction(ISD::SETCC, MVT::f32, Expand);
    setOperationAction(ISD::BITCAST, MVT::i32, Expand);
    setOperationAction(ISD::BITCAST, MVT::f32, Expand);
    setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);
    setOperationAction(ISD::SINT_TO_FP, MVT::i32, Expand);
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
    setOperationAction(ISD::FP_TO_SINT, MVT::i32, Expand);
    setOperationAction(ISD::UINT_TO_FP, MVT::i64, Expand);
    setOperationAction(ISD::SINT_TO_FP, MVT::i64, Expand);
    setOperationAction(ISD::FP_TO_UINT, MVT::i64, Expand);
    setOperationAction(ISD::FP_TO_SINT, MVT::i64, Expand);
  }
  setOperationAction(ISD::FMA, MVT::f64, Expand);
  setOperationAction(ISD::SETCC, MVT::f64, Expand);
  setOperationAction(ISD::BITCAST, MVT::i64, Expand);
  setOperationAction(ISD::BITCAST, MVT::f64, Expand);

  // Needed so that we don't try to implement f128 constant loads using
  // a load-and-extend of a f80 constant (in cases where the constant
  // would fit in an f80).
  for (MVT VT : MVT::fp_valuetypes())
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::f80, Expand);

  // Floating-point truncation and stores need to be done separately.
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  // We have 64-bit FPR<->GPR moves, but need special handling for
  // 32-bit forms.

  // VASTART and VACOPY need to deal with the Xtensa-specific varargs
  // structure, but VAEND is a no-op.
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  // we use special va_list structure so we have to customize this
  setOperationAction(ISD::VAARG, MVT::Other, Custom);
  setOperationAction(ISD::VACOPY, MVT::Other, Custom);

  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  // to have the best chance and doing something good with fences custom lower
  // them
  setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Custom);

  if (!Subtarget.hasS32C1I()) {
    for (unsigned I = MVT::FIRST_INTEGER_VALUETYPE;
         I <= MVT::LAST_INTEGER_VALUETYPE; ++I) {
      MVT VT = MVT::SimpleValueType(I);
      if (isTypeLegal(VT)) {
        setOperationAction(ISD::ATOMIC_CMP_SWAP, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_ADD, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_SUB, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_AND, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_OR, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_XOR, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_NAND, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_MIN, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_MAX, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_UMIN, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_UMAX, VT, Expand);
      }
    }
  }

  if (Subtarget.hasSingleFloat()) {
    setCondCodeAction(ISD::SETOGT, MVT::f32, Expand);
    setCondCodeAction(ISD::SETOGE, MVT::f32, Expand);
    setCondCodeAction(ISD::SETONE, MVT::f32, Expand);
    setCondCodeAction(ISD::SETUGE, MVT::f32, Expand);
    setCondCodeAction(ISD::SETUGT, MVT::f32, Expand);

    setTargetDAGCombine(ISD::FADD);
    setTargetDAGCombine(ISD::FSUB);
    setTargetDAGCombine(ISD::BRCOND);
  }

  // Compute derived properties from the register classes
  computeRegisterProperties(STI.getRegisterInfo());

  if (Subtarget.hasBoolean()) {
    addRegisterClass(MVT::i1, &Xtensa::BRRegClass);
  }
}

/// If a physical register, this returns the register that receives the
/// exception address on entry to an EH pad.
unsigned XtensaTargetLowering::getExceptionPointerRegister(
    const Constant *PersonalityFn) const {
  return Xtensa::A2;
}

/// If a physical register, this returns the register that receives the
/// exception typeid on entry to a landing pad.
unsigned XtensaTargetLowering::getExceptionSelectorRegister(
    const Constant *PersonalityFn) const {
  return Xtensa::A3;
}

bool XtensaTargetLowering::isOffsetFoldingLegal(
    const GlobalAddressSDNode *GA) const {
  // The Xtensa target isn't yet aware of offsets.
  return false;
}

bool XtensaTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT,
                                        bool ForCodeSize) const {
  return false;
}

unsigned XtensaTargetLowering::getVaListSizeInBits(const DataLayout &DL) const {
  // 2 * sizeof(int*) + sizeof(int)
  return 3 * 4;
}

//===----------------------------------------------------------------------===//
// Inline asm support
//===----------------------------------------------------------------------===//
TargetLowering::ConstraintType
XtensaTargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'a':
    case 'd':
    case 'f':
    case 'r':
      return C_RegisterClass;

    default:
      break;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

TargetLowering::ConstraintWeight
XtensaTargetLowering::getSingleConstraintMatchWeight(
    AsmOperandInfo &info, const char *constraint) const {
  ConstraintWeight weight = CW_Invalid;
  Value *CallOperandVal = info.CallOperandVal;
  // If we don't have a value, we can't do a match,
  // but allow it at the lowest weight.
  if (CallOperandVal == NULL)
    return CW_Default;
  Type *type = CallOperandVal->getType();
  // Look at the constraint type.
  switch (*constraint) {
  default:
    weight = TargetLowering::getSingleConstraintMatchWeight(info, constraint);
    break;

  case 'a':
  case 'd':
  case 'r':
    if (CallOperandVal->getType()->isIntegerTy())
      weight = CW_Register;
    break;

  case 'f':
    if (type->isFloatingPointTy())
      weight = CW_Register;
    break;
  }
  return weight;
}

std::pair<unsigned, const TargetRegisterClass *>
XtensaTargetLowering::getRegForInlineAsmConstraint(
    const TargetRegisterInfo *TRI, StringRef Constraint, MVT VT) const {
  if (Constraint.size() == 1) {
    // GCC Constraint Letters
    switch (Constraint[0]) {
    default:
      break;
    case 'a': // Address register
    case 'd': // Data register (equivalent to 'r')
    case 'r': // General-purpose register
      return std::make_pair(0U, &Xtensa::ARRegClass);

    case 'f': // Floating-point register
      if (Subtarget.hasSingleFloat())
        return std::make_pair(
            0U, &Xtensa::ARRegClass /* TODO Xtensa::FP32BitRegClass */);
      return std::make_pair(0U, &Xtensa::ARRegClass);
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void XtensaTargetLowering::LowerAsmOperandForConstraint(
    SDValue Op, std::string &Constraint, std::vector<SDValue> &Ops,
    SelectionDAG &DAG) const {
  SDLoc DL(Op);

  // Only support length 1 constraints for now.
  if (Constraint.length() > 1)
    return;

  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}
//===----------------------------------------------------------------------===//
//  DAG Combine functions
//===----------------------------------------------------------------------===//
static SDValue performMADD_MSUBCombine(SDNode *ROOTNode, SelectionDAG &CurDAG,
                                       const XtensaSubtarget &Subtarget) {
  if (ROOTNode->getOperand(0).getValueType() != MVT::f32)
    return SDValue();

  if (ROOTNode->getOperand(0).getOpcode() != ISD::FMUL &&
      ROOTNode->getOperand(1).getOpcode() != ISD::FMUL)
    return SDValue();

  SDValue Mult = ROOTNode->getOperand(0).getOpcode() == ISD::FMUL
                     ? ROOTNode->getOperand(0)
                     : ROOTNode->getOperand(1);

  SDValue AddOperand = ROOTNode->getOperand(0).getOpcode() == ISD::FMUL
                           ? ROOTNode->getOperand(1)
                           : ROOTNode->getOperand(0);

  if (!Mult.hasOneUse())
    return SDValue();

  SDLoc DL(ROOTNode);

  bool IsAdd = ROOTNode->getOpcode() == ISD::FADD;
  unsigned Opcode = IsAdd ? XtensaISD::MADD : XtensaISD::MSUB;
  SDValue MAddOps[3] = {AddOperand, Mult->getOperand(0), Mult->getOperand(1)};
  EVT VTs[3] = {MVT::f32, MVT::f32, MVT::f32};
  SDValue MAdd = CurDAG.getNode(Opcode, DL, VTs, MAddOps);

  return MAdd;
}

static SDValue performSUBCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const XtensaSubtarget &Subtarget) {
  if (DCI.isBeforeLegalizeOps()) {
    if (Subtarget.hasSingleFloat() && N->getValueType(0) == MVT::f32)
      return performMADD_MSUBCombine(N, DAG, Subtarget);
  }
  return SDValue();
}

static SDValue performADDCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const XtensaSubtarget &Subtarget) {
  if (DCI.isBeforeLegalizeOps()) {
    if (Subtarget.hasSingleFloat() && N->getValueType(0) == MVT::f32)
      return performMADD_MSUBCombine(N, DAG, Subtarget);
  }
  return SDValue();
}

static SDValue performBRCONDCombine(SDNode *N, SelectionDAG &DAG,
                                    TargetLowering::DAGCombinerInfo &DCI,
                                    const XtensaSubtarget &Subtarget) {
  if (DCI.isBeforeLegalizeOps()) {
    SDValue Chain = N->getOperand(0);

    if (N->getOperand(1).getOpcode() != ISD::SETCC)
      return SDValue();

    SDLoc DL(N);
    SDValue SetCC = N->getOperand(1);
    SDValue Dest = N->getOperand(2);
    ISD::CondCode CC = cast<CondCodeSDNode>(SetCC->getOperand(2))->get();
    SDValue LHS = SetCC->getOperand(0);
    SDValue RHS = SetCC->getOperand(1);

    if (LHS.getValueType() != MVT::i32)
      return SDValue();

    return DAG.getNode(ISD::BR_CC, DL, MVT::isVoid, Chain, DAG.getCondCode(CC),
                       LHS, RHS, Dest);
  }
  return SDValue();
}

SDValue XtensaTargetLowering::PerformDAGCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  unsigned Opc = N->getOpcode();

  switch (Opc) {
  default:
    break;
  case ISD::FADD:
    return performADDCombine(N, DAG, DCI, Subtarget);
  case ISD::FSUB:
    return performSUBCombine(N, DAG, DCI, Subtarget);
  case ISD::BRCOND:
    return performBRCONDCombine(N, DAG, DCI, Subtarget);
  }

  return SDValue();
}

//===----------------------------------------------------------------------===//
//  Lower helper functions
//===----------------------------------------------------------------------===//
#if 0
// addLiveIn - This helper function adds the specified physical register to
// the MachineFunction as a live in value.  It also creates a corresponding
// virtual register for it.
static unsigned addLiveIn(MachineFunction &MF, unsigned PReg,
                          const TargetRegisterClass *RC) {
  unsigned VReg = MF.getRegInfo().createVirtualRegister(RC);
  MF.getRegInfo().addLiveIn(PReg, VReg);
  return VReg;
}
#endif
//===----------------------------------------------------------------------===//
// Calling conventions
//===----------------------------------------------------------------------===//

#include "XtensaGenCallingConv.inc"

static bool CC_Xtensa_Custom(unsigned ValNo, MVT ValVT, MVT LocVT,
                             CCValAssign::LocInfo LocInfo,
                             ISD::ArgFlagsTy ArgFlags, CCState &State) {
  static const MCPhysReg IntRegs[] = {Xtensa::A2, Xtensa::A3, Xtensa::A4,
                                      Xtensa::A5, Xtensa::A6, Xtensa::A7};

  // Do not process byval args here.
  if (ArgFlags.isByVal())
    return true;

  // Promote i8 and i16
  if (LocVT == MVT::i8 || LocVT == MVT::i16) {
    LocVT = MVT::i32;
    if (ArgFlags.isSExt())
      LocInfo = CCValAssign::SExt;
    else if (ArgFlags.isZExt())
      LocInfo = CCValAssign::ZExt;
    else
      LocInfo = CCValAssign::AExt;
  }

  unsigned Reg;

  unsigned OrigAlign = ArgFlags.getOrigAlign();
  bool isI64 = (ValVT == MVT::i32 && OrigAlign == 8);

  if (ValVT == MVT::i32 || ValVT == MVT::f32) {
    Reg = State.AllocateReg(IntRegs);
    // If this is the first part of an i64 arg,
    // the allocated register must be either A0 or A2.
    if (isI64 && (Reg == Xtensa::A3 || Reg == Xtensa::A5 || Reg == Xtensa::A7))
      Reg = State.AllocateReg(IntRegs);
    LocVT = MVT::i32;
  } else if (ValVT == MVT::f64) {
    // Allocate int register and shadow next int register.
    Reg = State.AllocateReg(IntRegs);
    if (Reg == Xtensa::A3 || Reg == Xtensa::A5 || Reg == Xtensa::A7)
      Reg = State.AllocateReg(IntRegs);
    State.AllocateReg(IntRegs);
    LocVT = MVT::i32;
  } else
    llvm_unreachable("Cannot handle this ValVT.");

  if (!Reg) {
    unsigned Offset = State.AllocateStack(ValVT.getStoreSize(), OrigAlign);
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
  } else
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));

  return false;
}

CCAssignFn *XtensaTargetLowering::CCAssignFnForCall(CallingConv::ID CC,
                                                    bool isVarArg) const {
  // return isVarArg ? CC_Xtensa_VAR : CC_Xtensa;
  return CC_Xtensa_Custom;
}

// Value is a value that has been passed to us in the location described by VA
// (and so has type VA.getLocVT()).  Convert Value to VA.getValVT(), chaining
// any loads onto Chain.
static SDValue convertLocVTToValVT(SelectionDAG &DAG, const SDLoc &DL,
                                   CCValAssign &VA, SDValue Chain,
                                   SDValue Value) {
  // If the argument has been promoted from a smaller type, insert an
  // assertion to capture this.
  if (VA.getLocInfo() == CCValAssign::SExt)
    Value = DAG.getNode(ISD::AssertSext, DL, VA.getLocVT(), Value,
                        DAG.getValueType(VA.getValVT()));
  else if (VA.getLocInfo() == CCValAssign::ZExt)
    Value = DAG.getNode(ISD::AssertZext, DL, VA.getLocVT(), Value,
                        DAG.getValueType(VA.getValVT()));

  if (VA.isExtInLoc())
    Value = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Value);
  else if (VA.getLocInfo() == CCValAssign::Indirect)
    Value = DAG.getLoad(VA.getValVT(), DL, Chain, Value, MachinePointerInfo());
  else if (VA.getValVT() == MVT::f32)
    Value = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Value);
  else
    assert(VA.getLocInfo() == CCValAssign::Full && "Unsupported getLocInfo");
  return Value;
}

// Value is a value of type VA.getValVT() that we need to copy into
// the location described by VA.  Return a copy of Value converted to
// VA.getValVT().  The caller is responsible for handling indirect values.
static SDValue convertValVTToLocVT(SelectionDAG &DAG, SDLoc DL, CCValAssign &VA,
                                   SDValue Value) {
  switch (VA.getLocInfo()) {
  case CCValAssign::SExt:
    return DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Value);
  case CCValAssign::ZExt:
    return DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Value);
  case CCValAssign::AExt:
    return DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Value);
  case CCValAssign::BCvt:
    return DAG.getNode(ISD::BITCAST, DL, VA.getLocVT(), Value);
  case CCValAssign::Full:
    return Value;
  default:
    llvm_unreachable("Unhandled getLocInfo()");
  }
}

SDValue XtensaTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  XtensaFunctionInfo *XtensaFI = MF.getInfo<XtensaFunctionInfo>();
  EVT PtrVT = getPointerTy(MF.getDataLayout());

  // errs() >> "function " << MF.getName() << "\n";

  XtensaFI->setVarArgsFrameIndex(0);

  // Used with vargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  CCInfo.AnalyzeFormalArguments(Ins, CCAssignFnForCall(CallConv, IsVarArg));

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    // Arguments stored on registers
    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();
      const TargetRegisterClass *RC;

      if (RegVT == MVT::i32) {
        RC = &Xtensa::ARRegClass;
      } else
        llvm_unreachable("RegVT not supported by FormalArguments Lowering");

      // Transform the arguments stored on
      // physical registers into virtual ones
      unsigned Reg = MF.addLiveIn(VA.getLocReg(), RC);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, RegVT);

      // If this is an 8 or 16-bit value, it has been passed promoted
      // to 32 bits.  Insert an assert[sz]ext to capture this, then
      // truncate to the right size.
      if (VA.getLocInfo() != CCValAssign::Full) {
        unsigned Opcode = 0;
        if (VA.getLocInfo() == CCValAssign::SExt)
          Opcode = ISD::AssertSext;
        else if (VA.getLocInfo() == CCValAssign::ZExt)
          Opcode = ISD::AssertZext;
        if (Opcode)
          ArgValue = DAG.getNode(Opcode, DL, RegVT, ArgValue,
                                 DAG.getValueType(VA.getValVT()));
        if (VA.getValVT() == MVT::f32)
          ArgValue = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), ArgValue);
        else
          ArgValue = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), ArgValue);
      }

      InVals.push_back(ArgValue);

    } else { // !VA.isRegLoc()
      // sanity check
      assert(VA.isMemLoc());

      EVT ValVT = VA.getValVT();

      // The stack pointer offset is relative to the caller stack frame.
      int FI = MFI.CreateFixedObject(ValVT.getSizeInBits() / 8,
                                     VA.getLocMemOffset(), true);

      // Create load nodes to retrieve arguments from the stack
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      InVals.push_back(DAG.getLoad(
          ValVT, DL, Chain, FIN,
          MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI)));
    }
  }

  if (IsVarArg) {
    ArrayRef<MCPhysReg> ArgRegs = makeArrayRef(XtensaArgRegs);
    unsigned Idx = CCInfo.getFirstUnallocated(ArgRegs);
    const TargetRegisterClass *RC = &Xtensa::ARRegClass;
    MachineFrameInfo &MFI = MF.getFrameInfo();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    unsigned RegSize = 4;
    MVT RegTy = MVT::getIntegerVT(RegSize * 8);

    XtensaFI->setVarArgsFirstGPR(Idx + 2); // 2 - number of a2 register

    XtensaFI->setVarArgsStackOffset(MFI.CreateFixedObject(
        PtrVT.getSizeInBits() / 8, CCInfo.getNextStackOffset(), true));

    // Offset of the first variable argument from stack pointer, and size of
    // the vararg save area. For now, the varargs save area is either zero or
    // large enough to hold a0-a7.
    int VaArgOffset, VarArgsSaveSize;

    // If all registers are allocated, then all varargs must be passed on the
    // stack and we don't need to save any argregs.
    if (ArgRegs.size() == Idx) {
      VaArgOffset = CCInfo.getNextStackOffset();
      VarArgsSaveSize = 0;
    } else {
      VarArgsSaveSize = RegSize * (ArgRegs.size() - Idx);
      VaArgOffset = -VarArgsSaveSize;
    }

    // Record the frame index of the first variable argument
    // which is a value necessary to VASTART.
    int FI = MFI.CreateFixedObject(RegSize, VaArgOffset, true);
    XtensaFI->setVarArgsFrameIndex(FI);

    // Copy the integer registers that may have been used for passing varargs
    // to the vararg save area.
    for (unsigned I = Idx; I < ArgRegs.size(); ++I, VaArgOffset += RegSize) {
      const unsigned Reg = RegInfo.createVirtualRegister(RC);
      RegInfo.addLiveIn(ArgRegs[I], Reg);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, RegTy);
      FI = MFI.CreateFixedObject(RegSize, VaArgOffset, true);
      SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue Store = DAG.getStore(Chain, DL, ArgValue, PtrOff,
                                   MachinePointerInfo::getFixedStack(MF, FI));
      cast<StoreSDNode>(Store.getNode())
          ->getMemOperand()
          ->setValue((Value *)nullptr);
      OutChains.push_back(Store);
    }
  }

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens when on varg functions
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  }

  return Chain;
}

SDValue XtensaTargetLowering::getTargetNode(SDValue Op, SelectionDAG &DAG,
                                            unsigned Flag) const {
  EVT Ty = getPointerTy(DAG.getDataLayout());

  if (GlobalAddressSDNode *N = dyn_cast<GlobalAddressSDNode>(Op))
    return DAG.getTargetGlobalAddress(N->getGlobal(), SDLoc(Op), Ty, 0, Flag);
  if (ExternalSymbolSDNode *N = dyn_cast<ExternalSymbolSDNode>(Op))
    return DAG.getTargetExternalSymbol(N->getSymbol(), Ty, Flag);
  if (BlockAddressSDNode *N = dyn_cast<BlockAddressSDNode>(Op))
    return DAG.getTargetBlockAddress(N->getBlockAddress(), Ty, 0, Flag);
  if (JumpTableSDNode *N = dyn_cast<JumpTableSDNode>(Op))
    return DAG.getTargetJumpTable(N->getIndex(), Ty, Flag);
  if (ConstantPoolSDNode *N = dyn_cast<ConstantPoolSDNode>(Op))
    return DAG.getTargetConstantPool(N->getConstVal(), Ty, N->getAlignment(),
                                     N->getOffset(), Flag);

  llvm_unreachable("Unexpected node type.");
  return SDValue();
}

SDValue XtensaTargetLowering::getAddrPCRel(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  return DAG.getNode(XtensaISD::PCREL_WRAPPER, DL, Ty, Op);
}

SDValue
XtensaTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVector<ISD::OutputArg, 32> &Outs = CLI.Outs;
  SmallVector<SDValue, 32> &OutVals = CLI.OutVals;
  SmallVector<ISD::InputArg, 32> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &isTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;

  MachineFunction &MF = DAG.getMachineFunction();
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  const TargetFrameLowering *TFL = Subtarget.getFrameLowering();

  // Xtensa target does not yet support tail call optimization.
  isTailCall = false;

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  CCAssignFn *CC = CCAssignFnForCall(CallConv, IsVarArg);

  CCInfo.AnalyzeCallOperands(Outs, CC);

  //
  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();

  unsigned StackAlignment = TFL->getStackAlignment();
  unsigned NextStackOffset = alignTo(NumBytes, StackAlignment);

  // Mark the start of the call.

  // TODO
  // if (!IsTailCall)
  Chain = DAG.getCALLSEQ_START(Chain, NextStackOffset, 0, DL);

  // Copy argument values to their designated locations.
  std::deque<std::pair<unsigned, SDValue>> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;
  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    CCValAssign &VA = ArgLocs[I];
    SDValue ArgValue = OutVals[I];
    ISD::ArgFlagsTy Flags = Outs[I].Flags;

    ArgValue = convertValVTToLocVT(DAG, DL, VA, ArgValue);

    if (VA.isRegLoc())
      // Queue up the argument copies and emit them at the end.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ArgValue));
    else if (Flags.isByVal()) {
      assert(VA.isMemLoc());
      assert(Flags.getByValSize() &&
             "ByVal args of size 0 should have been ignored by front-end.");
      assert(!isTailCall &&
             "Do not tail-call optimize if there is a byval argument.");

      // True if this byval aggregate will be split between registers
      // and memory.
      unsigned ByValArgsCount = CCInfo.getInRegsParamsCount();
      unsigned CurByValIdx = CCInfo.getInRegsParamsProcessed();
      if (CurByValIdx < ByValArgsCount) {
        unsigned RegBegin, RegEnd;
        CCInfo.getInRegsParamInfo(CurByValIdx, RegBegin, RegEnd);

        EVT PtrVT =
            DAG.getTargetLoweringInfo().getPointerTy(DAG.getDataLayout());
        unsigned int i, j;
        for (i = 0, j = RegBegin; j < RegEnd; i++, j++) {
          SDValue Const = DAG.getConstant(
              4 * i, DL, MVT::i32); // TODO:should this i32 be ptrTy
          SDValue AddArg = DAG.getNode(ISD::ADD, DL, PtrVT, ArgValue, Const);
          SDValue Load =
              DAG.getLoad(PtrVT, DL, Chain, AddArg, MachinePointerInfo(),
                          DAG.InferPtrAlignment(AddArg));
          MemOpChains.push_back(Load.getValue(1));
          RegsToPass.push_back(std::make_pair(j, Load));
        }

        CCInfo.nextInRegsParam();
      }

      // TODO: Handle byvals partially or entirely not in registers

    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");

      // Work out the address of the stack slot.  Unpromoted ints and
      // floats are passed as right-justified 8-byte values.
      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, Xtensa::SP, PtrVT);
      unsigned Offset = VA.getLocMemOffset();
      SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr,
                                    DAG.getIntPtrConstant(Offset, DL));

      // Emit the store.
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, Address, MachinePointerInfo()));
    }
  }

  // Join the stores, which are independent of one another.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  // Build a sequence of copy-to-reg nodes, chained and glued together.
  SDValue Glue;
  for (unsigned I = 0, E = RegsToPass.size(); I != E; ++I) {
    unsigned Reg = RegsToPass[I].first;
    if (Subtarget.isWinABI())
      Reg = toCallerWindow(Reg);
    Chain = DAG.getCopyToReg(Chain, DL, Reg, RegsToPass[I].second, Glue);
    Glue = Chain.getValue(1);
  }

  // const char *name = 0;
  std::string name;

  unsigned char TF = 0;

  // Accept direct calls by converting symbolic call addresses to the
  // associated Target* opcodes.
  if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    name = E->getSymbol();
    TF = E->getTargetFlags();
    if (isPositionIndependent()) {
      report_fatal_error("PIC relocations is not supported");
    } else
      Callee = DAG.getTargetExternalSymbol(E->getSymbol(), PtrVT, TF);
  } else if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    // TODO replace GlobalAddress to some special operand instead of
    // ExternalSymbol
    //   Callee =
    //   DAG.getTargetExternalSymbol(strdup(G->getGlobal()->getName().str().c_str()),
    //   PtrVT);

    const GlobalValue *GV = G->getGlobal();
    name = GV->getName().str();
  }

  if ((!name.empty()) && isLongCall(name.c_str())) {
    // Create a constant pool entry for the callee address
    XtensaCP::XtensaCPModifier Modifier = XtensaCP::no_modifier;

    XtensaConstantPoolValue *CPV = XtensaConstantPoolSymbol::Create(
        *DAG.getContext(), name.c_str(), 0 /* XtensaCLabelIndex */, false,
        Modifier);

    // Get the address of the callee into a register
    SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 4, 0, TF);
    SDValue CPWrap = getAddrPCRel(CPAddr, DAG);
    Callee = CPWrap;
  }

  // The first call operand is the chain and the second is the target address.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // TODO  if (!IsTailCall)
  {
    // Add a register mask operand representing the call-preserved registers.
    const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
    const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
    assert(Mask && "Missing call preserved mask for calling convention");
    Ops.push_back(DAG.getRegisterMask(Mask));
  }

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (unsigned I = 0, E = RegsToPass.size(); I != E; ++I) {
    unsigned Reg = RegsToPass[I].first;
    if (Subtarget.isWinABI())
      Reg = toCallerWindow(Reg);
    Ops.push_back(DAG.getRegister(Reg, RegsToPass[I].second.getValueType()));
  }

  // Glue the call to the argument copies, if any.
  if (Glue.getNode())
    Ops.push_back(Glue);

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  Chain = DAG.getNode(Subtarget.isWinABI() ? XtensaISD::CALLW : XtensaISD::CALL,
                      DL, NodeTys, Ops);
  Glue = Chain.getValue(1);

  // Mark the end of the call, which is glued to the call itself.
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getConstant(NumBytes, DL, PtrVT, true),
                             DAG.getConstant(0, DL, PtrVT, true), Glue, DL);
  Glue = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RetLocs, *DAG.getContext());
  RetCCInfo.AnalyzeCallResult(Ins, Subtarget.isWinABI() ? RetCCW_Xtensa
                                                        : RetCC_Xtensa);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    CCValAssign &VA = RetLocs[I];

    // Copy the value out, gluing the copy to the end of the call sequence.
    unsigned Reg = VA.getLocReg();
    //    if (Subtarget.isWinABI())
    //      Reg = toCallerWindow(Reg);
    SDValue RetValue = DAG.getCopyFromReg(Chain, DL, Reg, VA.getLocVT(), Glue);
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    // Convert the value of the return register into the value that's
    // being returned.
    InVals.push_back(convertLocVTToValVT(DAG, DL, VA, Chain, RetValue));
  }
  return Chain;
}

/// This hook should be implemented to check whether the return values
/// described by the Outs array can fit into the return registers.  If false
/// is returned, an sret-demotion is performed.
bool XtensaTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);
  return CCInfo.CheckReturn(Outs, RetCC_Xtensa);
}

SDValue
XtensaTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                  bool IsVarArg,
                                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                                  const SmallVectorImpl<SDValue> &OutVals,
                                  const SDLoc &DL, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();

  // Assign locations to each returned value.
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RetLocs, *DAG.getContext());
  RetCCInfo.AnalyzeReturn(Outs, RetCC_Xtensa);

  SDValue Glue;
  // Quick exit for void returns
  if (RetLocs.empty())
    return DAG.getNode(Subtarget.isWinABI() ? XtensaISD::RETW_FLAG
                                            : XtensaISD::RET_FLAG,
                       DL, MVT::Other, Chain);

  // Copy the result values into the output registers.
  SmallVector<SDValue, 4> RetOps;
  RetOps.push_back(Chain);
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    CCValAssign &VA = RetLocs[I];
    SDValue RetValue = OutVals[I];

    // Make the return register live on exit.
    assert(VA.isRegLoc() && "Can only return in registers!");

    // Promote the value as required.
    RetValue = convertValVTToLocVT(DAG, DL, VA, RetValue);

    // Chain and glue the copies together.
    unsigned Reg = VA.getLocReg();
    Chain = DAG.getCopyToReg(Chain, DL, Reg, RetValue, Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(Reg, VA.getLocVT()));
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(Subtarget.isWinABI() ? XtensaISD::RETW_FLAG
                                          : XtensaISD::RET_FLAG,
                     DL, MVT::Other, RetOps);
}

static SDValue EmitCMP(SDValue &LHS, SDValue &RHS, ISD::CondCode CC, SDLoc dl,
                       SelectionDAG &DAG, int &br_code) {
  // Minor optimization: if LHS is a constant, swap operands, then the
  // constant can be folded into comparison.
  if (LHS.getOpcode() == ISD::Constant)
    std::swap(LHS, RHS);
  int cmp_code = 0;

  switch (CC) {
  default:
    llvm_unreachable("Invalid condition!");
    break;
  case ISD::SETUNE:
    br_code = XtensaISD::BR_CC_F;
    cmp_code = XtensaISD::CMPOEQ;
    break;
  case ISD::SETUO:
    br_code = XtensaISD::BR_CC_T;
    cmp_code = XtensaISD::CMPUO;
    break;
  case ISD::SETO:
    br_code = XtensaISD::BR_CC_F;
    cmp_code = XtensaISD::CMPUO;
    break;
  case ISD::SETUEQ:
    br_code = XtensaISD::BR_CC_T;
    cmp_code = XtensaISD::CMPUEQ;
    break;
  case ISD::SETULE:
    br_code = XtensaISD::BR_CC_T;
    cmp_code = XtensaISD::CMPULE;
    break;
  case ISD::SETULT:
    br_code = XtensaISD::BR_CC_T;
    cmp_code = XtensaISD::CMPULT;
    break;
  case ISD::SETEQ:
  case ISD::SETOEQ:
    br_code = XtensaISD::BR_CC_T;
    cmp_code = XtensaISD::CMPOEQ;
    break;
  case ISD::SETNE:
    br_code = XtensaISD::BR_CC_F;
    cmp_code = XtensaISD::CMPOEQ;
    break;
  case ISD::SETLE:
  case ISD::SETOLE:
    br_code = XtensaISD::BR_CC_T;
    cmp_code = XtensaISD::CMPOLE;
    break;
  case ISD::SETLT:
  case ISD::SETOLT:
    br_code = XtensaISD::BR_CC_T;
    cmp_code = XtensaISD::CMPOLT;
    break;
  case ISD::SETGE:
    br_code = XtensaISD::BR_CC_F;
    cmp_code = XtensaISD::CMPOLT;
    break;
  case ISD::SETGT:
    br_code = XtensaISD::BR_CC_F;
    cmp_code = XtensaISD::CMPOLE;
    break;
  }
  return DAG.getNode(cmp_code, dl, MVT::i1, LHS, RHS);
}

SDValue XtensaTargetLowering::lowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);
  SDLoc DL(Op);

  if (LHS.getValueType() == MVT::f32) {
    int br_code;
    SDValue Flag = EmitCMP(LHS, RHS, CC, DL, DAG, br_code);
    return DAG.getNode(br_code, DL, Op.getValueType(), Chain, Flag, Dest);
  } else { // MVT::i32
    SDValue setcc =
        DAG.getNode(ISD::SETCC, DL, MVT::i32, LHS, RHS, DAG.getCondCode(CC));
    return DAG.getNode(ISD::BRCOND, DL, Op.getValueType(), Chain, setcc, Dest);
  }
}

SDValue XtensaTargetLowering::lowerSELECT_CC(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getOperand(0).getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue TrueV = Op.getOperand(2);
  SDValue FalseV = Op.getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op->getOperand(4))->get();
  SDValue TargetCC = DAG.getConstant(CC, DL, MVT::i32);

  // Wrap select nodes
  if (LHS.getValueType() == MVT::f32)
    return DAG.getNode(XtensaISD::SELECT_CC_FP, DL, TrueV.getValueType(), LHS,
                       RHS, TrueV, FalseV, TargetCC);
  else if (TrueV.getValueType() == MVT::f32)
    return DAG.getNode(XtensaISD::SELECT_CC_FP, DL, TrueV.getValueType(), LHS,
                       RHS, TrueV, FalseV, TargetCC);
  else
    return DAG.getNode(XtensaISD::SELECT_CC, DL, Ty, LHS, RHS, TrueV, FalseV,
                       TargetCC);
}

SDValue XtensaTargetLowering::lowerSETCC(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getOperand(0).getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  SDValue TargetCC = DAG.getConstant(CC, DL, MVT::i32);

  // Check Op SDNode users
  // If there are only CALL/CALLW nodes, don't expand Global Address
  SDNode &OpNode = *Op.getNode();
  bool Val = false;
  for (SDNode::use_iterator UI = OpNode.use_begin(); UI != OpNode.use_end();
       ++UI) {
    SDNode &User = *UI.getUse().getUser();
    unsigned OpCode = User.getOpcode();
    if (OpCode == ISD::BRCOND) {
      Val = true;
      break;
    }
  }

  // SETCC has BRCOND predecessor, return original operation
  if (Val)
    return Op;

  // Expand to target SELECT_CC
  SDValue TrueV = DAG.getConstant(1, DL, Op.getValueType());
  SDValue FalseV = DAG.getConstant(0, DL, Op.getValueType());

  if (LHS.getValueType() == MVT::f32)
    return DAG.getNode(XtensaISD::SELECT_CC_FP, DL, TrueV.getValueType(), LHS,
                       RHS, TrueV, FalseV, TargetCC);
  else if (TrueV.getValueType() == MVT::f32)
    return DAG.getNode(XtensaISD::SELECT_CC_FP, DL, TrueV.getValueType(), LHS,
                       RHS, TrueV, FalseV, TargetCC);
  else
    return DAG.getNode(XtensaISD::SELECT_CC, DL, Ty, LHS, RHS, TrueV, FalseV,
                       TargetCC);
}

SDValue XtensaTargetLowering::lowerRETURNADDR(SDValue Op,
                                              SelectionDAG &DAG) const {
  // check the depth
  // TODO: xtensa-gcc can handle this, by navigating through the stack, we
  // should be able to do this too
  assert((cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue() == 0) &&
         "Return address can be determined only for current frame.");

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MVT VT = Op.getSimpleValueType();
  unsigned RA = Xtensa::A0;
  MFI.setReturnAddressIsTaken(true);

  // Return RA, which contains the return address. Mark it an implicit
  // live-in.
  unsigned Reg = MF.addLiveIn(RA, getRegClassFor(VT));
  return DAG.getCopyFromReg(DAG.getEntryNode(), SDLoc(Op), Reg, VT);
}

SDValue XtensaTargetLowering::lowerImmediate(SDValue Op,
                                             SelectionDAG &DAG) const {
  const ConstantSDNode *CN = cast<ConstantSDNode>(Op);
  SDLoc DL(CN);
  APInt apval = CN->getAPIntValue();
  int64_t value = apval.getSExtValue();
  if (Op.getValueType() == MVT::i32) {
    if (value > -2048 && value <= 2047)
      return Op;
    Type *Ty = Type::getInt32Ty(*DAG.getContext());
    Constant *CV = ConstantInt::get(Ty, value);
    SDValue CP = DAG.getConstantPool(CV, MVT::i32, 0, 0, false);
    return CP;
  } else if (Op.getValueType() == MVT::i64) {
    // TODO long constants
  }
  return Op;
}

SDValue XtensaTargetLowering::lowerImmediateFP(SDValue Op,
                                               SelectionDAG &DAG) const {
  const ConstantFPSDNode *CN = cast<ConstantFPSDNode>(Op);
  SDLoc DL(CN);
  APFloat apval = CN->getValueAPF();
  int64_t value = FloatToBits(CN->getValueAPF().convertToFloat());
  if (Op.getValueType() == MVT::f32) {
    Type *Ty = Type::getInt32Ty(*DAG.getContext());
    Constant *CV = ConstantInt::get(Ty, value);
    SDValue CP = DAG.getConstantPool(CV, MVT::i32, 0, 0, false);
    return DAG.getNode(ISD::BITCAST, DL, MVT::f32, CP);
  } else if (Op.getValueType() == MVT::f64) {
    // TODO long constants
  }
  return Op;
}

#include <iostream>

SDValue XtensaTargetLowering::lowerGlobalAddress(SDValue Op,
                                                 SelectionDAG &DAG) const {
  //  Reloc::Model RM = DAG.getTarget().getRelocationModel();
  SDLoc DL(Op);

  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Op)) {
    auto PtrVt = getPointerTy(DAG.getDataLayout());
    const GlobalValue *GV = G->getGlobal();

    // Check Op SDNode users
    // If there are only CALL/CALLW nodes, don't expand Global Address
    SDNode &OpNode = *Op.getNode();
    bool Val = false;
    for (SDNode::use_iterator UI = OpNode.use_begin(); UI != OpNode.use_end();
         ++UI) {
      SDNode &User = *UI.getUse().getUser();
      unsigned OpCode = User.getOpcode();
      if (OpCode != XtensaISD::CALL && OpCode != XtensaISD::CALLW) {
        Val = true;
        break;
      }
    }
    if (!Val) {
      SDValue TargAddr = DAG.getTargetGlobalAddress(G->getGlobal(), DL, PtrVt,
                                                    0, 0 /* TargetFlags */);
      return TargAddr;
    }

    SDValue CPAddr = DAG.getTargetConstantPool(GV, PtrVt, 4);
    SDValue CPWrap = getAddrPCRel(CPAddr, DAG);

    return CPWrap;
  }
  llvm_unreachable("invalid global addresses to lower");
}

SDValue XtensaTargetLowering::lowerGlobalTLSAddress(GlobalAddressSDNode *GA,
                                                    SelectionDAG &DAG) const {
  SDLoc DL(GA);
  const GlobalValue *GV = GA->getGlobal();
  EVT PtrVT = getPointerTy(DAG.getDataLayout());

  if (DAG.getTarget().useEmulatedTLS())
    return LowerToTLSEmulatedModel(GA, DAG);

  TLSModel::Model model = getTargetMachine().getTLSModel(GV);

  if (!Subtarget.hasTHREADPTR()) {
    llvm_unreachable("only emulated TLS supported");
  }

  if ((model == TLSModel::LocalExec) || (model == TLSModel::InitialExec)) {
    auto PtrVt = getPointerTy(DAG.getDataLayout());

    bool Priv = GV->isPrivateLinkage(GV->getLinkage());
    // Create a constant pool entry for the callee address
    XtensaConstantPoolValue *CPV = XtensaConstantPoolSymbol::Create(
        *DAG.getContext(), GV->getName().str().c_str() /* Sym */,
        0 /* XtensaCLabelIndex */, Priv, XtensaCP::TPOFF);

    // Get the address of the callee into a register
    SDValue CPAddr =
        DAG.getTargetConstantPool(CPV, PtrVt, 4, 0, XtensaII::MO_TPOFF);
    SDValue CPWrap = getAddrPCRel(CPAddr, DAG);

    SDValue TPRegister = DAG.getRegister(Xtensa::THREADPTR, MVT::i32);
    SDValue ThreadPointer =
        DAG.getNode(XtensaISD::RUR, DL, MVT::i32, TPRegister);
    return DAG.getNode(ISD::ADD, DL, PtrVT, ThreadPointer, CPWrap);
  } else
    llvm_unreachable("only local-exec and initial-exec TLS mode supported");

  return SDValue();
}

SDValue XtensaTargetLowering::lowerBlockAddress(BlockAddressSDNode *Node,
                                                SelectionDAG &DAG) const {
  const BlockAddress *BA = Node->getBlockAddress();
  EVT PtrVT = getPointerTy(DAG.getDataLayout());

  XtensaConstantPoolValue *CPV =
      XtensaConstantPoolConstant::Create(BA, 0, XtensaCP::CPBlockAddress, 0);
  SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 4);

  SDValue CPWrap = getAddrPCRel(CPAddr, DAG);
  return CPWrap;
}

SDValue XtensaTargetLowering::lowerBR_JT(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue Table = Op.getOperand(1);
  SDValue Index = Op.getOperand(2);
  SDLoc DL(Op);
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Table);
  MachineFunction &MF = DAG.getMachineFunction();
  const MachineJumpTableInfo *MJTI = MF.getJumpTableInfo();

  SDValue TargetJT = DAG.getTargetJumpTable(JT->getIndex(), MVT::i32);

  const DataLayout &TD = DAG.getDataLayout();
  EVT PTy = getPointerTy(TD);

  unsigned EntrySize = MJTI->getEntrySize(TD);

  Index = DAG.getNode(ISD::MUL, DL, Index.getValueType(), Index,
                      DAG.getConstant(EntrySize, DL, Index.getValueType()));
  SDValue Addr = DAG.getNode(ISD::ADD, DL, Index.getValueType(), Index, Table);

  EVT MemVT = EVT::getIntegerVT(*DAG.getContext(), EntrySize * 8);
  SDValue LD = DAG.getExtLoad(ISD::SEXTLOAD, DL, PTy, Chain, Addr,
                              MachinePointerInfo::getJumpTable(MF), MemVT);
  Addr = LD;

  return DAG.getNode(XtensaISD::BR_JT, DL, MVT::Other, LD.getValue(1), Addr,
                     TargetJT);
}

SDValue XtensaTargetLowering::lowerJumpTable(JumpTableSDNode *JT,
                                             SelectionDAG &DAG) const {
  SDLoc DL(JT);
  EVT PtrVt = getPointerTy(DAG.getDataLayout());

  // Create a constant pool entry for the callee address
  XtensaConstantPoolValue *CPV =
      XtensaConstantPoolJumpTable::Create(*DAG.getContext(), JT->getIndex());

  // Get the address of the callee into a register
  SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVt, 4);
  SDValue CPWrap = getAddrPCRel(CPAddr, DAG);

  return CPWrap;
}

SDValue XtensaTargetLowering::lowerConstantPool(ConstantPoolSDNode *CP,
                                                SelectionDAG &DAG) const {
  EVT PtrVT = getPointerTy(DAG.getDataLayout());

  SDValue Result;
  if (CP->isMachineConstantPoolEntry())
    Result = DAG.getTargetConstantPool(CP->getMachineCPVal(), PtrVT,
                                       CP->getAlignment());
  else
    Result = DAG.getTargetConstantPool(CP->getConstVal(), PtrVT,
                                       CP->getAlignment(), CP->getOffset());

  return getAddrPCRel(Result, DAG);
}

SDValue XtensaTargetLowering::lowerVASTART(SDValue Op,
                                           SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  XtensaFunctionInfo *XtensaFI = MF.getInfo<XtensaFunctionInfo>();
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  SDLoc DL(Op);

  SDValue Chain = Op.getOperand(0);
  SDValue Addr = Op.getOperand(1);

  // typedef struct __va_list_tag {
  //   int32_t *__va_stk; /* Initialized to point  to the position of the
  //                       * first argument in memory offset to account for
  //                       the
  //                       * arguments passed in registers and to account for
  //                       * the size of the argument registers not being
  //                       16-byte
  //                       * aligned.  E.G., there are 6 argument registers
  //                       * of 4 bytes each, but we want the __va_ndx for the
  //                       * first stack argument to have the maximal
  //                       * alignment of 16 bytes, so we offset the __va_stk
  //                       address by
  //                       * 32 bytes so that __va_stk[32] references the
  //                       first
  //                       * argument on the stack.
  //                       */
  //   int32_t  *__va_reg; /* Points to a stack-allocated region holding the
  //                        * contents
  //                        * of the incoming argument registers
  //                        */
  //   int32_t __va_ndx;   /* Index initialized to the position of the first
  //                        * unnamed (variable) argument.  This same index is
  //                        also
  //                        * used to address the arguments passed in memory.
  //                       */
  //  } __va_list_tag[1];

  SDValue ArgAR =
      DAG.getConstant(XtensaFI->getVarArgsFirstGPR() * 4 - 8, DL, MVT::i32);
  SDValue StackOffsetFI =
      DAG.getFrameIndex(XtensaFI->getVarArgsStackOffset(), PtrVT);

  SDValue FR = DAG.getFrameIndex(XtensaFI->getVarArgsFrameIndex(), PtrVT);

  uint64_t FrameOffset = PtrVT.getSizeInBits() / 8;
  SDValue ConstFrameOffset1 = DAG.getConstant(FrameOffset, DL, PtrVT);
  SDValue ConstFrameOffset2 = DAG.getConstant(FrameOffset * 2, DL, PtrVT);

  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();

  // Store first word : arguments given in stack  (__va_stk)
  // Advance Argument Overflow pointer down, lest it will point to start
  // after register argument va_arg finished
  SDValue OverflowPtrAdvance = DAG.getConstant(32, DL, PtrVT);
  SDValue StackOffsetFICorr =
      DAG.getNode(ISD::SUB, DL, PtrVT, StackOffsetFI, OverflowPtrAdvance);
  SDValue firstStore =
      DAG.getStore(Chain, DL, StackOffsetFICorr, Addr, MachinePointerInfo(SV));

  uint64_t nextOffset = FrameOffset;
  SDValue nextPtr = DAG.getNode(ISD::ADD, DL, PtrVT, Addr, ConstFrameOffset1);

  // Store second word : arguments given on registers  (__va_reg)
  SDValue FRAdvance =
      DAG.getConstant(XtensaFI->getVarArgsFirstGPR() * 4 - 8, DL, PtrVT);
  SDValue FRDecr = DAG.getNode(ISD::SUB, DL, PtrVT, FR, FRAdvance);
  SDValue secondStore = DAG.getStore(firstStore, DL, FRDecr, nextPtr,
                                     MachinePointerInfo(SV, nextOffset));
  nextOffset += FrameOffset;
  nextPtr = DAG.getNode(ISD::ADD, DL, PtrVT, Addr, ConstFrameOffset2);

  // Store first word : number of int regs  (__va_ndx)
  return DAG.getStore(secondStore, DL, ArgAR, nextPtr,
                      MachinePointerInfo(SV, nextOffset));
}

SDValue XtensaTargetLowering::lowerVAARG(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  EVT VT = Node->getValueType(0);
  SDValue InChain = Node->getOperand(0);
  SDValue VAListPtr = Node->getOperand(1);
  EVT PtrVT = VAListPtr.getValueType();
  const Value *SV = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
  SDLoc DL(Node);
  int Offset = Node->getConstantOperandVal(3);

  SDValue ARAreaPtr = DAG.getNode(ISD::ADD, DL, PtrVT, VAListPtr,
                                  DAG.getConstant(8, DL, MVT::i32));
  SDValue RegSaveAreaPtr = DAG.getNode(ISD::ADD, DL, PtrVT, VAListPtr,
                                       DAG.getConstant(4, DL, MVT::i32));
  SDValue OverflowAreaPtr = DAG.getNode(ISD::ADD, DL, PtrVT, VAListPtr,
                                        DAG.getConstant(0, DL, MVT::i32));

  // areas
  SDValue ARIndex =
      DAG.getLoad(MVT::i32, DL, InChain, ARAreaPtr, MachinePointerInfo());
  InChain = ARIndex.getValue(1);

  SDValue OverflowArea =
      DAG.getLoad(MVT::i32, DL, InChain, OverflowAreaPtr, MachinePointerInfo());
  InChain = OverflowArea.getValue(1);

  SDValue RegSaveArea =
      DAG.getLoad(MVT::i32, DL, InChain, RegSaveAreaPtr, MachinePointerInfo());
  InChain = RegSaveArea.getValue(1);

  // We must align Argument register number to even for 64-bit arguments
  if (VT == MVT::i64 || Offset == 8) {
    SDValue Const4 = DAG.getConstant(4, DL, MVT::i32);
    SDValue IndexIncr = DAG.getNode(ISD::ADD, DL, MVT::i32, ARIndex, Const4);

    SDValue ConstN7 = DAG.getConstant(~7, DL, MVT::i32);
    SDValue IndexMasked =
        DAG.getNode(ISD::AND, DL, MVT::i32, IndexIncr, ConstN7);

    InChain = DAG.getStore(InChain, DL, IndexMasked, ARAreaPtr,
                           MachinePointerInfo(SV));
    ARIndex = IndexMasked;
  }

  int LastArgIdx = 4 * 6; // 6 - index of
                          // Xtensa::a7, last argument register + 1
  SDValue CC =
      DAG.getSetCC(DL, MVT::i32, ARIndex,
                   DAG.getConstant(LastArgIdx, DL, MVT::i32), ISD::SETLT);

  // OurReg = RegSaveArea + ARIndex
  SDValue OurReg = DAG.getNode(ISD::ADD, DL, PtrVT, RegSaveArea, ARIndex);
  // OurOverflow = OverflowArea + ARIndex
  SDValue ARIndexCorrect = DAG.getNode(
      ISD::ADD, DL, PtrVT, DAG.getConstant(8, DL, MVT::i32), ARIndex);
  SDValue OurOverflow =
      DAG.getNode(ISD::ADD, DL, PtrVT, OverflowArea, ARIndexCorrect);

  // determine if we should load from Register save area or Overflow area
  SDValue Result = DAG.getNode(ISD::SELECT, DL, PtrVT, CC, OurReg, OurOverflow);

  // increase AR Index by 4 (or 8 if VT is i64)
  SDValue IndexPlus1 =
      DAG.getNode(ISD::ADD, DL, MVT::i32, ARIndex,
                  DAG.getConstant(VT == MVT::i64 ? 8 : 4, DL, MVT::i32));

  InChain = DAG.getStore(InChain, DL, IndexPlus1, ARAreaPtr,
                         MachinePointerInfo(/*SV*/));

  return DAG.getLoad(VT, DL, InChain, Result, MachinePointerInfo());
}

SDValue XtensaTargetLowering::lowerVACOPY(SDValue Op, SelectionDAG &DAG) const {
  // We have to copy the entire va_list struct:
  // 2*sizeof(int*) + sizeof(int) = 12 Byte
  unsigned VAListSize = 12;
  return DAG.getMemcpy(Op.getOperand(0), Op, Op.getOperand(1), Op.getOperand(2),
                       DAG.getConstant(VAListSize, SDLoc(Op), MVT::i32), 8,
                       false, true, false, MachinePointerInfo(),
                       MachinePointerInfo());
}

SDValue XtensaTargetLowering::lowerATOMIC_FENCE(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  return DAG.getNode(XtensaISD::MEMW, DL, MVT::Other, Chain);
}

SDValue XtensaTargetLowering::lowerSTACKSAVE(SDValue Op,
                                             SelectionDAG &DAG) const {
  unsigned sp = Xtensa::SP;
  return DAG.getCopyFromReg(Op.getOperand(0), SDLoc(Op), sp, Op.getValueType());
}

SDValue XtensaTargetLowering::lowerSTACKRESTORE(SDValue Op,
                                                SelectionDAG &DAG) const {
  unsigned sp = Xtensa::SP;
  if (Subtarget.isWinABI()) {
    SDValue NewSP =
        DAG.getNode(XtensaISD::MOVSP, SDLoc(Op), MVT::i32, Op.getOperand(1));
    return DAG.getCopyToReg(Op.getOperand(0), SDLoc(Op), sp, NewSP);
  } else {
    return DAG.getCopyToReg(Op.getOperand(0), SDLoc(Op), sp, Op.getOperand(1));
  }
}

SDValue XtensaTargetLowering::lowerFRAMEADDR(SDValue Op,
                                             SelectionDAG &DAG) const {
  // check the depth
  assert((cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue() == 0) &&
         "Frame address can only be determined for current frame.");

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = DAG.getMachineFunction().getFrameInfo();
  MFI.setFrameAddressIsTaken(true);
  EVT VT = Op.getValueType();
  SDLoc DL(Op);

  unsigned FrameReg = Subtarget.getRegisterInfo()->getFrameRegister(MF);
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), DL, FrameReg, VT);
  return FrameAddr;
}

SDValue XtensaTargetLowering::lowerDYNAMIC_STACKALLOC(SDValue Op,
                                                      SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0); // Legalize the chain.
  SDValue Size = Op.getOperand(1);  // Legalize the size.
  EVT VT = Size->getValueType(0);
  SDLoc DL(Op);

  // Round up Size to 32
  SDValue Size1 =
      DAG.getNode(ISD::ADD, DL, VT, Size, DAG.getConstant(31, DL, MVT::i32));
  SDValue SizeRoundUp =
      DAG.getNode(ISD::AND, DL, VT, Size1, DAG.getConstant(~31, DL, MVT::i32));

  unsigned SPReg = Xtensa::SP;
  SDValue SP = DAG.getCopyFromReg(Chain, DL, SPReg, VT);
  SDValue NewSP = DAG.getNode(ISD::SUB, DL, VT, SP, SizeRoundUp); // Value
  if (Subtarget.isWinABI()) {
    SDValue NewSP1 = DAG.getNode(XtensaISD::MOVSP, DL, MVT::i32, NewSP);
    Chain = DAG.getCopyToReg(SP.getValue(1), DL, SPReg, NewSP1); // Output chain
  } else {
    Chain = DAG.getCopyToReg(SP.getValue(1), DL, SPReg, NewSP); // Output chain
  }

  SDValue NewVal = DAG.getCopyFromReg(Chain, DL, SPReg, MVT::i32);
  Chain = NewVal.getValue(1);

  SDValue Ops[2] = {NewVal, Chain};
  return DAG.getMergeValues(Ops, DL);
}

SDValue XtensaTargetLowering::lowerShiftLeftParts(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VT = MVT::i32;

  SDValue Lo = Op.getOperand(0), Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);

  SDValue SetShiftLeft = DAG.getNode(XtensaISD::SSL, DL, MVT::Glue, Shamt);
  SDValue ShiftLeftHi =
      DAG.getNode(XtensaISD::SRC, DL, VT, Hi, Lo, SetShiftLeft);
  SDValue SetShiftLeft1 = DAG.getNode(XtensaISD::SSL, DL, MVT::Glue, Shamt);
  SDValue ShiftLeftLo = DAG.getNode(XtensaISD::SHL, DL, VT, Lo, SetShiftLeft1);
  SDValue Cond = DAG.getNode(ISD::AND, DL, MVT::i32, Shamt,
                             DAG.getConstant(VT.getSizeInBits(), DL, MVT::i32));
  Lo = DAG.getNode(ISD::SELECT, DL, VT, Cond, DAG.getConstant(0, DL, VT),
                   ShiftLeftLo);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, Cond, ShiftLeftLo, ShiftLeftHi);

  SDValue Ops[2] = {Lo, Hi};
  return DAG.getMergeValues(Ops, DL);
}

SDValue XtensaTargetLowering::lowerShiftRightParts(SDValue Op,
                                                   SelectionDAG &DAG,
                                                   bool IsSRA) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0), Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  MVT VT = MVT::i32;

  if (IsSRA) {
    SDValue SetShiftRight1 = DAG.getNode(XtensaISD::SSR, DL, MVT::Glue, Shamt);
    SDValue ShiftRightLo1 =
        DAG.getNode(XtensaISD::SRC, DL, VT, Hi, Lo, SetShiftRight1);

    SDValue SetShiftRight2 = DAG.getNode(XtensaISD::SSR, DL, MVT::Glue, Shamt);
    SDValue ShiftRightHi1 =
        DAG.getNode(XtensaISD::SRA, DL, VT, Hi, SetShiftRight2);

    SDValue SetShiftRight3 = DAG.getNode(XtensaISD::SSR, DL, MVT::Glue, Shamt);
    SDValue ShiftRightLo2 =
        DAG.getNode(XtensaISD::SRA, DL, VT, Hi, SetShiftRight3);

    SDValue ShiftRightHi2 =
        DAG.getNode(ISD::SRA, DL, VT, Hi, DAG.getConstant(31, DL, VT));

    SDValue Cond =
        DAG.getNode(ISD::AND, DL, MVT::i32, Shamt,
                    DAG.getConstant(VT.getSizeInBits(), DL, MVT::i32));
    Hi = DAG.getNode(ISD::SELECT, DL, VT, Cond, ShiftRightHi2, ShiftRightHi1);
    Lo = DAG.getNode(ISD::SELECT, DL, VT, Cond, ShiftRightLo2, ShiftRightLo1);
  } else {
    SDValue SetShiftRight1 = DAG.getNode(XtensaISD::SSR, DL, MVT::Glue, Shamt);
    SDValue ShiftRightLo1 =
        DAG.getNode(XtensaISD::SRC, DL, VT, Hi, Lo, SetShiftRight1);

    SDValue SetShiftRight2 = DAG.getNode(XtensaISD::SSR, DL, MVT::Glue, Shamt);
    SDValue ShiftRightHi1 =
        DAG.getNode(XtensaISD::SRL, DL, VT, Hi, SetShiftRight2);

    SDValue SetShiftRight3 = DAG.getNode(XtensaISD::SSR, DL, MVT::Glue, Shamt);
    SDValue ShiftRightLo2 =
        DAG.getNode(XtensaISD::SRL, DL, VT, Hi, SetShiftRight3);

    SDValue Cond =
        DAG.getNode(ISD::AND, DL, MVT::i32, Shamt,
                    DAG.getConstant(VT.getSizeInBits(), DL, MVT::i32));
    Hi = DAG.getNode(ISD::SELECT, DL, VT, Cond, DAG.getConstant(0, DL, VT),
                     ShiftRightHi1);
    Lo = DAG.getNode(ISD::SELECT, DL, VT, Cond, ShiftRightLo2, ShiftRightLo1);
  }

  SDValue Ops[2] = {Lo, Hi};
  return DAG.getMergeValues(Ops, DL);
}

SDValue XtensaTargetLowering::LowerOperation(SDValue Op,
                                             SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::BR_JT:
    return lowerBR_JT(Op, DAG);
  case ISD::Constant:
    return lowerImmediate(Op, DAG);
  case ISD::ConstantFP:
    return lowerImmediateFP(Op, DAG);
  case ISD::RETURNADDR:
    return lowerRETURNADDR(Op, DAG);
  case ISD::BR_CC:
    return lowerBR_CC(Op, DAG);
  case ISD::SETCC:
    return lowerSETCC(Op, DAG);
  case ISD::SELECT_CC:
    return lowerSELECT_CC(Op, DAG);
  case ISD::GlobalAddress:
    return lowerGlobalAddress(Op, DAG);
  case ISD::GlobalTLSAddress:
    return lowerGlobalTLSAddress(cast<GlobalAddressSDNode>(Op), DAG);
  case ISD::BlockAddress:
    return lowerBlockAddress(cast<BlockAddressSDNode>(Op), DAG);
  case ISD::JumpTable:
    return lowerJumpTable(cast<JumpTableSDNode>(Op), DAG);
  case ISD::ConstantPool:
    return lowerConstantPool(cast<ConstantPoolSDNode>(Op), DAG);
  case ISD::VASTART:
    return lowerVASTART(Op, DAG);
  case ISD::VAARG:
    return lowerVAARG(Op, DAG);
  case ISD::VACOPY:
    return lowerVACOPY(Op, DAG);
  case ISD::ATOMIC_FENCE:
    return lowerATOMIC_FENCE(Op, DAG);
  case ISD::STACKSAVE:
    return lowerSTACKSAVE(Op, DAG);
  case ISD::STACKRESTORE:
    return lowerSTACKRESTORE(Op, DAG);
  case ISD::FRAMEADDR:
    return lowerFRAMEADDR(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC:
    return lowerDYNAMIC_STACKALLOC(Op, DAG);
  case ISD::SHL_PARTS:
    return lowerShiftLeftParts(Op, DAG);
  case ISD::SRA_PARTS:
    return lowerShiftRightParts(Op, DAG, true);
  case ISD::SRL_PARTS:
    return lowerShiftRightParts(Op, DAG, false);
  default:
    llvm_unreachable("Unexpected node to lower");
  }
}

const char *XtensaTargetLowering::getTargetNodeName(unsigned Opcode) const {
#define OPCODE(NAME)                                                           \
  case XtensaISD::NAME:                                                        \
    return "XtensaISD::" #NAME
  switch (Opcode) {
    OPCODE(RET_FLAG);
    OPCODE(RETW_FLAG);
    OPCODE(CALL);
    OPCODE(CALLW);
    OPCODE(PCREL_WRAPPER);
    OPCODE(SELECT);
    OPCODE(SELECT_CC);
    OPCODE(SELECT_CC_FP);
    OPCODE(BR_CC_T);
    OPCODE(BR_CC_F);
    OPCODE(BR_JT);
    OPCODE(CMPUO);
    OPCODE(CMPUEQ);
    OPCODE(CMPULE);
    OPCODE(CMPULT);
    OPCODE(CMPOEQ);
    OPCODE(CMPOLE);
    OPCODE(CMPOLT);
    OPCODE(MOVT);
    OPCODE(MOVF);
    OPCODE(MADD);
    OPCODE(MSUB);
    OPCODE(MOVS);
    OPCODE(MOVSP);
    OPCODE(SHL);
    OPCODE(SRA);
    OPCODE(SRL);
    OPCODE(SRC);
    OPCODE(SSL);
    OPCODE(SSR);
    OPCODE(MEMW);
    OPCODE(S32C1I);
    OPCODE(WSR);
    OPCODE(RUR);
  }
  return NULL;
#undef OPCODE
}

//===----------------------------------------------------------------------===//
// Custom insertion
//===----------------------------------------------------------------------===//

// Call pseduo ops for ABI compliant calls (output is always ra)
MachineBasicBlock *XtensaTargetLowering::emitCALL(MachineInstr *MI,
                                                  MachineBasicBlock *BB) const {
  // TODO
  return BB;
}

static int GetBranchKind(int Cond, bool &BrInv) {
  switch (Cond) {
  case ISD::SETEQ:
  case ISD::SETOEQ:
  case ISD::SETUEQ:
    return Xtensa::BEQ;
  case ISD::SETNE:
  case ISD::SETONE:
  case ISD::SETUNE:
    return Xtensa::BNE;
  case ISD::SETLT:
  case ISD::SETOLT:
    return Xtensa::BLT;
  case ISD::SETLE:
  case ISD::SETOLE:
    BrInv = true;
    return Xtensa::BGE;
  case ISD::SETGT:
  case ISD::SETOGT:
    BrInv = true;
    return Xtensa::BLT;
  case ISD::SETGE:
  case ISD::SETOGE:
    return Xtensa::BGE;
  case ISD::SETULT:
    return Xtensa::BLTU;
  case ISD::SETULE:
    BrInv = true;
    return Xtensa::BGEU;
  case ISD::SETUGT:
    BrInv = true;
    return Xtensa::BLTU;
  case ISD::SETUGE:
    return Xtensa::BGEU;
  default:
    return -1;
  }
}

static void GetFPBranchKind(int Cond, int &BrKind, int &CmpKind) {

  switch (Cond) {
  default:
    llvm_unreachable("Invalid condition!");
    break;
  case ISD::SETUNE:
    BrKind = Xtensa::BF;
    CmpKind = Xtensa::OEQ_S;
    break;
  case ISD::SETUO:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::UN_S;
    break;
  case ISD::SETO:
    BrKind = Xtensa::BF;
    CmpKind = Xtensa::UN_S;
    break;
  case ISD::SETUEQ:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::UEQ_S;
    break;
  case ISD::SETULE:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::ULE_S;
    break;
  case ISD::SETULT:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::ULT_S;
    break;
  case ISD::SETEQ:
  case ISD::SETOEQ:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::OEQ_S;
    break;
  case ISD::SETNE:
    BrKind = Xtensa::BF;
    CmpKind = Xtensa::OEQ_S;
    break;
  case ISD::SETLE:
  case ISD::SETOLE:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::OLE_S;
    break;
  case ISD::SETLT:
  case ISD::SETOLT:
    BrKind = Xtensa::BT;
    CmpKind = Xtensa::OLT_S;
    break;
  case ISD::SETGE:
    BrKind = Xtensa::BF;
    CmpKind = Xtensa::OLT_S;
    break;
  case ISD::SETGT:
    BrKind = Xtensa::BF;
    CmpKind = Xtensa::OLE_S;
    break;
  }
}

MachineBasicBlock *
XtensaTargetLowering::emitSelectCC(MachineInstr &MI,
                                   MachineBasicBlock *BB) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  MachineOperand &LHS = MI.getOperand(1);
  MachineOperand &RHS = MI.getOperand(2);
  MachineOperand &TrueV = MI.getOperand(3);
  MachineOperand &FalseV = MI.getOperand(4);
  MachineOperand &Cond = MI.getOperand(5);

  // To "insert" a SELECT_CC instruction, we actually have to insert the
  // diamond control-flow pattern.  The incoming instruction knows the
  // destination vreg to set, the condition code register to branch on, the
  // true/false values to select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  //  thisMBB:
  //  ...
  //   TrueVal = ...
  //   cmpTY ccX, r1, r2
  //   bCC copy1MBB
  //   fallthrough --> copy0MBB
  MachineBasicBlock *thisMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *copy0MBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(It, copy0MBB);
  F->insert(It, sinkMBB);

  // Transfer the remainder of BB and its successor edges to sinkMBB.
  sinkMBB->splice(sinkMBB->begin(), BB,
                  std::next(MachineBasicBlock::iterator(MI)), BB->end());
  sinkMBB->transferSuccessorsAndUpdatePHIs(BB);

  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(sinkMBB);

  if ((MI.getOpcode() == Xtensa::SELECT_CC_FP_FP) ||
      (MI.getOpcode() == Xtensa::SELECT_CC_FP_INT)) {
    int BrKind = 0;
    int CmpKind = 0;
    MachineFunction *MF = BB->getParent();
    MachineRegisterInfo &RegInfo = MF->getRegInfo();
    const TargetRegisterClass *RC = getRegClassFor(MVT::i1);
    unsigned b = RegInfo.createVirtualRegister(RC);
    GetFPBranchKind(Cond.getImm(), BrKind, CmpKind);
    BuildMI(BB, DL, TII.get(CmpKind), b)
        .addReg(LHS.getReg())
        .addReg(RHS.getReg());
    BuildMI(BB, DL, TII.get(BrKind)).addReg(b).addMBB(sinkMBB);
  } else {
    /*int BrKind = GetBranchKind(Cond.getImm());
    BuildMI(BB, DL, TII.get(BrKind))
        .addReg(LHS.getReg())
        .addReg(RHS.getReg())
        .addMBB(sinkMBB);*/
    bool BrInv = false;
    int BrKind = GetBranchKind(Cond.getImm(), BrInv);
    if (BrInv) {
      BuildMI(BB, DL, TII.get(BrKind))
          .addReg(RHS.getReg())
          .addReg(LHS.getReg())
          .addMBB(sinkMBB);
    } else {
      BuildMI(BB, DL, TII.get(BrKind))
          .addReg(LHS.getReg())
          .addReg(RHS.getReg())
          .addMBB(sinkMBB);
    }
  }

  //  copy0MBB:
  //   %FalseValue = ...
  //   # fallthrough to sinkMBB
  BB = copy0MBB;

  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  //  sinkMBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
  //  ...
  BB = sinkMBB;

  BuildMI(*BB, BB->begin(), DL, TII.get(Xtensa::PHI), MI.getOperand(0).getReg())
      .addReg(FalseV.getReg())
      .addMBB(copy0MBB)
      .addReg(TrueV.getReg())
      .addMBB(thisMBB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

// Emit instructions for atomic_cmp_swap node for 8/16 bit operands
MachineBasicBlock *
XtensaTargetLowering::emitAtomicCmpSwap(MachineInstr &MI, MachineBasicBlock *BB,
                                        int isByteOperand) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineBasicBlock *thisBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *BBLoop = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *BBExit = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(It, BBLoop);
  F->insert(It, BBExit);

  // Transfer the remainder of BB and its successor edges to BBExit.
  BBExit->splice(BBExit->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  BBExit->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(BBLoop);

  MachineOperand &Res = MI.getOperand(0);
  MachineOperand &AtomValAddr = MI.getOperand(1);
  MachineOperand &CmpVal = MI.getOperand(2);
  MachineOperand &SwpVal = MI.getOperand(3);

  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetRegisterClass *RC = getRegClassFor(MVT::i32);

  unsigned r1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), r1).addImm(3);

  unsigned byte_offs = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::AND), byte_offs)
      .addReg(r1)
      .addReg(AtomValAddr.getReg());

  unsigned addr_align = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SUB), addr_align)
      .addReg(AtomValAddr.getReg())
      .addReg(byte_offs);

  unsigned bit_offs = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLLI), bit_offs)
      .addReg(byte_offs)
      .addImm(3);

  unsigned mask1 = MRI.createVirtualRegister(RC);
  if (isByteOperand) {
    BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), mask1).addImm(0xff);
  } else {
    unsigned r2 = MRI.createVirtualRegister(RC);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), r2).addImm(1);
    unsigned r3 = MRI.createVirtualRegister(RC);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::SLLI), r3).addReg(r2).addImm(16);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::ADDI), mask1).addReg(r3).addImm(-1);
  }

  BuildMI(*BB, MI, DL, TII.get(Xtensa::SSL)).addReg(bit_offs);

  unsigned r2 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), r2).addImm(-1);

  unsigned mask2 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLL), mask2).addReg(mask1);

  unsigned mask3 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::XOR), mask3).addReg(mask2).addReg(r2);

  unsigned r3 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::L32I), r3).addReg(addr_align).addImm(0);

  unsigned r4 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::AND), r4).addReg(r3).addReg(mask3);

  unsigned cmp1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLL), cmp1).addReg(CmpVal.getReg());

  unsigned swp1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLL), swp1).addReg(SwpVal.getReg());

  BB = BBLoop;

  unsigned maskPhi = MRI.createVirtualRegister(RC);
  unsigned maskLoop = MRI.createVirtualRegister(RC);

  BuildMI(*BB, BB->begin(), DL, TII.get(Xtensa::PHI), maskPhi)
      .addReg(maskLoop)
      .addMBB(BBLoop)
      .addReg(r4)
      .addMBB(thisBB);

  unsigned cmp2 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::OR), cmp2).addReg(cmp1).addReg(maskPhi);

  unsigned swp2 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::OR), swp2).addReg(swp1).addReg(maskPhi);

  BuildMI(BB, DL, TII.get(Xtensa::WSR), Xtensa::SCOMPARE1).addReg(cmp2);

  unsigned swp3 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::S32C1I), swp3)
      .addReg(swp2)
      .addReg(addr_align)
      .addImm(0);

  BuildMI(BB, DL, TII.get(Xtensa::AND), maskLoop).addReg(swp3).addReg(mask3);

  BuildMI(BB, DL, TII.get(Xtensa::BNE))
      .addReg(maskLoop)
      .addReg(maskPhi)
      .addMBB(BBLoop);

  BB->addSuccessor(BBLoop);
  BB->addSuccessor(BBExit);

  BB = BBExit;
  auto st = BBExit->begin();

  unsigned r5 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, st, DL, TII.get(Xtensa::SSR)).addReg(bit_offs);

  BuildMI(*BB, st, DL, TII.get(Xtensa::SRL), r5).addReg(swp3);

  BuildMI(*BB, st, DL, TII.get(Xtensa::AND), Res.getReg())
      .addReg(r5)
      .addReg(mask1);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

MachineBasicBlock *XtensaTargetLowering::emitAtomicRMW(MachineInstr &MI,
                                                       MachineBasicBlock *BB,
                                                       unsigned Opcode,
                                                       bool inv,
                                                       bool minmax) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineBasicBlock *thisBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *BBLoop = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *BBExit = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(It, BBLoop);
  F->insert(It, BBExit);

  // Transfer the remainder of BB and its successor edges to BB2.
  BBExit->splice(BBExit->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  BBExit->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(BBLoop);

  MachineOperand &Res = MI.getOperand(0);
  MachineOperand &AtomicValAddr = MI.getOperand(1);
  MachineOperand &Val = MI.getOperand(2);
  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetRegisterClass *RC = getRegClassFor(MVT::i32);

  unsigned r1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::L32I), r1).add(AtomicValAddr).addImm(0);

  BB = BBLoop;

  unsigned atomicValPhi = MRI.createVirtualRegister(RC);
  unsigned atomicValLoop = MRI.createVirtualRegister(RC);

  BuildMI(*BB, BB->begin(), DL, TII.get(Xtensa::PHI), atomicValPhi)
      .addReg(atomicValLoop)
      .addMBB(BBLoop)
      .addReg(r1)
      .addMBB(thisBB);

  unsigned r2 = MRI.createVirtualRegister(RC);

  if (minmax) {
    MachineBasicBlock *BBLoop1 = F->CreateMachineBasicBlock(LLVM_BB);
    F->insert(It, BBLoop1);
    BB->addSuccessor(BBLoop1);
    MachineBasicBlock *BBLoop2 = F->CreateMachineBasicBlock(LLVM_BB);
    F->insert(It, BBLoop2);
    BB->addSuccessor(BBLoop2);

    BuildMI(BB, DL, TII.get(Opcode))
        .addReg(atomicValPhi)
        .addReg(Val.getReg())
        .addMBB(BBLoop1);

    unsigned r7 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::MOV_N), r7).addReg(Val.getReg());

    BB = BBLoop1;
    unsigned r8 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::MOV_N), r8).addReg(atomicValPhi);
    BB->addSuccessor(BBLoop2);

    BB = BBLoop2;
    unsigned r9 = MRI.createVirtualRegister(RC);

    BuildMI(*BB, BB->begin(), DL, TII.get(Xtensa::PHI), r9)
        .addReg(r7)
        .addMBB(BBLoop)
        .addReg(r8)
        .addMBB(BBLoop1);
    BuildMI(BB, DL, TII.get(Xtensa::MOV_N), r2).addReg(r9);
  } else {
    BuildMI(BB, DL, TII.get(Opcode), r2)
        .addReg(atomicValPhi)
        .addReg(Val.getReg());
    if (inv) {
      unsigned rtmp1 = MRI.createVirtualRegister(RC);
      BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), rtmp1).addImm(-1);
      unsigned rtmp2 = MRI.createVirtualRegister(RC);
      BuildMI(*BB, MI, DL, TII.get(Xtensa::XOR), rtmp2)
          .addReg(r2)
          .addReg(rtmp1);
      r2 = rtmp2;
    }
  }

  unsigned r4 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::WSR), Xtensa::SCOMPARE1).addReg(atomicValPhi);
  BuildMI(BB, DL, TII.get(Xtensa::S32C1I), r4)
      .addReg(r2)
      .addReg(AtomicValAddr.getReg())
      .addImm(0);

  BuildMI(BB, DL, TII.get(Xtensa::MOV_N), atomicValLoop).addReg(r4);

  BuildMI(BB, DL, TII.get(Xtensa::BNE))
      .addReg(atomicValPhi)
      .addReg(r4)
      .addMBB(BBLoop);

  BB->addSuccessor(BBLoop);
  BB->addSuccessor(BBExit);

  BB = BBExit;
  auto st = BBExit->begin();

  BuildMI(*BB, st, DL, TII.get(Xtensa::MOV_N), Res.getReg()).addReg(r4);

  MI.eraseFromParent(); // The pseudo instruction is gone now.

  return BB;
}

MachineBasicBlock *
XtensaTargetLowering::emitAtomicRMW(MachineInstr &MI, MachineBasicBlock *BB,
                                    bool isByteOperand, unsigned Opcode,
                                    bool inv, bool minmax) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineBasicBlock *thisBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *BBLoop = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *BBExit = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(It, BBLoop);
  F->insert(It, BBExit);

  // Transfer the remainder of BB and its successor edges to BB2.
  BBExit->splice(BBExit->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  BBExit->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(BBLoop);

  MachineOperand &Res = MI.getOperand(0);
  MachineOperand &AtomValAddr = MI.getOperand(1);
  MachineOperand &Val = MI.getOperand(2);

  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetRegisterClass *RC = getRegClassFor(MVT::i32);

  unsigned r1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), r1).addImm(3);

  unsigned byte_offs = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::AND), byte_offs)
      .addReg(r1)
      .addReg(AtomValAddr.getReg());

  unsigned addr_align = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SUB), addr_align)
      .addReg(AtomValAddr.getReg())
      .addReg(byte_offs);

  unsigned bit_offs = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLLI), bit_offs)
      .addReg(byte_offs)
      .addImm(3);

  unsigned mask1 = MRI.createVirtualRegister(RC);
  if (isByteOperand) {
    BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), mask1).addImm(0xff);
  } else {
    unsigned r2 = MRI.createVirtualRegister(RC);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), r2).addImm(1);
    unsigned r3 = MRI.createVirtualRegister(RC);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::SLLI), r3).addReg(r2).addImm(16);
    BuildMI(*BB, MI, DL, TII.get(Xtensa::ADDI), mask1).addReg(r3).addImm(-1);
  }

  BuildMI(*BB, MI, DL, TII.get(Xtensa::SSL)).addReg(bit_offs);

  unsigned r2 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::MOVI), r2).addImm(-1);

  unsigned mask2 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLL), mask2).addReg(mask1);

  unsigned mask3 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::XOR), mask3).addReg(mask2).addReg(r2);

  unsigned r3 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::L32I), r3).addReg(addr_align).addImm(0);

  unsigned val1 = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(Xtensa::SLL), val1).addReg(Val.getReg());

  BB = BBLoop;

  unsigned atomicValPhi = MRI.createVirtualRegister(RC);
  unsigned atomicValLoop = MRI.createVirtualRegister(RC);

  BuildMI(*BB, BB->begin(), DL, TII.get(Xtensa::PHI), atomicValPhi)
      .addReg(atomicValLoop)
      .addMBB(BBLoop)
      .addReg(r3)
      .addMBB(thisBB);

  unsigned swp2;

  if (minmax) {
    MachineBasicBlock *BBLoop1 = F->CreateMachineBasicBlock(LLVM_BB);
    F->insert(It, BBLoop1);
    BB->addSuccessor(BBLoop1);
    MachineBasicBlock *BBLoop2 = F->CreateMachineBasicBlock(LLVM_BB);
    F->insert(It, BBLoop2);
    BB->addSuccessor(BBLoop2);

    unsigned r1 = MRI.createVirtualRegister(RC);
    unsigned r2 = MRI.createVirtualRegister(RC);
    unsigned r3 = MRI.createVirtualRegister(RC);
    unsigned r4 = MRI.createVirtualRegister(RC);

    unsigned r5 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::AND), r5)
        .addReg(atomicValPhi)
        .addReg(mask2);

    BuildMI(BB, DL, TII.get(Xtensa::SRL), r1).addReg(r5);
    BuildMI(BB, DL, TII.get(Xtensa::SRL), r2).addReg(val1);

    if ((Opcode == Xtensa::BLT) || (Opcode == Xtensa::BGE)) {
      if (isByteOperand) {
        BuildMI(BB, DL, TII.get(Xtensa::SEXT), r3).addReg(r1).addImm(7);
        BuildMI(BB, DL, TII.get(Xtensa::SEXT), r4).addReg(r2).addImm(7);
      } else {
        BuildMI(BB, DL, TII.get(Xtensa::SEXT), r3).addReg(r1).addImm(15);
        BuildMI(BB, DL, TII.get(Xtensa::SEXT), r4).addReg(r2).addImm(15);
      }
    } else {
      r3 = r1;
      r4 = r2;
    }

    BuildMI(BB, DL, TII.get(Opcode)).addReg(r3).addReg(r4).addMBB(BBLoop1);

    unsigned r7 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::MOV_N), r7).addReg(val1);

    BB = BBLoop1;
    unsigned r8 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::MOV_N), r8).addReg(atomicValPhi);
    BB->addSuccessor(BBLoop2);

    BB = BBLoop2;
    unsigned r9 = MRI.createVirtualRegister(RC);

    BuildMI(*BB, BB->begin(), DL, TII.get(Xtensa::PHI), r9)
        .addReg(r7)
        .addMBB(BBLoop)
        .addReg(r8)
        .addMBB(BBLoop1);

    unsigned r10 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::AND), r10)
        .addReg(atomicValPhi)
        .addReg(mask3);

    unsigned r11 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::AND), r11).addReg(r9).addReg(mask2);

    swp2 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::OR), swp2).addReg(r10).addReg(r11);
  } else {
    unsigned r4 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::AND), r4)
        .addReg(atomicValPhi)
        .addReg(mask2);

    unsigned res1 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Opcode), res1).addReg(r4).addReg(val1);

    unsigned swp1 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::AND), swp1).addReg(res1).addReg(mask2);

    unsigned r5 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::AND), r5)
        .addReg(atomicValPhi)
        .addReg(mask3);

    if (inv) {
      unsigned rtmp1 = MRI.createVirtualRegister(RC);
      BuildMI(BB, DL, TII.get(Xtensa::XOR), rtmp1)
          .addReg(atomicValPhi)
          .addReg(mask2);
      r5 = rtmp1;
    }

    swp2 = MRI.createVirtualRegister(RC);
    BuildMI(BB, DL, TII.get(Xtensa::OR), swp2).addReg(swp1).addReg(r5);
  }

  unsigned swp3 = MRI.createVirtualRegister(RC);
  BuildMI(BB, DL, TII.get(Xtensa::WSR), Xtensa::SCOMPARE1).addReg(atomicValPhi);
  BuildMI(BB, DL, TII.get(Xtensa::S32C1I), swp3)
      .addReg(swp2)
      .addReg(addr_align)
      .addImm(0);

  BuildMI(BB, DL, TII.get(Xtensa::MOV_N), atomicValLoop).addReg(swp3);

  BuildMI(BB, DL, TII.get(Xtensa::BNE))
      .addReg(swp3)
      .addReg(atomicValPhi)
      .addMBB(BBLoop);

  BB->addSuccessor(BBLoop);
  BB->addSuccessor(BBExit);
  BB = BBExit;
  auto st = BBExit->begin();

  unsigned r6 = MRI.createVirtualRegister(RC);

  BuildMI(*BB, st, DL, TII.get(Xtensa::SSR)).addReg(bit_offs);

  BuildMI(*BB, st, DL, TII.get(Xtensa::SRL), r6).addReg(atomicValLoop);

  BuildMI(*BB, st, DL, TII.get(Xtensa::AND), Res.getReg())
      .addReg(r6)
      .addReg(mask1);

  MI.eraseFromParent(); // The pseudo instruction is gone now.

  return BB;
}

MachineBasicBlock *XtensaTargetLowering::EmitInstrWithCustomInserter(
    MachineInstr &MI, MachineBasicBlock *MBB) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  MachineFunction *MF = MBB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  DebugLoc DL = MI.getDebugLoc();

  switch (MI.getOpcode()) {
  case Xtensa::SELECT_CC_FP_FP:
  case Xtensa::SELECT_CC_FP_INT:
  case Xtensa::SELECT_CC_INT_FP:
  case Xtensa::SELECT:
    return emitSelectCC(MI, MBB);

  case Xtensa::SLL_P: {
    MachineOperand &R = MI.getOperand(0);
    MachineOperand &S = MI.getOperand(1);
    MachineOperand &SA = MI.getOperand(2);

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::SSL)).addReg(SA.getReg());
    BuildMI(*MBB, MI, DL, TII.get(Xtensa::SLL), R.getReg()).addReg(S.getReg());
    MI.eraseFromParent();
    return MBB;
  }

  case Xtensa::SRA_P: {
    MachineOperand &R = MI.getOperand(0);
    MachineOperand &T = MI.getOperand(1);
    MachineOperand &SA = MI.getOperand(2);

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::SSR)).addReg(SA.getReg());
    BuildMI(*MBB, MI, DL, TII.get(Xtensa::SRA), R.getReg()).addReg(T.getReg());
    MI.eraseFromParent();
    return MBB;
  }

  case Xtensa::SRL_P: {
    MachineOperand &R = MI.getOperand(0);
    MachineOperand &T = MI.getOperand(1);
    MachineOperand &SA = MI.getOperand(2);

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::SSR)).addReg(SA.getReg());
    BuildMI(*MBB, MI, DL, TII.get(Xtensa::SRL), R.getReg()).addReg(T.getReg());
    MI.eraseFromParent();
    return MBB;
  }

  case Xtensa::L8I_P: {
    MachineOperand &R = MI.getOperand(0);
    MachineOperand &Op1 = MI.getOperand(1);
    MachineOperand &Op2 = MI.getOperand(2);

    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned r_new = MRI.createVirtualRegister(RC);

    const MachineMemOperand &MMO = **MI.memoperands_begin();
    if (MMO.isVolatile()) {
      BuildMI(*MBB, MI, DL, TII.get(Xtensa::MEMW));
    }

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::L8UI), r_new).add(Op1).add(Op2);
    if (Subtarget.hasSEXT()) {
      BuildMI(*MBB, MI, DL, TII.get(Xtensa::SEXT), R.getReg())
          .addReg(r_new)
          .addImm(7);
    } else {
      unsigned r_new1 = MRI.createVirtualRegister(RC);
      BuildMI(*MBB, MI, DL, TII.get(Xtensa::SLLI), r_new1)
          .addReg(r_new)
          .addImm(24);
      BuildMI(*MBB, MI, DL, TII.get(Xtensa::SRAI), R.getReg())
          .addReg(r_new1)
          .addImm(24);
    }
    MI.eraseFromParent();
    return MBB;
  }

  case Xtensa::ATOMIC_CMP_SWAP_8_P: {
    return emitAtomicCmpSwap(MI, MBB, 1);
  }

  case Xtensa::ATOMIC_CMP_SWAP_16_P: {
    return emitAtomicCmpSwap(MI, MBB, 0);
  }

  case Xtensa::ATOMIC_CMP_SWAP_32_P: {
    MachineOperand &R = MI.getOperand(0);
    MachineOperand &Addr = MI.getOperand(1);
    MachineOperand &Cmp = MI.getOperand(2);
    MachineOperand &Swap = MI.getOperand(3);

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::WSR), Xtensa::SCOMPARE1)
        .addReg(Cmp.getReg());

    BuildMI(*MBB, MI, DL, TII.get(Xtensa::S32C1I), R.getReg())
        .addReg(Swap.getReg())
        .addReg(Addr.getReg())
        .addImm(0);

    MI.eraseFromParent();
    return MBB;
  }

  case Xtensa::ATOMIC_LOAD_ADD_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::ADD, false, false);
  case Xtensa::ATOMIC_LOAD_SUB_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::SUB, false, false);
  case Xtensa::ATOMIC_LOAD_OR_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::OR, false, false);
  case Xtensa::ATOMIC_LOAD_XOR_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::XOR, false, false);
  case Xtensa::ATOMIC_LOAD_AND_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::AND, false, false);
  case Xtensa::ATOMIC_LOAD_NAND_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::AND, true, false);
  case Xtensa::ATOMIC_LOAD_MIN_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::BGE, false, true);
  case Xtensa::ATOMIC_LOAD_MAX_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::BLT, false, true);
  case Xtensa::ATOMIC_LOAD_UMIN_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::BGEU, false, true);
  case Xtensa::ATOMIC_LOAD_UMAX_8_P:
    return emitAtomicRMW(MI, MBB, true, Xtensa::BLTU, false, true);

  case Xtensa::ATOMIC_LOAD_ADD_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::ADD, false, false);
  case Xtensa::ATOMIC_LOAD_SUB_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::SUB, false, false);
  case Xtensa::ATOMIC_LOAD_OR_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::OR, false, false);
  case Xtensa::ATOMIC_LOAD_XOR_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::XOR, false, false);
  case Xtensa::ATOMIC_LOAD_AND_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::AND, false, false);
  case Xtensa::ATOMIC_LOAD_NAND_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::AND, true, false);
  case Xtensa::ATOMIC_LOAD_MIN_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::BGE, false, true);
  case Xtensa::ATOMIC_LOAD_MAX_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::BLT, false, true);
  case Xtensa::ATOMIC_LOAD_UMIN_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::BGEU, false, true);
  case Xtensa::ATOMIC_LOAD_UMAX_16_P:
    return emitAtomicRMW(MI, MBB, false, Xtensa::BLTU, false, true);

  case Xtensa::ATOMIC_LOAD_ADD_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::ADD, false, false);
  case Xtensa::ATOMIC_LOAD_SUB_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::SUB, false, false);
  case Xtensa::ATOMIC_LOAD_OR_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::OR, false, false);
  case Xtensa::ATOMIC_LOAD_XOR_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::XOR, false, false);
  case Xtensa::ATOMIC_LOAD_AND_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::AND, false, false);
  case Xtensa::ATOMIC_LOAD_NAND_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::AND, true, false);
  case Xtensa::ATOMIC_LOAD_MIN_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::BGE, false, true);
  case Xtensa::ATOMIC_LOAD_MAX_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::BLT, false, true);
  case Xtensa::ATOMIC_LOAD_UMIN_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::BGEU, false, true);
  case Xtensa::ATOMIC_LOAD_UMAX_32_P:
    return emitAtomicRMW(MI, MBB, Xtensa::BLTU, false, true);
  case Xtensa::S8I:
  case Xtensa::S16I:
  case Xtensa::S32I:
  case Xtensa::S32I_N:
  case Xtensa::S32F:
  case Xtensa::L8UI:
  case Xtensa::L16SI:
  case Xtensa::L16UI:
  case Xtensa::L32I:
  case Xtensa::L32I_N:
  case Xtensa::L32F: {
    const MachineMemOperand &MMO = **MI.memoperands_begin();
    if (MMO.isVolatile()) {
      BuildMI(*MBB, MI, DL, TII.get(Xtensa::MEMW));
    }
    return MBB;
  }
  default:
    llvm_unreachable("Unexpected instr type to insert");
  }
}
