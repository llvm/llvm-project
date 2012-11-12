//===-------- LegalizeFloatTypes.cpp - Legalization of float types --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements float type expansion and softening for LegalizeTypes.
// Softening is the act of turning a computation in an illegal floating point
// type into a computation in an integer type of the same size; also known as
// "soft float".  For example, turning f32 arithmetic into operations using i32.
// The resulting integer value is the same as what you would get by performing
// the floating point operation and bitcasting the result to the integer type.
// Expansion is the act of changing a computation in an illegal type to be a
// computation in two identical registers of a smaller type.  For example,
// implementing ppcf128 arithmetic in two f64 registers.
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

/// GetFPLibCall - Return the right libcall for the given floating point type.
static RTLIB::Libcall GetFPLibCall(EVT VT,
                                   RTLIB::Libcall Call_F32,
                                   RTLIB::Libcall Call_F64,
                                   RTLIB::Libcall Call_F80,
                                   RTLIB::Libcall Call_PPCF128) {
  return
    VT == MVT::f32 ? Call_F32 :
    VT == MVT::f64 ? Call_F64 :
    VT == MVT::f80 ? Call_F80 :
    VT == MVT::ppcf128 ? Call_PPCF128 :
    RTLIB::UNKNOWN_LIBCALL;
}

//===----------------------------------------------------------------------===//
//  Result Float to Integer Conversion.
//===----------------------------------------------------------------------===//

void DAGTypeLegalizer::SoftenFloatResult(SDNode *N, unsigned ResNo) {
  DEBUG(dbgs() << "Soften float result " << ResNo << ": "; N->dump(&DAG);
        dbgs() << "\n");
  SDValue R = SDValue();

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    dbgs() << "SoftenFloatResult #" << ResNo << ": ";
    N->dump(&DAG); dbgs() << "\n";
#endif
    llvm_unreachable("Do not know how to soften the result of this operator!");

    case ISD::MERGE_VALUES:R = SoftenFloatRes_MERGE_VALUES(N, ResNo); break;
    case ISD::BITCAST:     R = SoftenFloatRes_BITCAST(N); break;
    case ISD::BUILD_PAIR:  R = SoftenFloatRes_BUILD_PAIR(N); break;
    case ISD::ConstantFP:
      R = SoftenFloatRes_ConstantFP(cast<ConstantFPSDNode>(N));
      break;
    case ISD::EXTRACT_VECTOR_ELT:
      R = SoftenFloatRes_EXTRACT_VECTOR_ELT(N); break;
    case ISD::FABS:        R = SoftenFloatRes_FABS(N); break;
    case ISD::FADD:        R = SoftenFloatRes_FADD(N); break;
    case ISD::FCEIL:       R = SoftenFloatRes_FCEIL(N); break;
    case ISD::FCOPYSIGN:   R = SoftenFloatRes_FCOPYSIGN(N); break;
    case ISD::FCOS:        R = SoftenFloatRes_FCOS(N); break;
    case ISD::FDIV:        R = SoftenFloatRes_FDIV(N); break;
    case ISD::FEXP:        R = SoftenFloatRes_FEXP(N); break;
    case ISD::FEXP2:       R = SoftenFloatRes_FEXP2(N); break;
    case ISD::FFLOOR:      R = SoftenFloatRes_FFLOOR(N); break;
    case ISD::FLOG:        R = SoftenFloatRes_FLOG(N); break;
    case ISD::FLOG2:       R = SoftenFloatRes_FLOG2(N); break;
    case ISD::FLOG10:      R = SoftenFloatRes_FLOG10(N); break;
    case ISD::FMA:         R = SoftenFloatRes_FMA(N); break;
    case ISD::FMUL:        R = SoftenFloatRes_FMUL(N); break;
    case ISD::FNEARBYINT:  R = SoftenFloatRes_FNEARBYINT(N); break;
    case ISD::FNEG:        R = SoftenFloatRes_FNEG(N); break;
    case ISD::FP_EXTEND:   R = SoftenFloatRes_FP_EXTEND(N); break;
    case ISD::FP_ROUND:    R = SoftenFloatRes_FP_ROUND(N); break;
    case ISD::FP16_TO_FP32:R = SoftenFloatRes_FP16_TO_FP32(N); break;
    case ISD::FPOW:        R = SoftenFloatRes_FPOW(N); break;
    case ISD::FPOWI:       R = SoftenFloatRes_FPOWI(N); break;
    case ISD::FREM:        R = SoftenFloatRes_FREM(N); break;
    case ISD::FRINT:       R = SoftenFloatRes_FRINT(N); break;
    case ISD::FSIN:        R = SoftenFloatRes_FSIN(N); break;
    case ISD::FSQRT:       R = SoftenFloatRes_FSQRT(N); break;
    case ISD::FSUB:        R = SoftenFloatRes_FSUB(N); break;
    case ISD::FTRUNC:      R = SoftenFloatRes_FTRUNC(N); break;
    case ISD::LOAD:        R = SoftenFloatRes_LOAD(N); break;
    case ISD::SELECT:      R = SoftenFloatRes_SELECT(N); break;
    case ISD::SELECT_CC:   R = SoftenFloatRes_SELECT_CC(N); break;
    case ISD::SINT_TO_FP:
    case ISD::UINT_TO_FP:  R = SoftenFloatRes_XINT_TO_FP(N); break;
    case ISD::UNDEF:       R = SoftenFloatRes_UNDEF(N); break;
    case ISD::VAARG:       R = SoftenFloatRes_VAARG(N); break;
  }

  // If R is null, the sub-method took care of registering the result.
  if (R.getNode())
    SetSoftenedFloat(SDValue(N, ResNo), R);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_BITCAST(SDNode *N) {
  return BitConvertToInteger(N->getOperand(0));
}

SDValue DAGTypeLegalizer::SoftenFloatRes_MERGE_VALUES(SDNode *N,
                                                      unsigned ResNo) {
  SDValue Op = DisintegrateMERGE_VALUES(N, ResNo);
  return BitConvertToInteger(Op);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_BUILD_PAIR(SDNode *N) {
  // Convert the inputs to integers, and build a new pair out of them.
  return DAG.getNode(ISD::BUILD_PAIR, N->getDebugLoc(),
                     TLI.getTypeToTransformTo(*DAG.getContext(),
                                              N->getValueType(0)),
                     BitConvertToInteger(N->getOperand(0)),
                     BitConvertToInteger(N->getOperand(1)));
}

SDValue DAGTypeLegalizer::SoftenFloatRes_ConstantFP(ConstantFPSDNode *N) {
  return DAG.getConstant(N->getValueAPF().bitcastToAPInt(),
                         TLI.getTypeToTransformTo(*DAG.getContext(),
                                                  N->getValueType(0)));
}

SDValue DAGTypeLegalizer::SoftenFloatRes_EXTRACT_VECTOR_ELT(SDNode *N) {
  SDValue NewOp = BitConvertVectorToIntegerVector(N->getOperand(0));
  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, N->getDebugLoc(),
                     NewOp.getValueType().getVectorElementType(),
                     NewOp, N->getOperand(1));
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FABS(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  unsigned Size = NVT.getSizeInBits();

  // Mask = ~(1 << (Size-1))
  APInt API = APInt::getAllOnesValue(Size);
  API.clearBit(Size-1);
  SDValue Mask = DAG.getConstant(API, NVT);
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return DAG.getNode(ISD::AND, N->getDebugLoc(), NVT, Op, Mask);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FADD(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                     GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::ADD_F32,
                                  RTLIB::ADD_F64,
                                  RTLIB::ADD_F80,
                                  RTLIB::ADD_PPCF128),
                     NVT, Ops, 2, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FCEIL(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::CEIL_F32,
                                  RTLIB::CEIL_F64,
                                  RTLIB::CEIL_F80,
                                  RTLIB::CEIL_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FCOPYSIGN(SDNode *N) {
  SDValue LHS = GetSoftenedFloat(N->getOperand(0));
  SDValue RHS = BitConvertToInteger(N->getOperand(1));
  DebugLoc dl = N->getDebugLoc();

  EVT LVT = LHS.getValueType();
  EVT RVT = RHS.getValueType();

  unsigned LSize = LVT.getSizeInBits();
  unsigned RSize = RVT.getSizeInBits();

  // First get the sign bit of second operand.
  SDValue SignBit = DAG.getNode(ISD::SHL, dl, RVT, DAG.getConstant(1, RVT),
                                  DAG.getConstant(RSize - 1,
                                                  TLI.getShiftAmountTy(RVT)));
  SignBit = DAG.getNode(ISD::AND, dl, RVT, RHS, SignBit);

  // Shift right or sign-extend it if the two operands have different types.
  int SizeDiff = RVT.getSizeInBits() - LVT.getSizeInBits();
  if (SizeDiff > 0) {
    SignBit = DAG.getNode(ISD::SRL, dl, RVT, SignBit,
                          DAG.getConstant(SizeDiff,
                                 TLI.getShiftAmountTy(SignBit.getValueType())));
    SignBit = DAG.getNode(ISD::TRUNCATE, dl, LVT, SignBit);
  } else if (SizeDiff < 0) {
    SignBit = DAG.getNode(ISD::ANY_EXTEND, dl, LVT, SignBit);
    SignBit = DAG.getNode(ISD::SHL, dl, LVT, SignBit,
                          DAG.getConstant(-SizeDiff,
                                 TLI.getShiftAmountTy(SignBit.getValueType())));
  }

  // Clear the sign bit of the first operand.
  SDValue Mask = DAG.getNode(ISD::SHL, dl, LVT, DAG.getConstant(1, LVT),
                               DAG.getConstant(LSize - 1,
                                               TLI.getShiftAmountTy(LVT)));
  Mask = DAG.getNode(ISD::SUB, dl, LVT, Mask, DAG.getConstant(1, LVT));
  LHS = DAG.getNode(ISD::AND, dl, LVT, LHS, Mask);

  // Or the value with the sign bit.
  return DAG.getNode(ISD::OR, dl, LVT, LHS, SignBit);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FCOS(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::COS_F32,
                                  RTLIB::COS_F64,
                                  RTLIB::COS_F80,
                                  RTLIB::COS_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FDIV(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                     GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::DIV_F32,
                                  RTLIB::DIV_F64,
                                  RTLIB::DIV_F80,
                                  RTLIB::DIV_PPCF128),
                     NVT, Ops, 2, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FEXP(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::EXP_F32,
                                  RTLIB::EXP_F64,
                                  RTLIB::EXP_F80,
                                  RTLIB::EXP_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FEXP2(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::EXP2_F32,
                                  RTLIB::EXP2_F64,
                                  RTLIB::EXP2_F80,
                                  RTLIB::EXP2_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FFLOOR(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::FLOOR_F32,
                                  RTLIB::FLOOR_F64,
                                  RTLIB::FLOOR_F80,
                                  RTLIB::FLOOR_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FLOG(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::LOG_F32,
                                  RTLIB::LOG_F64,
                                  RTLIB::LOG_F80,
                                  RTLIB::LOG_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FLOG2(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::LOG2_F32,
                                  RTLIB::LOG2_F64,
                                  RTLIB::LOG2_F80,
                                  RTLIB::LOG2_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FLOG10(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::LOG10_F32,
                                  RTLIB::LOG10_F64,
                                  RTLIB::LOG10_F80,
                                  RTLIB::LOG10_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FMA(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Ops[3] = { GetSoftenedFloat(N->getOperand(0)),
                     GetSoftenedFloat(N->getOperand(1)),
                     GetSoftenedFloat(N->getOperand(2)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::FMA_F32,
                                  RTLIB::FMA_F64,
                                  RTLIB::FMA_F80,
                                  RTLIB::FMA_PPCF128),
                     NVT, Ops, 3, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FMUL(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                     GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::MUL_F32,
                                  RTLIB::MUL_F64,
                                  RTLIB::MUL_F80,
                                  RTLIB::MUL_PPCF128),
                     NVT, Ops, 2, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FNEARBYINT(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::NEARBYINT_F32,
                                  RTLIB::NEARBYINT_F64,
                                  RTLIB::NEARBYINT_F80,
                                  RTLIB::NEARBYINT_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FNEG(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  // Expand Y = FNEG(X) -> Y = SUB -0.0, X
  SDValue Ops[2] = { DAG.getConstantFP(-0.0, N->getValueType(0)),
                     GetSoftenedFloat(N->getOperand(0)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::SUB_F32,
                                  RTLIB::SUB_F64,
                                  RTLIB::SUB_F80,
                                  RTLIB::SUB_PPCF128),
                     NVT, Ops, 2, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FP_EXTEND(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = N->getOperand(0);
  RTLIB::Libcall LC = RTLIB::getFPEXT(Op.getValueType(), N->getValueType(0));
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_EXTEND!");
  return MakeLibCall(LC, NVT, &Op, 1, false, N->getDebugLoc());
}

// FIXME: Should we just use 'normal' FP_EXTEND / FP_TRUNC instead of special
// nodes?
SDValue DAGTypeLegalizer::SoftenFloatRes_FP16_TO_FP32(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = N->getOperand(0);
  return MakeLibCall(RTLIB::FPEXT_F16_F32, NVT, &Op, 1, false,
                     N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FP_ROUND(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = N->getOperand(0);
  RTLIB::Libcall LC = RTLIB::getFPROUND(Op.getValueType(), N->getValueType(0));
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_ROUND!");
  return MakeLibCall(LC, NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FPOW(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                     GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::POW_F32,
                                  RTLIB::POW_F64,
                                  RTLIB::POW_F80,
                                  RTLIB::POW_PPCF128),
                     NVT, Ops, 2, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FPOWI(SDNode *N) {
  assert(N->getOperand(1).getValueType() == MVT::i32 &&
         "Unsupported power type!");
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)), N->getOperand(1) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::POWI_F32,
                                  RTLIB::POWI_F64,
                                  RTLIB::POWI_F80,
                                  RTLIB::POWI_PPCF128),
                     NVT, Ops, 2, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FREM(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                     GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::REM_F32,
                                  RTLIB::REM_F64,
                                  RTLIB::REM_F80,
                                  RTLIB::REM_PPCF128),
                     NVT, Ops, 2, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FRINT(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::RINT_F32,
                                  RTLIB::RINT_F64,
                                  RTLIB::RINT_F80,
                                  RTLIB::RINT_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FSIN(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::SIN_F32,
                                  RTLIB::SIN_F64,
                                  RTLIB::SIN_F80,
                                  RTLIB::SIN_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FSQRT(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::SQRT_F32,
                                  RTLIB::SQRT_F64,
                                  RTLIB::SQRT_F80,
                                  RTLIB::SQRT_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FSUB(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Ops[2] = { GetSoftenedFloat(N->getOperand(0)),
                     GetSoftenedFloat(N->getOperand(1)) };
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::SUB_F32,
                                  RTLIB::SUB_F64,
                                  RTLIB::SUB_F80,
                                  RTLIB::SUB_PPCF128),
                     NVT, Ops, 2, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_FTRUNC(SDNode *N) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(GetFPLibCall(N->getValueType(0),
                                  RTLIB::TRUNC_F32,
                                  RTLIB::TRUNC_F64,
                                  RTLIB::TRUNC_F80,
                                  RTLIB::TRUNC_PPCF128),
                     NVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatRes_LOAD(SDNode *N) {
  LoadSDNode *L = cast<LoadSDNode>(N);
  EVT VT = N->getValueType(0);
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), VT);
  DebugLoc dl = N->getDebugLoc();

  SDValue NewL;
  if (L->getExtensionType() == ISD::NON_EXTLOAD) {
    NewL = DAG.getLoad(L->getAddressingMode(), L->getExtensionType(),
                       NVT, dl, L->getChain(), L->getBasePtr(), L->getOffset(),
                       L->getPointerInfo(), NVT, L->isVolatile(), 
                       L->isNonTemporal(), false, L->getAlignment());
    // Legalized the chain result - switch anything that used the old chain to
    // use the new one.
    ReplaceValueWith(SDValue(N, 1), NewL.getValue(1));
    return NewL;
  }

  // Do a non-extending load followed by FP_EXTEND.
  NewL = DAG.getLoad(L->getAddressingMode(), ISD::NON_EXTLOAD,
                     L->getMemoryVT(), dl, L->getChain(),
                     L->getBasePtr(), L->getOffset(), L->getPointerInfo(),
                     L->getMemoryVT(), L->isVolatile(),
                     L->isNonTemporal(), false, L->getAlignment());
  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDValue(N, 1), NewL.getValue(1));
  return BitConvertToInteger(DAG.getNode(ISD::FP_EXTEND, dl, VT, NewL));
}

SDValue DAGTypeLegalizer::SoftenFloatRes_SELECT(SDNode *N) {
  SDValue LHS = GetSoftenedFloat(N->getOperand(1));
  SDValue RHS = GetSoftenedFloat(N->getOperand(2));
  return DAG.getNode(ISD::SELECT, N->getDebugLoc(),
                     LHS.getValueType(), N->getOperand(0),LHS,RHS);
}

SDValue DAGTypeLegalizer::SoftenFloatRes_SELECT_CC(SDNode *N) {
  SDValue LHS = GetSoftenedFloat(N->getOperand(2));
  SDValue RHS = GetSoftenedFloat(N->getOperand(3));
  return DAG.getNode(ISD::SELECT_CC, N->getDebugLoc(),
                     LHS.getValueType(), N->getOperand(0),
                     N->getOperand(1), LHS, RHS, N->getOperand(4));
}

SDValue DAGTypeLegalizer::SoftenFloatRes_UNDEF(SDNode *N) {
  return DAG.getUNDEF(TLI.getTypeToTransformTo(*DAG.getContext(),
                                               N->getValueType(0)));
}

SDValue DAGTypeLegalizer::SoftenFloatRes_VAARG(SDNode *N) {
  SDValue Chain = N->getOperand(0); // Get the chain.
  SDValue Ptr = N->getOperand(1); // Get the pointer.
  EVT VT = N->getValueType(0);
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), VT);
  DebugLoc dl = N->getDebugLoc();

  SDValue NewVAARG;
  NewVAARG = DAG.getVAArg(NVT, dl, Chain, Ptr, N->getOperand(2),
                          N->getConstantOperandVal(3));

  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDValue(N, 1), NewVAARG.getValue(1));
  return NewVAARG;
}

SDValue DAGTypeLegalizer::SoftenFloatRes_XINT_TO_FP(SDNode *N) {
  bool Signed = N->getOpcode() == ISD::SINT_TO_FP;
  EVT SVT = N->getOperand(0).getValueType();
  EVT RVT = N->getValueType(0);
  EVT NVT = EVT();
  DebugLoc dl = N->getDebugLoc();

  // If the input is not legal, eg: i1 -> fp, then it needs to be promoted to
  // a larger type, eg: i8 -> fp.  Even if it is legal, no libcall may exactly
  // match.  Look for an appropriate libcall.
  RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
  for (unsigned t = MVT::FIRST_INTEGER_VALUETYPE;
       t <= MVT::LAST_INTEGER_VALUETYPE && LC == RTLIB::UNKNOWN_LIBCALL; ++t) {
    NVT = (MVT::SimpleValueType)t;
    // The source needs to big enough to hold the operand.
    if (NVT.bitsGE(SVT))
      LC = Signed ? RTLIB::getSINTTOFP(NVT, RVT):RTLIB::getUINTTOFP (NVT, RVT);
  }
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported XINT_TO_FP!");

  // Sign/zero extend the argument if the libcall takes a larger type.
  SDValue Op = DAG.getNode(Signed ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND, dl,
                           NVT, N->getOperand(0));
  return MakeLibCall(LC, TLI.getTypeToTransformTo(*DAG.getContext(), RVT),
                     &Op, 1, false, dl);
}


//===----------------------------------------------------------------------===//
//  Operand Float to Integer Conversion..
//===----------------------------------------------------------------------===//

bool DAGTypeLegalizer::SoftenFloatOperand(SDNode *N, unsigned OpNo) {
  DEBUG(dbgs() << "Soften float operand " << OpNo << ": "; N->dump(&DAG);
        dbgs() << "\n");
  SDValue Res = SDValue();

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    dbgs() << "SoftenFloatOperand Op #" << OpNo << ": ";
    N->dump(&DAG); dbgs() << "\n";
#endif
    llvm_unreachable("Do not know how to soften this operator's operand!");

  case ISD::BITCAST:     Res = SoftenFloatOp_BITCAST(N); break;
  case ISD::BR_CC:       Res = SoftenFloatOp_BR_CC(N); break;
  case ISD::FP_ROUND:    Res = SoftenFloatOp_FP_ROUND(N); break;
  case ISD::FP_TO_SINT:  Res = SoftenFloatOp_FP_TO_SINT(N); break;
  case ISD::FP_TO_UINT:  Res = SoftenFloatOp_FP_TO_UINT(N); break;
  case ISD::FP32_TO_FP16:Res = SoftenFloatOp_FP32_TO_FP16(N); break;
  case ISD::SELECT_CC:   Res = SoftenFloatOp_SELECT_CC(N); break;
  case ISD::SETCC:       Res = SoftenFloatOp_SETCC(N); break;
  case ISD::STORE:       Res = SoftenFloatOp_STORE(N, OpNo); break;
  }

  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.getNode()) return false;

  // If the result is N, the sub-method updated N in place.  Tell the legalizer
  // core about this.
  if (Res.getNode() == N)
    return true;

  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");

  ReplaceValueWith(SDValue(N, 0), Res);
  return false;
}

/// SoftenSetCCOperands - Soften the operands of a comparison.  This code is
/// shared among BR_CC, SELECT_CC, and SETCC handlers.
void DAGTypeLegalizer::SoftenSetCCOperands(SDValue &NewLHS, SDValue &NewRHS,
                                           ISD::CondCode &CCCode, DebugLoc dl) {
  SDValue LHSInt = GetSoftenedFloat(NewLHS);
  SDValue RHSInt = GetSoftenedFloat(NewRHS);
  EVT VT = NewLHS.getValueType();

  assert((VT == MVT::f32 || VT == MVT::f64) && "Unsupported setcc type!");

  // Expand into one or more soft-fp libcall(s).
  RTLIB::Libcall LC1 = RTLIB::UNKNOWN_LIBCALL, LC2 = RTLIB::UNKNOWN_LIBCALL;
  switch (CCCode) {
  case ISD::SETEQ:
  case ISD::SETOEQ:
    LC1 = (VT == MVT::f32) ? RTLIB::OEQ_F32 : RTLIB::OEQ_F64;
    break;
  case ISD::SETNE:
  case ISD::SETUNE:
    LC1 = (VT == MVT::f32) ? RTLIB::UNE_F32 : RTLIB::UNE_F64;
    break;
  case ISD::SETGE:
  case ISD::SETOGE:
    LC1 = (VT == MVT::f32) ? RTLIB::OGE_F32 : RTLIB::OGE_F64;
    break;
  case ISD::SETLT:
  case ISD::SETOLT:
    LC1 = (VT == MVT::f32) ? RTLIB::OLT_F32 : RTLIB::OLT_F64;
    break;
  case ISD::SETLE:
  case ISD::SETOLE:
    LC1 = (VT == MVT::f32) ? RTLIB::OLE_F32 : RTLIB::OLE_F64;
    break;
  case ISD::SETGT:
  case ISD::SETOGT:
    LC1 = (VT == MVT::f32) ? RTLIB::OGT_F32 : RTLIB::OGT_F64;
    break;
  case ISD::SETUO:
    LC1 = (VT == MVT::f32) ? RTLIB::UO_F32 : RTLIB::UO_F64;
    break;
  case ISD::SETO:
    LC1 = (VT == MVT::f32) ? RTLIB::O_F32 : RTLIB::O_F64;
    break;
  default:
    LC1 = (VT == MVT::f32) ? RTLIB::UO_F32 : RTLIB::UO_F64;
    switch (CCCode) {
    case ISD::SETONE:
      // SETONE = SETOLT | SETOGT
      LC1 = (VT == MVT::f32) ? RTLIB::OLT_F32 : RTLIB::OLT_F64;
      // Fallthrough
    case ISD::SETUGT:
      LC2 = (VT == MVT::f32) ? RTLIB::OGT_F32 : RTLIB::OGT_F64;
      break;
    case ISD::SETUGE:
      LC2 = (VT == MVT::f32) ? RTLIB::OGE_F32 : RTLIB::OGE_F64;
      break;
    case ISD::SETULT:
      LC2 = (VT == MVT::f32) ? RTLIB::OLT_F32 : RTLIB::OLT_F64;
      break;
    case ISD::SETULE:
      LC2 = (VT == MVT::f32) ? RTLIB::OLE_F32 : RTLIB::OLE_F64;
      break;
    case ISD::SETUEQ:
      LC2 = (VT == MVT::f32) ? RTLIB::OEQ_F32 : RTLIB::OEQ_F64;
      break;
    default: llvm_unreachable("Do not know how to soften this setcc!");
    }
  }

  // Use the target specific return value for comparions lib calls.
  EVT RetVT = TLI.getCmpLibcallReturnType();
  SDValue Ops[2] = { LHSInt, RHSInt };
  NewLHS = MakeLibCall(LC1, RetVT, Ops, 2, false/*sign irrelevant*/, dl);
  NewRHS = DAG.getConstant(0, RetVT);
  CCCode = TLI.getCmpLibcallCC(LC1);
  if (LC2 != RTLIB::UNKNOWN_LIBCALL) {
    SDValue Tmp = DAG.getNode(ISD::SETCC, dl, TLI.getSetCCResultType(RetVT),
                                NewLHS, NewRHS, DAG.getCondCode(CCCode));
    NewLHS = MakeLibCall(LC2, RetVT, Ops, 2, false/*sign irrelevant*/, dl);
    NewLHS = DAG.getNode(ISD::SETCC, dl, TLI.getSetCCResultType(RetVT), NewLHS,
                         NewRHS, DAG.getCondCode(TLI.getCmpLibcallCC(LC2)));
    NewLHS = DAG.getNode(ISD::OR, dl, Tmp.getValueType(), Tmp, NewLHS);
    NewRHS = SDValue();
  }
}

SDValue DAGTypeLegalizer::SoftenFloatOp_BITCAST(SDNode *N) {
  return DAG.getNode(ISD::BITCAST, N->getDebugLoc(), N->getValueType(0),
                     GetSoftenedFloat(N->getOperand(0)));
}

SDValue DAGTypeLegalizer::SoftenFloatOp_FP_ROUND(SDNode *N) {
  EVT SVT = N->getOperand(0).getValueType();
  EVT RVT = N->getValueType(0);

  RTLIB::Libcall LC = RTLIB::getFPROUND(SVT, RVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_ROUND libcall");

  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(LC, RVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatOp_BR_CC(SDNode *N) {
  SDValue NewLHS = N->getOperand(2), NewRHS = N->getOperand(3);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(1))->get();
  SoftenSetCCOperands(NewLHS, NewRHS, CCCode, N->getDebugLoc());

  // If SoftenSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.getNode() == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return SDValue(DAG.UpdateNodeOperands(N, N->getOperand(0),
                                DAG.getCondCode(CCCode), NewLHS, NewRHS,
                                N->getOperand(4)),
                 0);
}

SDValue DAGTypeLegalizer::SoftenFloatOp_FP_TO_SINT(SDNode *N) {
  EVT RVT = N->getValueType(0);
  RTLIB::Libcall LC = RTLIB::getFPTOSINT(N->getOperand(0).getValueType(), RVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_TO_SINT!");
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(LC, RVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatOp_FP_TO_UINT(SDNode *N) {
  EVT RVT = N->getValueType(0);
  RTLIB::Libcall LC = RTLIB::getFPTOUINT(N->getOperand(0).getValueType(), RVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_TO_UINT!");
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(LC, RVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatOp_FP32_TO_FP16(SDNode *N) {
  EVT RVT = N->getValueType(0);
  RTLIB::Libcall LC = RTLIB::FPROUND_F32_F16;
  SDValue Op = GetSoftenedFloat(N->getOperand(0));
  return MakeLibCall(LC, RVT, &Op, 1, false, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::SoftenFloatOp_SELECT_CC(SDNode *N) {
  SDValue NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(4))->get();
  SoftenSetCCOperands(NewLHS, NewRHS, CCCode, N->getDebugLoc());

  // If SoftenSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.getNode() == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return SDValue(DAG.UpdateNodeOperands(N, NewLHS, NewRHS,
                                N->getOperand(2), N->getOperand(3),
                                DAG.getCondCode(CCCode)),
                 0);
}

SDValue DAGTypeLegalizer::SoftenFloatOp_SETCC(SDNode *N) {
  SDValue NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(2))->get();
  SoftenSetCCOperands(NewLHS, NewRHS, CCCode, N->getDebugLoc());

  // If SoftenSetCCOperands returned a scalar, use it.
  if (NewRHS.getNode() == 0) {
    assert(NewLHS.getValueType() == N->getValueType(0) &&
           "Unexpected setcc expansion!");
    return NewLHS;
  }

  // Otherwise, update N to have the operands specified.
  return SDValue(DAG.UpdateNodeOperands(N, NewLHS, NewRHS,
                                DAG.getCondCode(CCCode)),
                 0);
}

SDValue DAGTypeLegalizer::SoftenFloatOp_STORE(SDNode *N, unsigned OpNo) {
  assert(ISD::isUNINDEXEDStore(N) && "Indexed store during type legalization!");
  assert(OpNo == 1 && "Can only soften the stored value!");
  StoreSDNode *ST = cast<StoreSDNode>(N);
  SDValue Val = ST->getValue();
  DebugLoc dl = N->getDebugLoc();

  if (ST->isTruncatingStore())
    // Do an FP_ROUND followed by a non-truncating store.
    Val = BitConvertToInteger(DAG.getNode(ISD::FP_ROUND, dl, ST->getMemoryVT(),
                                          Val, DAG.getIntPtrConstant(0)));
  else
    Val = GetSoftenedFloat(Val);

  return DAG.getStore(ST->getChain(), dl, Val, ST->getBasePtr(),
                      ST->getPointerInfo(),
                      ST->isVolatile(), ST->isNonTemporal(),
                      ST->getAlignment());
}


//===----------------------------------------------------------------------===//
//  Float Result Expansion
//===----------------------------------------------------------------------===//

/// ExpandFloatResult - This method is called when the specified result of the
/// specified node is found to need expansion.  At this point, the node may also
/// have invalid operands or may have other results that need promotion, we just
/// know that (at least) one result needs expansion.
void DAGTypeLegalizer::ExpandFloatResult(SDNode *N, unsigned ResNo) {
  DEBUG(dbgs() << "Expand float result: "; N->dump(&DAG); dbgs() << "\n");
  SDValue Lo, Hi;
  Lo = Hi = SDValue();

  // See if the target wants to custom expand this node.
  if (CustomLowerNode(N, N->getValueType(ResNo), true))
    return;

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    dbgs() << "ExpandFloatResult #" << ResNo << ": ";
    N->dump(&DAG); dbgs() << "\n";
#endif
    llvm_unreachable("Do not know how to expand the result of this operator!");

  case ISD::UNDEF:        SplitRes_UNDEF(N, Lo, Hi); break;
  case ISD::SELECT:       SplitRes_SELECT(N, Lo, Hi); break;
  case ISD::SELECT_CC:    SplitRes_SELECT_CC(N, Lo, Hi); break;

  case ISD::MERGE_VALUES:       ExpandRes_MERGE_VALUES(N, ResNo, Lo, Hi); break;
  case ISD::BITCAST:            ExpandRes_BITCAST(N, Lo, Hi); break;
  case ISD::BUILD_PAIR:         ExpandRes_BUILD_PAIR(N, Lo, Hi); break;
  case ISD::EXTRACT_ELEMENT:    ExpandRes_EXTRACT_ELEMENT(N, Lo, Hi); break;
  case ISD::EXTRACT_VECTOR_ELT: ExpandRes_EXTRACT_VECTOR_ELT(N, Lo, Hi); break;
  case ISD::VAARG:              ExpandRes_VAARG(N, Lo, Hi); break;

  case ISD::ConstantFP: ExpandFloatRes_ConstantFP(N, Lo, Hi); break;
  case ISD::FABS:       ExpandFloatRes_FABS(N, Lo, Hi); break;
  case ISD::FADD:       ExpandFloatRes_FADD(N, Lo, Hi); break;
  case ISD::FCEIL:      ExpandFloatRes_FCEIL(N, Lo, Hi); break;
  case ISD::FCOPYSIGN:  ExpandFloatRes_FCOPYSIGN(N, Lo, Hi); break;
  case ISD::FCOS:       ExpandFloatRes_FCOS(N, Lo, Hi); break;
  case ISD::FDIV:       ExpandFloatRes_FDIV(N, Lo, Hi); break;
  case ISD::FEXP:       ExpandFloatRes_FEXP(N, Lo, Hi); break;
  case ISD::FEXP2:      ExpandFloatRes_FEXP2(N, Lo, Hi); break;
  case ISD::FFLOOR:     ExpandFloatRes_FFLOOR(N, Lo, Hi); break;
  case ISD::FLOG:       ExpandFloatRes_FLOG(N, Lo, Hi); break;
  case ISD::FLOG2:      ExpandFloatRes_FLOG2(N, Lo, Hi); break;
  case ISD::FLOG10:     ExpandFloatRes_FLOG10(N, Lo, Hi); break;
  case ISD::FMA:        ExpandFloatRes_FMA(N, Lo, Hi); break;
  case ISD::FMUL:       ExpandFloatRes_FMUL(N, Lo, Hi); break;
  case ISD::FNEARBYINT: ExpandFloatRes_FNEARBYINT(N, Lo, Hi); break;
  case ISD::FNEG:       ExpandFloatRes_FNEG(N, Lo, Hi); break;
  case ISD::FP_EXTEND:  ExpandFloatRes_FP_EXTEND(N, Lo, Hi); break;
  case ISD::FPOW:       ExpandFloatRes_FPOW(N, Lo, Hi); break;
  case ISD::FPOWI:      ExpandFloatRes_FPOWI(N, Lo, Hi); break;
  case ISD::FRINT:      ExpandFloatRes_FRINT(N, Lo, Hi); break;
  case ISD::FSIN:       ExpandFloatRes_FSIN(N, Lo, Hi); break;
  case ISD::FSQRT:      ExpandFloatRes_FSQRT(N, Lo, Hi); break;
  case ISD::FSUB:       ExpandFloatRes_FSUB(N, Lo, Hi); break;
  case ISD::FTRUNC:     ExpandFloatRes_FTRUNC(N, Lo, Hi); break;
  case ISD::LOAD:       ExpandFloatRes_LOAD(N, Lo, Hi); break;
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: ExpandFloatRes_XINT_TO_FP(N, Lo, Hi); break;
  }

  // If Lo/Hi is null, the sub-method took care of registering results etc.
  if (Lo.getNode())
    SetExpandedFloat(SDValue(N, ResNo), Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_ConstantFP(SDNode *N, SDValue &Lo,
                                                 SDValue &Hi) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  assert(NVT.getSizeInBits() == integerPartWidth &&
         "Do not know how to expand this float constant!");
  APInt C = cast<ConstantFPSDNode>(N)->getValueAPF().bitcastToAPInt();
  Lo = DAG.getConstantFP(APFloat(APInt(integerPartWidth, C.getRawData()[1])),
                         NVT);
  Hi = DAG.getConstantFP(APFloat(APInt(integerPartWidth, C.getRawData()[0])),
                         NVT);
}

void DAGTypeLegalizer::ExpandFloatRes_FABS(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  assert(N->getValueType(0) == MVT::ppcf128 &&
         "Logic only correct for ppcf128!");
  DebugLoc dl = N->getDebugLoc();
  SDValue Tmp;
  GetExpandedFloat(N->getOperand(0), Lo, Tmp);
  Hi = DAG.getNode(ISD::FABS, dl, Tmp.getValueType(), Tmp);
  // Lo = Hi==fabs(Hi) ? Lo : -Lo;
  Lo = DAG.getNode(ISD::SELECT_CC, dl, Lo.getValueType(), Tmp, Hi, Lo,
                   DAG.getNode(ISD::FNEG, dl, Lo.getValueType(), Lo),
                   DAG.getCondCode(ISD::SETEQ));
}

void DAGTypeLegalizer::ExpandFloatRes_FADD(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::ADD_F32, RTLIB::ADD_F64,
                                         RTLIB::ADD_F80, RTLIB::ADD_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FCEIL(SDNode *N,
                                            SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::CEIL_F32, RTLIB::CEIL_F64,
                                         RTLIB::CEIL_F80, RTLIB::CEIL_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FCOPYSIGN(SDNode *N,
                                                SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::COPYSIGN_F32,
                                         RTLIB::COPYSIGN_F64,
                                         RTLIB::COPYSIGN_F80,
                                         RTLIB::COPYSIGN_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FCOS(SDNode *N,
                                           SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::COS_F32, RTLIB::COS_F64,
                                         RTLIB::COS_F80, RTLIB::COS_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FDIV(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SDValue Call = MakeLibCall(GetFPLibCall(N->getValueType(0),
                                          RTLIB::DIV_F32,
                                          RTLIB::DIV_F64,
                                          RTLIB::DIV_F80,
                                          RTLIB::DIV_PPCF128),
                             N->getValueType(0), Ops, 2, false,
                             N->getDebugLoc());
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FEXP(SDNode *N,
                                           SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::EXP_F32, RTLIB::EXP_F64,
                                         RTLIB::EXP_F80, RTLIB::EXP_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FEXP2(SDNode *N,
                                            SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::EXP2_F32, RTLIB::EXP2_F64,
                                         RTLIB::EXP2_F80, RTLIB::EXP2_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FFLOOR(SDNode *N,
                                             SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::FLOOR_F32,RTLIB::FLOOR_F64,
                                         RTLIB::FLOOR_F80,RTLIB::FLOOR_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FLOG(SDNode *N,
                                           SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::LOG_F32, RTLIB::LOG_F64,
                                         RTLIB::LOG_F80, RTLIB::LOG_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FLOG2(SDNode *N,
                                            SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::LOG2_F32, RTLIB::LOG2_F64,
                                         RTLIB::LOG2_F80, RTLIB::LOG2_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FLOG10(SDNode *N,
                                             SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::LOG10_F32,RTLIB::LOG10_F64,
                                         RTLIB::LOG10_F80,RTLIB::LOG10_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FMA(SDNode *N, SDValue &Lo,
                                          SDValue &Hi) {
  SDValue Ops[3] = { N->getOperand(0), N->getOperand(1), N->getOperand(2) };
  SDValue Call = MakeLibCall(GetFPLibCall(N->getValueType(0),
                                          RTLIB::FMA_F32,
                                          RTLIB::FMA_F64,
                                          RTLIB::FMA_F80,
                                          RTLIB::FMA_PPCF128),
                             N->getValueType(0), Ops, 3, false,
                             N->getDebugLoc());
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FMUL(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SDValue Call = MakeLibCall(GetFPLibCall(N->getValueType(0),
                                          RTLIB::MUL_F32,
                                          RTLIB::MUL_F64,
                                          RTLIB::MUL_F80,
                                          RTLIB::MUL_PPCF128),
                             N->getValueType(0), Ops, 2, false,
                             N->getDebugLoc());
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FNEARBYINT(SDNode *N,
                                                 SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::NEARBYINT_F32,
                                         RTLIB::NEARBYINT_F64,
                                         RTLIB::NEARBYINT_F80,
                                         RTLIB::NEARBYINT_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FNEG(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  GetExpandedFloat(N->getOperand(0), Lo, Hi);
  Lo = DAG.getNode(ISD::FNEG, dl, Lo.getValueType(), Lo);
  Hi = DAG.getNode(ISD::FNEG, dl, Hi.getValueType(), Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FP_EXTEND(SDNode *N, SDValue &Lo,
                                                SDValue &Hi) {
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), N->getValueType(0));
  Hi = DAG.getNode(ISD::FP_EXTEND, N->getDebugLoc(), NVT, N->getOperand(0));
  Lo = DAG.getConstantFP(APFloat(APInt(NVT.getSizeInBits(), 0)), NVT);
}

void DAGTypeLegalizer::ExpandFloatRes_FPOW(SDNode *N,
                                           SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::POW_F32, RTLIB::POW_F64,
                                         RTLIB::POW_F80, RTLIB::POW_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FPOWI(SDNode *N,
                                            SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::POWI_F32, RTLIB::POWI_F64,
                                         RTLIB::POWI_F80, RTLIB::POWI_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FRINT(SDNode *N,
                                            SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::RINT_F32, RTLIB::RINT_F64,
                                         RTLIB::RINT_F80, RTLIB::RINT_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FSIN(SDNode *N,
                                           SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::SIN_F32, RTLIB::SIN_F64,
                                         RTLIB::SIN_F80, RTLIB::SIN_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FSQRT(SDNode *N,
                                            SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::SQRT_F32, RTLIB::SQRT_F64,
                                         RTLIB::SQRT_F80, RTLIB::SQRT_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FSUB(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SDValue Call = MakeLibCall(GetFPLibCall(N->getValueType(0),
                                          RTLIB::SUB_F32,
                                          RTLIB::SUB_F64,
                                          RTLIB::SUB_F80,
                                          RTLIB::SUB_PPCF128),
                             N->getValueType(0), Ops, 2, false,
                             N->getDebugLoc());
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_FTRUNC(SDNode *N,
                                             SDValue &Lo, SDValue &Hi) {
  SDValue Call = LibCallify(GetFPLibCall(N->getValueType(0),
                                         RTLIB::TRUNC_F32, RTLIB::TRUNC_F64,
                                         RTLIB::TRUNC_F80, RTLIB::TRUNC_PPCF128),
                            N, false);
  GetPairElements(Call, Lo, Hi);
}

void DAGTypeLegalizer::ExpandFloatRes_LOAD(SDNode *N, SDValue &Lo,
                                           SDValue &Hi) {
  if (ISD::isNormalLoad(N)) {
    ExpandRes_NormalLoad(N, Lo, Hi);
    return;
  }

  assert(ISD::isUNINDEXEDLoad(N) && "Indexed load during type legalization!");
  LoadSDNode *LD = cast<LoadSDNode>(N);
  SDValue Chain = LD->getChain();
  SDValue Ptr = LD->getBasePtr();
  DebugLoc dl = N->getDebugLoc();

  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), LD->getValueType(0));
  assert(NVT.isByteSized() && "Expanded type not byte sized!");
  assert(LD->getMemoryVT().bitsLE(NVT) && "Float type not round?");

  Hi = DAG.getExtLoad(LD->getExtensionType(), dl, NVT, Chain, Ptr,
                      LD->getPointerInfo(), LD->getMemoryVT(), LD->isVolatile(),
                      LD->isNonTemporal(), LD->getAlignment());

  // Remember the chain.
  Chain = Hi.getValue(1);

  // The low part is zero.
  Lo = DAG.getConstantFP(APFloat(APInt(NVT.getSizeInBits(), 0)), NVT);

  // Modified the chain - switch anything that used the old chain to use the
  // new one.
  ReplaceValueWith(SDValue(LD, 1), Chain);
}

void DAGTypeLegalizer::ExpandFloatRes_XINT_TO_FP(SDNode *N, SDValue &Lo,
                                                 SDValue &Hi) {
  assert(N->getValueType(0) == MVT::ppcf128 && "Unsupported XINT_TO_FP!");
  EVT VT = N->getValueType(0);
  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), VT);
  SDValue Src = N->getOperand(0);
  EVT SrcVT = Src.getValueType();
  bool isSigned = N->getOpcode() == ISD::SINT_TO_FP;
  DebugLoc dl = N->getDebugLoc();

  // First do an SINT_TO_FP, whether the original was signed or unsigned.
  // When promoting partial word types to i32 we must honor the signedness,
  // though.
  if (SrcVT.bitsLE(MVT::i32)) {
    // The integer can be represented exactly in an f64.
    Src = DAG.getNode(isSigned ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND, dl,
                      MVT::i32, Src);
    Lo = DAG.getConstantFP(APFloat(APInt(NVT.getSizeInBits(), 0)), NVT);
    Hi = DAG.getNode(ISD::SINT_TO_FP, dl, NVT, Src);
  } else {
    RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
    if (SrcVT.bitsLE(MVT::i64)) {
      Src = DAG.getNode(isSigned ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND, dl,
                        MVT::i64, Src);
      LC = RTLIB::SINTTOFP_I64_PPCF128;
    } else if (SrcVT.bitsLE(MVT::i128)) {
      Src = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::i128, Src);
      LC = RTLIB::SINTTOFP_I128_PPCF128;
    }
    assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported XINT_TO_FP!");

    Hi = MakeLibCall(LC, VT, &Src, 1, true, dl);
    GetPairElements(Hi, Lo, Hi);
  }

  if (isSigned)
    return;

  // Unsigned - fix up the SINT_TO_FP value just calculated.
  Hi = DAG.getNode(ISD::BUILD_PAIR, dl, VT, Lo, Hi);
  SrcVT = Src.getValueType();

  // x>=0 ? (ppcf128)(iN)x : (ppcf128)(iN)x + 2^N; N=32,64,128.
  static const uint64_t TwoE32[]  = { 0x41f0000000000000LL, 0 };
  static const uint64_t TwoE64[]  = { 0x43f0000000000000LL, 0 };
  static const uint64_t TwoE128[] = { 0x47f0000000000000LL, 0 };
  ArrayRef<uint64_t> Parts;

  switch (SrcVT.getSimpleVT().SimpleTy) {
  default:
    llvm_unreachable("Unsupported UINT_TO_FP!");
  case MVT::i32:
    Parts = TwoE32;
    break;
  case MVT::i64:
    Parts = TwoE64;
    break;
  case MVT::i128:
    Parts = TwoE128;
    break;
  }

  Lo = DAG.getNode(ISD::FADD, dl, VT, Hi,
                   DAG.getConstantFP(APFloat(APInt(128, Parts)),
                                     MVT::ppcf128));
  Lo = DAG.getNode(ISD::SELECT_CC, dl, VT, Src, DAG.getConstant(0, SrcVT),
                   Lo, Hi, DAG.getCondCode(ISD::SETLT));
  GetPairElements(Lo, Lo, Hi);
}


//===----------------------------------------------------------------------===//
//  Float Operand Expansion
//===----------------------------------------------------------------------===//

/// ExpandFloatOperand - This method is called when the specified operand of the
/// specified node is found to need expansion.  At this point, all of the result
/// types of the node are known to be legal, but other operands of the node may
/// need promotion or expansion as well as the specified one.
bool DAGTypeLegalizer::ExpandFloatOperand(SDNode *N, unsigned OpNo) {
  DEBUG(dbgs() << "Expand float operand: "; N->dump(&DAG); dbgs() << "\n");
  SDValue Res = SDValue();

  // See if the target wants to custom expand this node.
  if (CustomLowerNode(N, N->getOperand(OpNo).getValueType(), false))
    return false;

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    dbgs() << "ExpandFloatOperand Op #" << OpNo << ": ";
    N->dump(&DAG); dbgs() << "\n";
#endif
    llvm_unreachable("Do not know how to expand this operator's operand!");

  case ISD::BITCAST:         Res = ExpandOp_BITCAST(N); break;
  case ISD::BUILD_VECTOR:    Res = ExpandOp_BUILD_VECTOR(N); break;
  case ISD::EXTRACT_ELEMENT: Res = ExpandOp_EXTRACT_ELEMENT(N); break;

  case ISD::BR_CC:      Res = ExpandFloatOp_BR_CC(N); break;
  case ISD::FP_ROUND:   Res = ExpandFloatOp_FP_ROUND(N); break;
  case ISD::FP_TO_SINT: Res = ExpandFloatOp_FP_TO_SINT(N); break;
  case ISD::FP_TO_UINT: Res = ExpandFloatOp_FP_TO_UINT(N); break;
  case ISD::SELECT_CC:  Res = ExpandFloatOp_SELECT_CC(N); break;
  case ISD::SETCC:      Res = ExpandFloatOp_SETCC(N); break;
  case ISD::STORE:      Res = ExpandFloatOp_STORE(cast<StoreSDNode>(N),
                                                  OpNo); break;
  }

  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.getNode()) return false;

  // If the result is N, the sub-method updated N in place.  Tell the legalizer
  // core about this.
  if (Res.getNode() == N)
    return true;

  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");

  ReplaceValueWith(SDValue(N, 0), Res);
  return false;
}

/// FloatExpandSetCCOperands - Expand the operands of a comparison.  This code
/// is shared among BR_CC, SELECT_CC, and SETCC handlers.
void DAGTypeLegalizer::FloatExpandSetCCOperands(SDValue &NewLHS,
                                                SDValue &NewRHS,
                                                ISD::CondCode &CCCode,
                                                DebugLoc dl) {
  SDValue LHSLo, LHSHi, RHSLo, RHSHi;
  GetExpandedFloat(NewLHS, LHSLo, LHSHi);
  GetExpandedFloat(NewRHS, RHSLo, RHSHi);

  assert(NewLHS.getValueType() == MVT::ppcf128 && "Unsupported setcc type!");

  // FIXME:  This generated code sucks.  We want to generate
  //         FCMPU crN, hi1, hi2
  //         BNE crN, L:
  //         FCMPU crN, lo1, lo2
  // The following can be improved, but not that much.
  SDValue Tmp1, Tmp2, Tmp3;
  Tmp1 = DAG.getSetCC(dl, TLI.getSetCCResultType(LHSHi.getValueType()),
                      LHSHi, RHSHi, ISD::SETOEQ);
  Tmp2 = DAG.getSetCC(dl, TLI.getSetCCResultType(LHSLo.getValueType()),
                      LHSLo, RHSLo, CCCode);
  Tmp3 = DAG.getNode(ISD::AND, dl, Tmp1.getValueType(), Tmp1, Tmp2);
  Tmp1 = DAG.getSetCC(dl, TLI.getSetCCResultType(LHSHi.getValueType()),
                      LHSHi, RHSHi, ISD::SETUNE);
  Tmp2 = DAG.getSetCC(dl, TLI.getSetCCResultType(LHSHi.getValueType()),
                      LHSHi, RHSHi, CCCode);
  Tmp1 = DAG.getNode(ISD::AND, dl, Tmp1.getValueType(), Tmp1, Tmp2);
  NewLHS = DAG.getNode(ISD::OR, dl, Tmp1.getValueType(), Tmp1, Tmp3);
  NewRHS = SDValue();   // LHS is the result, not a compare.
}

SDValue DAGTypeLegalizer::ExpandFloatOp_BR_CC(SDNode *N) {
  SDValue NewLHS = N->getOperand(2), NewRHS = N->getOperand(3);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(1))->get();
  FloatExpandSetCCOperands(NewLHS, NewRHS, CCCode, N->getDebugLoc());

  // If ExpandSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.getNode() == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return SDValue(DAG.UpdateNodeOperands(N, N->getOperand(0),
                                DAG.getCondCode(CCCode), NewLHS, NewRHS,
                                N->getOperand(4)), 0);
}

SDValue DAGTypeLegalizer::ExpandFloatOp_FP_ROUND(SDNode *N) {
  assert(N->getOperand(0).getValueType() == MVT::ppcf128 &&
         "Logic only correct for ppcf128!");
  SDValue Lo, Hi;
  GetExpandedFloat(N->getOperand(0), Lo, Hi);
  // Round it the rest of the way (e.g. to f32) if needed.
  return DAG.getNode(ISD::FP_ROUND, N->getDebugLoc(),
                     N->getValueType(0), Hi, N->getOperand(1));
}

SDValue DAGTypeLegalizer::ExpandFloatOp_FP_TO_SINT(SDNode *N) {
  EVT RVT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  // Expand ppcf128 to i32 by hand for the benefit of llvm-gcc bootstrap on
  // PPC (the libcall is not available).  FIXME: Do this in a less hacky way.
  if (RVT == MVT::i32) {
    assert(N->getOperand(0).getValueType() == MVT::ppcf128 &&
           "Logic only correct for ppcf128!");
    SDValue Res = DAG.getNode(ISD::FP_ROUND_INREG, dl, MVT::ppcf128,
                              N->getOperand(0), DAG.getValueType(MVT::f64));
    Res = DAG.getNode(ISD::FP_ROUND, dl, MVT::f64, Res,
                      DAG.getIntPtrConstant(1));
    return DAG.getNode(ISD::FP_TO_SINT, dl, MVT::i32, Res);
  }

  RTLIB::Libcall LC = RTLIB::getFPTOSINT(N->getOperand(0).getValueType(), RVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_TO_SINT!");
  return MakeLibCall(LC, RVT, &N->getOperand(0), 1, false, dl);
}

SDValue DAGTypeLegalizer::ExpandFloatOp_FP_TO_UINT(SDNode *N) {
  EVT RVT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  // Expand ppcf128 to i32 by hand for the benefit of llvm-gcc bootstrap on
  // PPC (the libcall is not available).  FIXME: Do this in a less hacky way.
  if (RVT == MVT::i32) {
    assert(N->getOperand(0).getValueType() == MVT::ppcf128 &&
           "Logic only correct for ppcf128!");
    const uint64_t TwoE31[] = {0x41e0000000000000LL, 0};
    APFloat APF = APFloat(APInt(128, TwoE31));
    SDValue Tmp = DAG.getConstantFP(APF, MVT::ppcf128);
    //  X>=2^31 ? (int)(X-2^31)+0x80000000 : (int)X
    // FIXME: generated code sucks.
    return DAG.getNode(ISD::SELECT_CC, dl, MVT::i32, N->getOperand(0), Tmp,
                       DAG.getNode(ISD::ADD, dl, MVT::i32,
                                   DAG.getNode(ISD::FP_TO_SINT, dl, MVT::i32,
                                               DAG.getNode(ISD::FSUB, dl,
                                                           MVT::ppcf128,
                                                           N->getOperand(0),
                                                           Tmp)),
                                   DAG.getConstant(0x80000000, MVT::i32)),
                       DAG.getNode(ISD::FP_TO_SINT, dl,
                                   MVT::i32, N->getOperand(0)),
                       DAG.getCondCode(ISD::SETGE));
  }

  RTLIB::Libcall LC = RTLIB::getFPTOUINT(N->getOperand(0).getValueType(), RVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_TO_UINT!");
  return MakeLibCall(LC, N->getValueType(0), &N->getOperand(0), 1, false, dl);
}

SDValue DAGTypeLegalizer::ExpandFloatOp_SELECT_CC(SDNode *N) {
  SDValue NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(4))->get();
  FloatExpandSetCCOperands(NewLHS, NewRHS, CCCode, N->getDebugLoc());

  // If ExpandSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.getNode() == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return SDValue(DAG.UpdateNodeOperands(N, NewLHS, NewRHS,
                                N->getOperand(2), N->getOperand(3),
                                DAG.getCondCode(CCCode)), 0);
}

SDValue DAGTypeLegalizer::ExpandFloatOp_SETCC(SDNode *N) {
  SDValue NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(2))->get();
  FloatExpandSetCCOperands(NewLHS, NewRHS, CCCode, N->getDebugLoc());

  // If ExpandSetCCOperands returned a scalar, use it.
  if (NewRHS.getNode() == 0) {
    assert(NewLHS.getValueType() == N->getValueType(0) &&
           "Unexpected setcc expansion!");
    return NewLHS;
  }

  // Otherwise, update N to have the operands specified.
  return SDValue(DAG.UpdateNodeOperands(N, NewLHS, NewRHS,
                                DAG.getCondCode(CCCode)), 0);
}

SDValue DAGTypeLegalizer::ExpandFloatOp_STORE(SDNode *N, unsigned OpNo) {
  if (ISD::isNormalStore(N))
    return ExpandOp_NormalStore(N, OpNo);

  assert(ISD::isUNINDEXEDStore(N) && "Indexed store during type legalization!");
  assert(OpNo == 1 && "Can only expand the stored value so far");
  StoreSDNode *ST = cast<StoreSDNode>(N);

  SDValue Chain = ST->getChain();
  SDValue Ptr = ST->getBasePtr();

  EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(),
                                     ST->getValue().getValueType());
  assert(NVT.isByteSized() && "Expanded type not byte sized!");
  assert(ST->getMemoryVT().bitsLE(NVT) && "Float type not round?");
  (void)NVT;

  SDValue Lo, Hi;
  GetExpandedOp(ST->getValue(), Lo, Hi);

  return DAG.getTruncStore(Chain, N->getDebugLoc(), Hi, Ptr,
                           ST->getPointerInfo(),
                           ST->getMemoryVT(), ST->isVolatile(),
                           ST->isNonTemporal(), ST->getAlignment());
}
