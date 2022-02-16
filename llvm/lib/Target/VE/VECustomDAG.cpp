//===-- VECustomDAG.h - VE Custom DAG Nodes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that VE uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "VECustomDAG.h"

#ifndef DEBUG_TYPE
#define DEBUG_TYPE "vecustomdag"
#endif

namespace llvm {

bool isPackedVectorType(EVT SomeVT) {
  if (!SomeVT.isVector())
    return false;
  return SomeVT.getVectorNumElements() > StandardVectorWidth;
}

MVT getLegalVectorType(Packing P, MVT ElemVT) {
  return MVT::getVectorVT(ElemVT, P == Packing::Normal ? StandardVectorWidth
                                                       : PackedVectorWidth);
}

Packing getTypePacking(EVT VT) {
  assert(VT.isVector());
  return isPackedVectorType(VT) ? Packing::Dense : Packing::Normal;
}

bool isMaskType(EVT SomeVT) {
  if (!SomeVT.isVector())
    return false;
  return SomeVT.getVectorElementType() == MVT::i1;
}

/// \returns the VVP_* SDNode opcode corresponsing to \p OC.
Optional<unsigned> getVVPOpcode(unsigned Opcode) {
  switch (Opcode) {
#define HANDLE_VP_TO_VVP(VPOPC, VVPNAME)                                       \
  case ISD::VPOPC:                                                             \
    return VEISD::VVPNAME;
#define ADD_VVP_OP(VVPNAME, SDNAME)                                            \
  case VEISD::VVPNAME:                                                         \
  case ISD::SDNAME:                                                            \
    return VEISD::VVPNAME;
#include "VVPNodes.def"
  }
  return None;
}

bool maySafelyIgnoreMask(SDValue Op) {
  auto VVPOpc = getVVPOpcode(Op->getOpcode());
  auto Opc = VVPOpc.getValueOr(Op->getOpcode());

  switch (Opc) {
  case VEISD::VVP_SDIV:
  case VEISD::VVP_UDIV:
  case VEISD::VVP_FDIV:
  case VEISD::VVP_SELECT:
    return false;

  default:
    return true;
  }
}

bool isVVPOrVEC(unsigned Opcode) {
  switch (Opcode) {
  case VEISD::VEC_BROADCAST:
#define ADD_VVP_OP(VVPNAME, ...) case VEISD::VVPNAME:
#include "VVPNodes.def"
    return true;
  }
  return false;
}

bool isVVPBinaryOp(unsigned VVPOpcode) {
  switch (VVPOpcode) {
#define ADD_BINARY_VVP_OP(VVPNAME, ...)                                        \
  case VEISD::VVPNAME:                                                         \
    return true;
#include "VVPNodes.def"
  }
  return false;
}

// Return the AVL operand position for this VVP or VEC Op.
Optional<int> getAVLPos(unsigned Opc) {
  // This is only available for VP SDNodes
  auto PosOpt = ISD::getVPExplicitVectorLengthIdx(Opc);
  if (PosOpt)
    return *PosOpt;

  // VVP Opcodes.
  if (isVVPBinaryOp(Opc))
    return 3;

  // VM Opcodes.
  switch (Opc) {
  case VEISD::VEC_BROADCAST:
    return 1;
  case VEISD::VVP_SELECT:
    return 3;
  }

  return None;
}

bool isLegalAVL(SDValue AVL) { return AVL->getOpcode() == VEISD::LEGALAVL; }

SDValue getNodeAVL(SDValue Op) {
  auto PosOpt = getAVLPos(Op->getOpcode());
  return PosOpt ? Op->getOperand(*PosOpt) : SDValue();
}

std::pair<SDValue, bool> getAnnotatedNodeAVL(SDValue Op) {
  SDValue AVL = getNodeAVL(Op);
  if (!AVL)
    return {SDValue(), true};
  if (isLegalAVL(AVL))
    return {AVL->getOperand(0), true};
  return {AVL, false};
}

SDValue VECustomDAG::getConstant(uint64_t Val, EVT VT, bool IsTarget,
                                 bool IsOpaque) const {
  return DAG.getConstant(Val, DL, VT, IsTarget, IsOpaque);
}

SDValue VECustomDAG::getConstantMask(Packing Packing, bool AllTrue) const {
  auto MaskVT = getLegalVectorType(Packing, MVT::i1);

  // VEISelDAGtoDAG will replace this pattern with the constant-true VM.
  auto TrueVal = DAG.getConstant(-1, DL, MVT::i32);
  auto AVL = getConstant(MaskVT.getVectorNumElements(), MVT::i32);
  auto Res = getNode(VEISD::VEC_BROADCAST, MaskVT, {TrueVal, AVL});
  if (AllTrue)
    return Res;

  return DAG.getNOT(DL, Res, Res.getValueType());
}

SDValue VECustomDAG::getMaskBroadcast(EVT ResultVT, SDValue Scalar,
                                      SDValue AVL) const {
  // Constant mask splat.
  if (auto BcConst = dyn_cast<ConstantSDNode>(Scalar))
    return getConstantMask(getTypePacking(ResultVT),
                           BcConst->getSExtValue() != 0);

  // Expand the broadcast to a vector comparison.
  auto ScalarBoolVT = Scalar.getSimpleValueType();
  assert(ScalarBoolVT == MVT::i32);

  // Cast to i32 ty.
  SDValue CmpElem = DAG.getSExtOrTrunc(Scalar, DL, MVT::i32);
  unsigned ElemCount = ResultVT.getVectorNumElements();
  MVT CmpVecTy = MVT::getVectorVT(ScalarBoolVT, ElemCount);

  // Broadcast to vector.
  SDValue BCVec =
      DAG.getNode(VEISD::VEC_BROADCAST, DL, CmpVecTy, {CmpElem, AVL});
  SDValue ZeroVec =
      getBroadcast(CmpVecTy, {DAG.getConstant(0, DL, ScalarBoolVT)}, AVL);

  MVT BoolVecTy = MVT::getVectorVT(MVT::i1, ElemCount);

  // Broadcast(Data) != Broadcast(0)
  // TODO: Use a VVP operation for this.
  return DAG.getSetCC(DL, BoolVecTy, BCVec, ZeroVec, ISD::CondCode::SETNE);
}

SDValue VECustomDAG::getBroadcast(EVT ResultVT, SDValue Scalar,
                                  SDValue AVL) const {
  assert(ResultVT.isVector());
  auto ScaVT = Scalar.getValueType();

  if (isMaskType(ResultVT))
    return getMaskBroadcast(ResultVT, Scalar, AVL);

  if (isPackedVectorType(ResultVT)) {
    // v512x packed mode broadcast
    // Replicate the scalar reg (f32 or i32) onto the opposing half of the full
    // scalar register. If it's an I64 type, assume that this has already
    // happened.
    if (ScaVT == MVT::f32) {
      Scalar = getNode(VEISD::REPL_F32, MVT::i64, Scalar);
    } else if (ScaVT == MVT::i32) {
      Scalar = getNode(VEISD::REPL_I32, MVT::i64, Scalar);
    }
  }

  return getNode(VEISD::VEC_BROADCAST, ResultVT, {Scalar, AVL});
}

SDValue VECustomDAG::annotateLegalAVL(SDValue AVL) const {
  if (isLegalAVL(AVL))
    return AVL;
  return getNode(VEISD::LEGALAVL, AVL.getValueType(), AVL);
}

} // namespace llvm
