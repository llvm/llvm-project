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

static const int StandardVectorWidth = 256;

bool isPackedVectorType(EVT SomeVT) {
  if (!SomeVT.isVector())
    return false;
  return SomeVT.getVectorNumElements() > StandardVectorWidth;
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

bool isVVPBinaryOp(unsigned VVPOpcode) {
  switch (VVPOpcode) {
#define ADD_BINARY_VVP_OP(VVPNAME, ...)                                        \
  case VEISD::VVPNAME:                                                         \
    return true;
#include "VVPNodes.def"
  }
  return false;
}

SDValue VECustomDAG::getConstant(uint64_t Val, EVT VT, bool IsTarget,
                                 bool IsOpaque) const {
  return DAG.getConstant(Val, DL, VT, IsTarget, IsOpaque);
}

SDValue VECustomDAG::getBroadcast(EVT ResultVT, SDValue Scalar,
                                  SDValue AVL) const {
  assert(ResultVT.isVector());
  auto ScaVT = Scalar.getValueType();
  assert(ScaVT != MVT::i1 && "TODO: Mask broadcasts");

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

} // namespace llvm
