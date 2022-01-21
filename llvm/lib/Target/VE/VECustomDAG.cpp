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
  return getNode(VEISD::VEC_BROADCAST, ResultVT, {Scalar, AVL});
}

} // namespace llvm
