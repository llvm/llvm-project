//===------------ VECustomDAG.h - VE Custom DAG Nodes -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the helper functions that VE uses to lower LLVM code into a
// selection DAG.  For example, hiding SDLoc, and easy to use SDNodeFlags.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_VE_VECUSTOMDAG_H
#define LLVM_LIB_TARGET_VE_VECUSTOMDAG_H

#include "VE.h"
#include "VEISelLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

Optional<unsigned> getVVPOpcode(unsigned Opcode);

bool isVVPBinaryOp(unsigned Opcode);

bool isPackedVectorType(EVT SomeVT);

class VECustomDAG {
  SelectionDAG &DAG;
  SDLoc DL;

public:
  SelectionDAG *getDAG() const { return &DAG; }

  VECustomDAG(SelectionDAG &DAG, SDLoc DL) : DAG(DAG), DL(DL) {}

  VECustomDAG(SelectionDAG &DAG, SDValue WhereOp) : DAG(DAG), DL(WhereOp) {}

  VECustomDAG(SelectionDAG &DAG, const SDNode *WhereN) : DAG(DAG), DL(WhereN) {}

  /// getNode {
  SDValue getNode(unsigned OC, SDVTList VTL, ArrayRef<SDValue> OpV,
                  Optional<SDNodeFlags> Flags = None) const {
    auto N = DAG.getNode(OC, DL, VTL, OpV);
    if (Flags)
      N->setFlags(*Flags);
    return N;
  }

  SDValue getNode(unsigned OC, ArrayRef<EVT> ResVT, ArrayRef<SDValue> OpV,
                  Optional<SDNodeFlags> Flags = None) const {
    auto N = DAG.getNode(OC, DL, ResVT, OpV);
    if (Flags)
      N->setFlags(*Flags);
    return N;
  }

  SDValue getNode(unsigned OC, EVT ResVT, ArrayRef<SDValue> OpV,
                  Optional<SDNodeFlags> Flags = None) const {
    auto N = DAG.getNode(OC, DL, ResVT, OpV);
    if (Flags)
      N->setFlags(*Flags);
    return N;
  }

  SDValue getUNDEF(EVT VT) const { return DAG.getUNDEF(VT); }
  /// } getNode

  SDValue getConstant(uint64_t Val, EVT VT, bool IsTarget = false,
                      bool IsOpaque = false) const;

  SDValue getBroadcast(EVT ResultVT, SDValue Scalar, SDValue AVL) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_VE_VECUSTOMDAG_H
