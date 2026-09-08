//===- MachineValueType.cpp - Machine-Level types ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGenTypes/MachineValueType.h"

using namespace llvm;

static constexpr MVT::VecVTTable buildVecVTTable() {
  MVT::VecVTTable T{};
  uint8_t NextSlot = 1;
#define GET_VT_ATTR(Ty, Sz, Any, Int, FP, Vec, Sc, Tup, NF, NElem, EltTy)      \
  if (Vec && !Tup && !T.Slots[NElem])                                          \
    T.Slots[NElem] = NextSlot++;
#include "llvm/CodeGen/GenVT.inc"
#undef GET_VT_ATTR
#define GET_VT_ATTR(Ty, Sz, Any, Int, FP, Vec, Sc, Tup, NF, NElem, EltTy)      \
  if (Vec && !Tup)                                                             \
    T.Tys[Sc][MVT::EltTy][T.Slots[NElem]] = MVT::Ty;
#include "llvm/CodeGen/GenVT.inc"
#undef GET_VT_ATTR
  return T;
}

const MVT::VecVTTable MVT::VecVTs = buildVecVTTable();
