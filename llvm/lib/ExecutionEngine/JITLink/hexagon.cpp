//===-------- hexagon.cpp - JITLink hexagon support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hexagon edge kind names.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/hexagon.h"

#define DEBUG_TYPE "jitlink"

namespace llvm::jitlink::hexagon {

const char *getEdgeKindName(Edge::Kind K) {
  switch (K) {
  case Pointer32:
    return "Pointer32";
  case PCRel32:
    return "PCRel32";
  case B22_PCREL:
    return "B22_PCREL";
  case B15_PCREL:
    return "B15_PCREL";
  case B13_PCREL:
    return "B13_PCREL";
  case B9_PCREL:
    return "B9_PCREL";
  case B7_PCREL:
    return "B7_PCREL";
  case HI16:
    return "HI16";
  case LO16:
    return "LO16";
  case Word32_6_X:
    return "Word32_6_X";
  case B32_PCREL_X:
    return "B32_PCREL_X";
  case B22_PCREL_X:
    return "B22_PCREL_X";
  case B15_PCREL_X:
    return "B15_PCREL_X";
  case B13_PCREL_X:
    return "B13_PCREL_X";
  case B9_PCREL_X:
    return "B9_PCREL_X";
  case B7_PCREL_X:
    return "B7_PCREL_X";
  case Word6_X:
    return "Word6_X";
  case Word6_PCREL_X:
    return "Word6_PCREL_X";
  case Word8_X:
    return "Word8_X";
  case Word9_X:
    return "Word9_X";
  case Word10_X:
    return "Word10_X";
  case Word11_X:
    return "Word11_X";
  case Word12_X:
    return "Word12_X";
  case Word16_X:
    return "Word16_X";
  default:
    return getGenericEdgeKindName(K);
  }
}

} // namespace llvm::jitlink::hexagon
