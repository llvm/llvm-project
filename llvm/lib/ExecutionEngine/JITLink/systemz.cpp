//===---- systemz.cpp - Generic JITLink systemz edge kinds, utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing systemz objects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/systemz.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {
namespace systemz {

const char NullPointerContent[8] = {0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x00, 0x00};

const char Pointer64JumpStubContent[8] = {
    static_cast<char>(0xC4u),
    0x18,
    0x00,
    0x00,
    0x00,
    0x00, // lgrl r1
    static_cast<char>(0x07u),
    static_cast<char>(0xF1u), // BCR 15, 1
};

const char *getEdgeKindName(Edge::Kind R) {
  switch (R) {
  case Pointer64:
    return "Pointer64";
  case Pointer32:
    return "Pointer32";
  case Pointer20:
    return "Pointer20";
  case Pointer16:
    return "Pointer16";
  case Pointer12:
    return "Pointer12";
  case Pointer8:
    return "Pointer8";
  case Delta64:
    return "Delta64";
  case Delta32:
    return "Delta32";
  case Delta16:
    return "Delta16";
  case Delta32dbl:
    return "Delta32dbl";
  case Delta24dbl:
    return "Delta24dbl";
  case Delta16dbl:
    return "Delta16dbl";
  case Delta12dbl:
    return "Delta12dbl";
  case NegDelta64:
    return "NegDelta64";
  case NegDelta32:
    return "NegDelta32";
  case DeltaPLT32dbl:
    return "DeltaPLT32dbl";
  case DeltaPLT24dbl:
    return "DeltaPLT24dbl";
  case DeltaPLT16dbl:
    return "DeltaPLT16dbl";
  case DeltaPLT12dbl:
    return "DeltaPLT12dbl";
  case DeltaPLT64:
    return "DeltaPLT64";
  case DeltaPLT32:
    return "DeltaPLT32";
  case Delta64FromGOT:
    return "Delta64FromGOT";
  case Delta32FromGOT:
    return "Delta32FromGOT";
  case Delta16FromGOT:
    return "Delta16FromGOT";
  case Delta64PLTFromGOT:
    return "Delta64PLTFromGOT";
  case Delta32PLTFromGOT:
    return "Delta32PLTFromGOT";
  case Delta16PLTFromGOT:
    return "Delta16PLTFromGOT";
  case Delta32GOTBase:
    return "Delta32GOTBase";
  case Delta32dblGOTBase:
    return "Delta32dblGOTBase";
  case RequestGOTAndTransformToDelta64FromGOT:
    return "RequestGOTAndTransformToDelta64FromGOT";
  case RequestGOTAndTransformToDelta32FromGOT:
    return "RequestGOTAndTransformToDelta32FromGOT";
  case RequestGOTAndTransformToDelta20FromGOT:
    return "RequestGOTAndTransformToDelta20FromGOT";
  case RequestGOTAndTransformToDelta16FromGOT:
    return "RequestGOTAndTransformToDelta16FromGOT";
  case RequestGOTAndTransformToDelta12FromGOT:
    return "RequestGOTAndTransformToDelta12FromGOT";
  case RequestGOTAndTransformToDelta32dbl:
    return "RequestGOTAndTransformToDelta32dbl";
  default:
    return getGenericEdgeKindName(static_cast<Edge::Kind>(R));
  }
}

} // namespace systemz
} // namespace jitlink
} // namespace llvm
