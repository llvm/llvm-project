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

const char Pointer64JumpStubContent[14] = {
    static_cast<char>(0xC0), 0x10, 0x00, 0x00, 0x00, 0x00, // larl r1
    static_cast<char>(0xE3), 0x10, 0x10, 0x00, 0x00, 0x04, // LG 1, 0(1)
    static_cast<char>(0x07), 0xF1,                         // BCR 15, 1
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
  case Delta64FromGOT:
    return "Delta64FromGOT";
  case Delta32FromGOT:
    return "Delta32FromGOT";
  case Delta16FromGOT:
    return "Delta16FromGOT";
  case BranchPCRelPLT32dbl:
    return "BranchPCRelPLT32dbl";
  case BranchPCRelPLT24dbl:
    return "BranchPCRelPLT24dbl";
  case BranchPCRelPLT16dbl:
    return "BranchPCRelPLT16dbl";
  case BranchPCRelPLT12dbl:
    return "BranchPCRelPLT12dbl";
  case PCRel32GOTEntry:
    return "PCRel32GOTENTRY";
  case BranchPCRelPLT64:
    return "BranchPCRelPLT64";
  case BranchPCRelPLT32:
    return "BranchPCRelPLT32";
  case DeltaPLT64FromGOT:
    return "DeltaPLT64FromGOT";
  case DeltaPLT32FromGOT:
    return "DeltaPLT32FromGOT";
  case DeltaPLT16FromGOT:
    return "DeltaPLT16FromGOT";
  case Delta64GOT:
    return "Delta64GOT";
  case Delta32GOT:
    return "Delta32GOT";
  case Delta20GOT:
    return "Delta20GOT";
  case Delta16GOT:
    return "Delta16GOT";
  case Delta12GOT:
    return "Delta12GOT";
  case DeltaPCRelGOT:
    return "DeltaPCRelGOT";
  case DeltaPCRelGOTdbl:
    return "DeltaPCRelGOTdbl";
  case Delta64JumpSlot:
    return "Delta64JumpSlot";
  case Delta32JumpSlot:
    return "Delta32JumpSlot";
  case Delta20JumpSlot:
    return "Delta20JumpSlot";
  case Delta16JumpSlot:
    return "Delta16JumpSlot";
  case Delta12JumpSlot:
    return "Delta12JumpSlot";
  case PCRel32JumpSlot:
    return "PCRel32JumpSlot";
  default:
    return getGenericEdgeKindName(static_cast<Edge::Kind>(R));
  }
}

} // namespace systemz
} // namespace jitlink
} // namespace llvm
