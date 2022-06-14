//===---- aarch64.cpp - Generic JITLink aarch64 edge kinds, utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing aarch64 objects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/aarch64.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {
namespace aarch64 {

const uint8_t NullGOTEntryContent[8] = {0x00, 0x00, 0x00, 0x00,
                                        0x00, 0x00, 0x00, 0x00};

const uint8_t StubContent[8] = {
    0x10, 0x00, 0x00, 0x58, // LDR x16, <literal>
    0x00, 0x02, 0x1f, 0xd6  // BR  x16
};

const char *getEdgeKindName(Edge::Kind R) {
  switch (R) {
  case Branch26:
    return "Branch26";
  case Pointer64:
    return "Pointer64";
  case Pointer64Anon:
    return "Pointer64Anon";
  case Page21:
    return "Page21";
  case PageOffset12:
    return "PageOffset12";
  case GOTPage21:
    return "GOTPage21";
  case GOTPageOffset12:
    return "GOTPageOffset12";
  case TLVPage21:
    return "TLVPage21";
  case TLVPageOffset12:
    return "TLVPageOffset12";
  case PointerToGOT:
    return "PointerToGOT";
  case PairedAddend:
    return "PairedAddend";
  case LDRLiteral19:
    return "LDRLiteral19";
  case Delta32:
    return "Delta32";
  case Delta64:
    return "Delta64";
  case NegDelta32:
    return "NegDelta32";
  case NegDelta64:
    return "NegDelta64";
  default:
    return getGenericEdgeKindName(static_cast<Edge::Kind>(R));
  }
}

} // namespace aarch64
} // namespace jitlink
} // namespace llvm
