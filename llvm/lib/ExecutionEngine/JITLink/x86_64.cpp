//===----- x86_64.cpp - Generic JITLink x86-64 edge kinds, utilities ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing x86-64 objects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/x86_64.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {
namespace x86_64 {

const char *getEdgeKindName(Edge::Kind K) {
  switch (K) {
  case Pointer64:
    return "Pointer64";
  case Pointer32:
    return "Pointer32";
  case Delta64:
    return "Delta64";
  case Delta32:
    return "Delta32";
  case NegDelta64:
    return "NegDelta64";
  case NegDelta32:
    return "NegDelta32";
  case Delta64FromGOT:
    return "Delta64FromGOT";
  case BranchPCRel32:
    return "BranchPCRel32";
  case BranchPCRel32ToPtrJumpStub:
    return "BranchPCRel32ToPtrJumpStub";
  case BranchPCRel32ToPtrJumpStubBypassable:
    return "BranchPCRel32ToPtrJumpStubBypassable";
  case RequestGOTAndTransformToDelta32:
    return "RequestGOTAndTransformToDelta32";
  case RequestGOTAndTransformToDelta64:
    return "RequestGOTAndTransformToDelta64";
  case RequestGOTAndTransformToDelta64FromGOT:
    return "RequestGOTAndTransformToDelta64FromGOT";
  case PCRel32GOTLoadREXRelaxable:
    return "PCRel32GOTLoadREXRelaxable";
  case RequestGOTAndTransformToPCRel32GOTLoadREXRelaxable:
    return "RequestGOTAndTransformToPCRel32GOTLoadREXRelaxable";
  case PCRel32TLVPLoadREXRelaxable:
    return "PCRel32TLVPLoadREXRelaxable";
  case RequestTLVPAndTransformToPCRel32TLVPLoadREXRelaxable:
    return "RequestTLVPAndTransformToPCRel32TLVPLoadREXRelaxable";
  default:
    return getGenericEdgeKindName(static_cast<Edge::Kind>(K));
  }
}

const char NullPointerContent[PointerSize] = {0x00, 0x00, 0x00, 0x00,
                                              0x00, 0x00, 0x00, 0x00};

const char PointerJumpStubContent[6] = {
    static_cast<char>(0xFFu), 0x25, 0x00, 0x00, 0x00, 0x00};

} // end namespace x86_64
} // end namespace jitlink
} // end namespace llvm
