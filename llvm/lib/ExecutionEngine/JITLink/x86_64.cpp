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
  case BranchPCRel32:
    return "BranchPCRel32";
  case BranchPCRel32ToPtrJumpStub:
    return "BranchPCRel32ToPtrJumpStub";
  case BranchPCRel32ToPtrJumpStubRelaxable:
    return "BranchPCRel32ToPtrJumpStubRelaxable";
  case RequestGOTAndTransformToDelta32:
    return "RequestGOTAndTransformToDelta32";
  case PCRel32GOTLoadRelaxable:
    return "PCRel32GOTLoadRelaxable";
  case RequestGOTAndTransformToPCRel32GOTLoadRelaxable:
    return "RequestGOTAndTransformToPCRel32GOTLoadRelaxable";
  case PCRel32TLVPLoadRelaxable:
    return "PCRel32TLVPLoadRelaxable";
  case RequestTLVPAndTransformToPCRel32TLVPLoadRelaxable:
    return "RequestTLVPAndTransformToPCRel32TLVPLoadRelaxable";
  default:
    return getGenericEdgeKindName(static_cast<Edge::Kind>(K));
  }
}

} // end namespace x86_64
} // end namespace jitlink
} // end namespace llvm
