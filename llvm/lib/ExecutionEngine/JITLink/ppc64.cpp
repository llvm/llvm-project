//===----- ppc64.cpp - Generic JITLink ppc64 edge kinds, utilities ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing 64-bit PowerPC objects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ppc64.h"

#define DEBUG_TYPE "jitlink"

namespace llvm::jitlink::ppc64 {

const char *getEdgeKindName(Edge::Kind K) {
  // TODO: Add edge names.
  return getGenericEdgeKindName(static_cast<Edge::Kind>(K));
}

} // end namespace llvm::jitlink::ppc64
