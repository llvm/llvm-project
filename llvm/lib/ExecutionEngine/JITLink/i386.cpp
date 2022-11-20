//===---- i386.cpp - Generic JITLink i386 edge kinds, utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing i386 objects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/i386.h"

#define DEBUG_TYPE "jitlink"

namespace llvm::jitlink::i386 {

const char *getEdgeKindName(Edge::Kind K) {
  switch (K) {
  case None:
    return "None";
  case Pointer32:
    return "Pointer32";
  case PCRel32:
    return "PCRel32";
  case Pointer16:
    return "Pointer16";
  case PCRel16:
    return "PCRel16";
  }
  return getGenericEdgeKindName(K);
}

} // namespace llvm::jitlink::i386
