//=== aarch64.h - Generic JITLink aarch64 edge kinds, utilities -*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_JITLINK_AARCH64_H
#define LLVM_EXECUTIONENGINE_JITLINK_AARCH64_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm {
namespace jitlink {
namespace aarch64 {

enum EdgeKind_aarch64 : Edge::Kind {
  Branch26 = Edge::FirstRelocation,
  Pointer32,
  Pointer64,
  Pointer64Anon,
  Page21,
  PageOffset12,
  GOTPage21,
  GOTPageOffset12,
  TLVPage21,
  TLVPageOffset12,
  PointerToGOT,
  PairedAddend,
  LDRLiteral19,
  Delta32,
  Delta64,
  NegDelta32,
  NegDelta64,
};

/// Returns a string name for the given aarch64 edge. For debugging purposes
/// only
const char *getEdgeKindName(Edge::Kind K);

unsigned getPageOffset12Shift(uint32_t Instr);

Error applyFixup(LinkGraph &G, Block &B, const Edge &E);

} // namespace aarch64
} // namespace jitlink
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_AARCH64_H
