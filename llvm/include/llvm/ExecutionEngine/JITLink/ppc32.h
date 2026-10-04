//===-- ppc32.h - Generic JITLink PPC32 edges ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_PPC32_H
#define LLVM_EXECUTIONENGINE_JITLINK_PPC32_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm::jitlink::ppc32 {

/// PPC32 fixups and GOT-entry requests, shared by both endiannesses.
enum EdgeKind_ppc32 : Edge::Kind {
  /// Write Target + Addend as an unsigned 32-bit pointer.
  Pointer32 = Edge::FirstRelocation,
  /// Write the low 16 bits of Target + Addend.
  Pointer16,
  /// Write the high 16 bits of Target + Addend.
  Pointer16HI,
  /// Write the high 16 bits of Target + Addend + 0x8000.
  Pointer16HA,
  /// Write Target + Addend - Fixup as a signed 32-bit displacement.
  Delta32,
  /// Write Fixup - (Target + Addend) as a 32-bit displacement.
  NegDelta32,
  /// Low, high, and adjusted high halves of Target + Addend - Fixup.
  Delta16,
  Delta16HI,
  Delta16HA,
  /// Relative branches with signed 26-bit and 16-bit byte displacements.
  /// The low two bits must be zero; other instruction bits are preserved.
  Branch24,
  Branch14,
  /// Absolute forms of Branch24 and Branch14.
  Branch24Absolute,
  Branch14Absolute,
  // A GOT entry's displacement from the GOT base. GOTDelta16 requires a signed
  // 16-bit displacement; the split forms select the low, high, or adjusted
  // high half of the displacement.
  GOTDelta16,
  GOTDelta16LO,
  GOTDelta16HI,
  GOTDelta16HA,
  // Request a GOT entry for the target, then apply the corresponding GOT
  // displacement fixup to the entry rather than the original target.
  RequestGOTAndTransformToGOTDelta16,
  RequestGOTAndTransformToGOTDelta16LO,
  RequestGOTAndTransformToGOTDelta16HI,
  RequestGOTAndTransformToGOTDelta16HA,
};

LLVM_ABI const char *getEdgeKindName(Edge::Kind K);
/// Apply a fixup. GOTSymbol is required for GOTDelta16 and its split forms.
LLVM_ABI Error applyFixup(LinkGraph &G, Block &B, const Edge &E,
                          const Symbol *GOTSymbol = nullptr);

/// Create a four-byte pointer, optionally initialized to a target plus addend.
LLVM_ABI Symbol &createAnonymousPointer(LinkGraph &G, Section &PointerSection,
                                        Symbol *InitialTarget = nullptr,
                                        Edge::AddendT InitialAddend = 0);
/// Create a jump stub that loads PointerSymbol and branches through CTR.
/// The stub clobbers r12 and CTR and preserves LR.
LLVM_ABI Symbol &createAnonymousPointerJumpStub(LinkGraph &G,
                                                Section &StubSection,
                                                Symbol &PointerSymbol);

} // namespace llvm::jitlink::ppc32

#endif // LLVM_EXECUTIONENGINE_JITLINK_PPC32_H
