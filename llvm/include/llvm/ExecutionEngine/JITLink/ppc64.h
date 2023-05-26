//===--- ppc64.h - Generic JITLink ppc64 edge kinds, utilities --*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_JITLINK_PPC64_H
#define LLVM_EXECUTIONENGINE_JITLINK_PPC64_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm::jitlink::ppc64 {

/// Represents ppc64 fixups and other ppc64-specific edge kinds.
/// TODO: Add edge kinds.
enum EdgeKind_ppc64 : Edge::Kind {};

/// Returns a string name for the given ppc64 edge. For debugging purposes
/// only.
const char *getEdgeKindName(Edge::Kind K);

/// Apply fixup expression for edge to block content.
/// TOOD: Add fixups as we add edges.
inline Error applyFixup(LinkGraph &G, Block &B, const Edge &E,
                        const Symbol *GOTSymbol) {
  return make_error<JITLinkError>(
      "In graph " + G.getName() + ", section " + B.getSection().getName() +
      " unsupported edge kind " + getEdgeKindName(E.getKind()));
}

} // end namespace llvm::jitlink::ppc64

#endif // LLVM_EXECUTIONENGINE_JITLINK_PPC64_H
