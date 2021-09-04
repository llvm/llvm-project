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

/// Represets aarch64 fixups
enum EdgeKind_aarch64 : Edge::Kind {

  /// Set a CALL immediate field to bits [27:2] of X = Target - Fixup + Addend
  R_AARCH64_CALL26 = Edge::FirstRelocation,

};

/// Returns a string name for the given aarch64 edge. For debugging purposes
/// only
const char *getEdgeKindName(Edge::Kind K);

} // namespace aarch64
} // namespace jitlink
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_AARCH64_H
