//=== i386.h - Generic JITLink i386 edge kinds, utilities -*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_JITLINK_I386_H
#define LLVM_EXECUTIONENGINE_JITLINK_I386_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm {
namespace jitlink {
namespace i386 {

/// Represets i386 fixups
enum EdgeKind_i386 : Edge::Kind {

  /// None
  None = Edge::FirstRelocation,

};

/// Returns a string name for the given i386 edge. For debugging purposes
/// only
const char *getEdgeKindName(Edge::Kind K);

} // namespace i386
} // namespace jitlink
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_I386_H
