//===--- MachO_x86_64.h - JIT link functions for MachO/x86-64 ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// jit-link functions for MachO/x86-64.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_MACHO_X86_64_H
#define LLVM_EXECUTIONENGINE_JITLINK_MACHO_X86_64_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm {
namespace jitlink {

namespace MachO_x86_64_Edges {

enum MachOX86RelocationKind : Edge::Kind {
  Branch32 = Edge::FirstRelocation,
  Pointer32,
  Pointer64,
  Pointer64Anon,
  PCRel32,
  PCRel32Minus1,
  PCRel32Minus2,
  PCRel32Minus4,
  PCRel32Anon,
  PCRel32Minus1Anon,
  PCRel32Minus2Anon,
  PCRel32Minus4Anon,
  PCRel32GOTLoad,
  PCRel32GOT,
  PCRel32TLV,
  Delta32,
  Delta64,
  NegDelta32,
  NegDelta64,
};

} // namespace MachO_x86_64_Edges

/// jit-link the given object buffer, which must be a MachO x86-64 object file.
///
/// If PrePrunePasses is empty then a default mark-live pass will be inserted
/// that will mark all exported atoms live. If PrePrunePasses is not empty, the
/// caller is responsible for including a pass to mark atoms as live.
///
/// If PostPrunePasses is empty then a default GOT-and-stubs insertion pass will
/// be inserted. If PostPrunePasses is not empty then the caller is responsible
/// for including a pass to insert GOT and stub edges.
void jitLink_MachO_x86_64(std::unique_ptr<JITLinkContext> Ctx);

/// Return the string name of the given MachO x86-64 edge kind.
StringRef getMachOX86RelocationKindName(Edge::Kind R);

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_MACHO_X86_64_H
