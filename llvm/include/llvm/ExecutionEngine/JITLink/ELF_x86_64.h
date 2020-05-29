//===--- ELF_x86_64.h - JIT link functions for ELF/x86-64 ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// jit-link functions for ELF/x86-64.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_ELF_X86_64_H
#define LLVM_EXECUTIONENGINE_JITLINK_ELF_X86_64_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm {
namespace jitlink {

namespace ELF_x86_64_Edges {
enum ELFX86RelocationKind : Edge::Kind {
  Branch32 = Edge::FirstRelocation,
  Branch32ToStub,
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

} // end namespace ELF_x86_64_Edges

/// jit-link the given object buffer, which must be a ELF x86-64 object file.
void jitLink_ELF_x86_64(std::unique_ptr<JITLinkContext> Ctx);
} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_ELF_X86_64_H
