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
  R_AMD64_NONE = Edge::FirstRelocation,
  R_AMD64_64,
  R_AMD64_PC32,
  R_AMD64_GOT32,
  R_AMD64_PLT32,
  R_AMD64_COPY,
  R_AMD64_GLOB_DAT,
  R_AMD64_JUMP_SLOT,
  R_AMD64_RELATIVE,
  R_AMD64_GOTPCREL,
  R_AMD64_32,
  R_AMD64_32S,
  R_AMD64_16,
  R_AMD64_PC16,
  R_AMD64_8,
  R_AMD64_PC8,
  R_AMD64_PC64,
  R_AMD64_GOTOFF64,
  R_AMD64_GOTPC32,
  R_AMD64_SIZE32,
  R_AMD64_SIZE64
};

} // end namespace ELF_x86_64_Edges

/// jit-link the given object buffer, which must be a ELF x86-64 object file.
void jitLink_ELF_x86_64(std::unique_ptr<JITLinkContext> Ctx);
StringRef getELFX86RelocationKindName(Edge::Kind R);
} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_ELF_X86_64_H
