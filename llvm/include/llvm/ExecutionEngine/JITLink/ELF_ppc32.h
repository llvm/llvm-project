//===-- ELF_ppc32.h - JIT link functions for ELF/PPC32 -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_ELF_PPC32_H
#define LLVM_EXECUTIONENGINE_JITLINK_ELF_PPC32_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm::jitlink {

/// Create a LinkGraph from a big- or little-endian ELF/PPC32 relocatable
/// object. The caller must ensure that the underlying object buffer outlives
/// the graph.
LLVM_ABI Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_ppc32(MemoryBufferRef ObjectBuffer,
                                   std::shared_ptr<orc::SymbolStringPool> SSP);

/// Link an ELF/PPC32 graph, using the endianness recorded in the graph.
LLVM_ABI void link_ELF_ppc32(std::unique_ptr<LinkGraph> G,
                             std::unique_ptr<JITLinkContext> Ctx);

} // namespace llvm::jitlink

#endif // LLVM_EXECUTIONENGINE_JITLINK_ELF_PPC32_H
