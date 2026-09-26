//===-- ELF_mips.h - JIT link functions for ELF/MIPS ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_ELF_MIPS_H
#define LLVM_EXECUTIONENGINE_JITLINK_ELF_MIPS_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm {
namespace jitlink {

/// Create a LinkGraph from an O32, N32, or N64 ELF/MIPS relocatable object.
/// Both ELF byte orders are supported.
LLVM_ABI Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_mips(MemoryBufferRef ObjectBuffer,
                                  std::shared_ptr<orc::SymbolStringPool> SSP);

/// JIT-link an ELF/MIPS graph.
LLVM_ABI void link_ELF_mips(std::unique_ptr<LinkGraph> G,
                            std::unique_ptr<JITLinkContext> Ctx);

} // namespace jitlink
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_ELF_MIPS_H
