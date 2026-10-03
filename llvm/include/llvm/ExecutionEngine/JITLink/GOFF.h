//===------- GOFF.h - Generic JIT link function for GOFF ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// JIT-Link functions for GOFF.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_GOFF_H
#define LLVM_EXECUTIONENGINE_JITLINK_GOFF_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm {
namespace jitlink {

/// Create a LinkGraph from an GOFF relocatable object.
///
/// Note: The graph does not take ownership of the underlying buffer, nor copy
/// its contents. The caller is responsible for ensuring that the object buffer
/// outlives the graph.
LLVM_ABI Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromGOFFObject(MemoryBufferRef ObjectBuffer,
                              std::shared_ptr<orc::SymbolStringPool> SSP);

/// Link the given graph.
LLVM_ABI void link_GOFF(std::unique_ptr<LinkGraph> G,
                        std::unique_ptr<JITLinkContext> Ctx);

} // namespace jitlink
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_GOFF_H
