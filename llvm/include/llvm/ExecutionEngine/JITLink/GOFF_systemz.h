//===--- GOFF_systemz.h -  JIT link functions for GOFF/systemz ---*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// JIT-Link functions for GOFF/systemz.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_GOFF_SYSTEMZ_H
#define LLVM_EXECUTIONENGINE_JITLINK_GOFF_SYSTEMZ_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm::jitlink {

/// Create a LinkGraph from an GOFF/systemz relocatable object.
///
/// Note: The graph does not take ownership of the underlying buffer, nor copy
/// its contents. The caller is responsible for ensuring that the object buffer
/// outlives the graph.
///
LLVM_ABI Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromGOFFObject_systemz(
    MemoryBufferRef ObjectBuffer, std::shared_ptr<orc::SymbolStringPool> SSP);

/// jit-link the given object buffer, which must be a GOFF systemz object file.
///
LLVM_ABI void link_GOFF_systemz(std::unique_ptr<LinkGraph> G,
                                std::unique_ptr<JITLinkContext> Ctx);

} // namespace llvm::jitlink

#endif // LLVM_EXECUTIONENGINE_JITLINK_GOFF_SYSTEMZ_H
