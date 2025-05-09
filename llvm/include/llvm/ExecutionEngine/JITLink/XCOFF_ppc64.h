//===------ XCOFF_ppc64.h - JIT link functions for XCOFF/ppc64 ------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// jit-link functions for XCOFF/ppc64.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_XCOFF_PPC64_H
#define LLVM_EXECUTIONENGINE_JITLINK_XCOFF_PPC64_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm::jitlink {

/// Create a LinkGraph from an XCOFF/ppc64 relocatable object.
///
/// Note: The graph does not take ownership of the underlying buffer, nor copy
/// its contents. The caller is responsible for ensuring that the object buffer
/// outlives the graph.
///
Expected<std::unique_ptr<LinkGraph>> createLinkGraphFromXCOFFObject_ppc64(
    MemoryBufferRef ObjectBuffer, std::shared_ptr<orc::SymbolStringPool> SSP);

/// jit-link the given object buffer, which must be a XCOFF ppc64 object file.
///
void link_XCOFF_ppc64(std::unique_ptr<LinkGraph> G,
                      std::unique_ptr<JITLinkContext> Ctx);

} // end namespace llvm::jitlink

#endif // LLVM_EXECUTIONENGINE_JITLINK_XCOFF_PPC64_H
