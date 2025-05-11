//===-------------- XCOFF.cpp - JIT linker function for XCOFF -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// XCOFF jit-link function.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/XCOFF.h"
#include "llvm/ExecutionEngine/JITLink/XCOFF_ppc64.h"
#include "llvm/Object/XCOFFObjectFile.h"

using namespace llvm;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromXCOFFObject(MemoryBufferRef ObjectBuffer,
                               std::shared_ptr<orc::SymbolStringPool> SSP) {
  // Check magic
  file_magic Magic = identify_magic(ObjectBuffer.getBuffer());
  if (Magic != file_magic::xcoff_object_64)
    return make_error<JITLinkError>("Invalid XCOFF 64 Header");

  // TODO: See if we need to add more checks
  //
  return createLinkGraphFromXCOFFObject_ppc64(ObjectBuffer, std::move(SSP));
}

void link_XCOFF(std::unique_ptr<LinkGraph> G,
                std::unique_ptr<JITLinkContext> Ctx) {
  link_XCOFF_ppc64(std::move(G), std::move(Ctx));
}

} // namespace jitlink
} // namespace llvm
