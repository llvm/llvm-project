//===--------------- GOFF.cpp - JIT linker function for GOFF --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// JIT-link functions for GOFF.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/GOFF.h"
#include "llvm/ExecutionEngine/JITLink/GOFF_systemz.h"
#include "llvm/Object/GOFFObjectFile.h"

using namespace llvm;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromGOFFObject(MemoryBufferRef ObjectBuffer,
                              std::shared_ptr<orc::SymbolStringPool> SSP) {
  return createLinkGraphFromGOFFObject_systemz(ObjectBuffer, std::move(SSP));
}

void link_GOFF(std::unique_ptr<LinkGraph> G,
               std::unique_ptr<JITLinkContext> Ctx) {
  link_GOFF_systemz(std::move(G), std::move(Ctx));
}

} // namespace jitlink
} // namespace llvm
