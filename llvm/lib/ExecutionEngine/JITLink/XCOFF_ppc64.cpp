//===------- XCOFF_ppc64.cpp -JIT linker implementation for XCOFF/ppc64
//-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// XCOFF/ppc64 jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/XCOFF_ppc64.h"
#include "XCOFFLinkGraphBuilder.h"
#include "llvm/ExecutionEngine/JITLink/ppc64.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/Support/Error.h"
#include <system_error>

using namespace llvm;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

Expected<std::unique_ptr<LinkGraph>> createLinkGraphFromXCOFFObject_ppc64(
    MemoryBufferRef ObjectBuffer, std::shared_ptr<orc::SymbolStringPool> SSP) {
  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << ObjectBuffer.getBufferIdentifier() << "...\n";
  });

  auto Obj = object::ObjectFile::createObjectFile(ObjectBuffer);
  if (!Obj)
    return Obj.takeError();
  assert((**Obj).isXCOFF() && "Expects and XCOFF Object");

  auto Features = (*Obj)->getFeatures();
  if (!Features)
    return Features.takeError();
  LLVM_DEBUG({
    dbgs() << " Features: ";
    (*Features).print(dbgs());
  });

  return XCOFFLinkGraphBuilder(cast<object::XCOFFObjectFile>(**Obj),
                               std::move(SSP), Triple("powerpc64-ibm-aix"),
                               std::move(*Features), ppc64::getEdgeKindName)
      .buildGraph();
}

void link_XCOFF_ppc64(std::unique_ptr<LinkGraph> G,
                      std::unique_ptr<JITLinkContext> Ctx) {
  Ctx->notifyFailed(make_error<StringError>(
      "link_XCOFF_ppc64 is not implemented", std::error_code()));
}

} // namespace jitlink
} // namespace llvm
