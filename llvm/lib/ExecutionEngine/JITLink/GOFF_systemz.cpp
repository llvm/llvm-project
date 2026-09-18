//===----------- GOFF_systemz.cpp - JIT linker function for GOFF ----------===//
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

#include "llvm/ExecutionEngine/JITLink/GOFF_systemz.h"
#include "GOFFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"
#include "llvm/ExecutionEngine/JITLink/GOFF.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/systemz.h"
#include "llvm/Object/GOFFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <system_error>

using namespace llvm;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

class GOFFJITLinker_systemz : public JITLinker<GOFFJITLinker_systemz> {
  using JITLinkerBase = JITLinker<GOFFJITLinker_systemz>;
  friend JITLinkerBase;

public:
  GOFFJITLinker_systemz(std::unique_ptr<JITLinkContext> Ctx,
                        std::unique_ptr<LinkGraph> G,
                        PassConfiguration PassConfig)
      : JITLinkerBase(std::move(Ctx), std::move(G), std::move(PassConfig)) {}

private:
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    if (auto Err = systemz::applyGOFFFixup(G, B, E)) {
      return make_error<StringError>("Unsupported goff relocation type",
                                     std::error_code());
    }
    return Error::success();
  }
};

Expected<std::unique_ptr<LinkGraph>> createLinkGraphFromGOFFObject_systemz(
    MemoryBufferRef ObjectBuffer, std::shared_ptr<orc::SymbolStringPool> SSP) {
  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << ObjectBuffer.getBufferIdentifier() << "...\n";
  });

  file_magic Magic = identify_magic(ObjectBuffer.getBuffer());
  if (Magic != file_magic::goff_object)
    return make_error<JITLinkError>("Invalid GOFF Header");

  auto GOFFObj = object::ObjectFile::createObjectFile(ObjectBuffer);
  if (!GOFFObj)
    return GOFFObj.takeError();
  assert((*GOFFObj)->isGOFF() && "Expects an GOFF Object");
  assert((*GOFFObj)->getArch() == Triple::systemz && "Only support systemz");

  auto Features = (*GOFFObj)->getFeatures();
  if (!Features)
    return Features.takeError();
  LLVM_DEBUG({
    dbgs() << " Features: ";
    (*Features).print(dbgs());
  });

  return GOFFLinkGraphBuilder(cast<object::GOFFObjectFile>(**GOFFObj),
                              std::move(SSP), (*GOFFObj)->makeTriple(),
                              std::move(*Features), systemz::getEdgeKindName)
      .buildGraph();
}

void link_GOFF_systemz(std::unique_ptr<LinkGraph> G,
                       std::unique_ptr<JITLinkContext> Ctx) {

  PassConfiguration PassCfg;
  if (auto Err = Ctx->modifyPassConfig(*G, PassCfg))
    return Ctx->notifyFailed(std::move(Err));

  if (G->getTargetTriple().getArch() != Triple::systemz)
    return Ctx->notifyFailed(make_error<JITLinkError>(
        "Unsupported target machine architecture in GOFF link graph " +
        G->getName()));

  GOFFJITLinker_systemz::link(std::move(Ctx), std::move(G), std::move(PassCfg));
  return;
}

} // namespace jitlink
} // namespace llvm
