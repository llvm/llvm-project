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
#include "JITLinkGeneric.h"
#include "XCOFFLinkGraphBuilder.h"
#include "llvm/ADT/bit.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/ppc64.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
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

class XCOFFJITLinker_ppc64 : public JITLinker<XCOFFJITLinker_ppc64> {
  using JITLinkerBase = JITLinker<XCOFFJITLinker_ppc64>;
  friend JITLinkerBase;

public:
  XCOFFJITLinker_ppc64(std::unique_ptr<JITLinkContext> Ctx,
                       std::unique_ptr<LinkGraph> G,
                       PassConfiguration PassConfig)
      : JITLinkerBase(std::move(Ctx), std::move(G), std::move(PassConfig)) {
    // FIXME: Post allocation pass define TOC base, this is temporary to support
    // building until we can build the required toc entries
    defineTOCSymbol(getGraph());
  }

  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    LLVM_DEBUG(dbgs() << "  Applying fixup for " << G.getName()
                      << ", address = " << B.getAddress()
                      << ", target = " << E.getTarget().getName() << ", kind = "
                      << ppc64::getEdgeKindName(E.getKind()) << "\n");
    switch (E.getKind()) {
    case ppc64::Pointer64:
      if (auto Err = ppc64::applyFixup<endianness::big>(G, B, E, TOCSymbol))
        return Err;
      break;
    default:
      return make_error<StringError>("Unsupported relocation type",
                                     std::error_code());
    }
    return Error::success();
  }

private:
  void defineTOCSymbol(LinkGraph &G) {
    for (Symbol *S : G.defined_symbols()) {
      if (S->hasName() && *S->getName() == StringRef("TOC")) {
        TOCSymbol = S;
        return;
      }
    }
    llvm_unreachable("LinkGraph does not contan an TOC Symbol");
  }

private:
  Symbol *TOCSymbol = nullptr;
};

void link_XCOFF_ppc64(std::unique_ptr<LinkGraph> G,
                      std::unique_ptr<JITLinkContext> Ctx) {
  // Ctx->notifyFailed(make_error<StringError>(
  //     "link_XCOFF_ppc64 is not implemented", std::error_code()));

  PassConfiguration Config;

  // Pass insertions

  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  XCOFFJITLinker_ppc64::link(std::move(Ctx), std::move(G), std::move(Config));
}

} // namespace jitlink
} // namespace llvm
