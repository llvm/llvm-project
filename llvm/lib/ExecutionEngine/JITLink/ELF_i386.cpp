//===----- ELF_i386.cpp - JIT linker implementation for ELF/i386 ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ELF/i386 jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF_i386.h"
#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/i386.h"
#include "llvm/Object/ELFObjectFile.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;

namespace llvm {
namespace jitlink {

class ELFJITLinker_i386 : public JITLinker<ELFJITLinker_i386> {
  friend class JITLinker<ELFJITLinker_i386>;

public:
  ELFJITLinker_i386(std::unique_ptr<JITLinkContext> Ctx,
                    std::unique_ptr<LinkGraph> G, PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {}

private:
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    using namespace i386;
    using namespace llvm::support;

    switch (E.getKind()) {
    case i386::None: {
      break;
    }
    }
    return Error::success();
  }
};

template <typename ELFT>
class ELFLinkGraphBuilder_i386 : public ELFLinkGraphBuilder<ELFT> {
private:
  static Expected<i386::EdgeKind_i386> getRelocationKind(const uint32_t Type) {
    using namespace i386;
    switch (Type) {
    case ELF::R_386_NONE:
      return EdgeKind_i386::None;
    }

    return make_error<JITLinkError>("Unsupported i386 relocation:" +
                                    formatv("{0:d}", Type));
  }

  Error addRelocations() override {
    LLVM_DEBUG(dbgs() << "Adding relocations\n");
    return Error::success();
  }

public:
  ELFLinkGraphBuilder_i386(StringRef FileName, const object::ELFFile<ELFT> &Obj,
                           const Triple T)
      : ELFLinkGraphBuilder<ELFT>(Obj, std::move(T), FileName,
                                  i386::getEdgeKindName) {}
};

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_i386(MemoryBufferRef ObjectBuffer) {
  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << ObjectBuffer.getBufferIdentifier() << "...\n";
  });

  auto ELFObj = object::ObjectFile::createELFObjectFile(ObjectBuffer);
  if (!ELFObj)
    return ELFObj.takeError();

  assert((*ELFObj)->getArch() == Triple::x86 &&
         "Only i386 (little endian) is supported for now");

  auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF32LE>>(**ELFObj);
  return ELFLinkGraphBuilder_i386<object::ELF32LE>((*ELFObj)->getFileName(),
                                                   ELFObjFile.getELFFile(),
                                                   (*ELFObj)->makeTriple())
      .buildGraph();
}

void link_ELF_i386(std::unique_ptr<LinkGraph> G,
                   std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;
  const Triple &TT = G->getTargetTriple();
  if (Ctx->shouldAddDefaultTargetPasses(TT)) {
    if (auto MarkLive = Ctx->getMarkLivePass(TT))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);
  }
  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  ELFJITLinker_i386::link(std::move(Ctx), std::move(G), std::move(Config));
}

} // namespace jitlink
} // namespace llvm
