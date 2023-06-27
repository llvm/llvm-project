//===------- ELF_ppc64.cpp -JIT linker implementation for ELF/ppc64 -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ELF/ppc64 jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF_ppc64.h"
#include "llvm/ExecutionEngine/JITLink/ppc64.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Endian.h"

#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"

#define DEBUG_TYPE "jitlink"

namespace llvm::jitlink {

template <support::endianness Endianness>
class ELFLinkGraphBuilder_ppc64
    : public ELFLinkGraphBuilder<object::ELFType<Endianness, true>> {
private:
  using ELFT = object::ELFType<Endianness, true>;
  using Base = ELFLinkGraphBuilder<ELFT>;

  using Base::G; // Use LinkGraph pointer from base class.

  Error addRelocations() override {
    LLVM_DEBUG(dbgs() << "Processing relocations:\n");

    using Self = ELFLinkGraphBuilder_ppc64<Endianness>;
    for (const auto &RelSect : Base::Sections) {
      // Validate the section to read relocation entries from.
      if (RelSect.sh_type == ELF::SHT_REL)
        return make_error<StringError>("No SHT_REL in valid " +
                                           G->getTargetTriple().getArchName() +
                                           " ELF object files",
                                       inconvertibleErrorCode());

      if (Error Err = Base::forEachRelaRelocation(RelSect, this,
                                                  &Self::addSingleRelocation))
        return Err;
    }

    return Error::success();
  }

  Error addSingleRelocation(const typename ELFT::Rela &Rel,
                            const typename ELFT::Shdr &FixupSection,
                            Block &BlockToFix) {
    auto ELFReloc = Rel.getType(false);
    return make_error<JITLinkError>(
        "In " + G->getName() + ": Unsupported ppc64 relocation type " +
        object::getELFRelocationTypeName(ELF::EM_PPC64, ELFReloc));
  }

public:
  ELFLinkGraphBuilder_ppc64(StringRef FileName,
                            const object::ELFFile<ELFT> &Obj, Triple TT,
                            SubtargetFeatures Features)
      : ELFLinkGraphBuilder<ELFT>(Obj, std::move(TT), std::move(Features),
                                  FileName, ppc64::getEdgeKindName) {}
};

template <support::endianness Endianness>
class ELFJITLinker_ppc64 : public JITLinker<ELFJITLinker_ppc64<Endianness>> {
  using JITLinkerBase = JITLinker<ELFJITLinker_ppc64<Endianness>>;
  friend JITLinkerBase;

public:
  ELFJITLinker_ppc64(std::unique_ptr<JITLinkContext> Ctx,
                     std::unique_ptr<LinkGraph> G, PassConfiguration PassConfig)
      : JITLinkerBase(std::move(Ctx), std::move(G), std::move(PassConfig)) {}

private:
  Symbol *GOTSymbol = nullptr;

  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    return ppc64::applyFixup(G, B, E, GOTSymbol);
  }
};

template <support::endianness Endianness>
Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_ppc64(MemoryBufferRef ObjectBuffer) {
  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << ObjectBuffer.getBufferIdentifier() << "...\n";
  });

  auto ELFObj = object::ObjectFile::createELFObjectFile(ObjectBuffer);
  if (!ELFObj)
    return ELFObj.takeError();

  auto Features = (*ELFObj)->getFeatures();
  if (!Features)
    return Features.takeError();

  using ELFT = object::ELFType<Endianness, true>;
  auto &ELFObjFile = cast<object::ELFObjectFile<ELFT>>(**ELFObj);
  return ELFLinkGraphBuilder_ppc64<Endianness>(
             (*ELFObj)->getFileName(), ELFObjFile.getELFFile(),
             (*ELFObj)->makeTriple(), std::move(*Features))
      .buildGraph();
}

template <support::endianness Endianness>
void link_ELF_ppc64(std::unique_ptr<LinkGraph> G,
                    std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;

  if (Ctx->shouldAddDefaultTargetPasses(G->getTargetTriple())) {
    // Construct a JITLinker and run the link function.
    // Add a mark-live pass.
    if (auto MarkLive = Ctx->getMarkLivePass(G->getTargetTriple()))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);
  }

  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  ELFJITLinker_ppc64<Endianness>::link(std::move(Ctx), std::move(G),
                                       std::move(Config));
}

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_ppc64(MemoryBufferRef ObjectBuffer) {
  return createLinkGraphFromELFObject_ppc64<support::big>(
      std::move(ObjectBuffer));
}

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_ppc64le(MemoryBufferRef ObjectBuffer) {
  return createLinkGraphFromELFObject_ppc64<support::little>(
      std::move(ObjectBuffer));
}

/// jit-link the given object buffer, which must be a ELF ppc64 object file.
void link_ELF_ppc64(std::unique_ptr<LinkGraph> G,
                    std::unique_ptr<JITLinkContext> Ctx) {
  return link_ELF_ppc64<support::big>(std::move(G), std::move(Ctx));
}

/// jit-link the given object buffer, which must be a ELF ppc64le object file.
void link_ELF_ppc64le(std::unique_ptr<LinkGraph> G,
                      std::unique_ptr<JITLinkContext> Ctx) {
  return link_ELF_ppc64<support::little>(std::move(G), std::move(Ctx));
}

} // end namespace llvm::jitlink
