//===----- ELF_aarch64.cpp - JIT linker implementation for ELF/aarch64 ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ELF/aarch64 jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF_aarch64.h"
#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/aarch64.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;

namespace llvm {
namespace jitlink {

class ELFJITLinker_aarch64 : public JITLinker<ELFJITLinker_aarch64> {
  friend class JITLinker<ELFJITLinker_aarch64>;

public:
  ELFJITLinker_aarch64(std::unique_ptr<JITLinkContext> Ctx,
                       std::unique_ptr<LinkGraph> G,
                       PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {}

private:
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    return aarch64::applyFixup(G, B, E);
  }
};

template <typename ELFT>
class ELFLinkGraphBuilder_aarch64 : public ELFLinkGraphBuilder<ELFT> {
private:
  enum ELFAArch64RelocationKind : Edge::Kind {
    ELFCall26 = Edge::FirstRelocation,
    ELFAdrPage21,
    ELFAddAbs12,
  };

  static Expected<ELFAArch64RelocationKind>
  getRelocationKind(const uint32_t Type) {
    using namespace aarch64;
    switch (Type) {
    case ELF::R_AARCH64_CALL26:
      return ELFCall26;
    case ELF::R_AARCH64_ADR_PREL_PG_HI21:
      return ELFAdrPage21;
    case ELF::R_AARCH64_ADD_ABS_LO12_NC:
      return ELFAddAbs12;
    }

    return make_error<JITLinkError>("Unsupported aarch64 relocation:" +
                                    formatv("{0:d}", Type));
  }

  Error addRelocations() override {
    LLVM_DEBUG(dbgs() << "Processing relocations:\n");

    using Base = ELFLinkGraphBuilder<ELFT>;
    using Self = ELFLinkGraphBuilder_aarch64<ELFT>;
    for (const auto &RelSect : Base::Sections)
      if (Error Err = Base::forEachRelocation(RelSect, this,
                                              &Self::addSingleRelocation))
        return Err;

    return Error::success();
  }

  Error addSingleRelocation(const typename ELFT::Rela &Rel,
                            const typename ELFT::Shdr &FixupSect,
                            Block &BlockToFix) {
    using Base = ELFLinkGraphBuilder<ELFT>;

    uint32_t SymbolIndex = Rel.getSymbol(false);
    auto ObjSymbol = Base::Obj.getRelocationSymbol(Rel, Base::SymTabSec);
    if (!ObjSymbol)
      return ObjSymbol.takeError();

    Symbol *GraphSymbol = Base::getGraphSymbol(SymbolIndex);
    if (!GraphSymbol)
      return make_error<StringError>(
          formatv("Could not find symbol at given index, did you add it to "
                  "JITSymbolTable? index: {0}, shndx: {1} Size of table: {2}",
                  SymbolIndex, (*ObjSymbol)->st_shndx,
                  Base::GraphSymbols.size()),
          inconvertibleErrorCode());

    uint32_t Type = Rel.getType(false);
    Expected<ELFAArch64RelocationKind> RelocKind = getRelocationKind(Type);
    if (!RelocKind)
      return RelocKind.takeError();

    int64_t Addend = Rel.r_addend;
    orc::ExecutorAddr FixupAddress =
        orc::ExecutorAddr(FixupSect.sh_addr) + Rel.r_offset;
    Edge::OffsetT Offset = FixupAddress - BlockToFix.getAddress();

    Edge::Kind Kind = Edge::Invalid;

    switch (*RelocKind) {
    case ELFCall26: {
      Kind = aarch64::Branch26;
      break;
    }
    case ELFAdrPage21: {
      Kind = aarch64::Page21;
      break;
    }
    case ELFAddAbs12: {
      Kind = aarch64::PageOffset12;
      break;
    }
    };

    Edge GE(Kind, Offset, *GraphSymbol, Addend);
    LLVM_DEBUG({
      dbgs() << "    ";
      printEdge(dbgs(), BlockToFix, GE, aarch64::getEdgeKindName(Kind));
      dbgs() << "\n";
    });

    BlockToFix.addEdge(std::move(GE));
    return Error::success();
  }

  /// Return the string name of the given ELF aarch64 edge kind.
  const char *getELFAArch64RelocationKindName(Edge::Kind R) {
    switch (R) {
    case ELFCall26:
      return "ELFCall26";
    case ELFAdrPage21:
      return "ELFAdrPage21";
    case ELFAddAbs12:
      return "ELFAddAbs12";
    default:
      return getGenericEdgeKindName(static_cast<Edge::Kind>(R));
    }
  }

public:
  ELFLinkGraphBuilder_aarch64(StringRef FileName,
                              const object::ELFFile<ELFT> &Obj, const Triple T)
      : ELFLinkGraphBuilder<ELFT>(Obj, std::move(T), FileName,
                                  aarch64::getEdgeKindName) {}
};

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_aarch64(MemoryBufferRef ObjectBuffer) {
  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << ObjectBuffer.getBufferIdentifier() << "...\n";
  });

  auto ELFObj = object::ObjectFile::createELFObjectFile(ObjectBuffer);
  if (!ELFObj)
    return ELFObj.takeError();

  assert((*ELFObj)->getArch() == Triple::aarch64 &&
         "Only AArch64 (little endian) is supported for now");

  auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF64LE>>(**ELFObj);
  return ELFLinkGraphBuilder_aarch64<object::ELF64LE>((*ELFObj)->getFileName(),
                                                      ELFObjFile.getELFFile(),
                                                      (*ELFObj)->makeTriple())
      .buildGraph();
}

void link_ELF_aarch64(std::unique_ptr<LinkGraph> G,
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

  ELFJITLinker_aarch64::link(std::move(Ctx), std::move(G), std::move(Config));
}

} // namespace jitlink
} // namespace llvm
