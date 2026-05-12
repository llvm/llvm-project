//===------ ELF_hexagon.cpp - JIT linker for ELF/hexagon ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ELF/hexagon jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF_hexagon.h"
#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/hexagon.h"
#include "llvm/Object/ELFObjectFile.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;

namespace {

class ELFJITLinker_hexagon : public JITLinker<ELFJITLinker_hexagon> {
  friend class JITLinker<ELFJITLinker_hexagon>;

public:
  ELFJITLinker_hexagon(std::unique_ptr<JITLinkContext> Ctx,
                       std::unique_ptr<LinkGraph> G,
                       PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {}

private:
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    return hexagon::applyFixup(G, B, E);
  }
};

class ELFLinkGraphBuilder_hexagon
    : public ELFLinkGraphBuilder<object::ELF32LE> {
private:
  using ELFT = object::ELF32LE;

  Expected<hexagon::EdgeKind_hexagon> getRelocationKind(const uint32_t Type) {
    switch (Type) {
    case ELF::R_HEX_32:
      return hexagon::Pointer32;
    case ELF::R_HEX_32_PCREL:
      return hexagon::PCRel32;
    case ELF::R_HEX_B22_PCREL:
    case ELF::R_HEX_PLT_B22_PCREL:
    // PLT and GD_PLT variants are mapped to plain branch edges since JITLink
    // resolves all symbols directly within contiguous JIT memory. When the
    // GOT/PLT stubs builder is added (see TODO in link_ELF_hexagon), these
    // should map to a distinct edge kind that triggers stub generation.
    // GD_PLT does not handle TLS __tls_get_addr calls.
    case ELF::R_HEX_GD_PLT_B22_PCREL:
      return hexagon::B22_PCREL;
    case ELF::R_HEX_B15_PCREL:
      return hexagon::B15_PCREL;
    case ELF::R_HEX_B13_PCREL:
      return hexagon::B13_PCREL;
    case ELF::R_HEX_B9_PCREL:
      return hexagon::B9_PCREL;
    case ELF::R_HEX_B7_PCREL:
      return hexagon::B7_PCREL;
    case ELF::R_HEX_HI16:
      return hexagon::HI16;
    case ELF::R_HEX_LO16:
      return hexagon::LO16;
    case ELF::R_HEX_32_6_X:
      return hexagon::Word32_6_X;
    case ELF::R_HEX_B32_PCREL_X:
    case ELF::R_HEX_GD_PLT_B32_PCREL_X: // See PLT/GD_PLT note above.
      return hexagon::B32_PCREL_X;
    case ELF::R_HEX_B22_PCREL_X:
    case ELF::R_HEX_GD_PLT_B22_PCREL_X: // See PLT/GD_PLT note above.
      return hexagon::B22_PCREL_X;
    case ELF::R_HEX_B15_PCREL_X:
      return hexagon::B15_PCREL_X;
    case ELF::R_HEX_B13_PCREL_X:
      return hexagon::B13_PCREL_X;
    case ELF::R_HEX_B9_PCREL_X:
      return hexagon::B9_PCREL_X;
    case ELF::R_HEX_B7_PCREL_X:
      return hexagon::B7_PCREL_X;
    case ELF::R_HEX_6_X:
      return hexagon::Word6_X;
    case ELF::R_HEX_6_PCREL_X:
      return hexagon::Word6_PCREL_X;
    case ELF::R_HEX_8_X:
      return hexagon::Word8_X;
    case ELF::R_HEX_9_X:
      return hexagon::Word9_X;
    case ELF::R_HEX_10_X:
      return hexagon::Word10_X;
    case ELF::R_HEX_11_X:
      return hexagon::Word11_X;
    case ELF::R_HEX_12_X:
      return hexagon::Word12_X;
    case ELF::R_HEX_16_X:
      return hexagon::Word16_X;
    }

    return make_error<JITLinkError>(
        "In " + G->getName() + ": Unsupported Hexagon relocation type " +
        object::getELFRelocationTypeName(ELF::EM_HEXAGON, Type));
  }

  Error addRelocations() override {
    LLVM_DEBUG(dbgs() << "Adding relocations\n");
    using Base = ELFLinkGraphBuilder<ELFT>;
    using Self = ELFLinkGraphBuilder_hexagon;

    for (const auto &RelSect : Base::Sections) {
      // Hexagon uses SHT_RELA.
      if (RelSect.sh_type == ELF::SHT_REL)
        return make_error<StringError>(
            "Unexpected SHT_REL section in Hexagon ELF object",
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
    using Base = ELFLinkGraphBuilder<ELFT>;

    auto ELFReloc = Rel.getType(false);

    if (LLVM_UNLIKELY(ELFReloc == ELF::R_HEX_NONE))
      return Error::success();

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

    Expected<hexagon::EdgeKind_hexagon> Kind = getRelocationKind(ELFReloc);
    if (!Kind)
      return Kind.takeError();

    auto FixupAddress = orc::ExecutorAddr(FixupSection.sh_addr) + Rel.r_offset;
    int64_t Addend = Rel.r_addend;

    Edge::OffsetT Offset = FixupAddress - BlockToFix.getAddress();
    Edge GE(*Kind, Offset, *GraphSymbol, Addend);
    LLVM_DEBUG({
      dbgs() << "    ";
      printEdge(dbgs(), BlockToFix, GE, hexagon::getEdgeKindName(*Kind));
      dbgs() << "\n";
    });

    BlockToFix.addEdge(std::move(GE));
    return Error::success();
  }

public:
  ELFLinkGraphBuilder_hexagon(StringRef FileName,
                              const object::ELFFile<ELFT> &Obj,
                              std::shared_ptr<orc::SymbolStringPool> SSP,
                              Triple TT, SubtargetFeatures Features)
      : ELFLinkGraphBuilder<ELFT>(Obj, std::move(SSP), std::move(TT),
                                  std::move(Features), FileName,
                                  hexagon::getEdgeKindName) {}
};

} // anonymous namespace

namespace llvm::jitlink {

Expected<std::unique_ptr<LinkGraph>> createLinkGraphFromELFObject_hexagon(
    MemoryBufferRef ObjectBuffer, std::shared_ptr<orc::SymbolStringPool> SSP) {
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

  assert((*ELFObj)->getArch() == Triple::hexagon &&
         "Only Hexagon is supported");

  auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF32LE>>(**ELFObj);

  return ELFLinkGraphBuilder_hexagon(
             (*ELFObj)->getFileName(), ELFObjFile.getELFFile(), std::move(SSP),
             (*ELFObj)->makeTriple(), std::move(*Features))
      .buildGraph();
}

void link_ELF_hexagon(std::unique_ptr<LinkGraph> G,
                      std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;
  const Triple &TT = G->getTargetTriple();
  if (Ctx->shouldAddDefaultTargetPasses(TT)) {
    // TODO: Add GOT/PLT stubs builder when external symbol support is needed.
    // TODO: Add eh-frame passes when exception handling support is needed.
    if (auto MarkLive = Ctx->getMarkLivePass(TT))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);
  }
  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  ELFJITLinker_hexagon::link(std::move(Ctx), std::move(G), std::move(Config));
}

} // namespace llvm::jitlink
