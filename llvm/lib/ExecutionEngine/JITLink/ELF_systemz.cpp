//===----- ELF_systemz.cpp - JIT linker implementation for ELF/systemz ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ELF/systemz jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/DWARFRecordSectionSplitter.h"
#include "llvm/ExecutionEngine/JITLink/systemz.h"
#include "llvm/Object/ELFObjectFile.h"

#include "DefineExternalSectionStartAndEndSymbols.h"
#include "EHFrameSupportImpl.h"
#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;

namespace {

constexpr StringRef ELFGOTSymbolName = "_GLOBAL_OFFSET_TABLE_";
constexpr StringRef ELFTLSInfoSectionName = "$__TLSINFO";

// TLS Info Builder.
class TLSInfoTableManager_ELF_systemz
    : public TableManager<TLSInfoTableManager_ELF_systemz> {
public:
  static StringRef getSectionName() { return ELFTLSInfoSectionName; }

  static const uint8_t TLSInfoEntryContent[16];

  bool visitEdge(LinkGraph &G, Block *B, Edge &E) {
    if (E.getKind() ==
        systemz::RequestTLSDescInGOTAndTransformToDelta64FromGOT) {
      LLVM_DEBUG({
        dbgs() << "  Fixing " << G.getEdgeKindName(E.getKind()) << " edge at "
               << formatv("{0:x}", B->getFixupAddress(E)) << " ("
               << formatv("{0:x}", B->getAddress()) << " + "
               << formatv("{0:x}", E.getOffset()) << ")\n";
      });
      E.setKind(systemz::Delta64FromGOT);
      E.setTarget(getEntryForTarget(G, E.getTarget()));
      return true;
    }
    return false;
  }

  Symbol &createEntry(LinkGraph &G, Symbol &Target) {
    // the TLS Info entry's key value will be written by the fixTLVSectionByName
    // pass, so create mutable content.
    auto &TLSInfoEntry = G.createMutableContentBlock(
        getTLSInfoSection(G), G.allocateContent(getTLSInfoEntryContent()),
        orc::ExecutorAddr(), 8, 0);
    TLSInfoEntry.addEdge(systemz::Pointer64, 8, Target, 0);
    return G.addAnonymousSymbol(TLSInfoEntry, 0, 16, false, false);
  }

private:
  Section &getTLSInfoSection(LinkGraph &G) {
    if (!TLSInfoTable)
      TLSInfoTable = &G.createSection(getSectionName(), orc::MemProt::Read);
    return *TLSInfoTable;
  }

  ArrayRef<char> getTLSInfoEntryContent() const {
    return {reinterpret_cast<const char *>(TLSInfoEntryContent),
            sizeof(TLSInfoEntryContent)};
  }

  Section *TLSInfoTable = nullptr;
};

const uint8_t TLSInfoTableManager_ELF_systemz::TLSInfoEntryContent[16] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

Error buildTables_ELF_systemz(LinkGraph &G) {
  LLVM_DEBUG(dbgs() << "Visiting edges in graph:\n");
  systemz::GOTTableManager GOT;
  systemz::PLTTableManager PLT(GOT);
  TLSInfoTableManager_ELF_systemz TLSInfo;
  visitExistingEdges(G, GOT, PLT, TLSInfo);
  return Error::success();
}

} // namespace

namespace llvm {
namespace jitlink {
class ELFJITLinker_systemz : public JITLinker<ELFJITLinker_systemz> {
  friend class JITLinker<ELFJITLinker_systemz>;

public:
  ELFJITLinker_systemz(std::unique_ptr<JITLinkContext> Ctx,
                       std::unique_ptr<LinkGraph> G,
                       PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {
    if (shouldAddDefaultTargetPasses(getGraph().getTargetTriple()))
      getPassConfig().PostAllocationPasses.push_back(
          [this](LinkGraph &G) { return getOrCreateGOTSymbol(G); });
  }

private:
  Symbol *GOTSymbol = nullptr;

  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    return systemz::applyFixup(G, B, E, GOTSymbol);
  }

  Error getOrCreateGOTSymbol(LinkGraph &G) {
    auto DefineExternalGOTSymbolIfPresent =
        createDefineExternalSectionStartAndEndSymbolsPass(
            [&](LinkGraph &LG, Symbol &Sym) -> SectionRangeSymbolDesc {
              if (Sym.getName() != nullptr &&
                  *Sym.getName() == ELFGOTSymbolName)
                if (auto *GOTSection = G.findSectionByName(
                        systemz::GOTTableManager::getSectionName())) {
                  GOTSymbol = &Sym;
                  return {*GOTSection, true};
                }
              return {};
            });

    // Try to attach _GLOBAL_OFFSET_TABLE_ to the GOT if it's defined as an
    // external.
    if (auto Err = DefineExternalGOTSymbolIfPresent(G))
      return Err;

    // If we succeeded then we're done.
    if (GOTSymbol)
      return Error::success();

    // Otherwise look for a GOT section: If it already has a start symbol we'll
    // record it, otherwise we'll create our own.
    // If there's a GOT section but we didn't find an external GOT symbol...
    if (auto *GOTSection =
            G.findSectionByName(systemz::GOTTableManager::getSectionName())) {

      // Check for an existing defined symbol.
      for (auto *Sym : GOTSection->symbols())
        if (Sym->getName() != nullptr && *Sym->getName() == ELFGOTSymbolName) {
          GOTSymbol = Sym;
          return Error::success();
        }

      // If there's no defined symbol then create one.
      SectionRange SR(*GOTSection);
      if (SR.empty())
        GOTSymbol =
            &G.addAbsoluteSymbol(ELFGOTSymbolName, orc::ExecutorAddr(), 0,
                                 Linkage::Strong, Scope::Local, true);
      else
        GOTSymbol =
            &G.addDefinedSymbol(*SR.getFirstBlock(), 0, ELFGOTSymbolName, 0,
                                Linkage::Strong, Scope::Local, false, true);
    }

    // If we still haven't found a GOT symbol then double check the externals.
    // We may have a GOT-relative reference but no GOT section, in which case
    // we just need to point the GOT symbol at some address in this graph.
    if (!GOTSymbol) {
      for (auto *Sym : G.external_symbols()) {
        if (Sym->getName() != nullptr && *Sym->getName() == ELFGOTSymbolName) {
          auto Blocks = G.blocks();
          if (!Blocks.empty()) {
            G.makeAbsolute(*Sym, (*Blocks.begin())->getAddress());
            GOTSymbol = Sym;
            break;
          }
        }
      }
    }

    return Error::success();
  }
};

class ELFLinkGraphBuilder_systemz
    : public ELFLinkGraphBuilder<object::ELF64BE> {
private:
  using ELFT = object::ELF64BE;
  using Base = ELFLinkGraphBuilder<ELFT>;
  using Base::G; // Use LinkGraph pointer from base class.

  Error addRelocations() override {
    LLVM_DEBUG(dbgs() << "Processing relocations:\n");

    using Base = ELFLinkGraphBuilder<ELFT>;
    using Self = ELFLinkGraphBuilder_systemz;
    for (const auto &RelSect : Base::Sections) {
      if (RelSect.sh_type == ELF::SHT_REL)
        // Validate the section to read relocation entries from.
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
                            const typename ELFT::Shdr &FixupSect,
                            Block &BlockToFix) {
    using support::big32_t;
    using Base = ELFLinkGraphBuilder<ELFT>;
    auto ELFReloc = Rel.getType(false);

    // No reloc.
    if (LLVM_UNLIKELY(ELFReloc == ELF::R_390_NONE))
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

    // Validate the relocation kind.
    int64_t Addend = Rel.r_addend;
    Edge::Kind Kind = Edge::Invalid;

    switch (ELFReloc) {
    case ELF::R_390_PC64: {
      Kind = systemz::Delta64;
      break;
    }
    case ELF::R_390_PC32: {
      Kind = systemz::Delta32;
      break;
    }
    case ELF::R_390_PC16: {
      Kind = systemz::Delta16;
      break;
    }
    case ELF::R_390_PC32DBL: {
      Kind = systemz::Delta32dbl;
      break;
    }
    case ELF::R_390_PC24DBL: {
      Kind = systemz::Delta24dbl;
      break;
    }
    case ELF::R_390_PC16DBL: {
      Kind = systemz::Delta16dbl;
      break;
    }
    case ELF::R_390_PC12DBL: {
      Kind = systemz::Delta12dbl;
      break;
    }
    case ELF::R_390_64: {
      Kind = systemz::Pointer64;
      break;
    }
    case ELF::R_390_32: {
      Kind = systemz::Pointer32;
      break;
    }
    case ELF::R_390_20: {
      Kind = systemz::Pointer20;
      break;
    }
    case ELF::R_390_16: {
      Kind = systemz::Pointer16;
      break;
    }
    case ELF::R_390_12: {
      Kind = systemz::Pointer12;
      break;
    }
    case ELF::R_390_8: {
      Kind = systemz::Pointer8;
      break;
    }
    // Relocations targeting the PLT associated with the symbol.
    case ELF::R_390_PLT64: {
      Kind = systemz::DeltaPLT64;
      break;
    }
    case ELF::R_390_PLT32: {
      Kind = systemz::DeltaPLT32;
      break;
    }
    case ELF::R_390_PLT32DBL: {
      Kind = systemz::DeltaPLT32dbl;
      break;
    }
    case ELF::R_390_PLT24DBL: {
      Kind = systemz::DeltaPLT24dbl;
      break;
    }
    case ELF::R_390_PLT16DBL: {
      Kind = systemz::DeltaPLT16dbl;
      break;
    }
    case ELF::R_390_PLT12DBL: {
      Kind = systemz::DeltaPLT12dbl;
      break;
    }
    case ELF::R_390_PLTOFF64: {
      Kind = systemz::Delta64PLTFromGOT;
      break;
    }
    case ELF::R_390_PLTOFF32: {
      Kind = systemz::Delta32PLTFromGOT;
      break;
    }
    case ELF::R_390_PLTOFF16: {
      Kind = systemz::Delta16PLTFromGOT;
      break;
    }
    // Relocations targeting the actual symbol (just relative to the GOT).
    case ELF::R_390_GOTOFF64: {
      Kind = systemz::Delta64FromGOT;
      break;
    }
    case ELF::R_390_GOTOFF: {
      Kind = systemz::Delta32FromGOT;
      break;
    }
    case ELF::R_390_GOTOFF16: {
      Kind = systemz::Delta16FromGOT;
      break;
    }
    // Relocations targeting the GOT entry associated with the symbol.
    case ELF::R_390_GOT64:
    case ELF::R_390_GOTPLT64: {
      Kind = systemz::RequestGOTAndTransformToDelta64FromGOT;
      break;
    }
    case ELF::R_390_GOT32:
    case ELF::R_390_GOTPLT32: {
      Kind = systemz::RequestGOTAndTransformToDelta32FromGOT;
      break;
    }
    case ELF::R_390_GOT20:
    case ELF::R_390_GOTPLT20: {
      Kind = systemz::RequestGOTAndTransformToDelta20FromGOT;
      break;
    }
    case ELF::R_390_GOT16:
    case ELF::R_390_GOTPLT16: {
      Kind = systemz::RequestGOTAndTransformToDelta16FromGOT;
      break;
    }
    case ELF::R_390_GOT12:
    case ELF::R_390_GOTPLT12: {
      Kind = systemz::RequestGOTAndTransformToDelta12FromGOT;
      break;
    }
    case ELF::R_390_GOTENT:
    case ELF::R_390_GOTPLTENT: {
      Kind = systemz::RequestGOTAndTransformToDelta32dbl;
      break;
    }
    // R_390_GOTPC and R_390_GOTPCDBL don't create GOT entry, they don't even
    // have symbol.
    case ELF::R_390_GOTPC: {
      Kind = systemz::Delta32GOTBase;
      break;
    }
    case ELF::R_390_GOTPCDBL: {
      Kind = systemz::Delta32dblGOTBase;
      break;
    }
    // Tag for function call in general dynamic TLS code.
    case ELF::R_390_TLS_GDCALL: {
      break;
    }
    // Direct 64 bit for general dynamic thread local data.
    case ELF::R_390_TLS_GD64: {
      Kind = systemz::RequestTLSDescInGOTAndTransformToDelta64FromGOT;
      break;
    }
    default:
      return make_error<JITLinkError>(
          "In " + G->getName() + ": Unsupported systemz relocation type " +
          object::getELFRelocationTypeName(ELF::EM_S390, ELFReloc));
    }
    auto FixupAddress = orc::ExecutorAddr(FixupSect.sh_addr) + Rel.r_offset;
    Edge::OffsetT Offset = FixupAddress - BlockToFix.getAddress();
    Edge GE(Kind, Offset, *GraphSymbol, Addend);
    LLVM_DEBUG({
      dbgs() << "    ";
      printEdge(dbgs(), BlockToFix, GE, systemz::getEdgeKindName(Kind));
      dbgs() << "\n";
    });

    BlockToFix.addEdge(std::move(GE));

    return Error::success();
  }

public:
  ELFLinkGraphBuilder_systemz(StringRef FileName,
                              const object::ELFFile<ELFT> &Obj,
                              std::shared_ptr<orc::SymbolStringPool> SSP,
                              Triple TT, SubtargetFeatures Features)
      : ELFLinkGraphBuilder<ELFT>(Obj, std::move(SSP), std::move(TT),
                                  std::move(Features), FileName,
                                  systemz::getEdgeKindName) {}
};

Expected<std::unique_ptr<LinkGraph>> createLinkGraphFromELFObject_systemz(
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

  assert((*ELFObj)->getArch() == Triple::systemz &&
         "Only SystemZ is supported");

  auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF64BE>>(**ELFObj);
  return ELFLinkGraphBuilder_systemz(
             (*ELFObj)->getFileName(), ELFObjFile.getELFFile(), std::move(SSP),
             (*ELFObj)->makeTriple(), std::move(*Features))
      .buildGraph();
}

void link_ELF_systemz(std::unique_ptr<LinkGraph> G,
                      std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;
  const Triple &TT = G->getTargetTriple();
  if (Ctx->shouldAddDefaultTargetPasses(TT)) {
    // Add eh-frame passes.
    Config.PrePrunePasses.push_back(DWARFRecordSectionSplitter(".eh_frame"));
    Config.PrePrunePasses.push_back(
        EHFrameEdgeFixer(".eh_frame", G->getPointerSize(), systemz::Pointer32,
                         systemz::Pointer64, systemz::Delta32, systemz::Delta64,
                         systemz::NegDelta32));
    Config.PrePrunePasses.push_back(EHFrameNullTerminator(".eh_frame"));

    // Add a mark-live pass.
    if (auto MarkLive = Ctx->getMarkLivePass(TT))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);

    // Add an in-place GOT/Stubs build pass.
    Config.PostPrunePasses.push_back(buildTables_ELF_systemz);

    // Resolve any external section start / end symbols.
    Config.PostAllocationPasses.push_back(
        createDefineExternalSectionStartAndEndSymbolsPass(
            identifyELFSectionStartAndEndSymbols));

    // Add GOT/Stubs optimizer pass.
    Config.PreFixupPasses.push_back(systemz::optimizeGOTAndStubAccesses);
  }

  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  ELFJITLinker_systemz::link(std::move(Ctx), std::move(G), std::move(Config));
}

} // namespace jitlink
} // namespace llvm
