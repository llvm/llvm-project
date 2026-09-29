//===--------- ELF_x86.cpp - JIT linker implementation for ELF/x86 --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ELF/x86 jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF_x86.h"
#include "DefineExternalSectionStartAndEndSymbols.h"
#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/x86.h"
#include "llvm/Object/ELFObjectFile.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;

namespace {
constexpr StringRef ELFGOTSymbolName = "_GLOBAL_OFFSET_TABLE_";

Error buildTables_ELF_x86(LinkGraph &G) {
  LLVM_DEBUG(dbgs() << "Visiting edges in graph:\n");

  x86::GOTTableManager GOT;
  x86::PLTTableManager PLT(GOT);
  visitExistingEdges(G, GOT, PLT);
  return Error::success();
}
} // namespace

namespace llvm::jitlink {

class ELFJITLinker_x86 : public JITLinker<ELFJITLinker_x86> {
  friend class JITLinker<ELFJITLinker_x86>;

public:
  ELFJITLinker_x86(std::unique_ptr<JITLinkContext> Ctx,
                   std::unique_ptr<LinkGraph> G, PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {
    getPassConfig().PostAllocationPasses.push_back(
        [this](LinkGraph &G) { return getOrCreateGOTSymbol(G); });
  }

private:
  Symbol *GOTSymbol = nullptr;

  Error getOrCreateGOTSymbol(LinkGraph &G) {
    auto DefineExternalGOTSymbolIfPresent =
        createDefineExternalSectionStartAndEndSymbolsPass(
            [&](LinkGraph &LG, Symbol &Sym) -> SectionRangeSymbolDesc {
              if (Sym.getName() != nullptr &&
                  *Sym.getName() == ELFGOTSymbolName)
                if (auto *GOTSection = G.findSectionByName(
                        x86::GOTTableManager::getSectionName())) {
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
            G.findSectionByName(x86::GOTTableManager::getSectionName())) {

      // Check for an existing defined symbol.
      for (auto *Sym : GOTSection->symbols())
        if (Sym->getName() != nullptr && *Sym->getName() == ELFGOTSymbolName) {
          GOTSymbol = Sym;
          return Error::success();
        }

      // If there's no defined symbol then create one.
      SectionRange SR(*GOTSection);

      if (SR.empty()) {
        GOTSymbol =
            &G.addAbsoluteSymbol(ELFGOTSymbolName, orc::ExecutorAddr(), 0,
                                 Linkage::Strong, Scope::Local, true);
      } else {
        GOTSymbol =
            &G.addDefinedSymbol(*SR.getFirstBlock(), 0, ELFGOTSymbolName, 0,
                                Linkage::Strong, Scope::Local, false, true);
      }
    }

    return Error::success();
  }

  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    return x86::applyFixup(G, B, E, GOTSymbol);
  }
};

class ELFLinkGraphBuilder_x86 : public ELFLinkGraphBuilder<object::ELF32LE> {
private:
  using ELFT = object::ELF32LE;

  Expected<x86::EdgeKind_x86> getRelocationKind(const uint32_t Type) {
    switch (Type) {
    case ELF::R_386_32:
      return x86::Pointer32;
    case ELF::R_386_PC32:
      return x86::PCRel32;
    case ELF::R_386_16:
      return x86::Pointer16;
    case ELF::R_386_PC16:
      return x86::PCRel16;
    case ELF::R_386_GOT32:
      return x86::RequestGOTAndTransformToDelta32FromGOT;
    case ELF::R_386_GOT32X:
      // TODO: Add a relaxable edge kind and update relaxation optimization.
      return x86::RequestGOTAndTransformToDelta32FromGOT;
    case ELF::R_386_GOTPC:
      return x86::Delta32;
    case ELF::R_386_GOTOFF:
      return x86::Delta32FromGOT;
    case ELF::R_386_PLT32:
      return x86::BranchPCRel32;
    }

    return make_error<JITLinkError>(
        "In " + G->getName() + ": Unsupported x86 relocation type " +
        object::getELFRelocationTypeName(ELF::EM_386, Type));
  }

  Error addRelocations() override {
    LLVM_DEBUG(dbgs() << "Adding relocations\n");
    using Base = ELFLinkGraphBuilder<ELFT>;
    using Self = ELFLinkGraphBuilder_x86;

    for (const auto &RelSect : Base::Sections) {
      // Validate the section to read relocation entries from.
      if (RelSect.sh_type == ELF::SHT_RELA)
        return make_error<StringError>(
            "No SHT_RELA in valid x86 ELF object files",
            inconvertibleErrorCode());

      if (Error Err = Base::forEachRelRelocation(RelSect, this,
                                                 &Self::addSingleRelocation))
        return Err;
    }

    return Error::success();
  }

  Error addSingleRelocation(const typename ELFT::Rel &Rel,
                            const typename ELFT::Shdr &FixupSection,
                            Block &BlockToFix) {
    using Base = ELFLinkGraphBuilder<ELFT>;

    auto ELFReloc = Rel.getType(false);

    // R_386_NONE is a no-op.
    if (LLVM_UNLIKELY(ELFReloc == ELF::R_386_NONE))
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

    Expected<x86::EdgeKind_x86> Kind = getRelocationKind(ELFReloc);
    if (!Kind)
      return Kind.takeError();

    auto FixupAddress = orc::ExecutorAddr(FixupSection.sh_addr) + Rel.r_offset;
    int64_t Addend = 0;

    switch (*Kind) {
    case x86::Pointer32:
    case x86::PCRel32:
    case x86::RequestGOTAndTransformToDelta32FromGOT:
    case x86::Delta32:
    case x86::Delta32FromGOT:
    case x86::BranchPCRel32:
    case x86::BranchPCRel32ToPtrJumpStub:
    case x86::BranchPCRel32ToPtrJumpStubBypassable: {
      const char *FixupContent = BlockToFix.getContent().data() +
                                 (FixupAddress - BlockToFix.getAddress());
      Addend = *(const support::little32_t *)FixupContent;
      break;
    }
    case x86::Pointer16:
    case x86::PCRel16: {
      const char *FixupContent = BlockToFix.getContent().data() +
                                 (FixupAddress - BlockToFix.getAddress());
      Addend = *(const support::little16_t *)FixupContent;
      break;
    }
    }

    Edge::OffsetT Offset = FixupAddress - BlockToFix.getAddress();
    Edge GE(*Kind, Offset, *GraphSymbol, Addend);
    LLVM_DEBUG({
      dbgs() << "    ";
      printEdge(dbgs(), BlockToFix, GE, x86::getEdgeKindName(*Kind));
      dbgs() << "\n";
    });

    BlockToFix.addEdge(std::move(GE));
    return Error::success();
  }

public:
  ELFLinkGraphBuilder_x86(StringRef FileName, const object::ELFFile<ELFT> &Obj,
                          std::shared_ptr<orc::SymbolStringPool> SSP, Triple TT,
                          SubtargetFeatures Features)
      : ELFLinkGraphBuilder<ELFT>(Obj, std::move(SSP), std::move(TT),
                                  std::move(Features), FileName,
                                  x86::getEdgeKindName) {}
};

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_x86(MemoryBufferRef ObjectBuffer,
                                 std::shared_ptr<orc::SymbolStringPool> SSP) {
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

  assert((*ELFObj)->getArch() == Triple::x86 &&
         "Only x86 (little endian) is supported for now");

  auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF32LE>>(**ELFObj);

  return ELFLinkGraphBuilder_x86((*ELFObj)->getFileName(),
                                 ELFObjFile.getELFFile(), std::move(SSP),
                                 (*ELFObj)->makeTriple(), std::move(*Features))
      .buildGraph();
}

void link_ELF_x86(std::unique_ptr<LinkGraph> G,
                  std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;
  const Triple &TT = G->getTargetTriple();
  if (Ctx->shouldAddDefaultTargetPasses(TT)) {
    if (auto MarkLive = Ctx->getMarkLivePass(TT))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);

    // Add an in-place GOT and PLT build pass.
    Config.PostPrunePasses.push_back(buildTables_ELF_x86);

    // Add GOT/Stubs optimizer pass.
    Config.PreFixupPasses.push_back(x86::optimizeGOTAndStubAccesses);
  }
  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  ELFJITLinker_x86::link(std::move(Ctx), std::move(G), std::move(Config));
}

} // namespace llvm::jitlink
