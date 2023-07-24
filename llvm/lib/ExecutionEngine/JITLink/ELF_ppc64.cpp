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
#include "llvm/ExecutionEngine/JITLink/DWARFRecordSectionSplitter.h"
#include "llvm/ExecutionEngine/JITLink/TableManager.h"
#include "llvm/ExecutionEngine/JITLink/ppc64.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Endian.h"

#include "EHFrameSupportImpl.h"
#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"

#define DEBUG_TYPE "jitlink"

namespace {

using namespace llvm;
using namespace llvm::jitlink;

constexpr StringRef ELFTOCSymbolName = ".TOC.";
constexpr StringRef TOCSymbolAliasIdent = "__TOC__";
constexpr uint64_t ELFTOCBaseOffset = 0x8000;

template <support::endianness Endianness>
Symbol &createELFGOTHeader(LinkGraph &G,
                           ppc64::TOCTableManager<Endianness> &TOC) {
  Symbol *TOCSymbol = nullptr;

  for (Symbol *Sym : G.defined_symbols())
    if (LLVM_UNLIKELY(Sym->getName() == ELFTOCSymbolName)) {
      TOCSymbol = Sym;
      break;
    }

  if (LLVM_LIKELY(TOCSymbol == nullptr)) {
    for (Symbol *Sym : G.external_symbols())
      if (Sym->getName() == ELFTOCSymbolName) {
        TOCSymbol = Sym;
        break;
      }
  }

  if (!TOCSymbol)
    TOCSymbol = &G.addExternalSymbol(ELFTOCSymbolName, 0, false);

  return TOC.getEntryForTarget(G, *TOCSymbol);
}

// Register preexisting GOT entries with TOC table manager.
template <support::endianness Endianness>
inline void
registerExistingGOTEntries(LinkGraph &G,
                           ppc64::TOCTableManager<Endianness> &TOC) {
  auto isGOTEntry = [](const Edge &E) {
    return E.getKind() == ppc64::Pointer64 && E.getTarget().isExternal();
  };
  if (Section *dotTOCSection = G.findSectionByName(".toc")) {
    for (Block *B : dotTOCSection->blocks())
      for (Edge &E : B->edges())
        if (isGOTEntry(E))
          TOC.registerPreExistingEntry(E.getTarget(),
                                       G.addAnonymousSymbol(*B, E.getOffset(),
                                                            G.getPointerSize(),
                                                            false, false));
  }
}

template <support::endianness Endianness>
Error buildTables_ELF_ppc64(LinkGraph &G) {
  LLVM_DEBUG(dbgs() << "Visiting edges in graph:\n");
  ppc64::TOCTableManager<Endianness> TOC;
  // Before visiting edges, we create a header containing the address of TOC
  // base as ELFABIv2 suggests:
  //  > The GOT consists of an 8-byte header that contains the TOC base (the
  //  first TOC base when multiple TOCs are present), followed by an array of
  //  8-byte addresses.
  createELFGOTHeader(G, TOC);

  // There might be compiler-generated GOT entries in ELF relocatable file.
  registerExistingGOTEntries(G, TOC);

  ppc64::PLTTableManager<Endianness> PLT(TOC);
  visitExistingEdges(G, TOC, PLT);
  // TODO: Add TLS support.

  // After visiting edges in LinkGraph, we have GOT entries built in the
  // synthesized section.
  // Merge sections included in TOC into synthesized TOC section,
  // thus TOC is compact and reducing chances of relocation
  // overflow.
  if (Section *TOCSection = G.findSectionByName(TOC.getSectionName())) {
    // .got and .plt are not normally present in a relocatable object file
    // because they are linker generated.
    if (Section *gotSection = G.findSectionByName(".got"))
      G.mergeSections(*TOCSection, *gotSection);
    if (Section *tocSection = G.findSectionByName(".toc"))
      G.mergeSections(*TOCSection, *tocSection);
    if (Section *sdataSection = G.findSectionByName(".sdata"))
      G.mergeSections(*TOCSection, *sdataSection);
    if (Section *sbssSection = G.findSectionByName(".sbss"))
      G.mergeSections(*TOCSection, *sbssSection);
    // .tocbss no longer appears in ELFABIv2. Leave it here to be compatible
    // with rtdyld.
    if (Section *tocbssSection = G.findSectionByName(".tocbss"))
      G.mergeSections(*TOCSection, *tocbssSection);
    if (Section *pltSection = G.findSectionByName(".plt"))
      G.mergeSections(*TOCSection, *pltSection);
  }

  return Error::success();
}

} // namespace

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
    using Base = ELFLinkGraphBuilder<ELFT>;
    auto ELFReloc = Rel.getType(false);

    // R_PPC64_NONE is a no-op.
    if (LLVM_UNLIKELY(ELFReloc == ELF::R_PPC64_NONE))
      return Error::success();

    auto ObjSymbol = Base::Obj.getRelocationSymbol(Rel, Base::SymTabSec);
    if (!ObjSymbol)
      return ObjSymbol.takeError();

    uint32_t SymbolIndex = Rel.getSymbol(false);
    Symbol *GraphSymbol = Base::getGraphSymbol(SymbolIndex);
    if (!GraphSymbol)
      return make_error<StringError>(
          formatv("Could not find symbol at given index, did you add it to "
                  "JITSymbolTable? index: {0}, shndx: {1} Size of table: {2}",
                  SymbolIndex, (*ObjSymbol)->st_shndx,
                  Base::GraphSymbols.size()),
          inconvertibleErrorCode());

    int64_t Addend = Rel.r_addend;
    orc::ExecutorAddr FixupAddress =
        orc::ExecutorAddr(FixupSection.sh_addr) + Rel.r_offset;
    Edge::OffsetT Offset = FixupAddress - BlockToFix.getAddress();
    Edge::Kind Kind = Edge::Invalid;

    switch (ELFReloc) {
    default:
      return make_error<JITLinkError>(
          "In " + G->getName() + ": Unsupported ppc64 relocation type " +
          object::getELFRelocationTypeName(ELF::EM_PPC64, ELFReloc));
    case ELF::R_PPC64_ADDR64:
      Kind = ppc64::Pointer64;
      break;
    case ELF::R_PPC64_TOC16_HA:
      Kind = ppc64::TOCDelta16HA;
      break;
    case ELF::R_PPC64_TOC16_DS:
      Kind = ppc64::TOCDelta16DS;
      break;
    case ELF::R_PPC64_TOC16_LO:
      Kind = ppc64::TOCDelta16LO;
      break;
    case ELF::R_PPC64_TOC16_LO_DS:
      Kind = ppc64::TOCDelta16LODS;
      break;
    case ELF::R_PPC64_REL16:
      Kind = ppc64::Delta16;
      break;
    case ELF::R_PPC64_REL16_HA:
      Kind = ppc64::Delta16HA;
      break;
    case ELF::R_PPC64_REL16_LO:
      Kind = ppc64::Delta16LO;
      break;
    case ELF::R_PPC64_REL32:
      Kind = ppc64::Delta32;
      break;
    case ELF::R_PPC64_REL24_NOTOC:
    case ELF::R_PPC64_REL24: {
      bool isLocal = !GraphSymbol->isExternal();
      if (isLocal) {
        // TODO: There are cases a local function call need a call stub.
        // 1. Caller uses TOC, the callee doesn't, need a r2 save stub.
        // 2. Caller doesn't use TOC, the callee does, need a r12 setup stub.
        // FIXME: For a local call, we might need a thunk if branch target is
        // out of range.
        Kind = ppc64::CallBranchDelta;
        // Branch to local entry.
        Addend += ELF::decodePPC64LocalEntryOffset((*ObjSymbol)->st_other);
      } else {
        Kind = ELFReloc == ELF::R_PPC64_REL24 ? ppc64::RequestPLTCallStubSaveTOC
                                              : ppc64::RequestPLTCallStubNoTOC;
      }
      break;
    }
    case ELF::R_PPC64_REL64:
      Kind = ppc64::Delta64;
      break;
    }

    Edge GE(Kind, Offset, *GraphSymbol, Addend);
    BlockToFix.addEdge(std::move(GE));
    return Error::success();
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
      : JITLinkerBase(std::move(Ctx), std::move(G), std::move(PassConfig)) {
    JITLinkerBase::getPassConfig().PostAllocationPasses.push_back(
        [this](LinkGraph &G) { return defineTOCBase(G); });
  }

private:
  Symbol *TOCSymbol = nullptr;

  Error defineTOCBase(LinkGraph &G) {
    for (Symbol *Sym : G.defined_symbols()) {
      if (LLVM_UNLIKELY(Sym->getName() == ELFTOCSymbolName)) {
        TOCSymbol = Sym;
        return Error::success();
      }
    }

    assert(TOCSymbol == nullptr &&
           "TOCSymbol should not be defined at this point");

    for (Symbol *Sym : G.external_symbols()) {
      if (Sym->getName() == ELFTOCSymbolName) {
        TOCSymbol = Sym;
        break;
      }
    }

    if (Section *TOCSection = G.findSectionByName(
            ppc64::TOCTableManager<Endianness>::getSectionName())) {
      assert(!TOCSection->empty() && "TOC section should have reserved an "
                                     "entry for containing the TOC base");

      SectionRange SR(*TOCSection);
      orc::ExecutorAddr TOCBaseAddr(SR.getFirstBlock()->getAddress() +
                                    ELFTOCBaseOffset);
      assert(TOCSymbol && TOCSymbol->isExternal() &&
             ".TOC. should be a external symbol at this point");
      G.makeAbsolute(*TOCSymbol, TOCBaseAddr);
      // Create an alias of .TOC. so that rtdyld checker can recognize.
      G.addAbsoluteSymbol(TOCSymbolAliasIdent, TOCSymbol->getAddress(),
                          TOCSymbol->getSize(), TOCSymbol->getLinkage(),
                          TOCSymbol->getScope(), TOCSymbol->isLive());
      return Error::success();
    }

    // If TOC section doesn't exist, which means no TOC relocation is found, we
    // don't need a TOCSymbol.
    return Error::success();
  }

  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    return ppc64::applyFixup<Endianness>(G, B, E, TOCSymbol);
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

    // Add eh-frame passses.
    Config.PrePrunePasses.push_back(DWARFRecordSectionSplitter(".eh_frame"));
    Config.PrePrunePasses.push_back(EHFrameEdgeFixer(
        ".eh_frame", G->getPointerSize(), ppc64::Pointer32, ppc64::Pointer64,
        ppc64::Delta32, ppc64::Delta64, ppc64::NegDelta32));
    Config.PrePrunePasses.push_back(EHFrameNullTerminator(".eh_frame"));

    // Add a mark-live pass.
    if (auto MarkLive = Ctx->getMarkLivePass(G->getTargetTriple()))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);
  }

  Config.PostPrunePasses.push_back(buildTables_ELF_ppc64<Endianness>);

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
