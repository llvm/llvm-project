//===-- ELF_ppc32.cpp - JIT linker for ELF/PPC32 -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF_ppc32.h"
#include "llvm/ExecutionEngine/JITLink/DWARFRecordSectionSplitter.h"
#include "llvm/ExecutionEngine/JITLink/TableManager.h"
#include "llvm/ExecutionEngine/JITLink/ppc32.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Endian.h"

#include "EHFrameSupportImpl.h"
#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"

#define DEBUG_TYPE "jitlink"

namespace llvm::jitlink {
namespace {

constexpr StringLiteral ELFGOTSymbolName = "_GLOBAL_OFFSET_TABLE_";
constexpr StringLiteral ELFTLSBaseSymbolName = "__jitlink_ppc32_tls_base";

static Symbol &getOrCreateTLSBase(LinkGraph &G) {
  auto Name = G.intern(ELFTLSBaseSymbolName);
  if (auto *S = G.findDefinedSymbolByName(Name))
    return *S;
  auto *TLS = G.findSectionByName(".tdata");
  if (!TLS || TLS->empty())
    TLS = G.findSectionByName(".tbss");
  if (!TLS || TLS->empty()) {
    TLS = &G.createSection(".tdata", orc::MemProt::Read | orc::MemProt::Write);
    G.createMutableContentBlock(*TLS, 1, orc::ExecutorAddr(), 1, 0);
  }
  auto &B = **TLS->blocks().begin();
  return G.addDefinedSymbol(B, 0, Name, 0, Linkage::Strong, Scope::Local, false,
                            true);
}

class GOTTableManager_ELF_ppc32 {
public:
  static StringRef getSectionName() { return "$__GOT"; }
  bool visitEdge(LinkGraph &G, Block *, Edge &E) {
    switch (E.getKind()) {
    case ppc32::RequestGOTAndTransformToGOTDelta16:
      E.setKind(ppc32::GOTDelta16);
      break;
    case ppc32::RequestGOTAndTransformToGOTDelta16LO:
      E.setKind(ppc32::GOTDelta16LO);
      break;
    case ppc32::RequestGOTAndTransformToGOTDelta16HI:
      E.setKind(ppc32::GOTDelta16HI);
      break;
    case ppc32::RequestGOTAndTransformToGOTDelta16HA:
      E.setKind(ppc32::GOTDelta16HA);
      break;
    default:
      return false;
    }
    E.setTarget(getEntryForTarget(G, E.getTarget(), E.getAddend()));
    E.setAddend(0);
    return true;
  }

  /// Create the GOT and define or resolve its base symbol.
  Section &getOrCreateSection(LinkGraph &G, Symbol *ExternalGOTSym = nullptr) {
    if (GOT)
      return *GOT;
    // GOT setup branches to an executable blrl at _GLOBAL_OFFSET_TABLE_ - 4.
    GOT = &G.createSection(getSectionName(),
                           orc::MemProt::Read | orc::MemProt::Exec);
    auto Content = G.allocateBuffer(4);
    support::endian::write32(Content.data(), 0x4e800021, // blrl
                             G.getEndianness());
    auto &B =
        G.createMutableContentBlock(*GOT, Content, orc::ExecutorAddr(), 4, 0);
    if (ExternalGOTSym)
      G.makeDefined(*ExternalGOTSym, B, 4, 0, Linkage::Strong, Scope::Local,
                    true);
    else
      G.addDefinedSymbol(B, 4, ELFGOTSymbolName, 0, Linkage::Strong,
                         Scope::Local, false, true);
    return *GOT;
  }

  Symbol &getEntryForTarget(LinkGraph &G, Symbol &Target,
                            Edge::AddendT Addend = 0) {
    // GOT references can name anonymous section symbols as well as globals.
    auto &Entry = Entries[{&Target, Addend}];
    if (!Entry)
      Entry = &createEntry(G, Target, Addend);
    return *Entry;
  }

  Symbol &createEntry(LinkGraph &G, Symbol &Target, Edge::AddendT Addend) {
    auto &Entry = ppc32::createAnonymousPointer(G, getOrCreateSection(G),
                                                &Target, Addend);
    // Order GOT entries after blrl when BasicLayout sorts blocks by address.
    Entry.getBlock().setAddress(orc::ExecutorAddr(NextEntryAddress));
    NextEntryAddress += 4;
    return Entry;
  }

private:
  Section *GOT = nullptr;
  uint64_t NextEntryAddress = 4;
  DenseMap<std::pair<Symbol *, Edge::AddendT>, Symbol *> Entries;
};

class PLTTableManager_ELF_ppc32 {
public:
  explicit PLTTableManager_ELF_ppc32(GOTTableManager_ELF_ppc32 &GOT)
      : GOT(GOT) {}
  static StringRef getSectionName() { return "$__STUBS"; }

  bool visitEdge(LinkGraph &G, Block *, Edge &E) {
    if (E.getKind() != ppc32::Branch24 || !E.getTarget().isExternal())
      return false;
    // The addend belongs to the destination, not the branch to the stub.
    auto &Pointer = GOT.getEntryForTarget(G, E.getTarget(), E.getAddend());
    auto &Stub = Entries[&Pointer];
    if (!Stub)
      Stub = &createEntry(G, Pointer);
    E.setTarget(*Stub);
    E.setAddend(0);
    return true;
  }

  Symbol &createEntry(LinkGraph &G, Symbol &Pointer) {
    if (!Stubs)
      Stubs = &G.createSection(getSectionName(),
                               orc::MemProt::Read | orc::MemProt::Exec);
    return ppc32::createAnonymousPointerJumpStub(G, *Stubs, Pointer);
  }

private:
  GOTTableManager_ELF_ppc32 &GOT;
  Section *Stubs = nullptr;
  DenseMap<Symbol *, Symbol *> Entries;
};

class TLSInfoTableManager_ELF_ppc32 {
public:
  explicit TLSInfoTableManager_ELF_ppc32(GOTTableManager_ELF_ppc32 &GOT)
      : GOT(GOT) {}

  bool visitEdge(LinkGraph &G, Block *, Edge &E) {
    switch (E.getKind()) {
    case ppc32::RequestTLSDescInGOTAndTransformToGOTDelta16:
      return rewriteTLS(G, E, ppc32::GOTDelta16, false);
    case ppc32::RequestTLSDescInGOTAndTransformToGOTDelta16LO:
      return rewriteTLS(G, E, ppc32::GOTDelta16LO, false);
    case ppc32::RequestTLSDescInGOTAndTransformToGOTDelta16HI:
      return rewriteTLS(G, E, ppc32::GOTDelta16HI, false);
    case ppc32::RequestTLSDescInGOTAndTransformToGOTDelta16HA:
      return rewriteTLS(G, E, ppc32::GOTDelta16HA, false);
    case ppc32::RequestTLSLDAndTransformToGOTDelta16:
      return rewriteTLS(G, E, ppc32::GOTDelta16, true);
    case ppc32::RequestTLSLDAndTransformToGOTDelta16LO:
      return rewriteTLS(G, E, ppc32::GOTDelta16LO, true);
    case ppc32::RequestTLSLDAndTransformToGOTDelta16HI:
      return rewriteTLS(G, E, ppc32::GOTDelta16HI, true);
    case ppc32::RequestTLSLDAndTransformToGOTDelta16HA:
      return rewriteTLS(G, E, ppc32::GOTDelta16HA, true);
    case ppc32::DTPRel32:
    case ppc32::DTPRel16:
    case ppc32::DTPRel16LO:
    case ppc32::DTPRel16HI:
    case ppc32::DTPRel16HA:
      getOrCreateTLSBase(G);
      return false;
    default:
      return false;
    }
  }

private:
  bool rewriteTLS(LinkGraph &G, Edge &E, Edge::Kind K, bool IsLocalDynamic) {
    auto &Target = IsLocalDynamic ? getOrCreateTLSBase(G) : E.getTarget();
    auto Addend = IsLocalDynamic ? 0 : E.getAddend();
    auto &Entry = Entries[{&Target, Addend}];
    if (!Entry)
      Entry = &createEntry(G, Target, Addend);
    GOT.getOrCreateSection(G);
    E.setKind(K);
    E.setTarget(*Entry);
    E.setAddend(0);
    return true;
  }

  Symbol &createEntry(LinkGraph &G, Symbol &Target, Edge::AddendT Addend) {
    if (!TLSInfo)
      TLSInfo = &G.createSection("$__TLSINFO", orc::MemProt::Read);
    // ELFNixPlatform fills in the pthread key.
    auto &B =
        G.createMutableContentBlock(*TLSInfo, 8, orc::ExecutorAddr(), 4, 0);
    B.addEdge(ppc32::Pointer32, 4, Target, Addend);
    return G.addAnonymousSymbol(B, 0, 8, false, false);
  }

  GOTTableManager_ELF_ppc32 &GOT;
  Section *TLSInfo = nullptr;
  DenseMap<std::pair<Symbol *, Edge::AddendT>, Symbol *> Entries;
};

static Error buildTables_ELF_ppc32(LinkGraph &G) {
  GOTTableManager_ELF_ppc32 GOT;
  PLTTableManager_ELF_ppc32 PLT(GOT);
  TLSInfoTableManager_ELF_ppc32 TLSInfo(GOT);
  // Resolve the GOT base before PLT stub creation.
  Symbol *ExternalGOTSym = nullptr;
  for (auto *Sym : G.external_symbols())
    if (*Sym->getName() == ELFGOTSymbolName) {
      ExternalGOTSym = Sym;
      break;
    }
  if (ExternalGOTSym)
    GOT.getOrCreateSection(G, ExternalGOTSym);
  visitExistingEdges(G, GOT, PLT, TLSInfo);
  return Error::success();
}

static Error mergeTLSInfoIntoGOT(LinkGraph &G) {
  if (auto *TLSInfo = G.findSectionByName("$__TLSINFO")) {
    auto *GOT =
        G.findSectionByName(GOTTableManager_ELF_ppc32::getSectionName());
    assert(GOT && "TLS info entries require a GOT base");
    // Keep TLS info in GOT reach after initializing its keys.
    G.mergeSections(*GOT, *TLSInfo);
  }
  return Error::success();
}

template <endianness Endian>
class ELFLinkGraphBuilder_ppc32
    : public ELFLinkGraphBuilder<object::ELFType<Endian, false>> {
  using ELFT = object::ELFType<Endian, false>;
  using Base = ELFLinkGraphBuilder<ELFT>;

  Error addRelocations() override {
    for (const auto &RelSect : Base::Sections) {
      if (RelSect.sh_type == ELF::SHT_REL)
        return make_error<JITLinkError>(
            "No SHT_REL in valid PPC32 ELF object files");
      if (Error Err = Base::forEachRelaRelocation(
              RelSect, this, &ELFLinkGraphBuilder_ppc32::addRelaRelocation))
        return Err;
    }
    return Error::success();
  }

  Error addRelaRelocation(const typename ELFT::Rela &Rel,
                          const typename ELFT::Shdr &Sect, Block &B) {
    int64_t Addend = Rel.r_addend;
    uint32_t Type = Rel.getType(false);
    if (Type == ELF::R_PPC_NONE || Type == ELF::R_PPC_TLSGD ||
        Type == ELF::R_PPC_TLSLD)
      return Error::success();

    Edge::Kind K;
    switch (Type) {
    case ELF::R_PPC_GOT_TLSGD16:
      K = ppc32::RequestTLSDescInGOTAndTransformToGOTDelta16;
      break;
    case ELF::R_PPC_GOT_TLSGD16_LO:
      K = ppc32::RequestTLSDescInGOTAndTransformToGOTDelta16LO;
      break;
    case ELF::R_PPC_GOT_TLSGD16_HI:
      K = ppc32::RequestTLSDescInGOTAndTransformToGOTDelta16HI;
      break;
    case ELF::R_PPC_GOT_TLSGD16_HA:
      K = ppc32::RequestTLSDescInGOTAndTransformToGOTDelta16HA;
      break;
    case ELF::R_PPC_GOT_TLSLD16:
      K = ppc32::RequestTLSLDAndTransformToGOTDelta16;
      break;
    case ELF::R_PPC_GOT_TLSLD16_LO:
      K = ppc32::RequestTLSLDAndTransformToGOTDelta16LO;
      break;
    case ELF::R_PPC_GOT_TLSLD16_HI:
      K = ppc32::RequestTLSLDAndTransformToGOTDelta16HI;
      break;
    case ELF::R_PPC_GOT_TLSLD16_HA:
      K = ppc32::RequestTLSLDAndTransformToGOTDelta16HA;
      break;
    case ELF::R_PPC_DTPREL32:
      K = ppc32::DTPRel32;
      break;
    case ELF::R_PPC_DTPREL16:
      K = ppc32::DTPRel16;
      break;
    case ELF::R_PPC_DTPREL16_LO:
      K = ppc32::DTPRel16LO;
      break;
    case ELF::R_PPC_DTPREL16_HI:
      K = ppc32::DTPRel16HI;
      break;
    case ELF::R_PPC_DTPREL16_HA:
      K = ppc32::DTPRel16HA;
      break;
    case ELF::R_PPC_ADDR32:
    case ELF::R_PPC_UADDR32:
      K = ppc32::Pointer32;
      break;
    case ELF::R_PPC_ADDR16:
    case ELF::R_PPC_UADDR16:
      K = ppc32::Pointer16;
      break;
    case ELF::R_PPC_ADDR16_LO:
      K = ppc32::Pointer16LO;
      break;
    case ELF::R_PPC_ADDR16_HI:
      K = ppc32::Pointer16HI;
      break;
    case ELF::R_PPC_ADDR16_HA:
      K = ppc32::Pointer16HA;
      break;
    case ELF::R_PPC_REL32:
      K = ppc32::Delta32;
      break;
    case ELF::R_PPC_GOT16:
      K = ppc32::RequestGOTAndTransformToGOTDelta16;
      break;
    case ELF::R_PPC_GOT16_LO:
      K = ppc32::RequestGOTAndTransformToGOTDelta16LO;
      break;
    case ELF::R_PPC_GOT16_HI:
      K = ppc32::RequestGOTAndTransformToGOTDelta16HI;
      break;
    case ELF::R_PPC_GOT16_HA:
      K = ppc32::RequestGOTAndTransformToGOTDelta16HA;
      break;
    case ELF::R_PPC_REL16:
      K = ppc32::Delta16;
      break;
    case ELF::R_PPC_REL16_LO:
      K = ppc32::Delta16LO;
      break;
    case ELF::R_PPC_REL16_HI:
      K = ppc32::Delta16HI;
      break;
    case ELF::R_PPC_REL16_HA:
      K = ppc32::Delta16HA;
      break;
    case ELF::R_PPC_REL24:
    case ELF::R_PPC_LOCAL24PC:
    case ELF::R_PPC_PLTREL24:
      K = ppc32::Branch24;
      break;
    case ELF::R_PPC_REL14:
      K = ppc32::Branch14;
      break;
    case ELF::R_PPC_ADDR24:
      K = ppc32::Branch24Absolute;
      break;
    case ELF::R_PPC_ADDR14:
      K = ppc32::Branch14Absolute;
      break;
    default:
      return make_error<JITLinkError>(
          "Unsupported PPC32 relocation " +
          object::getELFRelocationTypeName(ELF::EM_PPC, Type));
    }

    auto *Target = Base::getGraphSymbol(Rel.getSymbol(false));
    if (!Target)
      return make_error<JITLinkError>(
          "PPC32 relocation references missing symbol");

    auto FixupAddress = orc::ExecutorAddr(Sect.sh_addr) + Rel.r_offset;
    Edge::OffsetT Offset = FixupAddress - B.getAddress();
    unsigned Size = 2;
    switch (K) {
    case ppc32::Pointer32:
    case ppc32::Delta32:
    case ppc32::DTPRel32:
    case ppc32::Branch24:
    case ppc32::Branch14:
    case ppc32::Branch24Absolute:
    case ppc32::Branch14Absolute:
      Size = 4;
      break;
    default:
      break;
    }
    if (Offset > B.getSize() || Size > B.getSize() - Offset)
      return make_error<JITLinkError>("PPC32 relocation outside block");
    // The PLTREL24 addend selects a GOT base in the static linker. Our stub
    // uses its own GOT entry and therefore branches to the stub start.
    if (Type == ELF::R_PPC_PLTREL24)
      Addend = 0;
    B.addEdge(K, Offset, *Target, Addend);
    return Error::success();
  }

public:
  ELFLinkGraphBuilder_ppc32(StringRef Name, const object::ELFFile<ELFT> &Obj,
                            std::shared_ptr<orc::SymbolStringPool> SSP,
                            Triple TT, SubtargetFeatures Features)
      : Base(Obj, std::move(SSP), std::move(TT), std::move(Features), Name,
             ppc32::getEdgeKindName) {}
};

class ELFJITLinker_ppc32 : public JITLinker<ELFJITLinker_ppc32> {
  friend class JITLinker<ELFJITLinker_ppc32>;

public:
  ELFJITLinker_ppc32(std::unique_ptr<JITLinkContext> Ctx,
                     std::unique_ptr<LinkGraph> G, PassConfiguration Config)
      : JITLinker(std::move(Ctx), std::move(G), std::move(Config)) {
    getPassConfig().PostAllocationPasses.push_back(
        [this](LinkGraph &G) { return findBaseSymbols(G); });
  }

private:
  Symbol *GOTSymbol = nullptr;
  Symbol *TLSBaseSymbol = nullptr;

  Error findBaseSymbols(LinkGraph &G) {
    TLSBaseSymbol = G.findDefinedSymbolByName(G.intern(ELFTLSBaseSymbolName));
    if (auto *GOT =
            G.findSectionByName(GOTTableManager_ELF_ppc32::getSectionName()))
      for (auto *Sym : GOT->symbols())
        if (Sym->hasName() && *Sym->getName() == ELFGOTSymbolName) {
          GOTSymbol = Sym;
          break;
        }
    return Error::success();
  }

  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    return ppc32::applyFixup(G, B, E, GOTSymbol, TLSBaseSymbol);
  }
};

} // namespace

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_ppc32(MemoryBufferRef ObjectBuffer,
                                   std::shared_ptr<orc::SymbolStringPool> SSP) {
  auto Obj = object::ObjectFile::createELFObjectFile(ObjectBuffer);
  if (!Obj)
    return Obj.takeError();
  auto Features = (*Obj)->getFeatures();
  if (!Features)
    return Features.takeError();
  auto TT = (*Obj)->makeTriple();
  if (TT.getArch() == Triple::ppc) {
    auto &File =
        cast<object::ELFObjectFile<object::ELF32BE>>(**Obj).getELFFile();
    return ELFLinkGraphBuilder_ppc32<endianness::big>((*Obj)->getFileName(),
                                                      File, std::move(SSP), TT,
                                                      std::move(*Features))
        .buildGraph();
  }
  if (TT.getArch() == Triple::ppcle) {
    auto &File =
        cast<object::ELFObjectFile<object::ELF32LE>>(**Obj).getELFFile();
    return ELFLinkGraphBuilder_ppc32<endianness::little>(
               (*Obj)->getFileName(), File, std::move(SSP), TT,
               std::move(*Features))
        .buildGraph();
  }
  return make_error<JITLinkError>("Invalid PPC32 ELF target triple");
}

void link_ELF_ppc32(std::unique_ptr<LinkGraph> G,
                    std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;
  if (Ctx->shouldAddDefaultTargetPasses(G->getTargetTriple())) {
    Config.PrePrunePasses.push_back(DWARFRecordSectionSplitter(".eh_frame"));
    Config.PrePrunePasses.push_back(
        EHFrameEdgeFixer(".eh_frame", 4, ppc32::Pointer32, ppc32::Pointer32,
                         ppc32::Delta32, ppc32::Delta32, ppc32::NegDelta32));
    Config.PrePrunePasses.push_back(EHFrameNullTerminator(".eh_frame"));
    if (auto MarkLive = Ctx->getMarkLivePass(G->getTargetTriple()))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);
  }
  Config.PostPrunePasses.push_back(buildTables_ELF_ppc32);
  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));
  Config.PostPrunePasses.push_back(mergeTLSInfoIntoGOT);
  ELFJITLinker_ppc32::link(std::move(Ctx), std::move(G), std::move(Config));
}

} // namespace llvm::jitlink
