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
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

#include "PerGraphGOTAndPLTStubsBuilder.h"

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
    ELFLdSt8Abs12,
    ELFLdSt16Abs12,
    ELFLdSt32Abs12,
    ELFLdSt64Abs12,
    ELFLdSt128Abs12,
    ELFAbs64,
    ELFAdrGOTPage21,
    ELFLd64GOTLo12,
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
    case ELF::R_AARCH64_LDST8_ABS_LO12_NC:
      return ELFLdSt8Abs12;
    case ELF::R_AARCH64_LDST16_ABS_LO12_NC:
      return ELFLdSt16Abs12;
    case ELF::R_AARCH64_LDST32_ABS_LO12_NC:
      return ELFLdSt32Abs12;
    case ELF::R_AARCH64_LDST64_ABS_LO12_NC:
      return ELFLdSt64Abs12;
    case ELF::R_AARCH64_LDST128_ABS_LO12_NC:
      return ELFLdSt128Abs12;
    case ELF::R_AARCH64_ABS64:
      return ELFAbs64;
    case ELF::R_AARCH64_ADR_GOT_PAGE:
      return ELFAdrGOTPage21;
    case ELF::R_AARCH64_LD64_GOT_LO12_NC:
      return ELFLd64GOTLo12;
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
    using support::ulittle32_t;
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

    // Get a pointer to the fixup content.
    const void *FixupContent = BlockToFix.getContent().data() +
                               (FixupAddress - BlockToFix.getAddress());

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
    case ELFLdSt8Abs12: {
      uint32_t Instr = *(const ulittle32_t *)FixupContent;
      if (!aarch64::isLoadStoreImm12(Instr) ||
          aarch64::getPageOffset12Shift(Instr) != 0)
        return make_error<JITLinkError>(
            "R_AARCH64_LDST8_ABS_LO12_NC target is not a "
            "LDRB/STRB (imm12) instruction");

      Kind = aarch64::PageOffset12;
      break;
    }
    case ELFLdSt16Abs12: {
      uint32_t Instr = *(const ulittle32_t *)FixupContent;
      if (!aarch64::isLoadStoreImm12(Instr) ||
          aarch64::getPageOffset12Shift(Instr) != 1)
        return make_error<JITLinkError>(
            "R_AARCH64_LDST16_ABS_LO12_NC target is not a "
            "LDRH/STRH (imm12) instruction");

      Kind = aarch64::PageOffset12;
      break;
    }
    case ELFLdSt32Abs12: {
      uint32_t Instr = *(const ulittle32_t *)FixupContent;
      if (!aarch64::isLoadStoreImm12(Instr) ||
          aarch64::getPageOffset12Shift(Instr) != 2)
        return make_error<JITLinkError>(
            "R_AARCH64_LDST32_ABS_LO12_NC target is not a "
            "LDR/STR (imm12, 32 bit) instruction");

      Kind = aarch64::PageOffset12;
      break;
    }
    case ELFLdSt64Abs12: {
      uint32_t Instr = *(const ulittle32_t *)FixupContent;
      if (!aarch64::isLoadStoreImm12(Instr) ||
          aarch64::getPageOffset12Shift(Instr) != 3)
        return make_error<JITLinkError>(
            "R_AARCH64_LDST64_ABS_LO12_NC target is not a "
            "LDR/STR (imm12, 64 bit) instruction");

      Kind = aarch64::PageOffset12;
      break;
    }
    case ELFLdSt128Abs12: {
      uint32_t Instr = *(const ulittle32_t *)FixupContent;
      if (!aarch64::isLoadStoreImm12(Instr) ||
          aarch64::getPageOffset12Shift(Instr) != 4)
        return make_error<JITLinkError>(
            "R_AARCH64_LDST128_ABS_LO12_NC target is not a "
            "LDR/STR (imm12, 128 bit) instruction");

      Kind = aarch64::PageOffset12;
      break;
    }
    case ELFAbs64: {
      Kind = aarch64::Pointer64;
      break;
    }
    case ELFAdrGOTPage21: {
      Kind = aarch64::GOTPage21;
      break;
    }
    case ELFLd64GOTLo12: {
      Kind = aarch64::GOTPageOffset12;
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
    case ELFLdSt8Abs12:
      return "ELFLdSt8Abs12";
    case ELFLdSt16Abs12:
      return "ELFLdSt16Abs12";
    case ELFLdSt32Abs12:
      return "ELFLdSt32Abs12";
    case ELFLdSt64Abs12:
      return "ELFLdSt64Abs12";
    case ELFLdSt128Abs12:
      return "ELFLdSt128Abs12";
    case ELFAbs64:
      return "ELFAbs64";
    case ELFAdrGOTPage21:
      return "ELFAdrGOTPage21";
    case ELFLd64GOTLo12:
      return "ELFLd64GOTLo12";
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

class PerGraphGOTAndPLTStubsBuilder_ELF_arm64
    : public PerGraphGOTAndPLTStubsBuilder<
          PerGraphGOTAndPLTStubsBuilder_ELF_arm64> {
public:
  using PerGraphGOTAndPLTStubsBuilder<
      PerGraphGOTAndPLTStubsBuilder_ELF_arm64>::PerGraphGOTAndPLTStubsBuilder;

  bool isGOTEdgeToFix(Edge &E) const {
    return E.getKind() == aarch64::GOTPage21 ||
           E.getKind() == aarch64::GOTPageOffset12;
  }

  Symbol &createGOTEntry(Symbol &Target) {
    auto &GOTEntryBlock = G.createContentBlock(
        getGOTSection(), getGOTEntryBlockContent(), orc::ExecutorAddr(), 8, 0);
    GOTEntryBlock.addEdge(aarch64::Pointer64, 0, Target, 0);
    return G.addAnonymousSymbol(GOTEntryBlock, 0, 8, false, false);
  }

  void fixGOTEdge(Edge &E, Symbol &GOTEntry) {
    if (E.getKind() == aarch64::GOTPage21) {
      E.setKind(aarch64::Page21);
      E.setTarget(GOTEntry);
    } else if (E.getKind() == aarch64::GOTPageOffset12) {
      E.setKind(aarch64::PageOffset12);
      E.setTarget(GOTEntry);
    } else
      llvm_unreachable("Not a GOT edge?");
  }

  bool isExternalBranchEdge(Edge &E) { return false; }

  Symbol &createPLTStub(Symbol &Target) {
    assert(false && "unimplemetned");
    return Target;
  }

  void fixPLTEdge(Edge &E, Symbol &Stub) { assert(false && "unimplemetned"); }

private:
  Section &getGOTSection() {
    if (!GOTSection)
      GOTSection = &G.createSection("$__GOT", MemProt::Read | MemProt::Exec);
    return *GOTSection;
  }

  ArrayRef<char> getGOTEntryBlockContent() {
    return {reinterpret_cast<const char *>(NullGOTEntryContent),
            sizeof(NullGOTEntryContent)};
  }

  static const uint8_t NullGOTEntryContent[8];
  Section *GOTSection = nullptr;
};

const uint8_t PerGraphGOTAndPLTStubsBuilder_ELF_arm64::NullGOTEntryContent[8] =
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

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

  Config.PostPrunePasses.push_back(
      PerGraphGOTAndPLTStubsBuilder_ELF_arm64::asPass);

  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  ELFJITLinker_aarch64::link(std::move(Ctx), std::move(G), std::move(Config));
}

} // namespace jitlink
} // namespace llvm
