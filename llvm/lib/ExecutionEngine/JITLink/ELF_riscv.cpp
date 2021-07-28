//===------- ELF_riscv.cpp -JIT linker implementation for ELF/riscv -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ELF/riscv jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF_riscv.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/riscv.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"

#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"

#define DEBUG_TYPE "jitlink"
using namespace llvm;

namespace llvm {
namespace jitlink {

static Expected<const Edge &> getRISCVPCRelHi20(const Edge &E) {
  using namespace riscv;
  assert((E.getKind() == R_RISCV_PCREL_LO12_I ||
          E.getKind() == R_RISCV_PCREL_LO12_S) &&
         "Can only have high relocation for R_RISCV_PCREL_LO12_I or "
         "R_RISCV_PCREL_LO12_S");

  const Symbol &Sym = E.getTarget();
  const Block &B = Sym.getBlock();
  JITTargetAddress Offset = Sym.getOffset();

  struct Comp {
    bool operator()(const Edge &Lhs, JITTargetAddress Offset) {
      return Lhs.getOffset() < Offset;
    }
    bool operator()(JITTargetAddress Offset, const Edge &Rhs) {
      return Offset < Rhs.getOffset();
    }
  };

  auto Bound =
      std::equal_range(B.edges().begin(), B.edges().end(), Offset, Comp{});

  for (auto It = Bound.first; It != Bound.second; ++It) {
    if (It->getKind() == R_RISCV_PCREL_HI20)
      return *It;
  }

  return make_error<JITLinkError>(
      "No HI20 PCREL relocation type be found for LO12 PCREL relocation type");
}

static uint32_t extractBits(uint64_t Num, unsigned High, unsigned Low) {
  return (Num & ((1ULL << (High + 1)) - 1)) >> Low;
}

class ELFJITLinker_riscv : public JITLinker<ELFJITLinker_riscv> {
  friend class JITLinker<ELFJITLinker_riscv>;

public:
  ELFJITLinker_riscv(std::unique_ptr<JITLinkContext> Ctx,
                     std::unique_ptr<LinkGraph> G, PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {}

private:
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    using namespace riscv;
    using namespace llvm::support;

    char *BlockWorkingMem = B.getAlreadyMutableContent().data();
    char *FixupPtr = BlockWorkingMem + E.getOffset();
    JITTargetAddress FixupAddress = B.getAddress() + E.getOffset();
    switch (E.getKind()) {
    case R_RISCV_HI20: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend();
      int32_t Hi = (Value + 0x800) & 0xFFFFF000;
      uint32_t RawInstr = *(little32_t *)FixupPtr;
      *(little32_t *)FixupPtr = (RawInstr & 0xFFF) | static_cast<uint32_t>(Hi);
      break;
    }
    case R_RISCV_LO12_I: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend();
      int32_t Lo = Value & 0xFFF;
      uint32_t RawInstr = *(little32_t *)FixupPtr;
      *(little32_t *)FixupPtr =
          (RawInstr & 0xFFFFF) | (static_cast<uint32_t>(Lo & 0xFFF) << 20);
      break;
    }
    case R_RISCV_CALL: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend() - FixupAddress;
      int32_t Hi = (Value + 0x800) & 0xFFFFF000;
      int32_t Lo = Value & 0xFFF;
      uint32_t RawInstrAuipc = *(little32_t *)FixupPtr;
      uint32_t RawInstrJalr = *(little32_t *)(FixupPtr + 4);
      *(little32_t *)FixupPtr = RawInstrAuipc | static_cast<uint32_t>(Hi);
      *(little32_t *)(FixupPtr + 4) =
          RawInstrJalr | (static_cast<uint32_t>(Lo) << 20);
      break;
    }
    case R_RISCV_PCREL_HI20: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend() - FixupAddress;
      int32_t Hi = (Value + 0x800) & 0xFFFFF000;
      uint32_t RawInstr = *(little32_t *)FixupPtr;
      *(little32_t *)FixupPtr = (RawInstr & 0xFFF) | static_cast<uint32_t>(Hi);
      break;
    }
    case R_RISCV_PCREL_LO12_I: {
      auto RelHI20 = getRISCVPCRelHi20(E);
      if (!RelHI20)
        return RelHI20.takeError();
      int64_t Value = RelHI20->getTarget().getAddress() +
                      RelHI20->getAddend() - E.getTarget().getAddress();
      int64_t Lo = Value & 0xFFF;
      uint32_t RawInstr = *(little32_t *)FixupPtr;
      *(little32_t *)FixupPtr =
          (RawInstr & 0xFFFFF) | (static_cast<uint32_t>(Lo & 0xFFF) << 20);
      break;
    }
    case R_RISCV_PCREL_LO12_S: {
      auto RelHI20 = getRISCVPCRelHi20(E);
      int64_t Value = RelHI20->getTarget().getAddress() +
                      RelHI20->getAddend() - E.getTarget().getAddress();
      int64_t Lo = Value & 0xFFF;
      uint32_t Imm31_25 = extractBits(Lo, 11, 5) << 25;
      uint32_t Imm11_7 = extractBits(Lo, 4, 0) << 7;
      uint32_t RawInstr = *(little32_t *)FixupPtr;

      *(little32_t *)FixupPtr = (RawInstr & 0x1FFF07F) | Imm31_25 | Imm11_7;
      break;
    }
    }
    return Error::success();
  }
};

template <typename ELFT>
class ELFLinkGraphBuilder_riscv : public ELFLinkGraphBuilder<ELFT> {
private:
  static Expected<riscv::EdgeKind_riscv>
  getRelocationKind(const uint32_t Type) {
    using namespace riscv;
    switch (Type) {
    case ELF::R_RISCV_32:
      return EdgeKind_riscv::R_RISCV_32;
    case ELF::R_RISCV_64:
      return EdgeKind_riscv::R_RISCV_64;
    case ELF::R_RISCV_HI20:
      return EdgeKind_riscv::R_RISCV_HI20;
    case ELF::R_RISCV_LO12_I:
      return EdgeKind_riscv::R_RISCV_LO12_I;
    case ELF::R_RISCV_CALL:
      return EdgeKind_riscv::R_RISCV_CALL;
    case ELF::R_RISCV_PCREL_HI20:
      return EdgeKind_riscv::R_RISCV_PCREL_HI20;
    case ELF::R_RISCV_PCREL_LO12_I:
      return EdgeKind_riscv::R_RISCV_PCREL_LO12_I;
    case ELF::R_RISCV_PCREL_LO12_S:
      return EdgeKind_riscv::R_RISCV_PCREL_LO12_S;
    }

    return make_error<JITLinkError>("Unsupported riscv relocation:" +
                                    formatv("{0:d}", Type));
  }

  Error addRelocations() override {
    using Base = ELFLinkGraphBuilder<ELFT>;
    LLVM_DEBUG(dbgs() << "Adding relocations\n");

    // TODO a partern is forming of iterate some sections but only give me
    // ones I am interested, I should abstract that concept some where
    for (auto &SecRef : Base::Sections) {
      if (SecRef.sh_type != ELF::SHT_RELA && SecRef.sh_type != ELF::SHT_REL)
        continue;
      auto RelSectName = Base::Obj.getSectionName(SecRef);
      if (!RelSectName)
        return RelSectName.takeError();

      LLVM_DEBUG({
        dbgs() << "Adding relocations from section " << *RelSectName << "\n";
      });

      auto UpdateSection = Base::Obj.getSection(SecRef.sh_info);
      if (!UpdateSection)
        return UpdateSection.takeError();

      auto UpdateSectionName = Base::Obj.getSectionName(**UpdateSection);
      if (!UpdateSectionName)
        return UpdateSectionName.takeError();
      // Don't process relocations for debug sections.
      if (Base::isDwarfSection(*UpdateSectionName)) {
        LLVM_DEBUG({
          dbgs() << "  Target is dwarf section " << *UpdateSectionName
                 << ". Skipping.\n";
        });
        continue;
      } else
        LLVM_DEBUG({
          dbgs() << "  For target section " << *UpdateSectionName << "\n";
        });

      auto *JITSection = Base::G->findSectionByName(*UpdateSectionName);
      if (!JITSection)
        return make_error<llvm::StringError>(
            "Refencing a section that wasn't added to graph" +
                *UpdateSectionName,
            llvm::inconvertibleErrorCode());

      auto Relocations = Base::Obj.relas(SecRef);
      if (!Relocations)
        return Relocations.takeError();

      for (const auto &Rela : *Relocations) {
        auto Type = Rela.getType(false);

        LLVM_DEBUG({
          dbgs() << "Relocation Type: " << Type << "\n"
                 << "Name: " << Base::Obj.getRelocationTypeName(Type) << "\n";
        });

        auto SymbolIndex = Rela.getSymbol(false);
        auto Symbol = Base::Obj.getRelocationSymbol(Rela, Base::SymTabSec);
        if (!Symbol)
          return Symbol.takeError();

        auto BlockToFix = *(JITSection->blocks().begin());
        auto *TargetSymbol = Base::getGraphSymbol(SymbolIndex);

        if (!TargetSymbol) {
          return make_error<llvm::StringError>(
              "Could not find symbol at given index, did you add it to "
              "JITSymbolTable? index: " +
                  std::to_string(SymbolIndex) + ", shndx: " +
                  std::to_string((*Symbol)->st_shndx) + " Size of table: " +
                  std::to_string(Base::GraphSymbols.size()),
              llvm::inconvertibleErrorCode());
        }
        int64_t Addend = Rela.r_addend;
        JITTargetAddress FixupAddress =
            (*UpdateSection)->sh_addr + Rela.r_offset;

        LLVM_DEBUG({
          dbgs() << "Processing relocation at "
                 << format("0x%016" PRIx64, FixupAddress) << "\n";
        });
        auto Kind = getRelocationKind(Type);
        if (!Kind)
          return Kind.takeError();

        BlockToFix->addEdge(*Kind, FixupAddress - BlockToFix->getAddress(),
                            *TargetSymbol, Addend);
      }
    }
    return Error::success();
  }

public:
  ELFLinkGraphBuilder_riscv(StringRef FileName,
                            const object::ELFFile<ELFT> &Obj, const Triple T)
      : ELFLinkGraphBuilder<ELFT>(Obj, std::move(T), FileName,
                                  riscv::getEdgeKindName) {}
};

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_riscv(MemoryBufferRef ObjectBuffer) {
  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << ObjectBuffer.getBufferIdentifier() << "...\n";
  });

  auto ELFObj = object::ObjectFile::createELFObjectFile(ObjectBuffer);
  if (!ELFObj)
    return ELFObj.takeError();

  if ((*ELFObj)->getArch() == Triple::riscv64) {
    auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF64LE>>(**ELFObj);
    return ELFLinkGraphBuilder_riscv<object::ELF64LE>(
               (*ELFObj)->getFileName(), ELFObjFile.getELFFile(),
               (*ELFObj)->makeTriple())
        .buildGraph();
  } else {
    assert((*ELFObj)->getArch() == Triple::riscv32 &&
           "Invalid triple for RISCV ELF object file");
    auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF32LE>>(**ELFObj);
    return ELFLinkGraphBuilder_riscv<object::ELF32LE>(
               (*ELFObj)->getFileName(), ELFObjFile.getELFFile(),
               (*ELFObj)->makeTriple())
        .buildGraph();
  }
}

void link_ELF_riscv(std::unique_ptr<LinkGraph> G,
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

  ELFJITLinker_riscv::link(std::move(Ctx), std::move(G), std::move(Config));
}

} // namespace jitlink
} // namespace llvm
