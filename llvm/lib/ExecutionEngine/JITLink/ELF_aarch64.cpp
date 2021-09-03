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
    using namespace aarch64;
    using namespace llvm::support;

    char *BlockWorkingMem = B.getAlreadyMutableContent().data();
    char *FixupPtr = BlockWorkingMem + E.getOffset();
    JITTargetAddress FixupAddress = B.getAddress() + E.getOffset();
    switch (E.getKind()) {
    case aarch64::R_AARCH64_CALL26: {
      assert((FixupAddress & 0x3) == 0 && "Call-inst is not 32-bit aligned");
      int64_t Value = E.getTarget().getAddress() - FixupAddress + E.getAddend();

      if (static_cast<uint64_t>(Value) & 0x3)
        return make_error<JITLinkError>("Call target is not 32-bit aligned");

      if (!fitsRangeSignedInt<27>(Value))
        return makeTargetOutOfRangeError(G, B, E);

      uint32_t RawInstr = *(little32_t *)FixupPtr;
      assert((RawInstr & 0x7fffffff) == 0x14000000 &&
             "RawInstr isn't a B or BR immediate instruction");
      uint32_t Imm = (static_cast<uint32_t>(Value) & ((1 << 28) - 1)) >> 2;
      uint32_t FixedInstr = RawInstr | Imm;
      *(little32_t *)FixupPtr = FixedInstr;
      break;
    }
    }
    return Error::success();
  }

  template <uint8_t Bits> static bool fitsRangeSignedInt(int64_t Value) {
    return Value >= -(1 << Bits) && Value < (1 << Bits);
  }
};

template <typename ELFT>
class ELFLinkGraphBuilder_aarch64 : public ELFLinkGraphBuilder<ELFT> {
private:
  static Expected<aarch64::EdgeKind_aarch64>
  getRelocationKind(const uint32_t Type) {
    using namespace aarch64;
    switch (Type) {
    case ELF::R_AARCH64_CALL26:
      return EdgeKind_aarch64::R_AARCH64_CALL26;
    }

    return make_error<JITLinkError>("Unsupported aarch64 relocation:" +
                                    formatv("{0:d}", Type));
  }

  Error addRelocations() override {
    using Base = ELFLinkGraphBuilder<ELFT>;
    LLVM_DEBUG(dbgs() << "Adding relocations\n");

    // Iterate sections and only process the interesting ones.
    for (auto &SecRef : Base::Sections) {
      if (SecRef.sh_type != ELF::SHT_RELA && SecRef.sh_type != ELF::SHT_REL)
        continue;
      auto RelSectName = Base::Obj.getSectionName(SecRef);
      if (!RelSectName)
        return RelSectName.takeError();

      LLVM_DEBUG({
        dbgs() << "Adding relocations from section " << *RelSectName << "\n";
      });

      auto UpdateSect = Base::Obj.getSection(SecRef.sh_info);
      if (!UpdateSect)
        return UpdateSect.takeError();

      auto UpdateSectName = Base::Obj.getSectionName(**UpdateSect);
      if (!UpdateSectName)
        return UpdateSectName.takeError();

      // Don't process relocations for debug sections.
      if (Base::isDwarfSection(*UpdateSectName)) {
        LLVM_DEBUG({
          dbgs() << "  Target is dwarf section " << *UpdateSectName
                 << ". Skipping.\n";
        });
        continue;
      }
      LLVM_DEBUG(dbgs() << "  For target section " << *UpdateSectName << "\n");

      auto *JITSection = Base::G->findSectionByName(*UpdateSectName);
      if (!JITSection)
        return make_error<llvm::StringError>(
            "Refencing a section that wasn't added to graph" + *UpdateSectName,
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
        JITTargetAddress FixupAddress = (*UpdateSect)->sh_addr + Rela.r_offset;

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
