//===---- ELF_x86_64.cpp -JIT linker implementation for ELF/x86-64 ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ELF/x86-64 jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF_x86_64.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Endian.h"

#include "DefineExternalSectionStartAndEndSymbols.h"
#include "EHFrameSupportImpl.h"
#include "ELFLinkGraphBuilder.h"
#include "JITLinkGeneric.h"
#include "PerGraphGOTAndPLTStubsBuilder.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::ELF_x86_64_Edges;

namespace {

constexpr StringRef ELFGOTSectionName = "$__GOT";
constexpr StringRef ELFGOTSymbolName = "_GLOBAL_OFFSET_TABLE_";

class PerGraphGOTAndPLTStubsBuilder_ELF_x86_64
    : public PerGraphGOTAndPLTStubsBuilder<
          PerGraphGOTAndPLTStubsBuilder_ELF_x86_64> {
public:
  static const uint8_t NullGOTEntryContent[8];
  static const uint8_t StubContent[6];

  using PerGraphGOTAndPLTStubsBuilder<
      PerGraphGOTAndPLTStubsBuilder_ELF_x86_64>::PerGraphGOTAndPLTStubsBuilder;

  bool isGOTEdgeToFix(Edge &E) const {
    if (E.getKind() == GOTOFF64) {
      // We need to make sure that the GOT section exists, but don't otherwise
      // need to fix up this edge.
      getGOTSection();
      return false;
    }

    return E.getKind() == PCRel32GOT || E.getKind() == PCRel32GOTLoad ||
           E.getKind() == PCRel64GOT || E.getKind() == GOT64;
  }

  Symbol &createGOTEntry(Symbol &Target) {
    auto &GOTEntryBlock = G.createContentBlock(
        getGOTSection(), getGOTEntryBlockContent(), 0, 8, 0);
    GOTEntryBlock.addEdge(Pointer64, 0, Target, 0);
    return G.addAnonymousSymbol(GOTEntryBlock, 0, 8, false, false);
  }

  void fixGOTEdge(Edge &E, Symbol &GOTEntry) {
    // If this is a PCRel32GOT/PCRel64GOT then change it to an ordinary
    // PCRel32/PCRel64. If it is a PCRel32GOTLoad then leave it as-is for now:
    // We will use the kind to check for GOT optimization opportunities in the
    // optimizeMachO_x86_64_GOTAndStubs pass below.
    // If it's a GOT64 leave it as is.
    switch (E.getKind()) {
    case PCRel32GOT:
      E.setKind(PCRel32);
      break;
    case PCRel64GOT:
      E.setKind(PCRel64);
      break;
    case GOT64:
      break;
    case PCRel32GOTLoad:
      break;
    default:
      llvm_unreachable("Unexpected GOT edge kind");
    }

    E.setTarget(GOTEntry);
    // Leave the edge addend as-is.
  }

  bool isExternalBranchEdge(Edge &E) {
    return E.getKind() == Branch32 && !E.getTarget().isDefined();
  }

  Symbol &createPLTStub(Symbol &Target) {
    auto &StubContentBlock =
        G.createContentBlock(getStubsSection(), getStubBlockContent(), 0, 1, 0);
    // Re-use GOT entries for stub targets.
    auto &GOTEntrySymbol = getGOTEntry(Target);
    StubContentBlock.addEdge(PCRel32, 2, GOTEntrySymbol, -4);
    return G.addAnonymousSymbol(StubContentBlock, 0, 6, true, false);
  }

  void fixPLTEdge(Edge &E, Symbol &Stub) {
    assert(E.getKind() == Branch32 && "Not a Branch32 edge?");

    // Set the edge kind to Branch32ToStub. We will use this to check for stub
    // optimization opportunities in the optimize ELF_x86_64_GOTAndStubs pass
    // below.
    E.setKind(Branch32ToStub);
    E.setTarget(Stub);
  }

private:
  Section &getGOTSection() const {
    if (!GOTSection)
      GOTSection = &G.createSection(ELFGOTSectionName, sys::Memory::MF_READ);
    return *GOTSection;
  }

  Section &getStubsSection() const {
    if (!StubsSection) {
      auto StubsProt = static_cast<sys::Memory::ProtectionFlags>(
          sys::Memory::MF_READ | sys::Memory::MF_EXEC);
      StubsSection = &G.createSection("$__STUBS", StubsProt);
    }
    return *StubsSection;
  }

  ArrayRef<char> getGOTEntryBlockContent() {
    return {reinterpret_cast<const char *>(NullGOTEntryContent),
            sizeof(NullGOTEntryContent)};
  }

  ArrayRef<char> getStubBlockContent() {
    return {reinterpret_cast<const char *>(StubContent), sizeof(StubContent)};
  }

  mutable Section *GOTSection = nullptr;
  mutable Section *StubsSection = nullptr;
};

} // namespace

const uint8_t PerGraphGOTAndPLTStubsBuilder_ELF_x86_64::NullGOTEntryContent[8] =
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const uint8_t PerGraphGOTAndPLTStubsBuilder_ELF_x86_64::StubContent[6] = {
    0xFF, 0x25, 0x00, 0x00, 0x00, 0x00};

static Error optimizeELF_x86_64_GOTAndStubs(LinkGraph &G) {
  LLVM_DEBUG(dbgs() << "Optimizing GOT entries and stubs:\n");

  for (auto *B : G.blocks())
    for (auto &E : B->edges())
      if (E.getKind() == PCRel32GOTLoad) {
        // Replace GOT load with LEA only for MOVQ instructions.
        constexpr uint8_t MOVQRIPRel[] = {0x48, 0x8b};
        if (E.getOffset() < 3 ||
            strncmp(B->getContent().data() + E.getOffset() - 3,
                    reinterpret_cast<const char *>(MOVQRIPRel), 2) != 0)
          continue;

        auto &GOTBlock = E.getTarget().getBlock();
        assert(GOTBlock.getSize() == G.getPointerSize() &&
               "GOT entry block should be pointer sized");
        assert(GOTBlock.edges_size() == 1 &&
               "GOT entry should only have one outgoing edge");

        auto &GOTTarget = GOTBlock.edges().begin()->getTarget();
        JITTargetAddress EdgeAddr = B->getAddress() + E.getOffset();
        JITTargetAddress TargetAddr = GOTTarget.getAddress();

        int64_t Displacement = TargetAddr - EdgeAddr + 4;
        if (Displacement >= std::numeric_limits<int32_t>::min() &&
            Displacement <= std::numeric_limits<int32_t>::max()) {
          // Change the edge kind as we don't go through GOT anymore. This is
          // for formal correctness only. Technically, the two relocation kinds
          // are resolved the same way.
          E.setKind(PCRel32);
          E.setTarget(GOTTarget);
          auto *BlockData = reinterpret_cast<uint8_t *>(
              const_cast<char *>(B->getContent().data()));
          BlockData[E.getOffset() - 2] = 0x8d;
          LLVM_DEBUG({
            dbgs() << "  Replaced GOT load wih LEA:\n    ";
            printEdge(dbgs(), *B, E, getELFX86RelocationKindName(E.getKind()));
            dbgs() << "\n";
          });
        }
      } else if (E.getKind() == Branch32ToStub) {
        auto &StubBlock = E.getTarget().getBlock();
        assert(
            StubBlock.getSize() ==
                sizeof(PerGraphGOTAndPLTStubsBuilder_ELF_x86_64::StubContent) &&
            "Stub block should be stub sized");
        assert(StubBlock.edges_size() == 1 &&
               "Stub block should only have one outgoing edge");

        auto &GOTBlock = StubBlock.edges().begin()->getTarget().getBlock();
        assert(GOTBlock.getSize() == G.getPointerSize() &&
               "GOT block should be pointer sized");
        assert(GOTBlock.edges_size() == 1 &&
               "GOT block should only have one outgoing edge");

        auto &GOTTarget = GOTBlock.edges().begin()->getTarget();
        JITTargetAddress EdgeAddr = B->getAddress() + E.getOffset();
        JITTargetAddress TargetAddr = GOTTarget.getAddress();

        int64_t Displacement = TargetAddr - EdgeAddr + 4;
        if (Displacement >= std::numeric_limits<int32_t>::min() &&
            Displacement <= std::numeric_limits<int32_t>::max()) {
          E.setKind(Branch32);
          E.setTarget(GOTTarget);
          LLVM_DEBUG({
            dbgs() << "  Replaced stub branch with direct branch:\n    ";
            printEdge(dbgs(), *B, E, getELFX86RelocationKindName(E.getKind()));
            dbgs() << "\n";
          });
        }
      }

  return Error::success();
}

namespace llvm {
namespace jitlink {

// This should become a template as the ELFFile is so a lot of this could become
// generic
class ELFLinkGraphBuilder_x86_64 : public ELFLinkGraphBuilder<object::ELF64LE> {
private:

  static Expected<ELF_x86_64_Edges::ELFX86RelocationKind>
  getRelocationKind(const uint32_t Type) {
    switch (Type) {
    case ELF::R_X86_64_PC32:
      return ELF_x86_64_Edges::ELFX86RelocationKind::PCRel32;
    case ELF::R_X86_64_PC64:
    case ELF::R_X86_64_GOTPC64:
      return ELF_x86_64_Edges::ELFX86RelocationKind::Delta64;
    case ELF::R_X86_64_64:
      return ELF_x86_64_Edges::ELFX86RelocationKind::Pointer64;
    case ELF::R_X86_64_GOTPCREL:
    case ELF::R_X86_64_GOTPCRELX:
    case ELF::R_X86_64_REX_GOTPCRELX:
      return ELF_x86_64_Edges::ELFX86RelocationKind::PCRel32GOTLoad;
    case ELF::R_X86_64_GOTPCREL64:
      return ELF_x86_64_Edges::ELFX86RelocationKind::PCRel64GOT;
    case ELF::R_X86_64_GOT64:
      return ELF_x86_64_Edges::ELFX86RelocationKind::GOT64;
    case ELF::R_X86_64_GOTOFF64:
      return ELF_x86_64_Edges::ELFX86RelocationKind::GOTOFF64;
    case ELF::R_X86_64_PLT32:
      return ELF_x86_64_Edges::ELFX86RelocationKind::Branch32;
    }
    return make_error<JITLinkError>("Unsupported x86-64 relocation:" +
                                    formatv("{0:d}", Type));
  }

  Error addRelocations() override {
    LLVM_DEBUG(dbgs() << "Adding relocations\n");
    // TODO a partern is forming of iterate some sections but only give me
    // ones I am interested, i should abstract that concept some where
    for (auto &SecRef : Sections) {
      if (SecRef.sh_type != ELF::SHT_RELA && SecRef.sh_type != ELF::SHT_REL)
        continue;
      // TODO can the elf obj file do this for me?
      if (SecRef.sh_type == ELF::SHT_REL)
        return make_error<llvm::StringError>("Shouldn't have REL in x64",
                                             llvm::inconvertibleErrorCode());

      auto RelSectName = Obj.getSectionName(SecRef);
      if (!RelSectName)
        return RelSectName.takeError();

      LLVM_DEBUG({
        dbgs() << "Adding relocations from section " << *RelSectName << "\n";
      });

      auto UpdateSection = Obj.getSection(SecRef.sh_info);
      if (!UpdateSection)
        return UpdateSection.takeError();

      auto UpdateSectionName = Obj.getSectionName(**UpdateSection);
      if (!UpdateSectionName)
        return UpdateSectionName.takeError();

      // Don't process relocations for debug sections.
      if (isDwarfSection(*UpdateSectionName)) {
        LLVM_DEBUG({
          dbgs() << "  Target is dwarf section " << *UpdateSectionName
                 << ". Skipping.\n";
        });
        continue;
      } else
        LLVM_DEBUG({
          dbgs() << "  For target section " << *UpdateSectionName << "\n";
        });

      auto JITSection = G->findSectionByName(*UpdateSectionName);
      if (!JITSection)
        return make_error<llvm::StringError>(
            "Refencing a a section that wasn't added to graph" +
                *UpdateSectionName,
            llvm::inconvertibleErrorCode());

      auto Relocations = Obj.relas(SecRef);
      if (!Relocations)
        return Relocations.takeError();

      for (const auto &Rela : *Relocations) {
        auto Type = Rela.getType(false);

        LLVM_DEBUG({
          dbgs() << "Relocation Type: " << Type << "\n"
                 << "Name: " << Obj.getRelocationTypeName(Type) << "\n";
        });
        auto SymbolIndex = Rela.getSymbol(false);
        auto Symbol = Obj.getRelocationSymbol(Rela, SymTabSec);
        if (!Symbol)
          return Symbol.takeError();

        auto BlockToFix = *(JITSection->blocks().begin());
        auto *TargetSymbol = getGraphSymbol(SymbolIndex);

        if (!TargetSymbol) {
          return make_error<llvm::StringError>(
              "Could not find symbol at given index, did you add it to "
              "JITSymbolTable? index: " +
                  std::to_string(SymbolIndex) +
                  ", shndx: " + std::to_string((*Symbol)->st_shndx) +
                  " Size of table: " + std::to_string(GraphSymbols.size()),
              llvm::inconvertibleErrorCode());
        }
        uint64_t Addend = Rela.r_addend;
        JITTargetAddress FixupAddress =
            (*UpdateSection)->sh_addr + Rela.r_offset;

        LLVM_DEBUG({
          dbgs() << "Processing relocation at "
                 << format("0x%016" PRIx64, FixupAddress) << "\n";
        });
        auto Kind = getRelocationKind(Type);
        if (!Kind)
          return Kind.takeError();

        LLVM_DEBUG({
          Edge GE(*Kind, FixupAddress - BlockToFix->getAddress(), *TargetSymbol,
                  Addend);
          printEdge(dbgs(), *BlockToFix, GE,
                    getELFX86RelocationKindName(*Kind));
          dbgs() << "\n";
        });
        BlockToFix->addEdge(*Kind, FixupAddress - BlockToFix->getAddress(),
                            *TargetSymbol, Addend);
      }
    }
    return Error::success();
  }

public:
  ELFLinkGraphBuilder_x86_64(StringRef FileName,
                             const object::ELFFile<object::ELF64LE> &Obj)
      : ELFLinkGraphBuilder(Obj, Triple("x86_64-unknown-linux"), FileName,
                            getELFX86RelocationKindName) {}
};

class ELFJITLinker_x86_64 : public JITLinker<ELFJITLinker_x86_64> {
  friend class JITLinker<ELFJITLinker_x86_64>;

public:
  ELFJITLinker_x86_64(std::unique_ptr<JITLinkContext> Ctx,
                      std::unique_ptr<LinkGraph> G,
                      PassConfiguration PassConfig)
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
              if (Sym.getName() == ELFGOTSymbolName)
                if (auto *GOTSection = G.findSectionByName(ELFGOTSectionName)) {
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
    if (auto *GOTSection = G.findSectionByName(ELFGOTSectionName)) {

      // Check for an existing defined symbol.
      for (auto *Sym : GOTSection->symbols())
        if (Sym->getName() == ELFGOTSymbolName) {
          GOTSymbol = Sym;
          return Error::success();
        }

      // If there's no defined symbol then create one.
      SectionRange SR(*GOTSection);
      if (SR.empty())
        GOTSymbol = &G.addAbsoluteSymbol(ELFGOTSymbolName, 0, 0,
                                         Linkage::Strong, Scope::Local, true);
      else
        GOTSymbol =
            &G.addDefinedSymbol(*SR.getFirstBlock(), 0, ELFGOTSymbolName, 0,
                                Linkage::Strong, Scope::Local, false, true);
    }

    return Error::success();
  }

  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    using namespace ELF_x86_64_Edges;
    using namespace llvm::support;

    char *BlockWorkingMem = B.getAlreadyMutableContent().data();
    char *FixupPtr = BlockWorkingMem + E.getOffset();
    JITTargetAddress FixupAddress = B.getAddress() + E.getOffset();
    switch (E.getKind()) {
    case ELFX86RelocationKind::Branch32:
    case ELFX86RelocationKind::Branch32ToStub:
    case ELFX86RelocationKind::PCRel32:
    case ELFX86RelocationKind::PCRel32GOTLoad: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend() - FixupAddress;
      if (LLVM_LIKELY(x86_64::isInRangeForImmS32(Value)))
        *(little32_t *)FixupPtr = Value;
      else
        return makeTargetOutOfRangeError(G, B, E);
      break;
    }
    case ELFX86RelocationKind::PCRel64: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend() - FixupAddress;
      *(little64_t *)FixupPtr = Value;
      break;
    }
    case ELFX86RelocationKind::Pointer64: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend();
      *(ulittle64_t *)FixupPtr = Value;
      break;
    }
    case ELFX86RelocationKind::Delta32: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend() - FixupAddress;
      if (LLVM_LIKELY(x86_64::isInRangeForImmS32(Value)))
        *(little32_t *)FixupPtr = Value;
      else
        return makeTargetOutOfRangeError(G, B, E);
      break;
    }
    case ELFX86RelocationKind::Delta64: {
      int64_t Value = E.getTarget().getAddress() + E.getAddend() - FixupAddress;
      *(little64_t *)FixupPtr = Value;
      break;
    }
    case ELFX86RelocationKind::NegDelta32: {
      int64_t Value = FixupAddress - E.getTarget().getAddress() + E.getAddend();
      if (LLVM_LIKELY(x86_64::isInRangeForImmS32(Value)))
        *(little32_t *)FixupPtr = Value;
      else
        return makeTargetOutOfRangeError(G, B, E);
      break;
    }
    case ELFX86RelocationKind::NegDelta64: {
      int64_t Value = FixupAddress - E.getTarget().getAddress() + E.getAddend();
      *(little64_t *)FixupPtr = Value;
      break;
    }
    case ELFX86RelocationKind::GOT64:
    case ELFX86RelocationKind::GOTOFF64: {
      // GOT64: Offset of GOT entry within GOT.
      // GOTOFF64: Offset from GOT base to target.
      // The expressions are the same in both cases, but in the GOT64 case the
      // edge will have been fixed to point at the GOT entry, and in the
      // GOTOFF64 case it will still point at the original target.
      assert(GOTSymbol && "No GOT section symbol");
      int64_t Value =
          E.getTarget().getAddress() - GOTSymbol->getAddress() + E.getAddend();
      *(little64_t *)FixupPtr = Value;
      break;
    }
    default:
      LLVM_DEBUG({
        dbgs() << "Bad edge: " << getELFX86RelocationKindName(E.getKind())
               << "\n";
      });
      llvm_unreachable("Unsupported relocation");
    }
    return Error::success();
  }
};

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject_x86_64(MemoryBufferRef ObjectBuffer) {
  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << ObjectBuffer.getBufferIdentifier() << "...\n";
  });

  auto ELFObj = object::ObjectFile::createELFObjectFile(ObjectBuffer);
  if (!ELFObj)
    return ELFObj.takeError();

  auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF64LE>>(**ELFObj);
  return ELFLinkGraphBuilder_x86_64((*ELFObj)->getFileName(),
                                    ELFObjFile.getELFFile())
      .buildGraph();
}

static SectionRangeSymbolDesc
identifyELFSectionStartAndEndSymbols(LinkGraph &G, Symbol &Sym) {
  constexpr StringRef StartSymbolPrefix = "__start";
  constexpr StringRef EndSymbolPrefix = "__end";

  auto SymName = Sym.getName();
  if (SymName.startswith(StartSymbolPrefix)) {
    if (auto *Sec =
            G.findSectionByName(SymName.drop_front(StartSymbolPrefix.size())))
      return {*Sec, true};
  } else if (SymName.startswith(EndSymbolPrefix)) {
    if (auto *Sec =
            G.findSectionByName(SymName.drop_front(EndSymbolPrefix.size())))
      return {*Sec, false};
  }
  return {};
}

void link_ELF_x86_64(std::unique_ptr<LinkGraph> G,
                     std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;

  if (Ctx->shouldAddDefaultTargetPasses(G->getTargetTriple())) {

    Config.PrePrunePasses.push_back(EHFrameSplitter(".eh_frame"));
    Config.PrePrunePasses.push_back(EHFrameEdgeFixer(
        ".eh_frame", G->getPointerSize(), Delta64, Delta32, NegDelta32));
    Config.PrePrunePasses.push_back(EHFrameNullTerminator(".eh_frame"));

    // Construct a JITLinker and run the link function.
    // Add a mark-live pass.
    if (auto MarkLive = Ctx->getMarkLivePass(G->getTargetTriple()))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);

    // Add an in-place GOT/Stubs pass.
    Config.PostPrunePasses.push_back(
        PerGraphGOTAndPLTStubsBuilder_ELF_x86_64::asPass);

    // Resolve any external section start / end symbols.
    Config.PostAllocationPasses.push_back(
        createDefineExternalSectionStartAndEndSymbolsPass(
            identifyELFSectionStartAndEndSymbols));

    // Add GOT/Stubs optimizer pass.
    Config.PreFixupPasses.push_back(optimizeELF_x86_64_GOTAndStubs);
  }

  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));

  ELFJITLinker_x86_64::link(std::move(Ctx), std::move(G), std::move(Config));
}
const char *getELFX86RelocationKindName(Edge::Kind R) {
  switch (R) {
  case Branch32:
    return "Branch32";
  case Branch32ToStub:
    return "Branch32ToStub";
  case Pointer32:
    return "Pointer32";
  case Pointer64:
    return "Pointer64";
  case Pointer64Anon:
    return "Pointer64Anon";
  case PCRel32:
    return "PCRel32";
  case PCRel32Minus1:
    return "PCRel32Minus1";
  case PCRel32Minus2:
    return "PCRel32Minus2";
  case PCRel32Minus4:
    return "PCRel32Minus4";
  case PCRel32Anon:
    return "PCRel32Anon";
  case PCRel32Minus1Anon:
    return "PCRel32Minus1Anon";
  case PCRel32Minus2Anon:
    return "PCRel32Minus2Anon";
  case PCRel32Minus4Anon:
    return "PCRel32Minus4Anon";
  case PCRel32GOTLoad:
    return "PCRel32GOTLoad";
  case PCRel32GOT:
    return "PCRel32GOT";
  case PCRel32TLV:
    return "PCRel32TLV";
  case Delta32:
    return "Delta32";
  case Delta64:
    return "Delta64";
  case NegDelta32:
    return "NegDelta32";
  case NegDelta64:
    return "NegDelta64";
  }
  return getGenericEdgeKindName(static_cast<Edge::Kind>(R));
}
} // end namespace jitlink
} // end namespace llvm
