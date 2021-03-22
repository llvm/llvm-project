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

#include "BasicGOTAndStubsBuilder.h"
#include "EHFrameSupportImpl.h"
#include "JITLinkGeneric.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::ELF_x86_64_Edges;

namespace {

class ELF_x86_64_GOTAndStubsBuilder
    : public BasicGOTAndStubsBuilder<ELF_x86_64_GOTAndStubsBuilder> {
public:
  static const uint8_t NullGOTEntryContent[8];
  static const uint8_t StubContent[6];

  ELF_x86_64_GOTAndStubsBuilder(LinkGraph &G)
      : BasicGOTAndStubsBuilder<ELF_x86_64_GOTAndStubsBuilder>(G) {}

  bool isGOTEdgeToFix(Edge &E) const {
    return E.getKind() == PCRel32GOT || E.getKind() == PCRel32GOTLoad;
  }

  Symbol &createGOTEntry(Symbol &Target) {
    auto &GOTEntryBlock = G.createContentBlock(
        getGOTSection(), getGOTEntryBlockContent(), 0, 8, 0);
    GOTEntryBlock.addEdge(Pointer64, 0, Target, 0);
    return G.addAnonymousSymbol(GOTEntryBlock, 0, 8, false, false);
  }

  void fixGOTEdge(Edge &E, Symbol &GOTEntry) {
    assert((E.getKind() == PCRel32GOT || E.getKind() == PCRel32GOTLoad) &&
           "Not a GOT edge?");
    // If this is a PCRel32GOT then change it to an ordinary PCRel32. If it is
    // a PCRel32GOTLoad then leave it as-is for now. We will use the kind to
    // check for GOT optimization opportunities in the
    // optimizeMachO_x86_64_GOTAndStubs pass below.
    if (E.getKind() == PCRel32GOT)
      E.setKind(PCRel32);

    E.setTarget(GOTEntry);
    // Leave the edge addend as-is.
  }

  bool isExternalBranchEdge(Edge &E) {
    return E.getKind() == Branch32 && !E.getTarget().isDefined();
  }

  Symbol &createStub(Symbol &Target) {
    auto &StubContentBlock =
        G.createContentBlock(getStubsSection(), getStubBlockContent(), 0, 1, 0);
    // Re-use GOT entries for stub targets.
    auto &GOTEntrySymbol = getGOTEntrySymbol(Target);
    StubContentBlock.addEdge(PCRel32, 2, GOTEntrySymbol, -4);
    return G.addAnonymousSymbol(StubContentBlock, 0, 6, true, false);
  }

  void fixExternalBranchEdge(Edge &E, Symbol &Stub) {
    assert(E.getKind() == Branch32 && "Not a Branch32 edge?");

    // Set the edge kind to Branch32ToStub. We will use this to check for stub
    // optimization opportunities in the optimize ELF_x86_64_GOTAndStubs pass
    // below.
    E.setKind(Branch32ToStub);
    E.setTarget(Stub);
  }

private:
  Section &getGOTSection() {
    if (!GOTSection)
      GOTSection = &G.createSection("$__GOT", sys::Memory::MF_READ);
    return *GOTSection;
  }

  Section &getStubsSection() {
    if (!StubsSection) {
      auto StubsProt = static_cast<sys::Memory::ProtectionFlags>(
          sys::Memory::MF_READ | sys::Memory::MF_EXEC);
      StubsSection = &G.createSection("$__STUBS", StubsProt);
    }
    return *StubsSection;
  }

  StringRef getGOTEntryBlockContent() {
    return StringRef(reinterpret_cast<const char *>(NullGOTEntryContent),
                     sizeof(NullGOTEntryContent));
  }

  StringRef getStubBlockContent() {
    return StringRef(reinterpret_cast<const char *>(StubContent),
                     sizeof(StubContent));
  }

  Section *GOTSection = nullptr;
  Section *StubsSection = nullptr;
};

const char *const DwarfSectionNames[] = {
#define HANDLE_DWARF_SECTION(ENUM_NAME, ELF_NAME, CMDLINE_NAME, OPTION)        \
  ELF_NAME,
#include "llvm/BinaryFormat/Dwarf.def"
#undef HANDLE_DWARF_SECTION
};

} // namespace

const uint8_t ELF_x86_64_GOTAndStubsBuilder::NullGOTEntryContent[8] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const uint8_t ELF_x86_64_GOTAndStubsBuilder::StubContent[6] = {
    0xFF, 0x25, 0x00, 0x00, 0x00, 0x00};

static const char *CommonSectionName = "__common";
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
        assert(StubBlock.getSize() ==
                   sizeof(ELF_x86_64_GOTAndStubsBuilder::StubContent) &&
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

static bool isDwarfSection(StringRef SectionName) {
  return llvm::is_contained(DwarfSectionNames, SectionName);
}

namespace llvm {
namespace jitlink {

// This should become a template as the ELFFile is so a lot of this could become
// generic
class ELFLinkGraphBuilder_x86_64 {

private:
  Section *CommonSection = nullptr;
  // TODO hack to get this working
  // Find a better way
  using SymbolTable = object::ELFFile<object::ELF64LE>::Elf_Shdr;
  // For now we just assume
  using SymbolMap = std::map<int32_t, Symbol *>;
  SymbolMap JITSymbolTable;

  Section &getCommonSection() {
    if (!CommonSection) {
      auto Prot = static_cast<sys::Memory::ProtectionFlags>(
          sys::Memory::MF_READ | sys::Memory::MF_WRITE);
      CommonSection = &G->createSection(CommonSectionName, Prot);
    }
    return *CommonSection;
  }

  static Expected<ELF_x86_64_Edges::ELFX86RelocationKind>
  getRelocationKind(const uint32_t Type) {
    switch (Type) {
    case ELF::R_X86_64_PC32:
      return ELF_x86_64_Edges::ELFX86RelocationKind::PCRel32;
    case ELF::R_X86_64_PC64:
      return ELF_x86_64_Edges::ELFX86RelocationKind::Delta64;
    case ELF::R_X86_64_64:
      return ELF_x86_64_Edges::ELFX86RelocationKind::Pointer64;
    case ELF::R_X86_64_GOTPCREL:
    case ELF::R_X86_64_GOTPCRELX:
    case ELF::R_X86_64_REX_GOTPCRELX:
      return ELF_x86_64_Edges::ELFX86RelocationKind::PCRel32GOTLoad;
    case ELF::R_X86_64_PLT32:
      return ELF_x86_64_Edges::ELFX86RelocationKind::Branch32;
    }
    return make_error<JITLinkError>("Unsupported x86-64 relocation:" +
                                    formatv("{0:d}", Type));
  }

  std::unique_ptr<LinkGraph> G;
  // This could be a template
  const object::ELFFile<object::ELF64LE> &Obj;
  object::ELFFile<object::ELF64LE>::Elf_Shdr_Range sections;
  SymbolTable SymTab;

  bool isRelocatable() { return Obj.getHeader().e_type == llvm::ELF::ET_REL; }

  support::endianness
  getEndianness(const object::ELFFile<object::ELF64LE> &Obj) {
    return Obj.isLE() ? support::little : support::big;
  }

  // This could also just become part of a template
  unsigned getPointerSize(const object::ELFFile<object::ELF64LE> &Obj) {
    return Obj.getHeader().getFileClass() == ELF::ELFCLASS64 ? 8 : 4;
  }

  // We don't technically need this right now
  // But for now going to keep it as it helps me to debug things

  Error createNormalizedSymbols() {
    LLVM_DEBUG(dbgs() << "Creating normalized symbols...\n");

    for (auto SecRef : sections) {
      if (SecRef.sh_type != ELF::SHT_SYMTAB &&
          SecRef.sh_type != ELF::SHT_DYNSYM)
        continue;

      auto Symbols = Obj.symbols(&SecRef);
      // TODO: Currently I use this function to test things 
      // I also want to leave it to see if its common between MACH and elf
      // so for now I just want to continue even if there is an error
      if (errorToBool(Symbols.takeError()))
        continue;

      auto StrTabSec = Obj.getSection(SecRef.sh_link);
      if (!StrTabSec)
        return StrTabSec.takeError();
      auto StringTable = Obj.getStringTable(**StrTabSec);
      if (!StringTable)
        return StringTable.takeError();

      for (auto SymRef : *Symbols) {
        Optional<StringRef> Name;

        if (auto NameOrErr = SymRef.getName(*StringTable))
          Name = *NameOrErr;
        else
          return NameOrErr.takeError();

        LLVM_DEBUG({
          dbgs() << "  value = " << formatv("{0:x16}", SymRef.getValue())
                 << ", type = " << formatv("{0:x2}", SymRef.getType())
                 << ", binding = " << formatv("{0:x2}", SymRef.getBinding())
                 << ", size = "
                 << formatv("{0:x16}", static_cast<uint64_t>(SymRef.st_size))
                 << ", info = " << formatv("{0:x2}", SymRef.st_info)
                 << " :" << (Name ? *Name : "<anonymous symbol>") << "\n";
        });
      }
    }
    return Error::success();
  }

  Error createNormalizedSections() {
    LLVM_DEBUG(dbgs() << "Creating normalized sections...\n");
    for (auto &SecRef : sections) {
      auto Name = Obj.getSectionName(SecRef);
      if (!Name)
        return Name.takeError();

      // Skip Dwarf sections.
      if (isDwarfSection(*Name)) {
        LLVM_DEBUG({
          dbgs() << *Name
                 << " is a debug section: No graph section will be created.\n";
        });
        continue;
      }

      sys::Memory::ProtectionFlags Prot;
      if (SecRef.sh_flags & ELF::SHF_EXECINSTR) {
        Prot = static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                         sys::Memory::MF_EXEC);
      } else {
        Prot = static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                         sys::Memory::MF_WRITE);
      }
      uint64_t Address = SecRef.sh_addr;
      uint64_t Size = SecRef.sh_size;
      uint64_t Flags = SecRef.sh_flags;
      uint64_t Alignment = SecRef.sh_addralign;
      const char *Data = nullptr;
      // for now we just use this to skip the "undefined" section, probably need
      // to revist
      if (Size == 0)
        continue;

      // FIXME: Use flags.
      (void)Flags;

      LLVM_DEBUG({
        dbgs() << "  " << *Name << ": " << formatv("{0:x16}", Address) << " -- "
               << formatv("{0:x16}", Address + Size) << ", align: " << Alignment
               << " Flags: " << formatv("{0:x}", Flags) << "\n";
      });

      if (SecRef.sh_type != ELF::SHT_NOBITS) {
        // .sections() already checks that the data is not beyond the end of
        // file
        auto contents = Obj.getSectionContentsAsArray<char>(SecRef);
        if (!contents)
          return contents.takeError();

        Data = contents->data();
        // TODO protection flags.
        // for now everything is
        auto &section = G->createSection(*Name, Prot);
        // Do this here because we have it, but move it into graphify later
        G->createContentBlock(section, StringRef(Data, Size), Address,
                              Alignment, 0);
        if (SecRef.sh_type == ELF::SHT_SYMTAB)
          // TODO: Dynamic?
          SymTab = SecRef;
      } else {
        auto &Section = G->createSection(*Name, Prot);
        G->createZeroFillBlock(Section, Size, Address, Alignment, 0);
      }
    }

    return Error::success();
  }

  Error addRelocations() {
    LLVM_DEBUG(dbgs() << "Adding relocations\n");
    // TODO a partern is forming of iterate some sections but only give me
    // ones I am interested, i should abstract that concept some where
    for (auto &SecRef : sections) {
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
        auto Symbol = Obj.getRelocationSymbol(Rela, &SymTab);
        if (!Symbol)
          return Symbol.takeError();

        auto BlockToFix = *(JITSection->blocks().begin());
        auto *TargetSymbol = JITSymbolTable[SymbolIndex];

        if (!TargetSymbol) {
          return make_error<llvm::StringError>(
              "Could not find symbol at given index, did you add it to "
              "JITSymbolTable? index: " + std::to_string(SymbolIndex)
              + ", shndx: " + std::to_string((*Symbol)->st_shndx) +
                  " Size of table: " + std::to_string(JITSymbolTable.size()),
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

  Error graphifyRegularSymbols() {

    // TODO: ELF supports beyond SHN_LORESERVE,
    // need to perf test how a vector vs map handles those cases

    std::vector<std::vector<object::ELFFile<object::ELF64LE>::Elf_Shdr_Range *>>
        SecIndexToSymbols;

    LLVM_DEBUG(dbgs() << "Creating graph symbols...\n");

    for (auto SecRef : sections) {

      if (SecRef.sh_type != ELF::SHT_SYMTAB &&
          SecRef.sh_type != ELF::SHT_DYNSYM)
        continue;
      auto Symbols = Obj.symbols(&SecRef);
      if (!Symbols)
        return Symbols.takeError();

      auto StrTabSec = Obj.getSection(SecRef.sh_link);
      if (!StrTabSec)
        return StrTabSec.takeError();
      auto StringTable = Obj.getStringTable(**StrTabSec);
      if (!StringTable)
        return StringTable.takeError();
      auto Name = Obj.getSectionName(SecRef);
      if (!Name)
        return Name.takeError();

      LLVM_DEBUG(dbgs() << "Processing symbol section " << *Name << ":\n");

      auto Section = G->findSectionByName(*Name);
      if (!Section)
        return make_error<llvm::StringError>("Could not find a section " +
                                             *Name,
                                             llvm::inconvertibleErrorCode());
      // we only have one for now
      auto blocks = Section->blocks();
      if (blocks.empty())
        return make_error<llvm::StringError>("Section has no block",
                                             llvm::inconvertibleErrorCode());
      int SymbolIndex = -1;
      for (auto SymRef : *Symbols) {
        ++SymbolIndex;
        auto Type = SymRef.getType();

        if (Type == ELF::STT_FILE || SymbolIndex == 0)
          continue;
        // these should do it for now
        // if(Type != ELF::STT_NOTYPE &&
        //   Type != ELF::STT_OBJECT &&
        //   Type != ELF::STT_FUNC    &&
        //   Type != ELF::STT_SECTION &&
        //   Type != ELF::STT_COMMON) {
        //     continue;
        //   }
        auto Name = SymRef.getName(*StringTable);
        // I am not sure on If this is going to hold as an invariant. Revisit.
        if (!Name)
          return Name.takeError();

        if (SymRef.isCommon()) {
          // Symbols in SHN_COMMON refer to uninitialized data. The st_value
          // field holds alignment constraints.
          Symbol &S =
              G->addCommonSymbol(*Name, Scope::Default, getCommonSection(), 0,
                                 SymRef.st_size, SymRef.getValue(), false);
          JITSymbolTable[SymbolIndex] = &S;
          continue;
        }

        // Map Visibility and Binding to Scope and Linkage:
        Linkage L = Linkage::Strong;
        Scope S = Scope::Default;

        switch (SymRef.getBinding()) {
        case ELF::STB_LOCAL:
          S = Scope::Local;
          break;
        case ELF::STB_GLOBAL:
          // Nothing to do here.
          break;
        case ELF::STB_WEAK:
          L = Linkage::Weak;
          break;
        default:
          return make_error<StringError>("Unrecognized symbol binding for " +
                                             *Name,
                                         inconvertibleErrorCode());
        }

        switch (SymRef.getVisibility()) {
        case ELF::STV_DEFAULT:
        case ELF::STV_PROTECTED:
          // FIXME: Make STV_DEFAULT symbols pre-emptible? This probably needs
          // Orc support.
          // Otherwise nothing to do here.
          break;
        case ELF::STV_HIDDEN:
          // Default scope -> Hidden scope. No effect on local scope.
          if (S == Scope::Default)
            S = Scope::Hidden;
          break;
        case ELF::STV_INTERNAL:
          return make_error<StringError>("Unrecognized symbol visibility for " +
                                             *Name,
                                         inconvertibleErrorCode());
        }

        if (SymRef.isDefined() &&
            (Type == ELF::STT_NOTYPE || Type == ELF::STT_FUNC ||
             Type == ELF::STT_OBJECT || Type == ELF::STT_SECTION)) {

          auto DefinedSection = Obj.getSection(SymRef.st_shndx);
          if (!DefinedSection)
            return DefinedSection.takeError();
          auto sectName = Obj.getSectionName(**DefinedSection);
          if (!sectName)
            return Name.takeError();

          // Skip debug section symbols.
          if (isDwarfSection(*sectName))
            continue;

          auto JitSection = G->findSectionByName(*sectName);
          if (!JitSection)
            return make_error<llvm::StringError>(
                "Could not find the JitSection " + *sectName,
                llvm::inconvertibleErrorCode());
          auto bs = JitSection->blocks();
          if (bs.empty())
            return make_error<llvm::StringError>(
                "Section has no block", llvm::inconvertibleErrorCode());

          auto *B = *bs.begin();
          LLVM_DEBUG({ dbgs() << "  " << *Name << " at index " << SymbolIndex << "\n"; });
          if (SymRef.getType() == ELF::STT_SECTION)
            *Name = *sectName;
          auto &Sym = G->addDefinedSymbol(
              *B, SymRef.getValue(), *Name, SymRef.st_size, L, S,
              SymRef.getType() == ELF::STT_FUNC, false);
          JITSymbolTable[SymbolIndex] = &Sym;
        } else if (SymRef.isUndefined() && SymRef.isExternal()) {
          auto &Sym = G->addExternalSymbol(*Name, SymRef.st_size, L);
          JITSymbolTable[SymbolIndex] = &Sym;
        } else
          LLVM_DEBUG({
              dbgs()
                << "Not creating graph symbol for normalized symbol at index "
                << SymbolIndex << ", \"" << *Name << "\"\n";
            });

        // TODO: The following has to be implmented.
        // leaving commented out to save time for future patchs
        /*
          G->addAbsoluteSymbol(*Name, SymRef.getValue(), SymRef.st_size,
          Linkage::Strong, Scope::Default, false);
        */
      }
    }
    return Error::success();
  }

public:
  ELFLinkGraphBuilder_x86_64(StringRef FileName,
                             const object::ELFFile<object::ELF64LE> &Obj)
      : G(std::make_unique<LinkGraph>(
            FileName.str(), Triple("x86_64-unknown-linux"), getPointerSize(Obj),
            getEndianness(Obj), getELFX86RelocationKindName)),
        Obj(Obj) {}

  Expected<std::unique_ptr<LinkGraph>> buildGraph() {
    // Sanity check: we only operate on relocatable objects.
    if (!isRelocatable())
      return make_error<JITLinkError>("Object is not a relocatable ELF");

    auto Secs = Obj.sections();

    if (!Secs) {
      return Secs.takeError();
    }
    sections = *Secs;

    if (auto Err = createNormalizedSections())
      return std::move(Err);

    if (auto Err = createNormalizedSymbols())
      return std::move(Err);

    if (auto Err = graphifyRegularSymbols())
      return std::move(Err);

    if (auto Err = addRelocations())
      return std::move(Err);

    return std::move(G);
  }
};

class ELFJITLinker_x86_64 : public JITLinker<ELFJITLinker_x86_64> {
  friend class JITLinker<ELFJITLinker_x86_64>;

public:
  ELFJITLinker_x86_64(std::unique_ptr<JITLinkContext> Ctx,
                      std::unique_ptr<LinkGraph> G,
                      PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(G), std::move(PassConfig)) {}

private:
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E,
                   char *BlockWorkingMem) const {
    using namespace ELF_x86_64_Edges;
    using namespace llvm::support;
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
    Config.PostPrunePasses.push_back([](LinkGraph &G) -> Error {
      ELF_x86_64_GOTAndStubsBuilder(G).run();
      return Error::success();
    });

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
