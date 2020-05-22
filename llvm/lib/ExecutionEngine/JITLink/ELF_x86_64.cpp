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
#include "JITLinkGeneric.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Object/ELFObjectFile.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;

static const char *CommonSectionName = "__common";

namespace llvm {
namespace jitlink {
// This should become a template as the ELFFile is so a lot of this could become
// generic
class ELFLinkGraphBuilder_x86_64 {

private:
  Section *CommonSection = nullptr;
  Section &getCommonSection() {
    if (!CommonSection) {
      auto Prot = static_cast<sys::Memory::ProtectionFlags>(
          sys::Memory::MF_READ | sys::Memory::MF_WRITE);
      CommonSection = &G->createSection(CommonSectionName, Prot);
    }
    return *CommonSection;
  }

  std::unique_ptr<LinkGraph> G;
  // This could be a template
  const object::ELFFile<object::ELF64LE> &Obj;
  object::ELFFile<object::ELF64LE>::Elf_Shdr_Range sections;

  bool isRelocatable() { return Obj.getHeader()->e_type == llvm::ELF::ET_REL; }

  support::endianness
  getEndianness(const object::ELFFile<object::ELF64LE> &Obj) {
    return Obj.isLE() ? support::little : support::big;
  }

  // This could also just become part of a template
  unsigned getPointerSize(const object::ELFFile<object::ELF64LE> &Obj) {
    return Obj.getHeader()->getFileClass() == ELF::ELFCLASS64 ? 8 : 4;
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
      auto StringTable = Obj.getStringTable(*StrTabSec);
      if (!StringTable)
        return StringTable.takeError();

      for (auto SymRef : *Symbols) {
        Optional<StringRef> Name;
        unsigned char Binding;
        uint64_t Value;
        uint64_t Size = 0;

        // FIXME: Read size.
        (void)Size;

        if (auto NameOrErr = SymRef.getName(*StringTable)) {
          Name = *NameOrErr;
        } else {
          return NameOrErr.takeError();
        }
        Binding = SymRef.getBinding();
        Value = SymRef.getValue();
        LLVM_DEBUG({
          dbgs() << "  ";
          if (!Name)
            dbgs() << "<anonymous symbol>";
          else
            dbgs() << *Name;
          dbgs() << ": value = " << formatv("{0:x16}", Value)
                 << ", type = " << formatv("{0:x2}", SymRef.getType())
                 << ", binding = " << Binding
                 << ", size =" << Size;
          dbgs() << "\n";
        });
      }
    }
    return Error::success();
  }

  Error createNormalizedSections() {
    LLVM_DEBUG(dbgs() << "Creating normalized sections...\n");
    for (auto &SecRef : sections) {
      auto Name = Obj.getSectionName(&SecRef);
      if (!Name)
        return Name.takeError();
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
      // TODO: figure out what it is that has 0 size no name and address
      // 0000-0000
      if (Size == 0)
        continue;

      // FIXME: Use flags.
      (void)Flags;

      LLVM_DEBUG({
        dbgs() << "  " << *Name << ": " << formatv("{0:x16}", Address) << " -- "
               << formatv("{0:x16}", Address + Size) << ", align: " << Alignment
               << " Flags:" << Flags << "\n";
      });

      if (SecRef.sh_type != ELF::SHT_NOBITS) {
        // .sections() already checks that the data is not beyond the end of
        // file
        auto contents = Obj.getSectionContentsAsArray<char>(&SecRef);
        if (!contents)
          return contents.takeError();

        Data = contents->data();
        // TODO protection flags.
        // for now everything is
        auto &section = G->createSection(*Name, Prot);
        // Do this here because we have it, but move it into graphify later
        G->createContentBlock(section, StringRef(Data, Size), Address,
                              Alignment, 0);
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
      auto StringTable = Obj.getStringTable(*StrTabSec);
      if (!StringTable)
        return StringTable.takeError();
      auto Name = Obj.getSectionName(&SecRef);
      if (!Name)
        return Name.takeError();
      auto Section = G->findSectionByName(*Name);
      if (!Section)
        return make_error<llvm::StringError>("Could not find a section",
                                             llvm::inconvertibleErrorCode());
      // we only have one for now
      auto blocks = Section->blocks();
      if (blocks.empty())
        return make_error<llvm::StringError>("Section has no block",
                                             llvm::inconvertibleErrorCode());

      for (auto SymRef : *Symbols) {
        auto Type = SymRef.getType();
        if (Type == ELF::STT_NOTYPE || Type == ELF::STT_FILE)
          continue;
        // these should do it for now
        // if(Type != ELF::STT_NOTYPE &&
        //   Type != ELF::STT_OBJECT &&
        //   Type != ELF::STT_FUNC    &&
        //   Type != ELF::STT_SECTION &&
        //   Type != ELF::STT_COMMON) {
        //     continue;
        //   }
        std::pair<Linkage, Scope> bindings;
        auto Name = SymRef.getName(*StringTable);
        // I am not sure on If this is going to hold as an invariant. Revisit.
        if (!Name)
          return Name.takeError();
        // TODO: weak and hidden
        if (SymRef.isExternal()) {
          bindings = {Linkage::Strong, Scope::Default};
        } else {
          bindings = {Linkage::Strong, Scope::Local};
        }

        if (SymRef.isDefined() &&
            (Type == ELF::STT_FUNC || Type == ELF::STT_OBJECT)) {

          auto DefinedSection = Obj.getSection(SymRef.st_shndx);
          if (!DefinedSection)
            return DefinedSection.takeError();
          auto sectName = Obj.getSectionName(*DefinedSection);
          if (!sectName)
            return Name.takeError();

          auto JitSection = G->findSectionByName(*sectName);
          if (!JitSection)
            return make_error<llvm::StringError>(
                "Could not find a section", llvm::inconvertibleErrorCode());
          auto bs = JitSection->blocks();
          if (bs.empty())
            return make_error<llvm::StringError>(
                "Section has no block", llvm::inconvertibleErrorCode());

          auto B = *bs.begin();
          LLVM_DEBUG({ dbgs() << "  " << *Name << ": "; });

          G->addDefinedSymbol(*B, SymRef.getValue(), *Name, SymRef.st_size,
                              bindings.first, bindings.second,
                              SymRef.getType() == ELF::STT_FUNC, false);
        }
        //TODO: The following has to be implmented.
        // leaving commented out to save time for future patchs
        /*
          G->addAbsoluteSymbol(*Name, SymRef.getValue(), SymRef.st_size,
          Linkage::Strong, Scope::Default, false);

          if(SymRef.isCommon()) {
            G->addCommonSymbol(*Name, Scope::Default, getCommonSection(), 0, 0,
          SymRef.getValue(), false);
          }


          //G->addExternalSymbol(*Name, SymRef.st_size, Linkage::Strong);
  */
      }
    }
    return Error::success();
  }

public:
  ELFLinkGraphBuilder_x86_64(std::string filename,
                             const object::ELFFile<object::ELF64LE> &Obj)
      : G(std::make_unique<LinkGraph>(filename, getPointerSize(Obj),
                                      getEndianness(Obj))),
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

    return std::move(G);
  }
};

class ELFJITLinker_x86_64 : public JITLinker<ELFJITLinker_x86_64> {
  friend class JITLinker<ELFJITLinker_x86_64>;

public:
  ELFJITLinker_x86_64(std::unique_ptr<JITLinkContext> Ctx,
                      PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(PassConfig)) {}

private:
  StringRef getEdgeKindName(Edge::Kind R) const override {
    return getELFX86RelocationKindName(R);
  }

  Expected<std::unique_ptr<LinkGraph>>
  buildGraph(MemoryBufferRef ObjBuffer) override {
    auto ELFObj = object::ObjectFile::createELFObjectFile(ObjBuffer);
    if (!ELFObj)
      return ELFObj.takeError();

    auto &ELFObjFile = cast<object::ELFObjectFile<object::ELF64LE>>(**ELFObj);
    std::string fileName(ELFObj->get()->getFileName());
    return ELFLinkGraphBuilder_x86_64(std::move(fileName),
                                      *ELFObjFile.getELFFile())
        .buildGraph();
  }

  Error applyFixup(Block &B, const Edge &E, char *BlockWorkingMem) const {
    //TODO: add relocation handling
    return Error::success();
  }
};

void jitLink_ELF_x86_64(std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;
  Triple TT("x86_64-linux");
  // Construct a JITLinker and run the link function.
  // Add a mark-live pass.
  if (auto MarkLive = Ctx->getMarkLivePass(TT))
    Config.PrePrunePasses.push_back(std::move(MarkLive));
  else
    Config.PrePrunePasses.push_back(markAllSymbolsLive);

  ELFJITLinker_x86_64::link(std::move(Ctx), std::move(Config));
}

StringRef getELFX86RelocationKindName(Edge::Kind R) {
  switch (R) {
    //     case R_AMD64_NONE:
    //         return "None";
    //   case R_AMD64_PC32:
    //   case R_AMD64_GOT32:
    //   caseR_AMD64_PLT32,
    //   R_AMD64_COPY,
    //   R_AMD64_GLOB_DAT,
    //   R_AMD64_JUMP_SLOT,
    //   R_AMD64_RELATIVE,
    //   R_AMD64_GOTPCREL,
    //   R_AMD64_32,
    //   R_AMD64_32S,
    //   R_AMD64_16,
    //   R_AMD64_PC16,
    //   R_AMD64_8,
    //   R_AMD64_PC8,
    //   R_AMD64_PC64,
    //   R_AMD64_GOTOFF64,
    //   R_AMD64_GOTPC32,
    //   R_AMD64_SIZE32,
    //   R_AMD64_SIZE64
  default:
    return getGenericEdgeKindName(static_cast<Edge::Kind>(R));
  }
}
} // end namespace jitlink
} // end namespace llvm
