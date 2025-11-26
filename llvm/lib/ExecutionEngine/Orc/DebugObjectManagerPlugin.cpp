//===------- DebugObjectManagerPlugin.cpp - JITLink debug objects ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FIXME: Update Plugin to poke the debug object into a new JITLink section,
//        rather than creating a new allocation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/ELF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::object;

namespace {
StringRef DebugObjectSecttionName = "jitlink$debug_object";

Section *getDebugELFObjectSection(LinkGraph &LG) {
  return LG.findSectionByName(DebugObjectSecttionName);
}

}

namespace llvm {
namespace orc {

static const std::set<StringRef> DwarfSectionNames = {
#define HANDLE_DWARF_SECTION(ENUM_NAME, ELF_NAME, CMDLINE_NAME, OPTION)        \
  ELF_NAME,
#include "llvm/BinaryFormat/Dwarf.def"
#undef HANDLE_DWARF_SECTION
};

static bool isDwarfSection(StringRef SectionName) {
  return DwarfSectionNames.count(SectionName) == 1;
}

template <typename ELFT>
static Error fixUp(StringRef DebugObjContent, LinkGraph &LG) {
  using SectionHeader = typename ELFT::Shdr;

  Expected<ELFFile<ELFT>> ObjRef = ELFFile<ELFT>::create(DebugObjContent);
  if (!ObjRef)
    return ObjRef.takeError();

  Expected<ArrayRef<SectionHeader>> Sections = ObjRef->sections();
  if (!Sections)
    return Sections.takeError();

  for (const SectionHeader &Header : *Sections) {
    Expected<StringRef> Name = ObjRef->getSectionName(Header);
    if (!Name)
      return Name.takeError();
    if (Name->empty())
      continue;

    // Only record text and data sections (i.e. no bss, comments, rel, etc.)
    if (Header.sh_type != ELF::SHT_PROGBITS &&
        Header.sh_type != ELF::SHT_X86_64_UNWIND)
      continue;
    if (!(Header.sh_flags & ELF::SHF_ALLOC))
      continue;

    // Slide action section content.
    if (auto *GraphSec = LG.findSectionByName(*Name))
      const_cast<SectionHeader&>(Header).sh_addr =
        static_cast<typename ELFT::uint>(SectionRange(*GraphSec).getStart().getValue());
  }

  return Error::success();
}

static Error fixUpDebugObject(LinkGraph &LG) {
  auto *DebugObjSec = getDebugELFObjectSection(LG);
  if (!DebugObjSec)
    return Error::success();

  assert(DebugObjSec && "No ELF debug object section?");
  assert(DebugObjSec->blocks_size() == 1 && "ELF debug object contains multiple blocks?");
  auto DebugObjMutableContent = (*DebugObjSec->blocks().begin())->getAlreadyMutableContent();
  StringRef DebugObjContent(DebugObjMutableContent.data(),
			    DebugObjMutableContent.size());

  unsigned char Class, Endian;
  std::tie(Class, Endian) = getElfArchType(DebugObjContent);
  if (Class == ELF::ELFCLASS32) {
    if (Endian == ELF::ELFDATA2LSB)
      return fixUp<ELF32LE>(DebugObjContent, LG);
    else if (Endian == ELF::ELFDATA2MSB)
      return fixUp<ELF32BE>(DebugObjContent, LG);
  } else if (Class == ELF::ELFCLASS64) {
    if (Endian == ELF::ELFDATA2LSB)
      return fixUp<ELF64LE>(DebugObjContent, LG);
    else if (Endian == ELF::ELFDATA2MSB)
      return fixUp<ELF64BE>(DebugObjContent, LG);
  }

  // TDOO: Mark this path llvm_unreachable once we're checking the original object
  //       type below in the pre-allocation pass.

  return Error::success();
}                                                                 
// TODO: create from ES, check how to do this without using deprecated ES constructor
DebugObjectManagerPlugin::DebugObjectManagerPlugin(ExecutionSession &ES) {

  SymbolAddrs SAs;
    if (auto Err = ES.getBootstrapSymbols(
            {{SAs.Instance, rt::SimpleExecutorDylibManagerInstanceName},
            {SAs.Open, rt::SimpleExecutorDylibManagerOpenWrapperName},
            {SAs.Lookup, rt::SimpleExecutorDylibManagerLookupWrapperName}}))
      return std::move(Err);
    ExecutorProcessControl EPC = ES.getExecutorProcessControl();
    return EPCGenericDylibManager(EPC, std::move(SAs));
}

DebugObjectManagerPlugin::DebugObjectManagerPlugin(
    ExecutorSymbolDef RegisterDebugObject,
    ExecutorSymbolDef DeregisterDebugObject,
    SymbolAddrs SAs,
    bool RequireDebugSections, bool AutoRegisterCode)
  : RegisterDebugObject(RegisterDebugObject),
    DeregisterDebugObject(DeregisterDebugObject),
    RequireDebugSections(RequireDebugSections),
    AutoRegisterCode(AutoRegisterCode) {}
  
DebugObjectManagerPlugin::~DebugObjectManagerPlugin() = default;

void DebugObjectManagerPlugin::modifyPassConfig(MaterializationResponsibility &MR,
  jitlink::LinkGraph &LG,
  jitlink::PassConfiguration &PassConfig) {
  
  PassConfig.PrePrunePasses.push_back([](LinkGraph &LG) -> Error {
    bool HasDebugSections = false;
    for (auto &Sec : LG.sections()) {
      if (isDwarfSection(Sec.getName())) {
	HasDebugSections = true;
	break;
      }
    }

    if (!HasDebugSections)
      return Error::success();

    if (auto *OriginalObjectSection = getOriginalELFObjectSection(LG)) {
      // TODO: Check header of original object to ensure that it's a supported
      //       type.
      assert(OriginalObjectSection->blocks_size() == 1 &&
	     "Original object file sections should only contain a single block");
      auto &DebugSec = LG.createSection(DebugObjectSecttionName, MemProt::Read);
      LG.createContentBlock(DebugSec,
			    (*OriginalObjectSection->blocks().begin())->getContent(),
			    orc::ExecutorAddr(), 8, 0);
    }
    return Error::success();
  });
  
  PassConfig.PostAllocationPasses.push_back(fixUpDebugObject);
}

} // namespace orc
} // namespace llvm
