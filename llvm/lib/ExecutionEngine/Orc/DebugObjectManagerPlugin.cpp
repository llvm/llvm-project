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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkDylib.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

#define DEBUG_TYPE "orc"

using namespace llvm::jitlink;
using namespace llvm::object;

namespace llvm {
namespace orc {

enum DebugObjectFlags : int {
  // Request final target memory load-addresses for all sections.
  ReportFinalSectionLoadAddresses = 1 << 0,

  // We found sections with debug information when processing the input object.
  HasDebugSections = 1 << 1,
};

static const std::set<StringRef> DwarfSectionNames = {
#define HANDLE_DWARF_SECTION(ENUM_NAME, ELF_NAME, CMDLINE_NAME, OPTION)        \
  ELF_NAME,
#include "llvm/BinaryFormat/Dwarf.def"
#undef HANDLE_DWARF_SECTION
};

static bool isDwarfSection(StringRef SectionName) {
  return DwarfSectionNames.count(SectionName) == 1;
}

template <typename ELFT> Error fixUp(StringRef Buffer, LinkGraph &LG) {

  Error Err = Error::success();

  // TODO:replace debugObj
  // Expected<ELFFile<ELFT>> Buffer = ELFFile<ELFT>::create(LG.g);
  // if (!Buffer)
  //   return Buffer.takeError();

  Expected<ArrayRef<SectionHeader>> Sections = Buffer->sections();
  if (!Sections)
    return Sections.takeError();

  for (const SectionHeader &Header : *Sections) {
    Expected<StringRef> Name = Buffer->getSectionName(Header);
    if (!Name)
      return Name.takeError();
    if (Name->empty())
      continue;
    if (isDwarfSection(*Name))
      DebugObj->setFlags(HasDebugSections);

    // Only record text and data sections (i.e. no bss, comments, rel, etc.)
    if (Header.sh_type != ELF::SHT_PROGBITS &&
        Header.sh_type != ELF::SHT_X86_64_UNWIND)
      continue;
    if (!(Header.sh_flags & ELF::SHF_ALLOC))
      continue;

    if (auto *GraphSec = LG.findSectionByName(*Name))
      Header->sh_addr =
        static_cast<typename ELFT::uint>(SectionRange(*GraphSec).getStart().getValue());

  return std::move(DebugObj);
}


DebugObjectManagerPlugin::DebugObjectManagerPlugin(
    ExecutionSession &ES, std::unique_ptr<DebugObjectRegistrar> Target,
    bool RequireDebugSections, bool AutoRegisterCode)
    : ES(ES), Target(std::move(Target)),
      RequireDebugSections(RequireDebugSections),
      AutoRegisterCode(AutoRegisterCode) {}

DebugObjectManagerPlugin::DebugObjectManagerPlugin(
    ExecutionSession &ES, std::unique_ptr<DebugObjectRegistrar> Target)
    : DebugObjectManagerPlugin(ES, std::move(Target), true, true) {}

DebugObjectManagerPlugin::~DebugObjectManagerPlugin() = default;

void fixUpDebugObject(LinkGraph &LG) {
  auto *DebugObjSec = LG.getOriginalObjectContentSection();
  assert(DebugObjSec && "No ELF debug object section?");
  assert(DebugObjSec.blocks_size() == 1 && "ELF debug object contains multiple blocks?");
  auto DebugObjContent = (*DebugObjSec.blocks_begin())->getAlreadyMutableContent();

  // StringRef DebugObj(DebugObjContent.data(), DebugObjContent.size());

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
  // Unsupported combo. Remove the debug object section.
  LG.removeSection(*DebugObjSec);
  LLVM_DEBUG({
    dbgs() << "Can't emit debug object for " << LG.getName()
           << ": Unsupported ELF class / endianness.\n";
  });
  return Error::success();
}                                                                 

void DebugObjectManagerPlugin::modifyPassConfig(MaterializationResponsibility &MR,
  jitlink::LinkGraph &LG,
  jitlink::PassConfiguration &PassConfig) {
  // Not all link artifacts have associated debug objects.
  // std::lock_guard<std::mutex> Lock(PendingObjsLock);
  auto It = PendingObjs.find(&MR);
  if (It == PendingObjs.end())
    return;

  PassConfig.PrePrunePasses.push_back([](LinkGraph &LG) -> Error {
    
    // Copy existing object content into the new debug object section
    auto DebugObjContent = LG.getOriginalObjectContentSection();
    // Create new debug section in LinkGraph
    
    // Memory protection for reading graph
    orc::MemProt Prot = MemProt::Read;
    // Create debug section
    LG.createSection(DebugObjContent, Prot);

    return Error::success();
  });
  
  if (DebugObjContent.hasFlags(ReportFinalSectionLoadAddresses)) {
  // patch up the addresses in the debug object
    PassConfig.PostAllocationPasses.push_back(
        [&DebugObjContent](LinkGraph &LG) -> Error {
          for (const Section &GraphSection : LG.sections())
            DebugObjContent.reportSectionTargetMemoryRange(GraphSection.getName(),
                                                    SectionRange(GraphSection));
          fixUpDebugObject(LG)
          return Error::success();
        });
  }
}

Error DebugObjectManagerPlugin::notifyFailed(
    MaterializationResponsibility &MR) {
  // std::lock_guard<std::mutex> Lock(PendingObjsLock);
  PendingObjs.erase(&MR);
  return Error::success();
}

Error DebugObjectManagerPlugin::notifyRemovingResources(JITDylib &JD,
                                                        ResourceKey Key) {
  // Removing the resource for a pending object fails materialization, so they
  // get cleaned up in the notifyFailed() handler.
  // std::lock_guard<std::mutex> Lock(RegisteredObjsLock);
  RegisteredObjs.erase(Key);

  // TODO: Implement unregister notifications.
  return Error::success();
}

} // namespace orc
} // namespace llvm