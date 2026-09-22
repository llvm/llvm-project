//===- COFFObjcopy.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjCopy/COFF/COFFObjcopy.h"
#include "COFFObject.h"
#include "COFFReader.h"
#include "COFFWriter.h"
#include "llvm/ObjCopy/COFF/COFFConfig.h"
#include "llvm/ObjCopy/CommonConfig.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/WindowsResource.h"
#include "llvm/Support/CRC.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/Path.h"
#include <cassert>

namespace llvm {
namespace objcopy {
namespace coff {

using namespace object;
using namespace COFF;

static bool isDebugSection(const Section &Sec) {
  return Sec.Name.starts_with(".debug");
}

static uint64_t getNextRVA(const Object &Obj) {
  if (Obj.getSections().empty())
    return 0;
  const Section &Last = Obj.getSections().back();
  return alignTo(Last.Header.VirtualAddress + Last.Header.VirtualSize,
                 Obj.IsPE ? Obj.PeHeader.SectionAlignment : 1);
}

static Expected<std::vector<uint8_t>>
createGnuDebugLinkSectionContents(StringRef File) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> LinkTargetOrErr =
      MemoryBuffer::getFile(File);
  if (!LinkTargetOrErr)
    return createFileError(File, LinkTargetOrErr.getError());
  auto LinkTarget = std::move(*LinkTargetOrErr);
  uint32_t CRC32 = llvm::crc32(arrayRefFromStringRef(LinkTarget->getBuffer()));

  StringRef FileName = sys::path::filename(File);
  size_t CRCPos = alignTo(FileName.size() + 1, 4);
  std::vector<uint8_t> Data(CRCPos + 4);
  memcpy(Data.data(), FileName.data(), FileName.size());
  support::endian::write32le(Data.data() + CRCPos, CRC32);
  return Data;
}

// Adds named section with given contents to the object.
static void addSection(Object &Obj, StringRef Name, ArrayRef<uint8_t> Contents,
                       uint32_t Characteristics) {
  bool NeedVA = Characteristics & (IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ |
                                   IMAGE_SCN_MEM_WRITE);

  Section Sec;
  Sec.setOwnedContents(Contents);
  Sec.Name = Name;
  Sec.Header.VirtualSize = NeedVA ? Sec.getContents().size() : 0u;
  Sec.Header.VirtualAddress = NeedVA ? getNextRVA(Obj) : 0u;
  Sec.Header.SizeOfRawData =
      NeedVA ? alignTo(Sec.Header.VirtualSize,
                       Obj.IsPE ? Obj.PeHeader.FileAlignment : 1)
             : Sec.getContents().size();
  // Sec.Header.PointerToRawData is filled in by the writer.
  Sec.Header.PointerToRelocations = 0;
  Sec.Header.PointerToLinenumbers = 0;
  // Sec.Header.NumberOfRelocations is filled in by the writer.
  Sec.Header.NumberOfLinenumbers = 0;
  Sec.Header.Characteristics = Characteristics;

  Obj.addSections(Sec);
}

static Error addGnuDebugLink(Object &Obj, StringRef DebugLinkFile) {
  Expected<std::vector<uint8_t>> Contents =
      createGnuDebugLinkSectionContents(DebugLinkFile);
  if (!Contents)
    return Contents.takeError();

  addSection(Obj, ".gnu_debuglink", *Contents,
             IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ |
                 IMAGE_SCN_MEM_DISCARDABLE);

  return Error::success();
}

static uint32_t flagsToCharacteristics(SectionFlag AllFlags, uint32_t OldChar) {
  // Need to preserve alignment flags.
  const uint32_t PreserveMask =
      IMAGE_SCN_ALIGN_1BYTES | IMAGE_SCN_ALIGN_2BYTES | IMAGE_SCN_ALIGN_4BYTES |
      IMAGE_SCN_ALIGN_8BYTES | IMAGE_SCN_ALIGN_16BYTES |
      IMAGE_SCN_ALIGN_32BYTES | IMAGE_SCN_ALIGN_64BYTES |
      IMAGE_SCN_ALIGN_128BYTES | IMAGE_SCN_ALIGN_256BYTES |
      IMAGE_SCN_ALIGN_512BYTES | IMAGE_SCN_ALIGN_1024BYTES |
      IMAGE_SCN_ALIGN_2048BYTES | IMAGE_SCN_ALIGN_4096BYTES |
      IMAGE_SCN_ALIGN_8192BYTES;

  // Setup new section characteristics based on the flags provided in command
  // line.
  uint32_t NewCharacteristics = (OldChar & PreserveMask) | IMAGE_SCN_MEM_READ;

  if ((AllFlags & SectionFlag::SecAlloc) && !(AllFlags & SectionFlag::SecLoad))
    NewCharacteristics |= IMAGE_SCN_CNT_UNINITIALIZED_DATA;
  if (AllFlags & SectionFlag::SecNoload)
    NewCharacteristics |= IMAGE_SCN_LNK_REMOVE;
  if (!(AllFlags & SectionFlag::SecReadonly))
    NewCharacteristics |= IMAGE_SCN_MEM_WRITE;
  if (AllFlags & SectionFlag::SecDebug)
    NewCharacteristics |=
        IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_DISCARDABLE;
  if (AllFlags & SectionFlag::SecCode)
    NewCharacteristics |= IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE;
  if (AllFlags & SectionFlag::SecData)
    NewCharacteristics |= IMAGE_SCN_CNT_INITIALIZED_DATA;
  if (AllFlags & SectionFlag::SecShare)
    NewCharacteristics |= IMAGE_SCN_MEM_SHARED;
  if (AllFlags & SectionFlag::SecExclude)
    NewCharacteristics |= IMAGE_SCN_LNK_REMOVE;

  return NewCharacteristics;
}

static Error writeFile(StringRef FileName, ArrayRef<uint8_t> Contents) {
  std::unique_ptr<FileOutputBuffer> Buffer;
  if (auto B = FileOutputBuffer::create(FileName, Contents.size()))
    Buffer = std::move(*B);
  else
    return B.takeError();

  llvm::copy(Contents, Buffer->getBufferStart());
  return Buffer->commit();
}

static Error dumpSection(Object &O, StringRef SectionName, StringRef FileName) {
  for (const coff::Section &Section : O.getSections()) {
    if (Section.Name != SectionName)
      continue;
    return writeFile(FileName, Section.getContents());
  }
  return createStringError(object_error::parse_failed, "section '%s' not found",
                           SectionName.str().c_str());
}

// Returns the size of the address range that a section of a PE image occupies
// when loaded.
static uint64_t getMappedSize(const Section &Sec) {
  return std::max<uint64_t>(Sec.Header.VirtualSize, Sec.Header.SizeOfRawData);
}

static bool containsRVA(const Section &Sec, uint32_t RVA) {
  return RVA >= Sec.Header.VirtualAddress &&
         RVA < Sec.Header.VirtualAddress + getMappedSize(Sec);
}

// Applies Patch to the contents of the data directory with the given index of
// a PE image, if the image has it.
static Error
patchDataDirectory(Object &Obj, unsigned Index,
                   function_ref<Error(MutableArrayRef<uint8_t>)> Patch) {
  if (Obj.DataDirectories.size() <= Index)
    return Error::success();
  const data_directory &Dir = Obj.DataDirectories[Index];
  if (Dir.RelativeVirtualAddress == 0 || Dir.Size == 0)
    return Error::success();
  for (Section &Sec : Obj.getMutableSections()) {
    if (!containsRVA(Sec, Dir.RelativeVirtualAddress))
      continue;
    size_t Offset = Dir.RelativeVirtualAddress - Sec.Header.VirtualAddress;
    if (Offset + Dir.Size > Sec.getContents().size())
      return createStringError(object_error::parse_failed,
                               "data directory extends past end of section");
    std::vector<uint8_t> Contents = Sec.getContents().vec();
    if (Error E =
            Patch(MutableArrayRef<uint8_t>(Contents).slice(Offset, Dir.Size)))
      return E;
    // setOwnedContents sets the raw data size to the size of the contents,
    // which is not necessarily aligned to the file alignment.
    uint32_t SizeOfRawData = Sec.Header.SizeOfRawData;
    Sec.setOwnedContents(std::move(Contents));
    Sec.Header.SizeOfRawData = SizeOfRawData;
    return Error::success();
  }
  return createStringError(object_error::parse_failed,
                           "data directory not found in any section");
}

// Moves the sections of a PE image starting at the given index up by Delta
// bytes in the address space, updating the references to them from the data
// directories, the debug directory and the base relocations. References from
// other places, such as the code, are not updated, so only sections that are
// not referenced from there (such as discardable sections) can be moved.
static Error shiftSections(Object &Obj, size_t FirstIndex, uint32_t Delta) {
  ArrayRef<Section> Sections = Obj.getSections();
  if (FirstIndex >= Sections.size())
    return Error::success();
  uint32_t MovedStart = Sections[FirstIndex].Header.VirtualAddress;
  auto IsMoved = [MovedStart](uint32_t RVA) {
    return RVA != 0 && RVA >= MovedStart;
  };

  size_t Index = 0;
  for (Section &Sec : Obj.getMutableSections())
    if (Index++ >= FirstIndex)
      Sec.Header.VirtualAddress += Delta;

  // The certificate table's entry holds a file offset rather than an RVA.
  for (size_t I = 0; I < Obj.DataDirectories.size(); ++I) {
    data_directory &Dir = Obj.DataDirectories[I];
    if (I != CERTIFICATE_TABLE && IsMoved(Dir.RelativeVirtualAddress))
      Dir.RelativeVirtualAddress += Delta;
  }

  // The debug directory holds the RVAs of the payloads of its entries.
  if (Error E = patchDataDirectory(
          Obj, DEBUG_DIRECTORY,
          [&](MutableArrayRef<uint8_t> Contents) -> Error {
            for (size_t Offset = 0;
                 Offset + sizeof(debug_directory) <= Contents.size();
                 Offset += sizeof(debug_directory)) {
              auto *Entry =
                  reinterpret_cast<debug_directory *>(Contents.data() + Offset);
              if (IsMoved(Entry->AddressOfRawData))
                Entry->AddressOfRawData += Delta;
            }
            return Error::success();
          }))
    return E;

  // The base relocation blocks hold the RVAs of the pages they apply to.
  return patchDataDirectory(
      Obj, BASE_RELOCATION_TABLE,
      [&](MutableArrayRef<uint8_t> Contents) -> Error {
        size_t Offset = 0;
        while (Offset + sizeof(coff_base_reloc_block_header) <=
               Contents.size()) {
          auto *Block = reinterpret_cast<coff_base_reloc_block_header *>(
              Contents.data() + Offset);
          if (Block->BlockSize == 0)
            break;
          if (Block->BlockSize < sizeof(coff_base_reloc_block_header))
            return createStringError(object_error::parse_failed,
                                     "invalid base relocation block size");
          if (IsMoved(Block->PageRVA))
            Block->PageRVA += Delta;
          Offset += Block->BlockSize;
        }
        return Error::success();
      });
}

// Replaces the contents of a section of a PE image, resizing it as needed. If
// the section grows beyond its current address range, the sections following
// it are moved up, which requires them to be discardable.
static Error setSectionContents(Object &Obj, Section &Sec,
                                std::vector<uint8_t> Contents) {
  uint32_t SectionAlignment = Obj.PeHeader.SectionAlignment;
  uint64_t OldEnd = alignTo(getMappedSize(Sec), SectionAlignment);
  uint64_t NewEnd = alignTo(Contents.size(), SectionAlignment);
  if (NewEnd > OldEnd) {
    // The section index is one-based, so this skips the section itself.
    for (const Section &Next : Obj.getSections().drop_front(Sec.Index))
      if (!(Next.Header.Characteristics & IMAGE_SCN_MEM_DISCARDABLE))
        return createStringError(
            errc::invalid_argument,
            "section '%s' cannot grow past section '%s', which is not "
            "discardable and thus cannot be moved",
            Sec.Name.str().c_str(), Next.Name.str().c_str());
    if (Error E = shiftSections(Obj, Sec.Index, NewEnd - OldEnd))
      return E;
  }
  size_t Size = Contents.size();
  Sec.setOwnedContents(std::move(Contents));
  Sec.Header.VirtualSize = Size;
  Sec.Header.SizeOfRawData = alignTo(Size, Obj.PeHeader.FileAlignment);
  return Error::success();
}

struct ResourceSectionLocation {
  Section *Sec = nullptr;
  uint32_t Offset = 0;
};

// Returns the section and offset containing the resource directory of a PE
// image, or a null section if the image has no resources.
static Expected<ResourceSectionLocation> findResourceSection(Object &Obj) {
  if (Obj.DataDirectories.size() <= RESOURCE_TABLE)
    return ResourceSectionLocation();
  const data_directory &Dir = Obj.DataDirectories[RESOURCE_TABLE];
  if (Dir.RelativeVirtualAddress == 0 || Dir.Size == 0)
    return ResourceSectionLocation();
  for (Section &Sec : Obj.getMutableSections()) {
    if (containsRVA(Sec, Dir.RelativeVirtualAddress))
      return ResourceSectionLocation{&Sec, Dir.RelativeVirtualAddress -
                                               Sec.Header.VirtualAddress};
  }
  return createStringError(object_error::parse_failed,
                           "resource directory not found in any section");
}

static Expected<ArrayRef<uint8_t>>
findResourceData(const WindowsResourceParser &Parser,
                 const COFFResourceIdentifier &Resource) {
  if (const WindowsResourceParser::TreeNode *Node =
          Parser.findResource(Resource.Type, Resource.Name)) {
    const auto &Languages = Node->getIDChildren();
    auto Language = Resource.Language ? Languages.find(*Resource.Language)
                                      : Languages.begin();
    if (Language != Languages.end() && Language->second->checkIsDataNode())
      return ArrayRef<uint8_t>(
          Parser.getData()[Language->second->getDataIndex()]);
  }
  if (Resource.Language)
    return createStringError(
        errc::invalid_argument,
        "resource with type %u, name %u and language %u not found",
        Resource.Type, Resource.Name, unsigned(*Resource.Language));
  return createStringError(errc::invalid_argument,
                           "resource with type %u and name %u not found",
                           Resource.Type, Resource.Name);
}

static Error handleResources(const COFFConfig &COFFConfig,
                             const COFFObjectFile &In, Object &Obj) {
  if (COFFConfig.DumpResource.empty() && COFFConfig.UpdateResource.empty())
    return Error::success();
  if (!Obj.IsPE)
    return createStringError(
        errc::invalid_argument,
        "resources can only be dumped from or updated in PE images");

  WindowsResourceParser Parser;
  const data_directory *InputResourceDir = In.getDataDirectory(RESOURCE_TABLE);
  if (InputResourceDir && InputResourceDir->RelativeVirtualAddress != 0 &&
      InputResourceDir->Size != 0) {
    ResourceSectionRef RSR;
    if (Error E = RSR.load(&In))
      return E;
    std::vector<std::string> Duplicates;
    if (Error E = Parser.parse(RSR, In.getFileName(), Duplicates))
      return E;
    if (!Duplicates.empty())
      return createStringError(object_error::parse_failed, "%s",
                               Duplicates.front().c_str());
  }

  for (const COFFResourceDump &Dump : COFFConfig.DumpResource) {
    Expected<ArrayRef<uint8_t>> Data = findResourceData(Parser, Dump.Resource);
    if (!Data)
      return Data.takeError();
    if (Error E = writeFile(Dump.FileName, *Data))
      return E;
  }

  if (COFFConfig.UpdateResource.empty())
    return Error::success();

  Expected<ResourceSectionLocation> ResourceSection = findResourceSection(Obj);
  if (!ResourceSection)
    return ResourceSection.takeError();

  for (const COFFResourceUpdate &Update : COFFConfig.UpdateResource) {
    const COFFResourceIdentifier &Resource = Update.Resource;
    uint16_t Language = 0;
    if (Resource.Language) {
      Language = *Resource.Language;
    } else if (const WindowsResourceParser::TreeNode *Node =
                   Parser.findResource(Resource.Type, Resource.Name)) {
      // Replace the resource for all languages, keeping the language if there
      // is exactly one. Otherwise, the new resource is language-neutral.
      if (Node->getIDChildren().size() == 1)
        Language = Node->getIDChildren().begin()->first;
      else
        Parser.removeResource(Resource.Type, Resource.Name);
    }
    Parser.addResource(Resource.Type, Resource.Name, Language,
                       arrayRefFromStringRef(Update.Data->getBuffer()),
                       Update.Data->getBufferIdentifier());
  }

  if (Section *Sec = ResourceSection->Sec) {
    uint32_t Offset = ResourceSection->Offset;
    ArrayRef<uint8_t> OldContents = Sec->getContents();
    if (Offset > OldContents.size())
      return createStringError(
          object_error::parse_failed,
          "resource directory extends past end of section '%s'",
          Sec->Name.str().c_str());
    // Data following the resources cannot be preserved, as it would have to
    // move when the resources grow. Only allow zero padding there.
    uint64_t ResourcesEnd =
        uint64_t(Offset) + Obj.DataDirectories[RESOURCE_TABLE].Size;
    if (ResourcesEnd < OldContents.size() &&
        !llvm::all_of(
            OldContents.drop_front(ResourcesEnd),
            [](uint8_t Byte) { return Byte == 0; }))
      return createStringError(errc::invalid_argument,
                               "section '%s' contains data after its "
                               "resources that cannot be preserved",
                               Sec->Name.str().c_str());

    std::vector<uint8_t> ResourceContents = writeWindowsResourceSection(
        Parser, Sec->Header.VirtualAddress + Offset);
    std::vector<uint8_t> Contents(OldContents.begin(),
                                  OldContents.begin() + Offset);
    llvm::append_range(Contents, ResourceContents);
    Obj.DataDirectories[RESOURCE_TABLE].Size = ResourceContents.size();
    return setSectionContents(Obj, *Sec, std::move(Contents));
  }

  // Add a resource section after the last non-discardable section, so that
  // discardable sections such as .reloc stay at the end of the image.
  ArrayRef<Section> Sections = Obj.getSections();
  size_t Index = Sections.size();
  while (Index > 0 && (Sections[Index - 1].Header.Characteristics &
                       IMAGE_SCN_MEM_DISCARDABLE))
    --Index;
  if (Index == 0)
    Index = Sections.size();
  uint32_t SectionAlignment = Obj.PeHeader.SectionAlignment;
  uint32_t RVA = Index == 0
                     ? alignTo(Obj.PeHeader.SizeOfHeaders, SectionAlignment)
                     : alignTo(Sections[Index - 1].Header.VirtualAddress +
                                   getMappedSize(Sections[Index - 1]),
                               SectionAlignment);
  std::vector<uint8_t> Contents = writeWindowsResourceSection(Parser, RVA);
  if (Error E =
          shiftSections(Obj, Index, alignTo(Contents.size(), SectionAlignment)))
    return E;

  Section Sec;
  Sec.Name = ".rsrc";
  Sec.Header.VirtualSize = Contents.size();
  Sec.Header.VirtualAddress = RVA;
  Sec.setOwnedContents(std::move(Contents));
  Sec.Header.SizeOfRawData =
      alignTo(Sec.Header.VirtualSize, Obj.PeHeader.FileAlignment);
  // Sec.Header.PointerToRawData is filled in by the writer.
  Sec.Header.PointerToRelocations = 0;
  Sec.Header.PointerToLinenumbers = 0;
  Sec.Header.NumberOfRelocations = 0;
  Sec.Header.NumberOfLinenumbers = 0;
  Sec.Header.Characteristics =
      IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ;
  Obj.insertSection(Index, std::move(Sec));

  if (Obj.DataDirectories.size() <= RESOURCE_TABLE)
    Obj.DataDirectories.resize(RESOURCE_TABLE + 1);
  Obj.DataDirectories[RESOURCE_TABLE].RelativeVirtualAddress = RVA;
  Obj.DataDirectories[RESOURCE_TABLE].Size =
      Obj.getSections()[Index].Header.VirtualSize;
  return Error::success();
}

static Error handleArgs(const CommonConfig &Config,
                        const COFFConfig &COFFConfig, const COFFObjectFile &In,
                        Object &Obj) {
  for (StringRef Op : Config.DumpSection) {
    auto [Section, File] = Op.split('=');
    if (Error E = dumpSection(Obj, Section, File))
      return E;
  }

  // Perform the actual section removals.
  Obj.removeSections([&Config](const Section &Sec) {
    // Contrary to --only-keep-debug, --only-section fully removes sections that
    // aren't mentioned.
    if (!Config.OnlySection.empty() && !Config.OnlySection.matches(Sec.Name))
      return true;

    if (Config.StripDebug || Config.StripAll || Config.StripAllGNU ||
        Config.DiscardMode == DiscardType::All || Config.StripUnneeded) {
      if (isDebugSection(Sec) &&
          (Sec.Header.Characteristics & IMAGE_SCN_MEM_DISCARDABLE) != 0)
        return true;
    }

    if (Config.ToRemove.matches(Sec.Name))
      return true;

    return false;
  });

  if (Config.OnlyKeepDebug) {
    const data_directory *DebugDir =
        Obj.DataDirectories.size() > DEBUG_DIRECTORY
            ? &Obj.DataDirectories[DEBUG_DIRECTORY]
            : nullptr;
    // For --only-keep-debug, we keep all other sections, but remove their
    // content. The VirtualSize field in the section header is kept intact.
    Obj.truncateSections([DebugDir](const Section &Sec) {
      return !isDebugSection(Sec) && Sec.Name != ".buildid" &&
             !(DebugDir && DebugDir->Size > 0 &&
               DebugDir->RelativeVirtualAddress >= Sec.Header.VirtualAddress &&
               DebugDir->RelativeVirtualAddress <
                   Sec.Header.VirtualAddress + Sec.Header.SizeOfRawData) &&
             ((Sec.Header.Characteristics &
               (IMAGE_SCN_CNT_CODE | IMAGE_SCN_CNT_INITIALIZED_DATA)) != 0);
    });
  }

  // StripAll removes all symbols and thus also removes all relocations.
  if (Config.StripAll || Config.StripAllGNU)
    for (Section &Sec : Obj.getMutableSections())
      Sec.Relocs.clear();

  // If we need to do per-symbol removals, initialize the Referenced field.
  if (Config.StripUnneeded || Config.DiscardMode == DiscardType::All ||
      !Config.SymbolsToRemove.empty())
    if (Error E = Obj.markSymbols())
      return E;

  for (Symbol &Sym : Obj.getMutableSymbols()) {
    auto I = Config.SymbolsToRename.find(Sym.Name);
    if (I != Config.SymbolsToRename.end())
      Sym.Name = I->getValue();
  }

  auto ToRemove = [&](const Symbol &Sym) -> Expected<bool> {
    // For StripAll, all relocations have been stripped and we remove all
    // symbols.
    if (Config.StripAll || Config.StripAllGNU)
      return true;

    if (Config.SymbolsToRemove.matches(Sym.Name)) {
      // Explicitly removing a referenced symbol is an error.
      if (Sym.Referenced)
        return createStringError(llvm::errc::invalid_argument,
                                 "'" + Config.OutputFilename +
                                     "': not stripping symbol '" + Sym.Name +
                                     "' because it is named in a relocation");
      return true;
    }

    if (!Sym.Referenced) {
      // With --strip-unneeded, GNU objcopy removes all unreferenced local
      // symbols, and any unreferenced undefined external.
      // With --strip-unneeded-symbol we strip only specific unreferenced
      // local symbol instead of removing all of such.
      if (Sym.Sym.StorageClass == IMAGE_SYM_CLASS_STATIC ||
          Sym.Sym.SectionNumber == 0)
        if (Config.StripUnneeded ||
            Config.UnneededSymbolsToRemove.matches(Sym.Name))
          return true;

      // GNU objcopy keeps referenced local symbols and external symbols
      // if --discard-all is set, similar to what --strip-unneeded does,
      // but undefined local symbols are kept when --discard-all is set.
      if (Config.DiscardMode == DiscardType::All &&
          Sym.Sym.StorageClass == IMAGE_SYM_CLASS_STATIC &&
          Sym.Sym.SectionNumber != 0)
        return true;
    }

    return false;
  };

  // Actually do removals of symbols.
  if (Error Err = Obj.removeSymbols(ToRemove))
    return Err;

  if (!Config.SetSectionFlags.empty())
    for (Section &Sec : Obj.getMutableSections()) {
      const auto It = Config.SetSectionFlags.find(Sec.Name);
      if (It != Config.SetSectionFlags.end())
        Sec.Header.Characteristics = flagsToCharacteristics(
            It->second.NewFlags, Sec.Header.Characteristics);
    }

  for (const NewSectionInfo &NewSection : Config.AddSection) {
    uint32_t Characteristics;
    const auto It = Config.SetSectionFlags.find(NewSection.SectionName);
    if (It != Config.SetSectionFlags.end())
      Characteristics = flagsToCharacteristics(It->second.NewFlags, 0);
    else
      Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_ALIGN_1BYTES;

    addSection(Obj, NewSection.SectionName,
               ArrayRef(reinterpret_cast<const uint8_t *>(
                            NewSection.SectionData->getBufferStart()),
                        NewSection.SectionData->getBufferSize()),
               Characteristics);
  }

  for (const NewSectionInfo &NewSection : Config.UpdateSection) {
    auto It = llvm::find_if(Obj.getMutableSections(), [&](auto &Sec) {
      return Sec.Name == NewSection.SectionName;
    });
    if (It == Obj.getMutableSections().end())
      return createStringError(errc::invalid_argument,
                               "could not find section with name '%s'",
                               NewSection.SectionName.str().c_str());
    size_t ContentSize = It->getContents().size();
    if (!ContentSize)
      return createStringError(
          errc::invalid_argument,
          "section '%s' cannot be updated because it does not have contents",
          NewSection.SectionName.str().c_str());
    if (ContentSize < NewSection.SectionData->getBufferSize())
      return createStringError(
          errc::invalid_argument,
          "new section cannot be larger than previous section");
    It->setOwnedContents({NewSection.SectionData->getBufferStart(),
                          NewSection.SectionData->getBufferEnd()});
  }

  if (Error E = handleResources(COFFConfig, In, Obj))
    return E;

  if (!Config.AddGnuDebugLink.empty())
    if (Error E = addGnuDebugLink(Obj, Config.AddGnuDebugLink))
      return E;

  if (COFFConfig.Subsystem || COFFConfig.MajorSubsystemVersion ||
      COFFConfig.MinorSubsystemVersion) {
    if (!Obj.IsPE)
      return createStringError(
          errc::invalid_argument,
          "'" + Config.OutputFilename +
              "': unable to set subsystem on a relocatable object file");
    if (COFFConfig.Subsystem)
      Obj.PeHeader.Subsystem = *COFFConfig.Subsystem;
    if (COFFConfig.MajorSubsystemVersion)
      Obj.PeHeader.MajorSubsystemVersion = *COFFConfig.MajorSubsystemVersion;
    if (COFFConfig.MinorSubsystemVersion)
      Obj.PeHeader.MinorSubsystemVersion = *COFFConfig.MinorSubsystemVersion;
  }

  return Error::success();
}

Error executeObjcopyOnBinary(const CommonConfig &Config,
                             const COFFConfig &COFFConfig, COFFObjectFile &In,
                             raw_ostream &Out) {
  COFFReader Reader(In);
  Expected<std::unique_ptr<Object>> ObjOrErr = Reader.create();
  if (!ObjOrErr)
    return createFileError(Config.InputFilename, ObjOrErr.takeError());
  Object *Obj = ObjOrErr->get();
  assert(Obj && "Unable to deserialize COFF object");
  if (Error E = handleArgs(Config, COFFConfig, In, *Obj))
    return createFileError(Config.InputFilename, std::move(E));
  COFFWriter Writer(*Obj, Out);
  if (Error E = Writer.write())
    return createFileError(Config.OutputFilename, std::move(E));
  return Error::success();
}

} // end namespace coff
} // end namespace objcopy
} // end namespace llvm
