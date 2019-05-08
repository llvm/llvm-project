//===- COFFObjcopy.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "COFFObjcopy.h"
#include "Buffer.h"
#include "CopyConfig.h"
#include "Object.h"
#include "Reader.h"
#include "Writer.h"
#include "llvm-objcopy.h"

#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/JamCRC.h"
#include "llvm/Support/Path.h"
#include <cassert>

namespace llvm {
namespace objcopy {
namespace coff {

using namespace object;
using namespace COFF;

static bool isDebugSection(const Section &Sec) {
  return Sec.Name.startswith(".debug");
}

static uint64_t getNextRVA(const Object &Obj) {
  if (Obj.getSections().empty())
    return 0;
  const Section &Last = Obj.getSections().back();
  return alignTo(Last.Header.VirtualAddress + Last.Header.VirtualSize,
                 Obj.IsPE ? Obj.PeHeader.SectionAlignment : 1);
}

static uint32_t getCRC32(StringRef Data) {
  JamCRC CRC;
  CRC.update(ArrayRef<char>(Data.data(), Data.size()));
  // The CRC32 value needs to be complemented because the JamCRC dosn't
  // finalize the CRC32 value. It also dosn't negate the initial CRC32 value
  // but it starts by default at 0xFFFFFFFF which is the complement of zero.
  return ~CRC.getCRC();
}

static std::vector<uint8_t> createGnuDebugLinkSectionContents(StringRef File) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> LinkTargetOrErr =
      MemoryBuffer::getFile(File);
  if (!LinkTargetOrErr)
    error("'" + File + "': " + LinkTargetOrErr.getError().message());
  auto LinkTarget = std::move(*LinkTargetOrErr);
  uint32_t CRC32 = getCRC32(LinkTarget->getBuffer());

  StringRef FileName = sys::path::filename(File);
  size_t CRCPos = alignTo(FileName.size() + 1, 4);
  std::vector<uint8_t> Data(CRCPos + 4);
  memcpy(Data.data(), FileName.data(), FileName.size());
  support::endian::write32le(Data.data() + CRCPos, CRC32);
  return Data;
}

static void addGnuDebugLink(Object &Obj, StringRef DebugLinkFile) {
  uint32_t StartRVA = getNextRVA(Obj);

  std::vector<Section> Sections;
  Section Sec;
  Sec.setOwnedContents(createGnuDebugLinkSectionContents(DebugLinkFile));
  Sec.Name = ".gnu_debuglink";
  Sec.Header.VirtualSize = Sec.getContents().size();
  Sec.Header.VirtualAddress = StartRVA;
  Sec.Header.SizeOfRawData = alignTo(Sec.Header.VirtualSize,
                                     Obj.IsPE ? Obj.PeHeader.FileAlignment : 1);
  // Sec.Header.PointerToRawData is filled in by the writer.
  Sec.Header.PointerToRelocations = 0;
  Sec.Header.PointerToLinenumbers = 0;
  // Sec.Header.NumberOfRelocations is filled in by the writer.
  Sec.Header.NumberOfLinenumbers = 0;
  Sec.Header.Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA |
                               IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_DISCARDABLE;
  Sections.push_back(Sec);
  Obj.addSections(Sections);
}

static Error handleArgs(const CopyConfig &Config, Object &Obj) {
  // Perform the actual section removals.
  Obj.removeSections([&Config](const Section &Sec) {
    // Contrary to --only-keep-debug, --only-section fully removes sections that
    // aren't mentioned.
    if (!Config.OnlySection.empty() &&
        !is_contained(Config.OnlySection, Sec.Name))
      return true;

    if (Config.StripDebug || Config.StripAll || Config.StripAllGNU ||
        Config.DiscardMode == DiscardType::All || Config.StripUnneeded) {
      if (isDebugSection(Sec) &&
          (Sec.Header.Characteristics & IMAGE_SCN_MEM_DISCARDABLE) != 0)
        return true;
    }

    if (is_contained(Config.ToRemove, Sec.Name))
      return true;

    return false;
  });

  if (Config.OnlyKeepDebug) {
    // For --only-keep-debug, we keep all other sections, but remove their
    // content. The VirtualSize field in the section header is kept intact.
    Obj.truncateSections([](const Section &Sec) {
      return !isDebugSection(Sec) && Sec.Name != ".buildid" &&
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

  // Actually do removals of symbols.
  Obj.removeSymbols([&](const Symbol &Sym) {
    // For StripAll, all relocations have been stripped and we remove all
    // symbols.
    if (Config.StripAll || Config.StripAllGNU)
      return true;

    if (is_contained(Config.SymbolsToRemove, Sym.Name)) {
      // Explicitly removing a referenced symbol is an error.
      if (Sym.Referenced)
        reportError(Config.OutputFilename,
                    createStringError(llvm::errc::invalid_argument,
                                      "not stripping symbol '%s' because it is "
                                      "named in a relocation.",
                                      Sym.Name.str().c_str()));
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
            is_contained(Config.UnneededSymbolsToRemove, Sym.Name))
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
  });

  if (!Config.AddGnuDebugLink.empty())
    addGnuDebugLink(Obj, Config.AddGnuDebugLink);

  if (Config.AllowBrokenLinks || !Config.BuildIdLinkDir.empty() ||
      Config.BuildIdLinkInput || Config.BuildIdLinkOutput ||
      !Config.SplitDWO.empty() || !Config.SymbolsPrefix.empty() ||
      !Config.AllocSectionsPrefix.empty() || !Config.AddSection.empty() ||
      !Config.DumpSection.empty() || !Config.KeepSection.empty() ||
      !Config.SymbolsToGlobalize.empty() || !Config.SymbolsToKeep.empty() ||
      !Config.SymbolsToLocalize.empty() || !Config.SymbolsToWeaken.empty() ||
      !Config.SymbolsToKeepGlobal.empty() || !Config.SectionsToRename.empty() ||
      !Config.SetSectionFlags.empty() || !Config.SymbolsToRename.empty() ||
      Config.ExtractDWO || Config.KeepFileSymbols || Config.LocalizeHidden ||
      Config.PreserveDates || Config.StripDWO || Config.StripNonAlloc ||
      Config.StripSections || Config.Weaken || Config.DecompressDebugSections ||
      Config.DiscardMode == DiscardType::Locals ||
      !Config.SymbolsToAdd.empty() || Config.EntryExpr) {
    return createStringError(llvm::errc::invalid_argument,
                             "Option not supported by llvm-objcopy for COFF");
  }

  return Error::success();
}

Error executeObjcopyOnBinary(const CopyConfig &Config, COFFObjectFile &In,
                             Buffer &Out) {
  COFFReader Reader(In);
  Expected<std::unique_ptr<Object>> ObjOrErr = Reader.create();
  if (!ObjOrErr)
    return createFileError(Config.InputFilename, ObjOrErr.takeError());
  Object *Obj = ObjOrErr->get();
  assert(Obj && "Unable to deserialize COFF object");
  if (Error E = handleArgs(Config, *Obj))
    return createFileError(Config.InputFilename, std::move(E));
  COFFWriter Writer(*Obj, Out);
  if (Error E = Writer.write())
    return createFileError(Config.OutputFilename, std::move(E));
  return Error::success();
}

} // end namespace coff
} // end namespace objcopy
} // end namespace llvm
