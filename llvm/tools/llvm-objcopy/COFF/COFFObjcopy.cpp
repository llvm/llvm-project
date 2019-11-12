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
#include "llvm/Support/CRC.h"
#include "llvm/Support/Errc.h"
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

static std::vector<uint8_t> createGnuDebugLinkSectionContents(StringRef File) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> LinkTargetOrErr =
      MemoryBuffer::getFile(File);
  if (!LinkTargetOrErr)
    error("'" + File + "': " + LinkTargetOrErr.getError().message());
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

static void addGnuDebugLink(Object &Obj, StringRef DebugLinkFile) {
  std::vector<uint8_t> Contents =
      createGnuDebugLinkSectionContents(DebugLinkFile);
  addSection(Obj, ".gnu_debuglink", Contents,
             IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ |
                 IMAGE_SCN_MEM_DISCARDABLE);
}

static Error handleArgs(const CopyConfig &Config, Object &Obj) {
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

  for (Symbol &Sym : Obj.getMutableSymbols()) {
    auto I = Config.SymbolsToRename.find(Sym.Name);
    if (I != Config.SymbolsToRename.end())
      Sym.Name = I->getValue();
  }

  // Actually do removals of symbols.
  Obj.removeSymbols([&](const Symbol &Sym) {
    // For StripAll, all relocations have been stripped and we remove all
    // symbols.
    if (Config.StripAll || Config.StripAllGNU)
      return true;

    if (Config.SymbolsToRemove.matches(Sym.Name)) {
      // Explicitly removing a referenced symbol is an error.
      if (Sym.Referenced)
        reportError(Config.OutputFilename,
                    createStringError(llvm::errc::invalid_argument,
                                      "not stripping symbol '%s' because it is "
                                      "named in a relocation",
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
  });

  for (const auto &Flag : Config.AddSection) {
    StringRef SecName, FileName;
    std::tie(SecName, FileName) = Flag.split("=");

    auto BufOrErr = MemoryBuffer::getFile(FileName);
    if (!BufOrErr)
      return createFileError(FileName, errorCodeToError(BufOrErr.getError()));
    auto Buf = std::move(*BufOrErr);

    addSection(
        Obj, SecName,
        makeArrayRef(reinterpret_cast<const uint8_t *>(Buf->getBufferStart()),
                     Buf->getBufferSize()),
        IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_ALIGN_1BYTES);
  }

  if (!Config.AddGnuDebugLink.empty())
    addGnuDebugLink(Obj, Config.AddGnuDebugLink);

  if (Config.AllowBrokenLinks || !Config.BuildIdLinkDir.empty() ||
      Config.BuildIdLinkInput || Config.BuildIdLinkOutput ||
      !Config.SplitDWO.empty() || !Config.SymbolsPrefix.empty() ||
      !Config.AllocSectionsPrefix.empty() || !Config.DumpSection.empty() ||
      !Config.KeepSection.empty() || Config.NewSymbolVisibility ||
      !Config.SymbolsToGlobalize.empty() || !Config.SymbolsToKeep.empty() ||
      !Config.SymbolsToLocalize.empty() || !Config.SymbolsToWeaken.empty() ||
      !Config.SymbolsToKeepGlobal.empty() || !Config.SectionsToRename.empty() ||
      !Config.SetSectionAlignment.empty() || !Config.SetSectionFlags.empty() ||
      Config.ExtractDWO || Config.KeepFileSymbols || Config.LocalizeHidden ||
      Config.PreserveDates || Config.StripDWO || Config.StripNonAlloc ||
      Config.StripSections || Config.Weaken || Config.DecompressDebugSections ||
      Config.DiscardMode == DiscardType::Locals ||
      !Config.SymbolsToAdd.empty() || Config.EntryExpr) {
    return createStringError(llvm::errc::invalid_argument,
                             "option not supported by llvm-objcopy for COFF");
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
