//===- ELFObjcopy.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ELFObjcopy.h"
#include "Buffer.h"
#include "CopyConfig.h"
#include "Object.h"
#include "llvm-objcopy.h"

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/Error.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

namespace llvm {
namespace objcopy {
namespace elf {

using namespace object;
using namespace ELF;
using SectionPred = std::function<bool(const SectionBase &Sec)>;

static bool isDebugSection(const SectionBase &Sec) {
  return StringRef(Sec.Name).startswith(".debug") ||
         StringRef(Sec.Name).startswith(".zdebug") || Sec.Name == ".gdb_index";
}

static bool isDWOSection(const SectionBase &Sec) {
  return StringRef(Sec.Name).endswith(".dwo");
}

static bool onlyKeepDWOPred(const Object &Obj, const SectionBase &Sec) {
  // We can't remove the section header string table.
  if (&Sec == Obj.SectionNames)
    return false;
  // Short of keeping the string table we want to keep everything that is a DWO
  // section and remove everything else.
  return !isDWOSection(Sec);
}

uint64_t getNewShfFlags(SectionFlag AllFlags) {
  uint64_t NewFlags = 0;
  if (AllFlags & SectionFlag::SecAlloc)
    NewFlags |= ELF::SHF_ALLOC;
  if (!(AllFlags & SectionFlag::SecReadonly))
    NewFlags |= ELF::SHF_WRITE;
  if (AllFlags & SectionFlag::SecCode)
    NewFlags |= ELF::SHF_EXECINSTR;
  if (AllFlags & SectionFlag::SecMerge)
    NewFlags |= ELF::SHF_MERGE;
  if (AllFlags & SectionFlag::SecStrings)
    NewFlags |= ELF::SHF_STRINGS;
  return NewFlags;
}

static uint64_t getSectionFlagsPreserveMask(uint64_t OldFlags,
                                            uint64_t NewFlags) {
  // Preserve some flags which should not be dropped when setting flags.
  // Also, preserve anything OS/processor dependant.
  const uint64_t PreserveMask = ELF::SHF_COMPRESSED | ELF::SHF_EXCLUDE |
                                ELF::SHF_GROUP | ELF::SHF_LINK_ORDER |
                                ELF::SHF_MASKOS | ELF::SHF_MASKPROC |
                                ELF::SHF_TLS | ELF::SHF_INFO_LINK;
  return (OldFlags & PreserveMask) | (NewFlags & ~PreserveMask);
}

static void setSectionFlagsAndType(SectionBase &Sec, SectionFlag Flags) {
  Sec.Flags = getSectionFlagsPreserveMask(Sec.Flags, getNewShfFlags(Flags));

  // In GNU objcopy, certain flags promote SHT_NOBITS to SHT_PROGBITS. This rule
  // may promote more non-ALLOC sections than GNU objcopy, but it is fine as
  // non-ALLOC SHT_NOBITS sections do not make much sense.
  if (Sec.Type == SHT_NOBITS &&
      (!(Sec.Flags & ELF::SHF_ALLOC) ||
       Flags & (SectionFlag::SecContents | SectionFlag::SecLoad)))
    Sec.Type = SHT_PROGBITS;
}

static ElfType getOutputElfType(const Binary &Bin) {
  // Infer output ELF type from the input ELF object
  if (isa<ELFObjectFile<ELF32LE>>(Bin))
    return ELFT_ELF32LE;
  if (isa<ELFObjectFile<ELF64LE>>(Bin))
    return ELFT_ELF64LE;
  if (isa<ELFObjectFile<ELF32BE>>(Bin))
    return ELFT_ELF32BE;
  if (isa<ELFObjectFile<ELF64BE>>(Bin))
    return ELFT_ELF64BE;
  llvm_unreachable("Invalid ELFType");
}

static ElfType getOutputElfType(const MachineInfo &MI) {
  // Infer output ELF type from the binary arch specified
  if (MI.Is64Bit)
    return MI.IsLittleEndian ? ELFT_ELF64LE : ELFT_ELF64BE;
  else
    return MI.IsLittleEndian ? ELFT_ELF32LE : ELFT_ELF32BE;
}

static std::unique_ptr<Writer> createWriter(const CopyConfig &Config,
                                            Object &Obj, Buffer &Buf,
                                            ElfType OutputElfType) {
  if (Config.OutputFormat == "binary") {
    return llvm::make_unique<BinaryWriter>(Obj, Buf);
  }
  // Depending on the initial ELFT and OutputFormat we need a different Writer.
  switch (OutputElfType) {
  case ELFT_ELF32LE:
    return llvm::make_unique<ELFWriter<ELF32LE>>(Obj, Buf,
                                                 !Config.StripSections);
  case ELFT_ELF64LE:
    return llvm::make_unique<ELFWriter<ELF64LE>>(Obj, Buf,
                                                 !Config.StripSections);
  case ELFT_ELF32BE:
    return llvm::make_unique<ELFWriter<ELF32BE>>(Obj, Buf,
                                                 !Config.StripSections);
  case ELFT_ELF64BE:
    return llvm::make_unique<ELFWriter<ELF64BE>>(Obj, Buf,
                                                 !Config.StripSections);
  }
  llvm_unreachable("Invalid output format");
}

template <class ELFT>
static Expected<ArrayRef<uint8_t>>
findBuildID(const object::ELFFile<ELFT> &In) {
  for (const auto &Phdr : unwrapOrError(In.program_headers())) {
    if (Phdr.p_type != PT_NOTE)
      continue;
    Error Err = Error::success();
    for (const auto &Note : In.notes(Phdr, Err))
      if (Note.getType() == NT_GNU_BUILD_ID && Note.getName() == ELF_NOTE_GNU)
        return Note.getDesc();
    if (Err)
      return std::move(Err);
  }
  return createStringError(llvm::errc::invalid_argument,
                           "Could not find build ID.");
}

static Expected<ArrayRef<uint8_t>>
findBuildID(const object::ELFObjectFileBase &In) {
  if (auto *O = dyn_cast<ELFObjectFile<ELF32LE>>(&In))
    return findBuildID(*O->getELFFile());
  else if (auto *O = dyn_cast<ELFObjectFile<ELF64LE>>(&In))
    return findBuildID(*O->getELFFile());
  else if (auto *O = dyn_cast<ELFObjectFile<ELF32BE>>(&In))
    return findBuildID(*O->getELFFile());
  else if (auto *O = dyn_cast<ELFObjectFile<ELF64BE>>(&In))
    return findBuildID(*O->getELFFile());

  llvm_unreachable("Bad file format");
}

template <class... Ts>
static Error makeStringError(std::error_code EC, const Twine &Msg, Ts&&... Args) {
  std::string FullMsg = (EC.message() + ": " + Msg).str();
  return createStringError(EC, FullMsg.c_str(), std::forward<Ts>(Args)...);
}

#define MODEL_8 "%%%%%%%%"
#define MODEL_16 MODEL_8 MODEL_8
#define MODEL_32 (MODEL_16 MODEL_16)

static Error linkToBuildIdDir(const CopyConfig &Config, StringRef ToLink,
                              StringRef Suffix,
                              ArrayRef<uint8_t> BuildIdBytes) {
  SmallString<128> Path = Config.BuildIdLinkDir;
  sys::path::append(Path, llvm::toHex(BuildIdBytes[0], /*LowerCase*/ true));
  if (auto EC = sys::fs::create_directories(Path))
    return createFileError(
        Path.str(),
        makeStringError(EC, "cannot create build ID link directory"));

  sys::path::append(Path,
                    llvm::toHex(BuildIdBytes.slice(1), /*LowerCase*/ true));
  Path += Suffix;
  SmallString<128> TmpPath;
  // create_hard_link races so we need to link to a temporary path but
  // we want to make sure that we choose a filename that does not exist.
  // By using 32 model characters we get 128-bits of entropy. It is
  // unlikely that this string has ever existed before much less exists
  // on this disk or in the current working directory.
  // Additionally we prepend the original Path for debugging but also
  // because it ensures that we're linking within a directory on the same
  // partition on the same device which is critical. It has the added
  // win of yet further decreasing the odds of a conflict.
  sys::fs::createUniquePath(Twine(Path) + "-" + MODEL_32 + ".tmp", TmpPath,
                            /*MakeAbsolute*/ false);
  if (auto EC = sys::fs::create_hard_link(ToLink, TmpPath)) {
    Path.push_back('\0');
    return makeStringError(EC, "cannot link %s to %s", ToLink.data(),
                             Path.data());
  }
  // We then atomically rename the link into place which will just move the
  // link. If rename fails something is more seriously wrong so just return
  // an error.
  if (auto EC = sys::fs::rename(TmpPath, Path)) {
    Path.push_back('\0');
    return makeStringError(EC, "cannot link %s to %s", ToLink.data(),
                             Path.data());
  }
  // If `Path` was already a hard-link to the same underlying file then the
  // temp file will be left so we need to remove it. Remove will not cause
  // an error by default if the file is already gone so just blindly remove
  // it rather than checking.
  if (auto EC = sys::fs::remove(TmpPath)) {
    TmpPath.push_back('\0');
    return makeStringError(EC, "could not remove %s", TmpPath.data());
  }
  return Error::success();
}

static Error splitDWOToFile(const CopyConfig &Config, const Reader &Reader,
                            StringRef File, ElfType OutputElfType) {
  auto DWOFile = Reader.create();
  auto OnlyKeepDWOPred = [&DWOFile](const SectionBase &Sec) {
    return onlyKeepDWOPred(*DWOFile, Sec);
  };
  if (Error E = DWOFile->removeSections(Config.AllowBrokenLinks,
                                        OnlyKeepDWOPred))
    return E;
  if (Config.OutputArch) {
    DWOFile->Machine = Config.OutputArch.getValue().EMachine;
    DWOFile->OSABI = Config.OutputArch.getValue().OSABI;
  }
  FileBuffer FB(File);
  auto Writer = createWriter(Config, *DWOFile, FB, OutputElfType);
  if (Error E = Writer->finalize())
    return E;
  return Writer->write();
}

static Error dumpSectionToFile(StringRef SecName, StringRef Filename,
                               Object &Obj) {
  for (auto &Sec : Obj.sections()) {
    if (Sec.Name == SecName) {
      if (Sec.OriginalData.empty())
        return createStringError(
            object_error::parse_failed,
            "Can't dump section \"%s\": it has no contents",
            SecName.str().c_str());
      Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
          FileOutputBuffer::create(Filename, Sec.OriginalData.size());
      if (!BufferOrErr)
        return BufferOrErr.takeError();
      std::unique_ptr<FileOutputBuffer> Buf = std::move(*BufferOrErr);
      std::copy(Sec.OriginalData.begin(), Sec.OriginalData.end(),
                Buf->getBufferStart());
      if (Error E = Buf->commit())
        return E;
      return Error::success();
    }
  }
  return createStringError(object_error::parse_failed, "Section not found");
}

static bool isCompressable(const SectionBase &Section) {
  return !(Section.Flags & ELF::SHF_COMPRESSED) &&
         StringRef(Section.Name).startswith(".debug");
}

static void replaceDebugSections(
    Object &Obj, SectionPred &RemovePred,
    function_ref<bool(const SectionBase &)> shouldReplace,
    function_ref<SectionBase *(const SectionBase *)> addSection) {
  // Build a list of the debug sections we are going to replace.
  // We can't call `addSection` while iterating over sections,
  // because it would mutate the sections array.
  SmallVector<SectionBase *, 13> ToReplace;
  for (auto &Sec : Obj.sections())
    if (shouldReplace(Sec))
      ToReplace.push_back(&Sec);

  // Build a mapping from original section to a new one.
  DenseMap<SectionBase *, SectionBase *> FromTo;
  for (SectionBase *S : ToReplace)
    FromTo[S] = addSection(S);

  // Now we want to update the target sections of relocation
  // sections. Also we will update the relocations themselves
  // to update the symbol references.
  for (auto &Sec : Obj.sections())
    Sec.replaceSectionReferences(FromTo);

  RemovePred = [shouldReplace, RemovePred](const SectionBase &Sec) {
    return shouldReplace(Sec) || RemovePred(Sec);
  };
}

static bool isUnneededSymbol(const Symbol &Sym) {
  return !Sym.Referenced &&
         (Sym.Binding == STB_LOCAL || Sym.getShndx() == SHN_UNDEF) &&
         Sym.Type != STT_FILE && Sym.Type != STT_SECTION;
}

static Error updateAndRemoveSymbols(const CopyConfig &Config, Object &Obj) {
  // TODO: update or remove symbols only if there is an option that affects
  // them.
  if (!Obj.SymbolTable)
    return Error::success();

  Obj.SymbolTable->updateSymbols([&](Symbol &Sym) {
    // Common and undefined symbols don't make sense as local symbols, and can
    // even cause crashes if we localize those, so skip them.
    if (!Sym.isCommon() && Sym.getShndx() != SHN_UNDEF &&
        ((Config.LocalizeHidden &&
          (Sym.Visibility == STV_HIDDEN || Sym.Visibility == STV_INTERNAL)) ||
         is_contained(Config.SymbolsToLocalize, Sym.Name)))
      Sym.Binding = STB_LOCAL;

    // Note: these two globalize flags have very similar names but different
    // meanings:
    //
    // --globalize-symbol: promote a symbol to global
    // --keep-global-symbol: all symbols except for these should be made local
    //
    // If --globalize-symbol is specified for a given symbol, it will be
    // global in the output file even if it is not included via
    // --keep-global-symbol. Because of that, make sure to check
    // --globalize-symbol second.
    if (!Config.SymbolsToKeepGlobal.empty() &&
        !is_contained(Config.SymbolsToKeepGlobal, Sym.Name) &&
        Sym.getShndx() != SHN_UNDEF)
      Sym.Binding = STB_LOCAL;

    if (is_contained(Config.SymbolsToGlobalize, Sym.Name) &&
        Sym.getShndx() != SHN_UNDEF)
      Sym.Binding = STB_GLOBAL;

    if (is_contained(Config.SymbolsToWeaken, Sym.Name) &&
        Sym.Binding == STB_GLOBAL)
      Sym.Binding = STB_WEAK;

    if (Config.Weaken && Sym.Binding == STB_GLOBAL &&
        Sym.getShndx() != SHN_UNDEF)
      Sym.Binding = STB_WEAK;

    const auto I = Config.SymbolsToRename.find(Sym.Name);
    if (I != Config.SymbolsToRename.end())
      Sym.Name = I->getValue();

    if (!Config.SymbolsPrefix.empty() && Sym.Type != STT_SECTION)
      Sym.Name = (Config.SymbolsPrefix + Sym.Name).str();
  });

  // The purpose of this loop is to mark symbols referenced by sections
  // (like GroupSection or RelocationSection). This way, we know which
  // symbols are still 'needed' and which are not.
  if (Config.StripUnneeded || !Config.UnneededSymbolsToRemove.empty()) {
    for (auto &Section : Obj.sections())
      Section.markSymbols();
  }

  auto RemoveSymbolsPred = [&](const Symbol &Sym) {
    if (is_contained(Config.SymbolsToKeep, Sym.Name) ||
        (Config.KeepFileSymbols && Sym.Type == STT_FILE))
      return false;

    if ((Config.DiscardMode == DiscardType::All ||
         (Config.DiscardMode == DiscardType::Locals &&
          StringRef(Sym.Name).startswith(".L"))) &&
        Sym.Binding == STB_LOCAL && Sym.getShndx() != SHN_UNDEF &&
        Sym.Type != STT_FILE && Sym.Type != STT_SECTION)
      return true;

    if (Config.StripAll || Config.StripAllGNU)
      return true;

    if (is_contained(Config.SymbolsToRemove, Sym.Name))
      return true;

    if ((Config.StripUnneeded ||
         is_contained(Config.UnneededSymbolsToRemove, Sym.Name)) &&
        isUnneededSymbol(Sym))
      return true;

    return false;
  };

  return Obj.removeSymbols(RemoveSymbolsPred);
}

static Error replaceAndRemoveSections(const CopyConfig &Config, Object &Obj) {
  SectionPred RemovePred = [](const SectionBase &) { return false; };

  // Removes:
  if (!Config.ToRemove.empty()) {
    RemovePred = [&Config](const SectionBase &Sec) {
      return is_contained(Config.ToRemove, Sec.Name);
    };
  }

  if (Config.StripDWO || !Config.SplitDWO.empty())
    RemovePred = [RemovePred](const SectionBase &Sec) {
      return isDWOSection(Sec) || RemovePred(Sec);
    };

  if (Config.ExtractDWO)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      return onlyKeepDWOPred(Obj, Sec) || RemovePred(Sec);
    };

  if (Config.StripAllGNU)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if ((Sec.Flags & SHF_ALLOC) != 0)
        return false;
      if (&Sec == Obj.SectionNames)
        return false;
      switch (Sec.Type) {
      case SHT_SYMTAB:
      case SHT_REL:
      case SHT_RELA:
      case SHT_STRTAB:
        return true;
      }
      return isDebugSection(Sec);
    };

  if (Config.StripSections) {
    RemovePred = [RemovePred](const SectionBase &Sec) {
      return RemovePred(Sec) || Sec.ParentSegment == nullptr;
    };
  }

  if (Config.StripDebug) {
    RemovePred = [RemovePred](const SectionBase &Sec) {
      return RemovePred(Sec) || isDebugSection(Sec);
    };
  }

  if (Config.StripNonAlloc)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if (&Sec == Obj.SectionNames)
        return false;
      return (Sec.Flags & SHF_ALLOC) == 0 && Sec.ParentSegment == nullptr;
    };

  if (Config.StripAll)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if (&Sec == Obj.SectionNames)
        return false;
      if (StringRef(Sec.Name).startswith(".gnu.warning"))
        return false;
      if (Sec.ParentSegment != nullptr)
        return false;
      return (Sec.Flags & SHF_ALLOC) == 0;
    };

  // Explicit copies:
  if (!Config.OnlySection.empty()) {
    RemovePred = [&Config, RemovePred, &Obj](const SectionBase &Sec) {
      // Explicitly keep these sections regardless of previous removes.
      if (is_contained(Config.OnlySection, Sec.Name))
        return false;

      // Allow all implicit removes.
      if (RemovePred(Sec))
        return true;

      // Keep special sections.
      if (Obj.SectionNames == &Sec)
        return false;
      if (Obj.SymbolTable == &Sec ||
          (Obj.SymbolTable && Obj.SymbolTable->getStrTab() == &Sec))
        return false;

      // Remove everything else.
      return true;
    };
  }

  if (!Config.KeepSection.empty()) {
    RemovePred = [&Config, RemovePred](const SectionBase &Sec) {
      // Explicitly keep these sections regardless of previous removes.
      if (is_contained(Config.KeepSection, Sec.Name))
        return false;
      // Otherwise defer to RemovePred.
      return RemovePred(Sec);
    };
  }

  // This has to be the last predicate assignment.
  // If the option --keep-symbol has been specified
  // and at least one of those symbols is present
  // (equivalently, the updated symbol table is not empty)
  // the symbol table and the string table should not be removed.
  if ((!Config.SymbolsToKeep.empty() || Config.KeepFileSymbols) &&
      Obj.SymbolTable && !Obj.SymbolTable->empty()) {
    RemovePred = [&Obj, RemovePred](const SectionBase &Sec) {
      if (&Sec == Obj.SymbolTable || &Sec == Obj.SymbolTable->getStrTab())
        return false;
      return RemovePred(Sec);
    };
  }

  if (Config.CompressionType != DebugCompressionType::None)
    replaceDebugSections(Obj, RemovePred, isCompressable,
                         [&Config, &Obj](const SectionBase *S) {
                           return &Obj.addSection<CompressedSection>(
                               *S, Config.CompressionType);
                         });
  else if (Config.DecompressDebugSections)
    replaceDebugSections(
        Obj, RemovePred,
        [](const SectionBase &S) { return isa<CompressedSection>(&S); },
        [&Obj](const SectionBase *S) {
          auto CS = cast<CompressedSection>(S);
          return &Obj.addSection<DecompressedSection>(*CS);
        });

  return Obj.removeSections(Config.AllowBrokenLinks, RemovePred);
}

// This function handles the high level operations of GNU objcopy including
// handling command line options. It's important to outline certain properties
// we expect to hold of the command line operations. Any operation that "keeps"
// should keep regardless of a remove. Additionally any removal should respect
// any previous removals. Lastly whether or not something is removed shouldn't
// depend a) on the order the options occur in or b) on some opaque priority
// system. The only priority is that keeps/copies overrule removes.
static Error handleArgs(const CopyConfig &Config, Object &Obj,
                        const Reader &Reader, ElfType OutputElfType) {

  if (!Config.SplitDWO.empty())
    if (Error E =
            splitDWOToFile(Config, Reader, Config.SplitDWO, OutputElfType))
      return E;

  if (Config.OutputArch) {
    Obj.Machine = Config.OutputArch.getValue().EMachine;
    Obj.OSABI = Config.OutputArch.getValue().OSABI;
  }

  // It is important to remove the sections first. For example, we want to
  // remove the relocation sections before removing the symbols. That allows
  // us to avoid reporting the inappropriate errors about removing symbols
  // named in relocations.
  if (Error E = replaceAndRemoveSections(Config, Obj))
    return E;

  if (Error E = updateAndRemoveSymbols(Config, Obj))
    return E;

  if (!Config.SectionsToRename.empty()) {
    for (auto &Sec : Obj.sections()) {
      const auto Iter = Config.SectionsToRename.find(Sec.Name);
      if (Iter != Config.SectionsToRename.end()) {
        const SectionRename &SR = Iter->second;
        Sec.Name = SR.NewName;
        if (SR.NewFlags.hasValue())
          setSectionFlagsAndType(Sec, SR.NewFlags.getValue());
      }
    }
  }

  if (!Config.SetSectionFlags.empty()) {
    for (auto &Sec : Obj.sections()) {
      const auto Iter = Config.SetSectionFlags.find(Sec.Name);
      if (Iter != Config.SetSectionFlags.end()) {
        const SectionFlagsUpdate &SFU = Iter->second;
        setSectionFlagsAndType(Sec, SFU.NewFlags);
      }
    }
  }

  for (const auto &Flag : Config.AddSection) {
    std::pair<StringRef, StringRef> SecPair = Flag.split("=");
    StringRef SecName = SecPair.first;
    StringRef File = SecPair.second;
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
        MemoryBuffer::getFile(File);
    if (!BufOrErr)
      return createFileError(File, errorCodeToError(BufOrErr.getError()));
    std::unique_ptr<MemoryBuffer> Buf = std::move(*BufOrErr);
    ArrayRef<uint8_t> Data(
        reinterpret_cast<const uint8_t *>(Buf->getBufferStart()),
        Buf->getBufferSize());
    OwnedDataSection &NewSection =
        Obj.addSection<OwnedDataSection>(SecName, Data);
    if (SecName.startswith(".note") && SecName != ".note.GNU-stack")
      NewSection.Type = SHT_NOTE;
  }

  for (const auto &Flag : Config.DumpSection) {
    std::pair<StringRef, StringRef> SecPair = Flag.split("=");
    StringRef SecName = SecPair.first;
    StringRef File = SecPair.second;
    if (Error E = dumpSectionToFile(SecName, File, Obj))
      return createFileError(Config.InputFilename, std::move(E));
  }

  if (!Config.AddGnuDebugLink.empty())
    Obj.addSection<GnuDebugLinkSection>(Config.AddGnuDebugLink);

  for (const NewSymbolInfo &SI : Config.SymbolsToAdd) {
    SectionBase *Sec = Obj.findSection(SI.SectionName);
    uint64_t Value = Sec ? Sec->Addr + SI.Value : SI.Value;
    Obj.SymbolTable->addSymbol(
        SI.SymbolName, SI.Bind, SI.Type, Sec, Value, SI.Visibility,
        Sec ? (uint16_t)SYMBOL_SIMPLE_INDEX : (uint16_t)SHN_ABS, 0);
  }

  if (Config.EntryExpr)
    Obj.Entry = Config.EntryExpr(Obj.Entry);
  return Error::success();
}

Error executeObjcopyOnRawBinary(const CopyConfig &Config, MemoryBuffer &In,
                                Buffer &Out) {
  BinaryReader Reader(Config.BinaryArch, &In);
  std::unique_ptr<Object> Obj = Reader.create();

  // Prefer OutputArch (-O<format>) if set, otherwise fallback to BinaryArch
  // (-B<arch>).
  const ElfType OutputElfType = getOutputElfType(
      Config.OutputArch ? Config.OutputArch.getValue() : Config.BinaryArch);
  if (Error E = handleArgs(Config, *Obj, Reader, OutputElfType))
    return E;
  std::unique_ptr<Writer> Writer =
      createWriter(Config, *Obj, Out, OutputElfType);
  if (Error E = Writer->finalize())
    return E;
  return Writer->write();
}

Error executeObjcopyOnBinary(const CopyConfig &Config,
                             object::ELFObjectFileBase &In, Buffer &Out) {
  ELFReader Reader(&In);
  std::unique_ptr<Object> Obj = Reader.create();
  // Prefer OutputArch (-O<format>) if set, otherwise infer it from the input.
  const ElfType OutputElfType =
      Config.OutputArch ? getOutputElfType(Config.OutputArch.getValue())
                        : getOutputElfType(In);
  ArrayRef<uint8_t> BuildIdBytes;

  if (!Config.BuildIdLinkDir.empty()) {
    BuildIdBytes = unwrapOrError(findBuildID(In));
    if (BuildIdBytes.size() < 2)
      return createFileError(
          Config.InputFilename,
          createStringError(object_error::parse_failed,
                            "build ID is smaller than two bytes."));
  }

  if (!Config.BuildIdLinkDir.empty() && Config.BuildIdLinkInput)
    if (Error E =
            linkToBuildIdDir(Config, Config.InputFilename,
                             Config.BuildIdLinkInput.getValue(), BuildIdBytes))
      return E;

  if (Error E = handleArgs(Config, *Obj, Reader, OutputElfType))
    return E;
  std::unique_ptr<Writer> Writer =
      createWriter(Config, *Obj, Out, OutputElfType);
  if (Error E = Writer->finalize())
    return E;
  if (Error E = Writer->write())
    return E;
  if (!Config.BuildIdLinkDir.empty() && Config.BuildIdLinkOutput)
    if (Error E =
            linkToBuildIdDir(Config, Config.OutputFilename,
                             Config.BuildIdLinkOutput.getValue(), BuildIdBytes))
      return E;

  return Error::success();
}

} // end namespace elf
} // end namespace objcopy
} // end namespace llvm
