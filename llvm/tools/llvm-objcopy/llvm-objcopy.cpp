//===- llvm-objcopy.cpp ---------------------------------------------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-objcopy.h"

#include "Object.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/Error.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/WithColor.h"
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

using namespace llvm;
using namespace llvm::objcopy;
using namespace object;
using namespace ELF;

namespace {

enum ObjcopyID {
  OBJCOPY_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OBJCOPY_##ID,
#include "ObjcopyOpts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const OBJCOPY_##NAME[] = VALUE;
#include "ObjcopyOpts.inc"
#undef PREFIX

static const opt::OptTable::Info ObjcopyInfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {OBJCOPY_##PREFIX,                                                           \
   NAME,                                                                       \
   HELPTEXT,                                                                   \
   METAVAR,                                                                    \
   OBJCOPY_##ID,                                                               \
   opt::Option::KIND##Class,                                                   \
   PARAM,                                                                      \
   FLAGS,                                                                      \
   OBJCOPY_##GROUP,                                                            \
   OBJCOPY_##ALIAS,                                                            \
   ALIASARGS,                                                                  \
   VALUES},
#include "ObjcopyOpts.inc"
#undef OPTION
};

class ObjcopyOptTable : public opt::OptTable {
public:
  ObjcopyOptTable() : OptTable(ObjcopyInfoTable, true) {}
};

enum StripID {
  STRIP_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  STRIP_##ID,
#include "StripOpts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const STRIP_##NAME[] = VALUE;
#include "StripOpts.inc"
#undef PREFIX

static const opt::OptTable::Info StripInfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {STRIP_##PREFIX, NAME,       HELPTEXT,                                       \
   METAVAR,        STRIP_##ID, opt::Option::KIND##Class,                       \
   PARAM,          FLAGS,      STRIP_##GROUP,                                  \
   STRIP_##ALIAS,  ALIASARGS,  VALUES},
#include "StripOpts.inc"
#undef OPTION
};

class StripOptTable : public opt::OptTable {
public:
  StripOptTable() : OptTable(StripInfoTable, true) {}
};

struct SectionRename {
  StringRef OriginalName;
  StringRef NewName;
  Optional<uint64_t> NewFlags;
};

struct CopyConfig {
  // Main input/output options
  StringRef InputFilename;
  StringRef InputFormat;
  StringRef OutputFilename;
  StringRef OutputFormat;

  // Only applicable for --input-format=Binary
  MachineInfo BinaryArch;

  // Advanced options
  StringRef AddGnuDebugLink;
  StringRef SplitDWO;
  StringRef SymbolsPrefix;

  // Repeated options
  std::vector<StringRef> AddSection;
  std::vector<StringRef> DumpSection;
  std::vector<StringRef> Keep;
  std::vector<StringRef> OnlyKeep;
  std::vector<StringRef> SymbolsToGlobalize;
  std::vector<StringRef> SymbolsToKeep;
  std::vector<StringRef> SymbolsToLocalize;
  std::vector<StringRef> SymbolsToRemove;
  std::vector<StringRef> SymbolsToWeaken;
  std::vector<StringRef> ToRemove;
  std::vector<std::string> SymbolsToKeepGlobal;

  // Map options
  StringMap<SectionRename> SectionsToRename;
  StringMap<StringRef> SymbolsToRename;

  // Boolean options
  bool DiscardAll = false;
  bool ExtractDWO = false;
  bool KeepFileSymbols = false;
  bool LocalizeHidden = false;
  bool OnlyKeepDebug = false;
  bool PreserveDates = false;
  bool StripAll = false;
  bool StripAllGNU = false;
  bool StripDWO = false;
  bool StripDebug = false;
  bool StripNonAlloc = false;
  bool StripSections = false;
  bool StripUnneeded = false;
  bool Weaken = false;
};

using SectionPred = std::function<bool(const SectionBase &Sec)>;

enum SectionFlag {
  SecNone = 0,
  SecAlloc = 1 << 0,
  SecLoad = 1 << 1,
  SecNoload = 1 << 2,
  SecReadonly = 1 << 3,
  SecDebug = 1 << 4,
  SecCode = 1 << 5,
  SecData = 1 << 6,
  SecRom = 1 << 7,
  SecMerge = 1 << 8,
  SecStrings = 1 << 9,
  SecContents = 1 << 10,
  SecShare = 1 << 11,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ SecShare)
};

} // namespace

namespace llvm {
namespace objcopy {

// The name this program was invoked as.
StringRef ToolName;

LLVM_ATTRIBUTE_NORETURN void error(Twine Message) {
  WithColor::error(errs(), ToolName) << Message << ".\n";
  errs().flush();
  exit(1);
}

LLVM_ATTRIBUTE_NORETURN void reportError(StringRef File, std::error_code EC) {
  assert(EC);
  WithColor::error(errs(), ToolName)
      << "'" << File << "': " << EC.message() << ".\n";
  exit(1);
}

LLVM_ATTRIBUTE_NORETURN void reportError(StringRef File, Error E) {
  assert(E);
  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(std::move(E), OS, "");
  OS.flush();
  WithColor::error(errs(), ToolName) << "'" << File << "': " << Buf;
  exit(1);
}

} // end namespace objcopy
} // end namespace llvm

static SectionFlag parseSectionRenameFlag(StringRef SectionName) {
  return llvm::StringSwitch<SectionFlag>(SectionName)
      .Case("alloc", SectionFlag::SecAlloc)
      .Case("load", SectionFlag::SecLoad)
      .Case("noload", SectionFlag::SecNoload)
      .Case("readonly", SectionFlag::SecReadonly)
      .Case("debug", SectionFlag::SecDebug)
      .Case("code", SectionFlag::SecCode)
      .Case("data", SectionFlag::SecData)
      .Case("rom", SectionFlag::SecRom)
      .Case("merge", SectionFlag::SecMerge)
      .Case("strings", SectionFlag::SecStrings)
      .Case("contents", SectionFlag::SecContents)
      .Case("share", SectionFlag::SecShare)
      .Default(SectionFlag::SecNone);
}

static SectionRename parseRenameSectionValue(StringRef FlagValue) {
  if (!FlagValue.contains('='))
    error("Bad format for --rename-section: missing '='");

  // Initial split: ".foo" = ".bar,f1,f2,..."
  auto Old2New = FlagValue.split('=');
  SectionRename SR;
  SR.OriginalName = Old2New.first;

  // Flags split: ".bar" "f1" "f2" ...
  SmallVector<StringRef, 6> NameAndFlags;
  Old2New.second.split(NameAndFlags, ',');
  SR.NewName = NameAndFlags[0];

  if (NameAndFlags.size() > 1) {
    SectionFlag Flags = SectionFlag::SecNone;
    for (size_t I = 1, Size = NameAndFlags.size(); I < Size; ++I) {
      SectionFlag Flag = parseSectionRenameFlag(NameAndFlags[I]);
      if (Flag == SectionFlag::SecNone)
        error("Unrecognized section flag '" + NameAndFlags[I] +
              "'. Flags supported for GNU compatibility: alloc, load, noload, "
              "readonly, debug, code, data, rom, share, contents, merge, "
              "strings.");
      Flags |= Flag;
    }

    SR.NewFlags = 0;
    if (Flags & SectionFlag::SecAlloc)
      *SR.NewFlags |= ELF::SHF_ALLOC;
    if (!(Flags & SectionFlag::SecReadonly))
      *SR.NewFlags |= ELF::SHF_WRITE;
    if (Flags & SectionFlag::SecCode)
      *SR.NewFlags |= ELF::SHF_EXECINSTR;
    if (Flags & SectionFlag::SecMerge)
      *SR.NewFlags |= ELF::SHF_MERGE;
    if (Flags & SectionFlag::SecStrings)
      *SR.NewFlags |= ELF::SHF_STRINGS;
  }

  return SR;
}

static bool isDebugSection(const SectionBase &Sec) {
  return Sec.Name.startswith(".debug") || Sec.Name.startswith(".zdebug") ||
         Sec.Name == ".gdb_index";
}

static bool isDWOSection(const SectionBase &Sec) {
  return Sec.Name.endswith(".dwo");
}

static bool onlyKeepDWOPred(const Object &Obj, const SectionBase &Sec) {
  // We can't remove the section header string table.
  if (&Sec == Obj.SectionNames)
    return false;
  // Short of keeping the string table we want to keep everything that is a DWO
  // section and remove everything else.
  return !isDWOSection(Sec);
}

static const StringMap<MachineInfo> ArchMap{
    // Name, {EMachine, 64bit, LittleEndian}
    {"aarch64", {EM_AARCH64, true, true}},
    {"arm", {EM_ARM, false, true}},
    {"i386", {EM_386, false, true}},
    {"i386:x86-64", {EM_X86_64, true, true}},
    {"powerpc:common64", {EM_PPC64, true, true}},
    {"sparc", {EM_SPARC, false, true}},
    {"x86-64", {EM_X86_64, true, true}},
};

static const MachineInfo &getMachineInfo(StringRef Arch) {
  auto Iter = ArchMap.find(Arch);
  if (Iter == std::end(ArchMap))
    error("Invalid architecture: '" + Arch + "'");
  return Iter->getValue();
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

static void splitDWOToFile(const CopyConfig &Config, const Reader &Reader,
                           StringRef File, ElfType OutputElfType) {
  auto DWOFile = Reader.create();
  DWOFile->removeSections(
      [&](const SectionBase &Sec) { return onlyKeepDWOPred(*DWOFile, Sec); });
  FileBuffer FB(File);
  auto Writer = createWriter(Config, *DWOFile, FB, OutputElfType);
  Writer->finalize();
  Writer->write();
}

static Error dumpSectionToFile(StringRef SecName, StringRef Filename,
                               Object &Obj) {
  for (auto &Sec : Obj.sections()) {
    if (Sec.Name == SecName) {
      if (Sec.OriginalData.size() == 0)
        return make_error<StringError>("Can't dump section \"" + SecName +
                                           "\": it has no contents",
                                       object_error::parse_failed);
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
  return make_error<StringError>("Section not found",
                                 object_error::parse_failed);
}

// This function handles the high level operations of GNU objcopy including
// handling command line options. It's important to outline certain properties
// we expect to hold of the command line operations. Any operation that "keeps"
// should keep regardless of a remove. Additionally any removal should respect
// any previous removals. Lastly whether or not something is removed shouldn't
// depend a) on the order the options occur in or b) on some opaque priority
// system. The only priority is that keeps/copies overrule removes.
static void handleArgs(const CopyConfig &Config, Object &Obj,
                       const Reader &Reader, ElfType OutputElfType) {

  if (!Config.SplitDWO.empty()) {
    splitDWOToFile(Config, Reader, Config.SplitDWO, OutputElfType);
  }

  // TODO: update or remove symbols only if there is an option that affects
  // them.
  if (Obj.SymbolTable) {
    Obj.SymbolTable->updateSymbols([&](Symbol &Sym) {
      if ((Config.LocalizeHidden &&
           (Sym.Visibility == STV_HIDDEN || Sym.Visibility == STV_INTERNAL)) ||
          (!Config.SymbolsToLocalize.empty() &&
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
          !is_contained(Config.SymbolsToKeepGlobal, Sym.Name))
        Sym.Binding = STB_LOCAL;

      if (!Config.SymbolsToGlobalize.empty() &&
          is_contained(Config.SymbolsToGlobalize, Sym.Name))
        Sym.Binding = STB_GLOBAL;

      if (!Config.SymbolsToWeaken.empty() &&
          is_contained(Config.SymbolsToWeaken, Sym.Name) &&
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
    // symbols are still 'needed' and wich are not.
    if (Config.StripUnneeded) {
      for (auto &Section : Obj.sections())
        Section.markSymbols();
    }

    Obj.removeSymbols([&](const Symbol &Sym) {
      if ((!Config.SymbolsToKeep.empty() &&
           is_contained(Config.SymbolsToKeep, Sym.Name)) ||
          (Config.KeepFileSymbols && Sym.Type == STT_FILE))
        return false;

      if (Config.DiscardAll && Sym.Binding == STB_LOCAL &&
          Sym.getShndx() != SHN_UNDEF && Sym.Type != STT_FILE &&
          Sym.Type != STT_SECTION)
        return true;

      if (Config.StripAll || Config.StripAllGNU)
        return true;

      if (!Config.SymbolsToRemove.empty() &&
          is_contained(Config.SymbolsToRemove, Sym.Name)) {
        return true;
      }

      if (Config.StripUnneeded && !Sym.Referenced &&
          (Sym.Binding == STB_LOCAL || Sym.getShndx() == SHN_UNDEF) &&
          Sym.Type != STT_FILE && Sym.Type != STT_SECTION)
        return true;

      return false;
    });
  }

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
      return RemovePred(Sec) || (Sec.Flags & SHF_ALLOC) == 0;
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
      return (Sec.Flags & SHF_ALLOC) == 0;
    };

  if (Config.StripAll)
    RemovePred = [RemovePred, &Obj](const SectionBase &Sec) {
      if (RemovePred(Sec))
        return true;
      if (&Sec == Obj.SectionNames)
        return false;
      if (Sec.Name.startswith(".gnu.warning"))
        return false;
      return (Sec.Flags & SHF_ALLOC) == 0;
    };

  // Explicit copies:
  if (!Config.OnlyKeep.empty()) {
    RemovePred = [&Config, RemovePred, &Obj](const SectionBase &Sec) {
      // Explicitly keep these sections regardless of previous removes.
      if (is_contained(Config.OnlyKeep, Sec.Name))
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

  if (!Config.Keep.empty()) {
    RemovePred = [Config, RemovePred](const SectionBase &Sec) {
      // Explicitly keep these sections regardless of previous removes.
      if (is_contained(Config.Keep, Sec.Name))
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

  Obj.removeSections(RemovePred);

  if (!Config.SectionsToRename.empty()) {
    for (auto &Sec : Obj.sections()) {
      const auto Iter = Config.SectionsToRename.find(Sec.Name);
      if (Iter != Config.SectionsToRename.end()) {
        const SectionRename &SR = Iter->second;
        Sec.Name = SR.NewName;
        if (SR.NewFlags.hasValue()) {
          // Preserve some flags which should not be dropped when setting flags.
          // Also, preserve anything OS/processor dependant.
          const uint64_t PreserveMask = ELF::SHF_COMPRESSED | ELF::SHF_EXCLUDE |
                                        ELF::SHF_GROUP | ELF::SHF_LINK_ORDER |
                                        ELF::SHF_MASKOS | ELF::SHF_MASKPROC |
                                        ELF::SHF_TLS | ELF::SHF_INFO_LINK;
          Sec.Flags = (Sec.Flags & PreserveMask) |
                      (SR.NewFlags.getValue() & ~PreserveMask);
        }
      }
    }
  }

  if (!Config.AddSection.empty()) {
    for (const auto &Flag : Config.AddSection) {
      auto SecPair = Flag.split("=");
      auto SecName = SecPair.first;
      auto File = SecPair.second;
      auto BufOrErr = MemoryBuffer::getFile(File);
      if (!BufOrErr)
        reportError(File, BufOrErr.getError());
      auto Buf = std::move(*BufOrErr);
      auto BufPtr = reinterpret_cast<const uint8_t *>(Buf->getBufferStart());
      auto BufSize = Buf->getBufferSize();
      Obj.addSection<OwnedDataSection>(SecName,
                                       ArrayRef<uint8_t>(BufPtr, BufSize));
    }
  }

  if (!Config.DumpSection.empty()) {
    for (const auto &Flag : Config.DumpSection) {
      std::pair<StringRef, StringRef> SecPair = Flag.split("=");
      StringRef SecName = SecPair.first;
      StringRef File = SecPair.second;
      if (Error E = dumpSectionToFile(SecName, File, Obj))
        reportError(Config.InputFilename, std::move(E));
    }
  }

  if (!Config.AddGnuDebugLink.empty())
    Obj.addSection<GnuDebugLinkSection>(Config.AddGnuDebugLink);
}

static void executeElfObjcopyOnBinary(const CopyConfig &Config, Reader &Reader,
                                      Buffer &Out, ElfType OutputElfType) {
  std::unique_ptr<Object> Obj = Reader.create();

  handleArgs(Config, *Obj, Reader, OutputElfType);

  std::unique_ptr<Writer> Writer =
      createWriter(Config, *Obj, Out, OutputElfType);
  Writer->finalize();
  Writer->write();
}

// For regular archives this function simply calls llvm::writeArchive,
// For thin archives it writes the archive file itself as well as its members.
static Error deepWriteArchive(StringRef ArcName,
                              ArrayRef<NewArchiveMember> NewMembers,
                              bool WriteSymtab, object::Archive::Kind Kind,
                              bool Deterministic, bool Thin) {
  Error E =
      writeArchive(ArcName, NewMembers, WriteSymtab, Kind, Deterministic, Thin);
  if (!Thin || E)
    return E;
  for (const NewArchiveMember &Member : NewMembers) {
    // Internally, FileBuffer will use the buffer created by
    // FileOutputBuffer::create, for regular files (that is the case for
    // deepWriteArchive) FileOutputBuffer::create will return OnDiskBuffer.
    // OnDiskBuffer uses a temporary file and then renames it. So in reality
    // there is no inefficiency / duplicated in-memory buffers in this case. For
    // now in-memory buffers can not be completely avoided since
    // NewArchiveMember still requires them even though writeArchive does not
    // write them on disk.
    FileBuffer FB(Member.MemberName);
    FB.allocate(Member.Buf->getBufferSize());
    std::copy(Member.Buf->getBufferStart(), Member.Buf->getBufferEnd(),
              FB.getBufferStart());
    if (auto E = FB.commit())
      return E;
  }
  return Error::success();
}

static void executeElfObjcopyOnArchive(const CopyConfig &Config,
                                       const Archive &Ar) {
  std::vector<NewArchiveMember> NewArchiveMembers;
  Error Err = Error::success();
  for (const Archive::Child &Child : Ar.children(Err)) {
    Expected<std::unique_ptr<Binary>> ChildOrErr = Child.getAsBinary();
    if (!ChildOrErr)
      reportError(Ar.getFileName(), ChildOrErr.takeError());
    Binary *Bin = ChildOrErr->get();

    Expected<StringRef> ChildNameOrErr = Child.getName();
    if (!ChildNameOrErr)
      reportError(Ar.getFileName(), ChildNameOrErr.takeError());

    MemBuffer MB(ChildNameOrErr.get());
    ELFReader Reader(Bin);
    executeElfObjcopyOnBinary(Config, Reader, MB, getOutputElfType(*Bin));

    Expected<NewArchiveMember> Member =
        NewArchiveMember::getOldMember(Child, true);
    if (!Member)
      reportError(Ar.getFileName(), Member.takeError());
    Member->Buf = MB.releaseMemoryBuffer();
    Member->MemberName = Member->Buf->getBufferIdentifier();
    NewArchiveMembers.push_back(std::move(*Member));
  }

  if (Err)
    reportError(Config.InputFilename, std::move(Err));
  if (Error E =
          deepWriteArchive(Config.OutputFilename, NewArchiveMembers,
                           Ar.hasSymbolTable(), Ar.kind(), true, Ar.isThin()))
    reportError(Config.OutputFilename, std::move(E));
}

static void restoreDateOnFile(StringRef Filename,
                              const sys::fs::file_status &Stat) {
  int FD;

  if (auto EC =
          sys::fs::openFileForWrite(Filename, FD, sys::fs::CD_OpenExisting))
    reportError(Filename, EC);

  if (auto EC = sys::fs::setLastAccessAndModificationTime(
          FD, Stat.getLastAccessedTime(), Stat.getLastModificationTime()))
    reportError(Filename, EC);

  if (auto EC = sys::Process::SafelyCloseFileDescriptor(FD))
    reportError(Filename, EC);
}

static void executeElfObjcopy(const CopyConfig &Config) {
  sys::fs::file_status Stat;
  if (Config.PreserveDates)
    if (auto EC = sys::fs::status(Config.InputFilename, Stat))
      reportError(Config.InputFilename, EC);

  if (Config.InputFormat == "binary") {
    auto BufOrErr = MemoryBuffer::getFile(Config.InputFilename);
    if (!BufOrErr)
      reportError(Config.InputFilename, BufOrErr.getError());

    FileBuffer FB(Config.OutputFilename);
    BinaryReader Reader(Config.BinaryArch, BufOrErr->get());
    executeElfObjcopyOnBinary(Config, Reader, FB,
                              getOutputElfType(Config.BinaryArch));
  } else {
    Expected<OwningBinary<llvm::object::Binary>> BinaryOrErr =
        createBinary(Config.InputFilename);
    if (!BinaryOrErr)
      reportError(Config.InputFilename, BinaryOrErr.takeError());

    if (Archive *Ar = dyn_cast<Archive>(BinaryOrErr.get().getBinary())) {
      executeElfObjcopyOnArchive(Config, *Ar);
    } else {
      FileBuffer FB(Config.OutputFilename);
      Binary *Bin = BinaryOrErr.get().getBinary();
      ELFReader Reader(Bin);
      executeElfObjcopyOnBinary(Config, Reader, FB, getOutputElfType(*Bin));
    }
  }

  if (Config.PreserveDates) {
    restoreDateOnFile(Config.OutputFilename, Stat);
    if (!Config.SplitDWO.empty())
      restoreDateOnFile(Config.SplitDWO, Stat);
  }
}

static void addGlobalSymbolsFromFile(std::vector<std::string> &Symbols,
                                     StringRef Filename) {
  SmallVector<StringRef, 16> Lines;
  auto BufOrErr = MemoryBuffer::getFile(Filename);
  if (!BufOrErr)
    reportError(Filename, BufOrErr.getError());

  BufOrErr.get()->getBuffer().split(Lines, '\n');
  for (StringRef Line : Lines) {
    // Ignore everything after '#', trim whitespace, and only add the symbol if
    // it's not empty.
    auto TrimmedLine = Line.split('#').first.trim();
    if (!TrimmedLine.empty())
      Symbols.push_back(TrimmedLine.str());
  }
}

// ParseObjcopyOptions returns the config and sets the input arguments. If a
// help flag is set then ParseObjcopyOptions will print the help messege and
// exit.
static CopyConfig parseObjcopyOptions(ArrayRef<const char *> ArgsArr) {
  ObjcopyOptTable T;
  unsigned MissingArgumentIndex, MissingArgumentCount;
  llvm::opt::InputArgList InputArgs =
      T.ParseArgs(ArgsArr, MissingArgumentIndex, MissingArgumentCount);

  if (InputArgs.size() == 0) {
    T.PrintHelp(errs(), "llvm-objcopy <input> [ <output> ]", "objcopy tool");
    exit(1);
  }

  if (InputArgs.hasArg(OBJCOPY_help)) {
    T.PrintHelp(outs(), "llvm-objcopy <input> [ <output> ]", "objcopy tool");
    exit(0);
  }

  SmallVector<const char *, 2> Positional;

  for (auto Arg : InputArgs.filtered(OBJCOPY_UNKNOWN))
    error("unknown argument '" + Arg->getAsString(InputArgs) + "'");

  for (auto Arg : InputArgs.filtered(OBJCOPY_INPUT))
    Positional.push_back(Arg->getValue());

  if (Positional.empty())
    error("No input file specified");

  if (Positional.size() > 2)
    error("Too many positional arguments");

  CopyConfig Config;
  Config.InputFilename = Positional[0];
  Config.OutputFilename = Positional[Positional.size() == 1 ? 0 : 1];
  Config.InputFormat = InputArgs.getLastArgValue(OBJCOPY_input_target);
  Config.OutputFormat = InputArgs.getLastArgValue(OBJCOPY_output_target);
  if (Config.InputFormat == "binary") {
    auto BinaryArch = InputArgs.getLastArgValue(OBJCOPY_binary_architecture);
    if (BinaryArch.empty())
      error("Specified binary input without specifiying an architecture");
    Config.BinaryArch = getMachineInfo(BinaryArch);
  }

  Config.SplitDWO = InputArgs.getLastArgValue(OBJCOPY_split_dwo);
  Config.AddGnuDebugLink = InputArgs.getLastArgValue(OBJCOPY_add_gnu_debuglink);
  Config.SymbolsPrefix = InputArgs.getLastArgValue(OBJCOPY_prefix_symbols);

  for (auto Arg : InputArgs.filtered(OBJCOPY_redefine_symbol)) {
    if (!StringRef(Arg->getValue()).contains('='))
      error("Bad format for --redefine-sym");
    auto Old2New = StringRef(Arg->getValue()).split('=');
    if (!Config.SymbolsToRename.insert(Old2New).second)
      error("Multiple redefinition of symbol " + Old2New.first);
  }

  for (auto Arg : InputArgs.filtered(OBJCOPY_rename_section)) {
    SectionRename SR = parseRenameSectionValue(StringRef(Arg->getValue()));
    if (!Config.SectionsToRename.try_emplace(SR.OriginalName, SR).second)
      error("Multiple renames of section " + SR.OriginalName);
  }

  for (auto Arg : InputArgs.filtered(OBJCOPY_remove_section))
    Config.ToRemove.push_back(Arg->getValue());
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep))
    Config.Keep.push_back(Arg->getValue());
  for (auto Arg : InputArgs.filtered(OBJCOPY_only_keep))
    Config.OnlyKeep.push_back(Arg->getValue());
  for (auto Arg : InputArgs.filtered(OBJCOPY_add_section))
    Config.AddSection.push_back(Arg->getValue());
  for (auto Arg : InputArgs.filtered(OBJCOPY_dump_section))
    Config.DumpSection.push_back(Arg->getValue());
  Config.StripAll = InputArgs.hasArg(OBJCOPY_strip_all);
  Config.StripAllGNU = InputArgs.hasArg(OBJCOPY_strip_all_gnu);
  Config.StripDebug = InputArgs.hasArg(OBJCOPY_strip_debug);
  Config.StripDWO = InputArgs.hasArg(OBJCOPY_strip_dwo);
  Config.StripSections = InputArgs.hasArg(OBJCOPY_strip_sections);
  Config.StripNonAlloc = InputArgs.hasArg(OBJCOPY_strip_non_alloc);
  Config.StripUnneeded = InputArgs.hasArg(OBJCOPY_strip_unneeded);
  Config.ExtractDWO = InputArgs.hasArg(OBJCOPY_extract_dwo);
  Config.LocalizeHidden = InputArgs.hasArg(OBJCOPY_localize_hidden);
  Config.Weaken = InputArgs.hasArg(OBJCOPY_weaken);
  Config.DiscardAll = InputArgs.hasArg(OBJCOPY_discard_all);
  Config.OnlyKeepDebug = InputArgs.hasArg(OBJCOPY_only_keep_debug);
  Config.KeepFileSymbols = InputArgs.hasArg(OBJCOPY_keep_file_symbols);
  for (auto Arg : InputArgs.filtered(OBJCOPY_localize_symbol))
    Config.SymbolsToLocalize.push_back(Arg->getValue());
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_global_symbol))
    Config.SymbolsToKeepGlobal.push_back(Arg->getValue());
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_global_symbols))
    addGlobalSymbolsFromFile(Config.SymbolsToKeepGlobal, Arg->getValue());
  for (auto Arg : InputArgs.filtered(OBJCOPY_globalize_symbol))
    Config.SymbolsToGlobalize.push_back(Arg->getValue());
  for (auto Arg : InputArgs.filtered(OBJCOPY_weaken_symbol))
    Config.SymbolsToWeaken.push_back(Arg->getValue());
  for (auto Arg : InputArgs.filtered(OBJCOPY_strip_symbol))
    Config.SymbolsToRemove.push_back(Arg->getValue());
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_symbol))
    Config.SymbolsToKeep.push_back(Arg->getValue());

  Config.PreserveDates = InputArgs.hasArg(OBJCOPY_preserve_dates);

  return Config;
}

// ParseStripOptions returns the config and sets the input arguments. If a
// help flag is set then ParseStripOptions will print the help messege and
// exit.
static CopyConfig parseStripOptions(ArrayRef<const char *> ArgsArr) {
  StripOptTable T;
  unsigned MissingArgumentIndex, MissingArgumentCount;
  llvm::opt::InputArgList InputArgs =
      T.ParseArgs(ArgsArr, MissingArgumentIndex, MissingArgumentCount);

  if (InputArgs.size() == 0) {
    T.PrintHelp(errs(), "llvm-strip", "strip tool");
    exit(1);
  }

  if (InputArgs.hasArg(STRIP_help)) {
    T.PrintHelp(outs(), "llvm-strip", "strip tool");
    exit(0);
  }

  SmallVector<const char *, 2> Positional;
  for (auto Arg : InputArgs.filtered(STRIP_UNKNOWN))
    error("unknown argument '" + Arg->getAsString(InputArgs) + "'");
  for (auto Arg : InputArgs.filtered(STRIP_INPUT))
    Positional.push_back(Arg->getValue());

  if (Positional.empty())
    error("No input file specified");

  if (Positional.size() > 1)
    error("Support for multiple input files is not implemented yet");

  CopyConfig Config;
  Config.InputFilename = Positional[0];
  Config.OutputFilename =
      InputArgs.getLastArgValue(STRIP_output, Positional[0]);

  Config.StripDebug = InputArgs.hasArg(STRIP_strip_debug);

  Config.DiscardAll = InputArgs.hasArg(STRIP_discard_all);
  Config.StripUnneeded = InputArgs.hasArg(STRIP_strip_unneeded);
  Config.StripAll = InputArgs.hasArg(STRIP_strip_all);

  if (!Config.StripDebug && !Config.StripUnneeded && !Config.DiscardAll)
    Config.StripAll = true;

  for (auto Arg : InputArgs.filtered(STRIP_remove_section))
    Config.ToRemove.push_back(Arg->getValue());

  for (auto Arg : InputArgs.filtered(STRIP_keep_symbol))
    Config.SymbolsToKeep.push_back(Arg->getValue());

  Config.PreserveDates = InputArgs.hasArg(STRIP_preserve_dates);

  return Config;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  ToolName = argv[0];
  CopyConfig Config;
  if (sys::path::stem(ToolName).endswith_lower("strip"))
    Config = parseStripOptions(makeArrayRef(argv + 1, argc));
  else
    Config = parseObjcopyOptions(makeArrayRef(argv + 1, argc));
  executeElfObjcopy(Config);
}
