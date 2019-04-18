//===- CopyConfig.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CopyConfig.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/StringSaver.h"
#include <memory>

namespace llvm {
namespace objcopy {

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
  ObjcopyOptTable() : OptTable(ObjcopyInfoTable) {}
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
  StripOptTable() : OptTable(StripInfoTable) {}
};

} // namespace

static SectionFlag parseSectionRenameFlag(StringRef SectionName) {
  return llvm::StringSwitch<SectionFlag>(SectionName)
      .CaseLower("alloc", SectionFlag::SecAlloc)
      .CaseLower("load", SectionFlag::SecLoad)
      .CaseLower("noload", SectionFlag::SecNoload)
      .CaseLower("readonly", SectionFlag::SecReadonly)
      .CaseLower("debug", SectionFlag::SecDebug)
      .CaseLower("code", SectionFlag::SecCode)
      .CaseLower("data", SectionFlag::SecData)
      .CaseLower("rom", SectionFlag::SecRom)
      .CaseLower("merge", SectionFlag::SecMerge)
      .CaseLower("strings", SectionFlag::SecStrings)
      .CaseLower("contents", SectionFlag::SecContents)
      .CaseLower("share", SectionFlag::SecShare)
      .Default(SectionFlag::SecNone);
}

static Expected<SectionFlag>
parseSectionFlagSet(ArrayRef<StringRef> SectionFlags) {
  SectionFlag ParsedFlags = SectionFlag::SecNone;
  for (StringRef Flag : SectionFlags) {
    SectionFlag ParsedFlag = parseSectionRenameFlag(Flag);
    if (ParsedFlag == SectionFlag::SecNone)
      return createStringError(
          errc::invalid_argument,
          "Unrecognized section flag '%s'. Flags supported for GNU "
          "compatibility: alloc, load, noload, readonly, debug, code, data, "
          "rom, share, contents, merge, strings",
          Flag.str().c_str());
    ParsedFlags |= ParsedFlag;
  }

  return ParsedFlags;
}

static Expected<SectionRename> parseRenameSectionValue(StringRef FlagValue) {
  if (!FlagValue.contains('='))
    return createStringError(errc::invalid_argument,
                             "Bad format for --rename-section: missing '='");

  // Initial split: ".foo" = ".bar,f1,f2,..."
  auto Old2New = FlagValue.split('=');
  SectionRename SR;
  SR.OriginalName = Old2New.first;

  // Flags split: ".bar" "f1" "f2" ...
  SmallVector<StringRef, 6> NameAndFlags;
  Old2New.second.split(NameAndFlags, ',');
  SR.NewName = NameAndFlags[0];

  if (NameAndFlags.size() > 1) {
    Expected<SectionFlag> ParsedFlagSet =
        parseSectionFlagSet(makeArrayRef(NameAndFlags).drop_front());
    if (!ParsedFlagSet)
      return ParsedFlagSet.takeError();
    SR.NewFlags = *ParsedFlagSet;
  }

  return SR;
}

static Expected<SectionFlagsUpdate>
parseSetSectionFlagValue(StringRef FlagValue) {
  if (!StringRef(FlagValue).contains('='))
    return createStringError(errc::invalid_argument,
                             "Bad format for --set-section-flags: missing '='");

  // Initial split: ".foo" = "f1,f2,..."
  auto Section2Flags = StringRef(FlagValue).split('=');
  SectionFlagsUpdate SFU;
  SFU.Name = Section2Flags.first;

  // Flags split: "f1" "f2" ...
  SmallVector<StringRef, 6> SectionFlags;
  Section2Flags.second.split(SectionFlags, ',');
  Expected<SectionFlag> ParsedFlagSet = parseSectionFlagSet(SectionFlags);
  if (!ParsedFlagSet)
    return ParsedFlagSet.takeError();
  SFU.NewFlags = *ParsedFlagSet;

  return SFU;
}

static Expected<NewSymbolInfo> parseNewSymbolInfo(StringRef FlagValue) {
  // Parse value given with --add-symbol option and create the
  // new symbol if possible. The value format for --add-symbol is:
  //
  // <name>=[<section>:]<value>[,<flags>]
  //
  // where:
  // <name> - symbol name, can be empty string
  // <section> - optional section name. If not given ABS symbol is created
  // <value> - symbol value, can be decimal or hexadecimal number prefixed
  //           with 0x.
  // <flags> - optional flags affecting symbol type, binding or visibility:
  //           The following are currently supported:
  //
  //           global, local, weak, default, hidden, file, section, object,
  //           indirect-function.
  //
  //           The following flags are ignored and provided for GNU
  //           compatibility only:
  //
  //           warning, debug, constructor, indirect, synthetic,
  //           unique-object, before=<symbol>.
  NewSymbolInfo SI;
  StringRef Value;
  std::tie(SI.SymbolName, Value) = FlagValue.split('=');
  if (Value.empty())
    return createStringError(
        errc::invalid_argument,
        "bad format for --add-symbol, missing '=' after '%s'",
        SI.SymbolName.str().c_str());

  if (Value.contains(':')) {
    std::tie(SI.SectionName, Value) = Value.split(':');
    if (SI.SectionName.empty() || Value.empty())
      return createStringError(
          errc::invalid_argument,
          "bad format for --add-symbol, missing section name or symbol value");
  }

  SmallVector<StringRef, 6> Flags;
  Value.split(Flags, ',');
  if (Flags[0].getAsInteger(0, SI.Value))
    return createStringError(errc::invalid_argument, "bad symbol value: '%s'",
                             Flags[0].str().c_str());

  using Functor = std::function<void(void)>;
  SmallVector<StringRef, 6> UnsupportedFlags;
  for (size_t I = 1, NumFlags = Flags.size(); I < NumFlags; ++I)
    static_cast<Functor>(
        StringSwitch<Functor>(Flags[I])
            .CaseLower("global", [&SI] { SI.Bind = ELF::STB_GLOBAL; })
            .CaseLower("local", [&SI] { SI.Bind = ELF::STB_LOCAL; })
            .CaseLower("weak", [&SI] { SI.Bind = ELF::STB_WEAK; })
            .CaseLower("default", [&SI] { SI.Visibility = ELF::STV_DEFAULT; })
            .CaseLower("hidden", [&SI] { SI.Visibility = ELF::STV_HIDDEN; })
            .CaseLower("file", [&SI] { SI.Type = ELF::STT_FILE; })
            .CaseLower("section", [&SI] { SI.Type = ELF::STT_SECTION; })
            .CaseLower("object", [&SI] { SI.Type = ELF::STT_OBJECT; })
            .CaseLower("function", [&SI] { SI.Type = ELF::STT_FUNC; })
            .CaseLower("indirect-function",
                       [&SI] { SI.Type = ELF::STT_GNU_IFUNC; })
            .CaseLower("debug", [] {})
            .CaseLower("constructor", [] {})
            .CaseLower("warning", [] {})
            .CaseLower("indirect", [] {})
            .CaseLower("synthetic", [] {})
            .CaseLower("unique-object", [] {})
            .StartsWithLower("before", [] {})
            .Default([&] { UnsupportedFlags.push_back(Flags[I]); }))();
  if (!UnsupportedFlags.empty())
    return createStringError(errc::invalid_argument,
                             "unsupported flag%s for --add-symbol: '%s'",
                             UnsupportedFlags.size() > 1 ? "s" : "",
                             join(UnsupportedFlags, "', '").c_str());
  return SI;
}

static const StringMap<MachineInfo> ArchMap{
    // Name, {EMachine, 64bit, LittleEndian}
    {"aarch64", {ELF::EM_AARCH64, true, true}},
    {"arm", {ELF::EM_ARM, false, true}},
    {"i386", {ELF::EM_386, false, true}},
    {"i386:x86-64", {ELF::EM_X86_64, true, true}},
    {"powerpc:common64", {ELF::EM_PPC64, true, true}},
    {"sparc", {ELF::EM_SPARC, false, true}},
    {"x86-64", {ELF::EM_X86_64, true, true}},
};

static Expected<const MachineInfo &> getMachineInfo(StringRef Arch) {
  auto Iter = ArchMap.find(Arch);
  if (Iter == std::end(ArchMap))
    return createStringError(errc::invalid_argument,
                             "Invalid architecture: '%s'", Arch.str().c_str());
  return Iter->getValue();
}

// FIXME: consolidate with the bfd parsing used by lld.
static const StringMap<MachineInfo> OutputFormatMap{
    // Name, {EMachine, 64bit, LittleEndian}
    {"elf32-i386", {ELF::EM_386, false, true}},
    {"elf32-iamcu", {ELF::EM_IAMCU, false, true}},
    {"elf32-littlearm", {ELF::EM_ARM, false, true}},
    {"elf32-x86-64", {ELF::EM_X86_64, false, true}},
    {"elf64-aarch64", {ELF::EM_AARCH64, true, true}},
    {"elf64-littleaarch64", {ELF::EM_AARCH64, true, true}},
    {"elf32-powerpc", {ELF::EM_PPC, false, false}},
    {"elf32-powerpcle", {ELF::EM_PPC, false, true}},
    {"elf64-powerpc", {ELF::EM_PPC64, true, false}},
    {"elf64-powerpcle", {ELF::EM_PPC64, true, true}},
    {"elf64-x86-64", {ELF::EM_X86_64, true, true}},
    {"elf32-tradbigmips", {ELF::EM_MIPS, false, false}},
    {"elf32-bigmips", {ELF::EM_MIPS, false, false}},
    {"elf32-ntradbigmips", {ELF::EM_MIPS, false, false}},
    {"elf32-tradlittlemips", {ELF::EM_MIPS, false, true}},
    {"elf32-ntradlittlemips", {ELF::EM_MIPS, false, true}},
    {"elf64-tradbigmips", {ELF::EM_MIPS, true, false}},
    {"elf64-tradlittlemips", {ELF::EM_MIPS, true, true}},
};

static Expected<MachineInfo> getOutputFormatMachineInfo(StringRef Format) {
  StringRef OriginalFormat = Format;
  bool IsFreeBSD = Format.consume_back("-freebsd");
  auto Iter = OutputFormatMap.find(Format);
  if (Iter == std::end(OutputFormatMap))
    return createStringError(errc::invalid_argument,
                             "Invalid output format: '%s'",
                             OriginalFormat.str().c_str());
  MachineInfo MI = Iter->getValue();
  if (IsFreeBSD)
    MI.OSABI = ELF::ELFOSABI_FREEBSD;
  return {MI};
}

static Error addSymbolsFromFile(std::vector<NameOrRegex> &Symbols,
                                BumpPtrAllocator &Alloc, StringRef Filename,
                                bool UseRegex) {
  StringSaver Saver(Alloc);
  SmallVector<StringRef, 16> Lines;
  auto BufOrErr = MemoryBuffer::getFile(Filename);
  if (!BufOrErr)
    return createFileError(Filename, BufOrErr.getError());

  BufOrErr.get()->getBuffer().split(Lines, '\n');
  for (StringRef Line : Lines) {
    // Ignore everything after '#', trim whitespace, and only add the symbol if
    // it's not empty.
    auto TrimmedLine = Line.split('#').first.trim();
    if (!TrimmedLine.empty())
      Symbols.emplace_back(Saver.save(TrimmedLine), UseRegex);
  }

  return Error::success();
}

NameOrRegex::NameOrRegex(StringRef Pattern, bool IsRegex) {
  if (!IsRegex) {
    Name = Pattern;
    return;
  }

  SmallVector<char, 32> Data;
  R = std::make_shared<Regex>(
      ("^" + Pattern.ltrim('^').rtrim('$') + "$").toStringRef(Data));
}

static Error addSymbolsToRenameFromFile(StringMap<StringRef> &SymbolsToRename,
                                        BumpPtrAllocator &Alloc,
                                        StringRef Filename) {
  StringSaver Saver(Alloc);
  SmallVector<StringRef, 16> Lines;
  auto BufOrErr = MemoryBuffer::getFile(Filename);
  if (!BufOrErr)
    return createFileError(Filename, BufOrErr.getError());

  BufOrErr.get()->getBuffer().split(Lines, '\n');
  size_t NumLines = Lines.size();
  for (size_t LineNo = 0; LineNo < NumLines; ++LineNo) {
    StringRef TrimmedLine = Lines[LineNo].split('#').first.trim();
    if (TrimmedLine.empty())
      continue;

    std::pair<StringRef, StringRef> Pair = Saver.save(TrimmedLine).split(' ');
    StringRef NewName = Pair.second.trim();
    if (NewName.empty())
      return createStringError(errc::invalid_argument,
                               "%s:%zu: missing new symbol name",
                               Filename.str().c_str(), LineNo + 1);
    SymbolsToRename.insert({Pair.first, NewName});
  }
  return Error::success();
}

template <class T> static ErrorOr<T> getAsInteger(StringRef Val) {
  T Result;
  if (Val.getAsInteger(0, Result))
    return errc::invalid_argument;
  return Result;
}

// ParseObjcopyOptions returns the config and sets the input arguments. If a
// help flag is set then ParseObjcopyOptions will print the help messege and
// exit.
Expected<DriverConfig> parseObjcopyOptions(ArrayRef<const char *> ArgsArr) {
  DriverConfig DC;
  ObjcopyOptTable T;
  unsigned MissingArgumentIndex, MissingArgumentCount;
  llvm::opt::InputArgList InputArgs =
      T.ParseArgs(ArgsArr, MissingArgumentIndex, MissingArgumentCount);

  if (InputArgs.size() == 0) {
    T.PrintHelp(errs(), "llvm-objcopy input [output]", "objcopy tool");
    exit(1);
  }

  if (InputArgs.hasArg(OBJCOPY_help)) {
    T.PrintHelp(outs(), "llvm-objcopy input [output]", "objcopy tool");
    exit(0);
  }

  if (InputArgs.hasArg(OBJCOPY_version)) {
    outs() << "llvm-objcopy, compatible with GNU objcopy\n";
    cl::PrintVersionMessage();
    exit(0);
  }

  SmallVector<const char *, 2> Positional;

  for (auto Arg : InputArgs.filtered(OBJCOPY_UNKNOWN))
    return createStringError(errc::invalid_argument, "unknown argument '%s'",
                             Arg->getAsString(InputArgs).c_str());

  for (auto Arg : InputArgs.filtered(OBJCOPY_INPUT))
    Positional.push_back(Arg->getValue());

  if (Positional.empty())
    return createStringError(errc::invalid_argument, "No input file specified");

  if (Positional.size() > 2)
    return createStringError(errc::invalid_argument,
                             "Too many positional arguments");

  CopyConfig Config;
  Config.InputFilename = Positional[0];
  Config.OutputFilename = Positional[Positional.size() == 1 ? 0 : 1];
  if (InputArgs.hasArg(OBJCOPY_target) &&
      (InputArgs.hasArg(OBJCOPY_input_target) ||
       InputArgs.hasArg(OBJCOPY_output_target)))
    return createStringError(
        errc::invalid_argument,
        "--target cannot be used with --input-target or --output-target");

  bool UseRegex = InputArgs.hasArg(OBJCOPY_regex);
  if (InputArgs.hasArg(OBJCOPY_target)) {
    Config.InputFormat = InputArgs.getLastArgValue(OBJCOPY_target);
    Config.OutputFormat = InputArgs.getLastArgValue(OBJCOPY_target);
  } else {
    Config.InputFormat = InputArgs.getLastArgValue(OBJCOPY_input_target);
    Config.OutputFormat = InputArgs.getLastArgValue(OBJCOPY_output_target);
  }
  if (Config.InputFormat == "binary") {
    auto BinaryArch = InputArgs.getLastArgValue(OBJCOPY_binary_architecture);
    if (BinaryArch.empty())
      return createStringError(
          errc::invalid_argument,
          "Specified binary input without specifiying an architecture");
    Expected<const MachineInfo &> MI = getMachineInfo(BinaryArch);
    if (!MI)
      return MI.takeError();
    Config.BinaryArch = *MI;
  }
  if (!Config.OutputFormat.empty() && Config.OutputFormat != "binary") {
    Expected<MachineInfo> MI = getOutputFormatMachineInfo(Config.OutputFormat);
    if (!MI)
      return MI.takeError();
    Config.OutputArch = *MI;
  }

  if (auto Arg = InputArgs.getLastArg(OBJCOPY_compress_debug_sections,
                                      OBJCOPY_compress_debug_sections_eq)) {
    Config.CompressionType = DebugCompressionType::Z;

    if (Arg->getOption().getID() == OBJCOPY_compress_debug_sections_eq) {
      Config.CompressionType =
          StringSwitch<DebugCompressionType>(
              InputArgs.getLastArgValue(OBJCOPY_compress_debug_sections_eq))
              .Case("zlib-gnu", DebugCompressionType::GNU)
              .Case("zlib", DebugCompressionType::Z)
              .Default(DebugCompressionType::None);
      if (Config.CompressionType == DebugCompressionType::None)
        return createStringError(
            errc::invalid_argument,
            "Invalid or unsupported --compress-debug-sections format: %s",
            InputArgs.getLastArgValue(OBJCOPY_compress_debug_sections_eq)
                .str()
                .c_str());
    }
    if (!zlib::isAvailable())
      return createStringError(
          errc::invalid_argument,
          "LLVM was not compiled with LLVM_ENABLE_ZLIB: can not compress");
  }

  Config.AddGnuDebugLink = InputArgs.getLastArgValue(OBJCOPY_add_gnu_debuglink);
  Config.BuildIdLinkDir = InputArgs.getLastArgValue(OBJCOPY_build_id_link_dir);
  if (InputArgs.hasArg(OBJCOPY_build_id_link_input))
    Config.BuildIdLinkInput =
        InputArgs.getLastArgValue(OBJCOPY_build_id_link_input);
  if (InputArgs.hasArg(OBJCOPY_build_id_link_output))
    Config.BuildIdLinkOutput =
        InputArgs.getLastArgValue(OBJCOPY_build_id_link_output);
  Config.SplitDWO = InputArgs.getLastArgValue(OBJCOPY_split_dwo);
  Config.SymbolsPrefix = InputArgs.getLastArgValue(OBJCOPY_prefix_symbols);

  for (auto Arg : InputArgs.filtered(OBJCOPY_redefine_symbol)) {
    if (!StringRef(Arg->getValue()).contains('='))
      return createStringError(errc::invalid_argument,
                               "Bad format for --redefine-sym");
    auto Old2New = StringRef(Arg->getValue()).split('=');
    if (!Config.SymbolsToRename.insert(Old2New).second)
      return createStringError(errc::invalid_argument,
                               "Multiple redefinition of symbol %s",
                               Old2New.first.str().c_str());
  }

  for (auto Arg : InputArgs.filtered(OBJCOPY_redefine_symbols))
    if (Error E = addSymbolsToRenameFromFile(Config.SymbolsToRename, DC.Alloc,
                                             Arg->getValue()))
      return std::move(E);

  for (auto Arg : InputArgs.filtered(OBJCOPY_rename_section)) {
    Expected<SectionRename> SR =
        parseRenameSectionValue(StringRef(Arg->getValue()));
    if (!SR)
      return SR.takeError();
    if (!Config.SectionsToRename.try_emplace(SR->OriginalName, *SR).second)
      return createStringError(errc::invalid_argument,
                               "Multiple renames of section %s",
                               SR->OriginalName.str().c_str());
  }
  for (auto Arg : InputArgs.filtered(OBJCOPY_set_section_flags)) {
    Expected<SectionFlagsUpdate> SFU =
        parseSetSectionFlagValue(Arg->getValue());
    if (!SFU)
      return SFU.takeError();
    if (!Config.SetSectionFlags.try_emplace(SFU->Name, *SFU).second)
      return createStringError(
          errc::invalid_argument,
          "--set-section-flags set multiple times for section %s",
          SFU->Name.str().c_str());
  }
  // Prohibit combinations of --set-section-flags when the section name is used
  // by --rename-section, either as a source or a destination.
  for (const auto &E : Config.SectionsToRename) {
    const SectionRename &SR = E.second;
    if (Config.SetSectionFlags.count(SR.OriginalName))
      return createStringError(
          errc::invalid_argument,
          "--set-section-flags=%s conflicts with --rename-section=%s=%s",
          SR.OriginalName.str().c_str(), SR.OriginalName.str().c_str(),
          SR.NewName.str().c_str());
    if (Config.SetSectionFlags.count(SR.NewName))
      return createStringError(
          errc::invalid_argument,
          "--set-section-flags=%s conflicts with --rename-section=%s=%s",
          SR.NewName.str().c_str(), SR.OriginalName.str().c_str(),
          SR.NewName.str().c_str());
  }

  for (auto Arg : InputArgs.filtered(OBJCOPY_remove_section))
    Config.ToRemove.emplace_back(Arg->getValue(), UseRegex);
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_section))
    Config.KeepSection.emplace_back(Arg->getValue(), UseRegex);
  for (auto Arg : InputArgs.filtered(OBJCOPY_only_section))
    Config.OnlySection.emplace_back(Arg->getValue(), UseRegex);
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
  if (InputArgs.hasArg(OBJCOPY_discard_all, OBJCOPY_discard_locals))
    Config.DiscardMode =
        InputArgs.hasFlag(OBJCOPY_discard_all, OBJCOPY_discard_locals)
            ? DiscardType::All
            : DiscardType::Locals;
  Config.OnlyKeepDebug = InputArgs.hasArg(OBJCOPY_only_keep_debug);
  Config.KeepFileSymbols = InputArgs.hasArg(OBJCOPY_keep_file_symbols);
  Config.DecompressDebugSections =
      InputArgs.hasArg(OBJCOPY_decompress_debug_sections);
  for (auto Arg : InputArgs.filtered(OBJCOPY_localize_symbol))
    Config.SymbolsToLocalize.emplace_back(Arg->getValue(), UseRegex);
  for (auto Arg : InputArgs.filtered(OBJCOPY_localize_symbols))
    if (Error E = addSymbolsFromFile(Config.SymbolsToLocalize, DC.Alloc,
                                     Arg->getValue(), UseRegex))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_global_symbol))
    Config.SymbolsToKeepGlobal.emplace_back(Arg->getValue(), UseRegex);
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_global_symbols))
    if (Error E = addSymbolsFromFile(Config.SymbolsToKeepGlobal, DC.Alloc,
                                     Arg->getValue(), UseRegex))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_globalize_symbol))
    Config.SymbolsToGlobalize.emplace_back(Arg->getValue(), UseRegex);
  for (auto Arg : InputArgs.filtered(OBJCOPY_globalize_symbols))
    if (Error E = addSymbolsFromFile(Config.SymbolsToGlobalize, DC.Alloc,
                                     Arg->getValue(), UseRegex))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_weaken_symbol))
    Config.SymbolsToWeaken.emplace_back(Arg->getValue(), UseRegex);
  for (auto Arg : InputArgs.filtered(OBJCOPY_weaken_symbols))
    if (Error E = addSymbolsFromFile(Config.SymbolsToWeaken, DC.Alloc,
                                     Arg->getValue(), UseRegex))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_strip_symbol))
    Config.SymbolsToRemove.emplace_back(Arg->getValue(), UseRegex);
  for (auto Arg : InputArgs.filtered(OBJCOPY_strip_symbols))
    if (Error E = addSymbolsFromFile(Config.SymbolsToRemove, DC.Alloc,
                                     Arg->getValue(), UseRegex))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_strip_unneeded_symbol))
    Config.UnneededSymbolsToRemove.emplace_back(Arg->getValue(), UseRegex);
  for (auto Arg : InputArgs.filtered(OBJCOPY_strip_unneeded_symbols))
    if (Error E = addSymbolsFromFile(Config.UnneededSymbolsToRemove, DC.Alloc,
                                     Arg->getValue(), UseRegex))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_symbol))
    Config.SymbolsToKeep.emplace_back(Arg->getValue(), UseRegex);
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_symbols))
    if (Error E = addSymbolsFromFile(Config.SymbolsToKeep, DC.Alloc,
                                     Arg->getValue(), UseRegex))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_add_symbol)) {
    Expected<NewSymbolInfo> NSI = parseNewSymbolInfo(Arg->getValue());
    if (!NSI)
      return NSI.takeError();
    Config.SymbolsToAdd.push_back(*NSI);
  }

  Config.AllowBrokenLinks = InputArgs.hasArg(OBJCOPY_allow_broken_links);

  Config.DeterministicArchives = InputArgs.hasFlag(
      OBJCOPY_enable_deterministic_archives,
      OBJCOPY_disable_deterministic_archives, /*default=*/true);

  Config.PreserveDates = InputArgs.hasArg(OBJCOPY_preserve_dates);

  for (auto Arg : InputArgs)
    if (Arg->getOption().matches(OBJCOPY_set_start)) {
      auto EAddr = getAsInteger<uint64_t>(Arg->getValue());
      if (!EAddr)
        return createStringError(
            EAddr.getError(), "bad entry point address: '%s'", Arg->getValue());

      Config.EntryExpr = [EAddr](uint64_t) { return *EAddr; };
    } else if (Arg->getOption().matches(OBJCOPY_change_start)) {
      auto EIncr = getAsInteger<int64_t>(Arg->getValue());
      if (!EIncr)
        return createStringError(EIncr.getError(),
                                 "bad entry point increment: '%s'",
                                 Arg->getValue());
      auto Expr = Config.EntryExpr ? std::move(Config.EntryExpr)
                                   : [](uint64_t A) { return A; };
      Config.EntryExpr = [Expr, EIncr](uint64_t EAddr) {
        return Expr(EAddr) + *EIncr;
      };
    }

  if (Config.DecompressDebugSections &&
      Config.CompressionType != DebugCompressionType::None) {
    return createStringError(
        errc::invalid_argument,
        "Cannot specify --compress-debug-sections at the same time as "
        "--decompress-debug-sections at the same time");
  }

  if (Config.DecompressDebugSections && !zlib::isAvailable())
    return createStringError(
        errc::invalid_argument,
        "LLVM was not compiled with LLVM_ENABLE_ZLIB: cannot decompress");

  DC.CopyConfigs.push_back(std::move(Config));
  return std::move(DC);
}

// ParseStripOptions returns the config and sets the input arguments. If a
// help flag is set then ParseStripOptions will print the help messege and
// exit.
Expected<DriverConfig> parseStripOptions(ArrayRef<const char *> ArgsArr) {
  StripOptTable T;
  unsigned MissingArgumentIndex, MissingArgumentCount;
  llvm::opt::InputArgList InputArgs =
      T.ParseArgs(ArgsArr, MissingArgumentIndex, MissingArgumentCount);

  if (InputArgs.size() == 0) {
    T.PrintHelp(errs(), "llvm-strip [options] file...", "strip tool");
    exit(1);
  }

  if (InputArgs.hasArg(STRIP_help)) {
    T.PrintHelp(outs(), "llvm-strip [options] file...", "strip tool");
    exit(0);
  }

  if (InputArgs.hasArg(STRIP_version)) {
    outs() << "llvm-strip, compatible with GNU strip\n";
    cl::PrintVersionMessage();
    exit(0);
  }

  SmallVector<const char *, 2> Positional;
  for (auto Arg : InputArgs.filtered(STRIP_UNKNOWN))
    return createStringError(errc::invalid_argument, "unknown argument '%s'",
                             Arg->getAsString(InputArgs).c_str());
  for (auto Arg : InputArgs.filtered(STRIP_INPUT))
    Positional.push_back(Arg->getValue());

  if (Positional.empty())
    return createStringError(errc::invalid_argument, "No input file specified");

  if (Positional.size() > 1 && InputArgs.hasArg(STRIP_output))
    return createStringError(
        errc::invalid_argument,
        "Multiple input files cannot be used in combination with -o");

  CopyConfig Config;
  bool UseRegexp = InputArgs.hasArg(STRIP_regex);
  Config.AllowBrokenLinks = InputArgs.hasArg(STRIP_allow_broken_links);
  Config.StripDebug = InputArgs.hasArg(STRIP_strip_debug);

  if (InputArgs.hasArg(STRIP_discard_all, STRIP_discard_locals))
    Config.DiscardMode =
        InputArgs.hasFlag(STRIP_discard_all, STRIP_discard_locals)
            ? DiscardType::All
            : DiscardType::Locals;
  Config.StripUnneeded = InputArgs.hasArg(STRIP_strip_unneeded);
  Config.StripAll = InputArgs.hasArg(STRIP_strip_all);
  Config.StripAllGNU = InputArgs.hasArg(STRIP_strip_all_gnu);
  Config.OnlyKeepDebug = InputArgs.hasArg(STRIP_only_keep_debug);
  Config.KeepFileSymbols = InputArgs.hasArg(STRIP_keep_file_symbols);

  for (auto Arg : InputArgs.filtered(STRIP_keep_section))
    Config.KeepSection.emplace_back(Arg->getValue(), UseRegexp);

  for (auto Arg : InputArgs.filtered(STRIP_remove_section))
    Config.ToRemove.emplace_back(Arg->getValue(), UseRegexp);

  for (auto Arg : InputArgs.filtered(STRIP_strip_symbol))
    Config.SymbolsToRemove.emplace_back(Arg->getValue(), UseRegexp);

  for (auto Arg : InputArgs.filtered(STRIP_keep_symbol))
    Config.SymbolsToKeep.emplace_back(Arg->getValue(), UseRegexp);

  if (!Config.StripDebug && !Config.StripUnneeded &&
      Config.DiscardMode == DiscardType::None && !Config.StripAllGNU && Config.SymbolsToRemove.empty())
    Config.StripAll = true;

  Config.DeterministicArchives =
      InputArgs.hasFlag(STRIP_enable_deterministic_archives,
                        STRIP_disable_deterministic_archives, /*default=*/true);

  Config.PreserveDates = InputArgs.hasArg(STRIP_preserve_dates);

  DriverConfig DC;
  if (Positional.size() == 1) {
    Config.InputFilename = Positional[0];
    Config.OutputFilename =
        InputArgs.getLastArgValue(STRIP_output, Positional[0]);
    DC.CopyConfigs.push_back(std::move(Config));
  } else {
    for (const char *Filename : Positional) {
      Config.InputFilename = Filename;
      Config.OutputFilename = Filename;
      DC.CopyConfigs.push_back(Config);
    }
  }

  return std::move(DC);
}

} // namespace objcopy
} // namespace llvm
