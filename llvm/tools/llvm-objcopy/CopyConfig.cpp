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
#include "llvm/ADT/StringSet.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CRC.h"
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

enum InstallNameToolID {
  INSTALL_NAME_TOOL_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  INSTALL_NAME_TOOL_##ID,
#include "InstallNameToolOpts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE)                                                    \
  const char *const INSTALL_NAME_TOOL_##NAME[] = VALUE;
#include "InstallNameToolOpts.inc"
#undef PREFIX

static const opt::OptTable::Info InstallNameToolInfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {INSTALL_NAME_TOOL_##PREFIX,                                                 \
   NAME,                                                                       \
   HELPTEXT,                                                                   \
   METAVAR,                                                                    \
   INSTALL_NAME_TOOL_##ID,                                                     \
   opt::Option::KIND##Class,                                                   \
   PARAM,                                                                      \
   FLAGS,                                                                      \
   INSTALL_NAME_TOOL_##GROUP,                                                  \
   INSTALL_NAME_TOOL_##ALIAS,                                                  \
   ALIASARGS,                                                                  \
   VALUES},
#include "InstallNameToolOpts.inc"
#undef OPTION
};

class InstallNameToolOptTable : public opt::OptTable {
public:
  InstallNameToolOptTable() : OptTable(InstallNameToolInfoTable) {}
};

enum BitcodeStripID {
  BITCODE_STRIP_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  BITCODE_STRIP_##ID,
#include "BitcodeStripOpts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const BITCODE_STRIP_##NAME[] = VALUE;
#include "BitcodeStripOpts.inc"
#undef PREFIX

static const opt::OptTable::Info BitcodeStripInfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {BITCODE_STRIP_##PREFIX,                                                     \
   NAME,                                                                       \
   HELPTEXT,                                                                   \
   METAVAR,                                                                    \
   BITCODE_STRIP_##ID,                                                         \
   opt::Option::KIND##Class,                                                   \
   PARAM,                                                                      \
   FLAGS,                                                                      \
   BITCODE_STRIP_##GROUP,                                                      \
   BITCODE_STRIP_##ALIAS,                                                      \
   ALIASARGS,                                                                  \
   VALUES},
#include "BitcodeStripOpts.inc"
#undef OPTION
};

class BitcodeStripOptTable : public opt::OptTable {
public:
  BitcodeStripOptTable() : OptTable(BitcodeStripInfoTable) {}
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
      .CaseLower("exclude", SectionFlag::SecExclude)
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
          "unrecognized section flag '%s'. Flags supported for GNU "
          "compatibility: alloc, load, noload, readonly, exclude, debug, "
          "code, data, rom, share, contents, merge, strings",
          Flag.str().c_str());
    ParsedFlags |= ParsedFlag;
  }

  return ParsedFlags;
}

static Expected<SectionRename> parseRenameSectionValue(StringRef FlagValue) {
  if (!FlagValue.contains('='))
    return createStringError(errc::invalid_argument,
                             "bad format for --rename-section: missing '='");

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

static Expected<std::pair<StringRef, uint64_t>>
parseSetSectionAlignment(StringRef FlagValue) {
  if (!FlagValue.contains('='))
    return createStringError(
        errc::invalid_argument,
        "bad format for --set-section-alignment: missing '='");
  auto Split = StringRef(FlagValue).split('=');
  if (Split.first.empty())
    return createStringError(
        errc::invalid_argument,
        "bad format for --set-section-alignment: missing section name");
  uint64_t NewAlign;
  if (Split.second.getAsInteger(0, NewAlign))
    return createStringError(errc::invalid_argument,
                             "invalid alignment for --set-section-alignment: '%s'",
                             Split.second.str().c_str());
  return std::make_pair(Split.first, NewAlign);
}

static Expected<SectionFlagsUpdate>
parseSetSectionFlagValue(StringRef FlagValue) {
  if (!StringRef(FlagValue).contains('='))
    return createStringError(errc::invalid_argument,
                             "bad format for --set-section-flags: missing '='");

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

struct TargetInfo {
  FileFormat Format;
  MachineInfo Machine;
};

// FIXME: consolidate with the bfd parsing used by lld.
static const StringMap<MachineInfo> TargetMap{
    // Name, {EMachine, 64bit, LittleEndian}
    // x86
    {"elf32-i386", {ELF::EM_386, false, true}},
    {"elf32-x86-64", {ELF::EM_X86_64, false, true}},
    {"elf64-x86-64", {ELF::EM_X86_64, true, true}},
    // Intel MCU
    {"elf32-iamcu", {ELF::EM_IAMCU, false, true}},
    // ARM
    {"elf32-littlearm", {ELF::EM_ARM, false, true}},
    // ARM AArch64
    {"elf64-aarch64", {ELF::EM_AARCH64, true, true}},
    {"elf64-littleaarch64", {ELF::EM_AARCH64, true, true}},
    // RISC-V
    {"elf32-littleriscv", {ELF::EM_RISCV, false, true}},
    {"elf64-littleriscv", {ELF::EM_RISCV, true, true}},
    // PowerPC
    {"elf32-powerpc", {ELF::EM_PPC, false, false}},
    {"elf32-powerpcle", {ELF::EM_PPC, false, true}},
    {"elf64-powerpc", {ELF::EM_PPC64, true, false}},
    {"elf64-powerpcle", {ELF::EM_PPC64, true, true}},
    // MIPS
    {"elf32-bigmips", {ELF::EM_MIPS, false, false}},
    {"elf32-ntradbigmips", {ELF::EM_MIPS, false, false}},
    {"elf32-ntradlittlemips", {ELF::EM_MIPS, false, true}},
    {"elf32-tradbigmips", {ELF::EM_MIPS, false, false}},
    {"elf32-tradlittlemips", {ELF::EM_MIPS, false, true}},
    {"elf64-tradbigmips", {ELF::EM_MIPS, true, false}},
    {"elf64-tradlittlemips", {ELF::EM_MIPS, true, true}},
    // SPARC
    {"elf32-sparc", {ELF::EM_SPARC, false, false}},
    {"elf32-sparcel", {ELF::EM_SPARC, false, true}},
    {"elf32-hexagon", {ELF::EM_HEXAGON, false, true}},
};

static Expected<TargetInfo>
getOutputTargetInfoByTargetName(StringRef TargetName) {
  StringRef OriginalTargetName = TargetName;
  bool IsFreeBSD = TargetName.consume_back("-freebsd");
  auto Iter = TargetMap.find(TargetName);
  if (Iter == std::end(TargetMap))
    return createStringError(errc::invalid_argument,
                             "invalid output format: '%s'",
                             OriginalTargetName.str().c_str());
  MachineInfo MI = Iter->getValue();
  if (IsFreeBSD)
    MI.OSABI = ELF::ELFOSABI_FREEBSD;

  FileFormat Format;
  if (TargetName.startswith("elf"))
    Format = FileFormat::ELF;
  else
    // This should never happen because `TargetName` is valid (it certainly
    // exists in the TargetMap).
    llvm_unreachable("unknown target prefix");

  return {TargetInfo{Format, MI}};
}

static Error
addSymbolsFromFile(NameMatcher &Symbols, BumpPtrAllocator &Alloc,
                   StringRef Filename, MatchStyle MS,
                   llvm::function_ref<Error(Error)> ErrorCallback) {
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
      if (Error E = Symbols.addMatcher(NameOrPattern::create(
              Saver.save(TrimmedLine), MS, ErrorCallback)))
        return E;
  }

  return Error::success();
}

Expected<NameOrPattern>
NameOrPattern::create(StringRef Pattern, MatchStyle MS,
                      llvm::function_ref<Error(Error)> ErrorCallback) {
  switch (MS) {
  case MatchStyle::Literal:
    return NameOrPattern(Pattern);
  case MatchStyle::Wildcard: {
    SmallVector<char, 32> Data;
    bool IsPositiveMatch = true;
    if (Pattern[0] == '!') {
      IsPositiveMatch = false;
      Pattern = Pattern.drop_front();
    }
    Expected<GlobPattern> GlobOrErr = GlobPattern::create(Pattern);

    // If we couldn't create it as a glob, report the error, but try again with
    // a literal if the error reporting is non-fatal.
    if (!GlobOrErr) {
      if (Error E = ErrorCallback(GlobOrErr.takeError()))
        return std::move(E);
      return create(Pattern, MatchStyle::Literal, ErrorCallback);
    }

    return NameOrPattern(std::make_shared<GlobPattern>(*GlobOrErr),
                         IsPositiveMatch);
  }
  case MatchStyle::Regex: {
    SmallVector<char, 32> Data;
    return NameOrPattern(std::make_shared<Regex>(
        ("^" + Pattern.ltrim('^').rtrim('$') + "$").toStringRef(Data)));
  }
  }
  llvm_unreachable("Unhandled llvm.objcopy.MatchStyle enum");
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

namespace {

enum class ToolType { Objcopy, Strip, InstallNameTool, BitcodeStrip };

} // anonymous namespace

static void printHelp(const opt::OptTable &OptTable, raw_ostream &OS,
                      ToolType Tool) {
  StringRef HelpText, ToolName;
  switch (Tool) {
  case ToolType::Objcopy:
    ToolName = "llvm-objcopy";
    HelpText = " [options] input [output]";
    break;
  case ToolType::Strip:
    ToolName = "llvm-strip";
    HelpText = " [options] inputs...";
    break;
  case ToolType::InstallNameTool:
    ToolName = "llvm-install-name-tool";
    HelpText = " [options] input";
    break;
  case ToolType::BitcodeStrip:
    ToolName = "llvm-bitcode-strip";
    HelpText = " [options] input";
    break;
  }
  OptTable.PrintHelp(OS, (ToolName + HelpText).str().c_str(),
                     (ToolName + " tool").str().c_str());
  // TODO: Replace this with libOption call once it adds extrahelp support.
  // The CommandLine library has a cl::extrahelp class to support this,
  // but libOption does not have that yet.
  OS << "\nPass @FILE as argument to read options from FILE.\n";
}

// ParseObjcopyOptions returns the config and sets the input arguments. If a
// help flag is set then ParseObjcopyOptions will print the help messege and
// exit.
Expected<DriverConfig>
parseObjcopyOptions(ArrayRef<const char *> ArgsArr,
                    llvm::function_ref<Error(Error)> ErrorCallback) {
  DriverConfig DC;
  ObjcopyOptTable T;
  unsigned MissingArgumentIndex, MissingArgumentCount;
  llvm::opt::InputArgList InputArgs =
      T.ParseArgs(ArgsArr, MissingArgumentIndex, MissingArgumentCount);

  if (InputArgs.size() == 0) {
    printHelp(T, errs(), ToolType::Objcopy);
    exit(1);
  }

  if (InputArgs.hasArg(OBJCOPY_help)) {
    printHelp(T, outs(), ToolType::Objcopy);
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
    return createStringError(errc::invalid_argument, "no input file specified");

  if (Positional.size() > 2)
    return createStringError(errc::invalid_argument,
                             "too many positional arguments");

  CopyConfig Config;
  Config.InputFilename = Positional[0];
  Config.OutputFilename = Positional[Positional.size() == 1 ? 0 : 1];
  if (InputArgs.hasArg(OBJCOPY_target) &&
      (InputArgs.hasArg(OBJCOPY_input_target) ||
       InputArgs.hasArg(OBJCOPY_output_target)))
    return createStringError(
        errc::invalid_argument,
        "--target cannot be used with --input-target or --output-target");

  if (InputArgs.hasArg(OBJCOPY_regex) && InputArgs.hasArg(OBJCOPY_wildcard))
    return createStringError(errc::invalid_argument,
                             "--regex and --wildcard are incompatible");

  MatchStyle SectionMatchStyle = InputArgs.hasArg(OBJCOPY_regex)
                                     ? MatchStyle::Regex
                                     : MatchStyle::Wildcard;
  MatchStyle SymbolMatchStyle = InputArgs.hasArg(OBJCOPY_regex)
                                    ? MatchStyle::Regex
                                    : InputArgs.hasArg(OBJCOPY_wildcard)
                                          ? MatchStyle::Wildcard
                                          : MatchStyle::Literal;
  StringRef InputFormat, OutputFormat;
  if (InputArgs.hasArg(OBJCOPY_target)) {
    InputFormat = InputArgs.getLastArgValue(OBJCOPY_target);
    OutputFormat = InputArgs.getLastArgValue(OBJCOPY_target);
  } else {
    InputFormat = InputArgs.getLastArgValue(OBJCOPY_input_target);
    OutputFormat = InputArgs.getLastArgValue(OBJCOPY_output_target);
  }

  // FIXME:  Currently, we ignore the target for non-binary/ihex formats
  // explicitly specified by -I option (e.g. -Ielf32-x86-64) and guess the
  // format by llvm::object::createBinary regardless of the option value.
  Config.InputFormat = StringSwitch<FileFormat>(InputFormat)
                           .Case("binary", FileFormat::Binary)
                           .Case("ihex", FileFormat::IHex)
                           .Default(FileFormat::Unspecified);

  if (InputArgs.hasArg(OBJCOPY_new_symbol_visibility))
    Config.NewSymbolVisibility =
        InputArgs.getLastArgValue(OBJCOPY_new_symbol_visibility);

  Config.OutputFormat = StringSwitch<FileFormat>(OutputFormat)
                            .Case("binary", FileFormat::Binary)
                            .Case("ihex", FileFormat::IHex)
                            .Default(FileFormat::Unspecified);
  if (Config.OutputFormat == FileFormat::Unspecified) {
    if (OutputFormat.empty()) {
      Config.OutputFormat = Config.InputFormat;
    } else {
      Expected<TargetInfo> Target =
          getOutputTargetInfoByTargetName(OutputFormat);
      if (!Target)
        return Target.takeError();
      Config.OutputFormat = Target->Format;
      Config.OutputArch = Target->Machine;
    }
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
            "invalid or unsupported --compress-debug-sections format: %s",
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
  // The gnu_debuglink's target is expected to not change or else its CRC would
  // become invalidated and get rejected. We can avoid recalculating the
  // checksum for every target file inside an archive by precomputing the CRC
  // here. This prevents a significant amount of I/O.
  if (!Config.AddGnuDebugLink.empty()) {
    auto DebugOrErr = MemoryBuffer::getFile(Config.AddGnuDebugLink);
    if (!DebugOrErr)
      return createFileError(Config.AddGnuDebugLink, DebugOrErr.getError());
    auto Debug = std::move(*DebugOrErr);
    Config.GnuDebugLinkCRC32 =
        llvm::crc32(arrayRefFromStringRef(Debug->getBuffer()));
  }
  Config.BuildIdLinkDir = InputArgs.getLastArgValue(OBJCOPY_build_id_link_dir);
  if (InputArgs.hasArg(OBJCOPY_build_id_link_input))
    Config.BuildIdLinkInput =
        InputArgs.getLastArgValue(OBJCOPY_build_id_link_input);
  if (InputArgs.hasArg(OBJCOPY_build_id_link_output))
    Config.BuildIdLinkOutput =
        InputArgs.getLastArgValue(OBJCOPY_build_id_link_output);
  Config.SplitDWO = InputArgs.getLastArgValue(OBJCOPY_split_dwo);
  Config.SymbolsPrefix = InputArgs.getLastArgValue(OBJCOPY_prefix_symbols);
  Config.AllocSectionsPrefix =
      InputArgs.getLastArgValue(OBJCOPY_prefix_alloc_sections);
  if (auto Arg = InputArgs.getLastArg(OBJCOPY_extract_partition))
    Config.ExtractPartition = Arg->getValue();

  for (auto Arg : InputArgs.filtered(OBJCOPY_redefine_symbol)) {
    if (!StringRef(Arg->getValue()).contains('='))
      return createStringError(errc::invalid_argument,
                               "bad format for --redefine-sym");
    auto Old2New = StringRef(Arg->getValue()).split('=');
    if (!Config.SymbolsToRename.insert(Old2New).second)
      return createStringError(errc::invalid_argument,
                               "multiple redefinition of symbol '%s'",
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
                               "multiple renames of section '%s'",
                               SR->OriginalName.str().c_str());
  }
  for (auto Arg : InputArgs.filtered(OBJCOPY_set_section_alignment)) {
    Expected<std::pair<StringRef, uint64_t>> NameAndAlign =
        parseSetSectionAlignment(Arg->getValue());
    if (!NameAndAlign)
      return NameAndAlign.takeError();
    Config.SetSectionAlignment[NameAndAlign->first] = NameAndAlign->second;
  }
  for (auto Arg : InputArgs.filtered(OBJCOPY_set_section_flags)) {
    Expected<SectionFlagsUpdate> SFU =
        parseSetSectionFlagValue(Arg->getValue());
    if (!SFU)
      return SFU.takeError();
    if (!Config.SetSectionFlags.try_emplace(SFU->Name, *SFU).second)
      return createStringError(
          errc::invalid_argument,
          "--set-section-flags set multiple times for section '%s'",
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
    if (Error E = Config.ToRemove.addMatcher(NameOrPattern::create(
            Arg->getValue(), SectionMatchStyle, ErrorCallback)))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_section))
    if (Error E = Config.KeepSection.addMatcher(NameOrPattern::create(
            Arg->getValue(), SectionMatchStyle, ErrorCallback)))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_only_section))
    if (Error E = Config.OnlySection.addMatcher(NameOrPattern::create(
            Arg->getValue(), SectionMatchStyle, ErrorCallback)))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_add_section)) {
    StringRef ArgValue(Arg->getValue());
    if (!ArgValue.contains('='))
      return createStringError(errc::invalid_argument,
                               "bad format for --add-section: missing '='");
    if (ArgValue.split("=").second.empty())
      return createStringError(
          errc::invalid_argument,
          "bad format for --add-section: missing file name");
    Config.AddSection.push_back(ArgValue);
  }
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
  Config.ExtractMainPartition =
      InputArgs.hasArg(OBJCOPY_extract_main_partition);
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
  if (Config.DiscardMode == DiscardType::All) {
    Config.StripDebug = true;
    Config.KeepFileSymbols = true;
  }
  for (auto Arg : InputArgs.filtered(OBJCOPY_localize_symbol))
    if (Error E = Config.SymbolsToLocalize.addMatcher(NameOrPattern::create(
            Arg->getValue(), SymbolMatchStyle, ErrorCallback)))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_localize_symbols))
    if (Error E = addSymbolsFromFile(Config.SymbolsToLocalize, DC.Alloc,
                                     Arg->getValue(), SymbolMatchStyle,
                                     ErrorCallback))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_global_symbol))
    if (Error E = Config.SymbolsToKeepGlobal.addMatcher(NameOrPattern::create(
            Arg->getValue(), SymbolMatchStyle, ErrorCallback)))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_global_symbols))
    if (Error E = addSymbolsFromFile(Config.SymbolsToKeepGlobal, DC.Alloc,
                                     Arg->getValue(), SymbolMatchStyle,
                                     ErrorCallback))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_globalize_symbol))
    if (Error E = Config.SymbolsToGlobalize.addMatcher(NameOrPattern::create(
            Arg->getValue(), SymbolMatchStyle, ErrorCallback)))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_globalize_symbols))
    if (Error E = addSymbolsFromFile(Config.SymbolsToGlobalize, DC.Alloc,
                                     Arg->getValue(), SymbolMatchStyle,
                                     ErrorCallback))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_weaken_symbol))
    if (Error E = Config.SymbolsToWeaken.addMatcher(NameOrPattern::create(
            Arg->getValue(), SymbolMatchStyle, ErrorCallback)))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_weaken_symbols))
    if (Error E = addSymbolsFromFile(Config.SymbolsToWeaken, DC.Alloc,
                                     Arg->getValue(), SymbolMatchStyle,
                                     ErrorCallback))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_strip_symbol))
    if (Error E = Config.SymbolsToRemove.addMatcher(NameOrPattern::create(
            Arg->getValue(), SymbolMatchStyle, ErrorCallback)))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_strip_symbols))
    if (Error E = addSymbolsFromFile(Config.SymbolsToRemove, DC.Alloc,
                                     Arg->getValue(), SymbolMatchStyle,
                                     ErrorCallback))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_strip_unneeded_symbol))
    if (Error E =
            Config.UnneededSymbolsToRemove.addMatcher(NameOrPattern::create(
                Arg->getValue(), SymbolMatchStyle, ErrorCallback)))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_strip_unneeded_symbols))
    if (Error E = addSymbolsFromFile(Config.UnneededSymbolsToRemove, DC.Alloc,
                                     Arg->getValue(), SymbolMatchStyle,
                                     ErrorCallback))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_symbol))
    if (Error E = Config.SymbolsToKeep.addMatcher(NameOrPattern::create(
            Arg->getValue(), SymbolMatchStyle, ErrorCallback)))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_keep_symbols))
    if (Error E =
            addSymbolsFromFile(Config.SymbolsToKeep, DC.Alloc, Arg->getValue(),
                               SymbolMatchStyle, ErrorCallback))
      return std::move(E);
  for (auto Arg : InputArgs.filtered(OBJCOPY_add_symbol))
    Config.SymbolsToAdd.push_back(Arg->getValue());

  Config.AllowBrokenLinks = InputArgs.hasArg(OBJCOPY_allow_broken_links);

  Config.DeterministicArchives = InputArgs.hasFlag(
      OBJCOPY_enable_deterministic_archives,
      OBJCOPY_disable_deterministic_archives, /*default=*/true);

  Config.PreserveDates = InputArgs.hasArg(OBJCOPY_preserve_dates);

  if (Config.PreserveDates &&
      (Config.OutputFilename == "-" || Config.InputFilename == "-"))
    return createStringError(errc::invalid_argument,
                             "--preserve-dates requires a file");

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
        "cannot specify both --compress-debug-sections and "
        "--decompress-debug-sections");
  }

  if (Config.DecompressDebugSections && !zlib::isAvailable())
    return createStringError(
        errc::invalid_argument,
        "LLVM was not compiled with LLVM_ENABLE_ZLIB: cannot decompress");

  if (Config.ExtractPartition && Config.ExtractMainPartition)
    return createStringError(errc::invalid_argument,
                             "cannot specify --extract-partition together with "
                             "--extract-main-partition");

  DC.CopyConfigs.push_back(std::move(Config));
  return std::move(DC);
}

// ParseInstallNameToolOptions returns the config and sets the input arguments.
// If a help flag is set then ParseInstallNameToolOptions will print the help
// messege and exit.
Expected<DriverConfig>
parseInstallNameToolOptions(ArrayRef<const char *> ArgsArr) {
  DriverConfig DC;
  CopyConfig Config;
  InstallNameToolOptTable T;
  unsigned MissingArgumentIndex, MissingArgumentCount;
  llvm::opt::InputArgList InputArgs =
      T.ParseArgs(ArgsArr, MissingArgumentIndex, MissingArgumentCount);

  if (MissingArgumentCount)
    return createStringError(
        errc::invalid_argument,
        "missing argument to " +
            StringRef(InputArgs.getArgString(MissingArgumentIndex)) +
            " option");

  if (InputArgs.size() == 0) {
    printHelp(T, errs(), ToolType::InstallNameTool);
    exit(1);
  }

  if (InputArgs.hasArg(INSTALL_NAME_TOOL_help)) {
    printHelp(T, outs(), ToolType::InstallNameTool);
    exit(0);
  }

  if (InputArgs.hasArg(INSTALL_NAME_TOOL_version)) {
    outs() << "llvm-install-name-tool, compatible with cctools "
              "install_name_tool\n";
    cl::PrintVersionMessage();
    exit(0);
  }

  for (auto Arg : InputArgs.filtered(INSTALL_NAME_TOOL_add_rpath))
    Config.RPathToAdd.push_back(Arg->getValue());

  for (auto *Arg : InputArgs.filtered(INSTALL_NAME_TOOL_prepend_rpath))
    Config.RPathToPrepend.push_back(Arg->getValue());

  for (auto Arg : InputArgs.filtered(INSTALL_NAME_TOOL_delete_rpath)) {
    StringRef RPath = Arg->getValue();

    // Cannot add and delete the same rpath at the same time.
    if (is_contained(Config.RPathToAdd, RPath))
      return createStringError(
          errc::invalid_argument,
          "cannot specify both -add_rpath '%s' and -delete_rpath '%s'",
          RPath.str().c_str(), RPath.str().c_str());
    if (is_contained(Config.RPathToPrepend, RPath))
      return createStringError(
          errc::invalid_argument,
          "cannot specify both -prepend_rpath '%s' and -delete_rpath '%s'",
          RPath.str().c_str(), RPath.str().c_str());

    Config.RPathsToRemove.insert(RPath);
  }

  for (auto *Arg : InputArgs.filtered(INSTALL_NAME_TOOL_rpath)) {
    StringRef Old = Arg->getValue(0);
    StringRef New = Arg->getValue(1);

    auto Match = [=](StringRef RPath) { return RPath == Old || RPath == New; };

    // Cannot specify duplicate -rpath entries
    auto It1 = find_if(
        Config.RPathsToUpdate,
        [&Match](const DenseMap<StringRef, StringRef>::value_type &OldNew) {
          return Match(OldNew.getFirst()) || Match(OldNew.getSecond());
        });
    if (It1 != Config.RPathsToUpdate.end())
      return createStringError(errc::invalid_argument,
                               "cannot specify both -rpath '" +
                                   It1->getFirst() + "' '" + It1->getSecond() +
                                   "' and -rpath '" + Old + "' '" + New + "'");

    // Cannot specify the same rpath under both -delete_rpath and -rpath
    auto It2 = find_if(Config.RPathsToRemove, Match);
    if (It2 != Config.RPathsToRemove.end())
      return createStringError(errc::invalid_argument,
                               "cannot specify both -delete_rpath '" + *It2 +
                                   "' and -rpath '" + Old + "' '" + New + "'");

    // Cannot specify the same rpath under both -add_rpath and -rpath
    auto It3 = find_if(Config.RPathToAdd, Match);
    if (It3 != Config.RPathToAdd.end())
      return createStringError(errc::invalid_argument,
                               "cannot specify both -add_rpath '" + *It3 +
                                   "' and -rpath '" + Old + "' '" + New + "'");

    // Cannot specify the same rpath under both -prepend_rpath and -rpath.
    auto It4 = find_if(Config.RPathToPrepend, Match);
    if (It4 != Config.RPathToPrepend.end())
      return createStringError(errc::invalid_argument,
                               "cannot specify both -prepend_rpath '" + *It4 +
                                   "' and -rpath '" + Old + "' '" + New + "'");

    Config.RPathsToUpdate.insert({Old, New});
  }

  if (auto *Arg = InputArgs.getLastArg(INSTALL_NAME_TOOL_id)) {
    Config.SharedLibId = Arg->getValue();
    if (Config.SharedLibId->empty())
      return createStringError(errc::invalid_argument,
                               "cannot specify an empty id");
  }

  for (auto *Arg : InputArgs.filtered(INSTALL_NAME_TOOL_change))
    Config.InstallNamesToUpdate.insert({Arg->getValue(0), Arg->getValue(1)});

  Config.RemoveAllRpaths =
      InputArgs.hasArg(INSTALL_NAME_TOOL_delete_all_rpaths);

  SmallVector<StringRef, 2> Positional;
  for (auto Arg : InputArgs.filtered(INSTALL_NAME_TOOL_UNKNOWN))
    return createStringError(errc::invalid_argument, "unknown argument '%s'",
                             Arg->getAsString(InputArgs).c_str());
  for (auto Arg : InputArgs.filtered(INSTALL_NAME_TOOL_INPUT))
    Positional.push_back(Arg->getValue());
  if (Positional.empty())
    return createStringError(errc::invalid_argument, "no input file specified");
  if (Positional.size() > 1)
    return createStringError(
        errc::invalid_argument,
        "llvm-install-name-tool expects a single input file");
  Config.InputFilename = Positional[0];
  Config.OutputFilename = Positional[0];

  DC.CopyConfigs.push_back(std::move(Config));
  return std::move(DC);
}

Expected<DriverConfig>
parseBitcodeStripOptions(ArrayRef<const char *> ArgsArr) {
  DriverConfig DC;
  CopyConfig Config;
  BitcodeStripOptTable T;
  unsigned MissingArgumentIndex, MissingArgumentCount;
  opt::InputArgList InputArgs =
      T.ParseArgs(ArgsArr, MissingArgumentIndex, MissingArgumentCount);

  if (InputArgs.size() == 0) {
    printHelp(T, errs(), ToolType::BitcodeStrip);
    exit(1);
  }

  if (InputArgs.hasArg(BITCODE_STRIP_help)) {
    printHelp(T, outs(), ToolType::BitcodeStrip);
    exit(0);
  }

  if (InputArgs.hasArg(BITCODE_STRIP_version)) {
    outs() << "llvm-bitcode-strip, compatible with cctools "
              "bitcode_strip\n";
    cl::PrintVersionMessage();
    exit(0);
  }

  for (auto *Arg : InputArgs.filtered(BITCODE_STRIP_UNKNOWN))
    return createStringError(errc::invalid_argument, "unknown argument '%s'",
                             Arg->getAsString(InputArgs).c_str());

  SmallVector<StringRef, 2> Positional;
  for (auto *Arg : InputArgs.filtered(BITCODE_STRIP_INPUT))
    Positional.push_back(Arg->getValue());
  if (Positional.size() > 1)
    return createStringError(errc::invalid_argument,
                             "llvm-bitcode-strip expects a single input file");
  assert(!Positional.empty());
  Config.InputFilename = Positional[0];
  Config.OutputFilename = Positional[0];

  DC.CopyConfigs.push_back(std::move(Config));
  return std::move(DC);
}

// ParseStripOptions returns the config and sets the input arguments. If a
// help flag is set then ParseStripOptions will print the help messege and
// exit.
Expected<DriverConfig>
parseStripOptions(ArrayRef<const char *> ArgsArr,
                  llvm::function_ref<Error(Error)> ErrorCallback) {
  StripOptTable T;
  unsigned MissingArgumentIndex, MissingArgumentCount;
  llvm::opt::InputArgList InputArgs =
      T.ParseArgs(ArgsArr, MissingArgumentIndex, MissingArgumentCount);

  if (InputArgs.size() == 0) {
    printHelp(T, errs(), ToolType::Strip);
    exit(1);
  }

  if (InputArgs.hasArg(STRIP_help)) {
    printHelp(T, outs(), ToolType::Strip);
    exit(0);
  }

  if (InputArgs.hasArg(STRIP_version)) {
    outs() << "llvm-strip, compatible with GNU strip\n";
    cl::PrintVersionMessage();
    exit(0);
  }

  SmallVector<StringRef, 2> Positional;
  for (auto Arg : InputArgs.filtered(STRIP_UNKNOWN))
    return createStringError(errc::invalid_argument, "unknown argument '%s'",
                             Arg->getAsString(InputArgs).c_str());
  for (auto Arg : InputArgs.filtered(STRIP_INPUT))
    Positional.push_back(Arg->getValue());

  if (Positional.empty())
    return createStringError(errc::invalid_argument, "no input file specified");

  if (Positional.size() > 1 && InputArgs.hasArg(STRIP_output))
    return createStringError(
        errc::invalid_argument,
        "multiple input files cannot be used in combination with -o");

  CopyConfig Config;

  if (InputArgs.hasArg(STRIP_regex) && InputArgs.hasArg(STRIP_wildcard))
    return createStringError(errc::invalid_argument,
                             "--regex and --wildcard are incompatible");
  MatchStyle SectionMatchStyle =
      InputArgs.hasArg(STRIP_regex) ? MatchStyle::Regex : MatchStyle::Wildcard;
  MatchStyle SymbolMatchStyle = InputArgs.hasArg(STRIP_regex)
                                    ? MatchStyle::Regex
                                    : InputArgs.hasArg(STRIP_wildcard)
                                          ? MatchStyle::Wildcard
                                          : MatchStyle::Literal;
  Config.AllowBrokenLinks = InputArgs.hasArg(STRIP_allow_broken_links);
  Config.StripDebug = InputArgs.hasArg(STRIP_strip_debug);

  if (InputArgs.hasArg(STRIP_discard_all, STRIP_discard_locals))
    Config.DiscardMode =
        InputArgs.hasFlag(STRIP_discard_all, STRIP_discard_locals)
            ? DiscardType::All
            : DiscardType::Locals;
  Config.StripSections = InputArgs.hasArg(STRIP_strip_sections);
  Config.StripUnneeded = InputArgs.hasArg(STRIP_strip_unneeded);
  if (auto Arg = InputArgs.getLastArg(STRIP_strip_all, STRIP_no_strip_all))
    Config.StripAll = Arg->getOption().getID() == STRIP_strip_all;
  Config.StripAllGNU = InputArgs.hasArg(STRIP_strip_all_gnu);
  Config.StripSwiftSymbols = InputArgs.hasArg(STRIP_strip_swift_symbols);
  Config.OnlyKeepDebug = InputArgs.hasArg(STRIP_only_keep_debug);
  Config.KeepFileSymbols = InputArgs.hasArg(STRIP_keep_file_symbols);

  for (auto Arg : InputArgs.filtered(STRIP_keep_section))
    if (Error E = Config.KeepSection.addMatcher(NameOrPattern::create(
            Arg->getValue(), SectionMatchStyle, ErrorCallback)))
      return std::move(E);

  for (auto Arg : InputArgs.filtered(STRIP_remove_section))
    if (Error E = Config.ToRemove.addMatcher(NameOrPattern::create(
            Arg->getValue(), SectionMatchStyle, ErrorCallback)))
      return std::move(E);

  for (auto Arg : InputArgs.filtered(STRIP_strip_symbol))
    if (Error E = Config.SymbolsToRemove.addMatcher(NameOrPattern::create(
            Arg->getValue(), SymbolMatchStyle, ErrorCallback)))
      return std::move(E);

  for (auto Arg : InputArgs.filtered(STRIP_keep_symbol))
    if (Error E = Config.SymbolsToKeep.addMatcher(NameOrPattern::create(
            Arg->getValue(), SymbolMatchStyle, ErrorCallback)))
      return std::move(E);

  if (!InputArgs.hasArg(STRIP_no_strip_all) && !Config.StripDebug &&
      !Config.StripUnneeded && Config.DiscardMode == DiscardType::None &&
      !Config.StripAllGNU && Config.SymbolsToRemove.empty())
    Config.StripAll = true;

  if (Config.DiscardMode == DiscardType::All) {
    Config.StripDebug = true;
    Config.KeepFileSymbols = true;
  }

  Config.DeterministicArchives =
      InputArgs.hasFlag(STRIP_enable_deterministic_archives,
                        STRIP_disable_deterministic_archives, /*default=*/true);

  Config.PreserveDates = InputArgs.hasArg(STRIP_preserve_dates);
  Config.InputFormat = FileFormat::Unspecified;
  Config.OutputFormat = FileFormat::Unspecified;

  DriverConfig DC;
  if (Positional.size() == 1) {
    Config.InputFilename = Positional[0];
    Config.OutputFilename =
        InputArgs.getLastArgValue(STRIP_output, Positional[0]);
    DC.CopyConfigs.push_back(std::move(Config));
  } else {
    StringMap<unsigned> InputFiles;
    for (StringRef Filename : Positional) {
      if (InputFiles[Filename]++ == 1) {
        if (Filename == "-")
          return createStringError(
              errc::invalid_argument,
              "cannot specify '-' as an input file more than once");
        if (Error E = ErrorCallback(createStringError(
                errc::invalid_argument, "'%s' was already specified",
                Filename.str().c_str())))
          return std::move(E);
      }
      Config.InputFilename = Filename;
      Config.OutputFilename = Filename;
      DC.CopyConfigs.push_back(Config);
    }
  }

  if (Config.PreserveDates && (is_contained(Positional, "-") ||
                               InputArgs.getLastArgValue(STRIP_output) == "-"))
    return createStringError(errc::invalid_argument,
                             "--preserve-dates requires a file");

  return std::move(DC);
}

} // namespace objcopy
} // namespace llvm
