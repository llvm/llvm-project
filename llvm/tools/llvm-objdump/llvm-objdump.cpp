//===-- llvm-objdump.cpp - Object file dumping utility for llvm -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like binutils "objdump", that is, it
// dumps out a plethora of information about an object file depending on the
// flags.
//
// The flags and output of this program should be near identical to those of
// binutils objdump.
//
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "COFFDump.h"
#include "ELFDump.h"
#include "MachODump.h"
#include "WasmDump.h"
#include "XCOFFDump.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/FaultMaps.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCDisassembler/MCRelocationInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <system_error>
#include <unordered_map>
#include <utility>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::objdump;

static cl::OptionCategory ObjdumpCat("llvm-objdump Options");

static cl::opt<uint64_t> AdjustVMA(
    "adjust-vma",
    cl::desc("Increase the displayed address by the specified offset"),
    cl::value_desc("offset"), cl::init(0), cl::cat(ObjdumpCat));

static cl::opt<bool>
    AllHeaders("all-headers",
               cl::desc("Display all available header information"),
               cl::cat(ObjdumpCat));
static cl::alias AllHeadersShort("x", cl::desc("Alias for --all-headers"),
                                 cl::NotHidden, cl::Grouping,
                                 cl::aliasopt(AllHeaders));

static cl::opt<std::string>
    ArchName("arch-name",
             cl::desc("Target arch to disassemble for, "
                      "see -version for available targets"),
             cl::cat(ObjdumpCat));

cl::opt<bool>
    objdump::ArchiveHeaders("archive-headers",
                            cl::desc("Display archive header information"),
                            cl::cat(ObjdumpCat));
static cl::alias ArchiveHeadersShort("a",
                                     cl::desc("Alias for --archive-headers"),
                                     cl::NotHidden, cl::Grouping,
                                     cl::aliasopt(ArchiveHeaders));

cl::opt<bool> objdump::Demangle("demangle", cl::desc("Demangle symbols names"),
                                cl::init(false), cl::cat(ObjdumpCat));
static cl::alias DemangleShort("C", cl::desc("Alias for --demangle"),
                               cl::NotHidden, cl::Grouping,
                               cl::aliasopt(Demangle));

cl::opt<bool> objdump::Disassemble(
    "disassemble",
    cl::desc("Display assembler mnemonics for the machine instructions"),
    cl::cat(ObjdumpCat));
static cl::alias DisassembleShort("d", cl::desc("Alias for --disassemble"),
                                  cl::NotHidden, cl::Grouping,
                                  cl::aliasopt(Disassemble));

cl::opt<bool> objdump::DisassembleAll(
    "disassemble-all",
    cl::desc("Display assembler mnemonics for the machine instructions"),
    cl::cat(ObjdumpCat));
static cl::alias DisassembleAllShort("D",
                                     cl::desc("Alias for --disassemble-all"),
                                     cl::NotHidden, cl::Grouping,
                                     cl::aliasopt(DisassembleAll));

cl::opt<bool> objdump::SymbolDescription(
    "symbol-description",
    cl::desc("Add symbol description for disassembly. This "
             "option is for XCOFF files only"),
    cl::init(false), cl::cat(ObjdumpCat));

static cl::list<std::string>
    DisassembleSymbols("disassemble-symbols", cl::CommaSeparated,
                       cl::desc("List of symbols to disassemble. "
                                "Accept demangled names when --demangle is "
                                "specified, otherwise accept mangled names"),
                       cl::cat(ObjdumpCat));

static cl::opt<bool> DisassembleZeroes(
    "disassemble-zeroes",
    cl::desc("Do not skip blocks of zeroes when disassembling"),
    cl::cat(ObjdumpCat));
static cl::alias
    DisassembleZeroesShort("z", cl::desc("Alias for --disassemble-zeroes"),
                           cl::NotHidden, cl::Grouping,
                           cl::aliasopt(DisassembleZeroes));

static cl::list<std::string>
    DisassemblerOptions("disassembler-options",
                        cl::desc("Pass target specific disassembler options"),
                        cl::value_desc("options"), cl::CommaSeparated,
                        cl::cat(ObjdumpCat));
static cl::alias
    DisassemblerOptionsShort("M", cl::desc("Alias for --disassembler-options"),
                             cl::NotHidden, cl::Grouping, cl::Prefix,
                             cl::CommaSeparated,
                             cl::aliasopt(DisassemblerOptions));

cl::opt<DIDumpType> objdump::DwarfDumpType(
    "dwarf", cl::init(DIDT_Null), cl::desc("Dump of dwarf debug sections:"),
    cl::values(clEnumValN(DIDT_DebugFrame, "frames", ".debug_frame")),
    cl::cat(ObjdumpCat));

static cl::opt<bool> DynamicRelocations(
    "dynamic-reloc",
    cl::desc("Display the dynamic relocation entries in the file"),
    cl::cat(ObjdumpCat));
static cl::alias DynamicRelocationShort("R",
                                        cl::desc("Alias for --dynamic-reloc"),
                                        cl::NotHidden, cl::Grouping,
                                        cl::aliasopt(DynamicRelocations));

static cl::opt<bool>
    FaultMapSection("fault-map-section",
                    cl::desc("Display contents of faultmap section"),
                    cl::cat(ObjdumpCat));

static cl::opt<bool>
    FileHeaders("file-headers",
                cl::desc("Display the contents of the overall file header"),
                cl::cat(ObjdumpCat));
static cl::alias FileHeadersShort("f", cl::desc("Alias for --file-headers"),
                                  cl::NotHidden, cl::Grouping,
                                  cl::aliasopt(FileHeaders));

cl::opt<bool>
    objdump::SectionContents("full-contents",
                             cl::desc("Display the content of each section"),
                             cl::cat(ObjdumpCat));
static cl::alias SectionContentsShort("s",
                                      cl::desc("Alias for --full-contents"),
                                      cl::NotHidden, cl::Grouping,
                                      cl::aliasopt(SectionContents));

static cl::list<std::string> InputFilenames(cl::Positional,
                                            cl::desc("<input object files>"),
                                            cl::ZeroOrMore,
                                            cl::cat(ObjdumpCat));

static cl::opt<bool>
    PrintLines("line-numbers",
               cl::desc("Display source line numbers with "
                        "disassembly. Implies disassemble object"),
               cl::cat(ObjdumpCat));
static cl::alias PrintLinesShort("l", cl::desc("Alias for --line-numbers"),
                                 cl::NotHidden, cl::Grouping,
                                 cl::aliasopt(PrintLines));

static cl::opt<bool> MachOOpt("macho",
                              cl::desc("Use MachO specific object file parser"),
                              cl::cat(ObjdumpCat));
static cl::alias MachOm("m", cl::desc("Alias for --macho"), cl::NotHidden,
                        cl::Grouping, cl::aliasopt(MachOOpt));

cl::opt<std::string> objdump::MCPU(
    "mcpu", cl::desc("Target a specific cpu type (-mcpu=help for details)"),
    cl::value_desc("cpu-name"), cl::init(""), cl::cat(ObjdumpCat));

cl::list<std::string> objdump::MAttrs("mattr", cl::CommaSeparated,
                                      cl::desc("Target specific attributes"),
                                      cl::value_desc("a1,+a2,-a3,..."),
                                      cl::cat(ObjdumpCat));

cl::opt<bool> objdump::NoShowRawInsn(
    "no-show-raw-insn",
    cl::desc(
        "When disassembling instructions, do not print the instruction bytes."),
    cl::cat(ObjdumpCat));

cl::opt<bool> objdump::NoLeadingAddr("no-leading-addr",
                                     cl::desc("Print no leading address"),
                                     cl::cat(ObjdumpCat));

static cl::opt<bool> RawClangAST(
    "raw-clang-ast",
    cl::desc("Dump the raw binary contents of the clang AST section"),
    cl::cat(ObjdumpCat));

cl::opt<bool>
    objdump::Relocations("reloc",
                         cl::desc("Display the relocation entries in the file"),
                         cl::cat(ObjdumpCat));
static cl::alias RelocationsShort("r", cl::desc("Alias for --reloc"),
                                  cl::NotHidden, cl::Grouping,
                                  cl::aliasopt(Relocations));

cl::opt<bool>
    objdump::PrintImmHex("print-imm-hex",
                         cl::desc("Use hex format for immediate values"),
                         cl::cat(ObjdumpCat));

cl::opt<bool>
    objdump::PrivateHeaders("private-headers",
                            cl::desc("Display format specific file headers"),
                            cl::cat(ObjdumpCat));
static cl::alias PrivateHeadersShort("p",
                                     cl::desc("Alias for --private-headers"),
                                     cl::NotHidden, cl::Grouping,
                                     cl::aliasopt(PrivateHeaders));

cl::list<std::string>
    objdump::FilterSections("section",
                            cl::desc("Operate on the specified sections only. "
                                     "With -macho dump segment,section"),
                            cl::cat(ObjdumpCat));
static cl::alias FilterSectionsj("j", cl::desc("Alias for --section"),
                                 cl::NotHidden, cl::Grouping, cl::Prefix,
                                 cl::aliasopt(FilterSections));

cl::opt<bool> objdump::SectionHeaders(
    "section-headers",
    cl::desc("Display summaries of the headers for each section."),
    cl::cat(ObjdumpCat));
static cl::alias SectionHeadersShort("headers",
                                     cl::desc("Alias for --section-headers"),
                                     cl::NotHidden,
                                     cl::aliasopt(SectionHeaders));
static cl::alias SectionHeadersShorter("h",
                                       cl::desc("Alias for --section-headers"),
                                       cl::NotHidden, cl::Grouping,
                                       cl::aliasopt(SectionHeaders));

static cl::opt<bool>
    ShowLMA("show-lma",
            cl::desc("Display LMA column when dumping ELF section headers"),
            cl::cat(ObjdumpCat));

static cl::opt<bool> PrintSource(
    "source",
    cl::desc(
        "Display source inlined with disassembly. Implies disassemble object"),
    cl::cat(ObjdumpCat));
static cl::alias PrintSourceShort("S", cl::desc("Alias for -source"),
                                  cl::NotHidden, cl::Grouping,
                                  cl::aliasopt(PrintSource));

static cl::opt<uint64_t>
    StartAddress("start-address", cl::desc("Disassemble beginning at address"),
                 cl::value_desc("address"), cl::init(0), cl::cat(ObjdumpCat));
static cl::opt<uint64_t> StopAddress("stop-address",
                                     cl::desc("Stop disassembly at address"),
                                     cl::value_desc("address"),
                                     cl::init(UINT64_MAX), cl::cat(ObjdumpCat));

cl::opt<bool> objdump::SymbolTable("syms", cl::desc("Display the symbol table"),
                                   cl::cat(ObjdumpCat));
static cl::alias SymbolTableShort("t", cl::desc("Alias for --syms"),
                                  cl::NotHidden, cl::Grouping,
                                  cl::aliasopt(SymbolTable));

static cl::opt<bool> DynamicSymbolTable(
    "dynamic-syms",
    cl::desc("Display the contents of the dynamic symbol table"),
    cl::cat(ObjdumpCat));
static cl::alias DynamicSymbolTableShort("T",
                                         cl::desc("Alias for --dynamic-syms"),
                                         cl::NotHidden, cl::Grouping,
                                         cl::aliasopt(DynamicSymbolTable));

cl::opt<std::string> objdump::TripleName(
    "triple",
    cl::desc(
        "Target triple to disassemble for, see -version for available targets"),
    cl::cat(ObjdumpCat));

cl::opt<bool> objdump::UnwindInfo("unwind-info",
                                  cl::desc("Display unwind information"),
                                  cl::cat(ObjdumpCat));
static cl::alias UnwindInfoShort("u", cl::desc("Alias for --unwind-info"),
                                 cl::NotHidden, cl::Grouping,
                                 cl::aliasopt(UnwindInfo));

static cl::opt<bool>
    Wide("wide", cl::desc("Ignored for compatibility with GNU objdump"),
         cl::cat(ObjdumpCat));
static cl::alias WideShort("w", cl::Grouping, cl::aliasopt(Wide));

static cl::extrahelp
    HelpResponse("\nPass @FILE as argument to read options from FILE.\n");

static StringSet<> DisasmSymbolSet;
StringSet<> objdump::FoundSectionSet;
static StringRef ToolName;

namespace {
struct FilterResult {
  // True if the section should not be skipped.
  bool Keep;

  // True if the index counter should be incremented, even if the section should
  // be skipped. For example, sections may be skipped if they are not included
  // in the --section flag, but we still want those to count toward the section
  // count.
  bool IncrementIndex;
};
} // namespace

static FilterResult checkSectionFilter(object::SectionRef S) {
  if (FilterSections.empty())
    return {/*Keep=*/true, /*IncrementIndex=*/true};

  Expected<StringRef> SecNameOrErr = S.getName();
  if (!SecNameOrErr) {
    consumeError(SecNameOrErr.takeError());
    return {/*Keep=*/false, /*IncrementIndex=*/false};
  }
  StringRef SecName = *SecNameOrErr;

  // StringSet does not allow empty key so avoid adding sections with
  // no name (such as the section with index 0) here.
  if (!SecName.empty())
    FoundSectionSet.insert(SecName);

  // Only show the section if it's in the FilterSections list, but always
  // increment so the indexing is stable.
  return {/*Keep=*/is_contained(FilterSections, SecName),
          /*IncrementIndex=*/true};
}

namespace llvm {

SectionFilter ToolSectionFilter(object::ObjectFile const &O, uint64_t *Idx) {
  // Start at UINT64_MAX so that the first index returned after an increment is
  // zero (after the unsigned wrap).
  if (Idx)
    *Idx = UINT64_MAX;
  return SectionFilter(
      [Idx](object::SectionRef S) {
        FilterResult Result = checkSectionFilter(S);
        if (Idx != nullptr && Result.IncrementIndex)
          *Idx += 1;
        return Result.Keep;
      },
      O);
}

std::string getFileNameForError(const object::Archive::Child &C,
                                unsigned Index) {
  Expected<StringRef> NameOrErr = C.getName();
  if (NameOrErr)
    return std::string(NameOrErr.get());
  // If we have an error getting the name then we print the index of the archive
  // member. Since we are already in an error state, we just ignore this error.
  consumeError(NameOrErr.takeError());
  return "<file index: " + std::to_string(Index) + ">";
}

void reportWarning(Twine Message, StringRef File) {
  // Output order between errs() and outs() matters especially for archive
  // files where the output is per member object.
  outs().flush();
  WithColor::warning(errs(), ToolName)
      << "'" << File << "': " << Message << "\n";
}

LLVM_ATTRIBUTE_NORETURN void reportError(StringRef File, Twine Message) {
  outs().flush();
  WithColor::error(errs(), ToolName) << "'" << File << "': " << Message << "\n";
  exit(1);
}

LLVM_ATTRIBUTE_NORETURN void reportError(Error E, StringRef FileName,
                                         StringRef ArchiveName,
                                         StringRef ArchitectureName) {
  assert(E);
  outs().flush();
  WithColor::error(errs(), ToolName);
  if (ArchiveName != "")
    errs() << ArchiveName << "(" << FileName << ")";
  else
    errs() << "'" << FileName << "'";
  if (!ArchitectureName.empty())
    errs() << " (for architecture " << ArchitectureName << ")";
  errs() << ": ";
  logAllUnhandledErrors(std::move(E), errs());
  exit(1);
}

static void reportCmdLineWarning(Twine Message) {
  WithColor::warning(errs(), ToolName) << Message << "\n";
}

LLVM_ATTRIBUTE_NORETURN static void reportCmdLineError(Twine Message) {
  WithColor::error(errs(), ToolName) << Message << "\n";
  exit(1);
}

static void warnOnNoMatchForSections() {
  SetVector<StringRef> MissingSections;
  for (StringRef S : FilterSections) {
    if (FoundSectionSet.count(S))
      return;
    // User may specify a unnamed section. Don't warn for it.
    if (!S.empty())
      MissingSections.insert(S);
  }

  // Warn only if no section in FilterSections is matched.
  for (StringRef S : MissingSections)
    reportCmdLineWarning("section '" + S +
                         "' mentioned in a -j/--section option, but not "
                         "found in any input file");
}

static const Target *getTarget(const ObjectFile *Obj) {
  // Figure out the target triple.
  Triple TheTriple("unknown-unknown-unknown");
  if (TripleName.empty()) {
    TheTriple = Obj->makeTriple();
  } else {
    TheTriple.setTriple(Triple::normalize(TripleName));
    auto Arch = Obj->getArch();
    if (Arch == Triple::arm || Arch == Triple::armeb)
      Obj->setARMSubArch(TheTriple);
  }

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(ArchName, TheTriple,
                                                         Error);
  if (!TheTarget)
    reportError(Obj->getFileName(), "can't find target: " + Error);

  // Update the triple name and return the found target.
  TripleName = TheTriple.getTriple();
  return TheTarget;
}

bool isRelocAddressLess(RelocationRef A, RelocationRef B) {
  return A.getOffset() < B.getOffset();
}

static Error getRelocationValueString(const RelocationRef &Rel,
                                      SmallVectorImpl<char> &Result) {
  const ObjectFile *Obj = Rel.getObject();
  if (auto *ELF = dyn_cast<ELFObjectFileBase>(Obj))
    return getELFRelocationValueString(ELF, Rel, Result);
  if (auto *COFF = dyn_cast<COFFObjectFile>(Obj))
    return getCOFFRelocationValueString(COFF, Rel, Result);
  if (auto *Wasm = dyn_cast<WasmObjectFile>(Obj))
    return getWasmRelocationValueString(Wasm, Rel, Result);
  if (auto *MachO = dyn_cast<MachOObjectFile>(Obj))
    return getMachORelocationValueString(MachO, Rel, Result);
  if (auto *XCOFF = dyn_cast<XCOFFObjectFile>(Obj))
    return getXCOFFRelocationValueString(XCOFF, Rel, Result);
  llvm_unreachable("unknown object file format");
}

/// Indicates whether this relocation should hidden when listing
/// relocations, usually because it is the trailing part of a multipart
/// relocation that will be printed as part of the leading relocation.
static bool getHidden(RelocationRef RelRef) {
  auto *MachO = dyn_cast<MachOObjectFile>(RelRef.getObject());
  if (!MachO)
    return false;

  unsigned Arch = MachO->getArch();
  DataRefImpl Rel = RelRef.getRawDataRefImpl();
  uint64_t Type = MachO->getRelocationType(Rel);

  // On arches that use the generic relocations, GENERIC_RELOC_PAIR
  // is always hidden.
  if (Arch == Triple::x86 || Arch == Triple::arm || Arch == Triple::ppc)
    return Type == MachO::GENERIC_RELOC_PAIR;

  if (Arch == Triple::x86_64) {
    // On x86_64, X86_64_RELOC_UNSIGNED is hidden only when it follows
    // an X86_64_RELOC_SUBTRACTOR.
    if (Type == MachO::X86_64_RELOC_UNSIGNED && Rel.d.a > 0) {
      DataRefImpl RelPrev = Rel;
      RelPrev.d.a--;
      uint64_t PrevType = MachO->getRelocationType(RelPrev);
      if (PrevType == MachO::X86_64_RELOC_SUBTRACTOR)
        return true;
    }
  }

  return false;
}

namespace {
class SourcePrinter {
protected:
  DILineInfo OldLineInfo;
  const ObjectFile *Obj = nullptr;
  std::unique_ptr<symbolize::LLVMSymbolizer> Symbolizer;
  // File name to file contents of source.
  std::unordered_map<std::string, std::unique_ptr<MemoryBuffer>> SourceCache;
  // Mark the line endings of the cached source.
  std::unordered_map<std::string, std::vector<StringRef>> LineCache;
  // Keep track of missing sources.
  StringSet<> MissingSources;
  // Only emit 'no debug info' warning once.
  bool WarnedNoDebugInfo;

private:
  bool cacheSource(const DILineInfo& LineInfoFile);

  void printLines(raw_ostream &OS, const DILineInfo &LineInfo,
                  StringRef Delimiter);

  void printSources(raw_ostream &OS, const DILineInfo &LineInfo,
                    StringRef ObjectFilename, StringRef Delimiter);

public:
  SourcePrinter() = default;
  SourcePrinter(const ObjectFile *Obj, StringRef DefaultArch)
      : Obj(Obj), WarnedNoDebugInfo(false) {
    symbolize::LLVMSymbolizer::Options SymbolizerOpts;
    SymbolizerOpts.PrintFunctions =
        DILineInfoSpecifier::FunctionNameKind::LinkageName;
    SymbolizerOpts.Demangle = Demangle;
    SymbolizerOpts.DefaultArch = std::string(DefaultArch);
    Symbolizer.reset(new symbolize::LLVMSymbolizer(SymbolizerOpts));
  }
  virtual ~SourcePrinter() = default;
  virtual void printSourceLine(raw_ostream &OS,
                               object::SectionedAddress Address,
                               StringRef ObjectFilename,
                               StringRef Delimiter = "; ");
};

bool SourcePrinter::cacheSource(const DILineInfo &LineInfo) {
  std::unique_ptr<MemoryBuffer> Buffer;
  if (LineInfo.Source) {
    Buffer = MemoryBuffer::getMemBuffer(*LineInfo.Source);
  } else {
    auto BufferOrError = MemoryBuffer::getFile(LineInfo.FileName);
    if (!BufferOrError) {
      if (MissingSources.insert(LineInfo.FileName).second)
        reportWarning("failed to find source " + LineInfo.FileName,
                      Obj->getFileName());
      return false;
    }
    Buffer = std::move(*BufferOrError);
  }
  // Chomp the file to get lines
  const char *BufferStart = Buffer->getBufferStart(),
             *BufferEnd = Buffer->getBufferEnd();
  std::vector<StringRef> &Lines = LineCache[LineInfo.FileName];
  const char *Start = BufferStart;
  for (const char *I = BufferStart; I != BufferEnd; ++I)
    if (*I == '\n') {
      Lines.emplace_back(Start, I - Start - (BufferStart < I && I[-1] == '\r'));
      Start = I + 1;
    }
  if (Start < BufferEnd)
    Lines.emplace_back(Start, BufferEnd - Start);
  SourceCache[LineInfo.FileName] = std::move(Buffer);
  return true;
}

void SourcePrinter::printSourceLine(raw_ostream &OS,
                                    object::SectionedAddress Address,
                                    StringRef ObjectFilename,
                                    StringRef Delimiter) {
  if (!Symbolizer)
    return;

  DILineInfo LineInfo = DILineInfo();
  auto ExpectedLineInfo = Symbolizer->symbolizeCode(*Obj, Address);
  std::string ErrorMessage;
  if (!ExpectedLineInfo)
    ErrorMessage = toString(ExpectedLineInfo.takeError());
  else
    LineInfo = *ExpectedLineInfo;

  if (LineInfo.FileName == DILineInfo::BadString) {
    if (!WarnedNoDebugInfo) {
      std::string Warning =
          "failed to parse debug information for " + ObjectFilename.str();
      if (!ErrorMessage.empty())
        Warning += ": " + ErrorMessage;
      reportWarning(Warning, ObjectFilename);
      WarnedNoDebugInfo = true;
    }
  }

  if (PrintLines)
    printLines(OS, LineInfo, Delimiter);
  if (PrintSource)
    printSources(OS, LineInfo, ObjectFilename, Delimiter);
  OldLineInfo = LineInfo;
}

void SourcePrinter::printLines(raw_ostream &OS, const DILineInfo &LineInfo,
                               StringRef Delimiter) {
  bool PrintFunctionName = LineInfo.FunctionName != DILineInfo::BadString &&
                           LineInfo.FunctionName != OldLineInfo.FunctionName;
  if (PrintFunctionName) {
    OS << Delimiter << LineInfo.FunctionName;
    // If demangling is successful, FunctionName will end with "()". Print it
    // only if demangling did not run or was unsuccessful.
    if (!StringRef(LineInfo.FunctionName).endswith("()"))
      OS << "()";
    OS << ":\n";
  }
  if (LineInfo.FileName != DILineInfo::BadString && LineInfo.Line != 0 &&
      (OldLineInfo.Line != LineInfo.Line ||
       OldLineInfo.FileName != LineInfo.FileName || PrintFunctionName))
    OS << Delimiter << LineInfo.FileName << ":" << LineInfo.Line << "\n";
}

void SourcePrinter::printSources(raw_ostream &OS, const DILineInfo &LineInfo,
                                 StringRef ObjectFilename,
                                 StringRef Delimiter) {
  if (LineInfo.FileName == DILineInfo::BadString || LineInfo.Line == 0 ||
      (OldLineInfo.Line == LineInfo.Line &&
       OldLineInfo.FileName == LineInfo.FileName))
    return;

  if (SourceCache.find(LineInfo.FileName) == SourceCache.end())
    if (!cacheSource(LineInfo))
      return;
  auto LineBuffer = LineCache.find(LineInfo.FileName);
  if (LineBuffer != LineCache.end()) {
    if (LineInfo.Line > LineBuffer->second.size()) {
      reportWarning(
          formatv(
              "debug info line number {0} exceeds the number of lines in {1}",
              LineInfo.Line, LineInfo.FileName),
          ObjectFilename);
      return;
    }
    // Vector begins at 0, line numbers are non-zero
    OS << Delimiter << LineBuffer->second[LineInfo.Line - 1] << '\n';
  }
}

static bool isAArch64Elf(const ObjectFile *Obj) {
  const auto *Elf = dyn_cast<ELFObjectFileBase>(Obj);
  return Elf && Elf->getEMachine() == ELF::EM_AARCH64;
}

static bool isArmElf(const ObjectFile *Obj) {
  const auto *Elf = dyn_cast<ELFObjectFileBase>(Obj);
  return Elf && Elf->getEMachine() == ELF::EM_ARM;
}

static bool hasMappingSymbols(const ObjectFile *Obj) {
  return isArmElf(Obj) || isAArch64Elf(Obj);
}

static void printRelocation(StringRef FileName, const RelocationRef &Rel,
                            uint64_t Address, bool Is64Bits) {
  StringRef Fmt = Is64Bits ? "\t\t%016" PRIx64 ":  " : "\t\t\t%08" PRIx64 ":  ";
  SmallString<16> Name;
  SmallString<32> Val;
  Rel.getTypeName(Name);
  if (Error E = getRelocationValueString(Rel, Val))
    reportError(std::move(E), FileName);
  outs() << format(Fmt.data(), Address) << Name << "\t" << Val << "\n";
}

class PrettyPrinter {
public:
  virtual ~PrettyPrinter() = default;
  virtual void printInst(MCInstPrinter &IP, const MCInst *MI,
                         ArrayRef<uint8_t> Bytes,
                         object::SectionedAddress Address, raw_ostream &OS,
                         StringRef Annot, MCSubtargetInfo const &STI,
                         SourcePrinter *SP, StringRef ObjectFilename,
                         std::vector<RelocationRef> *Rels = nullptr) {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address, ObjectFilename);

    size_t Start = OS.tell();
    if (!NoLeadingAddr)
      OS << format("%8" PRIx64 ":", Address.Address);
    if (!NoShowRawInsn) {
      OS << ' ';
      dumpBytes(Bytes, OS);
    }

    // The output of printInst starts with a tab. Print some spaces so that
    // the tab has 1 column and advances to the target tab stop.
    unsigned TabStop = NoShowRawInsn ? 16 : 40;
    unsigned Column = OS.tell() - Start;
    OS.indent(Column < TabStop - 1 ? TabStop - 1 - Column : 7 - Column % 8);

    if (MI) {
      // See MCInstPrinter::printInst. On targets where a PC relative immediate
      // is relative to the next instruction and the length of a MCInst is
      // difficult to measure (x86), this is the address of the next
      // instruction.
      uint64_t Addr =
          Address.Address + (STI.getTargetTriple().isX86() ? Bytes.size() : 0);
      IP.printInst(MI, Addr, "", STI, OS);
    } else
      OS << "\t<unknown>";
  }
};
PrettyPrinter PrettyPrinterInst;

class HexagonPrettyPrinter : public PrettyPrinter {
public:
  void printLead(ArrayRef<uint8_t> Bytes, uint64_t Address,
                 raw_ostream &OS) {
    uint32_t opcode =
      (Bytes[3] << 24) | (Bytes[2] << 16) | (Bytes[1] << 8) | Bytes[0];
    if (!NoLeadingAddr)
      OS << format("%8" PRIx64 ":", Address);
    if (!NoShowRawInsn) {
      OS << "\t";
      dumpBytes(Bytes.slice(0, 4), OS);
      OS << format("\t%08" PRIx32, opcode);
    }
  }
  void printInst(MCInstPrinter &IP, const MCInst *MI, ArrayRef<uint8_t> Bytes,
                 object::SectionedAddress Address, raw_ostream &OS,
                 StringRef Annot, MCSubtargetInfo const &STI, SourcePrinter *SP,
                 StringRef ObjectFilename,
                 std::vector<RelocationRef> *Rels) override {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address, ObjectFilename, "");
    if (!MI) {
      printLead(Bytes, Address.Address, OS);
      OS << " <unknown>";
      return;
    }
    std::string Buffer;
    {
      raw_string_ostream TempStream(Buffer);
      IP.printInst(MI, Address.Address, "", STI, TempStream);
    }
    StringRef Contents(Buffer);
    // Split off bundle attributes
    auto PacketBundle = Contents.rsplit('\n');
    // Split off first instruction from the rest
    auto HeadTail = PacketBundle.first.split('\n');
    auto Preamble = " { ";
    auto Separator = "";

    // Hexagon's packets require relocations to be inline rather than
    // clustered at the end of the packet.
    std::vector<RelocationRef>::const_iterator RelCur = Rels->begin();
    std::vector<RelocationRef>::const_iterator RelEnd = Rels->end();
    auto PrintReloc = [&]() -> void {
      while ((RelCur != RelEnd) && (RelCur->getOffset() <= Address.Address)) {
        if (RelCur->getOffset() == Address.Address) {
          printRelocation(ObjectFilename, *RelCur, Address.Address, false);
          return;
        }
        ++RelCur;
      }
    };

    while (!HeadTail.first.empty()) {
      OS << Separator;
      Separator = "\n";
      if (SP && (PrintSource || PrintLines))
        SP->printSourceLine(OS, Address, ObjectFilename, "");
      printLead(Bytes, Address.Address, OS);
      OS << Preamble;
      Preamble = "   ";
      StringRef Inst;
      auto Duplex = HeadTail.first.split('\v');
      if (!Duplex.second.empty()) {
        OS << Duplex.first;
        OS << "; ";
        Inst = Duplex.second;
      }
      else
        Inst = HeadTail.first;
      OS << Inst;
      HeadTail = HeadTail.second.split('\n');
      if (HeadTail.first.empty())
        OS << " } " << PacketBundle.second;
      PrintReloc();
      Bytes = Bytes.slice(4);
      Address.Address += 4;
    }
  }
};
HexagonPrettyPrinter HexagonPrettyPrinterInst;

class AMDGCNPrettyPrinter : public PrettyPrinter {
public:
  void printInst(MCInstPrinter &IP, const MCInst *MI, ArrayRef<uint8_t> Bytes,
                 object::SectionedAddress Address, raw_ostream &OS,
                 StringRef Annot, MCSubtargetInfo const &STI, SourcePrinter *SP,
                 StringRef ObjectFilename,
                 std::vector<RelocationRef> *Rels) override {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address, ObjectFilename);

    if (MI) {
      SmallString<40> InstStr;
      raw_svector_ostream IS(InstStr);

      IP.printInst(MI, Address.Address, "", STI, IS);

      OS << left_justify(IS.str(), 60);
    } else {
      // an unrecognized encoding - this is probably data so represent it
      // using the .long directive, or .byte directive if fewer than 4 bytes
      // remaining
      if (Bytes.size() >= 4) {
        OS << format("\t.long 0x%08" PRIx32 " ",
                     support::endian::read32<support::little>(Bytes.data()));
        OS.indent(42);
      } else {
          OS << format("\t.byte 0x%02" PRIx8, Bytes[0]);
          for (unsigned int i = 1; i < Bytes.size(); i++)
            OS << format(", 0x%02" PRIx8, Bytes[i]);
          OS.indent(55 - (6 * Bytes.size()));
      }
    }

    OS << format("// %012" PRIX64 ":", Address.Address);
    if (Bytes.size() >= 4) {
      // D should be casted to uint32_t here as it is passed by format to
      // snprintf as vararg.
      for (uint32_t D : makeArrayRef(
               reinterpret_cast<const support::little32_t *>(Bytes.data()),
               Bytes.size() / 4))
        OS << format(" %08" PRIX32, D);
    } else {
      for (unsigned char B : Bytes)
        OS << format(" %02" PRIX8, B);
    }

    if (!Annot.empty())
      OS << " // " << Annot;
  }
};
AMDGCNPrettyPrinter AMDGCNPrettyPrinterInst;

class BPFPrettyPrinter : public PrettyPrinter {
public:
  void printInst(MCInstPrinter &IP, const MCInst *MI, ArrayRef<uint8_t> Bytes,
                 object::SectionedAddress Address, raw_ostream &OS,
                 StringRef Annot, MCSubtargetInfo const &STI, SourcePrinter *SP,
                 StringRef ObjectFilename,
                 std::vector<RelocationRef> *Rels) override {
    if (SP && (PrintSource || PrintLines))
      SP->printSourceLine(OS, Address, ObjectFilename);
    if (!NoLeadingAddr)
      OS << format("%8" PRId64 ":", Address.Address / 8);
    if (!NoShowRawInsn) {
      OS << "\t";
      dumpBytes(Bytes, OS);
    }
    if (MI)
      IP.printInst(MI, Address.Address, "", STI, OS);
    else
      OS << "\t<unknown>";
  }
};
BPFPrettyPrinter BPFPrettyPrinterInst;

PrettyPrinter &selectPrettyPrinter(Triple const &Triple) {
  switch(Triple.getArch()) {
  default:
    return PrettyPrinterInst;
  case Triple::hexagon:
    return HexagonPrettyPrinterInst;
  case Triple::amdgcn:
    return AMDGCNPrettyPrinterInst;
  case Triple::bpfel:
  case Triple::bpfeb:
    return BPFPrettyPrinterInst;
  }
}
}

static uint8_t getElfSymbolType(const ObjectFile *Obj, const SymbolRef &Sym) {
  assert(Obj->isELF());
  if (auto *Elf32LEObj = dyn_cast<ELF32LEObjectFile>(Obj))
    return Elf32LEObj->getSymbol(Sym.getRawDataRefImpl())->getType();
  if (auto *Elf64LEObj = dyn_cast<ELF64LEObjectFile>(Obj))
    return Elf64LEObj->getSymbol(Sym.getRawDataRefImpl())->getType();
  if (auto *Elf32BEObj = dyn_cast<ELF32BEObjectFile>(Obj))
    return Elf32BEObj->getSymbol(Sym.getRawDataRefImpl())->getType();
  if (auto *Elf64BEObj = cast<ELF64BEObjectFile>(Obj))
    return Elf64BEObj->getSymbol(Sym.getRawDataRefImpl())->getType();
  llvm_unreachable("Unsupported binary format");
}

template <class ELFT> static void
addDynamicElfSymbols(const ELFObjectFile<ELFT> *Obj,
                     std::map<SectionRef, SectionSymbolsTy> &AllSymbols) {
  for (auto Symbol : Obj->getDynamicSymbolIterators()) {
    uint8_t SymbolType = Symbol.getELFType();
    if (SymbolType == ELF::STT_SECTION)
      continue;

    uint64_t Address = unwrapOrError(Symbol.getAddress(), Obj->getFileName());
    // ELFSymbolRef::getAddress() returns size instead of value for common
    // symbols which is not desirable for disassembly output. Overriding.
    if (SymbolType == ELF::STT_COMMON)
      Address = Obj->getSymbol(Symbol.getRawDataRefImpl())->st_value;

    StringRef Name = unwrapOrError(Symbol.getName(), Obj->getFileName());
    if (Name.empty())
      continue;

    section_iterator SecI =
        unwrapOrError(Symbol.getSection(), Obj->getFileName());
    if (SecI == Obj->section_end())
      continue;

    AllSymbols[*SecI].emplace_back(Address, Name, SymbolType);
  }
}

static void
addDynamicElfSymbols(const ObjectFile *Obj,
                     std::map<SectionRef, SectionSymbolsTy> &AllSymbols) {
  assert(Obj->isELF());
  if (auto *Elf32LEObj = dyn_cast<ELF32LEObjectFile>(Obj))
    addDynamicElfSymbols(Elf32LEObj, AllSymbols);
  else if (auto *Elf64LEObj = dyn_cast<ELF64LEObjectFile>(Obj))
    addDynamicElfSymbols(Elf64LEObj, AllSymbols);
  else if (auto *Elf32BEObj = dyn_cast<ELF32BEObjectFile>(Obj))
    addDynamicElfSymbols(Elf32BEObj, AllSymbols);
  else if (auto *Elf64BEObj = cast<ELF64BEObjectFile>(Obj))
    addDynamicElfSymbols(Elf64BEObj, AllSymbols);
  else
    llvm_unreachable("Unsupported binary format");
}

static void addPltEntries(const ObjectFile *Obj,
                          std::map<SectionRef, SectionSymbolsTy> &AllSymbols,
                          StringSaver &Saver) {
  Optional<SectionRef> Plt = None;
  for (const SectionRef &Section : Obj->sections()) {
    Expected<StringRef> SecNameOrErr = Section.getName();
    if (!SecNameOrErr) {
      consumeError(SecNameOrErr.takeError());
      continue;
    }
    if (*SecNameOrErr == ".plt")
      Plt = Section;
  }
  if (!Plt)
    return;
  if (auto *ElfObj = dyn_cast<ELFObjectFileBase>(Obj)) {
    for (auto PltEntry : ElfObj->getPltAddresses()) {
      SymbolRef Symbol(PltEntry.first, ElfObj);
      uint8_t SymbolType = getElfSymbolType(Obj, Symbol);

      StringRef Name = unwrapOrError(Symbol.getName(), Obj->getFileName());
      if (!Name.empty())
        AllSymbols[*Plt].emplace_back(
            PltEntry.second, Saver.save((Name + "@plt").str()), SymbolType);
    }
  }
}

// Normally the disassembly output will skip blocks of zeroes. This function
// returns the number of zero bytes that can be skipped when dumping the
// disassembly of the instructions in Buf.
static size_t countSkippableZeroBytes(ArrayRef<uint8_t> Buf) {
  // Find the number of leading zeroes.
  size_t N = 0;
  while (N < Buf.size() && !Buf[N])
    ++N;

  // We may want to skip blocks of zero bytes, but unless we see
  // at least 8 of them in a row.
  if (N < 8)
    return 0;

  // We skip zeroes in multiples of 4 because do not want to truncate an
  // instruction if it starts with a zero byte.
  return N & ~0x3;
}

// Returns a map from sections to their relocations.
static std::map<SectionRef, std::vector<RelocationRef>>
getRelocsMap(object::ObjectFile const &Obj) {
  std::map<SectionRef, std::vector<RelocationRef>> Ret;
  uint64_t I = (uint64_t)-1;
  for (SectionRef Sec : Obj.sections()) {
    ++I;
    Expected<section_iterator> RelocatedOrErr = Sec.getRelocatedSection();
    if (!RelocatedOrErr)
      reportError(Obj.getFileName(),
                  "section (" + Twine(I) +
                      "): failed to get a relocated section: " +
                      toString(RelocatedOrErr.takeError()));

    section_iterator Relocated = *RelocatedOrErr;
    if (Relocated == Obj.section_end() || !checkSectionFilter(*Relocated).Keep)
      continue;
    std::vector<RelocationRef> &V = Ret[*Relocated];
    for (const RelocationRef &R : Sec.relocations())
      V.push_back(R);
    // Sort relocations by address.
    llvm::stable_sort(V, isRelocAddressLess);
  }
  return Ret;
}

// Used for --adjust-vma to check if address should be adjusted by the
// specified value for a given section.
// For ELF we do not adjust non-allocatable sections like debug ones,
// because they are not loadable.
// TODO: implement for other file formats.
static bool shouldAdjustVA(const SectionRef &Section) {
  const ObjectFile *Obj = Section.getObject();
  if (Obj->isELF())
    return ELFSectionRef(Section).getFlags() & ELF::SHF_ALLOC;
  return false;
}


typedef std::pair<uint64_t, char> MappingSymbolPair;
static char getMappingSymbolKind(ArrayRef<MappingSymbolPair> MappingSymbols,
                                 uint64_t Address) {
  auto It =
      partition_point(MappingSymbols, [Address](const MappingSymbolPair &Val) {
        return Val.first <= Address;
      });
  // Return zero for any address before the first mapping symbol; this means
  // we should use the default disassembly mode, depending on the target.
  if (It == MappingSymbols.begin())
    return '\x00';
  return (It - 1)->second;
}

static uint64_t dumpARMELFData(uint64_t SectionAddr, uint64_t Index,
                               uint64_t End, const ObjectFile *Obj,
                               ArrayRef<uint8_t> Bytes,
                               ArrayRef<MappingSymbolPair> MappingSymbols) {
  support::endianness Endian =
      Obj->isLittleEndian() ? support::little : support::big;
  outs() << format("%8" PRIx64 ":\t", SectionAddr + Index);
  if (Index + 4 <= End) {
    dumpBytes(Bytes.slice(Index, 4), outs());
    outs() << "\t.word\t"
           << format_hex(support::endian::read32(Bytes.data() + Index, Endian),
                         10);
    return 4;
  }
  if (Index + 2 <= End) {
    dumpBytes(Bytes.slice(Index, 2), outs());
    outs() << "\t\t.short\t"
           << format_hex(support::endian::read16(Bytes.data() + Index, Endian),
                         6);
    return 2;
  }
  dumpBytes(Bytes.slice(Index, 1), outs());
  outs() << "\t\t.byte\t" << format_hex(Bytes[0], 4);
  return 1;
}

static void dumpELFData(uint64_t SectionAddr, uint64_t Index, uint64_t End,
                        ArrayRef<uint8_t> Bytes) {
  // print out data up to 8 bytes at a time in hex and ascii
  uint8_t AsciiData[9] = {'\0'};
  uint8_t Byte;
  int NumBytes = 0;

  for (; Index < End; ++Index) {
    if (NumBytes == 0)
      outs() << format("%8" PRIx64 ":", SectionAddr + Index);
    Byte = Bytes.slice(Index)[0];
    outs() << format(" %02x", Byte);
    AsciiData[NumBytes] = isPrint(Byte) ? Byte : '.';

    uint8_t IndentOffset = 0;
    NumBytes++;
    if (Index == End - 1 || NumBytes > 8) {
      // Indent the space for less than 8 bytes data.
      // 2 spaces for byte and one for space between bytes
      IndentOffset = 3 * (8 - NumBytes);
      for (int Excess = NumBytes; Excess < 8; Excess++)
        AsciiData[Excess] = '\0';
      NumBytes = 8;
    }
    if (NumBytes == 8) {
      AsciiData[8] = '\0';
      outs() << std::string(IndentOffset, ' ') << "         ";
      outs() << reinterpret_cast<char *>(AsciiData);
      outs() << '\n';
      NumBytes = 0;
    }
  }
}

SymbolInfoTy createSymbolInfo(const ObjectFile *Obj, const SymbolRef &Symbol) {
  const StringRef FileName = Obj->getFileName();
  const uint64_t Addr = unwrapOrError(Symbol.getAddress(), FileName);
  const StringRef Name = unwrapOrError(Symbol.getName(), FileName);

  if (Obj->isXCOFF() && SymbolDescription) {
    const auto *XCOFFObj = cast<XCOFFObjectFile>(Obj);
    DataRefImpl SymbolDRI = Symbol.getRawDataRefImpl();

    const uint32_t SymbolIndex = XCOFFObj->getSymbolIndex(SymbolDRI.p);
    Optional<XCOFF::StorageMappingClass> Smc =
        getXCOFFSymbolCsectSMC(XCOFFObj, Symbol);
    return SymbolInfoTy(Addr, Name, Smc, SymbolIndex,
                        isLabel(XCOFFObj, Symbol));
  } else
    return SymbolInfoTy(Addr, Name,
                        Obj->isELF() ? getElfSymbolType(Obj, Symbol)
                                     : (uint8_t)ELF::STT_NOTYPE);
}

SymbolInfoTy createDummySymbolInfo(const ObjectFile *Obj, const uint64_t Addr,
                                   StringRef &Name, uint8_t Type) {
  if (Obj->isXCOFF() && SymbolDescription)
    return SymbolInfoTy(Addr, Name, None, None, false);
  else
    return SymbolInfoTy(Addr, Name, Type);
}

static void disassembleObject(const Target *TheTarget, const ObjectFile *Obj,
                              MCContext &Ctx, MCDisassembler *PrimaryDisAsm,
                              MCDisassembler *SecondaryDisAsm,
                              const MCInstrAnalysis *MIA, MCInstPrinter *IP,
                              const MCSubtargetInfo *PrimarySTI,
                              const MCSubtargetInfo *SecondarySTI,
                              PrettyPrinter &PIP,
                              SourcePrinter &SP, bool InlineRelocs) {
  const MCSubtargetInfo *STI = PrimarySTI;
  MCDisassembler *DisAsm = PrimaryDisAsm;
  bool PrimaryIsThumb = false;
  if (isArmElf(Obj))
    PrimaryIsThumb = STI->checkFeatures("+thumb-mode");

  std::map<SectionRef, std::vector<RelocationRef>> RelocMap;
  if (InlineRelocs)
    RelocMap = getRelocsMap(*Obj);
  bool Is64Bits = Obj->getBytesInAddress() > 4;

  // Create a mapping from virtual address to symbol name.  This is used to
  // pretty print the symbols while disassembling.
  std::map<SectionRef, SectionSymbolsTy> AllSymbols;
  SectionSymbolsTy AbsoluteSymbols;
  const StringRef FileName = Obj->getFileName();
  const MachOObjectFile *MachO = dyn_cast<const MachOObjectFile>(Obj);
  for (const SymbolRef &Symbol : Obj->symbols()) {
    StringRef Name = unwrapOrError(Symbol.getName(), FileName);
    if (Name.empty() && !(Obj->isXCOFF() && SymbolDescription))
      continue;

    if (Obj->isELF() && getElfSymbolType(Obj, Symbol) == ELF::STT_SECTION)
      continue;

    // Don't ask a Mach-O STAB symbol for its section unless you know that
    // STAB symbol's section field refers to a valid section index. Otherwise
    // the symbol may error trying to load a section that does not exist.
    if (MachO) {
      DataRefImpl SymDRI = Symbol.getRawDataRefImpl();
      uint8_t NType = (MachO->is64Bit() ?
                       MachO->getSymbol64TableEntry(SymDRI).n_type:
                       MachO->getSymbolTableEntry(SymDRI).n_type);
      if (NType & MachO::N_STAB)
        continue;
    }

    section_iterator SecI = unwrapOrError(Symbol.getSection(), FileName);
    if (SecI != Obj->section_end())
      AllSymbols[*SecI].push_back(createSymbolInfo(Obj, Symbol));
    else
      AbsoluteSymbols.push_back(createSymbolInfo(Obj, Symbol));
  }

  if (AllSymbols.empty() && Obj->isELF())
    addDynamicElfSymbols(Obj, AllSymbols);

  BumpPtrAllocator A;
  StringSaver Saver(A);
  addPltEntries(Obj, AllSymbols, Saver);

  // Create a mapping from virtual address to section. An empty section can
  // cause more than one section at the same address. Sort such sections to be
  // before same-addressed non-empty sections so that symbol lookups prefer the
  // non-empty section.
  std::vector<std::pair<uint64_t, SectionRef>> SectionAddresses;
  for (SectionRef Sec : Obj->sections())
    SectionAddresses.emplace_back(Sec.getAddress(), Sec);
  llvm::stable_sort(SectionAddresses, [](const auto &LHS, const auto &RHS) {
    if (LHS.first != RHS.first)
      return LHS.first < RHS.first;
    return LHS.second.getSize() < RHS.second.getSize();
  });

  // Linked executables (.exe and .dll files) typically don't include a real
  // symbol table but they might contain an export table.
  if (const auto *COFFObj = dyn_cast<COFFObjectFile>(Obj)) {
    for (const auto &ExportEntry : COFFObj->export_directories()) {
      StringRef Name;
      if (std::error_code EC = ExportEntry.getSymbolName(Name))
        reportError(errorCodeToError(EC), Obj->getFileName());
      if (Name.empty())
        continue;

      uint32_t RVA;
      if (std::error_code EC = ExportEntry.getExportRVA(RVA))
        reportError(errorCodeToError(EC), Obj->getFileName());

      uint64_t VA = COFFObj->getImageBase() + RVA;
      auto Sec = partition_point(
          SectionAddresses, [VA](const std::pair<uint64_t, SectionRef> &O) {
            return O.first <= VA;
          });
      if (Sec != SectionAddresses.begin()) {
        --Sec;
        AllSymbols[Sec->second].emplace_back(VA, Name, ELF::STT_NOTYPE);
      } else
        AbsoluteSymbols.emplace_back(VA, Name, ELF::STT_NOTYPE);
    }
  }

  // Sort all the symbols, this allows us to use a simple binary search to find
  // Multiple symbols can have the same address. Use a stable sort to stabilize
  // the output.
  StringSet<> FoundDisasmSymbolSet;
  for (std::pair<const SectionRef, SectionSymbolsTy> &SecSyms : AllSymbols)
    stable_sort(SecSyms.second);
  stable_sort(AbsoluteSymbols);

  for (const SectionRef &Section : ToolSectionFilter(*Obj)) {
    if (FilterSections.empty() && !DisassembleAll &&
        (!Section.isText() || Section.isVirtual()))
      continue;

    uint64_t SectionAddr = Section.getAddress();
    uint64_t SectSize = Section.getSize();
    if (!SectSize)
      continue;

    // Get the list of all the symbols in this section.
    SectionSymbolsTy &Symbols = AllSymbols[Section];
    std::vector<MappingSymbolPair> MappingSymbols;
    if (hasMappingSymbols(Obj)) {
      for (const auto &Symb : Symbols) {
        uint64_t Address = Symb.Addr;
        StringRef Name = Symb.Name;
        if (Name.startswith("$d"))
          MappingSymbols.emplace_back(Address - SectionAddr, 'd');
        if (Name.startswith("$x"))
          MappingSymbols.emplace_back(Address - SectionAddr, 'x');
        if (Name.startswith("$a"))
          MappingSymbols.emplace_back(Address - SectionAddr, 'a');
        if (Name.startswith("$t"))
          MappingSymbols.emplace_back(Address - SectionAddr, 't');
      }
    }

    llvm::sort(MappingSymbols);

    if (Obj->isELF() && Obj->getArch() == Triple::amdgcn) {
      // AMDGPU disassembler uses symbolizer for printing labels
      std::unique_ptr<MCRelocationInfo> RelInfo(
        TheTarget->createMCRelocationInfo(TripleName, Ctx));
      if (RelInfo) {
        std::unique_ptr<MCSymbolizer> Symbolizer(
          TheTarget->createMCSymbolizer(
            TripleName, nullptr, nullptr, &Symbols, &Ctx, std::move(RelInfo)));
        DisAsm->setSymbolizer(std::move(Symbolizer));
      }
    }

    StringRef SegmentName = "";
    if (MachO) {
      DataRefImpl DR = Section.getRawDataRefImpl();
      SegmentName = MachO->getSectionFinalSegmentName(DR);
    }

    StringRef SectionName = unwrapOrError(Section.getName(), Obj->getFileName());
    // If the section has no symbol at the start, just insert a dummy one.
    if (Symbols.empty() || Symbols[0].Addr != 0) {
      Symbols.insert(Symbols.begin(),
                     createDummySymbolInfo(Obj, SectionAddr, SectionName,
                                           Section.isText() ? ELF::STT_FUNC
                                                            : ELF::STT_OBJECT));
    }

    SmallString<40> Comments;
    raw_svector_ostream CommentStream(Comments);

    ArrayRef<uint8_t> Bytes = arrayRefFromStringRef(
        unwrapOrError(Section.getContents(), Obj->getFileName()));

    uint64_t VMAAdjustment = 0;
    if (shouldAdjustVA(Section))
      VMAAdjustment = AdjustVMA;

    uint64_t Size;
    uint64_t Index;
    bool PrintedSection = false;
    std::vector<RelocationRef> Rels = RelocMap[Section];
    std::vector<RelocationRef>::const_iterator RelCur = Rels.begin();
    std::vector<RelocationRef>::const_iterator RelEnd = Rels.end();
    // Disassemble symbol by symbol.
    for (unsigned SI = 0, SE = Symbols.size(); SI != SE; ++SI) {
      std::string SymbolName = Symbols[SI].Name.str();
      if (Demangle)
        SymbolName = demangle(SymbolName);

      // Skip if --disassemble-symbols is not empty and the symbol is not in
      // the list.
      if (!DisasmSymbolSet.empty() && !DisasmSymbolSet.count(SymbolName))
        continue;

      uint64_t Start = Symbols[SI].Addr;
      if (Start < SectionAddr || StopAddress <= Start)
        continue;
      else
        FoundDisasmSymbolSet.insert(SymbolName);

      // The end is the section end, the beginning of the next symbol, or
      // --stop-address.
      uint64_t End = std::min<uint64_t>(SectionAddr + SectSize, StopAddress);
      if (SI + 1 < SE)
        End = std::min(End, Symbols[SI + 1].Addr);
      if (Start >= End || End <= StartAddress)
        continue;
      Start -= SectionAddr;
      End -= SectionAddr;

      if (!PrintedSection) {
        PrintedSection = true;
        outs() << "\nDisassembly of section ";
        if (!SegmentName.empty())
          outs() << SegmentName << ",";
        outs() << SectionName << ":\n";
      }

      if (Obj->isELF() && Obj->getArch() == Triple::amdgcn) {
        if (Symbols[SI].Type == ELF::STT_AMDGPU_HSA_KERNEL) {
          // skip amd_kernel_code_t at the begining of kernel symbol (256 bytes)
          Start += 256;
        }
        if (SI == SE - 1 ||
            Symbols[SI + 1].Type == ELF::STT_AMDGPU_HSA_KERNEL) {
          // cut trailing zeroes at the end of kernel
          // cut up to 256 bytes
          const uint64_t EndAlign = 256;
          const auto Limit = End - (std::min)(EndAlign, End - Start);
          while (End > Limit &&
            *reinterpret_cast<const support::ulittle32_t*>(&Bytes[End - 4]) == 0)
            End -= 4;
        }
      }

      outs() << '\n';
      if (!NoLeadingAddr)
        outs() << format(Is64Bits ? "%016" PRIx64 " " : "%08" PRIx64 " ",
                         SectionAddr + Start + VMAAdjustment);
      if (Obj->isXCOFF() && SymbolDescription) {
        outs() << getXCOFFSymbolDescription(Symbols[SI], SymbolName) << ":\n";
      } else
        outs() << '<' << SymbolName << ">:\n";

      // Don't print raw contents of a virtual section. A virtual section
      // doesn't have any contents in the file.
      if (Section.isVirtual()) {
        outs() << "...\n";
        continue;
      }

      // Some targets (like WebAssembly) have a special prelude at the start
      // of each symbol.
      DisAsm->onSymbolStart(SymbolName, Size, Bytes.slice(Start, End - Start),
                            SectionAddr + Start, CommentStream);
      Start += Size;

      Index = Start;
      if (SectionAddr < StartAddress)
        Index = std::max<uint64_t>(Index, StartAddress - SectionAddr);

      // If there is a data/common symbol inside an ELF text section and we are
      // only disassembling text (applicable all architectures), we are in a
      // situation where we must print the data and not disassemble it.
      if (Obj->isELF() && !DisassembleAll && Section.isText()) {
        uint8_t SymTy = Symbols[SI].Type;
        if (SymTy == ELF::STT_OBJECT || SymTy == ELF::STT_COMMON) {
          dumpELFData(SectionAddr, Index, End, Bytes);
          Index = End;
        }
      }

      bool CheckARMELFData = hasMappingSymbols(Obj) &&
                             Symbols[SI].Type != ELF::STT_OBJECT &&
                             !DisassembleAll;
      bool DumpARMELFData = false;
      while (Index < End) {
        // ARM and AArch64 ELF binaries can interleave data and text in the
        // same section. We rely on the markers introduced to understand what
        // we need to dump. If the data marker is within a function, it is
        // denoted as a word/short etc.
        if (CheckARMELFData) {
          char Kind = getMappingSymbolKind(MappingSymbols, Index);
          DumpARMELFData = Kind == 'd';
          if (SecondarySTI) {
            if (Kind == 'a') {
              STI = PrimaryIsThumb ? SecondarySTI : PrimarySTI;
              DisAsm = PrimaryIsThumb ? SecondaryDisAsm : PrimaryDisAsm;
            } else if (Kind == 't') {
              STI = PrimaryIsThumb ? PrimarySTI : SecondarySTI;
              DisAsm = PrimaryIsThumb ? PrimaryDisAsm : SecondaryDisAsm;
            }
          }
        }

        if (DumpARMELFData) {
          Size = dumpARMELFData(SectionAddr, Index, End, Obj, Bytes,
                                MappingSymbols);
        } else {
          // When -z or --disassemble-zeroes are given we always dissasemble
          // them. Otherwise we might want to skip zero bytes we see.
          if (!DisassembleZeroes) {
            uint64_t MaxOffset = End - Index;
            // For --reloc: print zero blocks patched by relocations, so that
            // relocations can be shown in the dump.
            if (RelCur != RelEnd)
              MaxOffset = RelCur->getOffset() - Index;

            if (size_t N =
                    countSkippableZeroBytes(Bytes.slice(Index, MaxOffset))) {
              outs() << "\t\t..." << '\n';
              Index += N;
              continue;
            }
          }

          // Disassemble a real instruction or a data when disassemble all is
          // provided
          MCInst Inst;
          bool Disassembled =
              DisAsm->getInstruction(Inst, Size, Bytes.slice(Index),
                                     SectionAddr + Index, CommentStream);
          if (Size == 0)
            Size = 1;

          PIP.printInst(
              *IP, Disassembled ? &Inst : nullptr, Bytes.slice(Index, Size),
              {SectionAddr + Index + VMAAdjustment, Section.getIndex()}, outs(),
              "", *STI, &SP, Obj->getFileName(), &Rels);
          outs() << CommentStream.str();
          Comments.clear();

          // If disassembly has failed, avoid analysing invalid/incomplete
          // instruction information. Otherwise, try to resolve the target
          // address (jump target or memory operand address) and print it on the
          // right of the instruction.
          if (Disassembled && MIA) {
            uint64_t Target;
            bool PrintTarget =
                MIA->evaluateBranch(Inst, SectionAddr + Index, Size, Target);
            if (!PrintTarget)
              if (Optional<uint64_t> MaybeTarget =
                      MIA->evaluateMemoryOperandAddress(
                          Inst, SectionAddr + Index, Size)) {
                Target = *MaybeTarget;
                PrintTarget = true;
                outs() << "  # " << Twine::utohexstr(Target);
              }
            if (PrintTarget) {
              // In a relocatable object, the target's section must reside in
              // the same section as the call instruction or it is accessed
              // through a relocation.
              //
              // In a non-relocatable object, the target may be in any section.
              // In that case, locate the section(s) containing the target
              // address and find the symbol in one of those, if possible.
              //
              // N.B. We don't walk the relocations in the relocatable case yet.
              std::vector<const SectionSymbolsTy *> TargetSectionSymbols;
              if (!Obj->isRelocatableObject()) {
                auto It = llvm::partition_point(
                    SectionAddresses,
                    [=](const std::pair<uint64_t, SectionRef> &O) {
                      return O.first <= Target;
                    });
                uint64_t TargetSecAddr = 0;
                while (It != SectionAddresses.begin()) {
                  --It;
                  if (TargetSecAddr == 0)
                    TargetSecAddr = It->first;
                  if (It->first != TargetSecAddr)
                    break;
                  TargetSectionSymbols.push_back(&AllSymbols[It->second]);
                }
              } else {
                TargetSectionSymbols.push_back(&Symbols);
              }
              TargetSectionSymbols.push_back(&AbsoluteSymbols);

              // Find the last symbol in the first candidate section whose
              // offset is less than or equal to the target. If there are no
              // such symbols, try in the next section and so on, before finally
              // using the nearest preceding absolute symbol (if any), if there
              // are no other valid symbols.
              const SymbolInfoTy *TargetSym = nullptr;
              for (const SectionSymbolsTy *TargetSymbols :
                   TargetSectionSymbols) {
                auto It = llvm::partition_point(
                    *TargetSymbols,
                    [=](const SymbolInfoTy &O) { return O.Addr <= Target; });
                if (It != TargetSymbols->begin()) {
                  TargetSym = &*(It - 1);
                  break;
                }
              }

              if (TargetSym != nullptr) {
                uint64_t TargetAddress = TargetSym->Addr;
                std::string TargetName = TargetSym->Name.str();
                if (Demangle)
                  TargetName = demangle(TargetName);

                outs() << " <" << TargetName;
                uint64_t Disp = Target - TargetAddress;
                if (Disp)
                  outs() << "+0x" << Twine::utohexstr(Disp);
                outs() << '>';
              }
            }
          }
        }
        outs() << "\n";

        // Hexagon does this in pretty printer
        if (Obj->getArch() != Triple::hexagon) {
          // Print relocation for instruction and data.
          while (RelCur != RelEnd) {
            uint64_t Offset = RelCur->getOffset();
            // If this relocation is hidden, skip it.
            if (getHidden(*RelCur) || SectionAddr + Offset < StartAddress) {
              ++RelCur;
              continue;
            }

            // Stop when RelCur's offset is past the disassembled
            // instruction/data. Note that it's possible the disassembled data
            // is not the complete data: we might see the relocation printed in
            // the middle of the data, but this matches the binutils objdump
            // output.
            if (Offset >= Index + Size)
              break;

            // When --adjust-vma is used, update the address printed.
            if (RelCur->getSymbol() != Obj->symbol_end()) {
              Expected<section_iterator> SymSI =
                  RelCur->getSymbol()->getSection();
              if (SymSI && *SymSI != Obj->section_end() &&
                  shouldAdjustVA(**SymSI))
                Offset += AdjustVMA;
            }

            printRelocation(Obj->getFileName(), *RelCur, SectionAddr + Offset,
                            Is64Bits);
            ++RelCur;
          }
        }

        Index += Size;
      }
    }
  }
  StringSet<> MissingDisasmSymbolSet =
      set_difference(DisasmSymbolSet, FoundDisasmSymbolSet);
  for (StringRef Sym : MissingDisasmSymbolSet.keys())
    reportWarning("failed to disassemble missing symbol " + Sym, FileName);
}

static void disassembleObject(const ObjectFile *Obj, bool InlineRelocs) {
  const Target *TheTarget = getTarget(Obj);

  // Package up features to be passed to target/subtarget
  SubtargetFeatures Features = Obj->getFeatures();
  if (!MAttrs.empty())
    for (unsigned I = 0; I != MAttrs.size(); ++I)
      Features.AddFeature(MAttrs[I]);

  std::unique_ptr<const MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    reportError(Obj->getFileName(),
                "no register info for target " + TripleName);

  // Set up disassembler.
  MCTargetOptions MCOptions;
  std::unique_ptr<const MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  if (!AsmInfo)
    reportError(Obj->getFileName(),
                "no assembly info for target " + TripleName);
  std::unique_ptr<const MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, MCPU, Features.getString()));
  if (!STI)
    reportError(Obj->getFileName(),
                "no subtarget info for target " + TripleName);
  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII)
    reportError(Obj->getFileName(),
                "no instruction info for target " + TripleName);
  MCObjectFileInfo MOFI;
  MCContext Ctx(AsmInfo.get(), MRI.get(), &MOFI);
  // FIXME: for now initialize MCObjectFileInfo with default values
  MOFI.InitMCObjectFileInfo(Triple(TripleName), false, Ctx);

  std::unique_ptr<MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*STI, Ctx));
  if (!DisAsm)
    reportError(Obj->getFileName(), "no disassembler for target " + TripleName);

  // If we have an ARM object file, we need a second disassembler, because
  // ARM CPUs have two different instruction sets: ARM mode, and Thumb mode.
  // We use mapping symbols to switch between the two assemblers, where
  // appropriate.
  std::unique_ptr<MCDisassembler> SecondaryDisAsm;
  std::unique_ptr<const MCSubtargetInfo> SecondarySTI;
  if (isArmElf(Obj) && !STI->checkFeatures("+mclass")) {
    if (STI->checkFeatures("+thumb-mode"))
      Features.AddFeature("-thumb-mode");
    else
      Features.AddFeature("+thumb-mode");
    SecondarySTI.reset(TheTarget->createMCSubtargetInfo(TripleName, MCPU,
                                                        Features.getString()));
    SecondaryDisAsm.reset(TheTarget->createMCDisassembler(*SecondarySTI, Ctx));
  }

  std::unique_ptr<const MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
      Triple(TripleName), AsmPrinterVariant, *AsmInfo, *MII, *MRI));
  if (!IP)
    reportError(Obj->getFileName(),
                "no instruction printer for target " + TripleName);
  IP->setPrintImmHex(PrintImmHex);
  IP->setPrintBranchImmAsAddress(true);

  PrettyPrinter &PIP = selectPrettyPrinter(Triple(TripleName));
  SourcePrinter SP(Obj, TheTarget->getName());

  for (StringRef Opt : DisassemblerOptions)
    if (!IP->applyTargetSpecificCLOption(Opt))
      reportError(Obj->getFileName(),
                  "Unrecognized disassembler option: " + Opt);

  disassembleObject(TheTarget, Obj, Ctx, DisAsm.get(), SecondaryDisAsm.get(),
                    MIA.get(), IP.get(), STI.get(), SecondarySTI.get(), PIP,
                    SP, InlineRelocs);
}

void printRelocations(const ObjectFile *Obj) {
  StringRef Fmt = Obj->getBytesInAddress() > 4 ? "%016" PRIx64 :
                                                 "%08" PRIx64;
  // Regular objdump doesn't print relocations in non-relocatable object
  // files.
  if (!Obj->isRelocatableObject())
    return;

  // Build a mapping from relocation target to a vector of relocation
  // sections. Usually, there is an only one relocation section for
  // each relocated section.
  MapVector<SectionRef, std::vector<SectionRef>> SecToRelSec;
  uint64_t Ndx;
  for (const SectionRef &Section : ToolSectionFilter(*Obj, &Ndx)) {
    if (Section.relocation_begin() == Section.relocation_end())
      continue;
    Expected<section_iterator> SecOrErr = Section.getRelocatedSection();
    if (!SecOrErr)
      reportError(Obj->getFileName(),
                  "section (" + Twine(Ndx) +
                      "): unable to get a relocation target: " +
                      toString(SecOrErr.takeError()));
    SecToRelSec[**SecOrErr].push_back(Section);
  }

  for (std::pair<SectionRef, std::vector<SectionRef>> &P : SecToRelSec) {
    StringRef SecName = unwrapOrError(P.first.getName(), Obj->getFileName());
    outs() << "RELOCATION RECORDS FOR [" << SecName << "]:\n";
    uint32_t OffsetPadding = (Obj->getBytesInAddress() > 4 ? 16 : 8);
    uint32_t TypePadding = 24;
    outs() << left_justify("OFFSET", OffsetPadding) << " "
           << left_justify("TYPE", TypePadding) << " "
           << "VALUE\n";

    for (SectionRef Section : P.second) {
      for (const RelocationRef &Reloc : Section.relocations()) {
        uint64_t Address = Reloc.getOffset();
        SmallString<32> RelocName;
        SmallString<32> ValueStr;
        if (Address < StartAddress || Address > StopAddress || getHidden(Reloc))
          continue;
        Reloc.getTypeName(RelocName);
        if (Error E = getRelocationValueString(Reloc, ValueStr))
          reportError(std::move(E), Obj->getFileName());

        outs() << format(Fmt.data(), Address) << " "
               << left_justify(RelocName, TypePadding) << " " << ValueStr
               << "\n";
      }
    }
    outs() << "\n";
  }
}

void printDynamicRelocations(const ObjectFile *Obj) {
  // For the moment, this option is for ELF only
  if (!Obj->isELF())
    return;

  const auto *Elf = dyn_cast<ELFObjectFileBase>(Obj);
  if (!Elf || Elf->getEType() != ELF::ET_DYN) {
    reportError(Obj->getFileName(), "not a dynamic object");
    return;
  }

  std::vector<SectionRef> DynRelSec = Obj->dynamic_relocation_sections();
  if (DynRelSec.empty())
    return;

  outs() << "DYNAMIC RELOCATION RECORDS\n";
  StringRef Fmt = Obj->getBytesInAddress() > 4 ? "%016" PRIx64 : "%08" PRIx64;
  for (const SectionRef &Section : DynRelSec)
    for (const RelocationRef &Reloc : Section.relocations()) {
      uint64_t Address = Reloc.getOffset();
      SmallString<32> RelocName;
      SmallString<32> ValueStr;
      Reloc.getTypeName(RelocName);
      if (Error E = getRelocationValueString(Reloc, ValueStr))
        reportError(std::move(E), Obj->getFileName());
      outs() << format(Fmt.data(), Address) << " " << RelocName << " "
             << ValueStr << "\n";
    }
}

// Returns true if we need to show LMA column when dumping section headers. We
// show it only when the platform is ELF and either we have at least one section
// whose VMA and LMA are different and/or when --show-lma flag is used.
static bool shouldDisplayLMA(const ObjectFile *Obj) {
  if (!Obj->isELF())
    return false;
  for (const SectionRef &S : ToolSectionFilter(*Obj))
    if (S.getAddress() != getELFSectionLMA(S))
      return true;
  return ShowLMA;
}

static size_t getMaxSectionNameWidth(const ObjectFile *Obj) {
  // Default column width for names is 13 even if no names are that long.
  size_t MaxWidth = 13;
  for (const SectionRef &Section : ToolSectionFilter(*Obj)) {
    StringRef Name = unwrapOrError(Section.getName(), Obj->getFileName());
    MaxWidth = std::max(MaxWidth, Name.size());
  }
  return MaxWidth;
}

void printSectionHeaders(const ObjectFile *Obj) {
  size_t NameWidth = getMaxSectionNameWidth(Obj);
  size_t AddressWidth = 2 * Obj->getBytesInAddress();
  bool HasLMAColumn = shouldDisplayLMA(Obj);
  if (HasLMAColumn)
    outs() << "Sections:\n"
              "Idx "
           << left_justify("Name", NameWidth) << " Size     "
           << left_justify("VMA", AddressWidth) << " "
           << left_justify("LMA", AddressWidth) << " Type\n";
  else
    outs() << "Sections:\n"
              "Idx "
           << left_justify("Name", NameWidth) << " Size     "
           << left_justify("VMA", AddressWidth) << " Type\n";

  uint64_t Idx;
  for (const SectionRef &Section : ToolSectionFilter(*Obj, &Idx)) {
    StringRef Name = unwrapOrError(Section.getName(), Obj->getFileName());
    uint64_t VMA = Section.getAddress();
    if (shouldAdjustVA(Section))
      VMA += AdjustVMA;

    uint64_t Size = Section.getSize();

    std::string Type = Section.isText() ? "TEXT" : "";
    if (Section.isData())
      Type += Type.empty() ? "DATA" : " DATA";
    if (Section.isBSS())
      Type += Type.empty() ? "BSS" : " BSS";

    if (HasLMAColumn)
      outs() << format("%3" PRIu64 " %-*s %08" PRIx64 " ", Idx, NameWidth,
                       Name.str().c_str(), Size)
             << format_hex_no_prefix(VMA, AddressWidth) << " "
             << format_hex_no_prefix(getELFSectionLMA(Section), AddressWidth)
             << " " << Type << "\n";
    else
      outs() << format("%3" PRIu64 " %-*s %08" PRIx64 " ", Idx, NameWidth,
                       Name.str().c_str(), Size)
             << format_hex_no_prefix(VMA, AddressWidth) << " " << Type << "\n";
  }
  outs() << "\n";
}

void printSectionContents(const ObjectFile *Obj) {
  for (const SectionRef &Section : ToolSectionFilter(*Obj)) {
    StringRef Name = unwrapOrError(Section.getName(), Obj->getFileName());
    uint64_t BaseAddr = Section.getAddress();
    uint64_t Size = Section.getSize();
    if (!Size)
      continue;

    outs() << "Contents of section " << Name << ":\n";
    if (Section.isBSS()) {
      outs() << format("<skipping contents of bss section at [%04" PRIx64
                       ", %04" PRIx64 ")>\n",
                       BaseAddr, BaseAddr + Size);
      continue;
    }

    StringRef Contents = unwrapOrError(Section.getContents(), Obj->getFileName());

    // Dump out the content as hex and printable ascii characters.
    for (std::size_t Addr = 0, End = Contents.size(); Addr < End; Addr += 16) {
      outs() << format(" %04" PRIx64 " ", BaseAddr + Addr);
      // Dump line of hex.
      for (std::size_t I = 0; I < 16; ++I) {
        if (I != 0 && I % 4 == 0)
          outs() << ' ';
        if (Addr + I < End)
          outs() << hexdigit((Contents[Addr + I] >> 4) & 0xF, true)
                 << hexdigit(Contents[Addr + I] & 0xF, true);
        else
          outs() << "  ";
      }
      // Print ascii.
      outs() << "  ";
      for (std::size_t I = 0; I < 16 && Addr + I < End; ++I) {
        if (isPrint(static_cast<unsigned char>(Contents[Addr + I]) & 0xFF))
          outs() << Contents[Addr + I];
        else
          outs() << ".";
      }
      outs() << "\n";
    }
  }
}

void printSymbolTable(const ObjectFile *O, StringRef ArchiveName,
                      StringRef ArchitectureName, bool DumpDynamic) {
  if (O->isCOFF() && !DumpDynamic) {
    outs() << "SYMBOL TABLE:\n";
    printCOFFSymbolTable(cast<const COFFObjectFile>(O));
    return;
  }

  const StringRef FileName = O->getFileName();

  if (!DumpDynamic) {
    outs() << "SYMBOL TABLE:\n";
    for (auto I = O->symbol_begin(); I != O->symbol_end(); ++I)
      printSymbol(O, *I, FileName, ArchiveName, ArchitectureName, DumpDynamic);
    return;
  }

  outs() << "DYNAMIC SYMBOL TABLE:\n";
  if (!O->isELF()) {
    reportWarning(
        "this operation is not currently supported for this file format",
        FileName);
    return;
  }

  const ELFObjectFileBase *ELF = cast<const ELFObjectFileBase>(O);
  for (auto I = ELF->getDynamicSymbolIterators().begin();
       I != ELF->getDynamicSymbolIterators().end(); ++I)
    printSymbol(O, *I, FileName, ArchiveName, ArchitectureName, DumpDynamic);
}

void printSymbol(const ObjectFile *O, const SymbolRef &Symbol,
                 StringRef FileName, StringRef ArchiveName,
                 StringRef ArchitectureName, bool DumpDynamic) {
  const MachOObjectFile *MachO = dyn_cast<const MachOObjectFile>(O);
  uint64_t Address = unwrapOrError(Symbol.getAddress(), FileName, ArchiveName,
                                   ArchitectureName);
  if ((Address < StartAddress) || (Address > StopAddress))
    return;
  SymbolRef::Type Type =
      unwrapOrError(Symbol.getType(), FileName, ArchiveName, ArchitectureName);
  uint32_t Flags =
      unwrapOrError(Symbol.getFlags(), FileName, ArchiveName, ArchitectureName);

  // Don't ask a Mach-O STAB symbol for its section unless you know that
  // STAB symbol's section field refers to a valid section index. Otherwise
  // the symbol may error trying to load a section that does not exist.
  bool IsSTAB = false;
  if (MachO) {
    DataRefImpl SymDRI = Symbol.getRawDataRefImpl();
    uint8_t NType =
        (MachO->is64Bit() ? MachO->getSymbol64TableEntry(SymDRI).n_type
                          : MachO->getSymbolTableEntry(SymDRI).n_type);
    if (NType & MachO::N_STAB)
      IsSTAB = true;
  }
  section_iterator Section = IsSTAB
                                 ? O->section_end()
                                 : unwrapOrError(Symbol.getSection(), FileName,
                                                 ArchiveName, ArchitectureName);

  StringRef Name;
  if (Type == SymbolRef::ST_Debug && Section != O->section_end()) {
    if (Expected<StringRef> NameOrErr = Section->getName())
      Name = *NameOrErr;
    else
      consumeError(NameOrErr.takeError());

  } else {
    Name = unwrapOrError(Symbol.getName(), FileName, ArchiveName,
                         ArchitectureName);
  }

  bool Global = Flags & SymbolRef::SF_Global;
  bool Weak = Flags & SymbolRef::SF_Weak;
  bool Absolute = Flags & SymbolRef::SF_Absolute;
  bool Common = Flags & SymbolRef::SF_Common;
  bool Hidden = Flags & SymbolRef::SF_Hidden;

  char GlobLoc = ' ';
  if ((Section != O->section_end() || Absolute) && !Weak)
    GlobLoc = Global ? 'g' : 'l';
  char IFunc = ' ';
  if (O->isELF()) {
    if (ELFSymbolRef(Symbol).getELFType() == ELF::STT_GNU_IFUNC)
      IFunc = 'i';
    if (ELFSymbolRef(Symbol).getBinding() == ELF::STB_GNU_UNIQUE)
      GlobLoc = 'u';
  }

  char Debug = ' ';
  if (DumpDynamic)
    Debug = 'D';
  else if (Type == SymbolRef::ST_Debug || Type == SymbolRef::ST_File)
    Debug = 'd';

  char FileFunc = ' ';
  if (Type == SymbolRef::ST_File)
    FileFunc = 'f';
  else if (Type == SymbolRef::ST_Function)
    FileFunc = 'F';
  else if (Type == SymbolRef::ST_Data)
    FileFunc = 'O';

  const char *Fmt = O->getBytesInAddress() > 4 ? "%016" PRIx64 : "%08" PRIx64;

  outs() << format(Fmt, Address) << " "
         << GlobLoc            // Local -> 'l', Global -> 'g', Neither -> ' '
         << (Weak ? 'w' : ' ') // Weak?
         << ' '                // Constructor. Not supported yet.
         << ' '                // Warning. Not supported yet.
         << IFunc              // Indirect reference to another symbol.
         << Debug              // Debugging (d) or dynamic (D) symbol.
         << FileFunc           // Name of function (F), file (f) or object (O).
         << ' ';
  if (Absolute) {
    outs() << "*ABS*";
  } else if (Common) {
    outs() << "*COM*";
  } else if (Section == O->section_end()) {
    outs() << "*UND*";
  } else {
    if (MachO) {
      DataRefImpl DR = Section->getRawDataRefImpl();
      StringRef SegmentName = MachO->getSectionFinalSegmentName(DR);
      outs() << SegmentName << ",";
    }
    StringRef SectionName = unwrapOrError(Section->getName(), FileName);
    outs() << SectionName;
  }

  if (Common || O->isELF()) {
    uint64_t Val =
        Common ? Symbol.getAlignment() : ELFSymbolRef(Symbol).getSize();
    outs() << '\t' << format(Fmt, Val);
  }

  if (O->isELF()) {
    uint8_t Other = ELFSymbolRef(Symbol).getOther();
    switch (Other) {
    case ELF::STV_DEFAULT:
      break;
    case ELF::STV_INTERNAL:
      outs() << " .internal";
      break;
    case ELF::STV_HIDDEN:
      outs() << " .hidden";
      break;
    case ELF::STV_PROTECTED:
      outs() << " .protected";
      break;
    default:
      outs() << format(" 0x%02x", Other);
      break;
    }
  } else if (Hidden) {
    outs() << " .hidden";
  }

  if (Demangle)
    outs() << ' ' << demangle(std::string(Name)) << '\n';
  else
    outs() << ' ' << Name << '\n';
}

static void printUnwindInfo(const ObjectFile *O) {
  outs() << "Unwind info:\n\n";

  if (const COFFObjectFile *Coff = dyn_cast<COFFObjectFile>(O))
    printCOFFUnwindInfo(Coff);
  else if (const MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(O))
    printMachOUnwindInfo(MachO);
  else
    // TODO: Extract DWARF dump tool to objdump.
    WithColor::error(errs(), ToolName)
        << "This operation is only currently supported "
           "for COFF and MachO object files.\n";
}

/// Dump the raw contents of the __clangast section so the output can be piped
/// into llvm-bcanalyzer.
void printRawClangAST(const ObjectFile *Obj) {
  if (outs().is_displayed()) {
    WithColor::error(errs(), ToolName)
        << "The -raw-clang-ast option will dump the raw binary contents of "
           "the clang ast section.\n"
           "Please redirect the output to a file or another program such as "
           "llvm-bcanalyzer.\n";
    return;
  }

  StringRef ClangASTSectionName("__clangast");
  if (Obj->isCOFF()) {
    ClangASTSectionName = "clangast";
  }

  Optional<object::SectionRef> ClangASTSection;
  for (auto Sec : ToolSectionFilter(*Obj)) {
    StringRef Name;
    if (Expected<StringRef> NameOrErr = Sec.getName())
      Name = *NameOrErr;
    else
      consumeError(NameOrErr.takeError());

    if (Name == ClangASTSectionName) {
      ClangASTSection = Sec;
      break;
    }
  }
  if (!ClangASTSection)
    return;

  StringRef ClangASTContents = unwrapOrError(
      ClangASTSection.getValue().getContents(), Obj->getFileName());
  outs().write(ClangASTContents.data(), ClangASTContents.size());
}

static void printFaultMaps(const ObjectFile *Obj) {
  StringRef FaultMapSectionName;

  if (Obj->isELF()) {
    FaultMapSectionName = ".llvm_faultmaps";
  } else if (Obj->isMachO()) {
    FaultMapSectionName = "__llvm_faultmaps";
  } else {
    WithColor::error(errs(), ToolName)
        << "This operation is only currently supported "
           "for ELF and Mach-O executable files.\n";
    return;
  }

  Optional<object::SectionRef> FaultMapSection;

  for (auto Sec : ToolSectionFilter(*Obj)) {
    StringRef Name;
    if (Expected<StringRef> NameOrErr = Sec.getName())
      Name = *NameOrErr;
    else
      consumeError(NameOrErr.takeError());

    if (Name == FaultMapSectionName) {
      FaultMapSection = Sec;
      break;
    }
  }

  outs() << "FaultMap table:\n";

  if (!FaultMapSection.hasValue()) {
    outs() << "<not found>\n";
    return;
  }

  StringRef FaultMapContents =
      unwrapOrError(FaultMapSection.getValue().getContents(), Obj->getFileName());
  FaultMapParser FMP(FaultMapContents.bytes_begin(),
                     FaultMapContents.bytes_end());

  outs() << FMP;
}

static void printPrivateFileHeaders(const ObjectFile *O, bool OnlyFirst) {
  if (O->isELF()) {
    printELFFileHeader(O);
    printELFDynamicSection(O);
    printELFSymbolVersionInfo(O);
    return;
  }
  if (O->isCOFF())
    return printCOFFFileHeader(O);
  if (O->isWasm())
    return printWasmFileHeader(O);
  if (O->isMachO()) {
    printMachOFileHeader(O);
    if (!OnlyFirst)
      printMachOLoadCommands(O);
    return;
  }
  reportError(O->getFileName(), "Invalid/Unsupported object file format");
}

static void printFileHeaders(const ObjectFile *O) {
  if (!O->isELF() && !O->isCOFF())
    reportError(O->getFileName(), "Invalid/Unsupported object file format");

  Triple::ArchType AT = O->getArch();
  outs() << "architecture: " << Triple::getArchTypeName(AT) << "\n";
  uint64_t Address = unwrapOrError(O->getStartAddress(), O->getFileName());

  StringRef Fmt = O->getBytesInAddress() > 4 ? "%016" PRIx64 : "%08" PRIx64;
  outs() << "start address: "
         << "0x" << format(Fmt.data(), Address) << "\n\n";
}

static void printArchiveChild(StringRef Filename, const Archive::Child &C) {
  Expected<sys::fs::perms> ModeOrErr = C.getAccessMode();
  if (!ModeOrErr) {
    WithColor::error(errs(), ToolName) << "ill-formed archive entry.\n";
    consumeError(ModeOrErr.takeError());
    return;
  }
  sys::fs::perms Mode = ModeOrErr.get();
  outs() << ((Mode & sys::fs::owner_read) ? "r" : "-");
  outs() << ((Mode & sys::fs::owner_write) ? "w" : "-");
  outs() << ((Mode & sys::fs::owner_exe) ? "x" : "-");
  outs() << ((Mode & sys::fs::group_read) ? "r" : "-");
  outs() << ((Mode & sys::fs::group_write) ? "w" : "-");
  outs() << ((Mode & sys::fs::group_exe) ? "x" : "-");
  outs() << ((Mode & sys::fs::others_read) ? "r" : "-");
  outs() << ((Mode & sys::fs::others_write) ? "w" : "-");
  outs() << ((Mode & sys::fs::others_exe) ? "x" : "-");

  outs() << " ";

  outs() << format("%d/%d %6" PRId64 " ", unwrapOrError(C.getUID(), Filename),
                   unwrapOrError(C.getGID(), Filename),
                   unwrapOrError(C.getRawSize(), Filename));

  StringRef RawLastModified = C.getRawLastModified();
  unsigned Seconds;
  if (RawLastModified.getAsInteger(10, Seconds))
    outs() << "(date: \"" << RawLastModified
           << "\" contains non-decimal chars) ";
  else {
    // Since ctime(3) returns a 26 character string of the form:
    // "Sun Sep 16 01:03:52 1973\n\0"
    // just print 24 characters.
    time_t t = Seconds;
    outs() << format("%.24s ", ctime(&t));
  }

  StringRef Name = "";
  Expected<StringRef> NameOrErr = C.getName();
  if (!NameOrErr) {
    consumeError(NameOrErr.takeError());
    Name = unwrapOrError(C.getRawName(), Filename);
  } else {
    Name = NameOrErr.get();
  }
  outs() << Name << "\n";
}

// For ELF only now.
static bool shouldWarnForInvalidStartStopAddress(ObjectFile *Obj) {
  if (const auto *Elf = dyn_cast<ELFObjectFileBase>(Obj)) {
    if (Elf->getEType() != ELF::ET_REL)
      return true;
  }
  return false;
}

static void checkForInvalidStartStopAddress(ObjectFile *Obj,
                                            uint64_t Start, uint64_t Stop) {
  if (!shouldWarnForInvalidStartStopAddress(Obj))
    return;

  for (const SectionRef &Section : Obj->sections())
    if (ELFSectionRef(Section).getFlags() & ELF::SHF_ALLOC) {
      uint64_t BaseAddr = Section.getAddress();
      uint64_t Size = Section.getSize();
      if ((Start < BaseAddr + Size) && Stop > BaseAddr)
        return;
    }

  if (StartAddress.getNumOccurrences() == 0)
    reportWarning("no section has address less than 0x" +
                      Twine::utohexstr(Stop) + " specified by --stop-address",
                  Obj->getFileName());
  else if (StopAddress.getNumOccurrences() == 0)
    reportWarning("no section has address greater than or equal to 0x" +
                      Twine::utohexstr(Start) + " specified by --start-address",
                  Obj->getFileName());
  else
    reportWarning("no section overlaps the range [0x" +
                      Twine::utohexstr(Start) + ",0x" + Twine::utohexstr(Stop) +
                      ") specified by --start-address/--stop-address",
                  Obj->getFileName());
}

static void dumpObject(ObjectFile *O, const Archive *A = nullptr,
                       const Archive::Child *C = nullptr) {
  // Avoid other output when using a raw option.
  if (!RawClangAST) {
    outs() << '\n';
    if (A)
      outs() << A->getFileName() << "(" << O->getFileName() << ")";
    else
      outs() << O->getFileName();
    outs() << ":\tfile format " << O->getFileFormatName().lower() << "\n\n";
  }

  if (StartAddress.getNumOccurrences() || StopAddress.getNumOccurrences())
    checkForInvalidStartStopAddress(O, StartAddress, StopAddress);

  // Note: the order here matches GNU objdump for compatability.
  StringRef ArchiveName = A ? A->getFileName() : "";
  if (ArchiveHeaders && !MachOOpt && C)
    printArchiveChild(ArchiveName, *C);
  if (FileHeaders)
    printFileHeaders(O);
  if (PrivateHeaders || FirstPrivateHeader)
    printPrivateFileHeaders(O, FirstPrivateHeader);
  if (SectionHeaders)
    printSectionHeaders(O);
  if (SymbolTable)
    printSymbolTable(O, ArchiveName);
  if (DynamicSymbolTable)
    printSymbolTable(O, ArchiveName, /*ArchitectureName=*/"",
                     /*DumpDynamic=*/true);
  if (DwarfDumpType != DIDT_Null) {
    std::unique_ptr<DIContext> DICtx = DWARFContext::create(*O);
    // Dump the complete DWARF structure.
    DIDumpOptions DumpOpts;
    DumpOpts.DumpType = DwarfDumpType;
    DICtx->dump(outs(), DumpOpts);
  }
  if (Relocations && !Disassemble)
    printRelocations(O);
  if (DynamicRelocations)
    printDynamicRelocations(O);
  if (SectionContents)
    printSectionContents(O);
  if (Disassemble)
    disassembleObject(O, Relocations);
  if (UnwindInfo)
    printUnwindInfo(O);

  // Mach-O specific options:
  if (ExportsTrie)
    printExportsTrie(O);
  if (Rebase)
    printRebaseTable(O);
  if (Bind)
    printBindTable(O);
  if (LazyBind)
    printLazyBindTable(O);
  if (WeakBind)
    printWeakBindTable(O);

  // Other special sections:
  if (RawClangAST)
    printRawClangAST(O);
  if (FaultMapSection)
    printFaultMaps(O);
}

static void dumpObject(const COFFImportFile *I, const Archive *A,
                       const Archive::Child *C = nullptr) {
  StringRef ArchiveName = A ? A->getFileName() : "";

  // Avoid other output when using a raw option.
  if (!RawClangAST)
    outs() << '\n'
           << ArchiveName << "(" << I->getFileName() << ")"
           << ":\tfile format COFF-import-file"
           << "\n\n";

  if (ArchiveHeaders && !MachOOpt && C)
    printArchiveChild(ArchiveName, *C);
  if (SymbolTable)
    printCOFFSymbolTable(I);
}

/// Dump each object file in \a a;
static void dumpArchive(const Archive *A) {
  Error Err = Error::success();
  unsigned I = -1;
  for (auto &C : A->children(Err)) {
    ++I;
    Expected<std::unique_ptr<Binary>> ChildOrErr = C.getAsBinary();
    if (!ChildOrErr) {
      if (auto E = isNotObjectErrorInvalidFileType(ChildOrErr.takeError()))
        reportError(std::move(E), getFileNameForError(C, I), A->getFileName());
      continue;
    }
    if (ObjectFile *O = dyn_cast<ObjectFile>(&*ChildOrErr.get()))
      dumpObject(O, A, &C);
    else if (COFFImportFile *I = dyn_cast<COFFImportFile>(&*ChildOrErr.get()))
      dumpObject(I, A, &C);
    else
      reportError(errorCodeToError(object_error::invalid_file_type),
                  A->getFileName());
  }
  if (Err)
    reportError(std::move(Err), A->getFileName());
}

/// Open file and figure out how to dump it.
static void dumpInput(StringRef file) {
  // If we are using the Mach-O specific object file parser, then let it parse
  // the file and process the command line options.  So the -arch flags can
  // be used to select specific slices, etc.
  if (MachOOpt) {
    parseInputMachO(file);
    return;
  }

  // Attempt to open the binary.
  OwningBinary<Binary> OBinary = unwrapOrError(createBinary(file), file);
  Binary &Binary = *OBinary.getBinary();

  if (Archive *A = dyn_cast<Archive>(&Binary))
    dumpArchive(A);
  else if (ObjectFile *O = dyn_cast<ObjectFile>(&Binary))
    dumpObject(O);
  else if (MachOUniversalBinary *UB = dyn_cast<MachOUniversalBinary>(&Binary))
    parseInputMachO(UB);
  else
    reportError(errorCodeToError(object_error::invalid_file_type), file);
}
} // namespace llvm

int main(int argc, char **argv) {
  using namespace llvm;
  InitLLVM X(argc, argv);
  const cl::OptionCategory *OptionFilters[] = {&ObjdumpCat, &MachOCat};
  cl::HideUnrelatedOptions(OptionFilters);

  // Initialize targets and assembly printers/parsers.
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::ParseCommandLineOptions(argc, argv, "llvm object file dumper\n", nullptr,
                              /*EnvVar=*/nullptr,
                              /*LongOptionsUseDoubleDash=*/true);

  if (StartAddress >= StopAddress)
    reportCmdLineError("start address should be less than stop address");

  ToolName = argv[0];

  // Defaults to a.out if no filenames specified.
  if (InputFilenames.empty())
    InputFilenames.push_back("a.out");

  if (AllHeaders)
    ArchiveHeaders = FileHeaders = PrivateHeaders = Relocations =
        SectionHeaders = SymbolTable = true;

  if (DisassembleAll || PrintSource || PrintLines ||
      !DisassembleSymbols.empty())
    Disassemble = true;

  if (!ArchiveHeaders && !Disassemble && DwarfDumpType == DIDT_Null &&
      !DynamicRelocations && !FileHeaders && !PrivateHeaders && !RawClangAST &&
      !Relocations && !SectionHeaders && !SectionContents && !SymbolTable &&
      !DynamicSymbolTable && !UnwindInfo && !FaultMapSection &&
      !(MachOOpt &&
        (Bind || DataInCode || DylibId || DylibsUsed || ExportsTrie ||
         FirstPrivateHeader || IndirectSymbols || InfoPlist || LazyBind ||
         LinkOptHints || ObjcMetaData || Rebase || UniversalHeaders ||
         WeakBind || !FilterSections.empty()))) {
    cl::PrintHelpMessage();
    return 2;
  }

  DisasmSymbolSet.insert(DisassembleSymbols.begin(), DisassembleSymbols.end());

  llvm::for_each(InputFilenames, dumpInput);

  warnOnNoMatchForSections();

  return EXIT_SUCCESS;
}
