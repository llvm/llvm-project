//===- llvm-pdbdump.cpp - Dump debug info from a PDB file -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Dumps debug information present in PDB files.
//
//===----------------------------------------------------------------------===//

#include "llvm-pdbdump.h"

#include "Analyze.h"
#include "Diff.h"
#include "LLVMOutputStyle.h"
#include "LinePrinter.h"
#include "OutputStyle.h"
#include "PrettyCompilandDumper.h"
#include "PrettyExternalSymbolDumper.h"
#include "PrettyFunctionDumper.h"
#include "PrettyTypeDumper.h"
#include "PrettyVariableDumper.h"
#include "YAMLOutputStyle.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/config.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugFileChecksumFragment.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugInlineeLinesFragment.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugLineFragment.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/PDB/GenericError.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBRawSymbol.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptorBuilder.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/DbiStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/PDBFileBuilder.h"
#include "llvm/DebugInfo/PDB/Native/PDBStringTableBuilder.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/Native/TpiStreamBuilder.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolThunk.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/COM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

namespace opts {

cl::SubCommand RawSubcommand("raw", "Dump raw structure of the PDB file");
cl::SubCommand
    PrettySubcommand("pretty",
                     "Dump semantic information about types and symbols");

cl::SubCommand DiffSubcommand("diff", "Diff the contents of 2 PDB files");

cl::SubCommand
    YamlToPdbSubcommand("yaml2pdb",
                        "Generate a PDB file from a YAML description");
cl::SubCommand
    PdbToYamlSubcommand("pdb2yaml",
                        "Generate a detailed YAML description of a PDB File");

cl::SubCommand
    AnalyzeSubcommand("analyze",
                      "Analyze various aspects of a PDB's structure");

cl::OptionCategory TypeCategory("Symbol Type Options");
cl::OptionCategory FilterCategory("Filtering and Sorting Options");
cl::OptionCategory OtherOptions("Other Options");

namespace pretty {
cl::list<std::string> InputFilenames(cl::Positional,
                                     cl::desc("<input PDB files>"),
                                     cl::OneOrMore, cl::sub(PrettySubcommand));

cl::opt<bool> Compilands("compilands", cl::desc("Display compilands"),
                         cl::cat(TypeCategory), cl::sub(PrettySubcommand));
cl::opt<bool> Symbols("module-syms",
                      cl::desc("Display symbols for each compiland"),
                      cl::cat(TypeCategory), cl::sub(PrettySubcommand));
cl::opt<bool> Globals("globals", cl::desc("Dump global symbols"),
                      cl::cat(TypeCategory), cl::sub(PrettySubcommand));
cl::opt<bool> Externals("externals", cl::desc("Dump external symbols"),
                        cl::cat(TypeCategory), cl::sub(PrettySubcommand));
cl::list<SymLevel> SymTypes(
    "sym-types", cl::desc("Type of symbols to dump (default all)"),
    cl::cat(TypeCategory), cl::sub(PrettySubcommand), cl::ZeroOrMore,
    cl::values(
        clEnumValN(SymLevel::Thunks, "thunks", "Display thunk symbols"),
        clEnumValN(SymLevel::Data, "data", "Display data symbols"),
        clEnumValN(SymLevel::Functions, "funcs", "Display function symbols"),
        clEnumValN(SymLevel::All, "all", "Display all symbols (default)")));

cl::opt<bool>
    Types("types",
          cl::desc("Display all types (implies -classes, -enums, -typedefs)"),
          cl::cat(TypeCategory), cl::sub(PrettySubcommand));
cl::opt<bool> Classes("classes", cl::desc("Display class types"),
                      cl::cat(TypeCategory), cl::sub(PrettySubcommand));
cl::opt<bool> Enums("enums", cl::desc("Display enum types"),
                    cl::cat(TypeCategory), cl::sub(PrettySubcommand));
cl::opt<bool> Typedefs("typedefs", cl::desc("Display typedef types"),
                       cl::cat(TypeCategory), cl::sub(PrettySubcommand));
cl::opt<SymbolSortMode> SymbolOrder(
    "symbol-order", cl::desc("symbol sort order"),
    cl::init(SymbolSortMode::None),
    cl::values(clEnumValN(SymbolSortMode::None, "none",
                          "Undefined / no particular sort order"),
               clEnumValN(SymbolSortMode::Name, "name", "Sort symbols by name"),
               clEnumValN(SymbolSortMode::Size, "size",
                          "Sort symbols by size")),
    cl::cat(TypeCategory), cl::sub(PrettySubcommand));

cl::opt<ClassSortMode> ClassOrder(
    "class-order", cl::desc("Class sort order"), cl::init(ClassSortMode::None),
    cl::values(
        clEnumValN(ClassSortMode::None, "none",
                   "Undefined / no particular sort order"),
        clEnumValN(ClassSortMode::Name, "name", "Sort classes by name"),
        clEnumValN(ClassSortMode::Size, "size", "Sort classes by size"),
        clEnumValN(ClassSortMode::Padding, "padding",
                   "Sort classes by amount of padding"),
        clEnumValN(ClassSortMode::PaddingPct, "padding-pct",
                   "Sort classes by percentage of space consumed by padding"),
        clEnumValN(ClassSortMode::PaddingImmediate, "padding-imm",
                   "Sort classes by amount of immediate padding"),
        clEnumValN(ClassSortMode::PaddingPctImmediate, "padding-pct-imm",
                   "Sort classes by percentage of space consumed by immediate "
                   "padding")),
    cl::cat(TypeCategory), cl::sub(PrettySubcommand));

cl::opt<ClassDefinitionFormat> ClassFormat(
    "class-definitions", cl::desc("Class definition format"),
    cl::init(ClassDefinitionFormat::All),
    cl::values(
        clEnumValN(ClassDefinitionFormat::All, "all",
                   "Display all class members including data, constants, "
                   "typedefs, functions, etc"),
        clEnumValN(ClassDefinitionFormat::Layout, "layout",
                   "Only display members that contribute to class size."),
        clEnumValN(ClassDefinitionFormat::None, "none",
                   "Don't display class definitions")),
    cl::cat(TypeCategory), cl::sub(PrettySubcommand));
cl::opt<uint32_t> ClassRecursionDepth(
    "class-recurse-depth", cl::desc("Class recursion depth (0=no limit)"),
    cl::init(0), cl::cat(TypeCategory), cl::sub(PrettySubcommand));

cl::opt<bool> Lines("lines", cl::desc("Line tables"), cl::cat(TypeCategory),
                    cl::sub(PrettySubcommand));
cl::opt<bool>
    All("all", cl::desc("Implies all other options in 'Symbol Types' category"),
        cl::cat(TypeCategory), cl::sub(PrettySubcommand));

cl::opt<uint64_t> LoadAddress(
    "load-address",
    cl::desc("Assume the module is loaded at the specified address"),
    cl::cat(OtherOptions), cl::sub(PrettySubcommand));
cl::opt<bool> Native("native", cl::desc("Use native PDB reader instead of DIA"),
                     cl::cat(OtherOptions), cl::sub(PrettySubcommand));
cl::opt<cl::boolOrDefault>
    ColorOutput("color-output",
                cl::desc("Override use of color (default = isatty)"),
                cl::cat(OtherOptions), cl::sub(PrettySubcommand));
cl::list<std::string> ExcludeTypes(
    "exclude-types", cl::desc("Exclude types by regular expression"),
    cl::ZeroOrMore, cl::cat(FilterCategory), cl::sub(PrettySubcommand));
cl::list<std::string> ExcludeSymbols(
    "exclude-symbols", cl::desc("Exclude symbols by regular expression"),
    cl::ZeroOrMore, cl::cat(FilterCategory), cl::sub(PrettySubcommand));
cl::list<std::string> ExcludeCompilands(
    "exclude-compilands", cl::desc("Exclude compilands by regular expression"),
    cl::ZeroOrMore, cl::cat(FilterCategory), cl::sub(PrettySubcommand));

cl::list<std::string> IncludeTypes(
    "include-types",
    cl::desc("Include only types which match a regular expression"),
    cl::ZeroOrMore, cl::cat(FilterCategory), cl::sub(PrettySubcommand));
cl::list<std::string> IncludeSymbols(
    "include-symbols",
    cl::desc("Include only symbols which match a regular expression"),
    cl::ZeroOrMore, cl::cat(FilterCategory), cl::sub(PrettySubcommand));
cl::list<std::string> IncludeCompilands(
    "include-compilands",
    cl::desc("Include only compilands those which match a regular expression"),
    cl::ZeroOrMore, cl::cat(FilterCategory), cl::sub(PrettySubcommand));
cl::opt<uint32_t> SizeThreshold(
    "min-type-size", cl::desc("Displays only those types which are greater "
                              "than or equal to the specified size."),
    cl::init(0), cl::cat(FilterCategory), cl::sub(PrettySubcommand));
cl::opt<uint32_t> PaddingThreshold(
    "min-class-padding", cl::desc("Displays only those classes which have at "
                                  "least the specified amount of padding."),
    cl::init(0), cl::cat(FilterCategory), cl::sub(PrettySubcommand));
cl::opt<uint32_t> ImmediatePaddingThreshold(
    "min-class-padding-imm",
    cl::desc("Displays only those classes which have at least the specified "
             "amount of immediate padding, ignoring padding internal to bases "
             "and aggregates."),
    cl::init(0), cl::cat(FilterCategory), cl::sub(PrettySubcommand));

cl::opt<bool> ExcludeCompilerGenerated(
    "no-compiler-generated",
    cl::desc("Don't show compiler generated types and symbols"),
    cl::cat(FilterCategory), cl::sub(PrettySubcommand));
cl::opt<bool>
    ExcludeSystemLibraries("no-system-libs",
                           cl::desc("Don't show symbols from system libraries"),
                           cl::cat(FilterCategory), cl::sub(PrettySubcommand));

cl::opt<bool> NoEnumDefs("no-enum-definitions",
                         cl::desc("Don't display full enum definitions"),
                         cl::cat(FilterCategory), cl::sub(PrettySubcommand));
}

namespace diff {
cl::opt<bool> Pedantic("pedantic",
                       cl::desc("Finds all differences (even structural ones "
                                "that produce otherwise identical PDBs)"),
                       cl::sub(DiffSubcommand));

cl::list<std::string> InputFilenames(cl::Positional,
                                     cl::desc("<first> <second>"),
                                     cl::OneOrMore, cl::sub(DiffSubcommand));
}

namespace raw {

cl::OptionCategory MsfOptions("MSF Container Options");
cl::OptionCategory TypeOptions("Type Record Options");
cl::OptionCategory FileOptions("Module & File Options");
cl::OptionCategory SymbolOptions("Symbol Options");
cl::OptionCategory MiscOptions("Miscellaneous Options");

// MSF OPTIONS
cl::opt<bool> DumpHeaders("headers", cl::desc("dump PDB headers"),
                          cl::cat(MsfOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpStreamBlocks("stream-blocks",
                               cl::desc("dump PDB stream blocks"),
                               cl::cat(MsfOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpStreamSummary("stream-summary",
                                cl::desc("dump summary of the PDB streams"),
                                cl::cat(MsfOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpPageStats(
    "page-stats",
    cl::desc("dump allocation stats of the pages in the MSF file"),
    cl::cat(MsfOptions), cl::sub(RawSubcommand));
cl::opt<std::string>
    DumpBlockRangeOpt("block-data", cl::value_desc("start[-end]"),
                      cl::desc("Dump binary data from specified range."),
                      cl::cat(MsfOptions), cl::sub(RawSubcommand));
llvm::Optional<BlockRange> DumpBlockRange;

cl::list<std::string>
    DumpStreamData("stream-data", cl::CommaSeparated, cl::ZeroOrMore,
                   cl::desc("Dump binary data from specified streams.  Format "
                            "is SN[:Start][@Size]"),
                   cl::cat(MsfOptions), cl::sub(RawSubcommand));

// TYPE OPTIONS
cl::opt<bool>
    CompactRecords("compact-records",
                   cl::desc("Dump type and symbol records with less detail"),
                   cl::cat(TypeOptions), cl::sub(RawSubcommand));

cl::opt<bool>
    DumpTpiRecords("tpi-records",
                   cl::desc("dump CodeView type records from TPI stream"),
                   cl::cat(TypeOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpTpiRecordBytes(
    "tpi-record-bytes",
    cl::desc("dump CodeView type record raw bytes from TPI stream"),
    cl::cat(TypeOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpTpiHash("tpi-hash", cl::desc("dump CodeView TPI hash stream"),
                          cl::cat(TypeOptions), cl::sub(RawSubcommand));
cl::opt<bool>
    DumpIpiRecords("ipi-records",
                   cl::desc("dump CodeView type records from IPI stream"),
                   cl::cat(TypeOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpIpiRecordBytes(
    "ipi-record-bytes",
    cl::desc("dump CodeView type record raw bytes from IPI stream"),
    cl::cat(TypeOptions), cl::sub(RawSubcommand));

// MODULE & FILE OPTIONS
cl::opt<bool> DumpModules("modules", cl::desc("dump compiland information"),
                          cl::cat(FileOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpModuleFiles("module-files", cl::desc("dump file information"),
                              cl::cat(FileOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpLineInfo("line-info",
                           cl::desc("dump file and line information"),
                           cl::cat(FileOptions), cl::sub(RawSubcommand));

// SYMBOL OPTIONS
cl::opt<bool> DumpGlobals("globals", cl::desc("dump globals stream data"),
                          cl::cat(SymbolOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpModuleSyms("module-syms", cl::desc("dump module symbols"),
                             cl::cat(SymbolOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpPublics("publics", cl::desc("dump Publics stream data"),
                          cl::cat(SymbolOptions), cl::sub(RawSubcommand));
cl::opt<bool>
    DumpSymRecordBytes("sym-record-bytes",
                       cl::desc("dump CodeView symbol record raw bytes"),
                       cl::cat(SymbolOptions), cl::sub(RawSubcommand));

// MISCELLANEOUS OPTIONS
cl::opt<bool> DumpStringTable("string-table", cl::desc("dump PDB String Table"),
                              cl::cat(MiscOptions), cl::sub(RawSubcommand));

cl::opt<bool> DumpSectionContribs("section-contribs",
                                  cl::desc("dump section contributions"),
                                  cl::cat(MiscOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpSectionMap("section-map", cl::desc("dump section map"),
                             cl::cat(MiscOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpSectionHeaders("section-headers",
                                 cl::desc("dump section headers"),
                                 cl::cat(MiscOptions), cl::sub(RawSubcommand));
cl::opt<bool> DumpFpo("fpo", cl::desc("dump FPO records"), cl::cat(MiscOptions),
                      cl::sub(RawSubcommand));

cl::opt<bool> RawAll("all", cl::desc("Implies most other options."),
                     cl::cat(MiscOptions), cl::sub(RawSubcommand));

cl::list<std::string> InputFilenames(cl::Positional,
                                     cl::desc("<input PDB files>"),
                                     cl::OneOrMore, cl::sub(RawSubcommand));
}

namespace yaml2pdb {
cl::opt<std::string>
    YamlPdbOutputFile("pdb", cl::desc("the name of the PDB file to write"),
                      cl::sub(YamlToPdbSubcommand));

cl::list<std::string> InputFilename(cl::Positional,
                                    cl::desc("<input YAML file>"), cl::Required,
                                    cl::sub(YamlToPdbSubcommand));
}

namespace pdb2yaml {
cl::opt<bool>
    NoFileHeaders("no-file-headers",
                  cl::desc("Do not dump MSF file headers (you will not be able "
                           "to generate a fresh PDB from the resulting YAML)"),
                  cl::sub(PdbToYamlSubcommand), cl::init(false));
cl::opt<bool> Minimal("minimal",
                      cl::desc("Don't write fields with default values"),
                      cl::sub(PdbToYamlSubcommand), cl::init(false));

cl::opt<bool> StreamMetadata(
    "stream-metadata",
    cl::desc("Dump the number of streams and each stream's size"),
    cl::sub(PdbToYamlSubcommand), cl::init(false));
cl::opt<bool> StreamDirectory(
    "stream-directory",
    cl::desc("Dump each stream's block map (implies -stream-metadata)"),
    cl::sub(PdbToYamlSubcommand), cl::init(false));
cl::opt<bool> PdbStream("pdb-stream",
                        cl::desc("Dump the PDB Stream (Stream 1)"),
                        cl::sub(PdbToYamlSubcommand), cl::init(false));

cl::opt<bool> StringTable("string-table", cl::desc("Dump the PDB String Table"),
                          cl::sub(PdbToYamlSubcommand), cl::init(false));

cl::opt<bool> DbiStream("dbi-stream",
                        cl::desc("Dump the DBI Stream (Stream 2)"),
                        cl::sub(PdbToYamlSubcommand), cl::init(false));
cl::opt<bool>
    DbiModuleInfo("dbi-module-info",
                  cl::desc("Dump DBI Module Information (implies -dbi-stream)"),
                  cl::sub(PdbToYamlSubcommand), cl::init(false));

cl::opt<bool> DbiModuleSyms(
    "dbi-module-syms",
    cl::desc("Dump DBI Module Information (implies -dbi-module-info)"),
    cl::sub(PdbToYamlSubcommand), cl::init(false));

cl::opt<bool> DbiModuleSourceFileInfo(
    "dbi-module-source-info",
    cl::desc(
        "Dump DBI Module Source File Information (implies -dbi-module-info)"),
    cl::sub(PdbToYamlSubcommand), cl::init(false));

cl::opt<bool>
    DbiModuleSourceLineInfo("dbi-module-lines",
                            cl::desc("Dump DBI Module Source Line Information "
                                     "(implies -dbi-module-source-info)"),
                            cl::sub(PdbToYamlSubcommand), cl::init(false));

cl::opt<bool> TpiStream("tpi-stream",
                        cl::desc("Dump the TPI Stream (Stream 3)"),
                        cl::sub(PdbToYamlSubcommand), cl::init(false));

cl::opt<bool> IpiStream("ipi-stream",
                        cl::desc("Dump the IPI Stream (Stream 5)"),
                        cl::sub(PdbToYamlSubcommand), cl::init(false));

cl::list<std::string> InputFilename(cl::Positional,
                                    cl::desc("<input PDB file>"), cl::Required,
                                    cl::sub(PdbToYamlSubcommand));
}

namespace analyze {
cl::opt<bool> StringTable("hash-collisions", cl::desc("Find hash collisions"),
                          cl::sub(AnalyzeSubcommand), cl::init(false));
cl::list<std::string> InputFilename(cl::Positional,
                                    cl::desc("<input PDB file>"), cl::Required,
                                    cl::sub(AnalyzeSubcommand));
}
}

static ExitOnError ExitOnErr;

static void yamlToPdb(StringRef Path) {
  BumpPtrAllocator Allocator;
  ErrorOr<std::unique_ptr<MemoryBuffer>> ErrorOrBuffer =
      MemoryBuffer::getFileOrSTDIN(Path, /*FileSize=*/-1,
                                   /*RequiresNullTerminator=*/false);

  if (ErrorOrBuffer.getError()) {
    ExitOnErr(make_error<GenericError>(generic_error_code::invalid_path, Path));
  }

  std::unique_ptr<MemoryBuffer> &Buffer = ErrorOrBuffer.get();

  llvm::yaml::Input In(Buffer->getBuffer());
  pdb::yaml::PdbObject YamlObj(Allocator);
  In >> YamlObj;

  PDBFileBuilder Builder(Allocator);

  uint32_t BlockSize = 4096;
  if (YamlObj.Headers.hasValue())
    BlockSize = YamlObj.Headers->SuperBlock.BlockSize;
  ExitOnErr(Builder.initialize(BlockSize));
  // Add each of the reserved streams.  We ignore stream metadata in the
  // yaml, because we will reconstruct our own view of the streams.  For
  // example, the YAML may say that there were 20 streams in the original
  // PDB, but maybe we only dump a subset of those 20 streams, so we will
  // have fewer, and the ones we do have may end up with different indices
  // than the ones in the original PDB.  So we just start with a clean slate.
  for (uint32_t I = 0; I < kSpecialStreamCount; ++I)
    ExitOnErr(Builder.getMsfBuilder().addStream(0));

  if (YamlObj.StringTable.hasValue()) {
    auto &Strings = Builder.getStringTableBuilder();
    for (auto S : *YamlObj.StringTable)
      Strings.insert(S);
  }

  pdb::yaml::PdbInfoStream DefaultInfoStream;
  pdb::yaml::PdbDbiStream DefaultDbiStream;
  pdb::yaml::PdbTpiStream DefaultTpiStream;

  const auto &Info = YamlObj.PdbStream.getValueOr(DefaultInfoStream);

  auto &InfoBuilder = Builder.getInfoBuilder();
  InfoBuilder.setAge(Info.Age);
  InfoBuilder.setGuid(Info.Guid);
  InfoBuilder.setSignature(Info.Signature);
  InfoBuilder.setVersion(Info.Version);
  for (auto F : Info.Features)
    InfoBuilder.addFeature(F);

  auto &Strings = Builder.getStringTableBuilder().getStrings();

  const auto &Dbi = YamlObj.DbiStream.getValueOr(DefaultDbiStream);
  auto &DbiBuilder = Builder.getDbiBuilder();
  DbiBuilder.setAge(Dbi.Age);
  DbiBuilder.setBuildNumber(Dbi.BuildNumber);
  DbiBuilder.setFlags(Dbi.Flags);
  DbiBuilder.setMachineType(Dbi.MachineType);
  DbiBuilder.setPdbDllRbld(Dbi.PdbDllRbld);
  DbiBuilder.setPdbDllVersion(Dbi.PdbDllVersion);
  DbiBuilder.setVersionHeader(Dbi.VerHeader);
  for (const auto &MI : Dbi.ModInfos) {
    auto &ModiBuilder = ExitOnErr(DbiBuilder.addModuleInfo(MI.Mod));

    for (auto S : MI.SourceFiles)
      ExitOnErr(DbiBuilder.addModuleSourceFile(MI.Mod, S));
    if (MI.Modi.hasValue()) {
      const auto &ModiStream = *MI.Modi;
      ModiBuilder.setObjFileName(MI.Obj);
      for (auto Symbol : ModiStream.Symbols)
        ModiBuilder.addSymbol(Symbol.Record);
    }
    if (MI.FileLineInfo.hasValue()) {
      const auto &FLI = *MI.FileLineInfo;

      // File Checksums must be emitted before line information, because line
      // info records use offsets into the checksum buffer to reference a file's
      // source file name.
      auto Checksums =
          llvm::make_unique<ModuleDebugFileChecksumFragment>(Strings);
      auto &ChecksumRef = *Checksums;
      if (!FLI.FileChecksums.empty()) {
        for (auto &FC : FLI.FileChecksums)
          Checksums->addChecksum(FC.FileName, FC.Kind, FC.ChecksumBytes.Bytes);
      }
      ModiBuilder.setC13FileChecksums(std::move(Checksums));

      for (const auto &Fragment : FLI.LineFragments) {
        auto Lines =
            llvm::make_unique<ModuleDebugLineFragment>(ChecksumRef, Strings);
        Lines->setCodeSize(Fragment.CodeSize);
        Lines->setRelocationAddress(Fragment.RelocSegment,
                                    Fragment.RelocOffset);
        Lines->setFlags(Fragment.Flags);
        for (const auto &LC : Fragment.Blocks) {
          Lines->createBlock(LC.FileName);
          if (Lines->hasColumnInfo()) {
            for (const auto &Item : zip(LC.Lines, LC.Columns)) {
              auto &L = std::get<0>(Item);
              auto &C = std::get<1>(Item);
              uint32_t LE = L.LineStart + L.EndDelta;
              Lines->addLineAndColumnInfo(
                  L.Offset, LineInfo(L.LineStart, LE, L.IsStatement),
                  C.StartColumn, C.EndColumn);
            }
          } else {
            for (const auto &L : LC.Lines) {
              uint32_t LE = L.LineStart + L.EndDelta;
              Lines->addLineInfo(L.Offset,
                                 LineInfo(L.LineStart, LE, L.IsStatement));
            }
          }
        }
        ModiBuilder.addC13Fragment(std::move(Lines));
      }

      for (const auto &Inlinee : FLI.Inlinees) {
        auto Inlinees = llvm::make_unique<ModuleDebugInlineeLineFragment>(
            ChecksumRef, Inlinee.HasExtraFiles);
        for (const auto &Site : Inlinee.Sites) {
          Inlinees->addInlineSite(Site.Inlinee, Site.FileName,
                                  Site.SourceLineNum);
          if (!Inlinee.HasExtraFiles)
            continue;

          for (auto EF : Site.ExtraFiles) {
            Inlinees->addExtraFile(EF);
          }
        }
        ModiBuilder.addC13Fragment(std::move(Inlinees));
      }
    }
  }

  auto &TpiBuilder = Builder.getTpiBuilder();
  const auto &Tpi = YamlObj.TpiStream.getValueOr(DefaultTpiStream);
  TpiBuilder.setVersionHeader(Tpi.Version);
  for (const auto &R : Tpi.Records)
    TpiBuilder.addTypeRecord(R.Record.data(), R.Record.Hash);

  const auto &Ipi = YamlObj.IpiStream.getValueOr(DefaultTpiStream);
  auto &IpiBuilder = Builder.getIpiBuilder();
  IpiBuilder.setVersionHeader(Ipi.Version);
  for (const auto &R : Ipi.Records)
    TpiBuilder.addTypeRecord(R.Record.data(), R.Record.Hash);

  ExitOnErr(Builder.commit(opts::yaml2pdb::YamlPdbOutputFile));
}

static PDBFile &loadPDB(StringRef Path, std::unique_ptr<IPDBSession> &Session) {
  ExitOnErr(loadDataForPDB(PDB_ReaderType::Native, Path, Session));

  NativeSession *NS = static_cast<NativeSession *>(Session.get());
  return NS->getPDBFile();
}

static void pdb2Yaml(StringRef Path) {
  std::unique_ptr<IPDBSession> Session;
  auto &File = loadPDB(Path, Session);

  auto O = llvm::make_unique<YAMLOutputStyle>(File);
  O = llvm::make_unique<YAMLOutputStyle>(File);

  ExitOnErr(O->dump());
}

static void dumpRaw(StringRef Path) {
  std::unique_ptr<IPDBSession> Session;
  auto &File = loadPDB(Path, Session);

  auto O = llvm::make_unique<LLVMOutputStyle>(File);

  ExitOnErr(O->dump());
}

static void dumpAnalysis(StringRef Path) {
  std::unique_ptr<IPDBSession> Session;
  auto &File = loadPDB(Path, Session);
  auto O = llvm::make_unique<AnalysisStyle>(File);

  ExitOnErr(O->dump());
}

static void diff(StringRef Path1, StringRef Path2) {
  std::unique_ptr<IPDBSession> Session1;
  std::unique_ptr<IPDBSession> Session2;

  auto &File1 = loadPDB(Path1, Session1);
  auto &File2 = loadPDB(Path2, Session2);

  auto O = llvm::make_unique<DiffStyle>(File1, File2);

  ExitOnErr(O->dump());
}

bool opts::pretty::shouldDumpSymLevel(SymLevel Search) {
  if (SymTypes.empty())
    return true;
  if (llvm::find(SymTypes, Search) != SymTypes.end())
    return true;
  if (llvm::find(SymTypes, SymLevel::All) != SymTypes.end())
    return true;
  return false;
}

uint32_t llvm::pdb::getTypeLength(const PDBSymbolData &Symbol) {
  auto SymbolType = Symbol.getType();
  const IPDBRawSymbol &RawType = SymbolType->getRawSymbol();

  return RawType.getLength();
}

bool opts::pretty::compareFunctionSymbols(
    const std::unique_ptr<PDBSymbolFunc> &F1,
    const std::unique_ptr<PDBSymbolFunc> &F2) {
  assert(opts::pretty::SymbolOrder != opts::pretty::SymbolSortMode::None);

  if (opts::pretty::SymbolOrder == opts::pretty::SymbolSortMode::Name)
    return F1->getName() < F2->getName();

  // Note that we intentionally sort in descending order on length, since
  // long functions are more interesting than short functions.
  return F1->getLength() > F2->getLength();
}

bool opts::pretty::compareDataSymbols(
    const std::unique_ptr<PDBSymbolData> &F1,
    const std::unique_ptr<PDBSymbolData> &F2) {
  assert(opts::pretty::SymbolOrder != opts::pretty::SymbolSortMode::None);

  if (opts::pretty::SymbolOrder == opts::pretty::SymbolSortMode::Name)
    return F1->getName() < F2->getName();

  // Note that we intentionally sort in descending order on length, since
  // large types are more interesting than short ones.
  return getTypeLength(*F1) > getTypeLength(*F2);
}

static void dumpPretty(StringRef Path) {
  std::unique_ptr<IPDBSession> Session;

  const auto ReaderType =
      opts::pretty::Native ? PDB_ReaderType::Native : PDB_ReaderType::DIA;
  ExitOnErr(loadDataForPDB(ReaderType, Path, Session));

  if (opts::pretty::LoadAddress)
    Session->setLoadAddress(opts::pretty::LoadAddress);

  auto &Stream = outs();
  const bool UseColor = opts::pretty::ColorOutput == cl::BOU_UNSET
                            ? Stream.has_colors()
                            : opts::pretty::ColorOutput == cl::BOU_TRUE;
  LinePrinter Printer(2, UseColor, Stream);

  auto GlobalScope(Session->getGlobalScope());
  std::string FileName(GlobalScope->getSymbolsFileName());

  WithColor(Printer, PDB_ColorItem::None).get() << "Summary for ";
  WithColor(Printer, PDB_ColorItem::Path).get() << FileName;
  Printer.Indent();
  uint64_t FileSize = 0;

  Printer.NewLine();
  WithColor(Printer, PDB_ColorItem::Identifier).get() << "Size";
  if (!sys::fs::file_size(FileName, FileSize)) {
    Printer << ": " << FileSize << " bytes";
  } else {
    Printer << ": (Unable to obtain file size)";
  }

  Printer.NewLine();
  WithColor(Printer, PDB_ColorItem::Identifier).get() << "Guid";
  Printer << ": " << GlobalScope->getGuid();

  Printer.NewLine();
  WithColor(Printer, PDB_ColorItem::Identifier).get() << "Age";
  Printer << ": " << GlobalScope->getAge();

  Printer.NewLine();
  WithColor(Printer, PDB_ColorItem::Identifier).get() << "Attributes";
  Printer << ": ";
  if (GlobalScope->hasCTypes())
    outs() << "HasCTypes ";
  if (GlobalScope->hasPrivateSymbols())
    outs() << "HasPrivateSymbols ";
  Printer.Unindent();

  if (opts::pretty::Compilands) {
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::SectionHeader).get()
        << "---COMPILANDS---";
    Printer.Indent();
    auto Compilands = GlobalScope->findAllChildren<PDBSymbolCompiland>();
    CompilandDumper Dumper(Printer);
    CompilandDumpFlags options = CompilandDumper::Flags::None;
    if (opts::pretty::Lines)
      options = options | CompilandDumper::Flags::Lines;
    while (auto Compiland = Compilands->getNext())
      Dumper.start(*Compiland, options);
    Printer.Unindent();
  }

  if (opts::pretty::Classes || opts::pretty::Enums || opts::pretty::Typedefs) {
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::SectionHeader).get() << "---TYPES---";
    Printer.Indent();
    TypeDumper Dumper(Printer);
    Dumper.start(*GlobalScope);
    Printer.Unindent();
  }

  if (opts::pretty::Symbols) {
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::SectionHeader).get() << "---SYMBOLS---";
    Printer.Indent();
    auto Compilands = GlobalScope->findAllChildren<PDBSymbolCompiland>();
    CompilandDumper Dumper(Printer);
    while (auto Compiland = Compilands->getNext())
      Dumper.start(*Compiland, true);
    Printer.Unindent();
  }

  if (opts::pretty::Globals) {
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::SectionHeader).get() << "---GLOBALS---";
    Printer.Indent();
    if (shouldDumpSymLevel(opts::pretty::SymLevel::Functions)) {
      FunctionDumper Dumper(Printer);
      auto Functions = GlobalScope->findAllChildren<PDBSymbolFunc>();
      if (opts::pretty::SymbolOrder == opts::pretty::SymbolSortMode::None) {
        while (auto Function = Functions->getNext()) {
          Printer.NewLine();
          Dumper.start(*Function, FunctionDumper::PointerType::None);
        }
      } else {
        std::vector<std::unique_ptr<PDBSymbolFunc>> Funcs;
        while (auto Func = Functions->getNext())
          Funcs.push_back(std::move(Func));
        std::sort(Funcs.begin(), Funcs.end(),
                  opts::pretty::compareFunctionSymbols);
        for (const auto &Func : Funcs) {
          Printer.NewLine();
          Dumper.start(*Func, FunctionDumper::PointerType::None);
        }
      }
    }
    if (shouldDumpSymLevel(opts::pretty::SymLevel::Data)) {
      auto Vars = GlobalScope->findAllChildren<PDBSymbolData>();
      VariableDumper Dumper(Printer);
      if (opts::pretty::SymbolOrder == opts::pretty::SymbolSortMode::None) {
        while (auto Var = Vars->getNext())
          Dumper.start(*Var);
      } else {
        std::vector<std::unique_ptr<PDBSymbolData>> Datas;
        while (auto Var = Vars->getNext())
          Datas.push_back(std::move(Var));
        std::sort(Datas.begin(), Datas.end(), opts::pretty::compareDataSymbols);
        for (const auto &Var : Datas)
          Dumper.start(*Var);
      }
    }
    if (shouldDumpSymLevel(opts::pretty::SymLevel::Thunks)) {
      auto Thunks = GlobalScope->findAllChildren<PDBSymbolThunk>();
      CompilandDumper Dumper(Printer);
      while (auto Thunk = Thunks->getNext())
        Dumper.dump(*Thunk);
    }
    Printer.Unindent();
  }
  if (opts::pretty::Externals) {
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::SectionHeader).get() << "---EXTERNALS---";
    Printer.Indent();
    ExternalSymbolDumper Dumper(Printer);
    Dumper.start(*GlobalScope);
  }
  if (opts::pretty::Lines) {
    Printer.NewLine();
  }
  outs().flush();
}

int main(int argc_, const char *argv_[]) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv_[0]);
  PrettyStackTraceProgram X(argc_, argv_);

  ExitOnErr.setBanner("llvm-pdbdump: ");

  SmallVector<const char *, 256> argv;
  SpecificBumpPtrAllocator<char> ArgAllocator;
  ExitOnErr(errorCodeToError(sys::Process::GetArgumentVector(
      argv, makeArrayRef(argv_, argc_), ArgAllocator)));

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argv.size(), argv.data(), "LLVM PDB Dumper\n");
  if (!opts::raw::DumpBlockRangeOpt.empty()) {
    llvm::Regex R("^([0-9]+)(-([0-9]+))?$");
    llvm::SmallVector<llvm::StringRef, 2> Matches;
    if (!R.match(opts::raw::DumpBlockRangeOpt, &Matches)) {
      errs() << "Argument '" << opts::raw::DumpBlockRangeOpt
             << "' invalid format.\n";
      errs().flush();
      exit(1);
    }
    opts::raw::DumpBlockRange.emplace();
    Matches[1].getAsInteger(10, opts::raw::DumpBlockRange->Min);
    if (!Matches[3].empty()) {
      opts::raw::DumpBlockRange->Max.emplace();
      Matches[3].getAsInteger(10, *opts::raw::DumpBlockRange->Max);
    }
  }

  if (opts::RawSubcommand) {
    if (opts::raw::RawAll) {
      opts::raw::DumpHeaders = true;
      opts::raw::DumpModules = true;
      opts::raw::DumpModuleFiles = true;
      opts::raw::DumpModuleSyms = true;
      opts::raw::DumpGlobals = true;
      opts::raw::DumpPublics = true;
      opts::raw::DumpSectionHeaders = true;
      opts::raw::DumpStreamSummary = true;
      opts::raw::DumpPageStats = true;
      opts::raw::DumpStreamBlocks = true;
      opts::raw::DumpTpiRecords = true;
      opts::raw::DumpTpiHash = true;
      opts::raw::DumpIpiRecords = true;
      opts::raw::DumpSectionMap = true;
      opts::raw::DumpSectionContribs = true;
      opts::raw::DumpLineInfo = true;
      opts::raw::DumpFpo = true;
      opts::raw::DumpStringTable = true;
    }

    if (opts::raw::CompactRecords &&
        (opts::raw::DumpTpiRecordBytes || opts::raw::DumpIpiRecordBytes)) {
      errs() << "-compact-records is incompatible with -tpi-record-bytes and "
                "-ipi-record-bytes.\n";
      exit(1);
    }
  }

  llvm::sys::InitializeCOMRAII COM(llvm::sys::COMThreadingMode::MultiThreaded);

  if (opts::PdbToYamlSubcommand) {
    pdb2Yaml(opts::pdb2yaml::InputFilename.front());
  } else if (opts::YamlToPdbSubcommand) {
    yamlToPdb(opts::yaml2pdb::InputFilename.front());
  } else if (opts::AnalyzeSubcommand) {
    dumpAnalysis(opts::analyze::InputFilename.front());
  } else if (opts::PrettySubcommand) {
    if (opts::pretty::Lines)
      opts::pretty::Compilands = true;

    if (opts::pretty::All) {
      opts::pretty::Compilands = true;
      opts::pretty::Symbols = true;
      opts::pretty::Globals = true;
      opts::pretty::Types = true;
      opts::pretty::Externals = true;
      opts::pretty::Lines = true;
    }

    if (opts::pretty::Types) {
      opts::pretty::Classes = true;
      opts::pretty::Typedefs = true;
      opts::pretty::Enums = true;
    }

    // When adding filters for excluded compilands and types, we need to
    // remember that these are regexes.  So special characters such as * and \
    // need to be escaped in the regex.  In the case of a literal \, this means
    // it needs to be escaped again in the C++.  So matching a single \ in the
    // input requires 4 \es in the C++.
    if (opts::pretty::ExcludeCompilerGenerated) {
      opts::pretty::ExcludeTypes.push_back("__vc_attributes");
      opts::pretty::ExcludeCompilands.push_back("\\* Linker \\*");
    }
    if (opts::pretty::ExcludeSystemLibraries) {
      opts::pretty::ExcludeCompilands.push_back(
          "f:\\\\binaries\\\\Intermediate\\\\vctools\\\\crt_bld");
      opts::pretty::ExcludeCompilands.push_back("f:\\\\dd\\\\vctools\\\\crt");
      opts::pretty::ExcludeCompilands.push_back(
          "d:\\\\th.obj.x86fre\\\\minkernel");
    }
    std::for_each(opts::pretty::InputFilenames.begin(),
                  opts::pretty::InputFilenames.end(), dumpPretty);
  } else if (opts::RawSubcommand) {
    std::for_each(opts::raw::InputFilenames.begin(),
                  opts::raw::InputFilenames.end(), dumpRaw);
  } else if (opts::DiffSubcommand) {
    if (opts::diff::InputFilenames.size() != 2) {
      errs() << "diff subcommand expects exactly 2 arguments.\n";
      exit(1);
    }
    diff(opts::diff::InputFilenames[0], opts::diff::InputFilenames[1]);
  }

  outs().flush();
  return 0;
}
