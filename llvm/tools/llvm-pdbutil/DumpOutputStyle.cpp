//===- DumpOutputStyle.cpp ------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DumpOutputStyle.h"

#include "FormatUtil.h"
#include "MinimalSymbolDumper.h"
#include "MinimalTypeDumper.h"
#include "StreamUtil.h"
#include "llvm-pdbutil.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/CodeView/CVSymbolVisitor.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/DebugChecksumsSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugCrossExSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugCrossImpSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugFrameDataSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugInlineeLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugStringTableSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionVisitor.h"
#include "llvm/DebugInfo/CodeView/DebugSymbolsSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugUnknownSubsection.h"
#include "llvm/DebugInfo/CodeView/EnumTables.h"
#include "llvm/DebugInfo/CodeView/Formatters.h"
#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolDumper.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbacks.h"
#include "llvm/DebugInfo/CodeView/TypeDumpVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptor.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/EnumTables.h"
#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "llvm/DebugInfo/PDB/Native/ISectionContribVisitor.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/PublicsStream.h"
#include "llvm/DebugInfo/PDB/Native/SymbolStream.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/TpiHashing.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"

#include <cctype>
#include <unordered_map>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

DumpOutputStyle::DumpOutputStyle(PDBFile &File)
    : File(File), P(2, false, outs()) {}

Error DumpOutputStyle::dump() {
  if (opts::dump::DumpSummary) {
    if (auto EC = dumpFileSummary())
      return EC;
    P.NewLine();
  }

  if (opts::dump::DumpStreams) {
    if (auto EC = dumpStreamSummary())
      return EC;
    P.NewLine();
  }

  if (opts::dump::DumpSymbolStats.getNumOccurrences() > 0) {
    if (auto EC = dumpSymbolStats())
      return EC;
    P.NewLine();
  }

  if (opts::dump::DumpUdtStats.getNumOccurrences() > 0) {
    if (auto EC = dumpUdtStats())
      return EC;
    P.NewLine();
  }

  if (opts::dump::DumpStringTable) {
    if (auto EC = dumpStringTable())
      return EC;
    P.NewLine();
  }

  if (opts::dump::DumpModules) {
    if (auto EC = dumpModules())
      return EC;
  }

  if (opts::dump::DumpModuleFiles) {
    if (auto EC = dumpModuleFiles())
      return EC;
  }

  if (opts::dump::DumpLines) {
    if (auto EC = dumpLines())
      return EC;
  }

  if (opts::dump::DumpInlineeLines) {
    if (auto EC = dumpInlineeLines())
      return EC;
  }

  if (opts::dump::DumpXmi) {
    if (auto EC = dumpXmi())
      return EC;
  }

  if (opts::dump::DumpXme) {
    if (auto EC = dumpXme())
      return EC;
  }

  if (opts::dump::DumpTypes || !opts::dump::DumpTypeIndex.empty() ||
      opts::dump::DumpTypeExtras) {
    if (auto EC = dumpTpiStream(StreamTPI))
      return EC;
  }

  if (opts::dump::DumpIds || !opts::dump::DumpIdIndex.empty() ||
      opts::dump::DumpIdExtras) {
    if (auto EC = dumpTpiStream(StreamIPI))
      return EC;
  }

  if (opts::dump::DumpGlobals) {
    if (auto EC = dumpGlobals())
      return EC;
  }

  if (opts::dump::DumpPublics) {
    if (auto EC = dumpPublics())
      return EC;
  }

  if (opts::dump::DumpSymbols) {
    if (auto EC = dumpModuleSyms())
      return EC;
  }

  if (opts::dump::DumpSectionHeaders) {
    if (auto EC = dumpSectionHeaders())
      return EC;
  }

  if (opts::dump::DumpSectionContribs) {
    if (auto EC = dumpSectionContribs())
      return EC;
  }

  if (opts::dump::DumpSectionMap) {
    if (auto EC = dumpSectionMap())
      return EC;
  }

  return Error::success();
}

static void printHeader(LinePrinter &P, const Twine &S) {
  P.NewLine();
  P.formatLine("{0,=60}", S);
  P.formatLine("{0}", fmt_repeat('=', 60));
}

Error DumpOutputStyle::dumpFileSummary() {
  printHeader(P, "Summary");

  ExitOnError Err("Invalid PDB Format: ");

  AutoIndent Indent(P);
  P.formatLine("Block Size: {0}", File.getBlockSize());
  P.formatLine("Number of blocks: {0}", File.getBlockCount());
  P.formatLine("Number of streams: {0}", File.getNumStreams());

  auto &PS = Err(File.getPDBInfoStream());
  P.formatLine("Signature: {0}", PS.getSignature());
  P.formatLine("Age: {0}", PS.getAge());
  P.formatLine("GUID: {0}", fmt_guid(PS.getGuid().Guid));
  P.formatLine("Features: {0:x+}", static_cast<uint32_t>(PS.getFeatures()));
  P.formatLine("Has Debug Info: {0}", File.hasPDBDbiStream());
  P.formatLine("Has Types: {0}", File.hasPDBTpiStream());
  P.formatLine("Has IDs: {0}", File.hasPDBIpiStream());
  P.formatLine("Has Globals: {0}", File.hasPDBGlobalsStream());
  P.formatLine("Has Publics: {0}", File.hasPDBPublicsStream());
  if (File.hasPDBDbiStream()) {
    auto &DBI = Err(File.getPDBDbiStream());
    P.formatLine("Is incrementally linked: {0}", DBI.isIncrementallyLinked());
    P.formatLine("Has conflicting types: {0}", DBI.hasCTypes());
    P.formatLine("Is stripped: {0}", DBI.isStripped());
  }

  return Error::success();
}

static StatCollection getSymbolStats(ModuleDebugStreamRef MDS,
                                     StatCollection &CumulativeStats) {
  StatCollection Stats;
  for (const auto &S : MDS.symbols(nullptr)) {
    Stats.update(S.kind(), S.length());
    CumulativeStats.update(S.kind(), S.length());
  }
  return Stats;
}

static StatCollection getChunkStats(ModuleDebugStreamRef MDS,
                                    StatCollection &CumulativeStats) {
  StatCollection Stats;
  for (const auto &Chunk : MDS.subsections()) {
    Stats.update(uint32_t(Chunk.kind()), Chunk.getRecordLength());
    CumulativeStats.update(uint32_t(Chunk.kind()), Chunk.getRecordLength());
  }
  return Stats;
}

static inline std::string formatModuleDetailKind(DebugSubsectionKind K) {
  return formatChunkKind(K, false);
}

static inline std::string formatModuleDetailKind(SymbolKind K) {
  return formatSymbolKind(K);
}

template <typename Kind>
static void printModuleDetailStats(LinePrinter &P, StringRef Label,
                                   const StatCollection &Stats) {
  P.NewLine();
  P.formatLine("  {0}", Label);
  AutoIndent Indent(P);
  P.formatLine("{0,40}: {1,7} entries ({2,8} bytes)", "Total",
               Stats.Totals.Count, Stats.Totals.Size);
  P.formatLine("{0}", fmt_repeat('-', 74));
  for (const auto &K : Stats.Individual) {
    std::string KindName = formatModuleDetailKind(Kind(K.first));
    P.formatLine("{0,40}: {1,7} entries ({2,8} bytes)", KindName,
                 K.second.Count, K.second.Size);
  }
}

static bool isMyCode(const DbiModuleDescriptor &Desc) {
  StringRef Name = Desc.getModuleName();
  if (Name.startswith("Import:"))
    return false;
  if (Name.endswith_lower(".dll"))
    return false;
  if (Name.equals_lower("* linker *"))
    return false;
  if (Name.startswith_lower("f:\\binaries\\Intermediate\\vctools"))
    return false;
  if (Name.startswith_lower("f:\\dd\\vctools\\crt"))
    return false;
  return true;
}

static bool shouldDumpModule(uint32_t Modi, const DbiModuleDescriptor &Desc) {
  if (opts::dump::JustMyCode && !isMyCode(Desc))
    return false;

  // If the arg was not specified on the command line, always dump all modules.
  if (opts::dump::DumpModi.getNumOccurrences() == 0)
    return true;

  // Otherwise, only dump if this is the same module specified.
  return (opts::dump::DumpModi == Modi);
}

Error DumpOutputStyle::dumpStreamSummary() {
  printHeader(P, "Streams");

  if (StreamPurposes.empty())
    discoverStreamPurposes(File, StreamPurposes);

  AutoIndent Indent(P);
  uint32_t StreamCount = File.getNumStreams();
  uint32_t MaxStreamSize = File.getMaxStreamSize();

  for (uint16_t StreamIdx = 0; StreamIdx < StreamCount; ++StreamIdx) {
    P.formatLine(
        "Stream {0} ({1} bytes): [{2}]",
        fmt_align(StreamIdx, AlignStyle::Right, NumDigits(StreamCount)),
        fmt_align(File.getStreamByteSize(StreamIdx), AlignStyle::Right,
                  NumDigits(MaxStreamSize)),
        StreamPurposes[StreamIdx].getLongName());

    if (opts::dump::DumpStreamBlocks) {
      auto Blocks = File.getStreamBlockList(StreamIdx);
      std::vector<uint32_t> BV(Blocks.begin(), Blocks.end());
      P.formatLine("       {0}  Blocks: [{1}]",
                   fmt_repeat(' ', NumDigits(StreamCount)),
                   make_range(BV.begin(), BV.end()));
    }
  }

  return Error::success();
}

static Expected<ModuleDebugStreamRef> getModuleDebugStream(PDBFile &File,
                                                           uint32_t Index) {
  ExitOnError Err("Unexpected error: ");

  auto &Dbi = Err(File.getPDBDbiStream());
  const auto &Modules = Dbi.modules();
  auto Modi = Modules.getModuleDescriptor(Index);

  uint16_t ModiStream = Modi.getModuleStreamIndex();
  if (ModiStream == kInvalidStreamIndex)
    return make_error<RawError>(raw_error_code::no_stream,
                                "Module stream not present");

  auto ModStreamData = MappedBlockStream::createIndexedStream(
      File.getMsfLayout(), File.getMsfBuffer(), ModiStream,
      File.getAllocator());

  ModuleDebugStreamRef ModS(Modi, std::move(ModStreamData));
  if (auto EC = ModS.reload())
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid module stream");

  return std::move(ModS);
}

static std::string formatChecksumKind(FileChecksumKind Kind) {
  switch (Kind) {
    RETURN_CASE(FileChecksumKind, None, "None");
    RETURN_CASE(FileChecksumKind, MD5, "MD5");
    RETURN_CASE(FileChecksumKind, SHA1, "SHA-1");
    RETURN_CASE(FileChecksumKind, SHA256, "SHA-256");
  }
  return formatUnknownEnum(Kind);
}

namespace {
class StringsAndChecksumsPrinter {
  const DebugStringTableSubsectionRef &extractStringTable(PDBFile &File) {
    ExitOnError Err("Unexpected error processing modules: ");
    return Err(File.getStringTable()).getStringTable();
  }

  template <typename... Args>
  void formatInternal(LinePrinter &Printer, bool Append,
                      Args &&... args) const {
    if (Append)
      Printer.format(std::forward<Args>(args)...);
    else
      Printer.formatLine(std::forward<Args>(args)...);
  }

public:
  StringsAndChecksumsPrinter(PDBFile &File, uint32_t Modi)
      : Records(extractStringTable(File)) {
    auto MDS = getModuleDebugStream(File, Modi);
    if (!MDS) {
      consumeError(MDS.takeError());
      return;
    }

    DebugStream = llvm::make_unique<ModuleDebugStreamRef>(std::move(*MDS));
    Records.initialize(MDS->subsections());
    if (Records.hasChecksums()) {
      for (const auto &Entry : Records.checksums()) {
        auto S = Records.strings().getString(Entry.FileNameOffset);
        if (!S)
          continue;
        ChecksumsByFile[*S] = Entry;
      }
    }
  }

  Expected<StringRef> getNameFromStringTable(uint32_t Offset) const {
    return Records.strings().getString(Offset);
  }

  void formatFromFileName(LinePrinter &Printer, StringRef File,
                          bool Append = false) const {
    auto FC = ChecksumsByFile.find(File);
    if (FC == ChecksumsByFile.end()) {
      formatInternal(Printer, Append, "- (no checksum) {0}", File);
      return;
    }

    formatInternal(Printer, Append, "- ({0}: {1}) {2}",
                   formatChecksumKind(FC->getValue().Kind),
                   toHex(FC->getValue().Checksum), File);
  }

  void formatFromChecksumsOffset(LinePrinter &Printer, uint32_t Offset,
                                 bool Append = false) const {
    if (!Records.hasChecksums()) {
      formatInternal(Printer, Append, "(unknown file name offset {0})", Offset);
      return;
    }

    auto Iter = Records.checksums().getArray().at(Offset);
    if (Iter == Records.checksums().getArray().end()) {
      formatInternal(Printer, Append, "(unknown file name offset {0})", Offset);
      return;
    }

    uint32_t FO = Iter->FileNameOffset;
    auto ExpectedFile = getNameFromStringTable(FO);
    if (!ExpectedFile) {
      formatInternal(Printer, Append, "(unknown file name offset {0})", Offset);
      consumeError(ExpectedFile.takeError());
      return;
    }
    if (Iter->Kind == FileChecksumKind::None) {
      formatInternal(Printer, Append, "{0} (no checksum)", *ExpectedFile);
    } else {
      formatInternal(Printer, Append, "{0} ({1}: {2})", *ExpectedFile,
                     formatChecksumKind(Iter->Kind), toHex(Iter->Checksum));
    }
  }

  std::unique_ptr<ModuleDebugStreamRef> DebugStream;
  StringsAndChecksumsRef Records;
  StringMap<FileChecksumEntry> ChecksumsByFile;
};
} // namespace

template <typename CallbackT>
static void iterateOneModule(PDBFile &File, LinePrinter &P,
                             const DbiModuleDescriptor &Descriptor,
                             uint32_t Modi, uint32_t IndentLevel,
                             uint32_t Digits, CallbackT Callback) {
  P.formatLine(
      "Mod {0:4} | `{1}`: ", fmt_align(Modi, AlignStyle::Right, Digits),
      Descriptor.getModuleName());

  StringsAndChecksumsPrinter Strings(File, Modi);
  AutoIndent Indent2(P, IndentLevel);
  Callback(Modi, Strings);
}

template <typename CallbackT>
static void iterateModules(PDBFile &File, LinePrinter &P, uint32_t IndentLevel,
                           CallbackT Callback) {
  AutoIndent Indent(P);
  if (!File.hasPDBDbiStream()) {
    P.formatLine("DBI Stream not present");
    return;
  }

  ExitOnError Err("Unexpected error processing modules: ");

  auto &Stream = Err(File.getPDBDbiStream());

  const DbiModuleList &Modules = Stream.modules();

  if (opts::dump::DumpModi.getNumOccurrences() > 0) {
    assert(opts::dump::DumpModi.getNumOccurrences() == 1);
    uint32_t Modi = opts::dump::DumpModi;
    auto Descriptor = Modules.getModuleDescriptor(Modi);
    iterateOneModule(File, P, Descriptor, Modi, IndentLevel, NumDigits(Modi),
                     Callback);
    return;
  }

  uint32_t Count = Modules.getModuleCount();
  uint32_t Digits = NumDigits(Count);
  for (uint32_t I = 0; I < Count; ++I) {
    auto Desc = Modules.getModuleDescriptor(I);
    if (!shouldDumpModule(I, Desc))
      continue;
    iterateOneModule(File, P, Desc, I, IndentLevel, Digits, Callback);
  }
}

template <typename SubsectionT>
static void iterateModuleSubsections(
    PDBFile &File, LinePrinter &P, uint32_t IndentLevel,
    llvm::function_ref<void(uint32_t, StringsAndChecksumsPrinter &,
                            SubsectionT &)>
        Callback) {

  iterateModules(
      File, P, IndentLevel,
      [&File, &Callback](uint32_t Modi, StringsAndChecksumsPrinter &Strings) {
        auto MDS = getModuleDebugStream(File, Modi);
        if (!MDS) {
          consumeError(MDS.takeError());
          return;
        }

        for (const auto &SS : MDS->subsections()) {
          SubsectionT Subsection;

          if (SS.kind() != Subsection.kind())
            continue;

          BinaryStreamReader Reader(SS.getRecordData());
          if (auto EC = Subsection.initialize(Reader))
            continue;
          Callback(Modi, Strings, Subsection);
        }
      });
}

Error DumpOutputStyle::dumpModules() {
  printHeader(P, "Modules");

  AutoIndent Indent(P);
  if (!File.hasPDBDbiStream()) {
    P.formatLine("DBI Stream not present");
    return Error::success();
  }

  ExitOnError Err("Unexpected error processing modules: ");

  auto &Stream = Err(File.getPDBDbiStream());

  const DbiModuleList &Modules = Stream.modules();
  iterateModules(
      File, P, 11, [&](uint32_t Modi, StringsAndChecksumsPrinter &Strings) {
        auto Desc = Modules.getModuleDescriptor(Modi);
        P.formatLine("Obj: `{0}`: ", Desc.getObjFileName());
        P.formatLine("debug stream: {0}, # files: {1}, has ec info: {2}",
                     Desc.getModuleStreamIndex(), Desc.getNumberOfFiles(),
                     Desc.hasECInfo());
        StringRef PdbFilePath =
            Err(Stream.getECName(Desc.getPdbFilePathNameIndex()));
        StringRef SrcFilePath =
            Err(Stream.getECName(Desc.getSourceFileNameIndex()));
        P.formatLine("pdb file ni: {0} `{1}`, src file ni: {2} `{3}`",
                     Desc.getPdbFilePathNameIndex(), PdbFilePath,
                     Desc.getSourceFileNameIndex(), SrcFilePath);
      });
  return Error::success();
}

Error DumpOutputStyle::dumpModuleFiles() {
  printHeader(P, "Files");

  ExitOnError Err("Unexpected error processing modules: ");

  iterateModules(
      File, P, 11,
      [this, &Err](uint32_t Modi, StringsAndChecksumsPrinter &Strings) {
        auto &Stream = Err(File.getPDBDbiStream());

        const DbiModuleList &Modules = Stream.modules();
        for (const auto &F : Modules.source_files(Modi)) {
          Strings.formatFromFileName(P, F);
        }
      });
  return Error::success();
}

Error DumpOutputStyle::dumpSymbolStats() {
  printHeader(P, "Module Stats");

  ExitOnError Err("Unexpected error processing modules: ");

  StatCollection SymStats;
  StatCollection ChunkStats;
  auto &Stream = Err(File.getPDBDbiStream());

  const DbiModuleList &Modules = Stream.modules();
  uint32_t ModCount = Modules.getModuleCount();

  iterateModules(File, P, 0, [&](uint32_t Modi,
                                 StringsAndChecksumsPrinter &Strings) {
    DbiModuleDescriptor Desc = Modules.getModuleDescriptor(Modi);
    uint32_t StreamIdx = Desc.getModuleStreamIndex();

    if (StreamIdx == kInvalidStreamIndex) {
      P.formatLine("Mod {0} (debug info not present): [{1}]",
                   fmt_align(Modi, AlignStyle::Right, NumDigits(ModCount)),
                   Desc.getModuleName());
      return;
    }

    P.formatLine("Stream {0}, {1} bytes", StreamIdx,
                 File.getStreamByteSize(StreamIdx));

    ModuleDebugStreamRef MDS(Desc, File.createIndexedStream(StreamIdx));
    if (auto EC = MDS.reload()) {
      P.printLine("- Error parsing debug info stream");
      consumeError(std::move(EC));
      return;
    }

    printModuleDetailStats<SymbolKind>(P, "Symbols",
                                       getSymbolStats(MDS, SymStats));
    printModuleDetailStats<DebugSubsectionKind>(P, "Chunks",
                                                getChunkStats(MDS, ChunkStats));
  });

  P.printLine("  Summary |");
  AutoIndent Indent(P, 4);
  if (SymStats.Totals.Count > 0) {
    printModuleDetailStats<SymbolKind>(P, "Symbols", SymStats);
    printModuleDetailStats<DebugSubsectionKind>(P, "Chunks", ChunkStats);
  }

  return Error::success();
}

static bool isValidNamespaceIdentifier(StringRef S) {
  if (S.empty())
    return false;

  if (std::isdigit(S[0]))
    return false;

  return llvm::all_of(S, [](char C) { return std::isalnum(C); });
}

namespace {
constexpr uint32_t kNoneUdtKind = 0;
constexpr uint32_t kSimpleUdtKind = 1;
constexpr uint32_t kUnknownUdtKind = 2;
const StringRef NoneLabel("<none type>");
const StringRef SimpleLabel("<simple type>");
const StringRef UnknownLabel("<unknown type>");

} // namespace

static StringRef getUdtStatLabel(uint32_t Kind) {
  if (Kind == kNoneUdtKind)
    return NoneLabel;

  if (Kind == kSimpleUdtKind)
    return SimpleLabel;

  if (Kind == kUnknownUdtKind)
    return UnknownLabel;

  return formatTypeLeafKind(static_cast<TypeLeafKind>(Kind));
}

static uint32_t getLongestTypeLeafName(const StatCollection &Stats) {
  size_t L = 0;
  for (const auto &Stat : Stats.Individual) {
    StringRef Label = getUdtStatLabel(Stat.first);
    L = std::max(L, Label.size());
  }
  return static_cast<uint32_t>(L);
}

Error DumpOutputStyle::dumpUdtStats() {
  printHeader(P, "S_UDT Record Stats");

  StatCollection UdtStats;
  StatCollection UdtTargetStats;
  if (!File.hasPDBGlobalsStream()) {
    P.printLine("- Error: globals stream not present");
    return Error::success();
  }

  AutoIndent Indent(P, 4);

  auto &SymbolRecords = cantFail(File.getPDBSymbolStream());
  auto &Globals = cantFail(File.getPDBGlobalsStream());
  auto &TpiTypes = cantFail(initializeTypes(StreamTPI));

  StringMap<StatCollection::Stat> NamespacedStats;

  P.NewLine();

  size_t LongestNamespace = 0;
  for (uint32_t PubSymOff : Globals.getGlobalsTable()) {
    CVSymbol Sym = SymbolRecords.readRecord(PubSymOff);
    if (Sym.kind() != SymbolKind::S_UDT)
      continue;
    UdtStats.update(SymbolKind::S_UDT, Sym.length());

    UDTSym UDT = cantFail(SymbolDeserializer::deserializeAs<UDTSym>(Sym));

    uint32_t Kind = 0;
    uint32_t RecordSize = 0;
    if (UDT.Type.isSimple() ||
        (UDT.Type.toArrayIndex() >= TpiTypes.capacity())) {
      if (UDT.Type.isNoneType())
        Kind = kNoneUdtKind;
      else if (UDT.Type.isSimple())
        Kind = kSimpleUdtKind;
      else
        Kind = kUnknownUdtKind;
    } else {
      CVType T = TpiTypes.getType(UDT.Type);
      Kind = T.kind();
      RecordSize = T.length();
    }

    UdtTargetStats.update(Kind, RecordSize);

    size_t Pos = UDT.Name.find("::");
    if (Pos == StringRef::npos)
      continue;

    StringRef Scope = UDT.Name.take_front(Pos);
    if (Scope.empty() || !isValidNamespaceIdentifier(Scope))
      continue;

    LongestNamespace = std::max(LongestNamespace, Scope.size());
    NamespacedStats[Scope].update(RecordSize);
  }

  LongestNamespace += StringRef(" namespace ''").size();
  size_t LongestTypeLeafKind = getLongestTypeLeafName(UdtTargetStats);
  size_t FieldWidth = std::max(LongestNamespace, LongestTypeLeafKind);

  // Compute the max number of digits for count and size fields, including comma
  // separators.
  StringRef CountHeader("Count");
  StringRef SizeHeader("Size");
  size_t CD = NumDigits(UdtStats.Totals.Count);
  CD += (CD - 1) / 3;
  CD = std::max(CD, CountHeader.size());

  size_t SD = NumDigits(UdtStats.Totals.Size);
  SD += (SD - 1) / 3;
  SD = std::max(SD, SizeHeader.size());

  uint32_t TableWidth = FieldWidth + 3 + CD + 2 + SD + 1;

  P.formatLine("{0} | {1}  {2}",
               fmt_align("Record Kind", AlignStyle::Right, FieldWidth),
               fmt_align(CountHeader, AlignStyle::Right, CD),
               fmt_align(SizeHeader, AlignStyle::Right, SD));

  P.formatLine("{0}", fmt_repeat('-', TableWidth));
  for (const auto &Stat : UdtTargetStats.Individual) {
    StringRef Label = getUdtStatLabel(Stat.first);
    P.formatLine("{0} | {1:N}  {2:N}",
                 fmt_align(Label, AlignStyle::Right, FieldWidth),
                 fmt_align(Stat.second.Count, AlignStyle::Right, CD),
                 fmt_align(Stat.second.Size, AlignStyle::Right, SD));
  }
  P.formatLine("{0}", fmt_repeat('-', TableWidth));
  P.formatLine("{0} | {1:N}  {2:N}",
               fmt_align("Total (S_UDT)", AlignStyle::Right, FieldWidth),
               fmt_align(UdtStats.Totals.Count, AlignStyle::Right, CD),
               fmt_align(UdtStats.Totals.Size, AlignStyle::Right, SD));
  P.formatLine("{0}", fmt_repeat('-', TableWidth));
  for (const auto &Stat : NamespacedStats) {
    std::string Label = formatv("namespace '{0}'", Stat.getKey());
    P.formatLine("{0} | {1:N}  {2:N}",
                 fmt_align(Label, AlignStyle::Right, FieldWidth),
                 fmt_align(Stat.second.Count, AlignStyle::Right, CD),
                 fmt_align(Stat.second.Size, AlignStyle::Right, SD));
  }
  return Error::success();
}

static void typesetLinesAndColumns(PDBFile &File, LinePrinter &P,
                                   uint32_t Start, const LineColumnEntry &E) {
  const uint32_t kMaxCharsPerLineNumber = 4; // 4 digit line number
  uint32_t MinColumnWidth = kMaxCharsPerLineNumber + 5;

  // Let's try to keep it under 100 characters
  constexpr uint32_t kMaxRowLength = 100;
  // At least 3 spaces between columns.
  uint32_t ColumnsPerRow = kMaxRowLength / (MinColumnWidth + 3);
  uint32_t ItemsLeft = E.LineNumbers.size();
  auto LineIter = E.LineNumbers.begin();
  while (ItemsLeft != 0) {
    uint32_t RowColumns = std::min(ItemsLeft, ColumnsPerRow);
    for (uint32_t I = 0; I < RowColumns; ++I) {
      LineInfo Line(LineIter->Flags);
      std::string LineStr;
      if (Line.isAlwaysStepInto())
        LineStr = "ASI";
      else if (Line.isNeverStepInto())
        LineStr = "NSI";
      else
        LineStr = utostr(Line.getStartLine());
      char Statement = Line.isStatement() ? ' ' : '!';
      P.format("{0} {1:X-} {2} ",
               fmt_align(LineStr, AlignStyle::Right, kMaxCharsPerLineNumber),
               fmt_align(Start + LineIter->Offset, AlignStyle::Right, 8, '0'),
               Statement);
      ++LineIter;
      --ItemsLeft;
    }
    P.NewLine();
  }
}

Error DumpOutputStyle::dumpLines() {
  printHeader(P, "Lines");

  uint32_t LastModi = UINT32_MAX;
  uint32_t LastNameIndex = UINT32_MAX;
  iterateModuleSubsections<DebugLinesSubsectionRef>(
      File, P, 4,
      [this, &LastModi, &LastNameIndex](uint32_t Modi,
                                        StringsAndChecksumsPrinter &Strings,
                                        DebugLinesSubsectionRef &Lines) {
        uint16_t Segment = Lines.header()->RelocSegment;
        uint32_t Begin = Lines.header()->RelocOffset;
        uint32_t End = Begin + Lines.header()->CodeSize;
        for (const auto &Block : Lines) {
          if (LastModi != Modi || LastNameIndex != Block.NameIndex) {
            LastModi = Modi;
            LastNameIndex = Block.NameIndex;
            Strings.formatFromChecksumsOffset(P, Block.NameIndex);
          }

          AutoIndent Indent(P, 2);
          P.formatLine("{0:X-4}:{1:X-8}-{2:X-8}, ", Segment, Begin, End);
          uint32_t Count = Block.LineNumbers.size();
          if (Lines.hasColumnInfo())
            P.format("line/column/addr entries = {0}", Count);
          else
            P.format("line/addr entries = {0}", Count);

          P.NewLine();
          typesetLinesAndColumns(File, P, Begin, Block);
        }
      });

  return Error::success();
}

Error DumpOutputStyle::dumpInlineeLines() {
  printHeader(P, "Inlinee Lines");

  iterateModuleSubsections<DebugInlineeLinesSubsectionRef>(
      File, P, 2,
      [this](uint32_t Modi, StringsAndChecksumsPrinter &Strings,
             DebugInlineeLinesSubsectionRef &Lines) {
        P.formatLine("{0,+8} | {1,+5} | {2}", "Inlinee", "Line", "Source File");
        for (const auto &Entry : Lines) {
          P.formatLine("{0,+8} | {1,+5} | ", Entry.Header->Inlinee,
                       fmtle(Entry.Header->SourceLineNum));
          Strings.formatFromChecksumsOffset(P, Entry.Header->FileID, true);
        }
        P.NewLine();
      });

  return Error::success();
}

Error DumpOutputStyle::dumpXmi() {
  printHeader(P, "Cross Module Imports");
  iterateModuleSubsections<DebugCrossModuleImportsSubsectionRef>(
      File, P, 2,
      [this](uint32_t Modi, StringsAndChecksumsPrinter &Strings,
             DebugCrossModuleImportsSubsectionRef &Imports) {
        P.formatLine("{0,=32} | {1}", "Imported Module", "Type IDs");

        for (const auto &Xmi : Imports) {
          auto ExpectedModule =
              Strings.getNameFromStringTable(Xmi.Header->ModuleNameOffset);
          StringRef Module;
          SmallString<32> ModuleStorage;
          if (!ExpectedModule) {
            Module = "(unknown module)";
            consumeError(ExpectedModule.takeError());
          } else
            Module = *ExpectedModule;
          if (Module.size() > 32) {
            ModuleStorage = "...";
            ModuleStorage += Module.take_back(32 - 3);
            Module = ModuleStorage;
          }
          std::vector<std::string> TIs;
          for (const auto I : Xmi.Imports)
            TIs.push_back(formatv("{0,+10:X+}", fmtle(I)));
          std::string Result =
              typesetItemList(TIs, P.getIndentLevel() + 35, 12, " ");
          P.formatLine("{0,+32} | {1}", Module, Result);
        }
      });

  return Error::success();
}

Error DumpOutputStyle::dumpXme() {
  printHeader(P, "Cross Module Exports");

  iterateModuleSubsections<DebugCrossModuleExportsSubsectionRef>(
      File, P, 2,
      [this](uint32_t Modi, StringsAndChecksumsPrinter &Strings,
             DebugCrossModuleExportsSubsectionRef &Exports) {
        P.formatLine("{0,-10} | {1}", "Local ID", "Global ID");
        for (const auto &Export : Exports) {
          P.formatLine("{0,+10:X+} | {1}", TypeIndex(Export.Local),
                       TypeIndex(Export.Global));
        }
      });

  return Error::success();
}

Error DumpOutputStyle::dumpStringTable() {
  printHeader(P, "String Table");

  AutoIndent Indent(P);
  auto IS = File.getStringTable();
  if (!IS) {
    P.formatLine("Not present in file");
    consumeError(IS.takeError());
    return Error::success();
  }

  if (IS->name_ids().empty()) {
    P.formatLine("Empty");
    return Error::success();
  }

  auto MaxID = std::max_element(IS->name_ids().begin(), IS->name_ids().end());
  uint32_t Digits = NumDigits(*MaxID);

  P.formatLine("{0} | {1}", fmt_align("ID", AlignStyle::Right, Digits),
               "String");

  std::vector<uint32_t> SortedIDs(IS->name_ids().begin(), IS->name_ids().end());
  std::sort(SortedIDs.begin(), SortedIDs.end());
  for (uint32_t I : SortedIDs) {
    auto ES = IS->getStringForID(I);
    llvm::SmallString<32> Str;
    if (!ES) {
      consumeError(ES.takeError());
      Str = "Error reading string";
    } else if (!ES->empty()) {
      Str.append("'");
      Str.append(*ES);
      Str.append("'");
    }

    if (!Str.empty())
      P.formatLine("{0} | {1}", fmt_align(I, AlignStyle::Right, Digits), Str);
  }
  return Error::success();
}

static void buildDepSet(LazyRandomTypeCollection &Types,
                        ArrayRef<TypeIndex> Indices,
                        std::map<TypeIndex, CVType> &DepSet) {
  SmallVector<TypeIndex, 4> DepList;
  for (const auto &I : Indices) {
    TypeIndex TI(I);
    if (DepSet.find(TI) != DepSet.end() || TI.isSimple() || TI.isNoneType())
      continue;

    CVType Type = Types.getType(TI);
    DepSet[TI] = Type;
    codeview::discoverTypeIndices(Type, DepList);
    buildDepSet(Types, DepList, DepSet);
  }
}

static void dumpFullTypeStream(LinePrinter &Printer,
                               LazyRandomTypeCollection &Types,
                               TpiStream &Stream, bool Bytes, bool Extras) {
  Printer.formatLine("Showing {0:N} records", Stream.getNumTypeRecords());
  uint32_t Width =
      NumDigits(TypeIndex::FirstNonSimpleIndex + Stream.getNumTypeRecords());

  MinimalTypeDumpVisitor V(Printer, Width + 2, Bytes, Extras, Types,
                           Stream.getNumHashBuckets(), Stream.getHashValues());

  if (auto EC = codeview::visitTypeStream(Types, V)) {
    Printer.formatLine("An error occurred dumping type records: {0}",
                       toString(std::move(EC)));
  }
}

static void dumpPartialTypeStream(LinePrinter &Printer,
                                  LazyRandomTypeCollection &Types,
                                  TpiStream &Stream, ArrayRef<TypeIndex> TiList,
                                  bool Bytes, bool Extras, bool Deps) {
  uint32_t Width =
      NumDigits(TypeIndex::FirstNonSimpleIndex + Stream.getNumTypeRecords());

  MinimalTypeDumpVisitor V(Printer, Width + 2, Bytes, Extras, Types,
                           Stream.getNumHashBuckets(), Stream.getHashValues());

  if (opts::dump::DumpTypeDependents) {
    // If we need to dump all dependents, then iterate each index and find
    // all dependents, adding them to a map ordered by TypeIndex.
    std::map<TypeIndex, CVType> DepSet;
    buildDepSet(Types, TiList, DepSet);

    Printer.formatLine(
        "Showing {0:N} records and their dependents ({1:N} records total)",
        TiList.size(), DepSet.size());

    for (auto &Dep : DepSet) {
      if (auto EC = codeview::visitTypeRecord(Dep.second, Dep.first, V))
        Printer.formatLine("An error occurred dumping type record {0}: {1}",
                           Dep.first, toString(std::move(EC)));
    }
  } else {
    Printer.formatLine("Showing {0:N} records.", TiList.size());

    for (const auto &I : TiList) {
      TypeIndex TI(I);
      CVType Type = Types.getType(TI);
      if (auto EC = codeview::visitTypeRecord(Type, TI, V))
        Printer.formatLine("An error occurred dumping type record {0}: {1}", TI,
                           toString(std::move(EC)));
    }
  }
}

Error DumpOutputStyle::dumpTpiStream(uint32_t StreamIdx) {
  assert(StreamIdx == StreamTPI || StreamIdx == StreamIPI);

  bool Present = false;
  bool DumpTypes = false;
  bool DumpBytes = false;
  bool DumpExtras = false;
  std::vector<uint32_t> Indices;
  if (StreamIdx == StreamTPI) {
    printHeader(P, "Types (TPI Stream)");
    Present = File.hasPDBTpiStream();
    DumpTypes = opts::dump::DumpTypes;
    DumpBytes = opts::dump::DumpTypeData;
    DumpExtras = opts::dump::DumpTypeExtras;
    Indices.assign(opts::dump::DumpTypeIndex.begin(),
                   opts::dump::DumpTypeIndex.end());
  } else if (StreamIdx == StreamIPI) {
    printHeader(P, "Types (IPI Stream)");
    Present = File.hasPDBIpiStream();
    DumpTypes = opts::dump::DumpIds;
    DumpBytes = opts::dump::DumpIdData;
    DumpExtras = opts::dump::DumpIdExtras;
    Indices.assign(opts::dump::DumpIdIndex.begin(),
                   opts::dump::DumpIdIndex.end());
  }

  AutoIndent Indent(P);
  if (!Present) {
    P.formatLine("Stream not present");
    return Error::success();
  }

  ExitOnError Err("Unexpected error processing types: ");

  auto &Stream = Err((StreamIdx == StreamTPI) ? File.getPDBTpiStream()
                                              : File.getPDBIpiStream());

  auto &Types = Err(initializeTypes(StreamIdx));

  if (DumpTypes || !Indices.empty()) {
    if (Indices.empty())
      dumpFullTypeStream(P, Types, Stream, DumpBytes, DumpExtras);
    else {
      std::vector<TypeIndex> TiList(Indices.begin(), Indices.end());
      dumpPartialTypeStream(P, Types, Stream, TiList, DumpBytes, DumpExtras,
                            opts::dump::DumpTypeDependents);
    }
  }

  if (DumpExtras) {
    P.NewLine();
    auto IndexOffsets = Stream.getTypeIndexOffsets();
    P.formatLine("Type Index Offsets:");
    for (const auto &IO : IndexOffsets) {
      AutoIndent Indent2(P);
      P.formatLine("TI: {0}, Offset: {1}", IO.Type, fmtle(IO.Offset));
    }

    P.NewLine();
    P.formatLine("Hash Adjusters:");
    auto &Adjusters = Stream.getHashAdjusters();
    auto &Strings = Err(File.getStringTable());
    for (const auto &A : Adjusters) {
      AutoIndent Indent2(P);
      auto ExpectedStr = Strings.getStringForID(A.first);
      TypeIndex TI(A.second);
      if (ExpectedStr)
        P.formatLine("`{0}` -> {1}", *ExpectedStr, TI);
      else {
        P.formatLine("unknown str id ({0}) -> {1}", A.first, TI);
        consumeError(ExpectedStr.takeError());
      }
    }
  }
  return Error::success();
}

Expected<codeview::LazyRandomTypeCollection &>
DumpOutputStyle::initializeTypes(uint32_t SN) {
  auto &TypeCollection = (SN == StreamTPI) ? TpiTypes : IpiTypes;
  auto Tpi =
      (SN == StreamTPI) ? File.getPDBTpiStream() : File.getPDBIpiStream();
  if (!Tpi)
    return Tpi.takeError();

  if (!TypeCollection) {
    auto &Types = Tpi->typeArray();
    uint32_t Count = Tpi->getNumTypeRecords();
    auto Offsets = Tpi->getTypeIndexOffsets();
    TypeCollection =
        llvm::make_unique<LazyRandomTypeCollection>(Types, Count, Offsets);
  }

  return *TypeCollection;
}

Error DumpOutputStyle::dumpModuleSyms() {
  printHeader(P, "Symbols");

  AutoIndent Indent(P);
  if (!File.hasPDBDbiStream()) {
    P.formatLine("DBI Stream not present");
    return Error::success();
  }

  ExitOnError Err("Unexpected error processing symbols: ");

  auto &Ids = Err(initializeTypes(StreamIPI));
  auto &Types = Err(initializeTypes(StreamTPI));

  iterateModules(
      File, P, 2, [&](uint32_t I, StringsAndChecksumsPrinter &Strings) {
        auto ExpectedModS = getModuleDebugStream(File, I);
        if (!ExpectedModS) {
          P.formatLine("Error loading module stream {0}.  {1}", I,
                       toString(ExpectedModS.takeError()));
          return;
        }

        ModuleDebugStreamRef &ModS = *ExpectedModS;

        SymbolVisitorCallbackPipeline Pipeline;
        SymbolDeserializer Deserializer(nullptr, CodeViewContainer::Pdb);
        MinimalSymbolDumper Dumper(P, opts::dump::DumpSymRecordBytes, Ids,
                                   Types);

        Pipeline.addCallbackToPipeline(Deserializer);
        Pipeline.addCallbackToPipeline(Dumper);
        CVSymbolVisitor Visitor(Pipeline);
        auto SS = ModS.getSymbolsSubstream();
        if (auto EC =
                Visitor.visitSymbolStream(ModS.getSymbolArray(), SS.Offset)) {
          P.formatLine("Error while processing symbol records.  {0}",
                       toString(std::move(EC)));
          return;
        }
      });
  return Error::success();
}

Error DumpOutputStyle::dumpGlobals() {
  printHeader(P, "Global Symbols");
  AutoIndent Indent(P);
  if (!File.hasPDBGlobalsStream()) {
    P.formatLine("Globals stream not present");
    return Error::success();
  }
  ExitOnError Err("Error dumping globals stream: ");
  auto &Globals = Err(File.getPDBGlobalsStream());

  const GSIHashTable &Table = Globals.getGlobalsTable();
  Err(dumpSymbolsFromGSI(Table, opts::dump::DumpGlobalExtras));
  return Error::success();
}

Error DumpOutputStyle::dumpPublics() {
  printHeader(P, "Public Symbols");
  AutoIndent Indent(P);
  if (!File.hasPDBPublicsStream()) {
    P.formatLine("Publics stream not present");
    return Error::success();
  }
  ExitOnError Err("Error dumping publics stream: ");
  auto &Publics = Err(File.getPDBPublicsStream());

  const GSIHashTable &PublicsTable = Publics.getPublicsTable();
  if (opts::dump::DumpPublicExtras) {
    P.printLine("Publics Header");
    AutoIndent Indent(P);
    P.formatLine("sym hash = {0}, thunk table addr = {1}", Publics.getSymHash(),
                 formatSegmentOffset(Publics.getThunkTableSection(),
                                     Publics.getThunkTableOffset()));
  }
  Err(dumpSymbolsFromGSI(PublicsTable, opts::dump::DumpPublicExtras));

  // Skip the rest if we aren't dumping extras.
  if (!opts::dump::DumpPublicExtras)
    return Error::success();

  P.formatLine("Address Map");
  {
    // These are offsets into the publics stream sorted by secidx:secrel.
    AutoIndent Indent2(P);
    for (uint32_t Addr : Publics.getAddressMap())
      P.formatLine("off = {0}", Addr);
  }

  // The thunk map is optional debug info used for ILT thunks.
  if (!Publics.getThunkMap().empty()) {
    P.formatLine("Thunk Map");
    AutoIndent Indent2(P);
    for (uint32_t Addr : Publics.getThunkMap())
      P.formatLine("{0:x8}", Addr);
  }

  // The section offsets table appears to be empty when incremental linking
  // isn't in use.
  if (!Publics.getSectionOffsets().empty()) {
    P.formatLine("Section Offsets");
    AutoIndent Indent2(P);
    for (const SectionOffset &SO : Publics.getSectionOffsets())
      P.formatLine("{0:x4}:{1:x8}", uint16_t(SO.Isect), uint32_t(SO.Off));
  }

  return Error::success();
}

Error DumpOutputStyle::dumpSymbolsFromGSI(const GSIHashTable &Table,
                                          bool HashExtras) {
  auto ExpectedSyms = File.getPDBSymbolStream();
  if (!ExpectedSyms)
    return ExpectedSyms.takeError();
  auto ExpectedTypes = initializeTypes(StreamTPI);
  if (!ExpectedTypes)
    return ExpectedTypes.takeError();
  auto ExpectedIds = initializeTypes(StreamIPI);
  if (!ExpectedIds)
    return ExpectedIds.takeError();

  if (HashExtras) {
    P.printLine("GSI Header");
    AutoIndent Indent(P);
    P.formatLine("sig = {0:X}, hdr = {1:X}, hr size = {2}, num buckets = {3}",
                 Table.getVerSignature(), Table.getVerHeader(),
                 Table.getHashRecordSize(), Table.getNumBuckets());
  }

  {
    P.printLine("Records");
    SymbolVisitorCallbackPipeline Pipeline;
    SymbolDeserializer Deserializer(nullptr, CodeViewContainer::Pdb);
    MinimalSymbolDumper Dumper(P, opts::dump::DumpSymRecordBytes, *ExpectedIds,
                               *ExpectedTypes);

    Pipeline.addCallbackToPipeline(Deserializer);
    Pipeline.addCallbackToPipeline(Dumper);
    CVSymbolVisitor Visitor(Pipeline);

    BinaryStreamRef SymStream =
        ExpectedSyms->getSymbolArray().getUnderlyingStream();
    for (uint32_t PubSymOff : Table) {
      Expected<CVSymbol> Sym = readSymbolFromStream(SymStream, PubSymOff);
      if (!Sym)
        return Sym.takeError();
      if (auto E = Visitor.visitSymbolRecord(*Sym, PubSymOff))
        return E;
    }
  }

  // Return early if we aren't dumping public hash table and address map info.
  if (!HashExtras)
    return Error::success();

  P.formatLine("Hash Entries");
  {
    AutoIndent Indent2(P);
    for (const PSHashRecord &HR : Table.HashRecords)
      P.formatLine("off = {0}, refcnt = {1}", uint32_t(HR.Off),
                   uint32_t(HR.CRef));
  }

  // FIXME: Dump the bitmap.

  P.formatLine("Hash Buckets");
  {
    AutoIndent Indent2(P);
    for (uint32_t Hash : Table.HashBuckets)
      P.formatLine("{0:x8}", Hash);
  }

  return Error::success();
}

static std::string formatSegMapDescriptorFlag(uint32_t IndentLevel,
                                              OMFSegDescFlags Flags) {
  std::vector<std::string> Opts;
  if (Flags == OMFSegDescFlags::None)
    return "none";

  PUSH_FLAG(OMFSegDescFlags, Read, Flags, "read");
  PUSH_FLAG(OMFSegDescFlags, Write, Flags, "write");
  PUSH_FLAG(OMFSegDescFlags, Execute, Flags, "execute");
  PUSH_FLAG(OMFSegDescFlags, AddressIs32Bit, Flags, "32 bit addr");
  PUSH_FLAG(OMFSegDescFlags, IsSelector, Flags, "selector");
  PUSH_FLAG(OMFSegDescFlags, IsAbsoluteAddress, Flags, "absolute addr");
  PUSH_FLAG(OMFSegDescFlags, IsGroup, Flags, "group");
  return typesetItemList(Opts, IndentLevel, 4, " | ");
}

Error DumpOutputStyle::dumpSectionHeaders() {
  dumpSectionHeaders("Section Headers", DbgHeaderType::SectionHdr);
  dumpSectionHeaders("Original Section Headers", DbgHeaderType::SectionHdrOrig);
  return Error::success();
}

static Expected<std::pair<std::unique_ptr<MappedBlockStream>,
                          ArrayRef<llvm::object::coff_section>>>
loadSectionHeaders(PDBFile &File, DbgHeaderType Type) {
  if (!File.hasPDBDbiStream())
    return make_error<StringError>(
        "Section headers require a DBI Stream, which could not be loaded",
        inconvertibleErrorCode());

  auto &Dbi = cantFail(File.getPDBDbiStream());
  uint32_t SI = Dbi.getDebugStreamIndex(Type);

  if (SI == kInvalidStreamIndex)
    return make_error<StringError>(
        "PDB does not contain the requested image section header type",
        inconvertibleErrorCode());

  auto Stream = MappedBlockStream::createIndexedStream(
      File.getMsfLayout(), File.getMsfBuffer(), SI, File.getAllocator());
  if (!Stream)
    return make_error<StringError>("Could not load the required stream data",
                                   inconvertibleErrorCode());

  ArrayRef<object::coff_section> Headers;
  if (Stream->getLength() % sizeof(object::coff_section) != 0)
    return make_error<StringError>(
        "Section header array size is not a multiple of section header size",
        inconvertibleErrorCode());

  uint32_t NumHeaders = Stream->getLength() / sizeof(object::coff_section);
  BinaryStreamReader Reader(*Stream);
  cantFail(Reader.readArray(Headers, NumHeaders));
  return std::make_pair(std::move(Stream), Headers);
}

void DumpOutputStyle::dumpSectionHeaders(StringRef Label, DbgHeaderType Type) {
  printHeader(P, Label);
  ExitOnError Err("Error dumping publics stream: ");

  AutoIndent Indent(P);
  std::unique_ptr<MappedBlockStream> Stream;
  ArrayRef<object::coff_section> Headers;
  auto ExpectedHeaders = loadSectionHeaders(File, Type);
  if (!ExpectedHeaders) {
    P.printLine(toString(ExpectedHeaders.takeError()));
    return;
  }
  std::tie(Stream, Headers) = std::move(*ExpectedHeaders);

  uint32_t I = 1;
  for (const auto &Header : Headers) {
    P.NewLine();
    P.formatLine("SECTION HEADER #{0}", I);
    P.formatLine("{0,8} name", Header.Name);
    P.formatLine("{0,8:X-} virtual size", uint32_t(Header.VirtualSize));
    P.formatLine("{0,8:X-} virtual address", uint32_t(Header.VirtualAddress));
    P.formatLine("{0,8:X-} size of raw data", uint32_t(Header.SizeOfRawData));
    P.formatLine("{0,8:X-} file pointer to raw data",
                 uint32_t(Header.PointerToRawData));
    P.formatLine("{0,8:X-} file pointer to relocation table",
                 uint32_t(Header.PointerToRelocations));
    P.formatLine("{0,8:X-} file pointer to line numbers",
                 uint32_t(Header.PointerToLinenumbers));
    P.formatLine("{0,8:X-} number of relocations",
                 uint32_t(Header.NumberOfRelocations));
    P.formatLine("{0,8:X-} number of line numbers",
                 uint32_t(Header.NumberOfLinenumbers));
    P.formatLine("{0,8:X-} flags", uint32_t(Header.Characteristics));
    AutoIndent IndentMore(P, 9);
    P.formatLine("{0}", formatSectionCharacteristics(
                            P.getIndentLevel(), Header.Characteristics, 1, ""));
    ++I;
  }
  return;
}

std::vector<std::string> getSectionNames(PDBFile &File) {
  auto ExpectedHeaders = loadSectionHeaders(File, DbgHeaderType::SectionHdr);
  if (!ExpectedHeaders)
    return {};

  std::unique_ptr<MappedBlockStream> Stream;
  ArrayRef<object::coff_section> Headers;
  std::tie(Stream, Headers) = std::move(*ExpectedHeaders);
  std::vector<std::string> Names;
  for (const auto &H : Headers)
    Names.push_back(H.Name);
  return Names;
}

Error DumpOutputStyle::dumpSectionContribs() {
  printHeader(P, "Section Contributions");
  ExitOnError Err("Error dumping publics stream: ");

  AutoIndent Indent(P);
  if (!File.hasPDBDbiStream()) {
    P.formatLine(
        "Section contribs require a DBI Stream, which could not be loaded");
    return Error::success();
  }

  auto &Dbi = Err(File.getPDBDbiStream());

  class Visitor : public ISectionContribVisitor {
  public:
    Visitor(LinePrinter &P, ArrayRef<std::string> Names) : P(P), Names(Names) {
      auto Max = std::max_element(
          Names.begin(), Names.end(),
          [](StringRef S1, StringRef S2) { return S1.size() < S2.size(); });
      MaxNameLen = (Max == Names.end() ? 0 : Max->size());
    }
    void visit(const SectionContrib &SC) override {
      assert(SC.ISect > 0);
      std::string NameInsert;
      if (SC.ISect < Names.size()) {
        StringRef SectionName = Names[SC.ISect - 1];
        NameInsert = formatv("[{0}]", SectionName).str();
      } else
        NameInsert = "[???]";
      P.formatLine("SC{5}  | mod = {2}, {0}, size = {1}, data crc = {3}, reloc "
                   "crc = {4}",
                   formatSegmentOffset(SC.ISect, SC.Off), fmtle(SC.Size),
                   fmtle(SC.Imod), fmtle(SC.DataCrc), fmtle(SC.RelocCrc),
                   fmt_align(NameInsert, AlignStyle::Left, MaxNameLen + 2));
      AutoIndent Indent(P, MaxNameLen + 2);
      P.formatLine("      {0}",
                   formatSectionCharacteristics(P.getIndentLevel() + 6,
                                                SC.Characteristics, 3, " | "));
    }
    void visit(const SectionContrib2 &SC) override {
      P.formatLine(
          "SC2[{6}] | mod = {2}, {0}, size = {1}, data crc = {3}, reloc "
          "crc = {4}, coff section = {5}",
          formatSegmentOffset(SC.Base.ISect, SC.Base.Off), fmtle(SC.Base.Size),
          fmtle(SC.Base.Imod), fmtle(SC.Base.DataCrc), fmtle(SC.Base.RelocCrc),
          fmtle(SC.ISectCoff));
      P.formatLine("      {0}", formatSectionCharacteristics(
                                    P.getIndentLevel() + 6,
                                    SC.Base.Characteristics, 3, " | "));
    }

  private:
    LinePrinter &P;
    uint32_t MaxNameLen;
    ArrayRef<std::string> Names;
  };

  std::vector<std::string> Names = getSectionNames(File);
  Visitor V(P, makeArrayRef(Names));
  Dbi.visitSectionContributions(V);
  return Error::success();
}

Error DumpOutputStyle::dumpSectionMap() {
  printHeader(P, "Section Map");
  ExitOnError Err("Error dumping section map: ");

  AutoIndent Indent(P);
  if (!File.hasPDBDbiStream()) {
    P.formatLine("Dumping the section map requires a DBI Stream, which could "
                 "not be loaded");
    return Error::success();
  }

  auto &Dbi = Err(File.getPDBDbiStream());

  uint32_t I = 0;
  for (auto &M : Dbi.getSectionMap()) {
    P.formatLine(
        "Section {0:4} | ovl = {1}, group = {2}, frame = {3}, name = {4}", I,
        fmtle(M.Ovl), fmtle(M.Group), fmtle(M.Frame), fmtle(M.SecName));
    P.formatLine("               class = {0}, offset = {1}, size = {2}",
                 fmtle(M.ClassName), fmtle(M.Offset), fmtle(M.SecByteLength));
    P.formatLine("               flags = {0}",
                 formatSegMapDescriptorFlag(
                     P.getIndentLevel() + 13,
                     static_cast<OMFSegDescFlags>(uint16_t(M.Flags))));
    ++I;
  }
  return Error::success();
}
