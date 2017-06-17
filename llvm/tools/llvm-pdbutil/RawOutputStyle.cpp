//===- RawOutputStyle.cpp ------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RawOutputStyle.h"

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
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/TpiHashing.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"

#include <unordered_map>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

RawOutputStyle::RawOutputStyle(PDBFile &File)
    : File(File), P(2, false, outs()) {}

Error RawOutputStyle::dump() {
  if (opts::raw::DumpSummary) {
    if (auto EC = dumpFileSummary())
      return EC;
    P.NewLine();
  }

  if (opts::raw::DumpStreams) {
    if (auto EC = dumpStreamSummary())
      return EC;
    P.NewLine();
  }

  if (opts::raw::DumpBlockRange.hasValue()) {
    if (auto EC = dumpBlockRanges())
      return EC;
    P.NewLine();
  }

  if (!opts::raw::DumpStreamData.empty()) {
    if (auto EC = dumpStreamBytes())
      return EC;
    P.NewLine();
  }

  if (opts::raw::DumpStringTable) {
    if (auto EC = dumpStringTable())
      return EC;
    P.NewLine();
  }

  if (opts::raw::DumpModules) {
    if (auto EC = dumpModules())
      return EC;
  }

  if (opts::raw::DumpModuleFiles) {
    if (auto EC = dumpModuleFiles())
      return EC;
  }

  if (opts::raw::DumpLines) {
    if (auto EC = dumpLines())
      return EC;
  }

  if (opts::raw::DumpInlineeLines) {
    if (auto EC = dumpInlineeLines())
      return EC;
  }

  if (opts::raw::DumpXmi) {
    if (auto EC = dumpXmi())
      return EC;
  }

  if (opts::raw::DumpXme) {
    if (auto EC = dumpXme())
      return EC;
  }

  if (opts::raw::DumpTypes || opts::raw::DumpTypeExtras) {
    if (auto EC = dumpTpiStream(StreamTPI))
      return EC;
  }

  if (opts::raw::DumpIds || opts::raw::DumpIdExtras) {
    if (auto EC = dumpTpiStream(StreamIPI))
      return EC;
  }

  if (opts::raw::DumpPublics) {
    if (auto EC = dumpPublics())
      return EC;
  }

  if (opts::raw::DumpSymbols) {
    if (auto EC = dumpModuleSyms())
      return EC;
  }

  if (opts::raw::DumpSectionContribs) {
    if (auto EC = dumpSectionContribs())
      return EC;
  }

  if (opts::raw::DumpSectionMap) {
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

Error RawOutputStyle::dumpFileSummary() {
  printHeader(P, "Summary");

  ExitOnError Err("Invalid PDB Format");

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

Error RawOutputStyle::dumpStreamSummary() {
  printHeader(P, "Streams");

  if (StreamPurposes.empty())
    discoverStreamPurposes(File, StreamPurposes);

  AutoIndent Indent(P);
  uint32_t StreamCount = File.getNumStreams();

  for (uint16_t StreamIdx = 0; StreamIdx < StreamCount; ++StreamIdx) {
    P.formatLine(
        "Stream {0}: [{1}] ({2} bytes)",
        fmt_align(StreamIdx, AlignStyle::Right, NumDigits(StreamCount)),
        StreamPurposes[StreamIdx], File.getStreamByteSize(StreamIdx));
  }

  return Error::success();
}

Error RawOutputStyle::dumpBlockRanges() {
  printHeader(P, "MSF Blocks");

  auto &R = *opts::raw::DumpBlockRange;
  uint32_t Max = R.Max.getValueOr(R.Min);

  AutoIndent Indent(P);
  if (Max < R.Min)
    return make_error<StringError>(
        "Invalid block range specified.  Max < Min",
        std::make_error_code(std::errc::bad_address));
  if (Max >= File.getBlockCount())
    return make_error<StringError>(
        "Invalid block range specified.  Requested block out of bounds",
        std::make_error_code(std::errc::bad_address));

  for (uint32_t I = R.Min; I <= Max; ++I) {
    auto ExpectedData = File.getBlockData(I, File.getBlockSize());
    if (!ExpectedData)
      return ExpectedData.takeError();
    std::string Label = formatv("Block {0}", I).str();
    P.formatBinary(Label, *ExpectedData, 0);
  }

  return Error::success();
}

static Error parseStreamSpec(StringRef Str, uint32_t &SI, uint32_t &Offset,
                             uint32_t &Size) {
  if (Str.consumeInteger(0, SI))
    return make_error<RawError>(raw_error_code::invalid_format,
                                "Invalid Stream Specification");
  if (Str.consume_front(":")) {
    if (Str.consumeInteger(0, Offset))
      return make_error<RawError>(raw_error_code::invalid_format,
                                  "Invalid Stream Specification");
  }
  if (Str.consume_front("@")) {
    if (Str.consumeInteger(0, Size))
      return make_error<RawError>(raw_error_code::invalid_format,
                                  "Invalid Stream Specification");
  }
  if (!Str.empty())
    return make_error<RawError>(raw_error_code::invalid_format,
                                "Invalid Stream Specification");
  return Error::success();
}

Error RawOutputStyle::dumpStreamBytes() {
  if (StreamPurposes.empty())
    discoverStreamPurposes(File, StreamPurposes);

  printHeader(P, "Stream Data");
  ExitOnError Err("Unexpected error reading stream data");

  for (auto &Str : opts::raw::DumpStreamData) {
    uint32_t SI = 0;
    uint32_t Begin = 0;
    uint32_t Size = 0;
    uint32_t End = 0;

    if (auto EC = parseStreamSpec(Str, SI, Begin, Size))
      return EC;

    AutoIndent Indent(P);
    if (SI >= File.getNumStreams()) {
      P.formatLine("Stream {0}: Not present", SI);
      continue;
    }

    auto S = MappedBlockStream::createIndexedStream(
        File.getMsfLayout(), File.getMsfBuffer(), SI, File.getAllocator());
    if (!S) {
      P.NewLine();
      P.formatLine("Stream {0}: Not present", SI);
      continue;
    }

    if (Size == 0)
      End = S->getLength();
    else
      End = std::min(Begin + Size, S->getLength());

    P.formatLine("Stream {0} ({1:N} bytes): {2}", SI, S->getLength(),
                 StreamPurposes[SI]);
    AutoIndent Indent2(P);

    BinaryStreamReader R(*S);
    ArrayRef<uint8_t> StreamData;
    Err(R.readBytes(StreamData, S->getLength()));
    Size = End - Begin;
    StreamData = StreamData.slice(Begin, Size);
    P.formatBinary("Data", StreamData, Begin);
  }
  return Error::success();
}

static Expected<ModuleDebugStreamRef> getModuleDebugStream(PDBFile &File,
                                                           uint32_t Index) {
  ExitOnError Err("Unexpected error");

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
    ExitOnError Err("Unexpected error processing modules");
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
static void iterateModules(PDBFile &File, LinePrinter &P, uint32_t IndentLevel,
                           CallbackT Callback) {
  AutoIndent Indent(P);
  if (!File.hasPDBDbiStream()) {
    P.formatLine("DBI Stream not present");
    return;
  }

  ExitOnError Err("Unexpected error processing modules");

  auto &Stream = Err(File.getPDBDbiStream());

  const DbiModuleList &Modules = Stream.modules();
  uint32_t Count = Modules.getModuleCount();
  uint32_t Digits = NumDigits(Count);
  for (uint32_t I = 0; I < Count; ++I) {
    auto Modi = Modules.getModuleDescriptor(I);
    P.formatLine("Mod {0:4} | `{1}`: ", fmt_align(I, AlignStyle::Right, Digits),
                 Modi.getModuleName());

    StringsAndChecksumsPrinter Strings(File, I);
    AutoIndent Indent2(P, IndentLevel);
    Callback(I, Strings);
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

Error RawOutputStyle::dumpModules() {
  printHeader(P, "Modules");

  AutoIndent Indent(P);
  if (!File.hasPDBDbiStream()) {
    P.formatLine("DBI Stream not present");
    return Error::success();
  }

  ExitOnError Err("Unexpected error processing modules");

  auto &Stream = Err(File.getPDBDbiStream());

  const DbiModuleList &Modules = Stream.modules();
  uint32_t Count = Modules.getModuleCount();
  uint32_t Digits = NumDigits(Count);
  for (uint32_t I = 0; I < Count; ++I) {
    auto Modi = Modules.getModuleDescriptor(I);
    P.formatLine("Mod {0:4} | Name: `{1}`: ",
                 fmt_align(I, AlignStyle::Right, Digits), Modi.getModuleName());
    P.formatLine("           Obj: `{0}`: ", Modi.getObjFileName());
    P.formatLine("           debug stream: {0}, # files: {1}, has ec info: {2}",
                 Modi.getModuleStreamIndex(), Modi.getNumberOfFiles(),
                 Modi.hasECInfo());
  }
  return Error::success();
}

Error RawOutputStyle::dumpModuleFiles() {
  printHeader(P, "Files");

  ExitOnError Err("Unexpected error processing modules");

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

Error RawOutputStyle::dumpLines() {
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

Error RawOutputStyle::dumpInlineeLines() {
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

Error RawOutputStyle::dumpXmi() {
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

Error RawOutputStyle::dumpXme() {
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

Error RawOutputStyle::dumpStringTable() {
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

Error RawOutputStyle::dumpTpiStream(uint32_t StreamIdx) {
  assert(StreamIdx == StreamTPI || StreamIdx == StreamIPI);

  bool Present = false;
  bool DumpTypes = false;
  bool DumpBytes = false;
  bool DumpExtras = false;
  std::vector<uint32_t> Indices;
  if (StreamIdx == StreamTPI) {
    printHeader(P, "Types (TPI Stream)");
    Present = File.hasPDBTpiStream();
    DumpTypes = opts::raw::DumpTypes;
    DumpBytes = opts::raw::DumpTypeData;
    DumpExtras = opts::raw::DumpTypeExtras;
    Indices.assign(opts::raw::DumpTypeIndex.begin(),
                   opts::raw::DumpTypeIndex.end());
  } else if (StreamIdx == StreamIPI) {
    printHeader(P, "Types (IPI Stream)");
    Present = File.hasPDBIpiStream();
    DumpTypes = opts::raw::DumpIds;
    DumpBytes = opts::raw::DumpIdData;
    DumpExtras = opts::raw::DumpIdExtras;
    Indices.assign(opts::raw::DumpIdIndex.begin(),
                   opts::raw::DumpIdIndex.end());
  }

  AutoIndent Indent(P);
  if (!Present) {
    P.formatLine("Stream not present");
    return Error::success();
  }

  ExitOnError Err("Unexpected error processing types");

  auto &Stream = Err((StreamIdx == StreamTPI) ? File.getPDBTpiStream()
                                              : File.getPDBIpiStream());

  auto &Types = Err(initializeTypes(StreamIdx));

  if (DumpTypes) {
    P.formatLine("Showing {0:N} records", Stream.getNumTypeRecords());
    uint32_t Width =
        NumDigits(TypeIndex::FirstNonSimpleIndex + Stream.getNumTypeRecords());

    MinimalTypeDumpVisitor V(P, Width + 2, DumpBytes, DumpExtras, Types,
                             Stream.getHashValues());

    if (Indices.empty()) {
      if (auto EC = codeview::visitTypeStream(Types, V)) {
        P.formatLine("An error occurred dumping type records: {0}",
                     toString(std::move(EC)));
      }
    } else {
      for (const auto &I : Indices) {
        TypeIndex TI(I);
        CVType Type = Types.getType(TI);
        if (auto EC = codeview::visitTypeRecord(Type, TI, V))
          P.formatLine("An error occurred dumping type record {0}: {1}", TI,
                       toString(std::move(EC)));
      }
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
RawOutputStyle::initializeTypes(uint32_t SN) {
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

Error RawOutputStyle::dumpModuleSyms() {
  printHeader(P, "Symbols");

  AutoIndent Indent(P);
  if (!File.hasPDBDbiStream()) {
    P.formatLine("DBI Stream not present");
    return Error::success();
  }

  ExitOnError Err("Unexpected error processing symbols");

  auto &Stream = Err(File.getPDBDbiStream());

  auto &Types = Err(initializeTypes(StreamTPI));

  const DbiModuleList &Modules = Stream.modules();
  uint32_t Count = Modules.getModuleCount();
  uint32_t Digits = NumDigits(Count);
  for (uint32_t I = 0; I < Count; ++I) {
    auto Modi = Modules.getModuleDescriptor(I);
    P.formatLine("Mod {0:4} | `{1}`: ", fmt_align(I, AlignStyle::Right, Digits),
                 Modi.getModuleName());
    uint16_t ModiStream = Modi.getModuleStreamIndex();
    if (ModiStream == kInvalidStreamIndex) {
      P.formatLine("           <symbols not present>");
      continue;
    }
    auto ModStreamData = MappedBlockStream::createIndexedStream(
        File.getMsfLayout(), File.getMsfBuffer(), ModiStream,
        File.getAllocator());

    ModuleDebugStreamRef ModS(Modi, std::move(ModStreamData));
    if (auto EC = ModS.reload()) {
      P.formatLine("Error loading module stream {0}.  {1}", I,
                   toString(std::move(EC)));
      continue;
    }

    SymbolVisitorCallbackPipeline Pipeline;
    SymbolDeserializer Deserializer(nullptr, CodeViewContainer::Pdb);
    MinimalSymbolDumper Dumper(P, opts::raw::DumpSymRecordBytes, Types);

    Pipeline.addCallbackToPipeline(Deserializer);
    Pipeline.addCallbackToPipeline(Dumper);
    CVSymbolVisitor Visitor(Pipeline);
    if (auto EC = Visitor.visitSymbolStream(ModS.getSymbolArray())) {
      P.formatLine("Error while processing symbol records.  {0}",
                   toString(std::move(EC)));
      continue;
    }
  }
  return Error::success();
}

Error RawOutputStyle::dumpPublics() {
  printHeader(P, "Public Symbols");

  AutoIndent Indent(P);
  if (!File.hasPDBPublicsStream()) {
    P.formatLine("Publics stream not present");
    return Error::success();
  }

  ExitOnError Err("Error dumping publics stream");

  auto &Types = Err(initializeTypes(StreamTPI));
  auto &Publics = Err(File.getPDBPublicsStream());
  SymbolVisitorCallbackPipeline Pipeline;
  SymbolDeserializer Deserializer(nullptr, CodeViewContainer::Pdb);
  MinimalSymbolDumper Dumper(P, opts::raw::DumpSymRecordBytes, Types);

  Pipeline.addCallbackToPipeline(Deserializer);
  Pipeline.addCallbackToPipeline(Dumper);
  CVSymbolVisitor Visitor(Pipeline);
  auto ExpectedSymbols = Publics.getSymbolArray();
  if (!ExpectedSymbols) {
    P.formatLine("Could not read public symbol record stream");
    return Error::success();
  }

  if (auto EC = Visitor.visitSymbolStream(*ExpectedSymbols))
    P.formatLine("Error while processing public symbol records.  {0}",
                 toString(std::move(EC)));

  return Error::success();
}

static std::string formatSectionCharacteristics(uint32_t IndentLevel,
                                                uint32_t C) {
  using SC = COFF::SectionCharacteristics;
  std::vector<std::string> Opts;
  if (C == COFF::SC_Invalid)
    return "invalid";
  if (C == 0)
    return "none";

  PUSH_FLAG(SC, IMAGE_SCN_TYPE_NOLOAD, C, "IMAGE_SCN_TYPE_NOLOAD");
  PUSH_FLAG(SC, IMAGE_SCN_TYPE_NO_PAD, C, "IMAGE_SCN_TYPE_NO_PAD");
  PUSH_FLAG(SC, IMAGE_SCN_CNT_CODE, C, "IMAGE_SCN_CNT_CODE");
  PUSH_FLAG(SC, IMAGE_SCN_CNT_INITIALIZED_DATA, C,
            "IMAGE_SCN_CNT_INITIALIZED_DATA");
  PUSH_FLAG(SC, IMAGE_SCN_CNT_UNINITIALIZED_DATA, C,
            "IMAGE_SCN_CNT_UNINITIALIZED_DATA");
  PUSH_FLAG(SC, IMAGE_SCN_LNK_OTHER, C, "IMAGE_SCN_LNK_OTHER");
  PUSH_FLAG(SC, IMAGE_SCN_LNK_INFO, C, "IMAGE_SCN_LNK_INFO");
  PUSH_FLAG(SC, IMAGE_SCN_LNK_REMOVE, C, "IMAGE_SCN_LNK_REMOVE");
  PUSH_FLAG(SC, IMAGE_SCN_LNK_COMDAT, C, "IMAGE_SCN_LNK_COMDAT");
  PUSH_FLAG(SC, IMAGE_SCN_GPREL, C, "IMAGE_SCN_GPREL");
  PUSH_FLAG(SC, IMAGE_SCN_MEM_PURGEABLE, C, "IMAGE_SCN_MEM_PURGEABLE");
  PUSH_FLAG(SC, IMAGE_SCN_MEM_16BIT, C, "IMAGE_SCN_MEM_16BIT");
  PUSH_FLAG(SC, IMAGE_SCN_MEM_LOCKED, C, "IMAGE_SCN_MEM_LOCKED");
  PUSH_FLAG(SC, IMAGE_SCN_MEM_PRELOAD, C, "IMAGE_SCN_MEM_PRELOAD");
  PUSH_FLAG(SC, IMAGE_SCN_GPREL, C, "IMAGE_SCN_GPREL");
  PUSH_FLAG(SC, IMAGE_SCN_GPREL, C, "IMAGE_SCN_GPREL");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_1BYTES, C,
                   "IMAGE_SCN_ALIGN_1BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_2BYTES, C,
                   "IMAGE_SCN_ALIGN_2BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_4BYTES, C,
                   "IMAGE_SCN_ALIGN_4BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_8BYTES, C,
                   "IMAGE_SCN_ALIGN_8BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_16BYTES, C,
                   "IMAGE_SCN_ALIGN_16BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_32BYTES, C,
                   "IMAGE_SCN_ALIGN_32BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_64BYTES, C,
                   "IMAGE_SCN_ALIGN_64BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_128BYTES, C,
                   "IMAGE_SCN_ALIGN_128BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_256BYTES, C,
                   "IMAGE_SCN_ALIGN_256BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_512BYTES, C,
                   "IMAGE_SCN_ALIGN_512BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_1024BYTES, C,
                   "IMAGE_SCN_ALIGN_1024BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_2048BYTES, C,
                   "IMAGE_SCN_ALIGN_2048BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_4096BYTES, C,
                   "IMAGE_SCN_ALIGN_4096BYTES");
  PUSH_MASKED_FLAG(SC, 0xF00000, IMAGE_SCN_ALIGN_8192BYTES, C,
                   "IMAGE_SCN_ALIGN_8192BYTES");
  PUSH_FLAG(SC, IMAGE_SCN_LNK_NRELOC_OVFL, C, "IMAGE_SCN_LNK_NRELOC_OVFL");
  PUSH_FLAG(SC, IMAGE_SCN_MEM_DISCARDABLE, C, "IMAGE_SCN_MEM_DISCARDABLE");
  PUSH_FLAG(SC, IMAGE_SCN_MEM_NOT_CACHED, C, "IMAGE_SCN_MEM_NOT_CACHED");
  PUSH_FLAG(SC, IMAGE_SCN_MEM_NOT_PAGED, C, "IMAGE_SCN_MEM_NOT_PAGED");
  PUSH_FLAG(SC, IMAGE_SCN_MEM_SHARED, C, "IMAGE_SCN_MEM_SHARED");
  PUSH_FLAG(SC, IMAGE_SCN_MEM_EXECUTE, C, "IMAGE_SCN_MEM_EXECUTE");
  PUSH_FLAG(SC, IMAGE_SCN_MEM_READ, C, "IMAGE_SCN_MEM_READ");
  PUSH_FLAG(SC, IMAGE_SCN_MEM_WRITE, C, "IMAGE_SCN_MEM_WRITE");
  return typesetItemList(Opts, IndentLevel, 3, " | ");
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

Error RawOutputStyle::dumpSectionContribs() {
  printHeader(P, "Section Contributions");
  ExitOnError Err("Error dumping publics stream");

  AutoIndent Indent(P);
  if (!File.hasPDBDbiStream()) {
    P.formatLine(
        "Section contribs require a DBI Stream, which could not be loaded");
    return Error::success();
  }

  auto &Dbi = Err(File.getPDBDbiStream());

  class Visitor : public ISectionContribVisitor {
  public:
    Visitor(LinePrinter &P) : P(P) {}
    void visit(const SectionContrib &SC) override {
      P.formatLine(
          "SC  | mod = {2}, {0}, size = {1}, data crc = {3}, reloc crc = {4}",
          formatSegmentOffset(SC.ISect, SC.Off), fmtle(SC.Size), fmtle(SC.Imod),
          fmtle(SC.DataCrc), fmtle(SC.RelocCrc));
      P.formatLine("      {0}",
                   formatSectionCharacteristics(P.getIndentLevel() + 6,
                                                SC.Characteristics));
    }
    void visit(const SectionContrib2 &SC) override {
      P.formatLine("SC2 | mod = {2}, {0}, size = {1}, data crc = {3}, reloc "
                   "crc = {4}, coff section = {5}",
                   formatSegmentOffset(SC.Base.ISect, SC.Base.Off),
                   fmtle(SC.Base.Size), fmtle(SC.Base.Imod),
                   fmtle(SC.Base.DataCrc), fmtle(SC.Base.RelocCrc),
                   fmtle(SC.ISectCoff));
      P.formatLine("      {0}",
                   formatSectionCharacteristics(P.getIndentLevel() + 6,
                                                SC.Base.Characteristics));
    }

  private:
    LinePrinter &P;
  };

  Visitor V(P);
  Dbi.visitSectionContributions(V);
  return Error::success();
}

Error RawOutputStyle::dumpSectionMap() {
  printHeader(P, "Section Map");
  ExitOnError Err("Error dumping section map");

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
        "Section {0:4} | ovl = {0}, group = {1}, frame = {2}, name = {3}", I,
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
