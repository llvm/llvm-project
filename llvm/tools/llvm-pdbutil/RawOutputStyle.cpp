//===- RawOutputStyle.cpp ------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RawOutputStyle.h"

#include "CompactTypeDumpVisitor.h"
#include "FormatUtil.h"
#include "MinimalSymbolDumper.h"
#include "MinimalTypeDumper.h"
#include "StreamUtil.h"
#include "llvm-pdbutil.h"

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

  if (opts::raw::DumpTypes) {
    if (auto EC = dumpTpiStream(StreamTPI))
      return EC;
  }

  if (opts::raw::DumpIds) {
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

Error RawOutputStyle::dumpModules() {
  printHeader(P, "Modules");

  AutoIndent Indent(P);
  if (!File.hasPDBDbiStream()) {
    P.formatLine("DBI Stream not present");
    return Error::success();
  }

  ExitOnError Err("Unexpected error processing symbols");

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
    if (opts::raw::DumpModuleFiles) {
      P.formatLine("           contributing source files:");
      for (const auto &F : Modules.source_files(I)) {
        P.formatLine("           - {0}", F);
      }
    }
  }
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
  bool DumpBytes = false;
  if (StreamIdx == StreamTPI) {
    printHeader(P, "Types (TPI Stream)");
    Present = File.hasPDBTpiStream();
    DumpBytes = opts::raw::DumpTypeData;
  } else if (StreamIdx == StreamIPI) {
    printHeader(P, "Types (IPI Stream)");
    Present = File.hasPDBIpiStream();
    DumpBytes = opts::raw::DumpIdData;
  }

  AutoIndent Indent(P);
  if (!Present) {
    P.formatLine("Stream not present");
    return Error::success();
  }

  ExitOnError Err("Unexpected error processing types");

  auto &Stream = Err((StreamIdx == StreamTPI) ? File.getPDBTpiStream()
                                              : File.getPDBIpiStream());

  auto &Types = Err(initializeTypeDatabase(StreamIdx));

  P.formatLine("Showing {0:N} records", Stream.getNumTypeRecords());
  uint32_t Width =
      NumDigits(TypeIndex::FirstNonSimpleIndex + Stream.getNumTypeRecords());

  MinimalTypeDumpVisitor V(P, Width + 2, DumpBytes, Types);

  Optional<TypeIndex> I = Types.getFirst();
  if (auto EC = codeview::visitTypeStream(Types, V)) {
    P.formatLine("An error occurred dumping type records: {0}",
                 toString(std::move(EC)));
  }
  return Error::success();
}

Expected<codeview::LazyRandomTypeCollection &>
RawOutputStyle::initializeTypeDatabase(uint32_t SN) {
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

  auto &Types = Err(initializeTypeDatabase(StreamTPI));

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

  auto &Types = Err(initializeTypeDatabase(StreamTPI));
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
  return typesetItemList(Opts, 3, IndentLevel, " | ");
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
  return typesetItemList(Opts, 4, IndentLevel, " | ");
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
