//===- LLVMOutputStyle.cpp ------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LLVMOutputStyle.h"

#include "llvm-pdbdump.h"
#include "llvm/DebugInfo/CodeView/EnumTables.h"
#include "llvm/DebugInfo/CodeView/ModuleSubstreamVisitor.h"
#include "llvm/DebugInfo/CodeView/SymbolDumper.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/MSF/StreamReader.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/EnumTables.h"
#include "llvm/DebugInfo/PDB/Raw/ISectionContribVisitor.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/ModInfo.h"
#include "llvm/DebugInfo/PDB/Raw/ModStream.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/PublicsStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/TpiStream.h"
#include "llvm/Object/COFF.h"

#include <unordered_map>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

namespace {
struct PageStats {
  explicit PageStats(const BitVector &FreePages)
      : Upm(FreePages), ActualUsedPages(FreePages.size()),
        MultiUsePages(FreePages.size()), UseAfterFreePages(FreePages.size()) {
    const_cast<BitVector &>(Upm).flip();
    // To calculate orphaned pages, we start with the set of pages that the
    // MSF thinks are used.  Each time we find one that actually *is* used,
    // we unset it.  Whichever bits remain set at the end are orphaned.
    OrphanedPages = Upm;
  }

  // The inverse of the MSF File's copy of the Fpm.  The basis for which we
  // determine the allocation status of each page.
  const BitVector Upm;

  // Pages which are marked as used in the FPM and are used at least once.
  BitVector ActualUsedPages;

  // Pages which are marked as used in the FPM but are used more than once.
  BitVector MultiUsePages;

  // Pages which are marked as used in the FPM but are not used at all.
  BitVector OrphanedPages;

  // Pages which are marked free in the FPM but are used.
  BitVector UseAfterFreePages;
};
}

static void recordKnownUsedPage(PageStats &Stats, uint32_t UsedIndex) {
  if (Stats.Upm.test(UsedIndex)) {
    if (Stats.ActualUsedPages.test(UsedIndex))
      Stats.MultiUsePages.set(UsedIndex);
    Stats.ActualUsedPages.set(UsedIndex);
    Stats.OrphanedPages.reset(UsedIndex);
  } else {
    // The MSF doesn't think this page is used, but it is.
    Stats.UseAfterFreePages.set(UsedIndex);
  }
}

static void printSectionOffset(llvm::raw_ostream &OS,
                               const SectionOffset &Off) {
  OS << Off.Off << ", " << Off.Isect;
}

LLVMOutputStyle::LLVMOutputStyle(PDBFile &File)
    : File(File), P(outs()), Dumper(&P, false) {}

Error LLVMOutputStyle::dump() {
  if (auto EC = dumpFileHeaders())
    return EC;

  if (auto EC = dumpStreamSummary())
    return EC;

  if (auto EC = dumpFreePageMap())
    return EC;

  if (auto EC = dumpStreamBlocks())
    return EC;

  if (auto EC = dumpStreamData())
    return EC;

  if (auto EC = dumpInfoStream())
    return EC;

  if (auto EC = dumpNamedStream())
    return EC;

  if (auto EC = dumpTpiStream(StreamTPI))
    return EC;

  if (auto EC = dumpTpiStream(StreamIPI))
    return EC;

  if (auto EC = dumpDbiStream())
    return EC;

  if (auto EC = dumpSectionContribs())
    return EC;

  if (auto EC = dumpSectionMap())
    return EC;

  if (auto EC = dumpPublicsStream())
    return EC;

  if (auto EC = dumpSectionHeaders())
    return EC;

  if (auto EC = dumpFpoStream())
    return EC;

  flush();

  return Error::success();
}

Error LLVMOutputStyle::dumpFileHeaders() {
  if (!opts::raw::DumpHeaders)
    return Error::success();

  DictScope D(P, "FileHeaders");
  P.printNumber("BlockSize", File.getBlockSize());
  P.printNumber("FreeBlockMap", File.getFreeBlockMapBlock());
  P.printNumber("NumBlocks", File.getBlockCount());
  P.printNumber("NumDirectoryBytes", File.getNumDirectoryBytes());
  P.printNumber("Unknown1", File.getUnknown1());
  P.printNumber("BlockMapAddr", File.getBlockMapIndex());
  P.printNumber("NumDirectoryBlocks", File.getNumDirectoryBlocks());

  // The directory is not contiguous.  Instead, the block map contains a
  // contiguous list of block numbers whose contents, when concatenated in
  // order, make up the directory.
  P.printList("DirectoryBlocks", File.getDirectoryBlockArray());
  P.printNumber("NumStreams", File.getNumStreams());
  return Error::success();
}

Error LLVMOutputStyle::dumpStreamSummary() {
  if (!opts::raw::DumpStreamSummary)
    return Error::success();

  // It's OK if we fail to load some of these streams, we still attempt to print
  // what we can.
  auto Dbi = File.getPDBDbiStream();
  auto Tpi = File.getPDBTpiStream();
  auto Ipi = File.getPDBIpiStream();
  auto Info = File.getPDBInfoStream();

  ListScope L(P, "Streams");
  uint32_t StreamCount = File.getNumStreams();
  std::unordered_map<uint16_t, const ModuleInfoEx *> ModStreams;
  std::unordered_map<uint16_t, std::string> NamedStreams;

  if (Dbi) {
    for (auto &ModI : Dbi->modules()) {
      uint16_t SN = ModI.Info.getModuleStreamIndex();
      ModStreams[SN] = &ModI;
    }
  }
  if (Info) {
    for (auto &NSE : Info->named_streams()) {
      NamedStreams[NSE.second] = NSE.first();
    }
  }

  for (uint16_t StreamIdx = 0; StreamIdx < StreamCount; ++StreamIdx) {
    std::string Label("Stream ");
    Label += to_string(StreamIdx);
    std::string Value;
    if (StreamIdx == OldMSFDirectory)
      Value = "Old MSF Directory";
    else if (StreamIdx == StreamPDB)
      Value = "PDB Stream";
    else if (StreamIdx == StreamDBI)
      Value = "DBI Stream";
    else if (StreamIdx == StreamTPI)
      Value = "TPI Stream";
    else if (StreamIdx == StreamIPI)
      Value = "IPI Stream";
    else if (Dbi && StreamIdx == Dbi->getGlobalSymbolStreamIndex())
      Value = "Global Symbol Hash";
    else if (Dbi && StreamIdx == Dbi->getPublicSymbolStreamIndex())
      Value = "Public Symbol Hash";
    else if (Dbi && StreamIdx == Dbi->getSymRecordStreamIndex())
      Value = "Public Symbol Records";
    else if (Tpi && StreamIdx == Tpi->getTypeHashStreamIndex())
      Value = "TPI Hash";
    else if (Tpi && StreamIdx == Tpi->getTypeHashStreamAuxIndex())
      Value = "TPI Aux Hash";
    else if (Ipi && StreamIdx == Ipi->getTypeHashStreamIndex())
      Value = "IPI Hash";
    else if (Ipi && StreamIdx == Ipi->getTypeHashStreamAuxIndex())
      Value = "IPI Aux Hash";
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Exception))
      Value = "Exception Data";
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Fixup))
      Value = "Fixup Data";
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::FPO))
      Value = "FPO Data";
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::NewFPO))
      Value = "New FPO Data";
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::OmapFromSrc))
      Value = "Omap From Source Data";
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::OmapToSrc))
      Value = "Omap To Source Data";
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Pdata))
      Value = "Pdata";
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::SectionHdr))
      Value = "Section Header Data";
    else if (Dbi &&
             StreamIdx ==
                 Dbi->getDebugStreamIndex(DbgHeaderType::SectionHdrOrig))
      Value = "Section Header Original Data";
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::TokenRidMap))
      Value = "Token Rid Data";
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Xdata))
      Value = "Xdata";
    else {
      auto ModIter = ModStreams.find(StreamIdx);
      auto NSIter = NamedStreams.find(StreamIdx);
      if (ModIter != ModStreams.end()) {
        Value = "Module \"";
        Value += ModIter->second->Info.getModuleName().str();
        Value += "\"";
      } else if (NSIter != NamedStreams.end()) {
        Value = "Named Stream \"";
        Value += NSIter->second;
        Value += "\"";
      } else {
        Value = "???";
      }
    }
    Value = "[" + Value + "]";
    Value =
        Value + " (" + to_string(File.getStreamByteSize(StreamIdx)) + " bytes)";

    P.printString(Label, Value);
  }

  // Consume errors from missing streams.
  if (!Dbi)
    consumeError(Dbi.takeError());
  if (!Tpi)
    consumeError(Tpi.takeError());
  if (!Ipi)
    consumeError(Ipi.takeError());
  if (!Info)
    consumeError(Info.takeError());

  P.flush();
  return Error::success();
}

Error LLVMOutputStyle::dumpFreePageMap() {
  if (!opts::raw::DumpPageStats)
    return Error::success();

  // Start with used pages instead of free pages because
  // the number of free pages is far larger than used pages.
  BitVector FPM = File.getMsfLayout().FreePageMap;

  PageStats PS(FPM);

  recordKnownUsedPage(PS, 0); // MSF Super Block

  uint32_t BlocksPerSection = msf::getFpmIntervalLength(File.getMsfLayout());
  uint32_t NumSections = msf::getNumFpmIntervals(File.getMsfLayout());
  for (uint32_t I = 0; I < NumSections; ++I) {
    uint32_t Fpm0 = 1 + BlocksPerSection * I;
    // 2 Fpm blocks spaced at `getBlockSize()` block intervals
    recordKnownUsedPage(PS, Fpm0);
    recordKnownUsedPage(PS, Fpm0 + 1);
  }

  recordKnownUsedPage(PS, File.getBlockMapIndex()); // Stream Table

  for (auto DB : File.getDirectoryBlockArray())
    recordKnownUsedPage(PS, DB);

  // Record pages used by streams. Note that pages for stream 0
  // are considered being unused because that's what MSVC tools do.
  // Stream 0 doesn't contain actual data, so it makes some sense,
  // though it's a bit confusing to us.
  for (auto &SE : File.getStreamMap().drop_front(1))
    for (auto &S : SE)
      recordKnownUsedPage(PS, S);

  dumpBitVector("Msf Free Pages", FPM);
  dumpBitVector("Orphaned Pages", PS.OrphanedPages);
  dumpBitVector("Multiply Used Pages", PS.MultiUsePages);
  dumpBitVector("Use After Free Pages", PS.UseAfterFreePages);
  return Error::success();
}

void LLVMOutputStyle::dumpBitVector(StringRef Name, const BitVector &V) {
  std::vector<uint32_t> Vec;
  for (uint32_t I = 0, E = V.size(); I != E; ++I)
    if (V[I])
      Vec.push_back(I);
  P.printList(Name, Vec);
}

Error LLVMOutputStyle::dumpStreamBlocks() {
  if (!opts::raw::DumpStreamBlocks)
    return Error::success();

  ListScope L(P, "StreamBlocks");
  uint32_t StreamCount = File.getNumStreams();
  for (uint32_t StreamIdx = 0; StreamIdx < StreamCount; ++StreamIdx) {
    std::string Name("Stream ");
    Name += to_string(StreamIdx);
    auto StreamBlocks = File.getStreamBlockList(StreamIdx);
    P.printList(Name, StreamBlocks);
  }
  return Error::success();
}

Error LLVMOutputStyle::dumpStreamData() {
  uint32_t StreamCount = File.getNumStreams();
  StringRef DumpStreamStr = opts::raw::DumpStreamDataIdx;
  uint32_t DumpStreamNum;
  if (DumpStreamStr.getAsInteger(/*Radix=*/0U, DumpStreamNum))
    return Error::success();

  if (DumpStreamNum >= StreamCount)
    return make_error<RawError>(raw_error_code::no_stream);

  auto S = MappedBlockStream::createIndexedStream(
      File.getMsfLayout(), File.getMsfBuffer(), DumpStreamNum);
  StreamReader R(*S);
  while (R.bytesRemaining() > 0) {
    ArrayRef<uint8_t> Data;
    uint32_t BytesToReadInBlock = std::min(
        R.bytesRemaining(), static_cast<uint32_t>(File.getBlockSize()));
    if (auto EC = R.readBytes(Data, BytesToReadInBlock))
      return EC;
    P.printBinaryBlock(
        "Data",
        StringRef(reinterpret_cast<const char *>(Data.begin()), Data.size()));
  }
  return Error::success();
}

Error LLVMOutputStyle::dumpInfoStream() {
  if (!opts::raw::DumpHeaders)
    return Error::success();
  auto IS = File.getPDBInfoStream();
  if (!IS)
    return IS.takeError();

  DictScope D(P, "PDB Stream");
  P.printNumber("Version", IS->getVersion());
  P.printHex("Signature", IS->getSignature());
  P.printNumber("Age", IS->getAge());
  P.printObject("Guid", IS->getGuid());
  return Error::success();
}

Error LLVMOutputStyle::dumpNamedStream() {
  if (opts::raw::DumpStreamDataName.empty())
    return Error::success();

  auto IS = File.getPDBInfoStream();
  if (!IS)
    return IS.takeError();

  uint32_t NameStreamIndex =
      IS->getNamedStreamIndex(opts::raw::DumpStreamDataName);
  if (NameStreamIndex == 0 || NameStreamIndex >= File.getNumStreams())
    return make_error<RawError>(raw_error_code::no_stream);

  if (NameStreamIndex != 0) {
    std::string Name("Stream '");
    Name += opts::raw::DumpStreamDataName;
    Name += "'";
    DictScope D(P, Name);
    P.printNumber("Index", NameStreamIndex);

    auto NameStream = MappedBlockStream::createIndexedStream(
        File.getMsfLayout(), File.getMsfBuffer(), NameStreamIndex);
    StreamReader Reader(*NameStream);

    NameHashTable NameTable;
    if (auto EC = NameTable.load(Reader))
      return EC;

    P.printHex("Signature", NameTable.getSignature());
    P.printNumber("Version", NameTable.getHashVersion());
    P.printNumber("Name Count", NameTable.getNameCount());
    ListScope L(P, "Names");
    for (uint32_t ID : NameTable.name_ids()) {
      StringRef Str = NameTable.getStringForID(ID);
      if (!Str.empty())
        P.printString(to_string(ID), Str);
    }
  }
  return Error::success();
}

static void printTypeIndexOffset(raw_ostream &OS,
                                 const TypeIndexOffset &TIOff) {
  OS << "{" << TIOff.Type.getIndex() << ", " << TIOff.Offset << "}";
}

static void dumpTpiHash(ScopedPrinter &P, TpiStream &Tpi) {
  if (!opts::raw::DumpTpiHash)
    return;
  DictScope DD(P, "Hash");
  P.printNumber("Number of Hash Buckets", Tpi.NumHashBuckets());
  P.printNumber("Hash Key Size", Tpi.getHashKeySize());
  P.printList("Values", Tpi.getHashValues());
  P.printList("Type Index Offsets", Tpi.getTypeIndexOffsets(),
              printTypeIndexOffset);
  P.printList("Hash Adjustments", Tpi.getHashAdjustments(),
              printTypeIndexOffset);
}

Error LLVMOutputStyle::dumpTpiStream(uint32_t StreamIdx) {
  assert(StreamIdx == StreamTPI || StreamIdx == StreamIPI);

  bool DumpRecordBytes = false;
  bool DumpRecords = false;
  StringRef Label;
  StringRef VerLabel;
  if (StreamIdx == StreamTPI) {
    DumpRecordBytes = opts::raw::DumpTpiRecordBytes;
    DumpRecords = opts::raw::DumpTpiRecords;
    Label = "Type Info Stream (TPI)";
    VerLabel = "TPI Version";
  } else if (StreamIdx == StreamIPI) {
    DumpRecordBytes = opts::raw::DumpIpiRecordBytes;
    DumpRecords = opts::raw::DumpIpiRecords;
    Label = "Type Info Stream (IPI)";
    VerLabel = "IPI Version";
  }
  if (!DumpRecordBytes && !DumpRecords && !opts::raw::DumpModuleSyms)
    return Error::success();

  auto Tpi = (StreamIdx == StreamTPI) ? File.getPDBTpiStream()
                                      : File.getPDBIpiStream();
  if (!Tpi)
    return Tpi.takeError();

  if (DumpRecords || DumpRecordBytes) {
    DictScope D(P, Label);

    P.printNumber(VerLabel, Tpi->getTpiVersion());
    P.printNumber("Record count", Tpi->NumTypeRecords());

    ListScope L(P, "Records");

    bool HadError = false;
    for (auto &Type : Tpi->types(&HadError)) {
      DictScope DD(P, "");

      if (DumpRecords) {
        if (auto EC = Dumper.dump(Type))
          return EC;
      }

      if (DumpRecordBytes)
        P.printBinaryBlock("Bytes", Type.Data);
    }
    dumpTpiHash(P, *Tpi);
    if (HadError)
      return make_error<RawError>(raw_error_code::corrupt_file,
                                  "TPI stream contained corrupt record");
  } else if (opts::raw::DumpModuleSyms) {
    // Even if the user doesn't want to dump type records, we still need to
    // iterate them in order to build the list of types so that we can print
    // them when dumping module symbols. So when they want to dump symbols
    // but not types, use a null output stream.
    ScopedPrinter *OldP = Dumper.getPrinter();
    Dumper.setPrinter(nullptr);

    bool HadError = false;
    for (auto &Type : Tpi->types(&HadError)) {
      if (auto EC = Dumper.dump(Type))
        return EC;
    }

    Dumper.setPrinter(OldP);
    dumpTpiHash(P, *Tpi);
    if (HadError)
      return make_error<RawError>(raw_error_code::corrupt_file,
                                  "TPI stream contained corrupt record");
  }
  P.flush();
  return Error::success();
}

Error LLVMOutputStyle::dumpDbiStream() {
  bool DumpModules = opts::raw::DumpModules || opts::raw::DumpModuleSyms ||
                     opts::raw::DumpModuleFiles || opts::raw::DumpLineInfo;
  if (!opts::raw::DumpHeaders && !DumpModules)
    return Error::success();

  auto DS = File.getPDBDbiStream();
  if (!DS)
    return DS.takeError();

  DictScope D(P, "DBI Stream");
  P.printNumber("Dbi Version", DS->getDbiVersion());
  P.printNumber("Age", DS->getAge());
  P.printBoolean("Incremental Linking", DS->isIncrementallyLinked());
  P.printBoolean("Has CTypes", DS->hasCTypes());
  P.printBoolean("Is Stripped", DS->isStripped());
  P.printObject("Machine Type", DS->getMachineType());
  P.printNumber("Symbol Record Stream Index", DS->getSymRecordStreamIndex());
  P.printNumber("Public Symbol Stream Index", DS->getPublicSymbolStreamIndex());
  P.printNumber("Global Symbol Stream Index", DS->getGlobalSymbolStreamIndex());

  uint16_t Major = DS->getBuildMajorVersion();
  uint16_t Minor = DS->getBuildMinorVersion();
  P.printVersion("Toolchain Version", Major, Minor);

  std::string DllName;
  raw_string_ostream DllStream(DllName);
  DllStream << "mspdb" << Major << Minor << ".dll version";
  DllStream.flush();
  P.printVersion(DllName, Major, Minor, DS->getPdbDllVersion());

  if (DumpModules) {
    ListScope L(P, "Modules");
    for (auto &Modi : DS->modules()) {
      DictScope DD(P);
      P.printString("Name", Modi.Info.getModuleName().str());
      P.printNumber("Debug Stream Index", Modi.Info.getModuleStreamIndex());
      P.printString("Object File Name", Modi.Info.getObjFileName().str());
      P.printNumber("Num Files", Modi.Info.getNumberOfFiles());
      P.printNumber("Source File Name Idx", Modi.Info.getSourceFileNameIndex());
      P.printNumber("Pdb File Name Idx", Modi.Info.getPdbFilePathNameIndex());
      P.printNumber("Line Info Byte Size", Modi.Info.getLineInfoByteSize());
      P.printNumber("C13 Line Info Byte Size",
                    Modi.Info.getC13LineInfoByteSize());
      P.printNumber("Symbol Byte Size", Modi.Info.getSymbolDebugInfoByteSize());
      P.printNumber("Type Server Index", Modi.Info.getTypeServerIndex());
      P.printBoolean("Has EC Info", Modi.Info.hasECInfo());
      if (opts::raw::DumpModuleFiles) {
        std::string FileListName =
            to_string(Modi.SourceFiles.size()) + " Contributing Source Files";
        ListScope LL(P, FileListName);
        for (auto File : Modi.SourceFiles)
          P.printString(File.str());
      }
      bool HasModuleDI =
          (Modi.Info.getModuleStreamIndex() < File.getNumStreams());
      bool ShouldDumpSymbols =
          (opts::raw::DumpModuleSyms || opts::raw::DumpSymRecordBytes);
      if (HasModuleDI && (ShouldDumpSymbols || opts::raw::DumpLineInfo)) {
        auto ModStreamData = MappedBlockStream::createIndexedStream(
            File.getMsfLayout(), File.getMsfBuffer(),
            Modi.Info.getModuleStreamIndex());

        ModStream ModS(Modi.Info, std::move(ModStreamData));
        if (auto EC = ModS.reload())
          return EC;

        if (ShouldDumpSymbols) {
          ListScope SS(P, "Symbols");
          codeview::CVSymbolDumper SD(P, Dumper, nullptr, false);
          bool HadError = false;
          for (const auto &S : ModS.symbols(&HadError)) {
            DictScope DD(P, "");

            if (opts::raw::DumpModuleSyms)
              SD.dump(S);
            if (opts::raw::DumpSymRecordBytes)
              P.printBinaryBlock("Bytes", S.Data);
          }
          if (HadError)
            return make_error<RawError>(
                raw_error_code::corrupt_file,
                "DBI stream contained corrupt symbol record");
        }
        if (opts::raw::DumpLineInfo) {
          ListScope SS(P, "LineInfo");
          bool HadError = false;
          // Define a locally scoped visitor to print the different
          // substream types types.
          class RecordVisitor : public codeview::IModuleSubstreamVisitor {
          public:
            RecordVisitor(ScopedPrinter &P, PDBFile &F) : P(P), F(F) {}
            Error visitUnknown(ModuleSubstreamKind Kind,
                               ReadableStreamRef Stream) override {
              DictScope DD(P, "Unknown");
              ArrayRef<uint8_t> Data;
              StreamReader R(Stream);
              if (auto EC = R.readBytes(Data, R.bytesRemaining())) {
                return make_error<RawError>(
                    raw_error_code::corrupt_file,
                    "DBI stream contained corrupt line info record");
              }
              P.printBinaryBlock("Data", Data);
              return Error::success();
            }
            Error
            visitFileChecksums(ReadableStreamRef Data,
                               const FileChecksumArray &Checksums) override {
              DictScope DD(P, "FileChecksums");
              for (const auto &C : Checksums) {
                DictScope DDD(P, "Checksum");
                if (auto Result = getFileNameForOffset(C.FileNameOffset))
                  P.printString("FileName", Result.get());
                else
                  return Result.takeError();
                P.flush();
                P.printEnum("Kind", uint8_t(C.Kind), getFileChecksumNames());
                P.printBinaryBlock("Checksum", C.Checksum);
              }
              return Error::success();
            }

            Error visitLines(ReadableStreamRef Data,
                             const LineSubstreamHeader *Header,
                             const LineInfoArray &Lines) override {
              DictScope DD(P, "Lines");
              for (const auto &L : Lines) {
                if (auto Result = getFileNameForOffset2(L.NameIndex))
                  P.printString("FileName", Result.get());
                else
                  return Result.takeError();
                P.flush();
                for (const auto &N : L.LineNumbers) {
                  DictScope DDD(P, "Line");
                  LineInfo LI(N.Flags);
                  P.printNumber("Offset", N.Offset);
                  if (LI.isAlwaysStepInto())
                    P.printString("StepInto", StringRef("Always"));
                  else if (LI.isNeverStepInto())
                    P.printString("StepInto", StringRef("Never"));
                  else
                    P.printNumber("LineNumberStart", LI.getStartLine());
                  P.printNumber("EndDelta", LI.getLineDelta());
                  P.printBoolean("IsStatement", LI.isStatement());
                }
                for (const auto &C : L.Columns) {
                  DictScope DDD(P, "Column");
                  P.printNumber("Start", C.StartColumn);
                  P.printNumber("End", C.EndColumn);
                }
              }
              return Error::success();
            }

          private:
            Expected<StringRef> getFileNameForOffset(uint32_t Offset) {
              auto ST = F.getStringTable();
              if (!ST)
                return ST.takeError();

              return ST->getStringForID(Offset);
            }
            Expected<StringRef> getFileNameForOffset2(uint32_t Offset) {
              auto DS = F.getPDBDbiStream();
              if (!DS)
                return DS.takeError();
              return DS->getFileNameForIndex(Offset);
            }
            ScopedPrinter &P;
            PDBFile &F;
          };

          RecordVisitor V(P, File);
          for (const auto &L : ModS.lines(&HadError)) {
            if (auto EC = codeview::visitModuleSubstream(L, V))
              return EC;
          }
        }
      }
    }
  }
  return Error::success();
}

Error LLVMOutputStyle::dumpSectionContribs() {
  if (!opts::raw::DumpSectionContribs)
    return Error::success();

  auto Dbi = File.getPDBDbiStream();
  if (!Dbi)
    return Dbi.takeError();

  ListScope L(P, "Section Contributions");
  class Visitor : public ISectionContribVisitor {
  public:
    Visitor(ScopedPrinter &P, DbiStream &DS) : P(P), DS(DS) {}
    void visit(const SectionContrib &SC) override {
      DictScope D(P, "Contribution");
      P.printNumber("ISect", SC.ISect);
      P.printNumber("Off", SC.Off);
      P.printNumber("Size", SC.Size);
      P.printFlags("Characteristics", SC.Characteristics,
                   codeview::getImageSectionCharacteristicNames(),
                   COFF::SectionCharacteristics(0x00F00000));
      {
        DictScope DD(P, "Module");
        P.printNumber("Index", SC.Imod);
        auto M = DS.modules();
        if (M.size() > SC.Imod) {
          P.printString("Name", M[SC.Imod].Info.getModuleName());
        }
      }
      P.printNumber("Data CRC", SC.DataCrc);
      P.printNumber("Reloc CRC", SC.RelocCrc);
      P.flush();
    }
    void visit(const SectionContrib2 &SC) override {
      visit(SC.Base);
      P.printNumber("ISect Coff", SC.ISectCoff);
      P.flush();
    }

  private:
    ScopedPrinter &P;
    DbiStream &DS;
  };
  Visitor V(P, *Dbi);
  Dbi->visitSectionContributions(V);
  return Error::success();
}

Error LLVMOutputStyle::dumpSectionMap() {
  if (!opts::raw::DumpSectionMap)
    return Error::success();

  auto Dbi = File.getPDBDbiStream();
  if (!Dbi)
    return Dbi.takeError();

  ListScope L(P, "Section Map");
  for (auto &M : Dbi->getSectionMap()) {
    DictScope D(P, "Entry");
    P.printFlags("Flags", M.Flags, getOMFSegMapDescFlagNames());
    P.printNumber("Flags", M.Flags);
    P.printNumber("Ovl", M.Ovl);
    P.printNumber("Group", M.Group);
    P.printNumber("Frame", M.Frame);
    P.printNumber("SecName", M.SecName);
    P.printNumber("ClassName", M.ClassName);
    P.printNumber("Offset", M.Offset);
    P.printNumber("SecByteLength", M.SecByteLength);
    P.flush();
  }
  return Error::success();
}

Error LLVMOutputStyle::dumpPublicsStream() {
  if (!opts::raw::DumpPublics)
    return Error::success();

  DictScope D(P, "Publics Stream");
  auto Publics = File.getPDBPublicsStream();
  if (!Publics)
    return Publics.takeError();

  auto Dbi = File.getPDBDbiStream();
  if (!Dbi)
    return Dbi.takeError();

  P.printNumber("Stream number", Dbi->getPublicSymbolStreamIndex());
  P.printNumber("SymHash", Publics->getSymHash());
  P.printNumber("AddrMap", Publics->getAddrMap());
  P.printNumber("Number of buckets", Publics->getNumBuckets());
  P.printList("Hash Buckets", Publics->getHashBuckets());
  P.printList("Address Map", Publics->getAddressMap());
  P.printList("Thunk Map", Publics->getThunkMap());
  P.printList("Section Offsets", Publics->getSectionOffsets(),
              printSectionOffset);
  ListScope L(P, "Symbols");
  codeview::CVSymbolDumper SD(P, Dumper, nullptr, false);
  bool HadError = false;
  for (auto S : Publics->getSymbols(&HadError)) {
    DictScope DD(P, "");

    SD.dump(S);
    if (opts::raw::DumpSymRecordBytes)
      P.printBinaryBlock("Bytes", S.Data);
  }
  if (HadError)
    return make_error<RawError>(
        raw_error_code::corrupt_file,
        "Public symbol stream contained corrupt record");

  return Error::success();
}

Error LLVMOutputStyle::dumpSectionHeaders() {
  if (!opts::raw::DumpSectionHeaders)
    return Error::success();

  auto Dbi = File.getPDBDbiStream();
  if (!Dbi)
    return Dbi.takeError();

  ListScope D(P, "Section Headers");
  for (const object::coff_section &Section : Dbi->getSectionHeaders()) {
    DictScope DD(P, "");

    // If a name is 8 characters long, there is no NUL character at end.
    StringRef Name(Section.Name, strnlen(Section.Name, sizeof(Section.Name)));
    P.printString("Name", Name);
    P.printNumber("Virtual Size", Section.VirtualSize);
    P.printNumber("Virtual Address", Section.VirtualAddress);
    P.printNumber("Size of Raw Data", Section.SizeOfRawData);
    P.printNumber("File Pointer to Raw Data", Section.PointerToRawData);
    P.printNumber("File Pointer to Relocations", Section.PointerToRelocations);
    P.printNumber("File Pointer to Linenumbers", Section.PointerToLinenumbers);
    P.printNumber("Number of Relocations", Section.NumberOfRelocations);
    P.printNumber("Number of Linenumbers", Section.NumberOfLinenumbers);
    P.printFlags("Characteristics", Section.Characteristics,
                 getImageSectionCharacteristicNames());
  }
  return Error::success();
}

Error LLVMOutputStyle::dumpFpoStream() {
  if (!opts::raw::DumpFpo)
    return Error::success();

  auto Dbi = File.getPDBDbiStream();
  if (!Dbi)
    return Dbi.takeError();

  ListScope D(P, "New FPO");
  for (const object::FpoData &Fpo : Dbi->getFpoRecords()) {
    DictScope DD(P, "");
    P.printNumber("Offset", Fpo.Offset);
    P.printNumber("Size", Fpo.Size);
    P.printNumber("Number of locals", Fpo.NumLocals);
    P.printNumber("Number of params", Fpo.NumParams);
    P.printNumber("Size of Prolog", Fpo.getPrologSize());
    P.printNumber("Number of Saved Registers", Fpo.getNumSavedRegs());
    P.printBoolean("Has SEH", Fpo.hasSEH());
    P.printBoolean("Use BP", Fpo.useBP());
    P.printNumber("Frame Pointer", Fpo.getFP());
  }
  return Error::success();
}

void LLVMOutputStyle::flush() { P.flush(); }
