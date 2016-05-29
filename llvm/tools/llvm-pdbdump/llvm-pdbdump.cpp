//===- llvm-pdbdump.cpp - Dump debug info from a PDB file -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Dumps debug information present in PDB files.  This utility makes use of
// the Microsoft Windows SDK, so will not compile or run on non-Windows
// platforms.
//
//===----------------------------------------------------------------------===//

#include "llvm-pdbdump.h"
#include "CompilandDumper.h"
#include "ExternalSymbolDumper.h"
#include "FunctionDumper.h"
#include "LinePrinter.h"
#include "TypeDumper.h"
#include "VariableDumper.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/config.h"
#include "llvm/DebugInfo/CodeView/StreamReader.h"
#include "llvm/DebugInfo/CodeView/SymbolDumper.h"
#include "llvm/DebugInfo/CodeView/TypeDumper.h"
#include "llvm/DebugInfo/PDB/GenericError.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBRawSymbol.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolThunk.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/ModInfo.h"
#include "llvm/DebugInfo/PDB/Raw/ModStream.h"
#include "llvm/DebugInfo/PDB/Raw/NameHashTable.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/PublicsStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/RawSession.h"
#include "llvm/DebugInfo/PDB/Raw/TpiStream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#if defined(HAVE_DIA_SDK)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

using namespace llvm;
using namespace llvm::pdb;

namespace opts {

enum class PDB_DumpType { ByType, ByObjFile, Both };

cl::list<std::string> InputFilenames(cl::Positional,
                                     cl::desc("<input PDB files>"),
                                     cl::OneOrMore);

cl::OptionCategory TypeCategory("Symbol Type Options");
cl::OptionCategory FilterCategory("Filtering Options");
cl::OptionCategory OtherOptions("Other Options");
cl::OptionCategory NativeOptions("Native Options");

cl::opt<bool> Compilands("compilands", cl::desc("Display compilands"),
                         cl::cat(TypeCategory));
cl::opt<bool> Symbols("symbols", cl::desc("Display symbols for each compiland"),
                      cl::cat(TypeCategory));
cl::opt<bool> Globals("globals", cl::desc("Dump global symbols"),
                      cl::cat(TypeCategory));
cl::opt<bool> Externals("externals", cl::desc("Dump external symbols"),
                        cl::cat(TypeCategory));
cl::opt<bool> Types("types", cl::desc("Display types"), cl::cat(TypeCategory));
cl::opt<bool> Lines("lines", cl::desc("Line tables"), cl::cat(TypeCategory));
cl::opt<bool>
    All("all", cl::desc("Implies all other options in 'Symbol Types' category"),
        cl::cat(TypeCategory));

cl::opt<uint64_t> LoadAddress(
    "load-address",
    cl::desc("Assume the module is loaded at the specified address"),
    cl::cat(OtherOptions));

cl::opt<bool> DumpHeaders("raw-headers", cl::desc("dump PDB headers"),
                          cl::cat(NativeOptions));
cl::opt<bool> DumpStreamBlocks("raw-stream-blocks",
                               cl::desc("dump PDB stream blocks"),
                               cl::cat(NativeOptions));
cl::opt<bool> DumpStreamSummary("raw-stream-summary",
                                cl::desc("dump summary of the PDB streams"),
                                cl::cat(NativeOptions));
cl::opt<bool>
    DumpTpiRecords("raw-tpi-records",
                   cl::desc("dump CodeView type records from TPI stream"),
                   cl::cat(NativeOptions));
cl::opt<bool> DumpTpiRecordBytes(
    "raw-tpi-record-bytes",
    cl::desc("dump CodeView type record raw bytes from TPI stream"),
    cl::cat(NativeOptions));
cl::opt<bool>
    DumpIpiRecords("raw-ipi-records",
                   cl::desc("dump CodeView type records from IPI stream"),
                   cl::cat(NativeOptions));
cl::opt<bool> DumpIpiRecordBytes(
    "raw-ipi-record-bytes",
    cl::desc("dump CodeView type record raw bytes from IPI stream"),
    cl::cat(NativeOptions));
cl::opt<std::string> DumpStreamDataIdx("raw-stream",
                                       cl::desc("dump stream data"),
                                       cl::cat(NativeOptions));
cl::opt<std::string> DumpStreamDataName("raw-stream-name",
                                        cl::desc("dump stream data"),
                                        cl::cat(NativeOptions));
cl::opt<bool> DumpModules("raw-modules", cl::desc("dump compiland information"),
                          cl::cat(NativeOptions));
cl::opt<bool> DumpModuleFiles("raw-module-files",
                              cl::desc("dump file information"),
                              cl::cat(NativeOptions));
cl::opt<bool> DumpModuleSyms("raw-module-syms", cl::desc("dump module symbols"),
                             cl::cat(NativeOptions));
cl::opt<bool> DumpPublics("raw-publics", cl::desc("dump Publics stream data"),
                          cl::cat(NativeOptions));
cl::opt<bool>
    DumpSymRecordBytes("raw-sym-record-bytes",
                       cl::desc("dump CodeView symbol record raw bytes"),
                       cl::cat(NativeOptions));
cl::opt<bool>
    RawAll("raw-all",
           cl::desc("Implies most other options in 'Native Options' category"),
           cl::cat(NativeOptions));

cl::list<std::string>
    ExcludeTypes("exclude-types",
                 cl::desc("Exclude types by regular expression"),
                 cl::ZeroOrMore, cl::cat(FilterCategory));
cl::list<std::string>
    ExcludeSymbols("exclude-symbols",
                   cl::desc("Exclude symbols by regular expression"),
                   cl::ZeroOrMore, cl::cat(FilterCategory));
cl::list<std::string>
    ExcludeCompilands("exclude-compilands",
                      cl::desc("Exclude compilands by regular expression"),
                      cl::ZeroOrMore, cl::cat(FilterCategory));

cl::list<std::string> IncludeTypes(
    "include-types",
    cl::desc("Include only types which match a regular expression"),
    cl::ZeroOrMore, cl::cat(FilterCategory));
cl::list<std::string> IncludeSymbols(
    "include-symbols",
    cl::desc("Include only symbols which match a regular expression"),
    cl::ZeroOrMore, cl::cat(FilterCategory));
cl::list<std::string> IncludeCompilands(
    "include-compilands",
    cl::desc("Include only compilands those which match a regular expression"),
    cl::ZeroOrMore, cl::cat(FilterCategory));

cl::opt<bool> ExcludeCompilerGenerated(
    "no-compiler-generated",
    cl::desc("Don't show compiler generated types and symbols"),
    cl::cat(FilterCategory));
cl::opt<bool>
    ExcludeSystemLibraries("no-system-libs",
                           cl::desc("Don't show symbols from system libraries"),
                           cl::cat(FilterCategory));
cl::opt<bool> NoClassDefs("no-class-definitions",
                          cl::desc("Don't display full class definitions"),
                          cl::cat(FilterCategory));
cl::opt<bool> NoEnumDefs("no-enum-definitions",
                         cl::desc("Don't display full enum definitions"),
                         cl::cat(FilterCategory));
}

static ExitOnError ExitOnErr;

static Error dumpFileHeaders(ScopedPrinter &P, PDBFile &File) {
  if (!opts::DumpHeaders)
    return Error::success();

  DictScope D(P, "FileHeaders");
  P.printNumber("BlockSize", File.getBlockSize());
  P.printNumber("Unknown0", File.getUnknown0());
  P.printNumber("NumBlocks", File.getBlockCount());
  P.printNumber("NumDirectoryBytes", File.getNumDirectoryBytes());
  P.printNumber("Unknown1", File.getUnknown1());
  P.printNumber("BlockMapAddr", File.getBlockMapIndex());
  P.printNumber("NumDirectoryBlocks", File.getNumDirectoryBlocks());
  P.printNumber("BlockMapOffset", File.getBlockMapOffset());

  // The directory is not contiguous.  Instead, the block map contains a
  // contiguous list of block numbers whose contents, when concatenated in
  // order, make up the directory.
  P.printList("DirectoryBlocks", File.getDirectoryBlockArray());
  P.printNumber("NumStreams", File.getNumStreams());
  return Error::success();
}

static Error dumpStreamSummary(ScopedPrinter &P, PDBFile &File) {
  if (!opts::DumpStreamSummary)
    return Error::success();

  auto DbiS = File.getPDBDbiStream();
  if (auto EC = DbiS.takeError())
    return EC;
  auto TpiS = File.getPDBTpiStream();
  if (auto EC = TpiS.takeError())
    return EC;
  auto IpiS = File.getPDBIpiStream();
  if (auto EC = IpiS.takeError())
    return EC;
  auto InfoS = File.getPDBInfoStream();
  if (auto EC = InfoS.takeError())
    return EC;
  DbiStream &DS = DbiS.get();
  TpiStream &TS = TpiS.get();
  TpiStream &TIS = IpiS.get();
  InfoStream &IS = InfoS.get();

  ListScope L(P, "Streams");
  uint32_t StreamCount = File.getNumStreams();
  std::unordered_map<uint16_t, const ModuleInfoEx *> ModStreams;
  std::unordered_map<uint16_t, std::string> NamedStreams;

  for (auto &ModI : DS.modules()) {
    uint16_t SN = ModI.Info.getModuleStreamIndex();
    ModStreams[SN] = &ModI;
  }
  for (auto &NSE : IS.named_streams()) {
    NamedStreams[NSE.second] = NSE.first();
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
    else if (StreamIdx == DS.getGlobalSymbolStreamIndex())
      Value = "Global Symbol Hash";
    else if (StreamIdx == DS.getPublicSymbolStreamIndex())
      Value = "Public Symbol Hash";
    else if (StreamIdx == DS.getSymRecordStreamIndex())
      Value = "Public Symbol Records";
    else if (StreamIdx == TS.getTypeHashStreamIndex())
      Value = "TPI Hash";
    else if (StreamIdx == TS.getTypeHashStreamAuxIndex())
      Value = "TPI Aux Hash";
    else if (StreamIdx == TIS.getTypeHashStreamIndex())
      Value = "IPI Hash";
    else if (StreamIdx == TIS.getTypeHashStreamAuxIndex())
      Value = "IPI Aux Hash";
    else if (StreamIdx == DS.getDebugStreamIndex(DbgHeaderType::Exception))
      Value = "Exception Data";
    else if (StreamIdx == DS.getDebugStreamIndex(DbgHeaderType::Fixup))
      Value = "Fixup Data";
    else if (StreamIdx == DS.getDebugStreamIndex(DbgHeaderType::FPO))
      Value = "FPO Data";
    else if (StreamIdx == DS.getDebugStreamIndex(DbgHeaderType::NewFPO))
      Value = "New FPO Data";
    else if (StreamIdx == DS.getDebugStreamIndex(DbgHeaderType::OmapFromSrc))
      Value = "Omap From Source Data";
    else if (StreamIdx == DS.getDebugStreamIndex(DbgHeaderType::OmapToSrc))
      Value = "Omap To Source Data";
    else if (StreamIdx == DS.getDebugStreamIndex(DbgHeaderType::Pdata))
      Value = "Pdata";
    else if (StreamIdx == DS.getDebugStreamIndex(DbgHeaderType::SectionHdr))
      Value = "Section Header Data";
    else if (StreamIdx == DS.getDebugStreamIndex(DbgHeaderType::SectionHdrOrig))
      Value = "Section Header Original Data";
    else if (StreamIdx == DS.getDebugStreamIndex(DbgHeaderType::TokenRidMap))
      Value = "Token Rid Data";
    else if (StreamIdx == DS.getDebugStreamIndex(DbgHeaderType::Xdata))
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
  P.flush();
  return Error::success();
}

static Error dumpStreamBlocks(ScopedPrinter &P, PDBFile &File) {
  if (!opts::DumpStreamBlocks)
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

static Error dumpStreamData(ScopedPrinter &P, PDBFile &File) {
  uint32_t StreamCount = File.getNumStreams();
  StringRef DumpStreamStr = opts::DumpStreamDataIdx;
  uint32_t DumpStreamNum;
  if (DumpStreamStr.getAsInteger(/*Radix=*/0U, DumpStreamNum) ||
      DumpStreamNum >= StreamCount)
    return Error::success();

  MappedBlockStream S(DumpStreamNum, File);
  codeview::StreamReader R(S);
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

static Error dumpInfoStream(ScopedPrinter &P, PDBFile &File) {
  if (!opts::DumpHeaders)
    return Error::success();
  auto InfoS = File.getPDBInfoStream();
  if (auto EC = InfoS.takeError())
    return EC;

  InfoStream &IS = InfoS.get();

  DictScope D(P, "PDB Stream");
  P.printNumber("Version", IS.getVersion());
  P.printHex("Signature", IS.getSignature());
  P.printNumber("Age", IS.getAge());
  P.printObject("Guid", IS.getGuid());
  return Error::success();
}

static Error dumpNamedStream(ScopedPrinter &P, PDBFile &File) {
  if (opts::DumpStreamDataName.empty())
    return Error::success();

  auto InfoS = File.getPDBInfoStream();
  if (auto EC = InfoS.takeError())
    return EC;
  InfoStream &IS = InfoS.get();

  uint32_t NameStreamIndex = IS.getNamedStreamIndex(opts::DumpStreamDataName);

  if (NameStreamIndex != 0) {
    std::string Name("Stream '");
    Name += opts::DumpStreamDataName;
    Name += "'";
    DictScope D(P, Name);
    P.printNumber("Index", NameStreamIndex);

    MappedBlockStream NameStream(NameStreamIndex, File);
    codeview::StreamReader Reader(NameStream);

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
        P.printString(Str);
    }
  }
  return Error::success();
}

static Error dumpDbiStream(ScopedPrinter &P, PDBFile &File,
                           codeview::CVTypeDumper &TD) {
  bool DumpModules =
      opts::DumpModules || opts::DumpModuleSyms || opts::DumpModuleFiles;
  if (!opts::DumpHeaders && !DumpModules)
    return Error::success();

  auto DbiS = File.getPDBDbiStream();
  if (auto EC = DbiS.takeError())
    return EC;
  DbiStream &DS = DbiS.get();

  DictScope D(P, "DBI Stream");
  P.printNumber("Dbi Version", DS.getDbiVersion());
  P.printNumber("Age", DS.getAge());
  P.printBoolean("Incremental Linking", DS.isIncrementallyLinked());
  P.printBoolean("Has CTypes", DS.hasCTypes());
  P.printBoolean("Is Stripped", DS.isStripped());
  P.printObject("Machine Type", DS.getMachineType());
  P.printNumber("Symbol Record Stream Index", DS.getSymRecordStreamIndex());
  P.printNumber("Public Symbol Stream Index", DS.getPublicSymbolStreamIndex());
  P.printNumber("Global Symbol Stream Index", DS.getGlobalSymbolStreamIndex());

  uint16_t Major = DS.getBuildMajorVersion();
  uint16_t Minor = DS.getBuildMinorVersion();
  P.printVersion("Toolchain Version", Major, Minor);

  std::string DllName;
  raw_string_ostream DllStream(DllName);
  DllStream << "mspdb" << Major << Minor << ".dll version";
  DllStream.flush();
  P.printVersion(DllName, Major, Minor, DS.getPdbDllVersion());

  if (DumpModules) {
    ListScope L(P, "Modules");
    for (auto &Modi : DS.modules()) {
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
      if (opts::DumpModuleFiles) {
        std::string FileListName =
            to_string(Modi.SourceFiles.size()) + " Contributing Source Files";
        ListScope LL(P, FileListName);
        for (auto File : Modi.SourceFiles)
          P.printString(File.str());
      }
      bool HasModuleDI =
          (Modi.Info.getModuleStreamIndex() < File.getNumStreams());
      bool ShouldDumpSymbols =
          (opts::DumpModuleSyms || opts::DumpSymRecordBytes);
      if (HasModuleDI && ShouldDumpSymbols) {
        ListScope SS(P, "Symbols");
        ModStream ModS(File, Modi.Info);
        if (auto EC = ModS.reload())
          return EC;

        codeview::CVSymbolDumper SD(P, TD, nullptr, false);
        bool HadError = false;
        for (auto &S : ModS.symbols(&HadError)) {
          DictScope DD(P, "");

          if (opts::DumpModuleSyms)
            SD.dump(S);
          if (opts::DumpSymRecordBytes)
            P.printBinaryBlock("Bytes", S.Data);
        }
        if (HadError)
          return make_error<RawError>(raw_error_code::corrupt_file,
                                      "DBI stream contained corrupt record");
      }
    }
  }
  return Error::success();
}

static Error dumpTpiStream(ScopedPrinter &P, PDBFile &File,
                           codeview::CVTypeDumper &TD, uint32_t StreamIdx) {
  assert(StreamIdx == StreamTPI || StreamIdx == StreamIPI);

  bool DumpRecordBytes = false;
  bool DumpRecords = false;
  StringRef Label;
  StringRef VerLabel;
  if (StreamIdx == StreamTPI) {
    DumpRecordBytes = opts::DumpTpiRecordBytes;
    DumpRecords = opts::DumpTpiRecords;
    Label = "Type Info Stream (TPI)";
    VerLabel = "TPI Version";
  } else if (StreamIdx == StreamIPI) {
    DumpRecordBytes = opts::DumpIpiRecordBytes;
    DumpRecords = opts::DumpIpiRecords;
    Label = "Type Info Stream (IPI)";
    VerLabel = "IPI Version";
  }
  if (!DumpRecordBytes && !DumpRecords && !opts::DumpModuleSyms)
    return Error::success();

  auto TpiS = (StreamIdx == StreamTPI) ? File.getPDBTpiStream()
                                       : File.getPDBIpiStream();
    if (auto EC = TpiS.takeError())
      return EC;
    TpiStream &Tpi = TpiS.get();

    if (DumpRecords || DumpRecordBytes) {
      DictScope D(P, Label);

      P.printNumber(VerLabel, Tpi.getTpiVersion());
    P.printNumber("Record count", Tpi.NumTypeRecords());

    ListScope L(P, "Records");

    bool HadError = false;
    for (auto &Type : Tpi.types(&HadError)) {
      DictScope DD(P, "");

      if (DumpRecords)
        TD.dump(Type);

      if (DumpRecordBytes)
        P.printBinaryBlock("Bytes", Type.Data);
    }
    if (HadError)
      return make_error<RawError>(raw_error_code::corrupt_file,
                                  "TPI stream contained corrupt record");
  } else if (opts::DumpModuleSyms) {
    // Even if the user doesn't want to dump type records, we still need to
    // iterate them in order to build the list of types so that we can print
    // them when dumping module symbols. So when they want to dump symbols
    // but not types, use a null output stream.
    ScopedPrinter *OldP = TD.getPrinter();
    TD.setPrinter(nullptr);

    bool HadError = false;
    for (auto &Type : Tpi.types(&HadError))
      TD.dump(Type);

    TD.setPrinter(OldP);
    if (HadError)
      return make_error<RawError>(raw_error_code::corrupt_file,
                                  "TPI stream contained corrupt record");
  }
  P.flush();
  return Error::success();
}

static void printSectionOffset(llvm::raw_ostream &OS,
                               const SectionOffset &Off) {
  OS << Off.Off << ", " << Off.Isect;
}

static Error dumpPublicsStream(ScopedPrinter &P, PDBFile &File,
                               codeview::CVTypeDumper &TD) {
  if (!opts::DumpPublics)
    return Error::success();

  DictScope D(P, "Publics Stream");
  auto PublicsS = File.getPDBPublicsStream();
  if (auto EC = PublicsS.takeError())
    return EC;
  PublicsStream &Publics = PublicsS.get();
  P.printNumber("Stream number", Publics.getStreamNum());
  P.printNumber("SymHash", Publics.getSymHash());
  P.printNumber("AddrMap", Publics.getAddrMap());
  P.printNumber("Number of buckets", Publics.getNumBuckets());
  P.printList("Hash Buckets", Publics.getHashBuckets());
  P.printList("Address Map", Publics.getAddressMap());
  P.printList("Thunk Map", Publics.getThunkMap());
  P.printList("Section Offsets", Publics.getSectionOffsets(),
              printSectionOffset);
  ListScope L(P, "Symbols");
  codeview::CVSymbolDumper SD(P, TD, nullptr, false);
  bool HadError = false;
  for (auto S : Publics.getSymbols(&HadError)) {
    DictScope DD(P, "");

    SD.dump(S);
    if (opts::DumpSymRecordBytes)
      P.printBinaryBlock("Bytes", S.Data);
  }
  if (HadError)
    return make_error<RawError>(
        raw_error_code::corrupt_file,
        "Public symbol stream contained corrupt record");

  return Error::success();
}

static Error dumpStructure(RawSession &RS) {
  PDBFile &File = RS.getPDBFile();
  ScopedPrinter P(outs());

  if (auto EC = dumpFileHeaders(P, File))
    return EC;

  if (auto EC = dumpStreamSummary(P, File))
    return EC;

  if (auto EC = dumpStreamBlocks(P, File))
    return EC;

  if (auto EC = dumpStreamData(P, File))
    return EC;

  if (auto EC = dumpInfoStream(P, File))
    return EC;

  if (auto EC = dumpNamedStream(P, File))
    return EC;

  codeview::CVTypeDumper TD(P, false);
  if (auto EC = dumpTpiStream(P, File, TD, StreamTPI))
    return EC;
  if (auto EC = dumpTpiStream(P, File, TD, StreamIPI))
    return EC;

  if (auto EC = dumpDbiStream(P, File, TD))
    return EC;

  if (auto EC = dumpPublicsStream(P, File, TD))
    return EC;
  return Error::success();
}

bool isRawDumpEnabled() {
  if (opts::DumpHeaders)
    return true;
  if (opts::DumpModules)
    return true;
  if (opts::DumpModuleFiles)
    return true;
  if (opts::DumpModuleSyms)
    return true;
  if (!opts::DumpStreamDataIdx.empty())
    return true;
  if (!opts::DumpStreamDataName.empty())
    return true;
  if (opts::DumpPublics)
    return true;
  if (opts::DumpStreamSummary)
    return true;
  if (opts::DumpStreamBlocks)
    return true;
  if (opts::DumpSymRecordBytes)
    return true;
  if (opts::DumpTpiRecordBytes)
    return true;
  if (opts::DumpTpiRecords)
    return true;
  if (opts::DumpIpiRecords)
    return true;
  if (opts::DumpIpiRecordBytes)
    return true;
  return false;
}

static void dumpInput(StringRef Path) {
  std::unique_ptr<IPDBSession> Session;
  if (isRawDumpEnabled()) {
    ExitOnErr(loadDataForPDB(PDB_ReaderType::Raw, Path, Session));

    RawSession *RS = static_cast<RawSession *>(Session.get());
    ExitOnErr(dumpStructure(*RS));
    return;
  }

  ExitOnErr(loadDataForPDB(PDB_ReaderType::DIA, Path, Session));

  if (opts::LoadAddress)
    Session->setLoadAddress(opts::LoadAddress);

  LinePrinter Printer(2, outs());

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

  if (opts::Compilands) {
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::SectionHeader).get()
        << "---COMPILANDS---";
    Printer.Indent();
    auto Compilands = GlobalScope->findAllChildren<PDBSymbolCompiland>();
    CompilandDumper Dumper(Printer);
    CompilandDumpFlags options = CompilandDumper::Flags::None;
    if (opts::Lines)
      options = options | CompilandDumper::Flags::Lines;
    while (auto Compiland = Compilands->getNext())
      Dumper.start(*Compiland, options);
    Printer.Unindent();
  }

  if (opts::Types) {
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::SectionHeader).get() << "---TYPES---";
    Printer.Indent();
    TypeDumper Dumper(Printer);
    Dumper.start(*GlobalScope);
    Printer.Unindent();
  }

  if (opts::Symbols) {
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::SectionHeader).get() << "---SYMBOLS---";
    Printer.Indent();
    auto Compilands = GlobalScope->findAllChildren<PDBSymbolCompiland>();
    CompilandDumper Dumper(Printer);
    while (auto Compiland = Compilands->getNext())
      Dumper.start(*Compiland, true);
    Printer.Unindent();
  }

  if (opts::Globals) {
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::SectionHeader).get() << "---GLOBALS---";
    Printer.Indent();
    {
      FunctionDumper Dumper(Printer);
      auto Functions = GlobalScope->findAllChildren<PDBSymbolFunc>();
      while (auto Function = Functions->getNext()) {
        Printer.NewLine();
        Dumper.start(*Function, FunctionDumper::PointerType::None);
      }
    }
    {
      auto Vars = GlobalScope->findAllChildren<PDBSymbolData>();
      VariableDumper Dumper(Printer);
      while (auto Var = Vars->getNext())
        Dumper.start(*Var);
    }
    {
      auto Thunks = GlobalScope->findAllChildren<PDBSymbolThunk>();
      CompilandDumper Dumper(Printer);
      while (auto Thunk = Thunks->getNext())
        Dumper.dump(*Thunk);
    }
    Printer.Unindent();
  }
  if (opts::Externals) {
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::SectionHeader).get() << "---EXTERNALS---";
    Printer.Indent();
    ExternalSymbolDumper Dumper(Printer);
    Dumper.start(*GlobalScope);
  }
  if (opts::Lines) {
    Printer.NewLine();
  }
  outs().flush();
}

int main(int argc_, const char *argv_[]) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc_, argv_);

  ExitOnErr.setBanner("llvm-pdbdump: ");

  SmallVector<const char *, 256> argv;
  SpecificBumpPtrAllocator<char> ArgAllocator;
  ExitOnErr(errorCodeToError(sys::Process::GetArgumentVector(
      argv, makeArrayRef(argv_, argc_), ArgAllocator)));

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argv.size(), argv.data(), "LLVM PDB Dumper\n");
  if (opts::Lines)
    opts::Compilands = true;

  if (opts::All) {
    opts::Compilands = true;
    opts::Symbols = true;
    opts::Globals = true;
    opts::Types = true;
    opts::Externals = true;
    opts::Lines = true;
  }

  if (opts::RawAll) {
    opts::DumpHeaders = true;
    opts::DumpModules = true;
    opts::DumpModuleFiles = true;
    opts::DumpModuleSyms = true;
    opts::DumpPublics = true;
    opts::DumpStreamSummary = true;
    opts::DumpStreamBlocks = true;
    opts::DumpTpiRecords = true;
    opts::DumpIpiRecords = true;
  }

  // When adding filters for excluded compilands and types, we need to remember
  // that these are regexes.  So special characters such as * and \ need to be
  // escaped in the regex.  In the case of a literal \, this means it needs to
  // be escaped again in the C++.  So matching a single \ in the input requires
  // 4 \es in the C++.
  if (opts::ExcludeCompilerGenerated) {
    opts::ExcludeTypes.push_back("__vc_attributes");
    opts::ExcludeCompilands.push_back("\\* Linker \\*");
  }
  if (opts::ExcludeSystemLibraries) {
    opts::ExcludeCompilands.push_back(
        "f:\\\\binaries\\\\Intermediate\\\\vctools\\\\crt_bld");
    opts::ExcludeCompilands.push_back("f:\\\\dd\\\\vctools\\\\crt");
    opts::ExcludeCompilands.push_back("d:\\\\th.obj.x86fre\\\\minkernel");
  }

#if defined(HAVE_DIA_SDK)
  CoInitializeEx(nullptr, COINIT_MULTITHREADED);
#endif

  std::for_each(opts::InputFilenames.begin(), opts::InputFilenames.end(),
                dumpInput);

#if defined(HAVE_DIA_SDK)
  CoUninitialize();
#endif
  outs().flush();
  return 0;
}
