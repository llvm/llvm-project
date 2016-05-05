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
#include "llvm/DebugInfo/PDB/Raw/NameHashTable.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawSession.h"
#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"
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

cl::opt<bool> DumpHeaders("dump-headers", cl::desc("dump PDB headers"),
                          cl::cat(OtherOptions));
cl::opt<bool> DumpStreamSizes("dump-stream-sizes",
                              cl::desc("dump PDB stream sizes"),
                              cl::cat(OtherOptions));
cl::opt<bool> DumpStreamBlocks("dump-stream-blocks",
                               cl::desc("dump PDB stream blocks"),
                               cl::cat(OtherOptions));
cl::opt<bool> DumpTypeStream("dump-tpi-stream",
                             cl::desc("dump PDB TPI (Type Info) stream"),
                             cl::cat(OtherOptions));
cl::opt<bool>
    DumpTpiRecordBytes("dump-tpi-record-bytes",
                       cl::desc("dump CodeView type record raw bytes"),
                       cl::cat(OtherOptions));
cl::opt<std::string> DumpStreamData("dump-stream", cl::desc("dump stream data"),
                                    cl::cat(OtherOptions));

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

static void dumpFileHeaders(ScopedPrinter &P, PDBFile &File) {
  if (!opts::DumpHeaders)
    return;
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
}

static void dumpStreamSizes(ScopedPrinter &P, PDBFile &File) {
  if (!opts::DumpStreamSizes)
    return;

  ListScope L(P, "StreamSizes");
  uint32_t StreamCount = File.getNumStreams();
  for (uint32_t StreamIdx = 0; StreamIdx < StreamCount; ++StreamIdx) {
    std::string Name("Stream ");
    Name += to_string(StreamIdx);
    P.printNumber(Name, File.getStreamByteSize(StreamIdx));
  }
}

static void dumpStreamBlocks(ScopedPrinter &P, PDBFile &File) {
  if (!opts::DumpStreamBlocks)
    return;

  ListScope L(P, "StreamBlocks");
  uint32_t StreamCount = File.getNumStreams();
  for (uint32_t StreamIdx = 0; StreamIdx < StreamCount; ++StreamIdx) {
    std::string Name("Stream ");
    Name += to_string(StreamIdx);
    auto StreamBlocks = File.getStreamBlockList(StreamIdx);
    P.printList(Name, StreamBlocks);
  }
}

static void dumpStreamData(ScopedPrinter &P, PDBFile &File) {
  uint32_t StreamCount = File.getNumStreams();
  StringRef DumpStreamStr = opts::DumpStreamData;
  uint32_t DumpStreamNum;
  if (DumpStreamStr.getAsInteger(/*Radix=*/0U, DumpStreamNum) ||
      DumpStreamNum >= StreamCount)
    return;

  uint32_t StreamBytesRead = 0;
  uint32_t StreamSize = File.getStreamByteSize(DumpStreamNum);
  auto StreamBlocks = File.getStreamBlockList(DumpStreamNum);

  for (uint32_t StreamBlockAddr : StreamBlocks) {
    uint32_t BytesLeftToReadInStream = StreamSize - StreamBytesRead;
    if (BytesLeftToReadInStream == 0)
      break;

    uint32_t BytesToReadInBlock = std::min(
        BytesLeftToReadInStream, static_cast<uint32_t>(File.getBlockSize()));
    auto StreamBlockData =
        File.getBlockData(StreamBlockAddr, BytesToReadInBlock);

    outs() << StreamBlockData;
    StreamBytesRead += StreamBlockData.size();
  }
}

static void dumpInfoStream(ScopedPrinter &P, PDBFile &File) {
  InfoStream &IS = File.getPDBInfoStream();

  DictScope D(P, "PDB Stream");
  P.printNumber("Version", IS.getVersion());
  P.printHex("Signature", IS.getSignature());
  P.printNumber("Age", IS.getAge());
  P.printObject("Guid", IS.getGuid());
}

static void dumpNamedStream(ScopedPrinter &P, PDBFile &File, StringRef Stream) {
  InfoStream &IS = File.getPDBInfoStream();
  uint32_t NameStreamIndex = IS.getNamedStreamIndex(Stream);

  if (NameStreamIndex != 0) {
    std::string Name("Stream '");
    Name += Stream;
    Name += "'";
    DictScope D(P, Name);
    P.printNumber("Index", NameStreamIndex);

    MappedBlockStream NameStream(NameStreamIndex, File);
    StreamReader Reader(NameStream);

    NameHashTable NameTable;
    NameTable.load(Reader);
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
}

static void dumpDbiStream(ScopedPrinter &P, PDBFile &File) {
  DbiStream &DS = File.getPDBDbiStream();

  DictScope D(P, "DBI Stream");
  P.printNumber("Dbi Version", DS.getDbiVersion());
  P.printNumber("Age", DS.getAge());
  P.printBoolean("Incremental Linking", DS.isIncrementallyLinked());
  P.printBoolean("Has CTypes", DS.hasCTypes());
  P.printBoolean("Is Stripped", DS.isStripped());
  P.printObject("Machine Type", DS.getMachineType());
  P.printNumber("Number of Symbols", DS.getNumberOfSymbols());

  uint16_t Major = DS.getBuildMajorVersion();
  uint16_t Minor = DS.getBuildMinorVersion();
  P.printVersion("Toolchain Version", Major, Minor);

  std::string DllName;
  raw_string_ostream DllStream(DllName);
  DllStream << "mspdb" << Major << Minor << ".dll version";
  DllStream.flush();
  P.printVersion(DllName, Major, Minor, DS.getPdbDllVersion());

  ListScope L(P, "Modules");
  for (auto &Modi : DS.modules()) {
    DictScope DD(P);
    P.printString("Name", Modi.Info.getModuleName());
    P.printNumber("Debug Stream Index", Modi.Info.getModuleStreamIndex());
    P.printString("Object File Name", Modi.Info.getObjFileName());
    P.printNumber("Num Files", Modi.Info.getNumberOfFiles());
    P.printNumber("Source File Name Idx", Modi.Info.getSourceFileNameIndex());
    P.printNumber("Pdb File Name Idx", Modi.Info.getPdbFilePathNameIndex());
    P.printNumber("Line Info Byte Size", Modi.Info.getLineInfoByteSize());
    P.printNumber("C13 Line Info Byte Size",
                  Modi.Info.getC13LineInfoByteSize());
    P.printNumber("Symbol Byte Size", Modi.Info.getSymbolDebugInfoByteSize());
    P.printNumber("Type Server Index", Modi.Info.getTypeServerIndex());
    P.printBoolean("Has EC Info", Modi.Info.hasECInfo());
    std::string FileListName =
        to_string(Modi.SourceFiles.size()) + " Contributing Source Files";
    ListScope LL(P, FileListName);
    for (auto File : Modi.SourceFiles)
      P.printString(File);
  }
}

static void dumpTpiStream(ScopedPrinter &P, PDBFile &File) {
  if (!opts::DumpTypeStream)
    return;

  DictScope D(P, "Type Info Stream");

  TpiStream &Tpi = File.getPDBTpiStream();
  P.printNumber("TPI Version", Tpi.getTpiVersion());
  P.printNumber("Record count", Tpi.NumTypeRecords());

  if (!opts::DumpTpiRecordBytes)
    return;

  ListScope L(P, "Records");
  for (auto &Type : Tpi.types()) {
    DictScope DD(P, "");
    P.printHex("Kind", unsigned(Type.Leaf));
    P.printBinaryBlock("Bytes", Type.LeafData);
  }
}

static void dumpStructure(RawSession &RS) {
  PDBFile &File = RS.getPDBFile();
  ScopedPrinter P(outs());

  dumpFileHeaders(P, File);

  dumpStreamSizes(P, File);

  dumpStreamBlocks(P, File);

  dumpStreamData(P, File);

  dumpInfoStream(P, File);

  dumpNamedStream(P, File, "/names");

  dumpDbiStream(P, File);

  dumpTpiStream(P, File);
}

static void reportError(StringRef Path, PDB_ErrorCode Error) {
  switch (Error) {
  case PDB_ErrorCode::Success:
    break;
  case PDB_ErrorCode::NoDiaSupport:
    outs() << "LLVM was not compiled with support for DIA.  This usually means "
              "that either LLVM was not compiled with MSVC, or your MSVC "
              "installation is corrupt.\n";
    return;
  case PDB_ErrorCode::CouldNotCreateImpl:
    outs() << "Failed to connect to DIA at runtime.  Verify that Visual Studio "
              "is properly installed, or that msdiaXX.dll is in your PATH.\n";
    return;
  case PDB_ErrorCode::InvalidPath:
    outs() << "Unable to load PDB at '" << Path
           << "'.  Check that the file exists and is readable.\n";
    return;
  case PDB_ErrorCode::InvalidFileFormat:
    outs() << "Unable to load PDB at '" << Path
           << "'.  The file has an unrecognized format.\n";
    return;
  default:
    outs() << "Unable to load PDB at '" << Path
           << "'.  An unknown error occured.\n";
    return;
  }
}

static void dumpInput(StringRef Path) {
  std::unique_ptr<IPDBSession> Session;
  if (opts::DumpHeaders || !opts::DumpStreamData.empty()) {
    PDB_ErrorCode Error = loadDataForPDB(PDB_ReaderType::Raw, Path, Session);
    if (Error == PDB_ErrorCode::Success) {
      RawSession *RS = static_cast<RawSession *>(Session.get());
      dumpStructure(*RS);
    }

    reportError(Path, Error);
    outs().flush();
    return;
  }

  PDB_ErrorCode Error = loadDataForPDB(PDB_ReaderType::DIA, Path, Session);
  if (Error != PDB_ErrorCode::Success) {
    reportError(Path, Error);
    return;
  }

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

  SmallVector<const char *, 256> argv;
  SpecificBumpPtrAllocator<char> ArgAllocator;
  std::error_code EC = sys::Process::GetArgumentVector(
      argv, makeArrayRef(argv_, argc_), ArgAllocator);
  if (EC) {
    errs() << "error: couldn't get arguments: " << EC.message() << '\n';
    return 1;
  }

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

  return 0;
}
