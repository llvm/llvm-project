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
#include "llvm/DebugInfo/PDB/Raw/ModInfo.h"
#include "llvm/DebugInfo/PDB/Raw/PDBDbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/PDBInfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/PDBStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawSession.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#if defined(HAVE_DIA_SDK)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

using namespace llvm;

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


static void reportError(StringRef Input, StringRef Message) {
  if (Input == "-")
    Input = "<stdin>";
  errs() << Input << ": " << Message << "\n";
  errs().flush();
  exit(1);
}

static void reportError(StringRef Input, std::error_code EC) {
  reportError(Input, EC.message());
}

static void dumpStructure(RawSession &RS) {
  PDBFile &File = RS.getPDBFile();

  if (opts::DumpHeaders) {
    outs() << "BlockSize: " << File.getBlockSize() << '\n';
    outs() << "Unknown0: " << File.getUnknown0() << '\n';
    outs() << "NumBlocks: " << File.getBlockCount() << '\n';
    outs() << "NumDirectoryBytes: " << File.getNumDirectoryBytes() << '\n';
    outs() << "Unknown1: " << File.getUnknown1() << '\n';
    outs() << "BlockMapAddr: " << File.getBlockMapIndex() << '\n';
  }

  if (opts::DumpHeaders)
    outs() << "NumDirectoryBlocks: " << File.getNumDirectoryBlocks() << '\n';

  if (opts::DumpHeaders)
    outs() << "BlockMapOffset: " << File.getBlockMapOffset() << '\n';

  // The directory is not contiguous.  Instead, the block map contains a
  // contiguous list of block numbers whose contents, when concatenated in
  // order, make up the directory.
  auto DirectoryBlocks = File.getDirectoryBlockArray();

  if (opts::DumpHeaders) {
    outs() << "DirectoryBlocks: [";
    for (const auto &DirectoryBlockAddr : DirectoryBlocks) {
      if (&DirectoryBlockAddr != &DirectoryBlocks.front())
        outs() << ", ";
      outs() << DirectoryBlockAddr;
    }
    outs() << "]\n";
  }

  if (opts::DumpHeaders)
    outs() << "NumStreams: " << File.getNumStreams() << '\n';
  uint32_t StreamCount = File.getNumStreams();
  if (opts::DumpStreamSizes) {
    for (uint32_t StreamIdx = 0; StreamIdx < StreamCount; ++StreamIdx)
      outs() << "StreamSizes[" << StreamIdx
             << "]: " << File.getStreamByteSize(StreamIdx) << '\n';
  }

  if (opts::DumpStreamBlocks) {
    for (uint32_t StreamIdx = 0; StreamIdx < StreamCount; ++StreamIdx) {
      outs() << "StreamBlocks[" << StreamIdx << "]: [";
      auto StreamBlocks = File.getStreamBlockList(StreamIdx);
      for (size_t i = 0; i < StreamBlocks.size(); ++i) {
        if (i != 0)
          outs() << ", ";
        outs() << StreamBlocks[i];
      }
      outs() << "]\n";
    }
  }

  StringRef DumpStreamStr = opts::DumpStreamData;
  uint32_t DumpStreamNum;
  if (!DumpStreamStr.getAsInteger(/*Radix=*/0U, DumpStreamNum) &&
      DumpStreamNum < StreamCount) {
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

  PDBInfoStream &InfoStream = File.getPDBInfoStream();
  outs() << "Version: " << InfoStream.getVersion() << '\n';
  outs() << "Signature: ";
  outs().write_hex(InfoStream.getSignature()) << '\n';
  outs() << "Age: " << InfoStream.getAge() << '\n';
  outs() << "Guid: " << InfoStream.getGuid() << '\n';

  // Let's try to dump out the named stream "/names".
  uint32_t NameStreamIndex = InfoStream.getNamedStreamIndex("/names");
  if (NameStreamIndex != 0) {
    PDBStream NameStream(NameStreamIndex, File);
    outs() << "NameStream: " << NameStreamIndex << '\n';

    // The name stream appears to start with a signature and version.
    uint32_t NameStreamSignature;
    NameStream.readInteger(NameStreamSignature);
    outs() << "NameStreamSignature: ";
    outs().write_hex(NameStreamSignature) << '\n';

    uint32_t NameStreamVersion;
    NameStream.readInteger(NameStreamVersion);
    outs() << "NameStreamVersion: " << NameStreamVersion << '\n';

    // We only support this particular version of the name stream.
    if (NameStreamSignature != 0xeffeeffe || NameStreamVersion != 1)
      reportError("", std::make_error_code(std::errc::not_supported));
  }

  PDBDbiStream &DbiStream = File.getPDBDbiStream();
  outs() << "Dbi Version: " << DbiStream.getDbiVersion() << '\n';
  outs() << "Age: " << DbiStream.getAge() << '\n';
  outs() << "Incremental Linking: " << DbiStream.isIncrementallyLinked()
         << '\n';
  outs() << "Has CTypes: " << DbiStream.hasCTypes() << '\n';
  outs() << "Is Stripped: " << DbiStream.isStripped() << '\n';
  outs() << "Machine Type: " << DbiStream.getMachineType() << '\n';
  outs() << "Number of Symbols: " << DbiStream.getNumberOfSymbols() << '\n';

  uint16_t Major = DbiStream.getBuildMajorVersion();
  uint16_t Minor = DbiStream.getBuildMinorVersion();
  outs() << "Toolchain Version: " << Major << "." << Minor << '\n';
  outs() << "mspdb" << Major << Minor << ".dll version: " << Major << "."
         << Minor << "." << DbiStream.getPdbDllVersion() << '\n';

  outs() << "Modules: \n";
  for (auto Modi : DbiStream.modules()) {
    outs() << Modi.getModuleName() << '\n';
    outs().indent(4) << "Debug Stream Index: " << Modi.getModuleStreamIndex()
                     << '\n';
    outs().indent(4) << "Object File: " << Modi.getObjFileName() << '\n';
    outs().indent(4) << "Num Files: " << Modi.getNumberOfFiles() << '\n';
    outs().indent(4) << "Source File Name Idx: "
                     << Modi.getSourceFileNameIndex() << '\n';
    outs().indent(4) << "Pdb File Name Idx: " << Modi.getPdbFilePathNameIndex()
                     << '\n';
    outs().indent(4) << "Line Info Byte Size: " << Modi.getLineInfoByteSize()
                     << '\n';
    outs().indent(4) << "C13 Line Info Byte Size: "
                     << Modi.getC13LineInfoByteSize() << '\n';
    outs().indent(4) << "Symbol Byte Size: "
                     << Modi.getSymbolDebugInfoByteSize() << '\n';
    outs().indent(4) << "Type Server Index: " << Modi.getTypeServerIndex()
                     << '\n';
    outs().indent(4) << "Has EC Info: " << Modi.hasECInfo() << '\n';
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

    outs().flush();
    return;
  }

  PDB_ErrorCode Error = loadDataForPDB(PDB_ReaderType::DIA, Path, Session);
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
