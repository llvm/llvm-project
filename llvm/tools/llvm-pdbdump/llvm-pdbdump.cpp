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
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
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
    for (const uint32_t &DirectoryBlockAddr : DirectoryBlocks) {
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

  // Stream 1 starts with the following header:
  //   uint32_t Version;
  //   uint32_t Signature;
  //   uint32_t Age;
  //   GUID Guid;
  PDBStream Stream1(1, File);
  uint32_t Version;
  uint32_t Signature;
  uint32_t Age;
  PDB_UniqueId Guid;

  Stream1.readInteger(Version);
  outs() << "Version: " << Version << '\n';
  // PDB's with versions before PDBImpvVC70 might not have the Guid field, we
  // don't support them.
  if (Version < 20000404)
    reportError("", std::make_error_code(std::errc::not_supported));

  // This appears to be the time the PDB was last opened by an MSVC tool?
  // It is definitely a timestamp of some sort.
  Stream1.readInteger(Signature);
  outs() << "Signature: ";
  outs().write_hex(Signature) << '\n';

  // This appears to be a number which is used to determine that the PDB is kept
  // in sync with the EXE.
  Stream1.readInteger(Age);
  outs() << "Age: " << Age << '\n';

  // I'm not sure what the purpose of the GUID is.
  Stream1.readObject(&Guid);
  outs() << "Guid: " << Guid << '\n';

  // This is some sort of weird string-set/hash table encoded in the stream.
  // It starts with the number of bytes in the table.
  uint32_t NumberOfBytes;
  Stream1.readInteger(NumberOfBytes);
  outs() << "NumberOfBytes: " << NumberOfBytes << '\n';

  // Following that field is the starting offset of strings in the name table.
  uint32_t StringsOffset = Stream1.getOffset();
  Stream1.setOffset(StringsOffset + NumberOfBytes);

  // This appears to be equivalent to the total number of strings *actually*
  // in the name table.
  uint32_t HashSize;
  Stream1.readInteger(HashSize);
  outs() << "HashSize: " << HashSize << '\n';

  // This appears to be an upper bound on the number of strings in the name
  // table.
  uint32_t MaxNumberOfStrings;
  Stream1.readInteger(MaxNumberOfStrings);
  outs() << "MaxNumberOfStrings: " << MaxNumberOfStrings << '\n';

  // This appears to be a hash table which uses bitfields to determine whether
  // or not a bucket is 'present'.
  uint32_t NumPresentWords;
  Stream1.readInteger(NumPresentWords);
  outs() << "NumPresentWords: " << NumPresentWords << '\n';

  // Store all the 'present' bits in a vector for later processing.
  SmallVector<uint32_t, 1> PresentWords;
  for (uint32_t I = 0; I != NumPresentWords; ++I) {
    uint32_t Word;
    Stream1.readInteger(Word);
    PresentWords.push_back(Word);
    outs() << "Word: " << Word << '\n';
  }

  // This appears to be a hash table which uses bitfields to determine whether
  // or not a bucket is 'deleted'.
  uint32_t NumDeletedWords;
  Stream1.readInteger(NumDeletedWords);
  outs() << "NumDeletedWords: " << NumDeletedWords << '\n';

  // Store all the 'deleted' bits in a vector for later processing.
  SmallVector<uint32_t, 1> DeletedWords;
  for (uint32_t I = 0; I != NumDeletedWords; ++I) {
    uint32_t Word;
    Stream1.readInteger(Word);
    DeletedWords.push_back(Word);
    outs() << "Word: " << Word << '\n';
  }

  BitVector Present(MaxNumberOfStrings, false);
  if (!PresentWords.empty())
    Present.setBitsInMask(PresentWords.data(), PresentWords.size());
  BitVector Deleted(MaxNumberOfStrings, false);
  if (!DeletedWords.empty())
    Deleted.setBitsInMask(DeletedWords.data(), DeletedWords.size());

  StringMap<uint32_t> NamedStreams;
  for (uint32_t I = 0; I < MaxNumberOfStrings; ++I) {
    if (!Present.test(I))
      continue;

    // For all present entries, dump out their mapping.

    // This appears to be an offset relative to the start of the strings.
    // It tells us where the null-terminated string begins.
    uint32_t NameOffset;
    Stream1.readInteger(NameOffset);
    outs() << "NameOffset: " << NameOffset << '\n';

    // This appears to be a stream number into the stream directory.
    uint32_t NameIndex;
    Stream1.readInteger(NameIndex);
    outs() << "NameIndex: " << NameIndex << '\n';

    // Compute the offset of the start of the string relative to the stream.
    uint32_t StringOffset = StringsOffset + NameOffset;
    uint32_t OldOffset = Stream1.getOffset();
    // Pump out our c-string from the stream.
    std::string Str;
    Stream1.setOffset(StringOffset);
    Stream1.readZeroString(Str);
    outs() << "String: " << Str << "\n\n";

    Stream1.setOffset(OldOffset);
    // Add this to a string-map from name to stream number.
    NamedStreams.insert({Str, NameIndex});
  }

  // Let's try to dump out the named stream "/names".
  auto NameI = NamedStreams.find("/names");
  if (NameI != NamedStreams.end()) {
    PDBStream NameStream(NameI->second, File);
    outs() << "NameStream: " << NameI->second << '\n';

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
