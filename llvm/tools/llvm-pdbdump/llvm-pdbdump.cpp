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
#include "LLVMOutputStyle.h"
#include "LinePrinter.h"
#include "OutputStyle.h"
#include "TypeDumper.h"
#include "VariableDumper.h"
#include "YAMLOutputStyle.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/config.h"
#include "llvm/DebugInfo/CodeView/ByteStream.h"
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
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/RawSession.h"
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
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

namespace {
// A simple adapter that acts like a ByteStream but holds ownership over
// and underlying FileOutputBuffer.
class FileBufferByteStream : public ByteStream<true> {
public:
  FileBufferByteStream(std::unique_ptr<FileOutputBuffer> Buffer)
      : ByteStream(MutableArrayRef<uint8_t>(Buffer->getBufferStart(),
                                            Buffer->getBufferEnd())),
        FileBuffer(std::move(Buffer)) {}

private:
  std::unique_ptr<FileOutputBuffer> FileBuffer;
};
}

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

cl::opt<OutputStyleTy>
    RawOutputStyle("raw-output-style", cl::desc("Specify dump outpout style"),
                   cl::values(clEnumVal(LLVM, "LLVM default style"),
                              clEnumVal(YAML, "YAML style"), clEnumValEnd),
                   cl::init(LLVM), cl::cat(NativeOptions));

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
cl::opt<bool> DumpTpiHash("raw-tpi-hash",
                          cl::desc("dump CodeView TPI hash stream"),
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
cl::opt<bool> DumpSectionContribs("raw-section-contribs",
                                  cl::desc("dump section contributions"),
                                  cl::cat(NativeOptions));
cl::opt<bool> DumpLineInfo("raw-line-info",
                           cl::desc("dump file and line information"),
                           cl::cat(NativeOptions));
cl::opt<bool> DumpSectionMap("raw-section-map", cl::desc("dump section map"),
                             cl::cat(NativeOptions));
cl::opt<bool>
    DumpSymRecordBytes("raw-sym-record-bytes",
                       cl::desc("dump CodeView symbol record raw bytes"),
                       cl::cat(NativeOptions));
cl::opt<bool> DumpSectionHeaders("raw-section-headers",
                                 cl::desc("dump section headers"),
                                 cl::cat(NativeOptions));
cl::opt<bool> DumpFpo("raw-fpo", cl::desc("dump FPO records"),
                      cl::cat(NativeOptions));

cl::opt<bool>
    RawAll("raw-all",
           cl::desc("Implies most other options in 'Native Options' category"),
           cl::cat(NativeOptions));

cl::opt<bool>
    YamlToPdb("yaml-to-pdb",
              cl::desc("The input file is yaml, and the tool outputs a pdb"),
              cl::cat(NativeOptions));
cl::opt<std::string> YamlPdbOutputFile(
    "pdb-output", cl::desc("When yaml-to-pdb is specified, the output file"),
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

static Error dumpStructure(RawSession &RS) {
  PDBFile &File = RS.getPDBFile();
  std::unique_ptr<OutputStyle> O;
  if (opts::RawOutputStyle == opts::OutputStyleTy::LLVM)
    O = llvm::make_unique<LLVMOutputStyle>(File);
  else if (opts::RawOutputStyle == opts::OutputStyleTy::YAML)
    O = llvm::make_unique<YAMLOutputStyle>(File);
  else
    return make_error<RawError>(raw_error_code::feature_unsupported,
                                "Requested output style unsupported");

  if (auto EC = O->dumpFileHeaders())
    return EC;

  if (auto EC = O->dumpStreamSummary())
    return EC;

  if (auto EC = O->dumpStreamBlocks())
    return EC;

  if (auto EC = O->dumpStreamData())
    return EC;

  if (auto EC = O->dumpInfoStream())
    return EC;

  if (auto EC = O->dumpNamedStream())
    return EC;

  if (auto EC = O->dumpTpiStream(StreamTPI))
    return EC;

  if (auto EC = O->dumpTpiStream(StreamIPI))
    return EC;

  if (auto EC = O->dumpDbiStream())
    return EC;

  if (auto EC = O->dumpSectionContribs())
    return EC;

  if (auto EC = O->dumpSectionMap())
    return EC;

  if (auto EC = O->dumpPublicsStream())
    return EC;

  if (auto EC = O->dumpSectionHeaders())
    return EC;

  if (auto EC = O->dumpFpoStream())
    return EC;
  O->flush();
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
  if (opts::DumpTpiHash)
    return true;
  if (opts::DumpIpiRecords)
    return true;
  if (opts::DumpIpiRecordBytes)
    return true;
  if (opts::DumpSectionHeaders)
    return true;
  if (opts::DumpSectionContribs)
    return true;
  if (opts::DumpSectionMap)
    return true;
  if (opts::DumpLineInfo)
    return true;
  if (opts::DumpFpo)
    return true;
  return false;
}

static void yamlToPdb(StringRef Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> ErrorOrBuffer =
      MemoryBuffer::getFileOrSTDIN(Path, /*FileSize=*/-1,
                                   /*RequiresNullTerminator=*/false);

  if (ErrorOrBuffer.getError()) {
    ExitOnErr(make_error<GenericError>(generic_error_code::invalid_path, Path));
  }

  std::unique_ptr<MemoryBuffer> &Buffer = ErrorOrBuffer.get();

  llvm::yaml::Input In(Buffer->getBuffer());
  pdb::yaml::PdbObject YamlObj;
  In >> YamlObj;

  auto OutFileOrError = FileOutputBuffer::create(opts::YamlPdbOutputFile,
                                                 YamlObj.Headers.FileSize);
  if (OutFileOrError.getError())
    ExitOnErr(make_error<GenericError>(generic_error_code::invalid_path,
                                       opts::YamlPdbOutputFile));

  auto FileByteStream =
      llvm::make_unique<FileBufferByteStream>(std::move(*OutFileOrError));
  PDBFile Pdb(std::move(FileByteStream));
  Pdb.setSuperBlock(&YamlObj.Headers.SuperBlock);
  if (YamlObj.StreamMap.hasValue()) {
    std::vector<ArrayRef<support::ulittle32_t>> StreamMap;
    for (auto &E : YamlObj.StreamMap.getValue()) {
      StreamMap.push_back(E.Blocks);
    }
    Pdb.setStreamMap(StreamMap);
  }
  if (YamlObj.StreamSizes.hasValue()) {
    Pdb.setStreamSizes(YamlObj.StreamSizes.getValue());
  }

  Pdb.commit();
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
  sys::PrintStackTraceOnErrorSignal(argv_[0]);
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
    opts::DumpSectionHeaders = true;
    opts::DumpStreamSummary = true;
    opts::DumpStreamBlocks = true;
    opts::DumpTpiRecords = true;
    opts::DumpTpiHash = true;
    opts::DumpIpiRecords = true;
    opts::DumpSectionMap = true;
    opts::DumpSectionContribs = true;
    opts::DumpLineInfo = true;
    opts::DumpFpo = true;
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

  llvm::sys::InitializeCOMRAII COM(llvm::sys::COMThreadingMode::MultiThreaded);

  if (opts::YamlToPdb) {
    std::for_each(opts::InputFilenames.begin(), opts::InputFilenames.end(),
                  yamlToPdb);
  } else {
    std::for_each(opts::InputFilenames.begin(), opts::InputFilenames.end(),
                  dumpInput);
  }

  outs().flush();
  return 0;
}
