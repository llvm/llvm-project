//===-- llvm-cgdata.cpp - LLVM CodeGen Data Tool --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// llvm-cgdata parses raw codegen data embedded in compiled binary files, and
// merges them into a single .cgdata file. It can also inspect and maninuplate
// a .cgdata file. This .cgdata can contain various codegen data like outlining
// information, and it can be used to optimize the code in the subsequent build.
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/StringRef.h"
#include "llvm/CGData/CodeGenDataReader.h"
#include "llvm/CGData/CodeGenDataWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

enum CGDataFormat {
  Invalid,
  Text,
  Binary,
};

enum CGDataAction {
  Convert,
  Merge,
  Show,
};

// Command-line option boilerplate.
namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

using namespace llvm::opt;
static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

class CGDataOptTable : public opt::GenericOptTable {
public:
  CGDataOptTable()
      : GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable) {}
};
} // end anonymous namespace

// Options
static StringRef ToolName;
static std::string OutputFilename = "-";
static std::string Filename;
static bool ShowCGDataVersion;
static bool SkipTrim;
static CGDataAction Action;
static std::optional<CGDataFormat> OutputFormat;
static std::vector<std::string> InputFilenames;

static void exitWithError(Twine Message, StringRef Whence = "",
                          StringRef Hint = "") {
  WithColor::error();
  if (!Whence.empty())
    errs() << Whence << ": ";
  errs() << Message << "\n";
  if (!Hint.empty())
    WithColor::note() << Hint << "\n";
  ::exit(1);
}

static void exitWithError(Error E, StringRef Whence = "") {
  if (E.isA<CGDataError>()) {
    handleAllErrors(std::move(E), [&](const CGDataError &IPE) {
      exitWithError(IPE.message(), Whence);
    });
    return;
  }

  exitWithError(toString(std::move(E)), Whence);
}

static void exitWithErrorCode(std::error_code EC, StringRef Whence = "") {
  exitWithError(EC.message(), Whence);
}

static int convert_main(int argc, const char *argv[]) {
  std::error_code EC;
  raw_fd_ostream OS(OutputFilename, EC,
                    OutputFormat == CGDataFormat::Text
                        ? sys::fs::OF_TextWithCRLF
                        : sys::fs::OF_None);
  if (EC)
    exitWithErrorCode(EC, OutputFilename);

  auto FS = vfs::getRealFileSystem();
  auto ReaderOrErr = CodeGenDataReader::create(Filename, *FS);
  if (Error E = ReaderOrErr.takeError())
    exitWithError(std::move(E), Filename);

  CodeGenDataWriter Writer;
  auto Reader = ReaderOrErr->get();
  if (Reader->hasOutlinedHashTree()) {
    OutlinedHashTreeRecord Record(Reader->releaseOutlinedHashTree());
    Writer.addRecord(Record);
  }
  if (Reader->hasStableFunctionMap()) {
    StableFunctionMapRecord Record(Reader->releaseStableFunctionMap());
    Writer.addRecord(Record);
  }

  if (OutputFormat == CGDataFormat::Text) {
    if (Error E = Writer.writeText(OS))
      exitWithError(std::move(E));
  } else {
    if (Error E = Writer.write(OS))
      exitWithError(std::move(E));
  }

  return 0;
}

static bool handleBuffer(StringRef Filename, MemoryBufferRef Buffer,
                         OutlinedHashTreeRecord &GlobalOutlineRecord,
                         StableFunctionMapRecord &GlobalFunctionMapRecord);

static bool handleArchive(StringRef Filename, Archive &Arch,
                          OutlinedHashTreeRecord &GlobalOutlineRecord,
                          StableFunctionMapRecord &GlobalFunctionMapRecord) {
  bool Result = true;
  Error Err = Error::success();
  for (const auto &Child : Arch.children(Err)) {
    auto BuffOrErr = Child.getMemoryBufferRef();
    if (Error E = BuffOrErr.takeError())
      exitWithError(std::move(E), Filename);
    auto NameOrErr = Child.getName();
    if (Error E = NameOrErr.takeError())
      exitWithError(std::move(E), Filename);
    std::string Name = (Filename + "(" + NameOrErr.get() + ")").str();
    Result &= handleBuffer(Name, BuffOrErr.get(), GlobalOutlineRecord,
                           GlobalFunctionMapRecord);
  }
  if (Err)
    exitWithError(std::move(Err), Filename);
  return Result;
}

static bool handleBuffer(StringRef Filename, MemoryBufferRef Buffer,
                         OutlinedHashTreeRecord &GlobalOutlineRecord,
                         StableFunctionMapRecord &GlobalFunctionMapRecord) {
  Expected<std::unique_ptr<object::Binary>> BinOrErr =
      object::createBinary(Buffer);
  if (Error E = BinOrErr.takeError())
    exitWithError(std::move(E), Filename);

  bool Result = true;
  if (auto *Obj = dyn_cast<ObjectFile>(BinOrErr->get())) {
    if (Error E = CodeGenDataReader::mergeFromObjectFile(
            Obj, GlobalOutlineRecord, GlobalFunctionMapRecord))
      exitWithError(std::move(E), Filename);
  } else if (auto *Arch = dyn_cast<Archive>(BinOrErr->get())) {
    Result &= handleArchive(Filename, *Arch, GlobalOutlineRecord,
                            GlobalFunctionMapRecord);
  } else {
    // TODO: Support for the MachO universal binary format.
    errs() << "Error: unsupported binary file: " << Filename << "\n";
    Result = false;
  }

  return Result;
}

static bool handleFile(StringRef Filename,
                       OutlinedHashTreeRecord &GlobalOutlineRecord,
                       StableFunctionMapRecord &GlobalFunctionMapRecord) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = BuffOrErr.getError())
    exitWithErrorCode(EC, Filename);
  return handleBuffer(Filename, *BuffOrErr.get(), GlobalOutlineRecord,
                      GlobalFunctionMapRecord);
}

static int merge_main(int argc, const char *argv[]) {
  bool Result = true;
  OutlinedHashTreeRecord GlobalOutlineRecord;
  StableFunctionMapRecord GlobalFunctionMapRecord;
  for (auto &Filename : InputFilenames)
    Result &=
        handleFile(Filename, GlobalOutlineRecord, GlobalFunctionMapRecord);

  if (!Result)
    exitWithError("failed to merge codegen data files.");

  GlobalFunctionMapRecord.finalize(SkipTrim);

  CodeGenDataWriter Writer;
  if (!GlobalOutlineRecord.empty())
    Writer.addRecord(GlobalOutlineRecord);
  if (!GlobalFunctionMapRecord.empty())
    Writer.addRecord(GlobalFunctionMapRecord);

  std::error_code EC;
  raw_fd_ostream OS(OutputFilename, EC,
                    OutputFormat == CGDataFormat::Text
                        ? sys::fs::OF_TextWithCRLF
                        : sys::fs::OF_None);
  if (EC)
    exitWithErrorCode(EC, OutputFilename);

  if (OutputFormat == CGDataFormat::Text) {
    if (Error E = Writer.writeText(OS))
      exitWithError(std::move(E));
  } else {
    if (Error E = Writer.write(OS))
      exitWithError(std::move(E));
  }

  return 0;
}

static int show_main(int argc, const char *argv[]) {
  std::error_code EC;
  raw_fd_ostream OS(OutputFilename.data(), EC, sys::fs::OF_TextWithCRLF);
  if (EC)
    exitWithErrorCode(EC, OutputFilename);

  auto FS = vfs::getRealFileSystem();
  auto ReaderOrErr = CodeGenDataReader::create(Filename, *FS);
  if (Error E = ReaderOrErr.takeError())
    exitWithError(std::move(E), Filename);

  auto Reader = ReaderOrErr->get();
  if (ShowCGDataVersion)
    OS << "Version: " << Reader->getVersion() << "\n";

  if (Reader->hasOutlinedHashTree()) {
    auto Tree = Reader->releaseOutlinedHashTree();
    OS << "Outlined hash tree:\n";
    OS << "  Total Node Count: " << Tree->size() << "\n";
    OS << "  Terminal Node Count: " << Tree->size(/*GetTerminalCountOnly=*/true)
       << "\n";
    OS << "  Depth: " << Tree->depth() << "\n";
  }
  if (Reader->hasStableFunctionMap()) {
    auto Map = Reader->releaseStableFunctionMap();
    OS << "Stable function map:\n";
    OS << "  Unique hash Count: " << Map->size() << "\n";
    OS << "  Total function Count: "
       << Map->size(StableFunctionMap::TotalFunctionCount) << "\n";
    OS << "  Mergeable function Count: "
       << Map->size(StableFunctionMap::MergeableFunctionCount) << "\n";
  }

  return 0;
}

static void parseArgs(int argc, char **argv) {
  CGDataOptTable Tbl;
  ToolName = argv[0];
  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver{A};
  llvm::opt::InputArgList Args =
      Tbl.parseArgs(argc, argv, OPT_UNKNOWN, Saver, [&](StringRef Msg) {
        llvm::errs() << Msg << '\n';
        std::exit(1);
      });

  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(
        llvm::outs(),
        "llvm-cgdata <action> [options] (<binary files>|<.cgdata file>)",
        ToolName.str().c_str());
    std::exit(0);
  }
  if (Args.hasArg(OPT_version)) {
    cl::PrintVersionMessage();
    std::exit(0);
  }

  ShowCGDataVersion = Args.hasArg(OPT_cgdata_version);
  SkipTrim = Args.hasArg(OPT_skip_trim);

  if (opt::Arg *A = Args.getLastArg(OPT_format)) {
    StringRef OF = A->getValue();
    OutputFormat = StringSwitch<CGDataFormat>(OF)
                       .Case("text", CGDataFormat::Text)
                       .Case("binary", CGDataFormat::Binary)
                       .Default(CGDataFormat::Invalid);
    if (OutputFormat == CGDataFormat::Invalid)
      exitWithError("unsupported format '" + OF + "'");
  }

  InputFilenames = Args.getAllArgValues(OPT_INPUT);
  if (InputFilenames.empty())
    exitWithError("No input file is specified.");
  Filename = InputFilenames[0];

  if (Args.hasArg(OPT_output)) {
    OutputFilename = Args.getLastArgValue(OPT_output);
    for (auto &Filename : InputFilenames)
      if (Filename == OutputFilename)
        exitWithError(
            "Input file name cannot be the same as the output file name!\n");
  }

  opt::Arg *ActionArg = nullptr;
  for (opt::Arg *Arg : Args.filtered(OPT_action_group)) {
    if (ActionArg)
      exitWithError("Only one action is allowed.");
    ActionArg = Arg;
  }
  if (!ActionArg)
    exitWithError("One action is required.");

  switch (ActionArg->getOption().getID()) {
  case OPT_show:
    if (InputFilenames.size() != 1)
      exitWithError("only one input file is allowed.");
    Action = CGDataAction::Show;
    break;
  case OPT_convert:
    // The default output format is text for convert.
    if (!OutputFormat)
      OutputFormat = CGDataFormat::Text;
    if (InputFilenames.size() != 1)
      exitWithError("only one input file is allowed.");
    Action = CGDataAction::Convert;
    break;
  case OPT_merge:
    // The default output format is binary for merge.
    if (!OutputFormat)
      OutputFormat = CGDataFormat::Binary;
    Action = CGDataAction::Merge;
    break;
  default:
    llvm_unreachable("unrecognized action");
  }
}

int llvm_cgdata_main(int argc, char **argvNonConst, const llvm::ToolContext &) {
  const char **argv = const_cast<const char **>(argvNonConst);
  parseArgs(argc, argvNonConst);

  switch (Action) {
  case CGDataAction::Convert:
    return convert_main(argc, argv);
  case CGDataAction::Merge:
    return merge_main(argc, argv);
  case CGDataAction::Show:
    return show_main(argc, argv);
  }

  llvm_unreachable("unrecognized action");
}
