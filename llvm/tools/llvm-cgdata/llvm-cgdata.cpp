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
#include "llvm/CodeGenData/CodeGenDataReader.h"
#include "llvm/CodeGenData/CodeGenDataWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

// TODO: https://llvm.org/docs/CommandGuide/llvm-cgdata.html has documentations
// on each subcommand.
cl::SubCommand DumpSubcommand(
    "dump",
    "Dump the (indexed) codegen data file in either text or binary format.");
cl::SubCommand MergeSubcommand(
    "merge", "Takes binary files having raw codegen data in custom sections, "
             "and merge them into an index codegen data file.");
cl::SubCommand
    ShowSubcommand("show", "Show summary of the (indexed) codegen data file.");

enum CGDataFormat {
  CD_None = 0,
  CD_Text,
  CD_Binary,
};

cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                    cl::init("-"), cl::desc("Output file"),
                                    cl::sub(DumpSubcommand),
                                    cl::sub(MergeSubcommand));
cl::alias OutputFilenameA("o", cl::desc("Alias for --output"),
                          cl::aliasopt(OutputFilename));

cl::opt<std::string> Filename(cl::Positional, cl::desc("<cgdata-file>"),
                              cl::sub(DumpSubcommand), cl::sub(ShowSubcommand));
cl::list<std::string> InputFilenames(cl::Positional, cl::sub(MergeSubcommand),
                                     cl::desc("<binary-files...>"));
cl::opt<CGDataFormat> OutputFormat(
    cl::desc("Format of output data"), cl::sub(DumpSubcommand),
    cl::init(CD_Text),
    cl::values(clEnumValN(CD_Text, "text", "Text encoding"),
               clEnumValN(CD_Binary, "binary", "Binary encoding")));

cl::opt<bool> ShowCGDataVersion("cgdata-version", cl::init(false),
                                cl::desc("Show cgdata version. "),
                                cl::sub(ShowSubcommand));

static void exitWithError(Twine Message, std::string Whence = "",
                          std::string Hint = "") {
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
      exitWithError(IPE.message(), std::string(Whence));
    });
    return;
  }

  exitWithError(toString(std::move(E)), std::string(Whence));
}

static void exitWithErrorCode(std::error_code EC, StringRef Whence = "") {
  exitWithError(EC.message(), std::string(Whence));
}

static int dump_main(int argc, const char *argv[]) {
  if (Filename == OutputFilename) {
    errs() << sys::path::filename(argv[0]) << " " << argv[1]
           << ": Input file name cannot be the same as the output file name!\n";
    return 1;
  }

  std::error_code EC;
  raw_fd_ostream OS(OutputFilename.data(), EC,
                    OutputFormat == CD_Text ? sys::fs::OF_TextWithCRLF
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

  if (OutputFormat == CD_Text) {
    if (Error E = Writer.writeText(OS))
      exitWithError(std::move(E));
  } else {
    if (Error E = Writer.write(OS))
      exitWithError(std::move(E));
  }

  return 0;
}

static bool handleBuffer(StringRef Filename, MemoryBufferRef Buffer,
                         OutlinedHashTreeRecord &GlobalOutlineRecord);

static bool handleArchive(StringRef Filename, Archive &Arch,
                          OutlinedHashTreeRecord &GlobalOutlineRecord) {
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
    Result &= handleBuffer(Name, BuffOrErr.get(), GlobalOutlineRecord);
  }
  if (Err)
    exitWithError(std::move(Err), Filename);
  return Result;
}

static bool handleBuffer(StringRef Filename, MemoryBufferRef Buffer,
                         OutlinedHashTreeRecord &GlobalOutlineRecord) {
  Expected<std::unique_ptr<Binary>> BinOrErr = object::createBinary(Buffer);
  if (Error E = BinOrErr.takeError())
    exitWithError(std::move(E), Filename);

  bool Result = true;
  if (auto *Obj = dyn_cast<ObjectFile>(BinOrErr->get())) {
    if (Error E =
            CodeGenDataReader::mergeFromObjectFile(Obj, GlobalOutlineRecord))
      exitWithError(std::move(E), Filename);
  } else if (auto *Arch = dyn_cast<Archive>(BinOrErr->get())) {
    Result &= handleArchive(Filename, *Arch, GlobalOutlineRecord);
  } else {
    // TODO: Support for the MachO universal binary format.
    errs() << "Error: unsupported binary file: " << Filename << "\n";
    Result = false;
  }

  return Result;
}

static bool handleFile(StringRef Filename,
                       OutlinedHashTreeRecord &GlobalOutlineRecord) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = BuffOrErr.getError())
    exitWithErrorCode(EC, Filename);
  return handleBuffer(Filename, *BuffOrErr.get(), GlobalOutlineRecord);
}

static int merge_main(int argc, const char *argv[]) {
  bool Result = true;
  OutlinedHashTreeRecord GlobalOutlineRecord;
  for (auto &Filename : InputFilenames)
    Result &= handleFile(Filename, GlobalOutlineRecord);

  if (!Result) {
    errs() << "Error: failed to merge codegen data files.\n";
    return 1;
  }

  CodeGenDataWriter Writer;
  if (!GlobalOutlineRecord.empty())
    Writer.addRecord(GlobalOutlineRecord);

  std::error_code EC;
  raw_fd_ostream Output(OutputFilename, EC, sys::fs::OF_None);
  if (EC)
    exitWithErrorCode(EC, OutputFilename);

  if (auto E = Writer.write(Output))
    exitWithError(std::move(E));

  return 0;
}

static int show_main(int argc, const char *argv[]) {
  if (Filename == OutputFilename) {
    errs() << sys::path::filename(argv[0]) << " " << argv[1]
           << ": Input file name cannot be the same as the output file name!\n";
    return 1;
  }

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

  return 0;
}

int llvm_cgdata_main(int argc, char **argvNonConst, const llvm::ToolContext &) {
  const char **argv = const_cast<const char **>(argvNonConst);

  StringRef ProgName(sys::path::filename(argv[0]));

  if (argc < 2) {
    errs() << ProgName
           << ": No subcommand specified! Run llvm-cgdata --help for usage.\n";
    return 1;
  }

  cl::ParseCommandLineOptions(argc, argv, "LLVM codegen data\n");

  if (DumpSubcommand)
    return dump_main(argc, argv);

  if (MergeSubcommand)
    return merge_main(argc, argv);

  if (ShowSubcommand)
    return show_main(argc, argv);

  errs() << ProgName
         << ": Unknown command. Run llvm-cgdata --help for usage.\n";
  return 1;
}
