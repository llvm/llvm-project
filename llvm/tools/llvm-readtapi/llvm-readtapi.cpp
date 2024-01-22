//===-- llvm-readtapi.cpp - tapi file reader and transformer -----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the command-line driver for llvm-readtapi.
//
//===----------------------------------------------------------------------===//
#include "DiffEngine.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TextAPI/DylibReader.h"
#include "llvm/TextAPI/TextAPIError.h"
#include "llvm/TextAPI/TextAPIReader.h"
#include "llvm/TextAPI/TextAPIWriter.h"
#include "llvm/TextAPI/Utils.h"
#include <cstdlib>

using namespace llvm;
using namespace MachO;
using namespace object;

#if !defined(PATH_MAX)
#define PATH_MAX 1024
#endif

namespace {
using namespace llvm::opt;
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "TapiOpts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE)                                                    \
  static constexpr StringLiteral NAME##_init[] = VALUE;                        \
  static constexpr ArrayRef<StringLiteral> NAME(NAME##_init,                   \
                                                std::size(NAME##_init) - 1);
#include "TapiOpts.inc"
#undef PREFIX

static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "TapiOpts.inc"
#undef OPTION
};

class TAPIOptTable : public opt::GenericOptTable {
public:
  TAPIOptTable() : opt::GenericOptTable(InfoTable) {
    setGroupedShortOptions(true);
  }
};

struct StubOptions {
  bool DeleteInput = false;
};

struct Context {
  std::vector<std::string> Inputs;
  std::unique_ptr<llvm::raw_fd_stream> OutStream;
  FileType WriteFT = FileType::TBD_V5;
  StubOptions StubOpt;
  bool Compact = false;
  Architecture Arch = AK_unknown;
};

// Use unique exit code to differentiate failures not directly caused from
// TextAPI operations. This is used for wrapping `compare` operations in
// automation and scripting.
const int NON_TAPI_EXIT_CODE = 2;
const std::string TOOLNAME = "llvm-readtapi";
ExitOnError ExitOnErr;
} // anonymous namespace

// Handle error reporting in cases where `ExitOnError` is not used.
static void reportError(Twine Message, int ExitCode = EXIT_FAILURE) {
  errs() << TOOLNAME << ": error: " << Message << "\n";
  errs().flush();
  exit(ExitCode);
}

static std::unique_ptr<InterfaceFile>
getInterfaceFile(const StringRef Filename, bool ResetBanner = true) {
  ExitOnErr.setBanner(TOOLNAME + ": error: '" + Filename.str() + "' ");
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(Filename);
  if (BufferOrErr.getError())
    ExitOnErr(errorCodeToError(BufferOrErr.getError()));
  auto Buffer = std::move(*BufferOrErr);

  std::unique_ptr<InterfaceFile> IF;
  switch (identify_magic(Buffer->getBuffer())) {
  case file_magic::macho_dynamically_linked_shared_lib:
    LLVM_FALLTHROUGH;
  case file_magic::macho_dynamically_linked_shared_lib_stub:
    LLVM_FALLTHROUGH;
  case file_magic::macho_universal_binary:
    IF = ExitOnErr(DylibReader::get(Buffer->getMemBufferRef()));
    break;
  case file_magic::tapi_file:
    IF = ExitOnErr(TextAPIReader::get(Buffer->getMemBufferRef()));
    break;
  default:
    reportError(Filename + ": unsupported file type");
  }

  if (ResetBanner)
    ExitOnErr.setBanner(TOOLNAME + ": error: ");
  return IF;
}

static bool handleCompareAction(const Context &Ctx) {
  if (Ctx.Inputs.size() != 2)
    reportError("compare only supports two input files",
                /*ExitCode=*/NON_TAPI_EXIT_CODE);

  // Override default exit code.
  ExitOnErr = ExitOnError(TOOLNAME + ": error: ",
                          /*DefaultErrorExitCode=*/NON_TAPI_EXIT_CODE);
  auto LeftIF = getInterfaceFile(Ctx.Inputs.front());
  auto RightIF = getInterfaceFile(Ctx.Inputs.at(1));

  raw_ostream &OS = Ctx.OutStream ? *Ctx.OutStream : outs();
  return DiffEngine(LeftIF.get(), RightIF.get()).compareFiles(OS);
}

static bool handleWriteAction(const Context &Ctx,
                              std::unique_ptr<InterfaceFile> Out = nullptr) {
  if (!Out) {
    if (Ctx.Inputs.size() != 1)
      reportError("write only supports one input file");
    Out = getInterfaceFile(Ctx.Inputs.front());
  }
  raw_ostream &OS = Ctx.OutStream ? *Ctx.OutStream : outs();
  ExitOnErr(TextAPIWriter::writeToStream(OS, *Out, Ctx.WriteFT, Ctx.Compact));
  return EXIT_SUCCESS;
}

static bool handleMergeAction(const Context &Ctx) {
  if (Ctx.Inputs.size() < 2)
    reportError("merge requires at least two input files");

  std::unique_ptr<InterfaceFile> Out;
  for (StringRef FileName : Ctx.Inputs) {
    auto IF = getInterfaceFile(FileName);
    // On the first iteration copy the input file and skip merge.
    if (!Out) {
      Out = std::move(IF);
      continue;
    }
    Out = ExitOnErr(Out->merge(IF.get()));
  }
  return handleWriteAction(Ctx, std::move(Out));
}

static bool handleStubifyAction(Context &Ctx) {
  if (Ctx.Inputs.empty())
    reportError("stubify requires at least one input file");

  if ((Ctx.Inputs.size() > 1) && (Ctx.OutStream != nullptr))
    reportError("cannot write multiple inputs into single output file");

  for (StringRef FileName : Ctx.Inputs) {
    auto IF = getInterfaceFile(FileName);
    if (Ctx.StubOpt.DeleteInput) {
      std::error_code EC;
      SmallString<PATH_MAX> OutputLoc = FileName;
      MachO::replace_extension(OutputLoc, ".tbd");
      Ctx.OutStream = std::make_unique<llvm::raw_fd_stream>(OutputLoc, EC);
      if (EC)
        reportError("opening file '" + OutputLoc + ": " + EC.message());
      if (auto Err = sys::fs::remove(FileName))
        reportError("deleting file '" + FileName + ": " + EC.message());
    }
    handleWriteAction(Ctx, std::move(IF));
  }
  return EXIT_SUCCESS;
}

using IFOperation =
    std::function<llvm::Expected<std::unique_ptr<InterfaceFile>>(
        const llvm::MachO::InterfaceFile &, Architecture)>;
static bool handleSingleFileAction(const Context &Ctx, const StringRef Action,
                                   IFOperation act) {
  if (Ctx.Inputs.size() != 1)
    reportError(Action + " only supports one input file");
  if (Ctx.Arch == AK_unknown)
    reportError(Action + " requires -arch <arch>");

  auto IF = getInterfaceFile(Ctx.Inputs.front(), /*ResetBanner=*/false);
  auto OutIF = act(*IF, Ctx.Arch);
  if (!OutIF)
    ExitOnErr(OutIF.takeError());

  return handleWriteAction(Ctx, std::move(*OutIF));
}

static void setStubOptions(opt::InputArgList &Args, StubOptions &Opt) {
  Opt.DeleteInput = Args.hasArg(OPT_delete_input);
}

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  BumpPtrAllocator A;
  StringSaver Saver(A);
  TAPIOptTable Tbl;
  Context Ctx;
  ExitOnErr.setBanner(TOOLNAME + ": error:");
  opt::InputArgList Args = Tbl.parseArgs(
      Argc, Argv, OPT_UNKNOWN, Saver, [&](StringRef Msg) { reportError(Msg); });
  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(outs(),
                  "USAGE: llvm-readtapi <command> [-arch <architecture> "
                  "<options>]* <inputs> [-o "
                  "<output>]*",
                  "LLVM TAPI file reader and transformer");
    return EXIT_SUCCESS;
  }

  if (Args.hasArg(OPT_version)) {
    cl::PrintVersionMessage();
    return EXIT_SUCCESS;
  }

  // TODO: Add support for picking up libraries from directory input.
  for (opt::Arg *A : Args.filtered(OPT_INPUT))
    Ctx.Inputs.push_back(A->getValue());

  if (opt::Arg *A = Args.getLastArg(OPT_output_EQ)) {
    std::string OutputLoc = std::move(A->getValue());
    std::error_code EC;
    Ctx.OutStream = std::make_unique<llvm::raw_fd_stream>(OutputLoc, EC);
    if (EC)
      reportError("error opening the file '" + OutputLoc + EC.message(),
                  NON_TAPI_EXIT_CODE);
  }

  Ctx.Compact = Args.hasArg(OPT_compact);

  if (opt::Arg *A = Args.getLastArg(OPT_filetype_EQ)) {
    StringRef FT = A->getValue();
    Ctx.WriteFT = TextAPIWriter::parseFileType(FT);
    if (Ctx.WriteFT < FileType::TBD_V3)
      reportError("deprecated filetype '" + FT + "' is not supported to write");
    if (Ctx.WriteFT == FileType::Invalid)
      reportError("unsupported filetype '" + FT + "'");
  }

  if (opt::Arg *A = Args.getLastArg(OPT_arch_EQ)) {
    StringRef Arch = A->getValue();
    Ctx.Arch = getArchitectureFromName(Arch);
    if (Ctx.Arch == AK_unknown)
      reportError("unsupported architecture '" + Arch);
  }
  // Handle top level and exclusive operation.
  SmallVector<opt::Arg *, 1> ActionArgs(Args.filtered(OPT_action_group));

  if (ActionArgs.empty())
    // If no action specified, write out tapi file in requested format.
    return handleWriteAction(Ctx);

  if (ActionArgs.size() > 1) {
    std::string Buf;
    raw_string_ostream OS(Buf);
    OS << "only one of the following actions can be specified:";
    for (auto *Arg : ActionArgs)
      OS << " " << Arg->getSpelling();
    reportError(OS.str());
  }

  switch (ActionArgs.front()->getOption().getID()) {
  case OPT_compare:
    return handleCompareAction(Ctx);
  case OPT_merge:
    return handleMergeAction(Ctx);
  case OPT_extract:
    return handleSingleFileAction(Ctx, "extract", &InterfaceFile::extract);
  case OPT_remove:
    return handleSingleFileAction(Ctx, "remove", &InterfaceFile::remove);
  case OPT_stubify:
    setStubOptions(Args, Ctx.StubOpt);
    return handleStubifyAction(Ctx);
  }

  return EXIT_SUCCESS;
}
