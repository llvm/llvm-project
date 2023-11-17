//===-- llvm-readtapi.cpp - tapi file reader and manipulator -----*- C++-*-===//
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
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TextAPI/TextAPIError.h"
#include "llvm/TextAPI/TextAPIReader.h"
#include "llvm/TextAPI/TextAPIWriter.h"
#include <cstdlib>

using namespace llvm;
using namespace MachO;
using namespace object;

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

// Handle error reporting in cases where `ExitOnError` is not used.
void reportError(Twine Message, int ExitCode = EXIT_FAILURE) {
  WithColor::error(errs()) << Message << "\n";
  errs().flush();
  exit(ExitCode);
}

struct Context {
  std::vector<std::string> Inputs;
  std::unique_ptr<llvm::raw_fd_stream> OutStream;
  FileType WriteFT = FileType::TBD_V5;
  bool Compact = false;
};

std::unique_ptr<InterfaceFile> getInterfaceFile(const StringRef Filename,
                                                ExitOnError &ExitOnErr) {
  ExitOnErr.setBanner("error: '" + Filename.str() + "' ");
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(Filename);
  if (BufferOrErr.getError())
    ExitOnErr(errorCodeToError(BufferOrErr.getError()));
  Expected<std::unique_ptr<InterfaceFile>> IF =
      TextAPIReader::get((*BufferOrErr)->getMemBufferRef());
  if (!IF)
    ExitOnErr(IF.takeError());
  return std::move(*IF);
}

// Use unique exit code to differentiate failures not directly caused from
// TextAPI operations. This is used for wrapping `compare` operations in
// automation and scripting.
const int NON_TAPI_EXIT_CODE = 2;

bool handleCompareAction(const Context &Ctx) {
  if (Ctx.Inputs.size() != 2)
    reportError("compare only supports two input files",
                /*ExitCode=*/NON_TAPI_EXIT_CODE);

  ExitOnError ExitOnErr("error: ", /*DefaultErrorExitCode=*/NON_TAPI_EXIT_CODE);
  auto LeftIF = getInterfaceFile(Ctx.Inputs.front(), ExitOnErr);
  auto RightIF = getInterfaceFile(Ctx.Inputs.at(1), ExitOnErr);

  raw_ostream &OS = Ctx.OutStream ? *Ctx.OutStream : outs();
  return DiffEngine(LeftIF.get(), RightIF.get()).compareFiles(OS);
}

bool handleWriteAction(const Context &Ctx,
                       std::unique_ptr<InterfaceFile> Out = nullptr) {
  ExitOnError ExitOnErr("error: ");
  if (!Out) {
    if (Ctx.Inputs.size() != 1)
      reportError("write only supports one input file");
    Out = getInterfaceFile(Ctx.Inputs.front(), ExitOnErr);
  }
  raw_ostream &OS = Ctx.OutStream ? *Ctx.OutStream : outs();
  ExitOnErr(TextAPIWriter::writeToStream(OS, *Out, Ctx.WriteFT, Ctx.Compact));
  return EXIT_SUCCESS;
}

bool handleMergeAction(const Context &Ctx) {
  if (Ctx.Inputs.size() < 2)
    reportError("merge requires at least two input files");

  ExitOnError ExitOnErr("error: ");
  std::unique_ptr<InterfaceFile> Out;
  for (StringRef FileName : Ctx.Inputs) {
    auto IF = getInterfaceFile(FileName, ExitOnErr);
    // On the first iteration copy the input file and skip merge.
    if (!Out) {
      Out = std::move(IF);
      continue;
    }
    auto ResultIF = Out->merge(IF.get());
    if (!ResultIF)
      ExitOnErr(ResultIF.takeError());
    Out = std::move(ResultIF.get());
  }
  return handleWriteAction(Ctx, std::move(Out));
}

} // anonymous namespace

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  BumpPtrAllocator A;
  StringSaver Saver(A);
  TAPIOptTable Tbl;
  Context Ctx;
  opt::InputArgList Args =
      Tbl.parseArgs(Argc, Argv, OPT_UNKNOWN, Saver, [&](StringRef Msg) {
        WithColor::error(errs(), "llvm-readtapi") << Msg << "\n";
        exit(1);
      });
  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(outs(), "llvm-readtapi [options] <inputs>",
                  "LLVM TAPI file reader and manipulator");
    return EXIT_SUCCESS;
  }

  for (opt::Arg *A : Args.filtered(OPT_INPUT))
    Ctx.Inputs.push_back(A->getValue());

  if (opt::Arg *A = Args.getLastArg(OPT_output_EQ)) {
    std::string OutputLoc = std::move(A->getValue());
    std::error_code EC;
    Ctx.OutStream = std::make_unique<llvm::raw_fd_stream>(OutputLoc, EC);
    if (EC) {
      llvm::errs() << "error opening the file '" << OutputLoc
                   << "': " << EC.message() << "\n";
      return NON_TAPI_EXIT_CODE;
    }
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
  }

  return EXIT_SUCCESS;
}
