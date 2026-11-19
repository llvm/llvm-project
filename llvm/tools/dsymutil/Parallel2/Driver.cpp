//===- Driver.cpp - dsymutil --linker=parallel2 ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// --linker=parallel2 is a dsymutil mode optimized for concurrency and runtime.
// It uses lld's design: A sequence of parallel for loops. It also copies and
// patches DWARF instead of parsing and regenerating it.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::dsymutil::parallel2;

namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Options.inc"
#undef OPTION
};

struct Options {
  std::string InputFile;
};
} // namespace

[[noreturn]] static void fatal(const Twine &Message) {
  WithColor::error() << Message << '\n';
  exit(1);
}

static void warn(const Twine &Message) {
  WithColor::warning() << Message << '\n';
}

static bool isOptionImplemented(unsigned ID) {
  switch (ID) {
  case OPT_INPUT:
  case OPT_linker:
    return true;
  default:
    return false;
  }
}

int dsymutil::parallel2::link(const opt::InputArgList &Args) {
  // dsymutil has warned about the options that it does not know.
  for (const opt::Arg *Arg : Args) {
    unsigned ID = Arg->getOption().getID();
    if (ID != OPT_UNKNOWN && !isOptionImplemented(ID))
      warn("'" + Arg->getSpelling() +
           "' not yet implemented by --linker=parallel2");
  }

  Options Opts;
  std::vector<std::string> InputFiles = Args.getAllArgValues(OPT_INPUT);
  if (InputFiles.size() != 1)
    fatal("expected exactly one input file");
  Opts.InputFile = InputFiles[0];

  ErrorOr<std::unique_ptr<MemoryBuffer>> Binary =
      MemoryBuffer::getFile(Opts.InputFile, /*IsText=*/false,
                            /*RequiresNullTerminator=*/false);
  if (std::error_code EC = Binary.getError())
    fatal(Opts.InputFile + ": " + EC.message());

  // FIXME: Link the rest of the owl.

  return 0;
}
