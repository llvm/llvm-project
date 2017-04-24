//===- llvm-cvtres.cpp - Serialize .res files into .obj ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Serialize .res files into .obj files.  This is intended to be a
// platform-independent port of Microsoft's cvtres.exe.
//
//===----------------------------------------------------------------------===//

#include "llvm-cvtres.h"

#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR)                                              \
  OPT_##ID,
#include "Opts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Opts.inc"
#undef PREFIX

static const opt::OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR)                                              \
  {                                                                            \
      PREFIX,      NAME,     HELPTEXT,                                         \
      METAVAR,     OPT_##ID, opt::Option::KIND##Class,                         \
      PARAM,       FLAGS,    OPT_##GROUP,                                      \
      OPT_##ALIAS, ALIASARGS},
#include "Opts.inc"
#undef OPTION
};

class CvtResOptTable : public opt::OptTable {
public:
  CvtResOptTable() : OptTable(InfoTable, true) {}
};

static ExitOnError ExitOnErr;
}

int main(int argc_, const char *argv_[]) {
  sys::PrintStackTraceOnErrorSignal(argv_[0]);
  PrettyStackTraceProgram X(argc_, argv_);

  ExitOnErr.setBanner("llvm-cvtres: ");

  SmallVector<const char *, 256> argv;
  SpecificBumpPtrAllocator<char> ArgAllocator;
  ExitOnErr(errorCodeToError(sys::Process::GetArgumentVector(
      argv, makeArrayRef(argv_, argc_), ArgAllocator)));

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  CvtResOptTable T;
  unsigned MAI, MAC;
  ArrayRef<const char *> ArgsArr = makeArrayRef(argv_, argc_);
  opt::InputArgList InputArgs = T.ParseArgs(ArgsArr, MAI, MAC);

  if (InputArgs.hasArg(OPT_HELP))
    T.PrintHelp(outs(), "cvtres", "Resource Converter", false);

  return 0;
}
