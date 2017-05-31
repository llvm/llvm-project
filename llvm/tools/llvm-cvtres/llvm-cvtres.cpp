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

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/WindowsResource.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/BinaryStreamError.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace object;

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

LLVM_ATTRIBUTE_NORETURN void reportError(Twine Msg) {
  errs() << Msg;
  exit(1);
}

static void reportError(StringRef Input, std::error_code EC) {
  reportError(Twine(Input) + ": " + EC.message() + ".\n");
}

void error(std::error_code EC) {
  if (!EC)
    return;
  reportError(EC.message() + ".\n");
}

void error(Error EC) {
  if (!EC)
    return;
  handleAllErrors(std::move(EC),
                  [&](const ErrorInfoBase &EI) { reportError(EI.message()); });
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
  ArrayRef<const char *> ArgsArr = makeArrayRef(argv_ + 1, argc_);
  opt::InputArgList InputArgs = T.ParseArgs(ArgsArr, MAI, MAC);

  if (InputArgs.hasArg(OPT_HELP)) {
    T.PrintHelp(outs(), "cvtres", "Resource Converter", false);
    return 0;
  }

  machine Machine;

  if (InputArgs.hasArg(OPT_MACHINE)) {
    std::string MachineString = InputArgs.getLastArgValue(OPT_MACHINE).upper();
    Machine = StringSwitch<machine>(MachineString)
                  .Case("ARM", machine::ARM)
                  .Case("X64", machine::X64)
                  .Case("X86", machine::X86)
                  .Default(machine::UNKNOWN);
    if (Machine == machine::UNKNOWN)
      reportError("Unsupported machine architecture");
  } else {
    outs() << "Machine architecture not specified; assumed X64.\n";
    Machine = machine::X64;
  }

  std::vector<std::string> InputFiles = InputArgs.getAllArgValues(OPT_INPUT);

  if (InputFiles.size() == 0) {
    reportError("No input file specified.\n");
  }

  SmallString<128> OutputFile;

  if (InputArgs.hasArg(OPT_OUT)) {
    OutputFile = InputArgs.getLastArgValue(OPT_OUT);
  } else {
    OutputFile = StringRef(InputFiles[0]);
    llvm::sys::path::replace_extension(OutputFile, ".obj");
  }

  outs() << "Machine: ";
  switch (Machine) {
  case machine::ARM:
    outs() << "ARM\n";
    break;
  case machine::X86:
    outs() << "X86\n";
    break;
  default:
    outs() << "X64\n";
  }

  WindowsResourceParser Parser;

  for (const auto &File : InputFiles) {
    Expected<object::OwningBinary<object::Binary>> BinaryOrErr =
        object::createBinary(File);
    if (!BinaryOrErr)
      reportError(File, errorToErrorCode(BinaryOrErr.takeError()));

    Binary &Binary = *BinaryOrErr.get().getBinary();

    WindowsResource *RF = dyn_cast<WindowsResource>(&Binary);
    if (!RF)
      reportError(File + ": unrecognized file format.\n");

    int EntryNumber = 0;
    Expected<ResourceEntryRef> EntryOrErr = RF->getHeadEntry();
    if (!EntryOrErr)
      error(EntryOrErr.takeError());
    ResourceEntryRef Entry = EntryOrErr.get();
    bool End = false;
    while (!End) {
      error(Entry.moveNext(End));
      EntryNumber++;
    }
    outs() << "Number of resources: " << EntryNumber << "\n";

    error(Parser.parse(RF));
  }

  Parser.printTree();

  return 0;
}
