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

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/WindowsResource.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/BinaryStreamError.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>

using namespace llvm;
using namespace object;

namespace {

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OPT_##ID,
#include "Opts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Opts.inc"
#undef PREFIX

static const opt::OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {                                                                            \
      PREFIX,      NAME,      HELPTEXT,                                        \
      METAVAR,     OPT_##ID,  opt::Option::KIND##Class,                        \
      PARAM,       FLAGS,     OPT_##GROUP,                                     \
      OPT_##ALIAS, ALIASARGS, VALUES},
#include "Opts.inc"
#undef OPTION
};

class CvtResOptTable : public opt::OptTable {
public:
  CvtResOptTable() : OptTable(InfoTable, true) {}
};
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

template <typename T> T error(Expected<T> EC) {
  if (!EC)
    error(EC.takeError());
  return std::move(EC.get());
}

int main(int Argc, const char **Argv) {
  InitLLVM X(Argc, Argv);

  CvtResOptTable T;
  unsigned MAI, MAC;
  ArrayRef<const char *> ArgsArr = makeArrayRef(Argv + 1, Argc - 1);
  opt::InputArgList InputArgs = T.ParseArgs(ArgsArr, MAI, MAC);

  if (InputArgs.hasArg(OPT_HELP)) {
    T.PrintHelp(outs(), "cvtres", "Resource Converter", false);
    return 0;
  }

  bool Verbose = InputArgs.hasArg(OPT_VERBOSE);

  COFF::MachineTypes MachineType;

  if (InputArgs.hasArg(OPT_MACHINE)) {
    std::string MachineString = InputArgs.getLastArgValue(OPT_MACHINE).upper();
    MachineType = StringSwitch<COFF::MachineTypes>(MachineString)
                      .Case("ARM", COFF::IMAGE_FILE_MACHINE_ARMNT)
                      .Case("ARM64", COFF::IMAGE_FILE_MACHINE_ARM64)
                      .Case("X64", COFF::IMAGE_FILE_MACHINE_AMD64)
                      .Case("X86", COFF::IMAGE_FILE_MACHINE_I386)
                      .Default(COFF::IMAGE_FILE_MACHINE_UNKNOWN);
    if (MachineType == COFF::IMAGE_FILE_MACHINE_UNKNOWN)
      reportError("Unsupported machine architecture");
  } else {
    if (Verbose)
      outs() << "Machine architecture not specified; assumed X64.\n";
    MachineType = COFF::IMAGE_FILE_MACHINE_AMD64;
  }

  std::vector<std::string> InputFiles = InputArgs.getAllArgValues(OPT_INPUT);

  if (InputFiles.size() == 0) {
    reportError("No input file specified.\n");
  }

  SmallString<128> OutputFile;

  if (InputArgs.hasArg(OPT_OUT)) {
    OutputFile = InputArgs.getLastArgValue(OPT_OUT);
  } else {
    OutputFile = sys::path::filename(StringRef(InputFiles[0]));
    sys::path::replace_extension(OutputFile, ".obj");
  }

  if (Verbose) {
    outs() << "Machine: ";
    switch (MachineType) {
    case COFF::IMAGE_FILE_MACHINE_ARM64:
      outs() << "ARM64\n";
      break;
    case COFF::IMAGE_FILE_MACHINE_ARMNT:
      outs() << "ARM\n";
      break;
    case COFF::IMAGE_FILE_MACHINE_I386:
      outs() << "X86\n";
      break;
    default:
      outs() << "X64\n";
    }
  }

  WindowsResourceParser Parser;

  for (const auto &File : InputFiles) {
    Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(File);
    if (!BinaryOrErr)
      reportError(File, errorToErrorCode(BinaryOrErr.takeError()));

    Binary &Binary = *BinaryOrErr.get().getBinary();

    WindowsResource *RF = dyn_cast<WindowsResource>(&Binary);
    if (!RF)
      reportError(File + ": unrecognized file format.\n");

    if (Verbose) {
      int EntryNumber = 0;
      ResourceEntryRef Entry = error(RF->getHeadEntry());
      bool End = false;
      while (!End) {
        error(Entry.moveNext(End));
        EntryNumber++;
      }
      outs() << "Number of resources: " << EntryNumber << "\n";
    }

    error(Parser.parse(RF));
  }

  if (Verbose) {
    Parser.printTree(outs());
  }

  std::unique_ptr<MemoryBuffer> OutputBuffer =
      error(llvm::object::writeWindowsResourceCOFF(MachineType, Parser));
  auto FileOrErr =
      FileOutputBuffer::create(OutputFile, OutputBuffer->getBufferSize());
  if (!FileOrErr)
    reportError(OutputFile, errorToErrorCode(FileOrErr.takeError()));
  std::unique_ptr<FileOutputBuffer> FileBuffer = std::move(*FileOrErr);
  std::copy(OutputBuffer->getBufferStart(), OutputBuffer->getBufferEnd(),
            FileBuffer->getBufferStart());
  error(FileBuffer->commit());

  if (Verbose) {
    Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(OutputFile);
    if (!BinaryOrErr)
      reportError(OutputFile, errorToErrorCode(BinaryOrErr.takeError()));
    Binary &Binary = *BinaryOrErr.get().getBinary();
    ScopedPrinter W(errs());
    W.printBinaryBlock("Output File Raw Data", Binary.getData());
  }

  return 0;
}
