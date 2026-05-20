//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is an example tool used for testing the daemon driver functionality.
/// The tool reads input from stdin character-by-character and prints
/// uppercase letters on `stderr` and everything else on `stdout`. It
/// returns the number of times a letter was printed to `stderr`.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DaemonDriver.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolInterface.h"
#include <cctype>
#include <cstdlib>
#include <filesystem>
using namespace llvm;

// Global variable used to check that `resetState` is properly called between
// invocations.
constexpr int PersistentStateInit = 0;
static int PersistentState = PersistentStateInit;

static cl::opt<bool> SeparateLowercaseInstead(
    "separate-lowercase-instead",
    cl::desc("This option is used to test that command line arguments are "
             "passed correctly by the daemon driver."),
    cl::init(false));

static cl::opt<bool>
    PrintCurrentDirectory("print-current-directory",
                          cl::desc("This option is used to test the ability of "
                                   "the daemon to change working directory."),
                          cl::init(false));

static cl::opt<bool>
    ExitProcess("exit-process",
                cl::desc("This option is used to test the behaviour of "
                         "the daemon when the tool itself exits."),
                cl::init(false));

class ExampleTool : public LLVMTool {
public:
  virtual int run(int Argc, char **Argv,
                  const StandardInputSource &InputSource) override {
    // Make sure that `PersistentState` has been reset.
    assert(PersistentState == PersistentStateInit &&
           "Persistent state should have been reset.");
    PersistentState = 1;

    cl::ParseCommandLineOptions(Argc, Argv);

    if (PrintCurrentDirectory)
      llvm::outs() << std::filesystem::current_path().string() << "\n";

    if (ExitProcess)
      std::exit(0);

    // Read standard input or get it from `StdinOverride`.
    std::unique_ptr<MemoryBuffer> InputContent;
    ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
        InputSource.getFileOrInput("-", /*IsText=*/false);
    if (const std::error_code Err = FileOrErr.getError()) {
      errs() << "Error reading standard input: " << Err.message() << "\n";
      return 1;
    }
    InputContent.swap(FileOrErr.get());

    int StderrCount = 0;
    for (const char Char : InputContent->getBuffer()) {
      if (SeparateLowercaseInstead ? islower(Char) : isupper(Char)) {
        errs() << Char;
        errs().flush();
        StderrCount += 1;
      } else {
        outs() << Char;
        outs().flush();
      }
    }

    return StderrCount;
  };

  virtual void resetState() override {
    PersistentState = PersistentStateInit;
    cl::ResetAllOptionOccurrences();
  }
};

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  ExampleTool Tool;
  return runWithDaemonSupport(Tool, argc, argv);
}
