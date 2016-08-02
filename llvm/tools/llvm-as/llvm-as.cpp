//===--- llvm-as.cpp - The low-level LLVM assembler -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This utility may be invoked in the following manner:
//   llvm-as --help         - Output information about command line switches
//   llvm-as [options]      - Read LLVM asm from stdin, write bitcode to stdout
//   llvm-as [options] x.ll - Read LLVM asm from the x.ll file, write bitcode
//                            to the x.bc file.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include <memory>
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input .llvm file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Enable binary output on terminals"));

static cl::opt<bool>
DisableOutput("disable-output", cl::desc("Disable output"), cl::init(false));

static cl::opt<bool>
EmitFunctionSummary("function-summary", cl::desc("Emit function summary index"),
                    cl::init(false));

static cl::opt<bool>
DumpAsm("d", cl::desc("Print assembly as parsed"), cl::Hidden);

static cl::opt<bool>
DisableVerify("disable-verify", cl::Hidden,
              cl::desc("Do not run verifier on input LLVM (dangerous!)"));

static cl::opt<bool> PreserveBitcodeUseListOrder(
    "preserve-bc-uselistorder",
    cl::desc("Preserve use-list order when writing LLVM bitcode."),
    cl::init(true), cl::Hidden);

static void WriteOutputFile(const Module *M) {
  // Infer the output filename if needed.
  if (OutputFilename.empty()) {
    if (InputFilename == "-") {
      OutputFilename = "-";
    } else {
      StringRef IFN = InputFilename;
      OutputFilename = (IFN.endswith(".ll") ? IFN.drop_back(3) : IFN).str();
      OutputFilename += ".bc";
    }
  }

  std::error_code EC;
  std::unique_ptr<tool_output_file> Out(
      new tool_output_file(OutputFilename, EC, sys::fs::F_None));
  if (EC) {
    errs() << EC.message() << '\n';
    exit(1);
  }

  if (Force || !CheckBitcodeOutputToConsole(Out->os(), true))
    WriteBitcodeToFile(M, Out->os(), PreserveBitcodeUseListOrder,
                       EmitFunctionSummary);

  // Declare success.
  Out->keep();
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm .ll -> .bc assembler\n");

  // Parse the file now...
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyFile(InputFilename, Err, Context);
  if (!M.get()) {
    Err.print(argv[0], errs());
    return 1;
  }

  if (!DisableVerify) {
    std::string ErrorStr;
    raw_string_ostream OS(ErrorStr);
    if (verifyModule(*M.get(), &OS)) {
      errs() << argv[0]
             << ": assembly parsed, but does not verify as correct!\n";
      errs() << OS.str();
      return 1;
    }
  }

  if (DumpAsm) errs() << "Here's the assembly:\n" << *M.get();

  if (!DisableOutput)
    WriteOutputFile(M.get());

  return 0;
}
