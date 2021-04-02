//===- llvm-reduce.cpp - The LLVM Delta Reduction utility -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program tries to reduce an IR test case for a given interesting-ness
// test. It runs multiple delta debugging passes in order to minimize the input
// file. It's worth noting that this is a part of the bugpoint redesign
// proposal, and thus a *temporary* tool that will eventually be integrated
// into the bugpoint tool itself.
//
//===----------------------------------------------------------------------===//

#include "DeltaManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>
#include <vector>

using namespace llvm;

static cl::OptionCategory Options("llvm-reduce options");

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden,
                          cl::cat(Options));
static cl::opt<bool> Version("v", cl::desc("Alias for -version"), cl::Hidden,
                             cl::cat(Options));

static cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input llvm ll/bc file>"),
                                          cl::cat(Options));

static cl::opt<std::string>
    TestFilename("test", cl::Required,
                 cl::desc("Name of the interesting-ness test to be run"),
                 cl::cat(Options));

static cl::list<std::string>
    TestArguments("test-arg", cl::ZeroOrMore,
                  cl::desc("Arguments passed onto the interesting-ness test"),
                  cl::cat(Options));

static cl::opt<std::string>
    OutputFilename("output",
                   cl::desc("Specify the output file. default: reduced.ll"));
static cl::alias OutputFileAlias("o", cl::desc("Alias for -output"),
                                 cl::aliasopt(OutputFilename),
                                 cl::cat(Options));

static cl::opt<bool>
    ReplaceInput("in-place",
                 cl::desc("WARNING: This option will replace your input file "
                          "with the reduced version!"),
                 cl::cat(Options));

// Parses IR into a Module and verifies it
static std::unique_ptr<Module> parseInputFile(StringRef Filename,
                                              LLVMContext &Ctxt) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Result = parseIRFile(Filename, Err, Ctxt);
  if (!Result) {
    Err.print("llvm-reduce", errs());
    return Result;
  }

  if (verifyModule(*Result, &errs())) {
    errs() << "Error: " << Filename << " - input module is broken!\n";
    return std::unique_ptr<Module>();
  }

  return Result;
}

void writeOutput(Module *M, StringRef Message) {
  if (ReplaceInput) // In-place
    OutputFilename = InputFilename.c_str();
  else if (OutputFilename.empty() || OutputFilename == "-")
    OutputFilename = "reduced.ll";

  std::error_code EC;
  raw_fd_ostream Out(OutputFilename, EC);
  if (EC) {
    errs() << "Error opening output file: " << EC.message() << "!\n";
    exit(1);
  }
  M->print(Out, /*AnnotationWriter=*/nullptr);
  errs() << Message << OutputFilename << "\n";
}

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);

  cl::ParseCommandLineOptions(Argc, Argv, "LLVM automatic testcase reducer.\n");

  LLVMContext Context;
  std::unique_ptr<Module> OriginalProgram =
      parseInputFile(InputFilename, Context);

  // Initialize test environment
  TestRunner Tester(TestFilename, TestArguments);
  Tester.setProgram(std::move(OriginalProgram));

  // Try to reduce code
  runDeltaPasses(Tester);

  if (!Tester.getProgram()) {
    errs() << "\nCouldnt reduce input :/\n";
  } else {
    // Print reduced file to STDOUT
    if (OutputFilename == "-")
      Tester.getProgram()->print(outs(), nullptr);
    else
      writeOutput(Tester.getProgram(), "\nDone reducing! Reduced testcase: ");
  }

  return 0;
}
