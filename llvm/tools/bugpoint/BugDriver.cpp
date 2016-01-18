//===- BugDriver.cpp - Top-Level BugPoint class implementation ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class contains all of the shared state and information that is used by
// the BugPoint tool to track down errors in optimizations.  This class is the
// main driver class that invokes all sub-functionality.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "ToolRunner.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
using namespace llvm;

namespace llvm {
  Triple TargetTriple;
}

// Anonymous namespace to define command line options for debugging.
//
namespace {
  // Output - The user can specify a file containing the expected output of the
  // program.  If this filename is set, it is used as the reference diff source,
  // otherwise the raw input run through an interpreter is used as the reference
  // source.
  //
  cl::opt<std::string>
  OutputFile("output", cl::desc("Specify a reference program output "
                                "(for miscompilation detection)"));
}

/// setNewProgram - If we reduce or update the program somehow, call this method
/// to update bugdriver with it.  This deletes the old module and sets the
/// specified one as the current program.
void BugDriver::setNewProgram(Module *M) {
  delete Program;
  Program = M;
}


/// getPassesString - Turn a list of passes into a string which indicates the
/// command line options that must be passed to add the passes.
///
std::string llvm::getPassesString(const std::vector<std::string> &Passes) {
  std::string Result;
  for (unsigned i = 0, e = Passes.size(); i != e; ++i) {
    if (i) Result += " ";
    Result += "-";
    Result += Passes[i];
  }
  return Result;
}

BugDriver::BugDriver(const char *toolname, bool find_bugs,
                     unsigned timeout, unsigned memlimit, bool use_valgrind,
                     LLVMContext& ctxt)
  : Context(ctxt), ToolName(toolname), ReferenceOutputFile(OutputFile),
    Program(nullptr), Interpreter(nullptr), SafeInterpreter(nullptr),
    cc(nullptr), run_find_bugs(find_bugs), Timeout(timeout),
    MemoryLimit(memlimit), UseValgrind(use_valgrind) {}

BugDriver::~BugDriver() {
  delete Program;
  if (Interpreter != SafeInterpreter)
    delete Interpreter;
  delete SafeInterpreter;
  delete cc;
}

std::unique_ptr<Module> llvm::parseInputFile(StringRef Filename,
                                             LLVMContext &Ctxt) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Result = parseIRFile(Filename, Err, Ctxt);
  if (!Result) {
    Err.print("bugpoint", errs());
    return Result;
  }

  if (verifyModule(*Result, &errs())) {
    errs() << "bugpoint: " << Filename << ": error: input module is broken!\n";
    return std::unique_ptr<Module>();
  }

  // If we don't have an override triple, use the first one to configure
  // bugpoint, or use the host triple if none provided.
  if (TargetTriple.getTriple().empty()) {
    Triple TheTriple(Result->getTargetTriple());

    if (TheTriple.getTriple().empty())
      TheTriple.setTriple(sys::getDefaultTargetTriple());

    TargetTriple.setTriple(TheTriple.getTriple());
  }

  Result->setTargetTriple(TargetTriple.getTriple()); // override the triple
  return Result;
}

// This method takes the specified list of LLVM input files, attempts to load
// them, either as assembly or bitcode, then link them together. It returns
// true on failure (if, for example, an input bitcode file could not be
// parsed), and false on success.
//
bool BugDriver::addSources(const std::vector<std::string> &Filenames) {
  assert(!Program && "Cannot call addSources multiple times!");
  assert(!Filenames.empty() && "Must specify at least on input filename!");

  // Load the first input file.
  Program = parseInputFile(Filenames[0], Context).release();
  if (!Program) return true;

  outs() << "Read input file      : '" << Filenames[0] << "'\n";

  for (unsigned i = 1, e = Filenames.size(); i != e; ++i) {
    std::unique_ptr<Module> M = parseInputFile(Filenames[i], Context);
    if (!M.get()) return true;

    outs() << "Linking in input file: '" << Filenames[i] << "'\n";
    if (Linker::LinkModules(Program, M.get()))
      return true;
  }

  outs() << "*** All input ok\n";

  // All input files read successfully!
  return false;
}



/// run - The top level method that is invoked after all of the instance
/// variables are set up from command line arguments.
///
bool BugDriver::run(std::string &ErrMsg) {
  if (run_find_bugs) {
    // Rearrange the passes and apply them to the program. Repeat this process
    // until the user kills the program or we find a bug.
    return runManyPasses(PassesToRun, ErrMsg);
  }

  // If we're not running as a child, the first thing that we must do is
  // determine what the problem is. Does the optimization series crash the
  // compiler, or does it produce illegal code?  We make the top-level
  // decision by trying to run all of the passes on the input program,
  // which should generate a bitcode file.  If it does generate a bitcode
  // file, then we know the compiler didn't crash, so try to diagnose a
  // miscompilation.
  if (!PassesToRun.empty()) {
    outs() << "Running selected passes on program to test for crash: ";
    if (runPasses(Program, PassesToRun))
      return debugOptimizerCrash();
  }

  // Set up the execution environment, selecting a method to run LLVM bitcode.
  if (initializeExecutionEnvironment()) return true;

  // Test to see if we have a code generator crash.
  outs() << "Running the code generator to test for a crash: ";
  std::string Error;
  compileProgram(Program, &Error);
  if (!Error.empty()) {
    outs() << Error;
    return debugCodeGeneratorCrash(ErrMsg);
  }
  outs() << '\n';

  // Run the raw input to see where we are coming from.  If a reference output
  // was specified, make sure that the raw output matches it.  If not, it's a
  // problem in the front-end or the code generator.
  //
  bool CreatedOutput = false;
  if (ReferenceOutputFile.empty()) {
    outs() << "Generating reference output from raw program: ";
    if (!createReferenceFile(Program)) {
      return debugCodeGeneratorCrash(ErrMsg);
    }
    CreatedOutput = true;
  }

  // Make sure the reference output file gets deleted on exit from this
  // function, if appropriate.
  std::string ROF(ReferenceOutputFile);
  FileRemover RemoverInstance(ROF, CreatedOutput && !SaveTemps);

  // Diff the output of the raw program against the reference output.  If it
  // matches, then we assume there is a miscompilation bug and try to
  // diagnose it.
  outs() << "*** Checking the code generator...\n";
  bool Diff = diffProgram(Program, "", "", false, &Error);
  if (!Error.empty()) {
    errs() << Error;
    return debugCodeGeneratorCrash(ErrMsg);
  }
  if (!Diff) {
    outs() << "\n*** Output matches: Debugging miscompilation!\n";
    debugMiscompilation(&Error);
    if (!Error.empty()) {
      errs() << Error;
      return debugCodeGeneratorCrash(ErrMsg);
    }
    return false;
  }

  outs() << "\n*** Input program does not match reference diff!\n";
  outs() << "Debugging code generator problem!\n";
  bool Failure = debugCodeGenerator(&Error);
  if (!Error.empty()) {
    errs() << Error;
    return debugCodeGeneratorCrash(ErrMsg);
  }
  return Failure;
}

void llvm::PrintFunctionList(const std::vector<Function*> &Funcs) {
  unsigned NumPrint = Funcs.size();
  if (NumPrint > 10) NumPrint = 10;
  for (unsigned i = 0; i != NumPrint; ++i)
    outs() << " " << Funcs[i]->getName();
  if (NumPrint < Funcs.size())
    outs() << "... <" << Funcs.size() << " total>";
  outs().flush();
}

void llvm::PrintGlobalVariableList(const std::vector<GlobalVariable*> &GVs) {
  unsigned NumPrint = GVs.size();
  if (NumPrint > 10) NumPrint = 10;
  for (unsigned i = 0; i != NumPrint; ++i)
    outs() << " " << GVs[i]->getName();
  if (NumPrint < GVs.size())
    outs() << "... <" << GVs.size() << " total>";
  outs().flush();
}
