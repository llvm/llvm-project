//===- llvm-db.cpp - LLVM Debugger ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility implements a simple text-mode front-end to the LLVM debugger
// library.
//
//===----------------------------------------------------------------------===//

#include "CLIDebugger.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/System/Signals.h"
#include <iostream>

using namespace llvm;

namespace {
  // Command line options for specifying the program to debug and options to use
  cl::opt<std::string>
  InputFile(cl::desc("<program>"), cl::Positional, cl::init(""));

  cl::list<std::string>
  InputArgs("args", cl::Positional, cl::desc("<program and arguments>"),
            cl::ZeroOrMore);

  // Command line options to control various directory related stuff
  cl::list<std::string>
  SourceDirectories("directory", cl::value_desc("directory"),
                    cl::desc("Add directory to the search for source files"));
  cl::alias SDA("d", cl::desc("Alias for --directory"),
                cl::aliasopt(SourceDirectories));

  cl::opt<std::string>
  WorkingDirectory("cd", cl::desc("Use directory as current working directory"),
                   cl::value_desc("directory"));

  // Command line options specific to the llvm-db debugger driver
  cl::opt<bool> Quiet("quiet", cl::desc("Do not print introductory messages"));
  cl::alias QA1("silent", cl::desc("Alias for -quiet"), cl::aliasopt(Quiet));
  cl::alias QA2("q", cl::desc("Alias for -quiet"), cl::aliasopt(Quiet));
}

//===----------------------------------------------------------------------===//
// main Driver function
//
int main(int argc, char **argv, char * const *envp) {
  std::cout << "NOTE: llvm-db is known useless right now.\n";
  try {
    cl::ParseCommandLineOptions(argc, argv,
                                " llvm source-level debugger\n");
    sys::PrintStackTraceOnErrorSignal();

    if (!Quiet)
      std::cout << "llvm-db: The LLVM source-level debugger\n";

    // Merge Inputfile and InputArgs into the InputArgs list...
    if (!InputFile.empty() && InputArgs.empty())
      InputArgs.push_back(InputFile);

    // Create the CLI debugger...
    CLIDebugger D;

    // Initialize the debugger with the command line options we read...
    Debugger &Dbg = D.getDebugger();

    // Initialize the debugger environment.
    Dbg.initializeEnvironment(envp);
    Dbg.setWorkingDirectory(WorkingDirectory);
    for (unsigned i = 0, e = SourceDirectories.size(); i != e; ++i)
      D.addSourceDirectory(SourceDirectories[i]);

    if (!InputArgs.empty()) {
      try {
        D.fileCommand(InputArgs[0]);
      } catch (const std::string &Error) {
        std::cout << "Error: " << Error << "\n";
      }

      Dbg.setProgramArguments(InputArgs.begin()+1, InputArgs.end());
    }

    // Now that we have initialized the debugger, run it.
    return D.run();
  } catch (const std::string& msg) {
    std::cerr << argv[0] << ": " << msg << "\n";
  } catch (...) {
    std::cerr << argv[0] << ": Unexpected unknown exception occurred.\n";
  }
  return 1;
}
