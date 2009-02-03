//===- bugpoint.cpp - The LLVM Bugpoint utility ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is an automated compiler debugger tool.  It is used to narrow
// down miscompilations and crash problems to a specific pass in the compiler,
// and the specific Module or Function input that is causing the problem.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "ToolRunner.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/PassNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/System/Process.h"
#include "llvm/System/Signals.h"
#include "llvm/LinkAllVMCore.h"
#include <iostream>
using namespace llvm;

// AsChild - Specifies that this invocation of bugpoint is being generated
// from a parent process. It is not intended to be used by users so the 
// option is hidden.
static cl::opt<bool> 
AsChild("as-child", cl::desc("Run bugpoint as child process"), 
        cl::ReallyHidden);
          
static cl::opt<bool> 
FindBugs("find-bugs", cl::desc("Run many different optimization sequences "
                               "on program to find bugs"), cl::init(false));

static cl::list<std::string>
InputFilenames(cl::Positional, cl::OneOrMore,
               cl::desc("<input llvm ll/bc files>"));

static cl::opt<unsigned>
TimeoutValue("timeout", cl::init(300), cl::value_desc("seconds"),
             cl::desc("Number of seconds program is allowed to run before it "
                      "is killed (default is 300s), 0 disables timeout"));

static cl::opt<unsigned>
MemoryLimit("mlimit", cl::init(100), cl::value_desc("MBytes"),
             cl::desc("Maximum amount of memory to use. 0 disables check."));

// The AnalysesList is automatically populated with registered Passes by the
// PassNameParser.
//
static cl::list<const PassInfo*, bool, PassNameParser>
PassList(cl::desc("Passes available:"), cl::ZeroOrMore);

/// BugpointIsInterrupted - Set to true when the user presses ctrl-c.
bool llvm::BugpointIsInterrupted = false;

static void BugpointInterruptFunction() {
  BugpointIsInterrupted = true;
}

int main(int argc, char **argv) {
  llvm_shutdown_obj X;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv,
                              "LLVM automatic testcase reducer. See\nhttp://"
                              "llvm.org/docs/CommandGuide/bugpoint.html"
                              " for more information.\n");
  sys::PrintStackTraceOnErrorSignal();
  sys::SetInterruptFunction(BugpointInterruptFunction);
  
  BugDriver D(argv[0], AsChild, FindBugs, TimeoutValue, MemoryLimit);
  if (D.addSources(InputFilenames)) return 1;
  D.addPasses(PassList.begin(), PassList.end());

  // Bugpoint has the ability of generating a plethora of core files, so to
  // avoid filling up the disk, we prevent it
  sys::Process::PreventCoreFiles();

  try {
    return D.run();
  } catch (ToolExecutionError &TEE) {
    std::cerr << "Tool execution error: " << TEE.what() << '\n';
  } catch (const std::string& msg) {
    std::cerr << argv[0] << ": " << msg << "\n";
  } catch (...) {
    std::cerr << "Whoops, an exception leaked out of bugpoint.  "
              << "This is a bug in bugpoint!\n";
  }
  return 1;
}
