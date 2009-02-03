//===- lli.cpp - LLVM Interpreter / Dynamic compiler ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility provides a simple wrapper around the LLVM Execution Engines,
// which allow the direct execution of LLVM programs through a Just-In-Time
// compiler, or through an intepreter if no JIT is available for this platform.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Type.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/System/Process.h"
#include "llvm/System/Signals.h"
#include <iostream>
#include <cerrno>
using namespace llvm;

namespace {
  cl::opt<std::string>
  InputFile(cl::desc("<input bitcode>"), cl::Positional, cl::init("-"));

  cl::list<std::string>
  InputArgv(cl::ConsumeAfter, cl::desc("<program arguments>..."));

  cl::opt<bool> ForceInterpreter("force-interpreter",
                                 cl::desc("Force interpretation: disable JIT"),
                                 cl::init(false));

  cl::opt<bool> Fast("fast", 
                     cl::desc("Generate code quickly, "
                              "potentially sacrificing code quality"),
                     cl::init(false));

  cl::opt<std::string>
  TargetTriple("mtriple", cl::desc("Override target triple for module"));

  cl::opt<std::string>
  EntryFunc("entry-function",
            cl::desc("Specify the entry function (default = 'main') "
                     "of the executable"),
            cl::value_desc("function"),
            cl::init("main"));
  
  cl::opt<std::string>
  FakeArgv0("fake-argv0",
            cl::desc("Override the 'argv[0]' value passed into the executing"
                     " program"), cl::value_desc("executable"));
  
  cl::opt<bool>
  DisableCoreFiles("disable-core-files", cl::Hidden,
                   cl::desc("Disable emission of core files if possible"));

  cl::opt<bool>
  NoLazyCompilation("disable-lazy-compilation",
                  cl::desc("Disable JIT lazy compilation"),
                  cl::init(false));
}

static ExecutionEngine *EE = 0;

static void do_shutdown() {
  delete EE;
  llvm_shutdown();
}

//===----------------------------------------------------------------------===//
// main Driver function
//
int main(int argc, char **argv, char * const *envp) {
  atexit(do_shutdown);  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv,
                              "llvm interpreter & dynamic compiler\n");
  sys::PrintStackTraceOnErrorSignal();

  // If the user doesn't want core files, disable them.
  if (DisableCoreFiles)
    sys::Process::PreventCoreFiles();
  
  // Load the bitcode...
  std::string ErrorMsg;
  ModuleProvider *MP = NULL;
  if (MemoryBuffer *Buffer = MemoryBuffer::getFileOrSTDIN(InputFile,&ErrorMsg)) {
    MP = getBitcodeModuleProvider(Buffer, &ErrorMsg);
    if (!MP) delete Buffer;
  }
  
  if (!MP) {
    std::cerr << argv[0] << ": error loading program '" << InputFile << "': "
              << ErrorMsg << "\n";
    exit(1);
  }

  // Get the module as the MP could go away once EE takes over.
  Module *Mod = NoLazyCompilation
    ? MP->materializeModule(&ErrorMsg) : MP->getModule();
  if (!Mod) {
    std::cerr << argv[0] << ": bitcode didn't read correctly.\n";
    std::cerr << "Reason: " << ErrorMsg << "\n";
    exit(1);
  }

  // If we are supposed to override the target triple, do so now.
  if (!TargetTriple.empty())
    Mod->setTargetTriple(TargetTriple);

  EE = ExecutionEngine::create(MP, ForceInterpreter, &ErrorMsg, Fast);
  if (!EE && !ErrorMsg.empty()) {
    std::cerr << argv[0] << ":error creating EE: " << ErrorMsg << "\n";
    exit(1);
  }

  if (NoLazyCompilation)
    EE->DisableLazyCompilation();

  // If the user specifically requested an argv[0] to pass into the program,
  // do it now.
  if (!FakeArgv0.empty()) {
    InputFile = FakeArgv0;
  } else {
    // Otherwise, if there is a .bc suffix on the executable strip it off, it
    // might confuse the program.
    if (InputFile.rfind(".bc") == InputFile.length() - 3)
      InputFile.erase(InputFile.length() - 3);
  }

  // Add the module's name to the start of the vector of arguments to main().
  InputArgv.insert(InputArgv.begin(), InputFile);

  // Call the main function from M as if its signature were:
  //   int main (int argc, char **argv, const char **envp)
  // using the contents of Args to determine argc & argv, and the contents of
  // EnvVars to determine envp.
  //
  Function *EntryFn = Mod->getFunction(EntryFunc);
  if (!EntryFn) {
    std::cerr << '\'' << EntryFunc << "\' function not found in module.\n";
    return -1;
  }

  // If the program doesn't explicitly call exit, we will need the Exit 
  // function later on to make an explicit call, so get the function now. 
  Constant *Exit = Mod->getOrInsertFunction("exit", Type::VoidTy,
                                                        Type::Int32Ty, NULL);
  
  // Reset errno to zero on entry to main.
  errno = 0;
 
  // Run static constructors.
  EE->runStaticConstructorsDestructors(false);

  if (NoLazyCompilation) {
    for (Module::iterator I = Mod->begin(), E = Mod->end(); I != E; ++I) {
      Function *Fn = &*I;
      if (Fn != EntryFn && !Fn->isDeclaration())
        EE->getPointerToFunction(Fn);
    }
  }

  // Run main.
  int Result = EE->runFunctionAsMain(EntryFn, InputArgv, envp);

  // Run static destructors.
  EE->runStaticConstructorsDestructors(true);
  
  // If the program didn't call exit explicitly, we should call it now. 
  // This ensures that any atexit handlers get called correctly.
  if (Function *ExitF = dyn_cast<Function>(Exit)) {
    std::vector<GenericValue> Args;
    GenericValue ResultGV;
    ResultGV.IntVal = APInt(32, Result);
    Args.push_back(ResultGV);
    EE->runFunction(ExitF, Args);
    std::cerr << "ERROR: exit(" << Result << ") returned!\n";
    abort();
  } else {
    std::cerr << "ERROR: exit defined with wrong prototype!\n";
    abort();
  }
}
