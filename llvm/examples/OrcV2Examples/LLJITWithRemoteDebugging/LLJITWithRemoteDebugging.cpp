//===--- LLJITWithRemoteDebugging.cpp - LLJIT targeting a child process ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This example shows how to use LLJIT and JITLink for out-of-process execution
// with debug support.  A few notes beforehand:
//
//  * Debuggers must implement the GDB JIT interface (gdb, udb, lldb 12+).
//  * Debug support is currently limited to ELF on x86-64 platforms that run
//    Unix-like systems.
//  * There is a test for this example and it ships an IR file that is prepared
//    for the instructions below.
//
//
// The following command line session provides a complete walkthrough of the
// feature using LLDB 12:
//
// [Terminal 1] Prepare a debuggable out-of-process JIT session:
//
//    > cd llvm-project/build
//    > ninja LLJITWithRemoteDebugging llvm-jitlink-executor
//    > cp ../llvm/test/Examples/OrcV2Examples/Inputs/argc_sub1_elf.ll .
//    > bin/LLJITWithRemoteDebugging --wait-for-debugger argc_sub1_elf.ll
//    Found out-of-process executor: bin/llvm-jitlink-executor
//    Launched executor in subprocess: 65535
//    Attach a debugger and press any key to continue.
//
//
// [Terminal 2] Attach a debugger to the child process:
//
//    (lldb) log enable lldb jit
//    (lldb) settings set plugin.jit-loader.gdb.enable on
//    (lldb) settings set target.source-map Inputs/ \
//             /path/to/llvm-project/llvm/test/Examples/OrcV2Examples/Inputs/
//    (lldb) attach -p 65535
//     JITLoaderGDB::SetJITBreakpoint looking for JIT register hook
//     JITLoaderGDB::SetJITBreakpoint setting JIT breakpoint
//    Process 65535 stopped
//    (lldb) b sub1
//    Breakpoint 1: no locations (pending).
//    WARNING:  Unable to resolve breakpoint to any actual locations.
//    (lldb) c
//    Process 65535 resuming
//
//
// [Terminal 1] Press a key to start code generation and execution:
//
//    Parsed input IR code from: argc_sub1_elf.ll
//    Initialized LLJIT for remote executor
//    Running: argc_sub1_elf.ll
//
//
// [Terminal 2] Breakpoint hits; we change the argc value from 1 to 42:
//
//    (lldb)  JITLoaderGDB::JITDebugBreakpointHit hit JIT breakpoint
//     JITLoaderGDB::ReadJITDescriptorImpl registering JIT entry at 0x106b34000
//    1 location added to breakpoint 1
//    Process 65535 stopped
//    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
//        frame #0: JIT(0x106b34000)`sub1(x=1) at argc_sub1.c:1:28
//    -> 1   	int sub1(int x) { return x - 1; }
//       2   	int main(int argc, char **argv) { return sub1(argc); }
//    (lldb) p x
//    (int) $0 = 1
//    (lldb) expr x = 42
//    (int) $1 = 42
//    (lldb) c
//
//
// [Terminal 1] Example output reflects the modified value:
//
//    Exit code: 41
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "../ExampleModules.h"
#include "RemoteJITUtils.h"

#include <memory>
#include <string>

using namespace llvm;
using namespace llvm::orc;

// The LLVM IR file to run.
static cl::list<std::string> InputFiles(cl::Positional, cl::OneOrMore,
                                        cl::desc("<input files>"));

// Command line arguments to pass to the JITed main function.
static cl::list<std::string> InputArgv("args", cl::Positional,
                                       cl::desc("<program arguments>..."),
                                       cl::ZeroOrMore, cl::PositionalEatsArgs);

// Given paths must exist on the remote target.
static cl::list<std::string>
    Dylibs("dlopen", cl::desc("Dynamic libraries to load before linking"),
           cl::value_desc("filename"), cl::ZeroOrMore);

// File path of the executable to launch for execution in a child process.
// Inter-process communication will go through stdin/stdout pipes.
static cl::opt<std::string>
    OOPExecutor("executor", cl::desc("Set the out-of-process executor"),
                cl::value_desc("filename"));

// Network address of a running executor process that we can connected through a
// TCP socket. It may run locally or on a remote machine.
static cl::opt<std::string> OOPExecutorConnect(
    "connect",
    cl::desc("Connect to an out-of-process executor through a TCP socket"),
    cl::value_desc("<hostname>:<port>"));

// Give the user a chance to connect a debugger. Once we connected the executor
// process, wait for the user to press a key (and print out its PID if it's a
// child process).
static cl::opt<bool>
    WaitForDebugger("wait-for-debugger",
                    cl::desc("Wait for user input before entering JITed code"),
                    cl::init(false));

ExitOnError ExitOnErr;

static std::unique_ptr<JITLinkExecutor> connectExecutor(const char *Argv0) {
  // Connect to a running out-of-process executor through a TCP socket.
  if (!OOPExecutorConnect.empty()) {
    std::unique_ptr<TCPSocketJITLinkExecutor> Exec =
        ExitOnErr(JITLinkExecutor::ConnectTCPSocket(OOPExecutorConnect,
                                                    std::ref(ExitOnErr)));

    outs() << "Connected to executor at " << OOPExecutorConnect << "\n";
    if (WaitForDebugger) {
      outs() << "Attach a debugger and press any key to continue.\n";
      fflush(stdin);
      getchar();
    }

    return std::move(Exec);
  }

  // Launch a out-of-process executor locally in a child process.
  std::unique_ptr<ChildProcessJITLinkExecutor> Exec = ExitOnErr(
      OOPExecutor.empty() ? JITLinkExecutor::FindLocal(Argv0)
                          : JITLinkExecutor::CreateLocal(OOPExecutor));

  outs() << "Found out-of-process executor: " << Exec->getPath() << "\n";

  ExitOnErr(Exec->launch(std::ref(ExitOnErr)));
  if (WaitForDebugger) {
    outs() << "Launched executor in subprocess: " << Exec->getPID() << "\n"
           << "Attach a debugger and press any key to continue.\n";
    fflush(stdin);
    getchar();
  }

  return std::move(Exec);
}

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  ExitOnErr.setBanner(std::string(argv[0]) + ": ");
  cl::ParseCommandLineOptions(argc, argv, "LLJITWithRemoteDebugging");

  // Launch/connect the out-of-process executor.
  std::unique_ptr<JITLinkExecutor> Executor = connectExecutor(argv[0]);

  // Load the given IR files.
  std::vector<ThreadSafeModule> TSMs;
  for (const std::string &Path : InputFiles) {
    outs() << "Parsing input IR code from: " << Path << "\n";
    TSMs.push_back(ExitOnErr(parseExampleModuleFromFile(Path)));
  }

  StringRef TT;
  StringRef MainModuleName;
  TSMs.front().withModuleDo([&MainModuleName, &TT](Module &M) {
    MainModuleName = M.getName();
    TT = M.getTargetTriple();
  });

  for (const ThreadSafeModule &TSM : TSMs)
    ExitOnErr(TSM.withModuleDo([TT, MainModuleName](Module &M) -> Error {
      if (M.getTargetTriple() != TT)
        return make_error<StringError>(
            formatv("Different target triples in input files:\n"
                    "  '{0}' in '{1}'\n  '{2}' in '{3}'",
                    TT, MainModuleName, M.getTargetTriple(), M.getName()),
            inconvertibleErrorCode());
      return Error::success();
    }));

  // Create a target machine that matches the input triple.
  JITTargetMachineBuilder JTMB((Triple(TT)));
  JTMB.setCodeModel(CodeModel::Small);
  JTMB.setRelocationModel(Reloc::PIC_);

  // Create LLJIT and destroy it before disconnecting the target process.
  {
    std::unique_ptr<ExecutionSession> ES = Executor->startSession();

    outs() << "Initializing LLJIT for remote executor\n";
    auto J = ExitOnErr(LLJITBuilder()
                           .setExecutionSession(std::move(ES))
                           .setJITTargetMachineBuilder(std::move(JTMB))
                           .setObjectLinkingLayerCreator(std::ref(*Executor))
                           .create());

    // Add plugin for debug support.
    ExitOnErr(Executor->addDebugSupport(J->getObjLinkingLayer()));

    // Load required shared libraries on the remote target and add a generator
    // for each of it, so the compiler can lookup their symbols.
    for (const std::string &Path : Dylibs)
      J->getMainJITDylib().addGenerator(ExitOnErr(Executor->loadDylib(Path)));

    // Add the loaded IR module to the JIT. This will set up symbol tables and
    // prepare for materialization.
    for (ThreadSafeModule &TSM : TSMs)
      ExitOnErr(J->addIRModule(std::move(TSM)));

    // The example uses a non-lazy JIT for simplicity. Thus, looking up the main
    // function will materialize all reachable code. It also triggers debug
    // registration in the remote target process.
    JITEvaluatedSymbol MainFn = ExitOnErr(J->lookup("main"));

    outs() << "Running: main(";
    int Pos = 0;
    for (const std::string &Arg : InputArgv)
      outs() << (Pos++ == 0 ? "" : ", ") << "\"" << Arg << "\"";
    outs() << ")\n";

    // Execute the code in the remote target process and dump the result. With
    // the debugger attached to the target, it should be possible to inspect the
    // JITed code as if it was compiled statically.
    int Result = ExitOnErr(Executor->runAsMain(MainFn, InputArgv));
    outs() << "Exit code: " << Result << "\n";
  }

  return 0;
}
