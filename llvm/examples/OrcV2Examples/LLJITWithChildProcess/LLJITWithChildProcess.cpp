//===--- LLJITWithChildProcess.cpp - LLJIT targeting a child process ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In this example we will execute JITed code in a child process:
//
// 1. Launch a remote process.
// 2. Create a JITLink-compatible remote memory manager.
// 3. Use LLJITBuilder to create a (greedy) LLJIT instance.
// 4. Add the Add1Example module and execute add1().
// 5. Terminate the remote target session.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/OrcRemoteTargetClient.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "../ExampleModules.h"
#include "RemoteJITUtils.h"

#include <memory>
#include <string>

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;

// Executable running in the child process for remote execution. It communicates
// via stdin/stdout pipes.
cl::opt<std::string>
    ChildExecPath("remote-process", cl::Required,
                  cl::desc("Specify the filename of the process to launch for "
                           "remote JITing."),
                  cl::value_desc("filename"));

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  cl::ParseCommandLineOptions(argc, argv, "LLJITWithChildProcess");

  ExitOnError ExitOnErr;
  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  if (!sys::fs::can_execute(ChildExecPath)) {
    WithColor::error(errs(), argv[0])
        << "Child executable invalid: '" << ChildExecPath << "'\n";
    return -1;
  }

  ExecutionSession ES;
  ES.setErrorReporter([&](Error Err) { ExitOnErr(std::move(Err)); });

  // Launch the remote process and get a channel to it.
  pid_t ChildPID;
  std::unique_ptr<FDRawChannel> Ch = launchRemote(ChildExecPath, ChildPID);
  if (!Ch) {
    WithColor::error(errs(), argv[0]) << "Failed to launch remote JIT.\n";
    exit(1);
  }

  LLVM_DEBUG({
    dbgs()
        << "Launched executable in subprocess " << ChildPID << ":\n"
        << ChildExecPath << "\n\n"
        << "You may want to attach a debugger now. Press enter to continue.\n";
    fflush(stdin);
    getchar();
  });

  std::unique_ptr<remote::OrcRemoteTargetClient> Client =
      ExitOnErr(remote::OrcRemoteTargetClient::Create(*Ch, ES));

  // Create a JITLink-compatible remote memory manager.
  using MemManager = remote::OrcRemoteTargetClient::RemoteJITLinkMemoryManager;
  std::unique_ptr<MemManager> RemoteMM =
      ExitOnErr(Client->createRemoteJITLinkMemoryManager());

  // Our remote target is running on the host system.
  auto JTMB = ExitOnErr(JITTargetMachineBuilder::detectHost());
  JTMB.setCodeModel(CodeModel::Small);

  // Create an LLJIT instance with a JITLink ObjectLinkingLayer.
  auto J = ExitOnErr(
      LLJITBuilder()
          .setJITTargetMachineBuilder(std::move(JTMB))
          .setObjectLinkingLayerCreator(
              [&](ExecutionSession &ES,
                  const Triple &TT) -> std::unique_ptr<ObjectLayer> {
                return std::make_unique<ObjectLinkingLayer>(ES, *RemoteMM);
              })
          .create());

  auto M = ExitOnErr(parseExampleModule(Add1Example, "add1"));

  ExitOnErr(J->addIRModule(std::move(M)));

  // Look up the JIT'd function.
  auto Add1Sym = ExitOnErr(J->lookup("add1"));

  // Run in child target.
  Expected<int> Result = Client->callIntInt(Add1Sym.getAddress(), 42);
  if (Result)
    outs() << "add1(42) = " << *Result << "\n";
  else
    ES.reportError(Result.takeError());

  // Signal the remote target that we're done JITing.
  ExitOnErr(Client->terminateSession());
  LLVM_DEBUG(dbgs() << "Subprocess terminated\n");

  return 0;
}
