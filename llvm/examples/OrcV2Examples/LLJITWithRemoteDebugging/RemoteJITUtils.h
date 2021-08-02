//===-- RemoteJITUtils.h - Utilities for remote-JITing ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for ExecutorProcessControl-based remote JITing with Orc and
// JITLink.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXAMPLES_ORCV2EXAMPLES_LLJITWITHREMOTEDEBUGGING_REMOTEJITUTILS_H
#define LLVM_EXAMPLES_ORCV2EXAMPLES_LLJITWITHREMOTEDEBUGGING_REMOTEJITUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/FDRawByteChannel.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

namespace llvm {
namespace orc {

class ChildProcessJITLinkExecutor;
class RemoteExecutorProcessControl;
class TCPSocketJITLinkExecutor;

class JITLinkExecutor {
public:
  using RPCChannel = shared::FDRawByteChannel;

  /// Create a JITLinkExecutor for the given exectuable on disk.
  static Expected<std::unique_ptr<ChildProcessJITLinkExecutor>>
  CreateLocal(std::string ExecutablePath);

  /// Find the default exectuable on disk and create a JITLinkExecutor for it.
  static Expected<std::unique_ptr<ChildProcessJITLinkExecutor>>
  FindLocal(const char *JITArgv0);

  /// Create a JITLinkExecutor that connects to the given network address
  /// through a TCP socket. A valid NetworkAddress provides hostname and port,
  /// e.g. localhost:20000.
  static Expected<std::unique_ptr<TCPSocketJITLinkExecutor>>
  ConnectTCPSocket(StringRef NetworkAddress,
                   unique_function<void(Error)> ErrorReporter);

  // Implement ObjectLinkingLayerCreator
  Expected<std::unique_ptr<ObjectLayer>> operator()(ExecutionSession &,
                                                    const Triple &);

  std::unique_ptr<ExecutionSession> startSession();
  Error disconnect();

  Error addDebugSupport(ObjectLayer &ObjLayer);

  Expected<std::unique_ptr<DefinitionGenerator>>
  loadDylib(StringRef RemotePath);

  Expected<int> runAsMain(JITEvaluatedSymbol MainSym,
                          ArrayRef<std::string> Args);

  virtual ~JITLinkExecutor();

protected:
  std::unique_ptr<RemoteExecutorProcessControl> OwnedEPC{nullptr};
  RemoteExecutorProcessControl *EPC{nullptr};

  JITLinkExecutor();
};

/// JITLinkExecutor that runs in a child process on the local machine.
class ChildProcessJITLinkExecutor : public JITLinkExecutor {
public:
  Error launch(unique_function<void(Error)> ErrorReporter);

  pid_t getPID() const { return ProcessID; }
  StringRef getPath() const { return ExecutablePath; }

private:
  std::string ExecutablePath;
  pid_t ProcessID;

  ChildProcessJITLinkExecutor(std::string ExecutablePath)
      : ExecutablePath(std::move(ExecutablePath)) {}

  static std::string defaultPath(const char *HostArgv0, StringRef ExecutorName);
  friend class JITLinkExecutor;
};

/// JITLinkExecutor connected through a TCP socket.
class TCPSocketJITLinkExecutor : public JITLinkExecutor {
private:
  TCPSocketJITLinkExecutor(std::unique_ptr<RemoteExecutorProcessControl> EPC);

  friend class JITLinkExecutor;
};

} // namespace orc
} // namespace llvm

#endif
