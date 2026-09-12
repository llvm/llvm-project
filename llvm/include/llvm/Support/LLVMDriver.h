//===- LLVMDriver.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_LLVMDRIVER_H
#define LLVM_SUPPORT_LLVMDRIVER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorOr.h"

#include <memory>

namespace llvm {

class LLVMToolSession;
class ToolContext;

using ToolMainFn = int (*)(int, char **, const ToolContext &);

/// An LLVM command-line tool that can be invoked without creating a process.
struct CallableTool {
  StringRef Name;
  ToolMainFn Main;

  explicit operator bool() const { return Main != nullptr; }
};

/// Configures process-wide behavior owned by an LLVM tool session.
struct LLVMToolSessionOptions {
  /// Prefer a registered in-process tool even when its executable is not an
  /// alias of the session executable. This is useful for hosts whose tools do
  /// not exist as separate files, such as browser applications.
  bool PreferInProcessTools = false;

  bool InstallPipeSignalExitHandler = true;
  bool NeedsPOSIXUtilitySignalHandling = false;
};

/// Describes how a tool was invoked and provides access to its host session.
class ToolContext {
  LLVMToolSession *Session = nullptr;

  friend class LLVMToolSession;

public:
  const char *Path;
  const char *PrependArg;
  // PrependArg will be added unconditionally by the llvm-driver, but
  // NeedsPrependArg will be false if Path is adequate to reinvoke the tool.
  // This is useful if realpath is ever called on Path, in which case it will
  // point to the llvm-driver executable, where PrependArg will be needed to
  // invoke the correct tool.
  bool NeedsPrependArg;

  ToolContext(const char *Path, const char *PrependArg, bool NeedsPrependArg)
      : Path(Path), PrependArg(PrependArg), NeedsPrependArg(NeedsPrependArg) {}

  /// Finds a tool registered with the session that owns this context.
  LLVM_ABI ErrorOr<CallableTool> getCallableTool(StringRef Name) const;

  /// Invokes another tool registered with the same host session.
  LLVM_ABI int callTool(ArrayRef<const char *> Args) const;

  /// Returns true when Executable names a registered tool owned by this host.
  /// A tool is owned when it is an alias of the session executable or when the
  /// session explicitly prefers its registered in-process tools.
  LLVM_ABI bool canExecuteInProcess(StringRef Executable) const;

  /// Returns true when this invocation is owned by a long-lived tool session.
  bool isInProcess() const { return Session != nullptr; }
};

/// Owns LLVM process initialization and an in-process tool registry.
///
/// A long-lived host constructs one session and uses it for every embedded
/// tool invocation. The individual tools borrow a ToolContext and therefore do
/// not initialize or shut down LLVM themselves.
class LLVM_ABI LLVMToolSession {
public:
  LLVMToolSession(int &Argc, char **&Argv, ArrayRef<CallableTool> Tools,
                  LLVMToolSessionOptions Options = {});
  ~LLVMToolSession();

  LLVMToolSession(const LLVMToolSession &) = delete;
  LLVMToolSession &operator=(const LLVMToolSession &) = delete;

  /// Invokes the tool named by Args[0]. Args may instead contain a
  /// process-style argv beginning with the session executable or an LLVM
  /// multicall name.
  int callTool(ArrayRef<const char *> Args);

private:
  struct Impl;
  std::unique_ptr<Impl> PImpl;

  ErrorOr<CallableTool> findTool(StringRef Name) const;
  bool canExecuteInProcess(StringRef Executable) const;
  ToolContext makeContext(StringRef InvokedName);

  friend class ToolContext;
};

} // namespace llvm

#endif
