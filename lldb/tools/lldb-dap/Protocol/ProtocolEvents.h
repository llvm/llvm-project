//===-- ProtocolEvents.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains POD structs based on the DAP specification at
// https://microsoft.github.io/debug-adapter-protocol/specification
//
// This is not meant to be a complete implementation, new interfaces are added
// when they're needed.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSON.h)
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_PROTOCOL_PROTOCOL_EVENTS_H
#define LLDB_TOOLS_LLDB_DAP_PROTOCOL_PROTOCOL_EVENTS_H

#include "lldb/lldb-types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

namespace lldb_dap::protocol {

// MARK: Events

/// The event indicates that the debuggee has exited and returns its exit code.
struct ExitedEventBody {
  static llvm::StringLiteral getEvent() { return "exited"; }

  /// The exit code returned from the debuggee.
  int exitCode;
};
llvm::json::Value toJSON(const ExitedEventBody &);

/// The event indicates that the debugger has begun debugging a new process.
/// Either one that it has launched, or one that it has attached to.
struct ProcessEventBody {
  /// The logical name of the process. This is usually the full path to
  /// process's executable file. Example: /home/example/myproj/program.js.
  std::string name;

  /// The process ID of the debugged process, as assigned by the operating
  /// system. This property should be omitted for logical processes that do not
  /// map to operating system processes on the machine.
  std::optional<lldb::pid_t> systemProcessId;

  /// If true, the process is running on the same computer as the debug adapter.
  std::optional<bool> isLocalProcess;

  enum class StartMethod {
    /// Process was launched under the debugger.
    launch,
    /// Debugger attached to an existing process.
    attach,
    /// A project launcher component has launched a new process in a suspended
    /// state and then asked the debugger to attach.
    attachForSuspendedLaunch
  };

  /// Describes how the debug engine started debugging this process.
  std::optional<StartMethod> startMethod;

  /// The size of a pointer or address for this process, in bits. This value may
  /// be used by clients when formatting addresses for display.
  std::optional<int> pointerSize;
};
llvm::json::Value toJSON(const ProcessEventBody &);

} // namespace lldb_dap::protocol

#endif
