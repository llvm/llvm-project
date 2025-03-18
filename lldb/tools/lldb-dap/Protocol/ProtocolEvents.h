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

#include "Protocol/ProtocolTypes.h"
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

/// The event indicates that the target has produced some output.
struct OutputEventBody {
  enum class Category {
    /// Show the output in the client's default message UI, e.g. a
    /// 'debug console'. This category should only be used for informational
    /// output from the debugger (as opposed to the debuggee).
    Console,
    /// A hint for the client to show the output in the client's UI
    /// for important and highly visible information, e.g. as a popup
    /// notification. This category should only be used for important messages
    /// from the debugger (as opposed to the debuggee). Since this category
    /// value
    /// is a hint, clients might ignore the hint and assume the `console`
    /// category.
    Important,
    /// Show the output as normal program output from the debuggee.
    Stdout,
    /// Show the output as error program output from the debuggee.
    Stderr,
    /// Send the output to telemetry instead of showing it to the user.
    Telemetry,
  };

  /// The output category. If not specified or if the category is not
  /// understood by the client, `console` is assumed.
  std::optional<Category> category;

  /// The output to report.
  ///
  /// ANSI escape sequences may be used to influence text color and styling if
  /// `supportsANSIStyling` is present in both the adapter's `Capabilities` and
  /// the client's `InitializeRequestArguments`. A client may strip any
  /// unrecognized ANSI sequences.
  ///
  /// If the `supportsANSIStyling` capabilities are not both true, then the
  /// client should display the output literally.
  std::string output;

  enum class Group {
    /// Start a new group in expanded mode. Subsequent output events are members
    /// of the group and should be shown indented.
    ///
    /// The `output` attribute becomes the name of the group and is not
    /// indented.
    start,

    /// Start a new group in collapsed mode. Subsequent output events are
    /// members of the group and should be shown indented (as soon as the group
    /// is expanded).
    ///
    /// The `output` attribute becomes the name of the group and is not
    /// indented.
    startCollapsed,

    /// End the current group and decrease the indentation of subsequent output
    /// events.
    ///
    /// A non-empty `output` attribute is shown as the unindented end of the
    /// group.
    end,
  };

  /// Support for keeping an output log organized by grouping related messages.
  std::optional<Group> group;

  /// If an attribute `variablesReference` exists and its value is > 0, the
  /// output contains objects which can be retrieved by passing
  /// `variablesReference` to the `variables` request as long as execution
  /// remains suspended. See 'Lifetime of Object References' in the Overview
  /// section for details.
  std::optional<int> variablesReference;

  /// The source location where the output was produced.
  std::optional<Source> source;

  /// The source location's line where the output was produced.
  std::optional<int32_t> line;
  /// The position in `line` where the output was produced. It is measured in
  /// UTF-16 code units and the client capability `columnsStartAt1` determines
  /// whether it is 0- or 1-based.
  std::optional<int32_t> column;

  /// Additional data to report. For the `telemetry` category the data is
  /// sent to telemetry, for the other categories the data is shown in JSON
  /// format.
  std::optional<llvm::json::Value> data;

  /// A reference that allows the client to request the location where the new
  /// value is declared. For example, if the logged value is function pointer,
  /// the adapter may be able to look up the function's location. This should
  /// be present only if the adapter is likely to be able to resolve the
  /// location.
  ///
  /// This reference shares the same lifetime as the `variablesReference`. See
  /// 'Lifetime of Object References' in the Overview section for details.
  std::optional<int> locationReference;
};
llvm::json::Value toJSON(const OutputEventBody &);

} // namespace lldb_dap::protocol

#endif
