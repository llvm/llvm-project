//===-- JavaScript.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_JAVASCRIPT_JAVASCRIPT_H
#define LLVM_LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_JAVASCRIPT_JAVASCRIPT_H

#include "lldb/Utility/StructuredData.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <functional>
#include <memory>

// Forward declare V8 types to avoid including V8 headers here
namespace v8 {
class Isolate;
template <class T> class Global;
class Context;
class Platform;
class Function;
} // namespace v8

namespace lldb_private {

class JavaScript {
public:
  JavaScript(lldb::FileSP output_file = nullptr);
  ~JavaScript();

  // Execute JavaScript code
  llvm::Error Run(llvm::StringRef code);

  // Set callback for output (used by console.log)
  using OutputCallback = std::function<void(const std::string &)>;
  void SetOutputCallback(OutputCallback callback);

  // Load and execute a JavaScript module
  llvm::Error LoadModule(llvm::StringRef filename);

  // Check syntax without executing
  llvm::Error CheckSyntax(llvm::StringRef code);

  // Change IO streams
  llvm::Error ChangeIO(FILE *out, FILE *err);

  // Breakpoint callback support
  llvm::Error RegisterBreakpointCallback(void *baton,
                                         const char *command_body_text);

  llvm::Expected<bool>
  CallBreakpointCallback(void *baton, lldb::StackFrameSP stop_frame_sp,
                         lldb::BreakpointLocationSP bp_loc_sp,
                         StructuredData::ObjectSP extra_args_sp);

  // Watchpoint callback support
  llvm::Error RegisterWatchpointCallback(void *baton,
                                         const char *command_body_text);

  llvm::Expected<bool> CallWatchpointCallback(void *baton,
                                              lldb::StackFrameSP stop_frame_sp,
                                              lldb::WatchpointSP wp_sp);

  // Get the V8 isolate (for advanced usage)
  v8::Isolate *GetIsolate() { return m_isolate; }

  // Get the output file (for console.log implementation)
  lldb::FileSP GetOutputFile() { return m_output_file; }

  // Set the output file (for routing console.log to the correct stream)
  void SetOutputFile(lldb::FileSP output_file) { m_output_file = output_file; }

  // Write output (used by console.log)
  void WriteOutput(const std::string &text);

  // Set the debugger instance (exposes lldb.debugger to scripts)
  void SetDebugger(lldb::DebuggerSP debugger_sp);

private:
  static std::unique_ptr<v8::Platform> s_platform;
  static bool s_platform_initialized;

  v8::Isolate *m_isolate;
  v8::Global<v8::Context> *m_context;

  FILE *m_stdout;
  FILE *m_stderr;
  lldb::FileSP m_output_file;
  OutputCallback m_output_callback;

  // Map from baton pointer to JavaScript callback function
  std::map<void *, v8::Global<v8::Function>> m_breakpoint_callbacks;
  std::map<void *, v8::Global<v8::Function>> m_watchpoint_callbacks;

  lldb::DebuggerSP m_debugger;

  // Initialize V8 platform (called once)
  static void InitializePlatform();
};

} // namespace lldb_private

#endif // LLVM_LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_JAVASCRIPT_JAVASCRIPT_H
