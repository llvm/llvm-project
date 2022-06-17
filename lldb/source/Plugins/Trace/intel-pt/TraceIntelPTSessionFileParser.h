//===-- TraceIntelPTSessionFileParser.h -----------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTSESSIONFILEPARSER_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTSESSIONFILEPARSER_H

#include "../common/ThreadPostMortemTrace.h"
#include "TraceIntelPTJSONStructs.h"

namespace lldb_private {
namespace trace_intel_pt {

class TraceIntelPT;

class TraceIntelPTSessionFileParser {
public:
  /// Helper struct holding the objects created when parsing a process
  struct ParsedProcess {
    lldb::TargetSP target_sp;
    std::vector<lldb::ThreadPostMortemTraceSP> threads;
  };

  /// \param[in] debugger
  ///   The debugger that will own the targets to create.
  ///
  /// \param[in] trace_session_file
  ///   The contents of the main trace session definition file that follows the
  ///   schema of the intel pt trace plug-in.
  ///
  /// \param[in] session_file_dir
  ///   The folder where the trace session file is located.
  TraceIntelPTSessionFileParser(Debugger &debugger,
                                const llvm::json::Value &trace_session_file,
                                llvm::StringRef session_file_dir)
      : m_debugger(debugger), m_trace_session_file(trace_session_file),
        m_session_file_dir(session_file_dir) {}

  /// \return
  ///   The JSON schema for the session data.
  static llvm::StringRef GetSchema();

  /// Parse the structured data trace session and create the corresponding \a
  /// Target objects. In case of an error, no targets are created.
  ///
  /// \return
  ///   A \a lldb::TraceSP instance with the trace session data. In case of
  ///   errors, return a null pointer.
  llvm::Expected<lldb::TraceSP> Parse();

private:
  /// Resolve non-absolute paths relative to the session file folder.
  FileSpec NormalizePath(const std::string &path);

  /// Create a post-mortem thread associated with the given \p process
  /// using the definition from \p thread.
  lldb::ThreadPostMortemTraceSP ParseThread(Process &process,
                                            const JSONThread &thread);

  /// Given a session description and a list of fully parsed processes,
  /// create an actual Trace instance that "traces" these processes.
  llvm::Expected<lldb::TraceSP>
  CreateTraceIntelPTInstance(JSONTraceSession &session,
                             std::vector<ParsedProcess> &parsed_processes);

  /// Create the corresponding Threads and Process objects given the JSON
  /// process definition.
  ///
  /// \param[in] process
  ///   The JSON process definition
  llvm::Expected<ParsedProcess> ParseProcess(const JSONProcess &process);

  /// Create a moddule associated with the given \p target
  /// using the definition from \p module.
  llvm::Error ParseModule(Target &target, const JSONModule &module);

  /// Create a user-friendly error message upon a JSON-parsing failure using the
  /// \a json::ObjectMapper functionality.
  ///
  /// \param[in] root
  ///   The \a llvm::json::Path::Root used to parse the JSON \a value.
  ///
  /// \param[in] value
  ///   The json value that failed to parse.
  ///
  /// \return
  ///   An \a llvm::Error containing the user-friendly error message.
  llvm::Error CreateJSONError(llvm::json::Path::Root &root,
                              const llvm::json::Value &value);

  /// Create the corresponding Process, Thread and Module objects given this
  /// session file.
  llvm::Expected<std::vector<ParsedProcess>>
  ParseSessionFile(const JSONTraceSession &session);

  /// When applicable, augment the list of threads in the session file by
  /// inspecting the context switch trace. This only applies for threads of
  /// processes already specified in this session file.
  ///
  /// \return
  ///   An \a llvm::Error in case if failures, or \a llvm::Error::success
  ///   otherwise.
  llvm::Error AugmentThreadsFromContextSwitches(JSONTraceSession &session);

  /// Modifiy the session file by normalizing all the paths relative to the
  /// session file directory.
  void NormalizeAllPaths(JSONTraceSession &session);

  Debugger &m_debugger;
  const llvm::json::Value &m_trace_session_file;
  const std::string m_session_file_dir;
};

} // namespace trace_intel_pt
} // namespace lldb_private


#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTSESSIONFILEPARSER_H
