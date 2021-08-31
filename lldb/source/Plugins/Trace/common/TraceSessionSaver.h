//===-- SessionSaver.h ----------------------------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TRACESESSIONSAVER_H
#define LLDB_TARGET_TRACESESSIONSAVER_H

#include "TraceJSONStructs.h"

namespace lldb_private {

class TraceSessionSaver {

public:
  /// Save the trace session description JSON object inside the given directory
  /// as a file named \a trace.json.
  ///
  /// \param[in] trace_session_description
  ///     The trace's session, as JSON Object.
  ///
  /// \param[in] directory
  ///     The directory where the JSON file will be saved.
  ///
  /// \return
  ///     \a llvm::success if the operation was successful, or an \a llvm::Error
  ///     otherwise.
  static llvm::Error
  WriteSessionToFile(const llvm::json::Value &trace_session_description,
                     FileSpec directory);

  /// Build the processes section of the trace session description file. Besides
  /// returning the processes information, this method saves to disk all modules
  /// and raw traces corresponding to the traced threads of the given process.
  ///
  /// \param[in] live_process
  ///     The process being traced.
  ///
  /// \param[in] raw_trace_fetcher
  ///     Callback function that receives a thread ID and returns its raw trace.
  ///     This callback should return \a None if the thread is not being traced.
  ///     Otherwise, it should return the raw trace in bytes or an
  ///     \a llvm::Error in case of failures.
  ///
  /// \param[in] directory
  ///     The directory where files will be saved when building the processes
  ///     section.
  ///
  /// \return
  ///     The processes section or \a llvm::Error in case of failures.
  static llvm::Expected<JSONTraceSessionBase> BuildProcessesSection(
      Process &live_process,
      std::function<
          llvm::Expected<llvm::Optional<std::vector<uint8_t>>>(lldb::tid_t tid)>
          raw_trace_fetcher,
      FileSpec directory);

  /// Build the threads sub-section of the trace session description file.
  /// For each traced thread, its raw trace is also written to the file
  /// \a thread_id_.trace inside of the given directory.
  ///
  /// \param[in] live_process
  ///     The process being traced.
  ///
  /// \param[in] raw_trace_fetcher
  ///     Callback function that receives a thread ID and returns its raw trace.
  ///     This callback should return \a None if the thread is not being traced.
  ///     Otherwise, it should return the raw trace in bytes or an
  ///     \a llvm::Error in case of failures.
  ///
  /// \param[in] directory
  ///     The directory where files will be saved when building the threads
  ///     section.
  ///
  /// \return
  ///     The threads section or \a llvm::Error in case of failures.
  static llvm::Expected<std::vector<JSONThread>> BuildThreadsSection(
      Process &live_process,
      std::function<
          llvm::Expected<llvm::Optional<std::vector<uint8_t>>>(lldb::tid_t tid)>
          raw_trace_fetcher,
      FileSpec directory);

  /// Build modules sub-section of the trace's session. The original modules
  /// will be copied over to the \a <directory/modules> folder. Invalid modules
  /// are skipped.
  /// Copying the modules has the benefit of making these trace session
  /// directories self-contained, as the raw traces and modules are part of the
  /// output directory and can be sent to another machine, where lldb can load
  /// them and replicate exactly the same trace session.
  ///
  /// \param[in] live_process
  ///     The process being traced.
  ///
  /// \param[in] directory
  ///     The directory where the modules files will be saved when building
  ///     the modules section.
  ///     Example: If a module \a libbar.so exists in the path
  ///     \a /usr/lib/foo/libbar.so, then it will be copied to
  ///     \a <directory>/modules/usr/lib/foo/libbar.so.
  ///
  /// \return
  ///     The modules section or \a llvm::Error in case of failures.
  static llvm::Expected<std::vector<JSONModule>>
  BuildModulesSection(Process &live_process, FileSpec directory);
};
} // namespace lldb_private

#endif // LLDB_TARGET_TRACESESSIONSAVER_H
