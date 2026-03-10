//===-- TraceArmETMBundleLoader.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_ARM_ETM_TRACEARMETMBUNDLELOADER_H
#define LLDB_SOURCE_PLUGINS_TRACE_ARM_ETM_TRACEARMETMBUNDLELOADER_H

#include "../common/ThreadPostMortemTrace.h"
#include "../common/TraceBundleLoader.h"
#include "TraceArmETMJSONStructs.h"

namespace lldb_private {
namespace trace_arm_etm {

class TraceArmETM;

class TraceArmETMBundleLoader : public TraceBundleLoader {
public:
  /// \param[in] debugger
  ///   The debugger that will own the targets to create.
  ///
  /// \param[in] bundle_description
  ///   The JSON description of a trace bundle that follows the schema of the
  ///   arm etm trace plug-in.
  ///
  /// \param[in] bundle_dir
  ///   The folder where the trace bundle is located.
  TraceArmETMBundleLoader(Debugger &debugger,
                          const llvm::json::Value &bundle_description,
                          llvm::StringRef bundle_dir)
      : TraceBundleLoader(debugger, bundle_dir),
        m_bundle_description(bundle_description) {}

  /// \return
  ///   The JSON schema for the bundle description.
  static llvm::StringRef GetSchema();

  /// Parse the trace bundle description and create the corresponding \a
  /// Target objects. In case of an error, no targets are created.
  ///
  /// \return
  ///   A \a lldb::TraceSP instance created according to the trace bundle
  ///   information. In case of errors, return a null pointer.
  llvm::Expected<lldb::TraceSP> Load();

private:
  /// Create a post-mortem thread associated with the given \p process
  /// using the definition from \p thread.
  lldb::ThreadPostMortemTraceSP ParseThread(Process &process,
                                            const JSONThread &thread);

  /// Given a bundle description and a list of fully parsed processes,
  /// create an actual Trace instance that "traces" these processes.
  llvm::Expected<lldb::TraceSP>
  CreateTraceArmETMInstance(JSONTraceBundleDescription &bundle_description,
                            std::vector<ParsedProcess> &parsed_processes);

  /// Create the corresponding Threads and Process objects given the JSON
  /// process definition.
  ///
  /// \param[in] process
  ///   The JSON process definition
  llvm::Expected<ParsedProcess> ParseProcess(const JSONProcess &process);

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
  /// bundle description.
  llvm::Expected<std::vector<ParsedProcess>>
  LoadBundle(const JSONTraceBundleDescription &bundle_description);

  /// Modify the bundle description by normalizing all the paths relative to
  /// the session file directory.
  void NormalizeAllPaths(JSONTraceBundleDescription &bundle_description);

  const llvm::json::Value &m_bundle_description;
};

} // namespace trace_arm_etm
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_ARM_ETM_TRACEARMETMBUNDLELOADER_H
