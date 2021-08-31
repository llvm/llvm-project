//===-- TraceSessionFileParser.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TRACESESSIONPARSER_H
#define LLDB_TARGET_TRACESESSIONPARSER_H

#include "ThreadPostMortemTrace.h"
#include "TraceJSONStructs.h"

namespace lldb_private {

/// \class TraceSessionFileParser TraceSessionFileParser.h
///
/// Base class for parsing the common information of JSON trace session files.
/// Contains the basic C++ structs that represent the JSON data, which include
/// \a JSONTraceSession as the root object.
///
/// See \a Trace::FindPlugin for more information regarding these JSON files.
class TraceSessionFileParser {
public:

  /// Helper struct holding the objects created when parsing a process
  struct ParsedProcess {
    lldb::TargetSP target_sp;
    std::vector<lldb::ThreadPostMortemTraceSP> threads;
  };

  TraceSessionFileParser(Debugger &debugger, llvm::StringRef session_file_dir,
                         llvm::StringRef schema)
      : m_debugger(debugger), m_session_file_dir(session_file_dir),
        m_schema(schema) {}

  /// Build the full schema for a Trace plug-in.
  ///
  /// \param[in] plugin_schema
  ///   The subschema that corresponds to the "trace" section of the schema.
  ///
  /// \return
  ///   The full schema containing the common attributes and the plug-in
  ///   specific attributes.
  static std::string BuildSchema(llvm::StringRef plugin_schema);

  /// Parse the fields common to all trace session schemas.
  ///
  /// \param[in] session
  ///     The session json objects already deserialized.
  ///
  /// \return
  ///     A list of \a ParsedProcess containing all threads and targets created
  ///     during the parsing, or an error in case of failures. In case of
  ///     errors, no side effects are produced.
  llvm::Expected<std::vector<ParsedProcess>>
  ParseCommonSessionFile(const JSONTraceSessionBase &session);

protected:
  /// Resolve non-absolute paths relative to the session file folder. It
  /// modifies the given file_spec.
  void NormalizePath(lldb_private::FileSpec &file_spec);

  lldb::ThreadPostMortemTraceSP ParseThread(lldb::ProcessSP &process_sp,
                                            const JSONThread &thread);

  llvm::Expected<ParsedProcess> ParseProcess(const JSONProcess &process);

  llvm::Error ParseModule(lldb::TargetSP &target_sp, const JSONModule &module);

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

  Debugger &m_debugger;
  std::string m_session_file_dir;
  llvm::StringRef m_schema;
};
} // namespace lldb_private


#endif // LLDB_TARGET_TRACESESSIONPARSER_H
