//===-- TraceJSONStruct.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TRACEJSONSTRUCTS_H
#define LLDB_TARGET_TRACEJSONSTRUCTS_H

#include "lldb/lldb-types.h"
#include "llvm/Support/JSON.h"

namespace lldb_private {

struct JSONAddress {
  lldb::addr_t value;
};

struct JSONModule {
  std::string system_path;
  llvm::Optional<std::string> file;
  JSONAddress load_address;
  llvm::Optional<std::string> uuid;
};

struct JSONThread {
  int64_t tid;
  std::string trace_file;
};

struct JSONProcess {
  int64_t pid;
  std::string triple;
  std::vector<JSONThread> threads;
  std::vector<JSONModule> modules;
};

struct JSONTracePluginSettings {
  std::string type;
};

struct JSONTraceSessionBase {
  std::vector<JSONProcess> processes;
};

/// The trace plug-in implementation should provide its own TPluginSettings,
/// which corresponds to the "trace" section of the schema.
template <class TPluginSettings>
struct JSONTraceSession : JSONTraceSessionBase {
  TPluginSettings trace;
};

} // namespace lldb_private

namespace llvm {
namespace json {

llvm::json::Value toJSON(const lldb_private::JSONModule &module);

llvm::json::Value toJSON(const lldb_private::JSONThread &thread);

llvm::json::Value toJSON(const lldb_private::JSONProcess &process);

llvm::json::Value
toJSON(const lldb_private::JSONTraceSessionBase &session_base);

bool fromJSON(const Value &value, lldb_private::JSONAddress &address,
              Path path);

bool fromJSON(const Value &value, lldb_private::JSONModule &module, Path path);

bool fromJSON(const Value &value, lldb_private::JSONThread &thread, Path path);

bool fromJSON(const Value &value, lldb_private::JSONProcess &process,
              Path path);

bool fromJSON(const Value &value,
              lldb_private::JSONTracePluginSettings &plugin_settings,
              Path path);

bool fromJSON(const Value &value, lldb_private::JSONTraceSessionBase &session,
              Path path);

template <class TPluginSettings>
bool fromJSON(const Value &value,
              lldb_private::JSONTraceSession<TPluginSettings> &session,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("trace", session.trace) &&
         fromJSON(value, (lldb_private::JSONTraceSessionBase &)session, path);
}

} // namespace json
} // namespace llvm

#endif // LLDB_TARGET_TRACEJSONSTRUCTS_H
