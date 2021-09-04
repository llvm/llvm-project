//===-- TraceSessionFileStructs.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "TraceJSONStructs.h"
#include "ThreadPostMortemTrace.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include <sstream>

using namespace lldb_private;
namespace llvm {
namespace json {

llvm::json::Value toJSON(const JSONModule &module) {
  llvm::json::Object json_module;
  json_module["systemPath"] = module.system_path;
  if (module.file)
    json_module["file"] = *module.file;
  std::ostringstream oss;
  oss << "0x" << std::hex << module.load_address.value;
  json_module["loadAddress"] = oss.str();
  if (module.uuid)
    json_module["uuid"] = *module.uuid;
  return std::move(json_module);
}

llvm::json::Value toJSON(const JSONThread &thread) {
  return Value(Object{{"tid", thread.tid}, {"traceFile", thread.trace_file}});
}

llvm::json::Value toJSON(const JSONProcess &process) {
  llvm::json::Object json_process;
  json_process["pid"] = process.pid;
  json_process["triple"] = process.triple;

  llvm::json::Array threads_arr;
  for (JSONThread e : process.threads)
    threads_arr.push_back(toJSON(e));

  json_process["threads"] = llvm::json::Value(std::move(threads_arr));

  llvm::json::Array modules_arr;
  for (JSONModule e : process.modules)
    modules_arr.push_back(toJSON(e));

  json_process["modules"] = llvm::json::Value(std::move(modules_arr));

  return std::move(json_process);
}

llvm::json::Value toJSON(const JSONTraceSessionBase &session) {
  llvm::json::Array arr;
  for (JSONProcess e : session.processes)
    arr.push_back(toJSON(e));

  return std::move(arr);
}

bool fromJSON(const Value &value, JSONAddress &address, Path path) {
  Optional<StringRef> s = value.getAsString();
  if (s.hasValue() && !s->getAsInteger(0, address.value))
    return true;

  path.report("expected numeric string");
  return false;
}

bool fromJSON(const Value &value, JSONModule &module, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("systemPath", module.system_path) &&
         o.map("file", module.file) &&
         o.map("loadAddress", module.load_address) &&
         o.map("uuid", module.uuid);
}

bool fromJSON(const Value &value, JSONThread &thread, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("tid", thread.tid) && o.map("traceFile", thread.trace_file);
}

bool fromJSON(const Value &value, JSONProcess &process, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("pid", process.pid) && o.map("triple", process.triple) &&
         o.map("threads", process.threads) && o.map("modules", process.modules);
}

bool fromJSON(const Value &value, JSONTracePluginSettings &plugin_settings,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("type", plugin_settings.type);
}

bool fromJSON(const Value &value, JSONTraceSessionBase &session, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("processes", session.processes);
}

} // namespace json
} // namespace llvm
