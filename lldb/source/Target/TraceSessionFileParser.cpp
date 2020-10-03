//===-- TraceSessionFileParser.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "lldb/Target/TraceSessionFileParser.h"

#include <sstream>

#include "lldb/Core/Module.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

void TraceSessionFileParser::NormalizePath(lldb_private::FileSpec &file_spec) {
  if (file_spec.IsRelative())
    file_spec.PrependPathComponent(m_session_file_dir);
}

Error TraceSessionFileParser::ParseModule(lldb::TargetSP &target_sp,
                                          const JSONModule &module) {
  FileSpec system_file_spec(module.system_path);
  NormalizePath(system_file_spec);

  FileSpec local_file_spec(module.file.hasValue() ? *module.file
                                                  : module.system_path);
  NormalizePath(local_file_spec);

  ModuleSpec module_spec;
  module_spec.GetFileSpec() = local_file_spec;
  module_spec.GetPlatformFileSpec() = system_file_spec;
  module_spec.SetObjectOffset(module.load_address.value);

  if (module.uuid.hasValue())
    module_spec.GetUUID().SetFromStringRef(*module.uuid);

  Status error;
  ModuleSP module_sp =
      target_sp->GetOrCreateModule(module_spec, /*notify*/ false, &error);
  return error.ToError();
}

Error TraceSessionFileParser::CreateJSONError(json::Path::Root &root,
                                              const json::Value &value) {
  std::string err;
  raw_string_ostream os(err);
  root.printErrorContext(value, os);
  return createStringError(
      std::errc::invalid_argument, "%s\n\nContext:\n%s\n\nSchema:\n%s",
      toString(root.getError()).c_str(), os.str().c_str(), m_schema.data());
}

std::string TraceSessionFileParser::BuildSchema(StringRef plugin_schema) {
  std::ostringstream schema_builder;
  schema_builder << "{\n  \"trace\": ";
  schema_builder << plugin_schema.data() << ",";
  schema_builder << R"(
  "processes": [
    {
      "pid": integer,
      "triple": string, // llvm-triple
      "threads": [
        {
          "tid": integer,
          "traceFile": string
        }
      ],
      "modules": [
        {
          "systemPath": string, // original path of the module at runtime
          "file"?: string, // copy of the file if not available at "systemPath"
          "loadAddress": string, // string address in hex or decimal form
          "uuid"?: string,
        }
      ]
    }
  ]
  // Notes:
  // All paths are either absolute or relative to the session file.
}
)";
  return schema_builder.str();
}

namespace llvm {
namespace json {

bool fromJSON(const Value &value, TraceSessionFileParser::JSONAddress &address,
              Path path) {
  Optional<StringRef> s = value.getAsString();
  if (s.hasValue() && !s->getAsInteger(0, address.value))
    return true;

  path.report("expected numeric string");
  return false;
}

bool fromJSON(const Value &value, TraceSessionFileParser::JSONModule &module,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("systemPath", module.system_path) &&
         o.map("file", module.file) &&
         o.map("loadAddress", module.load_address) &&
         o.map("uuid", module.uuid);
}

bool fromJSON(const Value &value, TraceSessionFileParser::JSONThread &thread,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("tid", thread.tid) && o.map("traceFile", thread.trace_file);
}

bool fromJSON(const Value &value, TraceSessionFileParser::JSONProcess &process,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("pid", process.pid) && o.map("triple", process.triple) &&
         o.map("threads", process.threads) && o.map("modules", process.modules);
}

bool fromJSON(const Value &value,
              TraceSessionFileParser::JSONTracePluginSettings &plugin_settings,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("type", plugin_settings.type);
}

} // namespace json
} // namespace llvm
