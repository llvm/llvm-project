//===-- TraceArmETMJSONStructs.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceArmETMJSONStructs.h"
#include "llvm/Support/JSON.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_arm_etm;
using namespace llvm;
using namespace llvm::json;

namespace lldb_private {
namespace trace_arm_etm {

json::Value toJSON(const JSONThread &thread) {
  json::Object obj{{"tid", thread.tid}};
  if (thread.etm_trace)
    obj["etmTrace"] = *thread.etm_trace;
  return obj;
}

bool fromJSON(const json::Value &value, JSONThread &thread, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("tid", thread.tid) && o.map("etmTrace", thread.etm_trace);
}

json::Value toJSON(const JSONProcess &process) {
  return Object{
      {"pid", process.pid},
      {"triple", process.triple},
      {"threads", process.threads},
      {"modules", process.modules},
  };
}

bool fromJSON(const json::Value &value, JSONProcess &process, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("pid", process.pid) && o.map("triple", process.triple) &&
         o.map("threads", process.threads) && o.map("modules", process.modules);
}

json::Value toJSON(const JSONTraceBundleDescription &bundle_description) {
  return Object{
      {"type", bundle_description.type},
      {"processes", bundle_description.processes},
  };
}

bool fromJSON(const json::Value &value,
              JSONTraceBundleDescription &bundle_description, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("processes", bundle_description.processes) &&
         o.map("type", bundle_description.type);
}

} // namespace trace_arm_etm
} // namespace lldb_private
