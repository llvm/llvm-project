//===-- TraceJSONStructs.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceJSONStructs.h"
#include "llvm/Support/JSON.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;
using namespace llvm::json;

namespace lldb_private {

json::Value toJSON(const JSONModule &module) {
  json::Object json_module;
  json_module["systemPath"] = module.system_path;
  if (module.file)
    json_module["file"] = *module.file;
  json_module["loadAddress"] = toJSON(module.load_address, true);
  if (module.uuid)
    json_module["uuid"] = *module.uuid;
  return std::move(json_module);
}

bool fromJSON(const json::Value &value, JSONModule &module, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("systemPath", module.system_path) &&
         o.map("file", module.file) &&
         o.map("loadAddress", module.load_address) &&
         o.map("uuid", module.uuid);
}

} // namespace lldb_private
