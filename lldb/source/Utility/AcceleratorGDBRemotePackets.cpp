//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/AcceleratorGDBRemotePackets.h"

using namespace lldb_private;

llvm::json::Value lldb_private::toJSON(const AcceleratorActions &data) {
  return llvm::json::Object{
      {"plugin_name", data.plugin_name},
      {"session_name", data.session_name},
      {"identifier", data.identifier},
  };
}

bool lldb_private::fromJSON(const llvm::json::Value &value,
                            AcceleratorActions &data, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("plugin_name", data.plugin_name) &&
         o.map("session_name", data.session_name) &&
         o.map("identifier", data.identifier);
}
