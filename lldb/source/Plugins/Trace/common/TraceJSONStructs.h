//===-- TraceJSONStructs.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_COMMON_TRACEJSONSTRUCTS_H
#define LLDB_SOURCE_PLUGINS_TRACE_COMMON_TRACEJSONSTRUCTS_H

#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/JSON.h"
#include <optional>

namespace lldb_private {

struct JSONModule {
  std::string system_path;
  std::optional<std::string> file;
  JSONUINT64 load_address;
  std::optional<std::string> uuid;
};

llvm::json::Value toJSON(const JSONModule &module);

bool fromJSON(const llvm::json::Value &value, JSONModule &module,
              llvm::json::Path path);

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_COMMON_TRACEJSONSTRUCTS_H
