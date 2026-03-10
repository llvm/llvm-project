//===-- TraceArmETMJSONStructs.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_ARM_ETM_TRACEARMETMJSONSTRUCTS_H
#define LLDB_SOURCE_PLUGINS_TRACE_ARM_ETM_TRACEARMETMJSONSTRUCTS_H

#include "../common/TraceJSONStructs.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <vector>

namespace lldb_private {
namespace trace_arm_etm {

struct JSONThread {
  uint64_t tid;
  std::optional<std::string> etm_trace;
};

struct JSONProcess {
  uint64_t pid;
  std::optional<std::string> triple;
  std::vector<JSONThread> threads;
  std::vector<JSONModule> modules;
};

struct JSONTraceBundleDescription {
  std::string type;
  std::optional<std::vector<JSONProcess>> processes;
};

llvm::json::Value toJSON(const JSONThread &thread);

llvm::json::Value toJSON(const JSONProcess &process);

llvm::json::Value toJSON(const JSONTraceBundleDescription &bundle_description);

bool fromJSON(const llvm::json::Value &value, JSONThread &thread,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, JSONProcess &process,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value,
              JSONTraceBundleDescription &bundle_description,
              llvm::json::Path path);
} // namespace trace_arm_etm
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_ARM_ETM_TRACEARMETMJSONSTRUCTS_H
