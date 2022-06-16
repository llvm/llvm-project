//===-- TraceIntelPTJSONStructs.h -----------------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTJSONSTRUCTS_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTJSONSTRUCTS_H

#include "lldb/lldb-types.h"

#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"

#include "llvm/ADT/Optional.h"
#include "llvm/Support/JSON.h"

#include <intel-pt.h>
#include <vector>

namespace lldb_private {
namespace trace_intel_pt {

struct JSONModule {
  std::string system_path;
  llvm::Optional<std::string> file;
  uint64_t load_address;
  llvm::Optional<std::string> uuid;
};

struct JSONThread {
  int64_t tid;
  llvm::Optional<std::string> trace_buffer;
};

struct JSONProcess {
  int64_t pid;
  std::string triple;
  std::vector<JSONThread> threads;
  std::vector<JSONModule> modules;
};

struct JSONCore {
  lldb::core_id_t core_id;
  std::string trace_buffer;
  std::string context_switch_trace;
};

struct JSONTraceSession {
  std::string type;
  pt_cpu cpu_info;
  std::vector<JSONProcess> processes;
  llvm::Optional<std::vector<JSONCore>> cores;
  llvm::Optional<LinuxPerfZeroTscConversion> tsc_perf_zero_conversion;

  llvm::Optional<std::vector<lldb::core_id_t>> GetCoreIds();
};

llvm::json::Value toJSON(const JSONModule &module);

llvm::json::Value toJSON(const JSONThread &thread);

llvm::json::Value toJSON(const JSONProcess &process);

llvm::json::Value toJSON(const JSONCore &core);

llvm::json::Value toJSON(const pt_cpu &cpu_info);

llvm::json::Value toJSON(const JSONTraceSession &session);

bool fromJSON(const llvm::json::Value &value, JSONModule &module,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, JSONThread &thread,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, JSONProcess &process,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, JSONCore &core,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, pt_cpu &cpu_info,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, JSONTraceSession &session,
              llvm::json::Path path);
} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTJSONSTRUCTS_H
