//===-- TraceIntelPTJSONStructs.h -----------------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTJSONSTRUCTS_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTJSONSTRUCTS_H

#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/JSON.h"
#include <intel-pt.h>
#include <vector>

namespace lldb_private {
namespace trace_intel_pt {

struct JSONModule {
  std::string system_path;
  llvm::Optional<std::string> file;
  JSONUINT64 load_address;
  llvm::Optional<std::string> uuid;
};

struct JSONThread {
  uint64_t tid;
  llvm::Optional<std::string> ipt_trace;
};

struct JSONProcess {
  uint64_t pid;
  llvm::Optional<std::string> triple;
  std::vector<JSONThread> threads;
  std::vector<JSONModule> modules;
};

struct JSONCpu {
  lldb::cpu_id_t id;
  std::string ipt_trace;
  std::string context_switch_trace;
};

struct JSONKernel {
  llvm::Optional<JSONUINT64> load_address;
  std::string file;
};

struct JSONTraceBundleDescription {
  std::string type;
  pt_cpu cpu_info;
  llvm::Optional<std::vector<JSONProcess>> processes;
  llvm::Optional<std::vector<JSONCpu>> cpus;
  llvm::Optional<LinuxPerfZeroTscConversion> tsc_perf_zero_conversion;
  llvm::Optional<JSONKernel> kernel;

  llvm::Optional<std::vector<lldb::cpu_id_t>> GetCpuIds();
};

llvm::json::Value toJSON(const JSONModule &module);

llvm::json::Value toJSON(const JSONThread &thread);

llvm::json::Value toJSON(const JSONProcess &process);

llvm::json::Value toJSON(const JSONCpu &cpu);

llvm::json::Value toJSON(const pt_cpu &cpu_info);

llvm::json::Value toJSON(const JSONKernel &kernel);

llvm::json::Value toJSON(const JSONTraceBundleDescription &bundle_description);

bool fromJSON(const llvm::json::Value &value, JSONModule &module,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, JSONThread &thread,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, JSONProcess &process,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, JSONCpu &cpu,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, pt_cpu &cpu_info,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value, JSONModule &kernel,
              llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value,
              JSONTraceBundleDescription &bundle_description,
              llvm::json::Path path);
} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTJSONSTRUCTS_H
