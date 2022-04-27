//===-- TraceIntelPTGDBRemotePackets.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"

using namespace llvm;
using namespace llvm::json;

namespace lldb_private {

const char *IntelPTDataKinds::kProcFsCpuInfo = "procfsCpuInfo";
const char *IntelPTDataKinds::kThreadTraceBuffer = "threadTraceBuffer";

bool fromJSON(const json::Value &value, TraceIntelPTStartRequest &packet,
              Path path) {
  ObjectMapper o(value, path);
  if (!o || !fromJSON(value, (TraceStartRequest &)packet, path) ||
      !o.map("enableTsc", packet.enableTsc) ||
      !o.map("psbPeriod", packet.psbPeriod) ||
      !o.map("threadBufferSize", packet.threadBufferSize) ||
      !o.map("processBufferSizeLimit", packet.processBufferSizeLimit))
    return false;
  if (packet.tids && packet.processBufferSizeLimit) {
    path.report("processBufferSizeLimit must be provided");
    return false;
  }
  if (!packet.tids && !packet.processBufferSizeLimit) {
    path.report("processBufferSizeLimit must not be provided");
    return false;
  }
  return true;
}

json::Value toJSON(const TraceIntelPTStartRequest &packet) {
  json::Value base = toJSON((const TraceStartRequest &)packet);
  base.getAsObject()->try_emplace("threadBufferSize", packet.threadBufferSize);
  base.getAsObject()->try_emplace("processBufferSizeLimit",
                                  packet.processBufferSizeLimit);
  base.getAsObject()->try_emplace("psbPeriod", packet.psbPeriod);
  base.getAsObject()->try_emplace("enableTsc", packet.enableTsc);
  return base;
}

std::chrono::nanoseconds
LinuxPerfZeroTscConversion::Convert(uint64_t raw_counter_value) {
  uint64_t quot = raw_counter_value >> m_time_shift;
  uint64_t rem_flag = (((uint64_t)1 << m_time_shift) - 1);
  uint64_t rem = raw_counter_value & rem_flag;
  return std::chrono::nanoseconds{m_time_zero + quot * m_time_mult +
                                  ((rem * m_time_mult) >> m_time_shift)};
}

json::Value LinuxPerfZeroTscConversion::toJSON() {
  return json::Value(json::Object{
      {"kind", "tsc-perf-zero-conversion"},
      {"time_mult", static_cast<int64_t>(m_time_mult)},
      {"time_shift", static_cast<int64_t>(m_time_shift)},
      {"time_zero", static_cast<int64_t>(m_time_zero)},
  });
}

bool fromJSON(const json::Value &value, TraceTscConversionUP &tsc_conversion,
              json::Path path) {
  ObjectMapper o(value, path);

  int64_t time_mult, time_shift, time_zero;
  if (!o || !o.map("time_mult", time_mult) ||
      !o.map("time_shift", time_shift) || !o.map("time_zero", time_zero))
    return false;

  tsc_conversion = std::make_unique<LinuxPerfZeroTscConversion>(
      static_cast<uint32_t>(time_mult), static_cast<uint16_t>(time_shift),
      static_cast<uint64_t>(time_zero));

  return true;
}

bool fromJSON(const json::Value &value, TraceIntelPTGetStateResponse &packet,
              json::Path path) {
  ObjectMapper o(value, path);
  if (!o || !fromJSON(value, (TraceGetStateResponse &)packet, path))
    return false;

  const Object &obj = *(value.getAsObject());
  if (const json::Value *counters = obj.get("counters")) {
    json::Path subpath = path.field("counters");
    ObjectMapper ocounters(*counters, subpath);
    if (!ocounters || !ocounters.mapOptional("tsc-perf-zero-conversion",
                                             packet.tsc_conversion))
      return false;
  }
  return true;
}

json::Value toJSON(const TraceIntelPTGetStateResponse &packet) {
  json::Value base = toJSON((const TraceGetStateResponse &)packet);

  if (packet.tsc_conversion) {
    std::vector<json::Value> counters{};
    base.getAsObject()->try_emplace(
        "counters", json::Object{{"tsc-perf-zero-conversion",
                                  packet.tsc_conversion->toJSON()}});
  }

  return base;
}

} // namespace lldb_private
