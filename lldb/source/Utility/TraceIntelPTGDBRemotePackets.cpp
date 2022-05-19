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
const char *IntelPTDataKinds::kTraceBuffer = "traceBuffer";
const char *IntelPTDataKinds::kPerfContextSwitchTrace =
    "perfContextSwitchTrace";

bool TraceIntelPTStartRequest::IsPerCoreTracing() const {
  return per_core_tracing.getValueOr(false);
}

bool fromJSON(const json::Value &value, TraceIntelPTStartRequest &packet,
              Path path) {
  ObjectMapper o(value, path);
  if (!o || !fromJSON(value, (TraceStartRequest &)packet, path) ||
      !o.map("enableTsc", packet.enable_tsc) ||
      !o.map("psbPeriod", packet.psb_period) ||
      !o.map("traceBufferSize", packet.trace_buffer_size))
    return false;

  if (packet.IsProcessTracing()) {
    if (!o.map("processBufferSizeLimit", packet.process_buffer_size_limit) ||
        !o.map("perCoreTracing", packet.per_core_tracing))
      return false;
  }
  return true;
}

json::Value toJSON(const TraceIntelPTStartRequest &packet) {
  json::Value base = toJSON((const TraceStartRequest &)packet);
  json::Object &obj = *base.getAsObject();
  obj.try_emplace("traceBufferSize", packet.trace_buffer_size);
  obj.try_emplace("processBufferSizeLimit", packet.process_buffer_size_limit);
  obj.try_emplace("psbPeriod", packet.psb_period);
  obj.try_emplace("enableTsc", packet.enable_tsc);
  obj.try_emplace("perCoreTracing", packet.per_core_tracing);
  return base;
}

std::chrono::nanoseconds
LinuxPerfZeroTscConversion::ToNanos(uint64_t tsc) const {
  uint64_t quot = tsc >> time_shift;
  uint64_t rem_flag = (((uint64_t)1 << time_shift) - 1);
  uint64_t rem = tsc & rem_flag;
  return std::chrono::nanoseconds{time_zero + quot * time_mult +
                                  ((rem * time_mult) >> time_shift)};
}

uint64_t
LinuxPerfZeroTscConversion::ToTSC(std::chrono::nanoseconds nanos) const {
  uint64_t time = nanos.count() - time_zero;
  uint64_t quot = time / time_mult;
  uint64_t rem = time % time_mult;
  return (quot << time_shift) + (rem << time_shift) / time_mult;
}

json::Value toJSON(const LinuxPerfZeroTscConversion &packet) {
  return json::Value(json::Object{
      {"timeMult", packet.time_mult},
      {"timeShift", packet.time_shift},
      {"timeZero", packet.time_zero},
  });
}

bool fromJSON(const json::Value &value, LinuxPerfZeroTscConversion &packet,
              json::Path path) {
  ObjectMapper o(value, path);
  uint64_t time_mult, time_shift, time_zero;
  if (!o || !o.map("timeMult", time_mult) || !o.map("timeShift", time_shift) ||
      !o.map("timeZero", time_zero))
    return false;
  packet.time_mult = time_mult;
  packet.time_zero = time_zero;
  packet.time_shift = time_shift;
  return true;
}

bool fromJSON(const json::Value &value, TraceIntelPTGetStateResponse &packet,
              json::Path path) {
  ObjectMapper o(value, path);
  return o && fromJSON(value, (TraceGetStateResponse &)packet, path) &&
         o.map("tscPerfZeroConversion", packet.tsc_perf_zero_conversion);
}

json::Value toJSON(const TraceIntelPTGetStateResponse &packet) {
  json::Value base = toJSON((const TraceGetStateResponse &)packet);
  base.getAsObject()->insert(
      {"tscPerfZeroConversion", packet.tsc_perf_zero_conversion});
  return base;
}

} // namespace lldb_private
