//===-- TraceGDBRemotePackets.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/TraceGDBRemotePackets.h"

using namespace llvm;
using namespace llvm::json;

namespace lldb_private {
/// jLLDBTraceSupported
/// \{
bool fromJSON(const json::Value &value, TraceSupportedResponse &packet,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("description", packet.description) &&
         o.map("name", packet.name);
}

json::Value toJSON(const TraceSupportedResponse &packet) {
  return json::Value(
      Object{{"description", packet.description}, {"name", packet.name}});
}
/// \}

/// jLLDBTraceStart
/// \{
bool TraceStartRequest::IsProcessTracing() const { return !(bool)tids; }

bool fromJSON(const json::Value &value, TraceStartRequest &packet, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("type", packet.type) && o.map("tids", packet.tids);
}

json::Value toJSON(const TraceStartRequest &packet) {
  return json::Value(Object{{"tids", packet.tids}, {"type", packet.type}});
}
/// \}

/// jLLDBTraceStop
/// \{
TraceStopRequest::TraceStopRequest(llvm::StringRef type,
                                   const std::vector<lldb::tid_t> &tids_)
    : type(type) {
  tids.emplace();
  for (lldb::tid_t tid : tids_)
    tids->push_back(static_cast<int64_t>(tid));
}

bool TraceStopRequest::IsProcessTracing() const { return !(bool)tids; }

bool fromJSON(const json::Value &value, TraceStopRequest &packet, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("type", packet.type) && o.map("tids", packet.tids);
}

json::Value toJSON(const TraceStopRequest &packet) {
  return json::Value(Object{{"type", packet.type}, {"tids", packet.tids}});
}
/// \}

/// jLLDBTraceGetState
/// \{
bool fromJSON(const json::Value &value, TraceGetStateRequest &packet,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("type", packet.type);
}

json::Value toJSON(const TraceGetStateRequest &packet) {
  return json::Value(Object{{"type", packet.type}});
}

bool fromJSON(const json::Value &value, TraceBinaryData &packet, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("kind", packet.kind) && o.map("size", packet.size);
}

json::Value toJSON(const TraceBinaryData &packet) {
  return json::Value(Object{{"kind", packet.kind}, {"size", packet.size}});
}

bool fromJSON(const json::Value &value, TraceThreadState &packet, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("tid", packet.tid) &&
         o.map("binaryData", packet.binaryData);
}

json::Value toJSON(const TraceThreadState &packet) {
  return json::Value(
      Object{{"tid", packet.tid}, {"binaryData", packet.binaryData}});
}

bool fromJSON(const json::Value &value, TraceGetStateResponse &packet,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("tracedThreads", packet.tracedThreads) &&
         o.map("processBinaryData", packet.processBinaryData);
}

json::Value toJSON(const TraceGetStateResponse &packet) {
  return json::Value(Object{{"tracedThreads", packet.tracedThreads},
                            {"processBinaryData", packet.processBinaryData}});
}
/// \}

/// jLLDBTraceGetBinaryData
/// \{
json::Value toJSON(const TraceGetBinaryDataRequest &packet) {
  return json::Value(Object{{"type", packet.type},
                            {"kind", packet.kind},
                            {"offset", packet.offset},
                            {"tid", packet.tid},
                            {"size", packet.size}});
}

bool fromJSON(const json::Value &value, TraceGetBinaryDataRequest &packet,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("type", packet.type) && o.map("kind", packet.kind) &&
         o.map("tid", packet.tid) && o.map("offset", packet.offset) &&
         o.map("size", packet.size);
}
/// \}

} // namespace lldb_private
