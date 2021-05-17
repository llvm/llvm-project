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

bool fromJSON(const json::Value &value, TraceIntelPTStartRequest &packet,
              Path path) {
  ObjectMapper o(value, path);
  if (!o || !fromJSON(value, (TraceStartRequest &)packet, path) ||
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
  return base;
}

} // namespace lldb_private
