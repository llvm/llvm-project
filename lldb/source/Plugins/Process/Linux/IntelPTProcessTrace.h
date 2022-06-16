//===-- IntelPTProcessTrace.h --------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IntelPTProcessTrace_H_
#define liblldb_IntelPTProcessTrace_H_

#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"

#include <memory>

namespace lldb_private {
namespace process_linux {

// Abstract class to be inherited by all the process tracing strategies.
class IntelPTProcessTrace {
public:
  virtual ~IntelPTProcessTrace() = default;

  virtual void ProcessDidStop() {}

  virtual void ProcessWillResume() {}

  /// Construct a minimal jLLDBTraceGetState response for this process trace.
  virtual TraceIntelPTGetStateResponse GetState() = 0;

  virtual bool TracesThread(lldb::tid_t tid) const = 0;

  /// \copydoc IntelPTThreadTraceCollection::TraceStart()
  virtual llvm::Error TraceStart(lldb::tid_t tid) = 0;

  /// \copydoc IntelPTThreadTraceCollection::TraceStop()
  virtual llvm::Error TraceStop(lldb::tid_t tid) = 0;

  /// Get binary data owned by this instance.
  virtual llvm::Expected<std::vector<uint8_t>>
  GetBinaryData(const TraceGetBinaryDataRequest &request) = 0;
};

using IntelPTProcessTraceUP = std::unique_ptr<IntelPTProcessTrace>;

} // namespace process_linux
} // namespace lldb_private

#endif // liblldb_IntelPTProcessTrace_H_
