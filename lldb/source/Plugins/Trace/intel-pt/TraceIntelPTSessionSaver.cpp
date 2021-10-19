//===-- TraceIntelPTSessionSaver.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPTSessionSaver.h"
#include "../common/TraceSessionSaver.h"
#include "TraceIntelPT.h"
#include "TraceIntelPTJSONStructs.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/None.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

llvm::Error TraceIntelPTSessionSaver::SaveToDisk(TraceIntelPT &trace_ipt,
                                                 FileSpec directory) {
  Process *live_process = trace_ipt.GetLiveProcess();
  if (live_process == nullptr)
    return createStringError(inconvertibleErrorCode(),
                             "Saving a trace requires a live process.");

  if (std::error_code ec =
          sys::fs::create_directories(directory.GetPath().c_str()))
    return llvm::errorCodeToError(ec);

  llvm::Expected<JSONTraceIntelPTTrace> json_intel_pt_trace =
      BuildTraceSection(trace_ipt);
  if (!json_intel_pt_trace)
    return json_intel_pt_trace.takeError();

  llvm::Expected<JSONTraceSessionBase> json_session_description =
      TraceSessionSaver::BuildProcessesSection(
          *live_process,
          [&](lldb::tid_t tid)
              -> llvm::Expected<llvm::Optional<std::vector<uint8_t>>> {
            if (!trace_ipt.IsTraced(tid))
              return None;
            return trace_ipt.GetLiveThreadBuffer(tid);
          },
          directory);

  if (!json_session_description)
    return json_session_description.takeError();

  JSONTraceIntelPTSession json_intel_pt_session{json_intel_pt_trace.get(),
                                                json_session_description.get()};

  return TraceSessionSaver::WriteSessionToFile(
      llvm::json::toJSON(json_intel_pt_session), directory);
}

llvm::Expected<JSONTraceIntelPTTrace>
TraceIntelPTSessionSaver::BuildTraceSection(TraceIntelPT &trace_ipt) {
  llvm::Expected<pt_cpu> cpu_info = trace_ipt.GetCPUInfo();
  if (!cpu_info)
    return cpu_info.takeError();

  return JSONTraceIntelPTTrace{"intel-pt",
                               JSONTraceIntelPTCPUInfo(cpu_info.get())};
}
