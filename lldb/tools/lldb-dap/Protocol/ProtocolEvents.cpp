//===-- ProtocolEvents.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolEvents.h"
#include "JSONUtils.h"
#include "lldb/lldb-defines.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"

using namespace llvm;

namespace lldb_dap::protocol {

json::Value toJSON(const CapabilitiesEventBody &CEB) {
  return json::Object{{"capabilities", CEB.capabilities}};
}

json::Value toJSON(const ModuleEventBody::Reason &MEBR) {
  switch (MEBR) {
  case ModuleEventBody::eReasonNew:
    return "new";
  case ModuleEventBody::eReasonChanged:
    return "changed";
  case ModuleEventBody::eReasonRemoved:
    return "removed";
  }
  llvm_unreachable("unhandled module event reason!.");
}

json::Value toJSON(const ModuleEventBody &MEB) {
  return json::Object{{"reason", MEB.reason}, {"module", MEB.module}};
}

llvm::json::Value toJSON(const InvalidatedEventBody::Area &IEBA) {
  switch (IEBA) {
  case InvalidatedEventBody::eAreaAll:
    return "all";
  case InvalidatedEventBody::eAreaStacks:
    return "stacks";
  case InvalidatedEventBody::eAreaThreads:
    return "threads";
  case InvalidatedEventBody::eAreaVariables:
    return "variables";
  }
  llvm_unreachable("unhandled invalidated event area!.");
}

llvm::json::Value toJSON(const InvalidatedEventBody &IEB) {
  json::Object Result{{"areas", IEB.areas}};
  if (IEB.threadId)
    Result.insert({"threadId", IEB.threadId});
  if (IEB.stackFrameId)
    Result.insert({"stackFrameId", IEB.stackFrameId});
  return Result;
}

llvm::json::Value toJSON(const MemoryEventBody &MEB) {
  return json::Object{
      {"memoryReference", EncodeMemoryReference(MEB.memoryReference)},
      {"offset", MEB.offset},
      {"count", MEB.count}};
}

static llvm::json::Value toJSON(const StoppedReason &SR) {
  assert(SR != eStoppedReasonUninitialized && "StopReason Uninitialized");
  switch (SR) {
  case eStoppedReasonUninitialized:
    return "";
  case eStoppedReasonStep:
    return "step";
  case eStoppedReasonBreakpoint:
    return "breakpoint";
  case eStoppedReasonException:
    return "exception";
  case eStoppedReasonPause:
    return "pause";
  case eStoppedReasonEntry:
    return "entry";
  case eStoppedReasonGoto:
    return "goto";
  case eStoppedReasonFunctionBreakpoint:
    return "function breakpoint";
  case eStoppedReasonDataBreakpoint:
    return "data breakpoint";
  case eStoppedReasonInstructionBreakpoint:
    return "instruction breakpoint";
  }
}

llvm::json::Value toJSON(const StoppedEventBody &SEB) {
  llvm::json::Object Result{{"reason", SEB.reason}};

  if (!SEB.description.empty())
    Result.insert({"description", SEB.description});
  if (SEB.threadId != LLDB_INVALID_THREAD_ID)
    Result.insert({"threadId", SEB.threadId});
  if (SEB.preserveFocusHint)
    Result.insert({"preserveFocusHint", SEB.preserveFocusHint});
  if (!SEB.text.empty())
    Result.insert({"text", SEB.text});
  if (SEB.allThreadsStopped)
    Result.insert({"allThreadsStopped", SEB.allThreadsStopped});
  if (!SEB.hitBreakpointIds.empty())
    Result.insert({"hitBreakpointIds", SEB.hitBreakpointIds});

  return Result;
}

} // namespace lldb_dap::protocol
