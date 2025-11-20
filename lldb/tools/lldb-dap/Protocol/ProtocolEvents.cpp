//===-- ProtocolEvents.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolEvents.h"
#include "JSONUtils.h"
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

static llvm::json::Value toJSON(const OutputCategory &OC) {
  switch (OC) {
  case eOutputCategoryConsole:
    return "console";
  case eOutputCategoryImportant:
    return "important";
  case eOutputCategoryStdout:
    return "stdout";
  case eOutputCategoryStderr:
    return "stderr";
  case eOutputCategoryTelemetry:
    return "telemetry";
  }
  llvm_unreachable("unhandled output category!.");
}

static llvm::json::Value toJSON(const OutputGroup &OG) {
  switch (OG) {
  case eOutputGroupStart:
    return "start";
  case eOutputGroupStartCollapsed:
    return "startCollapsed";
  case eOutputGroupEnd:
    return "end";
  case eOutputGroupNone:
    break;
  }
  llvm_unreachable("unhandled output category!.");
}

llvm::json::Value toJSON(const OutputEventBody &OEB) {
  json::Object Result{{"output", OEB.output}, {"category", OEB.category}};

  if (OEB.group != eOutputGroupNone)
    Result.insert({"group", OEB.group});
  if (OEB.variablesReference)
    Result.insert({"variablesReference", OEB.variablesReference});
  if (OEB.source)
    Result.insert({"source", OEB.source});
  if (OEB.line != LLDB_INVALID_LINE_NUMBER)
    Result.insert({"line", OEB.line});
  if (OEB.column != LLDB_INVALID_COLUMN_NUMBER)
    Result.insert({"column", OEB.column});
  if (OEB.data)
    Result.insert({"data", OEB.data});
  if (OEB.locationReference)
    Result.insert({"locationReference", OEB.locationReference});

  return Result;
}

} // namespace lldb_dap::protocol
