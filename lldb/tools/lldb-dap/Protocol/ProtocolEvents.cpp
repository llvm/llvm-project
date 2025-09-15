//===-- ProtocolEvents.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolEvents.h"
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
    Result.insert({"threadID", IEB.threadId});
  if (IEB.frameId)
    Result.insert({"frameId", IEB.frameId});
  return Result;
}

} // namespace lldb_dap::protocol
