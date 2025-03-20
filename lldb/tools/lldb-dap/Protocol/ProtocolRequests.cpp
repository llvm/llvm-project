//===-- ProtocolRequests.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolRequests.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <utility>

using namespace llvm;

namespace lldb_dap::protocol {

bool fromJSON(const json::Value &Params, DisconnectArguments &DA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("restart", DA.restart) &&
         O.mapOptional("terminateDebuggee", DA.terminateDebuggee) &&
         O.mapOptional("suspendDebuggee", DA.suspendDebuggee);
}
bool fromJSON(const llvm::json::Value &Params, GotoArguments &GA,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("targetId", GA.targetId) && O.map("threadId", GA.threadId);
}

bool fromJSON(const llvm::json::Value &Params, GotoTargetsArguments &GTA,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("source", GTA.source) && O.map("line", GTA.line) &&
         O.mapOptional("column", GTA.column);
}

llvm::json::Value toJSON(const GotoTargetsResponseBody &GTA) {
  json::Array targets;
  for (const auto &target : GTA.targets) {
    targets.emplace_back(target);
  }
  return json::Object{{"targets", std::move(targets)}};
}

bool fromJSON(const json::Value &Params, SourceArguments &SA, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("source", SA.source) &&
         O.map("sourceReference", SA.sourceReference);
}

json::Value toJSON(const SourceResponseBody &SA) {
  json::Object Result{{"content", SA.content}};

  if (SA.mimeType)
    Result.insert({"mimeType", SA.mimeType});

  return std::move(Result);
}

} // namespace lldb_dap::protocol
