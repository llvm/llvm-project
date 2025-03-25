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

bool fromJSON(const llvm::json::Value &Params,
              InitializeRequestArguments::PathFormat &PF, llvm::json::Path P) {
  auto rawPathFormat = Params.getAsString();
  if (!rawPathFormat) {
    P.report("expected a string");
    return false;
  }

  std::optional<InitializeRequestArguments::PathFormat> pathFormat =
      StringSwitch<std::optional<InitializeRequestArguments::PathFormat>>(
          *rawPathFormat)
          .Case("path", InitializeRequestArguments::PathFormat::path)
          .Case("uri", InitializeRequestArguments::PathFormat::uri)
          .Default(std::nullopt);
  if (!pathFormat) {
    P.report("unexpected value, expected 'path' or 'uri'");
    return false;
  }

  PF = *pathFormat;
  return true;
}

bool fromJSON(const llvm::json::Value &Params, InitializeRequestArguments &IRA,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("adatperID", IRA.adatperID) &&
         O.mapOptional("clientID", IRA.clientID) &&
         O.mapOptional("clientName", IRA.clientName) &&
         O.mapOptional("locale", IRA.locale) &&
         O.mapOptional("linesStartAt1", IRA.linesStartAt1) &&
         O.mapOptional("columnsStartAt1", IRA.columnsStartAt1) &&
         O.mapOptional("pathFormat", IRA.pathFormat) &&
         O.mapOptional("supportsVariableType", IRA.supportsVariableType) &&
         O.mapOptional("supportsVariablePaging", IRA.supportsVariablePaging) &&
         O.mapOptional("supportsRunInTerminalRequest",
                       IRA.supportsRunInTerminalRequest) &&
         O.mapOptional("supportsMemoryReferences",
                       IRA.supportsMemoryReferences) &&
         O.mapOptional("supportsProgressReporting",
                       IRA.supportsProgressReporting) &&
         O.mapOptional("supportsInvalidatedEvent",
                       IRA.supportsInvalidatedEvent) &&
         O.mapOptional("supportsMemoryEvent", IRA.supportsMemoryEvent) &&
         O.mapOptional("supportsArgsCanBeInterpretedByShell",
                       IRA.supportsArgsCanBeInterpretedByShell) &&
         O.mapOptional("supportsStartDebuggingRequest",
                       IRA.supportsStartDebuggingRequest) &&
         O.mapOptional("supportsANSIStyling", IRA.supportsANSIStyling) &&
         O.mapOptional("$__lldb_sourceInitFile", IRA.sourceInitFile);
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
