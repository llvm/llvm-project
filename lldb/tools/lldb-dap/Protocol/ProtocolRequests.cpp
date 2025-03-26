//===-- ProtocolRequests.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolRequests.h"
#include "DAP.h"
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
  json::ObjectMapper OM(Params, P);
  if (!OM)
    return false;

  const json::Object *O = Params.getAsObject();
  if (std::optional<bool> v = O->getBoolean("supportsVariableType"); v && *v)
    IRA.supportedFeatures.insert(ClientFeature::supportsVariableType);
  if (std::optional<bool> v = O->getBoolean("supportsVariablePaging"); v && *v)
    IRA.supportedFeatures.insert(ClientFeature::supportsVariablePaging);
  if (std::optional<bool> v = O->getBoolean("supportsRunInTerminalRequest");
      v && *v)
    IRA.supportedFeatures.insert(ClientFeature::supportsRunInTerminalRequest);
  if (std::optional<bool> v = O->getBoolean("supportsMemoryReferences");
      v && *v)
    IRA.supportedFeatures.insert(ClientFeature::supportsMemoryReferences);
  if (std::optional<bool> v = O->getBoolean("supportsProgressReporting");
      v && *v)
    IRA.supportedFeatures.insert(ClientFeature::supportsProgressReporting);
  if (std::optional<bool> v = O->getBoolean("supportsInvalidatedEvent");
      v && *v)
    IRA.supportedFeatures.insert(ClientFeature::supportsInvalidatedEvent);
  if (std::optional<bool> v = O->getBoolean("supportsMemoryEvent"); v && *v)
    IRA.supportedFeatures.insert(ClientFeature::supportsMemoryEvent);
  if (std::optional<bool> v =
          O->getBoolean("supportsArgsCanBeInterpretedByShell");
      v && *v)
    IRA.supportedFeatures.insert(
        ClientFeature::supportsArgsCanBeInterpretedByShell);
  if (std::optional<bool> v = O->getBoolean("supportsStartDebuggingRequest");
      v && *v)
    IRA.supportedFeatures.insert(ClientFeature::supportsStartDebuggingRequest);
  if (std::optional<bool> v = O->getBoolean("supportsANSIStyling"); v && *v)
    IRA.supportedFeatures.insert(ClientFeature::supportsANSIStyling);

  return OM.mapOptional("adatperID", IRA.adatperID) &&
         OM.mapOptional("clientID", IRA.clientID) &&
         OM.mapOptional("clientName", IRA.clientName) &&
         OM.mapOptional("locale", IRA.locale) &&
         OM.mapOptional("linesStartAt1", IRA.linesStartAt1) &&
         OM.mapOptional("columnsStartAt1", IRA.columnsStartAt1) &&
         OM.mapOptional("pathFormat", IRA.pathFormat) &&
         OM.mapOptional("$__lldb_sourceInitFile", IRA.lldbExtSourceInitFile);
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
