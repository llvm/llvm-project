//===-- ProtocolRequests.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolRequests.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <utility>

using namespace llvm;

// The 'env' field is either an object as a map of strings or as an array of
// strings formatted like 'key=value'.
static bool parseEnv(const json::Value &Params, StringMap<std::string> &env,
                     json::Path P) {
  const json::Object *O = Params.getAsObject();
  if (!O) {
    P.report("expected object");
    return false;
  }

  const json::Value *value = O->get("env");
  if (!value)
    return true;

  if (const json::Object *env_obj = value->getAsObject()) {
    for (const auto &kv : *env_obj) {
      const std::optional<StringRef> value = kv.second.getAsString();
      if (!value) {
        P.field("env").field(kv.first).report("expected string value");
        return false;
      }
      env.insert({kv.first.str(), value->str()});
    }
    return true;
  }

  if (const json::Array *env_arr = value->getAsArray()) {
    for (size_t i = 0; i < env_arr->size(); ++i) {
      const std::optional<StringRef> value = (*env_arr)[i].getAsString();
      if (!value) {
        P.field("env").index(i).report("expected string");
        return false;
      }
      std::pair<StringRef, StringRef> kv = value->split("=");
      env.insert({kv.first, kv.second.str()});
    }

    return true;
  }

  P.field("env").report("invalid format, expected array or object");
  return false;
}

static bool parseTimeout(const json::Value &Params, std::chrono::seconds &S,
                         json::Path P) {
  const json::Object *O = Params.getAsObject();
  if (!O) {
    P.report("expected object");
    return false;
  }

  const json::Value *value = O->get("timeout");
  if (!value)
    return true;
  std::optional<double> timeout = value->getAsNumber();
  if (!timeout) {
    P.field("timeout").report("expected number");
    return false;
  }

  S = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::duration<double>(*value->getAsNumber()));
  return true;
}

static bool
parseSourceMap(const json::Value &Params,
               std::vector<std::pair<std::string, std::string>> &sourceMap,
               json::Path P) {
  const json::Object *O = Params.getAsObject();
  if (!O) {
    P.report("expected object");
    return false;
  }

  const json::Value *value = O->get("sourceMap");
  if (!value)
    return true;

  if (const json::Object *map_obj = value->getAsObject()) {
    for (const auto &kv : *map_obj) {
      const std::optional<StringRef> value = kv.second.getAsString();
      if (!value) {
        P.field("sourceMap").field(kv.first).report("expected string value");
        return false;
      }
      sourceMap.emplace_back(std::make_pair(kv.first.str(), value->str()));
    }
    return true;
  }

  if (const json::Array *env_arr = value->getAsArray()) {
    for (size_t i = 0; i < env_arr->size(); ++i) {
      const json::Array *kv = (*env_arr)[i].getAsArray();
      if (!kv) {
        P.field("sourceMap").index(i).report("expected array");
        return false;
      }
      if (kv->size() != 2) {
        P.field("sourceMap").index(i).report("expected array of pairs");
        return false;
      }
      const std::optional<StringRef> first = (*kv)[0].getAsString();
      if (!first) {
        P.field("sourceMap").index(0).report("expected string");
        return false;
      }
      const std::optional<StringRef> second = (*kv)[1].getAsString();
      if (!second) {
        P.field("sourceMap").index(1).report("expected string");
        return false;
      }
      sourceMap.emplace_back(std::make_pair(*first, second->str()));
    }

    return true;
  }

  P.report("invalid format, expected array or object");
  return false;
}

namespace lldb_dap::protocol {

bool fromJSON(const llvm::json::Value &Params, CancelArguments &CA,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("requestId", CA.requestId) &&
         O.map("progressId", CA.progressId);
}

bool fromJSON(const json::Value &Params, DisconnectArguments &DA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("restart", DA.restart) &&
         O.map("terminateDebuggee", DA.terminateDebuggee) &&
         O.map("suspendDebuggee", DA.suspendDebuggee);
}

bool fromJSON(const json::Value &Params, PathFormat &PF, json::Path P) {
  auto rawPathFormat = Params.getAsString();
  if (!rawPathFormat) {
    P.report("expected a string");
    return false;
  }

  std::optional<PathFormat> pathFormat =
      StringSwitch<std::optional<PathFormat>>(*rawPathFormat)
          .Case("path", ePatFormatPath)
          .Case("uri", ePathFormatURI)
          .Default(std::nullopt);
  if (!pathFormat) {
    P.report("unexpected value, expected 'path' or 'uri'");
    return false;
  }

  PF = *pathFormat;
  return true;
}

static const StringMap<ClientFeature> ClientFeatureByKey{
    {"supportsVariableType", eClientFeatureVariableType},
    {"supportsVariablePaging", eClientFeatureVariablePaging},
    {"supportsRunInTerminalRequest", eClientFeatureRunInTerminalRequest},
    {"supportsMemoryReferences", eClientFeatureMemoryReferences},
    {"supportsProgressReporting", eClientFeatureProgressReporting},
    {"supportsInvalidatedEvent", eClientFeatureInvalidatedEvent},
    {"supportsMemoryEvent", eClientFeatureMemoryEvent},
    {"supportsArgsCanBeInterpretedByShell",
     eClientFeatureArgsCanBeInterpretedByShell},
    {"supportsStartDebuggingRequest", eClientFeatureStartDebuggingRequest},
    {"supportsANSIStyling", eClientFeatureANSIStyling}};

bool fromJSON(const json::Value &Params, InitializeRequestArguments &IRA,
              json::Path P) {
  json::ObjectMapper OM(Params, P);
  if (!OM)
    return false;

  const json::Object *O = Params.getAsObject();

  for (auto &kv : ClientFeatureByKey) {
    const json::Value *value_ref = O->get(kv.first());
    if (!value_ref)
      continue;

    const std::optional<bool> value = value_ref->getAsBoolean();
    if (!value) {
      P.field(kv.first()).report("expected bool");
      return false;
    }

    if (*value)
      IRA.supportedFeatures.insert(kv.second);
  }

  return OM.map("adapterID", IRA.adapterID) &&
         OM.map("clientID", IRA.clientID) &&
         OM.map("clientName", IRA.clientName) && OM.map("locale", IRA.locale) &&
         OM.map("linesStartAt1", IRA.linesStartAt1) &&
         OM.map("columnsStartAt1", IRA.columnsStartAt1) &&
         OM.map("pathFormat", IRA.pathFormat) &&
         OM.map("$__lldb_sourceInitFile", IRA.lldbExtSourceInitFile);
}

bool fromJSON(const json::Value &Params, Configuration &C, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O.map("debuggerRoot", C.debuggerRoot) &&
         O.mapOptional("enableAutoVariableSummaries",
                       C.enableAutoVariableSummaries) &&
         O.mapOptional("enableSyntheticChildDebugging",
                       C.enableSyntheticChildDebugging) &&
         O.mapOptional("displayExtendedBacktrace",
                       C.displayExtendedBacktrace) &&
         O.mapOptional("commandEscapePrefix", C.commandEscapePrefix) &&
         O.map("customFrameFormat", C.customFrameFormat) &&
         O.map("customThreadFormat", C.customThreadFormat) &&
         O.map("sourcePath", C.sourcePath) &&
         O.mapOptional("initCommands", C.initCommands) &&
         O.mapOptional("preRunCommands", C.preRunCommands) &&
         O.mapOptional("postRunCommands", C.postRunCommands) &&
         O.mapOptional("stopCommands", C.stopCommands) &&
         O.mapOptional("exitCommands", C.exitCommands) &&
         O.mapOptional("terminateCommands", C.terminateCommands) &&
         O.map("program", C.program) && O.map("targetTriple", C.targetTriple) &&
         O.map("platformName", C.platformName) &&
         parseSourceMap(Params, C.sourceMap, P);
}

bool fromJSON(const json::Value &Params, LaunchRequestArguments &LRA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && fromJSON(Params, LRA.configuration, P) &&
         O.mapOptional("noDebug", LRA.noDebug) &&
         O.mapOptional("launchCommands", LRA.launchCommands) &&
         O.map("cwd", LRA.cwd) && O.mapOptional("args", LRA.args) &&
         O.mapOptional("detachOnError", LRA.detachOnError) &&
         O.mapOptional("disableASLR", LRA.disableASLR) &&
         O.mapOptional("disableSTDIO", LRA.disableSTDIO) &&
         O.mapOptional("shellExpandArguments", LRA.shellExpandArguments) &&
         O.mapOptional("stopOnEntry", LRA.stopOnEntry) &&
         O.mapOptional("runInTerminal", LRA.runInTerminal) &&
         parseEnv(Params, LRA.env, P) && parseTimeout(Params, LRA.timeout, P);
}

bool fromJSON(const llvm::json::Value &Params, SetVariableArguments &SVA,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("variablesReference", SVA.variablesReference) &&
         O.map("name", SVA.name) && O.map("value", SVA.value) &&
         O.mapOptional("format", SVA.format);
}

llvm::json::Value toJSON(const SetVariableResponseBody &SVR) {
  json::Object Body{{"value", SVR.value}};
  if (SVR.type.has_value())
    Body.insert({"type", SVR.type});

  if (SVR.variablesReference.has_value())
    Body.insert({"variablesReference", SVR.variablesReference});

  if (SVR.namedVariables.has_value())
    Body.insert({"namedVariables", SVR.namedVariables});

  if (SVR.indexedVariables.has_value())
    Body.insert({"indexedVariables", SVR.indexedVariables});

  if (SVR.memoryReference.has_value())
    Body.insert({"memoryReference", SVR.memoryReference});

  if (SVR.valueLocationReference.has_value())
    Body.insert({"valueLocationReference", SVR.valueLocationReference});

  return llvm::json::Value(std::move(Body));
}

bool fromJSON(const json::Value &Params, SourceArguments &SA, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("source", SA.source) &&
         O.map("sourceReference", SA.sourceReference);
}

json::Value toJSON(const SourceResponseBody &SA) {
  json::Object Result{{"content", SA.content}};

  if (SA.mimeType)
    Result.insert({"mimeType", SA.mimeType});

  return std::move(Result);
}

bool fromJSON(const llvm::json::Value &Params, NextArguments &NA,
              llvm::json::Path P) {
  json::ObjectMapper OM(Params, P);
  return OM && OM.map("threadId", NA.threadId) &&
         OM.mapOptional("singleThread", NA.singleThread) &&
         OM.mapOptional("granularity", NA.granularity);
}

bool fromJSON(const llvm::json::Value &Params, StepInArguments &SIA,
              llvm::json::Path P) {
  json::ObjectMapper OM(Params, P);
  return OM && OM.map("threadId", SIA.threadId) &&
         OM.map("targetId", SIA.targetId) &&
         OM.mapOptional("singleThread", SIA.singleThread) &&
         OM.mapOptional("granularity", SIA.granularity);
}

bool fromJSON(const llvm::json::Value &Params, StepOutArguments &SOA,
              llvm::json::Path P) {
  json::ObjectMapper OM(Params, P);
  return OM && OM.map("threadId", SOA.threadId) &&
         OM.mapOptional("singleThread", SOA.singleThread) &&
         OM.mapOptional("granularity", SOA.granularity);
}

} // namespace lldb_dap::protocol
