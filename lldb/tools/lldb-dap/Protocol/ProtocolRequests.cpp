//===-- ProtocolRequests.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolRequests.h"
#include "JSONUtils.h"
#include "lldb/lldb-defines.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Base64.h"
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

bool fromJSON(const json::Value &Params, CancelArguments &CA, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("requestId", CA.requestId) &&
         O.map("progressId", CA.progressId);
}

bool fromJSON(const json::Value &Params, DisconnectArguments &DA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("restart", DA.restart) &&
         O.mapOptional("terminateDebuggee", DA.terminateDebuggee) &&
         O.mapOptional("suspendDebuggee", DA.suspendDebuggee);
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
         OM.mapOptional("pathFormat", IRA.pathFormat) &&
         OM.map("$__lldb_sourceInitFile", IRA.lldbExtSourceInitFile);
}

bool fromJSON(const json::Value &Params, Configuration &C, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O.mapOptional("debuggerRoot", C.debuggerRoot) &&
         O.mapOptional("enableAutoVariableSummaries",
                       C.enableAutoVariableSummaries) &&
         O.mapOptional("enableSyntheticChildDebugging",
                       C.enableSyntheticChildDebugging) &&
         O.mapOptional("displayExtendedBacktrace",
                       C.displayExtendedBacktrace) &&
         O.mapOptional("stopOnEntry", C.stopOnEntry) &&
         O.mapOptional("commandEscapePrefix", C.commandEscapePrefix) &&
         O.mapOptional("customFrameFormat", C.customFrameFormat) &&
         O.mapOptional("customThreadFormat", C.customThreadFormat) &&
         O.mapOptional("sourcePath", C.sourcePath) &&
         O.mapOptional("initCommands", C.initCommands) &&
         O.mapOptional("preRunCommands", C.preRunCommands) &&
         O.mapOptional("postRunCommands", C.postRunCommands) &&
         O.mapOptional("stopCommands", C.stopCommands) &&
         O.mapOptional("exitCommands", C.exitCommands) &&
         O.mapOptional("terminateCommands", C.terminateCommands) &&
         O.mapOptional("program", C.program) &&
         O.mapOptional("targetTriple", C.targetTriple) &&
         O.mapOptional("platformName", C.platformName) &&
         parseSourceMap(Params, C.sourceMap, P) &&
         parseTimeout(Params, C.timeout, P);
}

bool fromJSON(const json::Value &Params, BreakpointLocationsArguments &BLA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("source", BLA.source) && O.map("line", BLA.line) &&
         O.mapOptional("column", BLA.column) &&
         O.mapOptional("endLine", BLA.endLine) &&
         O.mapOptional("endColumn", BLA.endColumn);
}

json::Value toJSON(const BreakpointLocationsResponseBody &BLRB) {
  return json::Object{{"breakpoints", BLRB.breakpoints}};
}

bool fromJSON(const json::Value &Params, Console &C, json::Path P) {
  auto oldFormatConsole = Params.getAsBoolean();
  if (oldFormatConsole) {
    C = *oldFormatConsole ? eConsoleIntegratedTerminal : eConsoleInternal;
    return true;
  }
  auto newFormatConsole = Params.getAsString();
  if (!newFormatConsole) {
    P.report("expected a string");
    return false;
  }

  std::optional<Console> console =
      StringSwitch<std::optional<Console>>(*newFormatConsole)
          .Case("internalConsole", eConsoleInternal)
          .Case("integratedTerminal", eConsoleIntegratedTerminal)
          .Case("externalTerminal", eConsoleExternalTerminal)
          .Default(std::nullopt);
  if (!console) {
    P.report("unexpected value, expected 'internalConsole', "
             "'integratedTerminal' or 'externalTerminal'");
    return false;
  }

  C = *console;
  return true;
}

bool fromJSON(const json::Value &Params, LaunchRequestArguments &LRA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && fromJSON(Params, LRA.configuration, P) &&
         O.mapOptional("noDebug", LRA.noDebug) &&
         O.mapOptional("launchCommands", LRA.launchCommands) &&
         O.mapOptional("cwd", LRA.cwd) && O.mapOptional("args", LRA.args) &&
         O.mapOptional("detachOnError", LRA.detachOnError) &&
         O.mapOptional("disableASLR", LRA.disableASLR) &&
         O.mapOptional("disableSTDIO", LRA.disableSTDIO) &&
         O.mapOptional("shellExpandArguments", LRA.shellExpandArguments) &&
         O.mapOptional("runInTerminal", LRA.console) &&
         O.mapOptional("console", LRA.console) &&
         O.mapOptional("stdio", LRA.stdio) && parseEnv(Params, LRA.env, P);
}

bool fromJSON(const json::Value &Params, AttachRequestArguments &ARA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && fromJSON(Params, ARA.configuration, P) &&
         O.mapOptional("attachCommands", ARA.attachCommands) &&
         O.mapOptional("pid", ARA.pid) &&
         O.mapOptional("waitFor", ARA.waitFor) &&
         O.mapOptional("gdb-remote-port", ARA.gdbRemotePort) &&
         O.mapOptional("gdb-remote-hostname", ARA.gdbRemoteHostname) &&
         O.mapOptional("coreFile", ARA.coreFile);
}

bool fromJSON(const json::Value &Params, ContinueArguments &CA, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("threadId", CA.threadId) &&
         O.mapOptional("singleThread", CA.singleThread);
}

json::Value toJSON(const ContinueResponseBody &CRB) {
  json::Object Body{{"allThreadsContinued", CRB.allThreadsContinued}};
  return std::move(Body);
}

bool fromJSON(const json::Value &Params, CompletionsArguments &CA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("text", CA.text) && O.map("column", CA.column) &&
         O.mapOptional("frameId", CA.frameId) && O.mapOptional("line", CA.line);
}

json::Value toJSON(const CompletionsResponseBody &CRB) {
  return json::Object{{"targets", CRB.targets}};
}

bool fromJSON(const json::Value &Params, SetVariableArguments &SVA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("variablesReference", SVA.variablesReference) &&
         O.map("name", SVA.name) && O.map("value", SVA.value) &&
         O.mapOptional("format", SVA.format);
}

json::Value toJSON(const SetVariableResponseBody &SVR) {
  json::Object Body{{"value", SVR.value}};

  if (!SVR.type.empty())
    Body.insert({"type", SVR.type});
  if (SVR.variablesReference)
    Body.insert({"variablesReference", SVR.variablesReference});
  if (SVR.namedVariables)
    Body.insert({"namedVariables", SVR.namedVariables});
  if (SVR.indexedVariables)
    Body.insert({"indexedVariables", SVR.indexedVariables});
  if (SVR.memoryReference != LLDB_INVALID_ADDRESS)
    Body.insert(
        {"memoryReference", EncodeMemoryReference(SVR.memoryReference)});
  if (SVR.valueLocationReference)
    Body.insert({"valueLocationReference", SVR.valueLocationReference});

  return json::Value(std::move(Body));
}

bool fromJSON(const json::Value &Params, ScopesArguments &SCA, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("frameId", SCA.frameId);
}

json::Value toJSON(const ScopesResponseBody &SCR) {
  return json::Object{{"scopes", SCR.scopes}};
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

bool fromJSON(const json::Value &Params, NextArguments &NA, json::Path P) {
  json::ObjectMapper OM(Params, P);
  return OM && OM.map("threadId", NA.threadId) &&
         OM.mapOptional("singleThread", NA.singleThread) &&
         OM.mapOptional("granularity", NA.granularity);
}

bool fromJSON(const json::Value &Params, StepInArguments &SIA, json::Path P) {
  json::ObjectMapper OM(Params, P);
  return OM && OM.map("threadId", SIA.threadId) &&
         OM.map("targetId", SIA.targetId) &&
         OM.mapOptional("singleThread", SIA.singleThread) &&
         OM.mapOptional("granularity", SIA.granularity);
}

bool fromJSON(const llvm::json::Value &Params, StepInTargetsArguments &SITA,
              llvm::json::Path P) {
  json::ObjectMapper OM(Params, P);
  return OM && OM.map("frameId", SITA.frameId);
}

llvm::json::Value toJSON(const StepInTargetsResponseBody &SITR) {
  return llvm::json::Object{{"targets", SITR.targets}};
}

bool fromJSON(const json::Value &Params, StepOutArguments &SOA, json::Path P) {
  json::ObjectMapper OM(Params, P);
  return OM && OM.map("threadId", SOA.threadId) &&
         OM.mapOptional("singleThread", SOA.singleThread) &&
         OM.mapOptional("granularity", SOA.granularity);
}

bool fromJSON(const json::Value &Params, SetBreakpointsArguments &SBA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("source", SBA.source) &&
         O.map("breakpoints", SBA.breakpoints) && O.map("lines", SBA.lines) &&
         O.map("sourceModified", SBA.sourceModified);
}

json::Value toJSON(const SetBreakpointsResponseBody &SBR) {
  return json::Object{{"breakpoints", SBR.breakpoints}};
}

bool fromJSON(const json::Value &Params, SetFunctionBreakpointsArguments &SFBA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("breakpoints", SFBA.breakpoints);
}

json::Value toJSON(const SetFunctionBreakpointsResponseBody &SFBR) {
  return json::Object{{"breakpoints", SFBR.breakpoints}};
}

bool fromJSON(const json::Value &Params,
              SetInstructionBreakpointsArguments &SIBA, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("breakpoints", SIBA.breakpoints);
}

json::Value toJSON(const SetInstructionBreakpointsResponseBody &SIBR) {
  return json::Object{{"breakpoints", SIBR.breakpoints}};
}

bool fromJSON(const json::Value &Params, DataBreakpointInfoArguments &DBIA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("variablesReference", DBIA.variablesReference) &&
         O.map("name", DBIA.name) && O.mapOptional("frameId", DBIA.frameId) &&
         O.map("bytes", DBIA.bytes) && O.map("asAddress", DBIA.asAddress) &&
         O.map("mode", DBIA.mode);
}

json::Value toJSON(const DataBreakpointInfoResponseBody &DBIRB) {
  json::Object result{{"dataId", DBIRB.dataId},
                      {"description", DBIRB.description}};

  if (DBIRB.accessTypes)
    result["accessTypes"] = *DBIRB.accessTypes;
  if (DBIRB.canPersist)
    result["canPersist"] = *DBIRB.canPersist;

  return result;
}

bool fromJSON(const json::Value &Params, SetDataBreakpointsArguments &SDBA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("breakpoints", SDBA.breakpoints);
}

json::Value toJSON(const SetDataBreakpointsResponseBody &SDBR) {
  return json::Object{{"breakpoints", SDBR.breakpoints}};
}

bool fromJSON(const json::Value &Params, SetExceptionBreakpointsArguments &Args,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("filters", Args.filters) &&
         O.mapOptional("filterOptions", Args.filterOptions);
}

json::Value toJSON(const SetExceptionBreakpointsResponseBody &B) {
  json::Object result;
  if (!B.breakpoints.empty())
    result.insert({"breakpoints", B.breakpoints});
  return result;
}

json::Value toJSON(const ThreadsResponseBody &TR) {
  return json::Object{{"threads", TR.threads}};
}

bool fromJSON(const llvm::json::Value &Params, DisassembleArguments &DA,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O &&
         DecodeMemoryReference(Params, "memoryReference", DA.memoryReference, P,
                               /*required=*/true) &&
         O.mapOptional("offset", DA.offset) &&
         O.mapOptional("instructionOffset", DA.instructionOffset) &&
         O.map("instructionCount", DA.instructionCount) &&
         O.mapOptional("resolveSymbols", DA.resolveSymbols);
}

json::Value toJSON(const DisassembleResponseBody &DRB) {
  return json::Object{{"instructions", DRB.instructions}};
}

bool fromJSON(const json::Value &Params, ReadMemoryArguments &RMA,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O &&
         DecodeMemoryReference(Params, "memoryReference", RMA.memoryReference,
                               P, /*required=*/true) &&
         O.map("count", RMA.count) && O.mapOptional("offset", RMA.offset);
}

json::Value toJSON(const ReadMemoryResponseBody &RMR) {
  json::Object result{{"address", EncodeMemoryReference(RMR.address)}};

  if (RMR.unreadableBytes != 0)
    result.insert({"unreadableBytes", RMR.unreadableBytes});
  if (!RMR.data.empty())
    result.insert({"data", llvm::encodeBase64(RMR.data)});

  return result;
}

bool fromJSON(const json::Value &Params, ModulesArguments &MA, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("startModule", MA.startModule) &&
         O.mapOptional("moduleCount", MA.moduleCount);
}

json::Value toJSON(const ModulesResponseBody &MR) {
  json::Object result{{"modules", MR.modules}};
  if (MR.totalModules != 0)
    result.insert({"totalModules", MR.totalModules});

  return result;
}

bool fromJSON(const json::Value &Param, VariablesArguments::VariablesFilter &VA,
              json::Path Path) {
  auto rawFilter = Param.getAsString();
  if (!rawFilter) {
    Path.report("expected a string");
    return false;
  }
  std::optional<VariablesArguments::VariablesFilter> filter =
      StringSwitch<std::optional<VariablesArguments::VariablesFilter>>(
          *rawFilter)
          .Case("indexed", VariablesArguments::eVariablesFilterIndexed)
          .Case("named", VariablesArguments::eVariablesFilterNamed)
          .Default(std::nullopt);
  if (!filter) {
    Path.report("unexpected value, expected 'named' or 'indexed'");
    return false;
  }

  VA = *filter;
  return true;
}

bool fromJSON(const json::Value &Param, VariablesArguments &VA,
              json::Path Path) {
  json::ObjectMapper O(Param, Path);
  return O && O.map("variablesReference", VA.variablesReference) &&
         O.mapOptional("filter", VA.filter) &&
         O.mapOptional("start", VA.start) && O.mapOptional("count", VA.count) &&
         O.mapOptional("format", VA.format);
}

json::Value toJSON(const VariablesResponseBody &VRB) {
  return json::Object{{"variables", VRB.variables}};
}

bool fromJSON(const json::Value &Params, WriteMemoryArguments &WMA,
              json::Path P) {
  json::ObjectMapper O(Params, P);

  return O &&
         DecodeMemoryReference(Params, "memoryReference", WMA.memoryReference,
                               P, /*required=*/true) &&
         O.mapOptional("allowPartial", WMA.allowPartial) &&
         O.mapOptional("offset", WMA.offset) && O.map("data", WMA.data);
}

json::Value toJSON(const WriteMemoryResponseBody &WMR) {
  json::Object result;

  if (WMR.bytesWritten != 0)
    result.insert({"bytesWritten", WMR.bytesWritten});
  return result;
}

bool fromJSON(const llvm::json::Value &Params, ModuleSymbolsArguments &Args,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("moduleId", Args.moduleId) &&
         O.map("moduleName", Args.moduleName) &&
         O.mapOptional("startIndex", Args.startIndex) &&
         O.mapOptional("count", Args.count);
}

llvm::json::Value toJSON(const ModuleSymbolsResponseBody &DGMSR) {
  json::Object result;
  result.insert({"symbols", DGMSR.symbols});
  return result;
}

} // namespace lldb_dap::protocol
