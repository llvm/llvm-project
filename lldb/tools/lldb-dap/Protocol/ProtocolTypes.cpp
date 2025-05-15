//===-- ProtocolTypes.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include <optional>

using namespace llvm;

namespace lldb_dap::protocol {

bool fromJSON(const json::Value &Params, PresentationHint &PH, json::Path P) {
  auto rawHint = Params.getAsString();
  if (!rawHint) {
    P.report("expected a string");
    return false;
  }
  std::optional<PresentationHint> hint =
      StringSwitch<std::optional<PresentationHint>>(*rawHint)
          .Case("normal", ePresentationHintNormal)
          .Case("emphasize", ePresentationHintEmphasize)
          .Case("deemphasize", ePresentationHintDeemphasize)
          .Default(std::nullopt);
  if (!hint) {
    P.report("unexpected value");
    return false;
  }
  PH = *hint;
  return true;
}

bool fromJSON(const json::Value &Params, Source &S, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("name", S.name) && O.map("path", S.path) &&
         O.map("presentationHint", S.presentationHint) &&
         O.map("sourceReference", S.sourceReference);
}

llvm::json::Value toJSON(PresentationHint hint) {
  switch (hint) {
  case ePresentationHintNormal:
    return "normal";
  case ePresentationHintEmphasize:
    return "emphasize";
  case ePresentationHintDeemphasize:
    return "deemphasize";
  }
  llvm_unreachable("unhandled presentation hint.");
}

llvm::json::Value toJSON(const Source &S) {
  json::Object result;
  if (S.name)
    result.insert({"name", *S.name});
  if (S.path)
    result.insert({"path", *S.path});
  if (S.sourceReference)
    result.insert({"sourceReference", *S.sourceReference});
  if (S.presentationHint)
    result.insert({"presentationHint", *S.presentationHint});

  return result;
}

bool fromJSON(const llvm::json::Value &Params, ExceptionBreakpointsFilter &EBF,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("filter", EBF.filter) && O.map("label", EBF.label) &&
         O.mapOptional("description", EBF.description) &&
         O.mapOptional("default", EBF.defaultState) &&
         O.mapOptional("supportsCondition", EBF.supportsCondition) &&
         O.mapOptional("conditionDescription", EBF.conditionDescription);
}

json::Value toJSON(const ExceptionBreakpointsFilter &EBF) {
  json::Object result{{"filter", EBF.filter}, {"label", EBF.label}};

  if (EBF.description)
    result.insert({"description", *EBF.description});
  if (EBF.defaultState)
    result.insert({"default", *EBF.defaultState});
  if (EBF.supportsCondition)
    result.insert({"supportsCondition", *EBF.supportsCondition});
  if (EBF.conditionDescription)
    result.insert({"conditionDescription", *EBF.conditionDescription});

  return result;
}

bool fromJSON(const json::Value &Params, ColumnType &CT, json::Path P) {
  auto rawColumnType = Params.getAsString();
  if (!rawColumnType) {
    P.report("expected a string");
    return false;
  }
  std::optional<ColumnType> columnType =
      StringSwitch<std::optional<ColumnType>>(*rawColumnType)
          .Case("string", eColumnTypeString)
          .Case("number", eColumnTypeNumber)
          .Case("boolean", eColumnTypeBoolean)
          .Case("unixTimestampUTC", eColumnTypeTimestamp)
          .Default(std::nullopt);
  if (!columnType) {
    P.report("unexpected value, expected 'string', 'number',  'boolean', or "
             "'unixTimestampUTC'");
    return false;
  }
  CT = *columnType;
  return true;
}

json::Value toJSON(const ColumnType &T) {
  switch (T) {
  case eColumnTypeString:
    return "string";
  case eColumnTypeNumber:
    return "number";
  case eColumnTypeBoolean:
    return "boolean";
  case eColumnTypeTimestamp:
    return "unixTimestampUTC";
  }
  llvm_unreachable("unhandled column type.");
}

bool fromJSON(const llvm::json::Value &Params, ColumnDescriptor &CD,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("attributeName", CD.attributeName) &&
         O.map("label", CD.label) && O.mapOptional("format", CD.format) &&
         O.mapOptional("type", CD.type) && O.mapOptional("width", CD.width);
}

json::Value toJSON(const ColumnDescriptor &CD) {
  json::Object result{{"attributeName", CD.attributeName}, {"label", CD.label}};

  if (CD.format)
    result.insert({"format", *CD.format});
  if (CD.type)
    result.insert({"type", *CD.type});
  if (CD.width)
    result.insert({"width", *CD.width});

  return result;
}

json::Value toJSON(const ChecksumAlgorithm &CA) {
  switch (CA) {
  case eChecksumAlgorithmMD5:
    return "MD5";
  case eChecksumAlgorithmSHA1:
    return "SHA1";
  case eChecksumAlgorithmSHA256:
    return "SHA256";
  case eChecksumAlgorithmTimestamp:
    return "timestamp";
  }
  llvm_unreachable("unhandled checksum algorithm.");
}

bool fromJSON(const llvm::json::Value &Params, ChecksumAlgorithm &CA,
              llvm::json::Path P) {
  auto rawAlgorithm = Params.getAsString();
  if (!rawAlgorithm) {
    P.report("expected a string");
    return false;
  }

  std::optional<ChecksumAlgorithm> algorithm =
      llvm::StringSwitch<std::optional<ChecksumAlgorithm>>(*rawAlgorithm)
          .Case("MD5", eChecksumAlgorithmMD5)
          .Case("SHA1", eChecksumAlgorithmSHA1)
          .Case("SHA256", eChecksumAlgorithmSHA256)
          .Case("timestamp", eChecksumAlgorithmTimestamp)
          .Default(std::nullopt);

  if (!algorithm) {
    P.report(
        "unexpected value, expected 'MD5', 'SHA1', 'SHA256', or 'timestamp'");
    return false;
  }

  CA = *algorithm;
  return true;
}

json::Value toJSON(const BreakpointModeApplicability &BMA) {
  switch (BMA) {
  case eBreakpointModeApplicabilitySource:
    return "source";
  case eBreakpointModeApplicabilityException:
    return "exception";
  case eBreakpointModeApplicabilityData:
    return "data";
  case eBreakpointModeApplicabilityInstruction:
    return "instruction";
  }
  llvm_unreachable("unhandled breakpoint mode applicability.");
}

bool fromJSON(const llvm::json::Value &Params, BreakpointModeApplicability &BMA,
              llvm::json::Path P) {
  auto rawApplicability = Params.getAsString();
  if (!rawApplicability) {
    P.report("expected a string");
    return false;
  }
  std::optional<BreakpointModeApplicability> applicability =
      llvm::StringSwitch<std::optional<BreakpointModeApplicability>>(
          *rawApplicability)
          .Case("source", eBreakpointModeApplicabilitySource)
          .Case("exception", eBreakpointModeApplicabilityException)
          .Case("data", eBreakpointModeApplicabilityData)
          .Case("instruction", eBreakpointModeApplicabilityInstruction)
          .Default(std::nullopt);
  if (!applicability) {
    P.report("unexpected value, expected 'source', 'exception', 'data', or "
             "'instruction'");
    return false;
  }
  BMA = *applicability;
  return true;
}

json::Value toJSON(const BreakpointMode &BM) {
  json::Object result{
      {"mode", BM.mode},
      {"label", BM.label},
      {"appliesTo", BM.appliesTo},
  };

  if (BM.description)
    result.insert({"description", *BM.description});

  return result;
}

bool fromJSON(const llvm::json::Value &Params, BreakpointMode &BM,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("mode", BM.mode) && O.map("label", BM.label) &&
         O.mapOptional("description", BM.description) &&
         O.map("appliesTo", BM.appliesTo);
}

static llvm::StringLiteral ToString(AdapterFeature feature) {
  switch (feature) {
  case eAdapterFeatureANSIStyling:
    return "supportsANSIStyling";
  case eAdapterFeatureBreakpointLocationsRequest:
    return "supportsBreakpointLocationsRequest";
  case eAdapterFeatureCancelRequest:
    return "supportsCancelRequest";
  case eAdapterFeatureClipboardContext:
    return "supportsClipboardContext";
  case eAdapterFeatureCompletionsRequest:
    return "supportsCompletionsRequest";
  case eAdapterFeatureConditionalBreakpoints:
    return "supportsConditionalBreakpoints";
  case eAdapterFeatureConfigurationDoneRequest:
    return "supportsConfigurationDoneRequest";
  case eAdapterFeatureDataBreakpointBytes:
    return "supportsDataBreakpointBytes";
  case eAdapterFeatureDataBreakpoints:
    return "supportsDataBreakpoints";
  case eAdapterFeatureDelayedStackTraceLoading:
    return "supportsDelayedStackTraceLoading";
  case eAdapterFeatureDisassembleRequest:
    return "supportsDisassembleRequest";
  case eAdapterFeatureEvaluateForHovers:
    return "supportsEvaluateForHovers";
  case eAdapterFeatureExceptionFilterOptions:
    return "supportsExceptionFilterOptions";
  case eAdapterFeatureExceptionInfoRequest:
    return "supportsExceptionInfoRequest";
  case eAdapterFeatureExceptionOptions:
    return "supportsExceptionOptions";
  case eAdapterFeatureFunctionBreakpoints:
    return "supportsFunctionBreakpoints";
  case eAdapterFeatureGotoTargetsRequest:
    return "supportsGotoTargetsRequest";
  case eAdapterFeatureHitConditionalBreakpoints:
    return "supportsHitConditionalBreakpoints";
  case eAdapterFeatureInstructionBreakpoints:
    return "supportsInstructionBreakpoints";
  case eAdapterFeatureLoadedSourcesRequest:
    return "supportsLoadedSourcesRequest";
  case eAdapterFeatureLogPoints:
    return "supportsLogPoints";
  case eAdapterFeatureModulesRequest:
    return "supportsModulesRequest";
  case eAdapterFeatureReadMemoryRequest:
    return "supportsReadMemoryRequest";
  case eAdapterFeatureRestartFrame:
    return "supportsRestartFrame";
  case eAdapterFeatureRestartRequest:
    return "supportsRestartRequest";
  case eAdapterFeatureSetExpression:
    return "supportsSetExpression";
  case eAdapterFeatureSetVariable:
    return "supportsSetVariable";
  case eAdapterFeatureSingleThreadExecutionRequests:
    return "supportsSingleThreadExecutionRequests";
  case eAdapterFeatureStepBack:
    return "supportsStepBack";
  case eAdapterFeatureStepInTargetsRequest:
    return "supportsStepInTargetsRequest";
  case eAdapterFeatureSteppingGranularity:
    return "supportsSteppingGranularity";
  case eAdapterFeatureTerminateRequest:
    return "supportsTerminateRequest";
  case eAdapterFeatureTerminateThreadsRequest:
    return "supportsTerminateThreadsRequest";
  case eAdapterFeatureSuspendDebuggee:
    return "supportSuspendDebuggee";
  case eAdapterFeatureValueFormattingOptions:
    return "supportsValueFormattingOptions";
  case eAdapterFeatureWriteMemoryRequest:
    return "supportsWriteMemoryRequest";
  case eAdapterFeatureTerminateDebuggee:
    return "supportTerminateDebuggee";
  }
  llvm_unreachable("unhandled adapter feature.");
}

llvm::json::Value toJSON(const AdapterFeature &feature) {
  return ToString(feature);
}

bool fromJSON(const llvm::json::Value &Params, AdapterFeature &feature,
              llvm::json::Path P) {
  auto rawFeature = Params.getAsString();
  if (!rawFeature) {
    P.report("expected a string");
    return false;
  }

  std::optional<AdapterFeature> parsedFeature =
      llvm::StringSwitch<std::optional<AdapterFeature>>(*rawFeature)
          .Case("supportsANSIStyling", eAdapterFeatureANSIStyling)
          .Case("supportsBreakpointLocationsRequest",
                eAdapterFeatureBreakpointLocationsRequest)
          .Case("supportsCancelRequest", eAdapterFeatureCancelRequest)
          .Case("supportsClipboardContext", eAdapterFeatureClipboardContext)
          .Case("supportsCompletionsRequest", eAdapterFeatureCompletionsRequest)
          .Case("supportsConditionalBreakpoints",
                eAdapterFeatureConditionalBreakpoints)
          .Case("supportsConfigurationDoneRequest",
                eAdapterFeatureConfigurationDoneRequest)
          .Case("supportsDataBreakpointBytes",
                eAdapterFeatureDataBreakpointBytes)
          .Case("supportsDataBreakpoints", eAdapterFeatureDataBreakpoints)
          .Case("supportsDelayedStackTraceLoading",
                eAdapterFeatureDelayedStackTraceLoading)
          .Case("supportsDisassembleRequest", eAdapterFeatureDisassembleRequest)
          .Case("supportsEvaluateForHovers", eAdapterFeatureEvaluateForHovers)
          .Case("supportsExceptionFilterOptions",
                eAdapterFeatureExceptionFilterOptions)
          .Case("supportsExceptionInfoRequest",
                eAdapterFeatureExceptionInfoRequest)
          .Case("supportsExceptionOptions", eAdapterFeatureExceptionOptions)
          .Case("supportsFunctionBreakpoints",
                eAdapterFeatureFunctionBreakpoints)
          .Case("supportsGotoTargetsRequest", eAdapterFeatureGotoTargetsRequest)
          .Case("supportsHitConditionalBreakpoints",
                eAdapterFeatureHitConditionalBreakpoints)
          .Case("supportsInstructionBreakpoints",
                eAdapterFeatureInstructionBreakpoints)
          .Case("supportsLoadedSourcesRequest",
                eAdapterFeatureLoadedSourcesRequest)
          .Case("supportsLogPoints", eAdapterFeatureLogPoints)
          .Case("supportsModulesRequest", eAdapterFeatureModulesRequest)
          .Case("supportsReadMemoryRequest", eAdapterFeatureReadMemoryRequest)
          .Case("supportsRestartFrame", eAdapterFeatureRestartFrame)
          .Case("supportsRestartRequest", eAdapterFeatureRestartRequest)
          .Case("supportsSetExpression", eAdapterFeatureSetExpression)
          .Case("supportsSetVariable", eAdapterFeatureSetVariable)
          .Case("supportsSingleThreadExecutionRequests",
                eAdapterFeatureSingleThreadExecutionRequests)
          .Case("supportsStepBack", eAdapterFeatureStepBack)
          .Case("supportsStepInTargetsRequest",
                eAdapterFeatureStepInTargetsRequest)
          .Case("supportsSteppingGranularity",
                eAdapterFeatureSteppingGranularity)
          .Case("supportsTerminateRequest", eAdapterFeatureTerminateRequest)
          .Case("supportsTerminateThreadsRequest",
                eAdapterFeatureTerminateThreadsRequest)
          .Case("supportSuspendDebuggee", eAdapterFeatureSuspendDebuggee)
          .Case("supportsValueFormattingOptions",
                eAdapterFeatureValueFormattingOptions)
          .Case("supportsWriteMemoryRequest", eAdapterFeatureWriteMemoryRequest)
          .Case("supportTerminateDebuggee", eAdapterFeatureTerminateDebuggee)
          .Default(std::nullopt);

  if (!parsedFeature) {
    P.report("unexpected value for AdapterFeature");
    return false;
  }

  feature = *parsedFeature;
  return true;
}

json::Value toJSON(const Capabilities &C) {
  json::Object result;

  for (const auto &feature : C.supportedFeatures)
    result.insert({ToString(feature), true});

  if (C.exceptionBreakpointFilters && !C.exceptionBreakpointFilters->empty())
    result.insert(
        {"exceptionBreakpointFilters", *C.exceptionBreakpointFilters});
  if (C.completionTriggerCharacters && !C.completionTriggerCharacters->empty())
    result.insert(
        {"completionTriggerCharacters", *C.completionTriggerCharacters});
  if (C.additionalModuleColumns && !C.additionalModuleColumns->empty())
    result.insert({"additionalModuleColumns", *C.additionalModuleColumns});
  if (C.supportedChecksumAlgorithms && !C.supportedChecksumAlgorithms->empty())
    result.insert(
        {"supportedChecksumAlgorithms", *C.supportedChecksumAlgorithms});
  if (C.breakpointModes && !C.breakpointModes->empty())
    result.insert({"breakpointModes", *C.breakpointModes});

  // lldb-dap extensions
  if (C.lldbExtVersion && !C.lldbExtVersion->empty())
    result.insert({"$__lldb_version", *C.lldbExtVersion});

  return result;
}

bool fromJSON(const llvm::json::Value &Params, Capabilities &C,
              llvm::json::Path P) {
  auto *Object = Params.getAsObject();
  if (!Object) {
    P.report("expected an object");
    return false;
  }
  // Check for the presence of supported features.
  for (unsigned i = eAdapterFeatureFirst; i <= eAdapterFeatureLast; ++i) {
    AdapterFeature feature = static_cast<AdapterFeature>(i);
    if (Object->getBoolean(ToString(feature)))
      C.supportedFeatures.insert(feature);
  }
  llvm::json::ObjectMapper O(Params, P);
  return O &&
         O.mapOptional("exceptionBreakpointFilters",
                       C.exceptionBreakpointFilters) &&
         O.mapOptional("completionTriggerCharacters",
                       C.completionTriggerCharacters) &&
         O.mapOptional("additionalModuleColumns", C.additionalModuleColumns) &&
         O.mapOptional("supportedChecksumAlgorithms",
                       C.supportedChecksumAlgorithms) &&
         O.mapOptional("breakpointModes", C.breakpointModes) &&
         O.mapOptional("$__lldb_version", C.lldbExtVersion);
}

bool fromJSON(const llvm::json::Value &Params, SteppingGranularity &SG,
              llvm::json::Path P) {
  auto raw_granularity = Params.getAsString();
  if (!raw_granularity) {
    P.report("expected a string");
    return false;
  }
  std::optional<SteppingGranularity> granularity =
      StringSwitch<std::optional<SteppingGranularity>>(*raw_granularity)
          .Case("statement", eSteppingGranularityStatement)
          .Case("line", eSteppingGranularityLine)
          .Case("instruction", eSteppingGranularityInstruction)
          .Default(std::nullopt);
  if (!granularity) {
    P.report("unexpected value");
    return false;
  }
  SG = *granularity;
  return true;
}

llvm::json::Value toJSON(const SteppingGranularity &SG) {
  switch (SG) {
  case eSteppingGranularityStatement:
    return "statement";
  case eSteppingGranularityLine:
    return "line";
  case eSteppingGranularityInstruction:
    return "instruction";
  }
  llvm_unreachable("unhandled stepping granularity.");
}

bool fromJSON(const llvm::json::Value &Params, ValueFormat &VF,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("hex", VF.hex);
}

json::Value toJSON(const BreakpointLocation &B) {
  json::Object result;

  result.insert({"line", B.line});
  if (B.column)
    result.insert({"column", *B.column});
  if (B.endLine)
    result.insert({"endLine", *B.endLine});
  if (B.endColumn)
    result.insert({"endColumn", *B.endColumn});

  return result;
}

llvm::json::Value toJSON(const BreakpointReason &BR) {
  switch (BR) {
  case BreakpointReason::eBreakpointReasonPending:
    return "pending";
  case BreakpointReason::eBreakpointReasonFailed:
    return "failed";
  }
  llvm_unreachable("unhandled breakpoint reason.");
}

bool fromJSON(const llvm::json::Value &Params, BreakpointReason &BR,
              llvm::json::Path P) {
  auto rawReason = Params.getAsString();
  if (!rawReason) {
    P.report("expected a string");
    return false;
  }
  std::optional<BreakpointReason> reason =
      llvm::StringSwitch<std::optional<BreakpointReason>>(*rawReason)
          .Case("pending", BreakpointReason::eBreakpointReasonPending)
          .Case("failed", BreakpointReason::eBreakpointReasonFailed)
          .Default(std::nullopt);
  if (!reason) {
    P.report("unexpected value, expected 'pending' or 'failed'");
    return false;
  }
  BR = *reason;
  return true;
}

json::Value toJSON(const Breakpoint &BP) {
  json::Object result{{"verified", BP.verified}};

  if (BP.id)
    result.insert({"id", *BP.id});
  if (BP.message)
    result.insert({"message", *BP.message});
  if (BP.source)
    result.insert({"source", *BP.source});
  if (BP.line)
    result.insert({"line", *BP.line});
  if (BP.column)
    result.insert({"column", *BP.column});
  if (BP.endLine)
    result.insert({"endLine", *BP.endLine});
  if (BP.endColumn)
    result.insert({"endColumn", *BP.endColumn});
  if (BP.instructionReference)
    result.insert({"instructionReference", *BP.instructionReference});
  if (BP.offset)
    result.insert({"offset", *BP.offset});
  if (BP.reason) {
    result.insert({"reason", *BP.reason});
  }

  return result;
}

bool fromJSON(const llvm::json::Value &Params, Breakpoint &BP,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.mapOptional("id", BP.id) && O.map("verified", BP.verified) &&
         O.mapOptional("message", BP.message) &&
         O.mapOptional("source", BP.source) && O.mapOptional("line", BP.line) &&
         O.mapOptional("column", BP.column) &&
         O.mapOptional("endLine", BP.endLine) &&
         O.mapOptional("endColumn", BP.endColumn) &&
         O.mapOptional("instructionReference", BP.instructionReference) &&
         O.mapOptional("offset", BP.offset) &&
         O.mapOptional("reason", BP.reason);
}

bool fromJSON(const llvm::json::Value &Params, SourceBreakpoint &SB,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("line", SB.line) && O.mapOptional("column", SB.column) &&
         O.mapOptional("condition", SB.condition) &&
         O.mapOptional("hitCondition", SB.hitCondition) &&
         O.mapOptional("logMessage", SB.logMessage) &&
         O.mapOptional("mode", SB.mode);
}

llvm::json::Value toJSON(const SourceBreakpoint &SB) {
  llvm::json::Object result{{"line", SB.line}};

  if (SB.column)
    result.insert({"column", *SB.column});
  if (SB.condition)
    result.insert({"condition", *SB.condition});
  if (SB.hitCondition)
    result.insert({"hitCondition", *SB.hitCondition});
  if (SB.logMessage)
    result.insert({"logMessage", *SB.logMessage});
  if (SB.mode)
    result.insert({"mode", *SB.mode});

  return result;
}

bool fromJSON(const llvm::json::Value &Params, FunctionBreakpoint &FB,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("name", FB.name) &&
         O.mapOptional("condition", FB.condition) &&
         O.mapOptional("hitCondition", FB.hitCondition);
}

llvm::json::Value toJSON(const FunctionBreakpoint &FB) {
  llvm::json::Object result{{"name", FB.name}};

  if (FB.condition)
    result.insert({"condition", *FB.condition});
  if (FB.hitCondition)
    result.insert({"hitCondition", *FB.hitCondition});

  return result;
}

bool fromJSON(const llvm::json::Value &Params, DataBreakpointAccessType &DBAT,
              llvm::json::Path P) {
  auto rawAccessType = Params.getAsString();
  if (!rawAccessType) {
    P.report("expected a string");
    return false;
  }
  std::optional<DataBreakpointAccessType> accessType =
      StringSwitch<std::optional<DataBreakpointAccessType>>(*rawAccessType)
          .Case("read", eDataBreakpointAccessTypeRead)
          .Case("write", eDataBreakpointAccessTypeWrite)
          .Case("readWrite", eDataBreakpointAccessTypeReadWrite)
          .Default(std::nullopt);
  if (!accessType) {
    P.report("unexpected value, expected 'read', 'write', or 'readWrite'");
    return false;
  }
  DBAT = *accessType;
  return true;
}

llvm::json::Value toJSON(const DataBreakpointAccessType &DBAT) {
  switch (DBAT) {
  case eDataBreakpointAccessTypeRead:
    return "read";
  case eDataBreakpointAccessTypeWrite:
    return "write";
  case eDataBreakpointAccessTypeReadWrite:
    return "readWrite";
  }
  llvm_unreachable("unhandled data breakpoint access type.");
}

bool fromJSON(const llvm::json::Value &Params, DataBreakpoint &DBI,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("dataId", DBI.dataId) &&
         O.mapOptional("accessType", DBI.accessType) &&
         O.mapOptional("condition", DBI.condition) &&
         O.mapOptional("hitCondition", DBI.hitCondition);
}

llvm::json::Value toJSON(const DataBreakpoint &DBI) {
  llvm::json::Object result{{"dataId", DBI.dataId}};

  if (DBI.accessType)
    result.insert({"accessType", *DBI.accessType});
  if (DBI.condition)
    result.insert({"condition", *DBI.condition});
  if (DBI.hitCondition)
    result.insert({"hitCondition", *DBI.hitCondition});

  return result;
}

bool fromJSON(const llvm::json::Value &Params, InstructionBreakpoint &IB,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("instructionReference", IB.instructionReference) &&
         O.mapOptional("offset", IB.offset) &&
         O.mapOptional("condition", IB.condition) &&
         O.mapOptional("hitCondition", IB.hitCondition) &&
         O.mapOptional("mode", IB.mode);
}

} // namespace lldb_dap::protocol
