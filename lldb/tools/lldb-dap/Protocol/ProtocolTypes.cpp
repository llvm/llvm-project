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
  return O && O.mapOptional("name", S.name) && O.mapOptional("path", S.path) &&
         O.mapOptional("presentationHint", S.presentationHint) &&
         O.mapOptional("sourceReference", S.sourceReference);
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

static llvm::StringLiteral ToString(AdapterFeature feature) {
  switch (feature) {
  case eAdapterFeatureSupportsANSIStyling:
    return "supportsANSIStyling";
  case eAdapterFeatureSupportsBreakpointLocationsRequest:
    return "supportsBreakpointLocationsRequest";
  case eAdapterFeatureSupportsCancelRequest:
    return "supportsCancelRequest";
  case eAdapterFeatureSupportsClipboardContext:
    return "supportsClipboardContext";
  case eAdapterFeatureSupportsCompletionsRequest:
    return "supportsCompletionsRequest";
  case eAdapterFeatureSupportsConditionalBreakpoints:
    return "supportsConditionalBreakpoints";
  case eAdapterFeatureSupportsConfigurationDoneRequest:
    return "supportsConfigurationDoneRequest";
  case eAdapterFeatureSupportsDataBreakpointBytes:
    return "supportsDataBreakpointBytes";
  case eAdapterFeatureSupportsDataBreakpoints:
    return "supportsDataBreakpoints";
  case eAdapterFeatureSupportsDelayedStackTraceLoading:
    return "supportsDelayedStackTraceLoading";
  case eAdapterFeatureSupportsDisassembleRequest:
    return "supportsDisassembleRequest";
  case eAdapterFeatureSupportsEvaluateForHovers:
    return "supportsEvaluateForHovers";
  case eAdapterFeatureSupportsExceptionFilterOptions:
    return "supportsExceptionFilterOptions";
  case eAdapterFeatureSupportsExceptionInfoRequest:
    return "supportsExceptionInfoRequest";
  case eAdapterFeatureSupportsExceptionOptions:
    return "supportsExceptionOptions";
  case eAdapterFeatureSupportsFunctionBreakpoints:
    return "supportsFunctionBreakpoints";
  case eAdapterFeatureSupportsGotoTargetsRequest:
    return "supportsGotoTargetsRequest";
  case eAdapterFeatureSupportsHitConditionalBreakpoints:
    return "supportsHitConditionalBreakpoints";
  case eAdapterFeatureSupportsInstructionBreakpoints:
    return "supportsInstructionBreakpoints";
  case eAdapterFeatureSupportsLoadedSourcesRequest:
    return "supportsLoadedSourcesRequest";
  case eAdapterFeatureSupportsLogPoints:
    return "supportsLogPoints";
  case eAdapterFeatureSupportsModulesRequest:
    return "supportsModulesRequest";
  case eAdapterFeatureSupportsReadMemoryRequest:
    return "supportsReadMemoryRequest";
  case eAdapterFeatureSupportsRestartFrame:
    return "supportsRestartFrame";
  case eAdapterFeatureSupportsRestartRequest:
    return "supportsRestartRequest";
  case eAdapterFeatureSupportsSetExpression:
    return "supportsSetExpression";
  case eAdapterFeatureSupportsSetVariable:
    return "supportsSetVariable";
  case eAdapterFeatureSupportsSingleThreadExecutionRequests:
    return "supportsSingleThreadExecutionRequests";
  case eAdapterFeatureSupportsStepBack:
    return "supportsStepBack";
  case eAdapterFeatureSupportsStepInTargetsRequest:
    return "supportsStepInTargetsRequest";
  case eAdapterFeatureSupportsSteppingGranularity:
    return "supportsSteppingGranularity";
  case eAdapterFeatureSupportsTerminateRequest:
    return "supportsTerminateRequest";
  case eAdapterFeatureSupportsTerminateThreadsRequest:
    return "supportsTerminateThreadsRequest";
  case eAdapterFeatureSupportSuspendDebuggee:
    return "supportSuspendDebuggee";
  case eAdapterFeatureSupportsValueFormattingOptions:
    return "supportsValueFormattingOptions";
  case eAdapterFeatureSupportsWriteMemoryRequest:
    return "supportsWriteMemoryRequest";
  case eAdapterFeatureSupportTerminateDebuggee:
    return "supportTerminateDebuggee";
  }
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

} // namespace lldb_dap::protocol
