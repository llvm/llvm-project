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

bool fromJSON(const json::Value &Params, Source::PresentationHint &PH,
              json::Path P) {
  auto rawHint = Params.getAsString();
  if (!rawHint) {
    P.report("expected a string");
    return false;
  }
  std::optional<Source::PresentationHint> hint =
      StringSwitch<std::optional<Source::PresentationHint>>(*rawHint)
          .Case("normal", Source::PresentationHint::normal)
          .Case("emphasize", Source::PresentationHint::emphasize)
          .Case("deemphasize", Source::PresentationHint::deemphasize)
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

json::Value toJSON(const ColumnDescriptor::Type &T) {
  switch (T) {
  case ColumnDescriptor::Type::String:
    return "string";
  case ColumnDescriptor::Type::Number:
    return "number";
  case ColumnDescriptor::Type::Boolean:
    return "boolean";
  case ColumnDescriptor::Type::Timestamp:
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
  case ChecksumAlgorithm::md5:
    return "MD5";
  case ChecksumAlgorithm::sha1:
    return "SHA1";
  case ChecksumAlgorithm::sha256:
    return "SHA256";
  case ChecksumAlgorithm::timestamp:
    return "timestamp";
  }
}

json::Value toJSON(const BreakpointModeApplicability &BMA) {
  switch (BMA) {
  case BreakpointModeApplicability::source:
    return "source";
  case BreakpointModeApplicability::exception:
    return "exception";
  case BreakpointModeApplicability::data:
    return "data";
  case BreakpointModeApplicability::instruction:
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

json::Value toJSON(const Capabilities &C) {
  json::Object result;

  for (const auto &feature : C.supportedFeatures)
    switch (feature) {
    case Capabilities::Feature::supportsANSIStyling:
      result.insert({"supportsANSIStyling", true});
      break;
    case Capabilities::Feature::supportsBreakpointLocationsRequest:
      result.insert({"supportsBreakpointLocationsRequest", true});
      break;
    case Capabilities::Feature::supportsCancelRequest:
      result.insert({"supportsCancelRequest", true});
      break;
    case Capabilities::Feature::supportsClipboardContext:
      result.insert({"supportsClipboardContext", true});
      break;
    case Capabilities::Feature::supportsCompletionsRequest:
      result.insert({"supportsCompletionsRequest", true});
      break;
    case Capabilities::Feature::supportsConditionalBreakpoints:
      result.insert({"supportsConditionalBreakpoints", true});
      break;
    case Capabilities::Feature::supportsConfigurationDoneRequest:
      result.insert({"supportsConfigurationDoneRequest", true});
      break;
    case Capabilities::Feature::supportsDataBreakpointBytes:
      result.insert({"supportsDataBreakpointBytes", true});
      break;
    case Capabilities::Feature::supportsDataBreakpoints:
      result.insert({"supportsDataBreakpoints", true});
      break;
    case Capabilities::Feature::supportsDelayedStackTraceLoading:
      result.insert({"supportsDelayedStackTraceLoading", true});
      break;
    case Capabilities::Feature::supportsDisassembleRequest:
      result.insert({"supportsDisassembleRequest", true});
      break;
    case Capabilities::Feature::supportsEvaluateForHovers:
      result.insert({"supportsEvaluateForHovers", true});
      break;
    case Capabilities::Feature::supportsExceptionFilterOptions:
      result.insert({"supportsExceptionFilterOptions", true});
      break;
    case Capabilities::Feature::supportsExceptionInfoRequest:
      result.insert({"supportsExceptionInfoRequest", true});
      break;
    case Capabilities::Feature::supportsExceptionOptions:
      result.insert({"supportsExceptionOptions", true});
      break;
    case Capabilities::Feature::supportsFunctionBreakpoints:
      result.insert({"supportsFunctionBreakpoints", true});
      break;
    case Capabilities::Feature::supportsGotoTargetsRequest:
      result.insert({"supportsGotoTargetsRequest", true});
      break;
    case Capabilities::Feature::supportsHitConditionalBreakpoints:
      result.insert({"supportsHitConditionalBreakpoints", true});
      break;
    case Capabilities::Feature::supportsInstructionBreakpoints:
      result.insert({"supportsInstructionBreakpoints", true});
      break;
    case Capabilities::Feature::supportsLoadedSourcesRequest:
      result.insert({"supportsLoadedSourcesRequest", true});
      break;
    case Capabilities::Feature::supportsLogPoints:
      result.insert({"supportsLogPoints", true});
      break;
    case Capabilities::Feature::supportsModulesRequest:
      result.insert({"supportsModulesRequest", true});
      break;
    case Capabilities::Feature::supportsReadMemoryRequest:
      result.insert({"supportsReadMemoryRequest", true});
      break;
    case Capabilities::Feature::supportsRestartFrame:
      result.insert({"supportsRestartFrame", true});
      break;
    case Capabilities::Feature::supportsRestartRequest:
      result.insert({"supportsRestartRequest", true});
      break;
    case Capabilities::Feature::supportsSetExpression:
      result.insert({"supportsSetExpression", true});
      break;
    case Capabilities::Feature::supportsSetVariable:
      result.insert({"supportsSetVariable", true});
      break;
    case Capabilities::Feature::supportsSingleThreadExecutionRequests:
      result.insert({"supportsSingleThreadExecutionRequests", true});
      break;
    case Capabilities::Feature::supportsStepBack:
      result.insert({"supportsStepBack", true});
      break;
    case Capabilities::Feature::supportsStepInTargetsRequest:
      result.insert({"supportsStepInTargetsRequest", true});
      break;
    case Capabilities::Feature::supportsSteppingGranularity:
      result.insert({"supportsSteppingGranularity", true});
      break;
    case Capabilities::Feature::supportsTerminateRequest:
      result.insert({"supportsTerminateRequest", true});
      break;
    case Capabilities::Feature::supportsTerminateThreadsRequest:
      result.insert({"supportsTerminateThreadsRequest", true});
      break;
    case Capabilities::Feature::supportSuspendDebuggee:
      result.insert({"supportSuspendDebuggee", true});
      break;
    case Capabilities::Feature::supportsValueFormattingOptions:
      result.insert({"supportsValueFormattingOptions", true});
      break;
    case Capabilities::Feature::supportsWriteMemoryRequest:
      result.insert({"supportsWriteMemoryRequest", true});
      break;
    case Capabilities::Feature::supportTerminateDebuggee:
      result.insert({"supportTerminateDebuggee", true});
      break;
    }

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
