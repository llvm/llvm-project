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
    result.insert({"defaultState", *EBF.defaultState});
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

  if (C.supportsConfigurationDoneRequest && *C.supportsConfigurationDoneRequest)
    result.insert({"supportsConfigurationDoneRequest",
                   *C.supportsConfigurationDoneRequest});
  if (C.supportsFunctionBreakpoints && *C.supportsFunctionBreakpoints)
    result.insert(
        {"supportsFunctionBreakpoints", *C.supportsFunctionBreakpoints});
  if (C.supportsConditionalBreakpoints && *C.supportsConditionalBreakpoints)
    result.insert(
        {"supportsConditionalBreakpoints", *C.supportsConditionalBreakpoints});
  if (C.supportsHitConditionalBreakpoints &&
      *C.supportsHitConditionalBreakpoints)
    result.insert({"supportsHitConditionalBreakpoints",
                   *C.supportsHitConditionalBreakpoints});
  if (C.supportsEvaluateForHovers && *C.supportsEvaluateForHovers)
    result.insert({"supportsEvaluateForHovers", *C.supportsEvaluateForHovers});
  if (C.exceptionBreakpointFilters && !C.exceptionBreakpointFilters->empty())
    result.insert(
        {"exceptionBreakpointFilters", *C.exceptionBreakpointFilters});
  if (C.supportsStepBack && *C.supportsStepBack)
    result.insert({"supportsStepBack", *C.supportsStepBack});
  if (C.supportsSetVariable && *C.supportsSetVariable)
    result.insert({"supportsSetVariable", *C.supportsSetVariable});
  if (C.supportsRestartFrame && *C.supportsRestartFrame)
    result.insert({"supportsRestartFrame", *C.supportsRestartFrame});
  if (C.supportsGotoTargetsRequest && *C.supportsGotoTargetsRequest)
    result.insert(
        {"supportsGotoTargetsRequest", *C.supportsGotoTargetsRequest});
  if (C.supportsStepInTargetsRequest && *C.supportsStepInTargetsRequest)
    result.insert(
        {"supportsStepInTargetsRequest", *C.supportsStepInTargetsRequest});
  if (C.supportsCompletionsRequest && *C.supportsCompletionsRequest)
    result.insert(
        {"supportsCompletionsRequest", *C.supportsCompletionsRequest});
  if (C.completionTriggerCharacters && !C.completionTriggerCharacters->empty())
    result.insert(
        {"completionTriggerCharacters", *C.completionTriggerCharacters});
  if (C.supportsModulesRequest && *C.supportsModulesRequest)
    result.insert({"supportsModulesRequest", *C.supportsModulesRequest});
  if (C.additionalModuleColumns && !C.additionalModuleColumns->empty())
    result.insert({"additionalModuleColumns", *C.additionalModuleColumns});
  if (C.supportedChecksumAlgorithms && !C.supportedChecksumAlgorithms->empty())
    result.insert(
        {"supportedChecksumAlgorithms", *C.supportedChecksumAlgorithms});
  if (C.supportsRestartRequest && *C.supportsRestartRequest)
    result.insert({"supportsRestartRequest", *C.supportsRestartRequest});
  if (C.supportsExceptionOptions && *C.supportsExceptionOptions)
    result.insert({"supportsExceptionOptions", *C.supportsExceptionOptions});
  if (C.supportsValueFormattingOptions && *C.supportsValueFormattingOptions)
    result.insert(
        {"supportsValueFormattingOptions", *C.supportsValueFormattingOptions});
  if (C.supportsExceptionInfoRequest && *C.supportsExceptionInfoRequest)
    result.insert(
        {"supportsExceptionInfoRequest", *C.supportsExceptionInfoRequest});
  if (C.supportTerminateDebuggee && *C.supportTerminateDebuggee)
    result.insert({"supportTerminateDebuggee", *C.supportTerminateDebuggee});
  if (C.supportSuspendDebuggee && *C.supportSuspendDebuggee)
    result.insert({"supportSuspendDebuggee", *C.supportSuspendDebuggee});
  if (C.supportsDelayedStackTraceLoading && *C.supportsDelayedStackTraceLoading)
    result.insert({"supportsDelayedStackTraceLoading",
                   *C.supportsDelayedStackTraceLoading});
  if (C.supportsLoadedSourcesRequest && *C.supportsLoadedSourcesRequest)
    result.insert(
        {"supportsLoadedSourcesRequest", *C.supportsLoadedSourcesRequest});
  if (C.supportsLogPoints && *C.supportsLogPoints)
    result.insert({"supportsLogPoints", *C.supportsLogPoints});
  if (C.supportsTerminateThreadsRequest && *C.supportsTerminateThreadsRequest)
    result.insert({"supportsTerminateThreadsRequest",
                   *C.supportsTerminateThreadsRequest});
  if (C.supportsSetExpression && *C.supportsSetExpression)
    result.insert({"supportsSetExpression", *C.supportsSetExpression});
  if (C.supportsTerminateRequest && *C.supportsTerminateRequest)
    result.insert({"supportsTerminateRequest", *C.supportsTerminateRequest});
  if (C.supportsDataBreakpoints && *C.supportsDataBreakpoints)
    result.insert({"supportsDataBreakpoints", *C.supportsDataBreakpoints});
  if (C.supportsReadMemoryRequest && *C.supportsReadMemoryRequest)
    result.insert({"supportsReadMemoryRequest", *C.supportsReadMemoryRequest});
  if (C.supportsWriteMemoryRequest && *C.supportsWriteMemoryRequest)
    result.insert(
        {"supportsWriteMemoryRequest", *C.supportsWriteMemoryRequest});
  if (C.supportsDisassembleRequest && *C.supportsDisassembleRequest)
    result.insert(
        {"supportsDisassembleRequest", *C.supportsDisassembleRequest});
  if (C.supportsCancelRequest && *C.supportsCancelRequest)
    result.insert({"supportsCancelRequest", *C.supportsCancelRequest});
  if (C.supportsBreakpointLocationsRequest &&
      *C.supportsBreakpointLocationsRequest)
    result.insert({"supportsBreakpointLocationsRequest",
                   *C.supportsBreakpointLocationsRequest});
  if (C.supportsClipboardContext && *C.supportsClipboardContext)
    result.insert({"supportsClipboardContext", *C.supportsClipboardContext});
  if (C.supportsSteppingGranularity && *C.supportsSteppingGranularity)
    result.insert(
        {"supportsSteppingGranularity", *C.supportsSteppingGranularity});
  if (C.supportsInstructionBreakpoints && *C.supportsInstructionBreakpoints)
    result.insert(
        {"supportsInstructionBreakpoints", *C.supportsInstructionBreakpoints});
  if (C.supportsExceptionFilterOptions && *C.supportsExceptionFilterOptions)
    result.insert(
        {"supportsExceptionFilterOptions", *C.supportsExceptionFilterOptions});
  if (C.supportsSingleThreadExecutionRequests &&
      *C.supportsSingleThreadExecutionRequests)
    result.insert({"supportsSingleThreadExecutionRequests",
                   *C.supportsSingleThreadExecutionRequests});
  if (C.supportsDataBreakpointBytes && *C.supportsDataBreakpointBytes)
    result.insert(
        {"supportsDataBreakpointBytes", *C.supportsDataBreakpointBytes});
  if (C.breakpointModes && !C.breakpointModes->empty())
    result.insert({"breakpointModes", *C.breakpointModes});
  if (C.supportsANSIStyling && *C.supportsANSIStyling)
    result.insert({"supportsANSIStyling", *C.supportsANSIStyling});
  if (C.lldbVersion && !C.lldbVersion->empty())
    result.insert({"$__lldb_version", *C.lldbVersion});

  return result;
}

} // namespace lldb_dap::protocol
