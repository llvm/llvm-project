//===-- ProtocolTypes.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolTypes.h"
#include "JSONUtils.h"
#include "ProtocolUtils.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/StringExtras.h"
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
          .Case("normal", Source::eSourcePresentationHintNormal)
          .Case("emphasize", Source::eSourcePresentationHintEmphasize)
          .Case("deemphasize", Source::eSourcePresentationHintDeemphasize)
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
         O.map("sourceReference", S.sourceReference) &&
         O.map("adapterData", S.adapterData);
}

llvm::json::Value toJSON(Source::PresentationHint hint) {
  switch (hint) {
  case Source::eSourcePresentationHintNormal:
    return "normal";
  case Source::eSourcePresentationHintEmphasize:
    return "emphasize";
  case Source::eSourcePresentationHintDeemphasize:
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
  if (S.sourceReference && (*S.sourceReference > LLDB_DAP_INVALID_SRC_REF))
    result.insert({"sourceReference", *S.sourceReference});
  if (S.presentationHint)
    result.insert({"presentationHint", *S.presentationHint});
  if (S.adapterData)
    result.insert({"adapterData", *S.adapterData});

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

  if (!EBF.description.empty())
    result.insert({"description", EBF.description});
  if (EBF.defaultState)
    result.insert({"default", EBF.defaultState});
  if (EBF.supportsCondition)
    result.insert({"supportsCondition", EBF.supportsCondition});
  if (!EBF.conditionDescription.empty())
    result.insert({"conditionDescription", EBF.conditionDescription});

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

  if (!C.exceptionBreakpointFilters.empty())
    result.insert({"exceptionBreakpointFilters", C.exceptionBreakpointFilters});
  if (!C.completionTriggerCharacters.empty())
    result.insert(
        {"completionTriggerCharacters", C.completionTriggerCharacters});
  if (!C.additionalModuleColumns.empty())
    result.insert({"additionalModuleColumns", C.additionalModuleColumns});
  if (!C.supportedChecksumAlgorithms.empty())
    result.insert(
        {"supportedChecksumAlgorithms", C.supportedChecksumAlgorithms});
  if (!C.breakpointModes.empty())
    result.insert({"breakpointModes", C.breakpointModes});

  // lldb-dap extensions
  if (!C.lldbExtVersion.empty())
    result.insert({"$__lldb_version", C.lldbExtVersion});

  return result;
}

bool fromJSON(const json::Value &Params, ExceptionFilterOptions &EFO,
              json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("filterId", EFO.filterId) &&
         O.mapOptional("condition", EFO.condition) &&
         O.mapOptional("mode", EFO.mode);
}

json::Value toJSON(const ExceptionFilterOptions &EFO) {
  json::Object result{{"filterId", EFO.filterId}};

  if (!EFO.condition.empty())
    result.insert({"condition", EFO.condition});
  if (!EFO.mode.empty())
    result.insert({"mode", EFO.mode});

  return result;
}

bool fromJSON(const json::Value &Params, Scope::PresentationHint &PH,
              json::Path P) {
  auto rawHint = Params.getAsString();
  if (!rawHint) {
    P.report("expected a string");
    return false;
  }
  const std::optional<Scope::PresentationHint> hint =
      StringSwitch<std::optional<Scope::PresentationHint>>(*rawHint)
          .Case("arguments", Scope::eScopePresentationHintArguments)
          .Case("locals", Scope::eScopePresentationHintLocals)
          .Case("registers", Scope::eScopePresentationHintRegisters)
          .Case("returnValue", Scope::eScopePresentationHintReturnValue)
          .Default(std::nullopt);
  if (!hint) {
    P.report("unexpected value");
    return false;
  }
  PH = *hint;
  return true;
}

bool fromJSON(const json::Value &Params, Scope &S, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("name", S.name) &&
         O.mapOptional("presentationHint", S.presentationHint) &&
         O.map("variablesReference", S.variablesReference) &&
         O.mapOptional("namedVariables", S.namedVariables) &&
         O.map("indexedVariables", S.indexedVariables) &&
         O.mapOptional("source", S.source) && O.map("expensive", S.expensive) &&
         O.mapOptional("line", S.line) && O.mapOptional("column", S.column) &&
         O.mapOptional("endLine", S.endLine) &&
         O.mapOptional("endColumn", S.endColumn);
}

llvm::json::Value toJSON(const Scope &SC) {
  llvm::json::Object result{{"name", SC.name},
                            {"variablesReference", SC.variablesReference},
                            {"expensive", SC.expensive}};

  if (SC.presentationHint.has_value()) {
    llvm::StringRef presentationHint;
    switch (*SC.presentationHint) {
    case Scope::eScopePresentationHintArguments:
      presentationHint = "arguments";
      break;
    case Scope::eScopePresentationHintLocals:
      presentationHint = "locals";
      break;
    case Scope::eScopePresentationHintRegisters:
      presentationHint = "registers";
      break;
    case Scope::eScopePresentationHintReturnValue:
      presentationHint = "returnValue";
      break;
    }

    result.insert({"presentationHint", presentationHint});
  }

  if (SC.namedVariables.has_value())
    result.insert({"namedVariables", SC.namedVariables});

  if (SC.indexedVariables.has_value())
    result.insert({"indexedVariables", SC.indexedVariables});

  if (SC.source.has_value())
    result.insert({"source", SC.source});

  if (SC.line.has_value())
    result.insert({"line", SC.line});

  if (SC.column.has_value())
    result.insert({"column", SC.column});

  if (SC.endLine.has_value())
    result.insert({"endLine", SC.endLine});

  if (SC.endColumn.has_value())
    result.insert({"endColumn", SC.endColumn});

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

bool fromJSON(const json::Value &Params, StepInTarget &SIT, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("id", SIT.id) && O.map("label", SIT.label) &&
         O.mapOptional("line", SIT.line) &&
         O.mapOptional("column", SIT.column) &&
         O.mapOptional("endLine", SIT.endLine) &&
         O.mapOptional("endColumn", SIT.endColumn);
}

llvm::json::Value toJSON(const StepInTarget &SIT) {
  json::Object target{{"id", SIT.id}, {"label", SIT.label}};

  if (SIT.line != LLDB_INVALID_LINE_NUMBER)
    target.insert({"line", SIT.line});
  if (SIT.column != LLDB_INVALID_COLUMN_NUMBER)
    target.insert({"column", SIT.column});
  if (SIT.endLine != LLDB_INVALID_LINE_NUMBER)
    target.insert({"endLine", SIT.endLine});
  if (SIT.endLine != LLDB_INVALID_COLUMN_NUMBER)
    target.insert({"endColumn", SIT.endColumn});

  return target;
}

bool fromJSON(const json::Value &Params, Thread &T, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("id", T.id) && O.map("name", T.name);
}

json::Value toJSON(const Thread &T) {
  return json::Object{{"id", T.id}, {"name", T.name}};
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

bool fromJSON(const llvm::json::Value &Params,
              DisassembledInstruction::PresentationHint &PH,
              llvm::json::Path P) {
  auto rawHint = Params.getAsString();
  if (!rawHint) {
    P.report("expected a string");
    return false;
  }
  std::optional<DisassembledInstruction::PresentationHint> hint =
      StringSwitch<std::optional<DisassembledInstruction::PresentationHint>>(
          *rawHint)
          .Case("normal", DisassembledInstruction::
                              eDisassembledInstructionPresentationHintNormal)
          .Case("invalid", DisassembledInstruction::
                               eDisassembledInstructionPresentationHintInvalid)
          .Default(std::nullopt);
  if (!hint) {
    P.report("unexpected value");
    return false;
  }
  PH = *hint;
  return true;
}

llvm::json::Value toJSON(const DisassembledInstruction::PresentationHint &PH) {
  switch (PH) {
  case DisassembledInstruction::eDisassembledInstructionPresentationHintNormal:
    return "normal";
  case DisassembledInstruction::eDisassembledInstructionPresentationHintInvalid:
    return "invalid";
  }
  llvm_unreachable("unhandled presentation hint.");
}

bool fromJSON(const llvm::json::Value &Params, DisassembledInstruction &DI,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O &&
         DecodeMemoryReference(Params, "address", DI.address, P,
                               /*required=*/true) &&
         O.map("instruction", DI.instruction) &&
         O.mapOptional("instructionBytes", DI.instructionBytes) &&
         O.mapOptional("symbol", DI.symbol) &&
         O.mapOptional("location", DI.location) &&
         O.mapOptional("line", DI.line) && O.mapOptional("column", DI.column) &&
         O.mapOptional("endLine", DI.endLine) &&
         O.mapOptional("endColumn", DI.endColumn) &&
         O.mapOptional("presentationHint", DI.presentationHint);
}

llvm::json::Value toJSON(const DisassembledInstruction &DI) {
  llvm::json::Object result{{"instruction", DI.instruction}};
  if (DI.address == LLDB_INVALID_ADDRESS) {
    // VS Code has explicit comparisons to the string "-1" in order to check for
    // invalid instructions. See
    // https://github.com/microsoft/vscode/blob/main/src/vs/workbench/contrib/debug/browser/disassemblyView.ts
    result.insert({"address", "-1"});
  } else {
    result.insert({"address", "0x" + llvm::utohexstr(DI.address)});
  }

  if (DI.instructionBytes)
    result.insert({"instructionBytes", *DI.instructionBytes});
  if (DI.symbol)
    result.insert({"symbol", *DI.symbol});
  if (DI.location)
    result.insert({"location", *DI.location});
  if (DI.line)
    result.insert({"line", *DI.line});
  if (DI.column)
    result.insert({"column", *DI.column});
  if (DI.endLine)
    result.insert({"endLine", *DI.endLine});
  if (DI.endColumn)
    result.insert({"endColumn", *DI.endColumn});
  if (DI.presentationHint)
    result.insert({"presentationHint", *DI.presentationHint});

  return result;
}

json::Value toJSON(const Module &M) {
  json::Object result{{"id", M.id}, {"name", M.name}};

  if (!M.path.empty())
    result.insert({"path", M.path});
  if (M.isOptimized)
    result.insert({"isOptimized", M.isOptimized});
  if (M.isUserCode)
    result.insert({"isUserCode", M.isUserCode});
  if (!M.version.empty())
    result.insert({"version", M.version});
  if (!M.symbolStatus.empty())
    result.insert({"symbolStatus", M.symbolStatus});
  if (!M.symbolFilePath.empty())
    result.insert({"symbolFilePath", M.symbolFilePath});
  if (!M.dateTimeStamp.empty())
    result.insert({"dateTimeStamp", M.dateTimeStamp});
  if (!M.addressRange.empty())
    result.insert({"addressRange", M.addressRange});
  if (M.debugInfoSizeBytes != 0)
    result.insert(
        {"debugInfoSize", ConvertDebugInfoSizeToString(M.debugInfoSizeBytes)});

  return result;
}

json::Value toJSON(const VariablePresentationHint &VPH) {
  json::Object result{};

  if (!VPH.kind.empty())
    result.insert({"kind", VPH.kind});
  if (!VPH.attributes.empty())
    result.insert({"attributes", VPH.attributes});
  if (!VPH.visibility.empty())
    result.insert({"visibility", VPH.visibility});
  if (VPH.lazy)
    result.insert({"lazy", VPH.lazy});

  return result;
}

bool fromJSON(const json::Value &Param, VariablePresentationHint &VPH,
              json::Path Path) {
  json::ObjectMapper O(Param, Path);
  return O && O.mapOptional("kind", VPH.kind) &&
         O.mapOptional("attributes", VPH.attributes) &&
         O.mapOptional("visibility", VPH.visibility) &&
         O.mapOptional("lazy", VPH.lazy);
}

json::Value toJSON(const Variable &V) {
  json::Object result{{"name", V.name},
                      {"variablesReference", V.variablesReference},
                      {"value", V.value}};

  if (!V.type.empty())
    result.insert({"type", V.type});
  if (V.presentationHint)
    result.insert({"presentationHint", *V.presentationHint});
  if (!V.evaluateName.empty())
    result.insert({"evaluateName", V.evaluateName});
  if (V.namedVariables)
    result.insert({"namedVariables", V.namedVariables});
  if (V.indexedVariables)
    result.insert({"indexedVariables", V.indexedVariables});
  if (V.memoryReference != LLDB_INVALID_ADDRESS)
    result.insert(
        {"memoryReference", EncodeMemoryReference(V.memoryReference)});
  if (V.declarationLocationReference)
    result.insert(
        {"declarationLocationReference", V.declarationLocationReference});
  if (V.valueLocationReference)
    result.insert({"valueLocationReference", V.valueLocationReference});

  return result;
}

bool fromJSON(const json::Value &Param, Variable &V, json::Path Path) {
  json::ObjectMapper O(Param, Path);
  return O && O.map("name", V.name) &&
         O.map("variablesReference", V.variablesReference) &&
         O.map("value", V.value) && O.mapOptional("type", V.type) &&
         O.mapOptional("presentationHint", *V.presentationHint) &&
         O.mapOptional("evaluateName", V.evaluateName) &&
         O.mapOptional("namedVariables", V.namedVariables) &&
         O.mapOptional("indexedVariables", V.indexedVariables) &&
         O.mapOptional("declarationLocationReference",
                       V.declarationLocationReference) &&
         O.mapOptional("valueLocationReference", V.valueLocationReference) &&
         DecodeMemoryReference(Param, "memoryReference", V.memoryReference,
                               Path, /*required=*/false);
}

} // namespace lldb_dap::protocol
