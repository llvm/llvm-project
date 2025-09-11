//===-- ProtocolTypesTest.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolTypes.h"
#include "Protocol/ProtocolEvents.h"
#include "Protocol/ProtocolRequests.h"
#include "TestingSupport/TestUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;
using lldb_private::roundtripJSON;
using llvm::json::parse;
using llvm::json::Value;

/// Returns a pretty printed json string of a `llvm::json::Value`.
static std::string pp(const json::Value &E) {
  return formatv("{0:2}", E).str();
}

TEST(ProtocolTypesTest, ExceptionBreakpointsFilter) {
  ExceptionBreakpointsFilter filter;
  filter.filter = "testFilter";
  filter.label = "Test Filter";
  filter.description = "This is a test filter";
  filter.defaultState = true;
  filter.supportsCondition = true;
  filter.conditionDescription = "Condition for test filter";

  llvm::Expected<ExceptionBreakpointsFilter> deserialized_filter =
      roundtripJSON(filter);
  ASSERT_THAT_EXPECTED(deserialized_filter, llvm::Succeeded());

  EXPECT_EQ(filter.filter, deserialized_filter->filter);
  EXPECT_EQ(filter.label, deserialized_filter->label);
  EXPECT_EQ(filter.description, deserialized_filter->description);
  EXPECT_EQ(filter.defaultState, deserialized_filter->defaultState);
  EXPECT_EQ(filter.supportsCondition, deserialized_filter->supportsCondition);
  EXPECT_EQ(filter.conditionDescription,
            deserialized_filter->conditionDescription);
}

TEST(ProtocolTypesTest, Source) {
  Source source;
  source.name = "testName";
  source.path = "/path/to/source";
  source.sourceReference = 12345;
  source.presentationHint = Source::eSourcePresentationHintEmphasize;

  llvm::Expected<Source> deserialized_source = roundtripJSON(source);
  ASSERT_THAT_EXPECTED(deserialized_source, llvm::Succeeded());

  EXPECT_EQ(source.name, deserialized_source->name);
  EXPECT_EQ(source.path, deserialized_source->path);
  EXPECT_EQ(source.sourceReference, deserialized_source->sourceReference);
  EXPECT_EQ(source.presentationHint, deserialized_source->presentationHint);
}

TEST(ProtocolTypesTest, ColumnDescriptor) {
  ColumnDescriptor column;
  column.attributeName = "moduleName";
  column.label = "Module Name";
  column.format = "uppercase";
  column.type = eColumnTypeString;
  column.width = 20;

  llvm::Expected<ColumnDescriptor> deserialized_column = roundtripJSON(column);
  ASSERT_THAT_EXPECTED(deserialized_column, llvm::Succeeded());

  EXPECT_EQ(column.attributeName, deserialized_column->attributeName);
  EXPECT_EQ(column.label, deserialized_column->label);
  EXPECT_EQ(column.format, deserialized_column->format);
  EXPECT_EQ(column.type, deserialized_column->type);
  EXPECT_EQ(column.width, deserialized_column->width);
}

TEST(ProtocolTypesTest, BreakpointMode) {
  BreakpointMode mode;
  mode.mode = "testMode";
  mode.label = "Test Mode";
  mode.description = "This is a test mode";
  mode.appliesTo = {eBreakpointModeApplicabilitySource,
                    eBreakpointModeApplicabilityException};

  llvm::Expected<BreakpointMode> deserialized_mode = roundtripJSON(mode);
  ASSERT_THAT_EXPECTED(deserialized_mode, llvm::Succeeded());

  EXPECT_EQ(mode.mode, deserialized_mode->mode);
  EXPECT_EQ(mode.label, deserialized_mode->label);
  EXPECT_EQ(mode.description, deserialized_mode->description);
  EXPECT_EQ(mode.appliesTo, deserialized_mode->appliesTo);
}

TEST(ProtocolTypesTest, Breakpoint) {
  Breakpoint breakpoint;
  breakpoint.id = 42;
  breakpoint.verified = true;
  breakpoint.message = "Breakpoint set successfully";
  breakpoint.source =
      Source{"test.cpp", "/path/to/test.cpp", 123,
             Source::eSourcePresentationHintNormal, std::nullopt};
  breakpoint.line = 10;
  breakpoint.column = 5;
  breakpoint.endLine = 15;
  breakpoint.endColumn = 10;
  breakpoint.instructionReference = "0x12345678";
  breakpoint.offset = 4;
  breakpoint.reason = BreakpointReason::eBreakpointReasonPending;

  llvm::Expected<Breakpoint> deserialized_breakpoint =
      roundtripJSON(breakpoint);
  ASSERT_THAT_EXPECTED(deserialized_breakpoint, llvm::Succeeded());

  EXPECT_EQ(breakpoint.id, deserialized_breakpoint->id);
  EXPECT_EQ(breakpoint.verified, deserialized_breakpoint->verified);
  EXPECT_EQ(breakpoint.message, deserialized_breakpoint->message);
  EXPECT_EQ(breakpoint.source->name, deserialized_breakpoint->source->name);
  EXPECT_EQ(breakpoint.source->path, deserialized_breakpoint->source->path);
  EXPECT_EQ(breakpoint.source->sourceReference,
            deserialized_breakpoint->source->sourceReference);
  EXPECT_EQ(breakpoint.source->presentationHint,
            deserialized_breakpoint->source->presentationHint);
  EXPECT_EQ(breakpoint.line, deserialized_breakpoint->line);
  EXPECT_EQ(breakpoint.column, deserialized_breakpoint->column);
  EXPECT_EQ(breakpoint.endLine, deserialized_breakpoint->endLine);
  EXPECT_EQ(breakpoint.endColumn, deserialized_breakpoint->endColumn);
  EXPECT_EQ(breakpoint.instructionReference,
            deserialized_breakpoint->instructionReference);
  EXPECT_EQ(breakpoint.offset, deserialized_breakpoint->offset);
  EXPECT_EQ(breakpoint.reason, deserialized_breakpoint->reason);
}

TEST(ProtocolTypesTest, SourceBreakpoint) {
  SourceBreakpoint source_breakpoint;
  source_breakpoint.line = 42;
  source_breakpoint.column = 5;
  source_breakpoint.condition = "x > 10";
  source_breakpoint.hitCondition = "5";
  source_breakpoint.logMessage = "Breakpoint hit at line 42";
  source_breakpoint.mode = "hardware";

  llvm::Expected<SourceBreakpoint> deserialized_source_breakpoint =
      roundtripJSON(source_breakpoint);
  ASSERT_THAT_EXPECTED(deserialized_source_breakpoint, llvm::Succeeded());

  EXPECT_EQ(source_breakpoint.line, deserialized_source_breakpoint->line);
  EXPECT_EQ(source_breakpoint.column, deserialized_source_breakpoint->column);
  EXPECT_EQ(source_breakpoint.condition,
            deserialized_source_breakpoint->condition);
  EXPECT_EQ(source_breakpoint.hitCondition,
            deserialized_source_breakpoint->hitCondition);
  EXPECT_EQ(source_breakpoint.logMessage,
            deserialized_source_breakpoint->logMessage);
  EXPECT_EQ(source_breakpoint.mode, deserialized_source_breakpoint->mode);
}

TEST(ProtocolTypesTest, FunctionBreakpoint) {
  FunctionBreakpoint function_breakpoint;
  function_breakpoint.name = "myFunction";
  function_breakpoint.condition = "x == 0";
  function_breakpoint.hitCondition = "3";

  llvm::Expected<FunctionBreakpoint> deserialized_function_breakpoint =
      roundtripJSON(function_breakpoint);
  ASSERT_THAT_EXPECTED(deserialized_function_breakpoint, llvm::Succeeded());

  EXPECT_EQ(function_breakpoint.name, deserialized_function_breakpoint->name);
  EXPECT_EQ(function_breakpoint.condition,
            deserialized_function_breakpoint->condition);
  EXPECT_EQ(function_breakpoint.hitCondition,
            deserialized_function_breakpoint->hitCondition);
}

TEST(ProtocolTypesTest, DataBreakpoint) {
  DataBreakpoint data_breakpoint_info;
  data_breakpoint_info.dataId = "variable1";
  data_breakpoint_info.accessType = eDataBreakpointAccessTypeReadWrite;
  data_breakpoint_info.condition = "x > 100";
  data_breakpoint_info.hitCondition = "10";

  llvm::Expected<DataBreakpoint> deserialized_data_breakpoint_info =
      roundtripJSON(data_breakpoint_info);
  ASSERT_THAT_EXPECTED(deserialized_data_breakpoint_info, llvm::Succeeded());

  EXPECT_EQ(data_breakpoint_info.dataId,
            deserialized_data_breakpoint_info->dataId);
  EXPECT_EQ(data_breakpoint_info.accessType,
            deserialized_data_breakpoint_info->accessType);
  EXPECT_EQ(data_breakpoint_info.condition,
            deserialized_data_breakpoint_info->condition);
  EXPECT_EQ(data_breakpoint_info.hitCondition,
            deserialized_data_breakpoint_info->hitCondition);
}

TEST(ProtocolTypesTest, Capabilities) {
  Capabilities capabilities;

  // Populate supported features.
  capabilities.supportedFeatures.insert(eAdapterFeatureANSIStyling);
  capabilities.supportedFeatures.insert(
      eAdapterFeatureBreakpointLocationsRequest);

  // Populate optional fields.
  capabilities.exceptionBreakpointFilters = {
      {{"filter1", "Filter 1", "Description 1", true, true, "Condition 1"},
       {"filter2", "Filter 2", "Description 2", false, false, "Condition 2"}}};

  capabilities.completionTriggerCharacters = {".", "->"};
  capabilities.additionalModuleColumns = {
      {"moduleName", "Module Name", "uppercase", eColumnTypeString, 20}};
  capabilities.supportedChecksumAlgorithms = {eChecksumAlgorithmMD5,
                                              eChecksumAlgorithmSHA256};
  capabilities.breakpointModes = {{"hardware",
                                   "Hardware Breakpoint",
                                   "Description",
                                   {eBreakpointModeApplicabilitySource}}};
  capabilities.lldbExtVersion = "1.0.0";

  // Perform roundtripJSON serialization and deserialization.
  llvm::Expected<Capabilities> deserialized_capabilities =
      roundtripJSON(capabilities);
  ASSERT_THAT_EXPECTED(deserialized_capabilities, llvm::Succeeded());

  // Verify supported features.
  EXPECT_EQ(capabilities.supportedFeatures,
            deserialized_capabilities->supportedFeatures);

  // Verify exception breakpoint filters.
  EXPECT_EQ(capabilities.exceptionBreakpointFilters.size(),
            deserialized_capabilities->exceptionBreakpointFilters.size());
  for (size_t i = 0; i < capabilities.exceptionBreakpointFilters.size(); ++i) {
    const auto &original = capabilities.exceptionBreakpointFilters.at(i);
    const auto &deserialized =
        deserialized_capabilities->exceptionBreakpointFilters.at(i);
    EXPECT_EQ(original.filter, deserialized.filter);
    EXPECT_EQ(original.label, deserialized.label);
    EXPECT_EQ(original.description, deserialized.description);
    EXPECT_EQ(original.defaultState, deserialized.defaultState);
    EXPECT_EQ(original.supportsCondition, deserialized.supportsCondition);
    EXPECT_EQ(original.conditionDescription, deserialized.conditionDescription);
  }

  // Verify completion trigger characters.
  EXPECT_EQ(capabilities.completionTriggerCharacters,
            deserialized_capabilities->completionTriggerCharacters);

  // Verify additional module columns.
  EXPECT_EQ(capabilities.additionalModuleColumns.size(),
            deserialized_capabilities->additionalModuleColumns.size());
  for (size_t i = 0; i < capabilities.additionalModuleColumns.size(); ++i) {
    const auto &original = capabilities.additionalModuleColumns.at(i);
    const auto &deserialized =
        deserialized_capabilities->additionalModuleColumns.at(i);
    EXPECT_EQ(original.attributeName, deserialized.attributeName);
    EXPECT_EQ(original.label, deserialized.label);
    EXPECT_EQ(original.format, deserialized.format);
    EXPECT_EQ(original.type, deserialized.type);
    EXPECT_EQ(original.width, deserialized.width);
  }

  // Verify supported checksum algorithms.
  EXPECT_EQ(capabilities.supportedChecksumAlgorithms,
            deserialized_capabilities->supportedChecksumAlgorithms);

  // Verify breakpoint modes.
  EXPECT_EQ(capabilities.breakpointModes.size(),
            deserialized_capabilities->breakpointModes.size());
  for (size_t i = 0; i < capabilities.breakpointModes.size(); ++i) {
    const auto &original = capabilities.breakpointModes.at(i);
    const auto &deserialized = deserialized_capabilities->breakpointModes.at(i);
    EXPECT_EQ(original.mode, deserialized.mode);
    EXPECT_EQ(original.label, deserialized.label);
    EXPECT_EQ(original.description, deserialized.description);
    EXPECT_EQ(original.appliesTo, deserialized.appliesTo);
  }

  // Verify lldb extension version.
  EXPECT_EQ(capabilities.lldbExtVersion,
            deserialized_capabilities->lldbExtVersion);
}

TEST(ProtocolTypesTest, Scope) {
  Scope scope;
  scope.name = "Locals";
  scope.presentationHint = Scope::eScopePresentationHintLocals;
  scope.variablesReference = 1;
  scope.namedVariables = 2;
  scope.indexedVariables = std::nullopt;
  scope.expensive = false;
  scope.line = 2;
  scope.column = 3;
  scope.endLine = 10;
  scope.endColumn = 20;

  Source source;
  source.name = "testName";
  source.path = "/path/to/source";
  source.sourceReference = 12345;
  source.presentationHint = Source::eSourcePresentationHintNormal;
  scope.source = source;

  llvm::Expected<Scope> deserialized_scope = roundtripJSON(scope);
  ASSERT_THAT_EXPECTED(deserialized_scope, llvm::Succeeded());
  EXPECT_EQ(scope.name, deserialized_scope->name);
  EXPECT_EQ(scope.presentationHint, deserialized_scope->presentationHint);
  EXPECT_EQ(scope.variablesReference, deserialized_scope->variablesReference);
  EXPECT_EQ(scope.namedVariables, deserialized_scope->namedVariables);
  EXPECT_EQ(scope.indexedVariables, deserialized_scope->indexedVariables);
  EXPECT_EQ(scope.expensive, deserialized_scope->expensive);
  EXPECT_EQ(scope.line, deserialized_scope->line);
  EXPECT_EQ(scope.column, deserialized_scope->column);
  EXPECT_EQ(scope.endLine, deserialized_scope->endLine);
  EXPECT_EQ(scope.endColumn, deserialized_scope->endColumn);

  EXPECT_THAT(deserialized_scope->source.has_value(), true);
  const Source &deserialized_source = deserialized_scope->source.value();

  EXPECT_EQ(source.path, deserialized_source.path);
  EXPECT_EQ(source.sourceReference, deserialized_source.sourceReference);
  EXPECT_EQ(source.presentationHint, deserialized_source.presentationHint);
}

TEST(ProtocolTypesTest, PresentationHint) {
  // Test all PresentationHint values.
  std::vector<std::pair<Source::PresentationHint, llvm::StringRef>> test_cases =
      {{Source::eSourcePresentationHintNormal, "normal"},
       {Source::eSourcePresentationHintEmphasize, "emphasize"},
       {Source::eSourcePresentationHintDeemphasize, "deemphasize"}};

  for (const auto &test_case : test_cases) {
    // Serialize the PresentationHint to JSON.
    llvm::json::Value serialized = toJSON(test_case.first);
    ASSERT_EQ(serialized.kind(), llvm::json::Value::Kind::String);
    EXPECT_EQ(serialized.getAsString(), test_case.second);

    // Deserialize the JSON back to PresentationHint.
    Source::PresentationHint deserialized;
    llvm::json::Path::Root root;
    ASSERT_TRUE(fromJSON(serialized, deserialized, root))
        << llvm::toString(root.getError());
    EXPECT_EQ(deserialized, test_case.first);
  }

  // Test invalid value.
  llvm::json::Value invalid_value = "invalid_hint";
  Source::PresentationHint deserialized_invalid;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(invalid_value, deserialized_invalid, root));
}

TEST(ProtocolTypesTest, SteppingGranularity) {
  // Test all SteppingGranularity values.
  std::vector<std::pair<SteppingGranularity, llvm::StringRef>> test_cases = {
      {eSteppingGranularityStatement, "statement"},
      {eSteppingGranularityLine, "line"},
      {eSteppingGranularityInstruction, "instruction"}};

  for (const auto &test_case : test_cases) {
    // Serialize the SteppingGranularity to JSON.
    llvm::json::Value serialized = toJSON(test_case.first);
    ASSERT_EQ(serialized.kind(), llvm::json::Value::Kind::String);
    EXPECT_EQ(serialized.getAsString(), test_case.second);

    // Deserialize the JSON back to SteppingGranularity.
    SteppingGranularity deserialized;
    llvm::json::Path::Root root;
    ASSERT_TRUE(fromJSON(serialized, deserialized, root))
        << llvm::toString(root.getError());
    EXPECT_EQ(deserialized, test_case.first);
  }

  // Test invalid value.
  llvm::json::Value invalid_value = "invalid_granularity";
  SteppingGranularity deserialized_invalid;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(invalid_value, deserialized_invalid, root));
}

TEST(ProtocolTypesTest, BreakpointReason) {
  // Test all BreakpointReason values.
  std::vector<std::pair<BreakpointReason, llvm::StringRef>> test_cases = {
      {BreakpointReason::eBreakpointReasonPending, "pending"},
      {BreakpointReason::eBreakpointReasonFailed, "failed"}};

  for (const auto &test_case : test_cases) {
    // Serialize the BreakpointReason to JSON.
    llvm::json::Value serialized = toJSON(test_case.first);
    ASSERT_EQ(serialized.kind(), llvm::json::Value::Kind::String);
    EXPECT_EQ(serialized.getAsString(), test_case.second);

    // Deserialize the JSON back to BreakpointReason.
    BreakpointReason deserialized;
    llvm::json::Path::Root root;
    ASSERT_TRUE(fromJSON(serialized, deserialized, root))
        << llvm::toString(root.getError());
    EXPECT_EQ(deserialized, test_case.first);
  }

  // Test invalid value.
  llvm::json::Value invalid_value = "invalid_reason";
  BreakpointReason deserialized_invalid;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(invalid_value, deserialized_invalid, root));
}

TEST(ProtocolTypesTest, DataBreakpointAccessType) {
  // Test all DataBreakpointAccessType values.
  std::vector<std::pair<DataBreakpointAccessType, llvm::StringRef>> test_cases =
      {{eDataBreakpointAccessTypeRead, "read"},
       {eDataBreakpointAccessTypeWrite, "write"},
       {eDataBreakpointAccessTypeReadWrite, "readWrite"}};

  for (const auto &test_case : test_cases) {
    // Serialize the DataBreakpointAccessType to JSON.
    llvm::json::Value serialized = toJSON(test_case.first);
    ASSERT_EQ(serialized.kind(), llvm::json::Value::Kind::String);
    EXPECT_EQ(serialized.getAsString(), test_case.second);

    // Deserialize the JSON back to DataBreakpointAccessType.
    DataBreakpointAccessType deserialized;
    llvm::json::Path::Root root;
    ASSERT_TRUE(fromJSON(serialized, deserialized, root))
        << llvm::toString(root.getError());
    EXPECT_EQ(deserialized, test_case.first);
  }

  // Test invalid value
  llvm::json::Value invalid_value = "invalid_access_type";
  DataBreakpointAccessType deserialized_invalid;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(invalid_value, deserialized_invalid, root));
}

TEST(ProtocolTypesTest, ColumnType) {
  // Test all ColumnType values.
  std::vector<std::pair<ColumnType, llvm::StringRef>> test_cases = {
      {eColumnTypeString, "string"},
      {eColumnTypeNumber, "number"},
      {eColumnTypeBoolean, "boolean"},
      {eColumnTypeTimestamp, "unixTimestampUTC"}};

  for (const auto &test_case : test_cases) {
    // Serialize the ColumnType to JSON.
    llvm::json::Value serialized = toJSON(test_case.first);
    ASSERT_EQ(serialized.kind(), llvm::json::Value::Kind::String);
    EXPECT_EQ(serialized.getAsString(), test_case.second);

    // Deserialize the JSON back to ColumnType.
    ColumnType deserialized;
    llvm::json::Path::Root root;
    ASSERT_TRUE(fromJSON(serialized, deserialized, root))
        << llvm::toString(root.getError());
    EXPECT_EQ(deserialized, test_case.first);
  }

  // Test invalid value.
  llvm::json::Value invalid_value = "invalid_column_type";
  ColumnType deserialized_invalid;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(invalid_value, deserialized_invalid, root));
}

TEST(ProtocolTypesTest, BreakpointModeApplicability) {
  // Test all BreakpointModeApplicability values.
  std::vector<std::pair<BreakpointModeApplicability, llvm::StringRef>>
      test_cases = {{eBreakpointModeApplicabilitySource, "source"},
                    {eBreakpointModeApplicabilityException, "exception"},
                    {eBreakpointModeApplicabilityData, "data"},
                    {eBreakpointModeApplicabilityInstruction, "instruction"}};

  for (const auto &test_case : test_cases) {
    // Serialize the BreakpointModeApplicability to JSON.
    llvm::json::Value serialized = toJSON(test_case.first);
    ASSERT_EQ(serialized.kind(), llvm::json::Value::Kind::String);
    EXPECT_EQ(serialized.getAsString(), test_case.second);

    // Deserialize the JSON back to BreakpointModeApplicability.
    BreakpointModeApplicability deserialized;
    llvm::json::Path::Root root;
    ASSERT_TRUE(fromJSON(serialized, deserialized, root))
        << llvm::toString(root.getError());
    EXPECT_EQ(deserialized, test_case.first);
  }

  // Test invalid value.
  llvm::json::Value invalid_value = "invalid_applicability";
  BreakpointModeApplicability deserialized_invalid;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(invalid_value, deserialized_invalid, root));
}

TEST(ProtocolTypesTest, ChecksumAlgorithm) {
  // Test all ChecksumAlgorithm values.
  std::vector<std::pair<ChecksumAlgorithm, llvm::StringRef>> test_cases = {
      {eChecksumAlgorithmMD5, "MD5"},
      {eChecksumAlgorithmSHA1, "SHA1"},
      {eChecksumAlgorithmSHA256, "SHA256"},
      {eChecksumAlgorithmTimestamp, "timestamp"}};

  for (const auto &test_case : test_cases) {
    // Serialize the ChecksumAlgorithm to JSON.
    llvm::json::Value serialized = toJSON(test_case.first);
    ASSERT_EQ(serialized.kind(), llvm::json::Value::Kind::String);
    EXPECT_EQ(serialized.getAsString(), test_case.second);

    // Deserialize the JSON back to ChecksumAlgorithm.
    ChecksumAlgorithm deserialized;
    llvm::json::Path::Root root;
    ASSERT_TRUE(fromJSON(serialized, deserialized, root))
        << llvm::toString(root.getError());
    EXPECT_EQ(deserialized, test_case.first);
  }

  // Test invalid value.
  llvm::json::Value invalid_value = "invalid_algorithm";
  ChecksumAlgorithm deserialized_invalid;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(invalid_value, deserialized_invalid, root));
}

TEST(ProtocolTypesTest, DisassembledInstructionPresentationHint) {
  // Test all PresentationHint values.
  std::vector<
      std::pair<DisassembledInstruction::PresentationHint, llvm::StringRef>>
      test_cases = {{DisassembledInstruction::
                         eDisassembledInstructionPresentationHintNormal,
                     "normal"},
                    {DisassembledInstruction::
                         eDisassembledInstructionPresentationHintInvalid,
                     "invalid"}};

  for (const auto &test_case : test_cases) {
    // Serialize the PresentationHint to JSON.
    llvm::json::Value serialized = toJSON(test_case.first);
    ASSERT_EQ(serialized.kind(), llvm::json::Value::Kind::String);
    EXPECT_EQ(serialized.getAsString(), test_case.second);

    // Deserialize the JSON back to PresentationHint.
    DisassembledInstruction::PresentationHint deserialized;
    llvm::json::Path::Root root;
    ASSERT_TRUE(fromJSON(serialized, deserialized, root))
        << llvm::toString(root.getError());
    EXPECT_EQ(deserialized, test_case.first);
  }

  // Test invalid value.
  llvm::json::Value invalid_value = "invalid_hint";
  DisassembledInstruction::PresentationHint deserialized_invalid;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(invalid_value, deserialized_invalid, root));
}

TEST(ProtocolTypesTest, DisassembledInstruction) {
  DisassembledInstruction instruction;
  instruction.address = 0x12345678;
  instruction.instructionBytes = "0F 1F 00";
  instruction.instruction = "mov eax, ebx";
  instruction.symbol = "main";
  instruction.location =
      Source{"test.cpp", "/path/to/test.cpp", 123,
             Source::eSourcePresentationHintNormal, std::nullopt};
  instruction.line = 10;
  instruction.column = 5;
  instruction.endLine = 15;
  instruction.endColumn = 10;
  instruction.presentationHint =
      DisassembledInstruction::eDisassembledInstructionPresentationHintNormal;

  StringLiteral json = R"({
  "address": "0x12345678",
  "column": 5,
  "endColumn": 10,
  "endLine": 15,
  "instruction": "mov eax, ebx",
  "instructionBytes": "0F 1F 00",
  "line": 10,
  "location": {
    "name": "test.cpp",
    "path": "/path/to/test.cpp",
    "presentationHint": "normal",
    "sourceReference": 123
  },
  "presentationHint": "normal",
  "symbol": "main"
})";

  // Validate toJSON
  EXPECT_EQ(json, pp(instruction));

  // Validate fromJSON
  EXPECT_THAT_EXPECTED(parse<DisassembledInstruction>(json),
                       HasValue(Value(instruction)));
  // Validate parsing errors
  EXPECT_THAT_EXPECTED(
      parse<DisassembledInstruction>(R"({"address":1})",
                                     "disassemblyInstruction"),
      FailedWithMessage("expected string at disassemblyInstruction.address"));
  EXPECT_THAT_EXPECTED(
      parse<DisassembledInstruction>(R"({"address":"-1"})",
                                     "disassemblyInstruction"),
      FailedWithMessage(
          "malformed memory reference at disassemblyInstruction.address"));
  EXPECT_THAT_EXPECTED(
      parse<DisassembledInstruction>(
          R"({"address":"0xfffffffffffffffffffffffffff"})",
          "disassemblyInstruction"),
      FailedWithMessage(
          "malformed memory reference at disassemblyInstruction.address"));
}

TEST(ProtocolTypesTest, Thread) {
  const Thread thread{1, "thr1"};
  const StringRef json = R"({
  "id": 1,
  "name": "thr1"
})";
  // Validate toJSON
  EXPECT_EQ(json, pp(thread));
  // Validate fromJSON
  EXPECT_THAT_EXPECTED(parse<Thread>(json), HasValue(Value(thread)));
  // Validate parsing errors
  EXPECT_THAT_EXPECTED(parse<Thread>(R"({"id":1})", "thread"),
                       FailedWithMessage("missing value at thread.name"));
  EXPECT_THAT_EXPECTED(parse<Thread>(R"({"id":"one"})", "thread"),
                       FailedWithMessage("expected uint64_t at thread.id"));
  EXPECT_THAT_EXPECTED(parse<Thread>(R"({"id":1,"name":false})", "thread"),
                       FailedWithMessage("expected string at thread.name"));
}

TEST(ProtocolTypesTest, ThreadResponseBody) {
  const ThreadsResponseBody body{{{1, "thr1"}, {2, "thr2"}}};
  const StringRef json = R"({
  "threads": [
    {
      "id": 1,
      "name": "thr1"
    },
    {
      "id": 2,
      "name": "thr2"
    }
  ]
})";
  // Validate toJSON
  EXPECT_EQ(json, pp(body));
}

TEST(ProtocolTypesTest, CapabilitiesEventBody) {
  Capabilities capabilities;
  capabilities.supportedFeatures = {
      eAdapterFeatureANSIStyling,
      eAdapterFeatureBreakpointLocationsRequest,
  };
  CapabilitiesEventBody body;
  body.capabilities = capabilities;
  StringRef json = R"({
  "capabilities": {
    "supportsANSIStyling": true,
    "supportsBreakpointLocationsRequest": true
  }
})";
  // Validate toJSON
  EXPECT_EQ(json, pp(body));
}

TEST(ProtocolTypesTest, ExceptionFilterOptions) {
  EXPECT_THAT_EXPECTED(parse<ExceptionFilterOptions>(R"({"filterId":"id"})"),
                       HasValue(Value(ExceptionFilterOptions{
                           /*filterId=*/"id", /*condition=*/"", /*mode*/ ""})));
  EXPECT_THAT_EXPECTED(
      parse<ExceptionFilterOptions>(R"({"filterId":"id","condition":"1+2"})"),
      HasValue(Value(ExceptionFilterOptions{
          /*filterId=*/"id", /*condition=*/"1+2", /*mode*/ ""})));
  EXPECT_THAT_EXPECTED(
      parse<ExceptionFilterOptions>(
          R"({"filterId":"id","condition":"1+2","mode":"m"})"),
      HasValue(Value(ExceptionFilterOptions{
          /*filterId=*/"id", /*condition=*/"1+2", /*mode*/ "m"})));

  // Validate parsing errors
  EXPECT_THAT_EXPECTED(
      parse<ExceptionFilterOptions>(R"({})", "exceptionFilterOptions"),
      FailedWithMessage("missing value at exceptionFilterOptions.filterId"));
  EXPECT_THAT_EXPECTED(
      parse<ExceptionFilterOptions>(R"({"filterId":"id","condition":42})",
                                    "exceptionFilterOptions"),
      FailedWithMessage("expected string at exceptionFilterOptions.condition"));
  EXPECT_THAT_EXPECTED(
      parse<ExceptionFilterOptions>(R"({"filterId":"id","mode":42})",
                                    "exceptionFilterOptions"),
      FailedWithMessage("expected string at exceptionFilterOptions.mode"));
}

TEST(ProtocolTypesTest, SetExceptionBreakpointsArguments) {
  EXPECT_THAT_EXPECTED(
      parse<SetExceptionBreakpointsArguments>(R"({"filters":[]})"),
      HasValue(testing::FieldsAre(/*filters=*/testing::IsEmpty(),
                                  /*filterOptions=*/testing::IsEmpty())));
  EXPECT_THAT_EXPECTED(
      parse<SetExceptionBreakpointsArguments>(R"({"filters":["abc"]})"),
      HasValue(testing::FieldsAre(/*filters=*/std::vector<std::string>{"abc"},
                                  /*filterOptions=*/testing::IsEmpty())));
  EXPECT_THAT_EXPECTED(
      parse<SetExceptionBreakpointsArguments>(
          R"({"filters":[],"filterOptions":[{"filterId":"abc"}]})"),
      HasValue(testing::FieldsAre(
          /*filters=*/testing::IsEmpty(),
          /*filterOptions=*/testing::Contains(testing::FieldsAre(
              /*filterId=*/"abc", /*condition=*/"", /*mode=*/"")))));

  // Validate parse errors
  EXPECT_THAT_EXPECTED(parse<SetExceptionBreakpointsArguments>(R"({})"),
                       FailedWithMessage("missing value at (root).filters"));
  EXPECT_THAT_EXPECTED(
      parse<SetExceptionBreakpointsArguments>(R"({"filters":false})"),
      FailedWithMessage("expected array at (root).filters"));
}

TEST(ProtocolTypesTest, SetExceptionBreakpointsResponseBody) {
  SetExceptionBreakpointsResponseBody body;
  Breakpoint bp;
  bp.id = 12, bp.verified = true;
  body.breakpoints = {bp};
  EXPECT_EQ(R"({
  "breakpoints": [
    {
      "id": 12,
      "verified": true
    }
  ]
})",
            pp(body));
}

TEST(ProtocolTypesTest, StepInTarget) {
  StepInTarget target;
  target.id = 230;
  target.label = "the_function_name";
  target.line = 2;
  target.column = 320;
  target.endLine = 32;
  target.endColumn = 23;

  llvm::Expected<StepInTarget> deserialized_target = roundtripJSON(target);
  ASSERT_THAT_EXPECTED(deserialized_target, llvm::Succeeded());

  EXPECT_EQ(target.id, deserialized_target->id);
  EXPECT_EQ(target.label, deserialized_target->label);
  EXPECT_EQ(target.line, deserialized_target->line);
  EXPECT_EQ(target.column, deserialized_target->column);
  EXPECT_EQ(target.endLine, deserialized_target->endLine);
  EXPECT_EQ(target.endColumn, deserialized_target->endColumn);
}

TEST(ProtocolTypesTest, ReadMemoryArguments) {
  ReadMemoryArguments args;
  args.count = 20;
  args.memoryReference = 43962;
  args.offset = 0;

  llvm::Expected<ReadMemoryArguments> expected =
      parse<ReadMemoryArguments>(R"({"memoryReference":"-4000", "count": 20})");
  ASSERT_THAT_EXPECTED(expected, llvm::Failed());
  expected = parse<ReadMemoryArguments>(
      R"({"memoryReference":"0xabba", "count": 20})");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());

  EXPECT_EQ(args.count, expected->count);
  EXPECT_EQ(args.memoryReference, expected->memoryReference);
  EXPECT_EQ(args.offset, expected->offset);
}

TEST(ProtocolTypesTest, ReadMemoryResponseBody) {
  ReadMemoryResponseBody response;
  response.address = 0xdeadbeef;
  const std::string data_str = "hello world!";
  std::transform(data_str.begin(), data_str.end(),
                 std::back_inserter(response.data),
                 [](char letter) { return std::byte(letter); });
  response.unreadableBytes = 1;

  Expected<Value> expected = json::parse(
      R"({ "address": "0xDEADBEEF", "data": "aGVsbG8gd29ybGQh", "unreadableBytes": 1})");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(pp(*expected), pp(response));
}

TEST(ProtocolTypesTest, Modules) {
  Module module;
  module.id = "AC805E8E-B6A4-CD92-4B05-5CFA7CE24AE8-8926C776";
  module.name = "libm.so.6";
  module.path = "/some/path/to/libm.so.6";
  module.isOptimized = true;
  module.isUserCode = true;
  module.version = "0.0.1";
  module.symbolStatus = "Symbol not found.";
  module.symbolFilePath = "/some/file/path/to/the/symbol/module";
  module.dateTimeStamp = "2020-12-09T16:09:53+00:00";
  module.addressRange = "0xcafeface";
  module.debugInfoSizeBytes = 1572864;

  Expected<json::Value> expected = json::parse(
      R"({
                  "id" : "AC805E8E-B6A4-CD92-4B05-5CFA7CE24AE8-8926C776",
                  "name": "libm.so.6",
                  "path": "/some/path/to/libm.so.6",
                  "isOptimized": true,
                  "isUserCode": true,
                  "version": "0.0.1",
                  "symbolStatus": "Symbol not found.",
                  "symbolFilePath": "/some/file/path/to/the/symbol/module",
                  "dateTimeStamp": "2020-12-09T16:09:53+00:00",
                  "addressRange": "0xcafeface",
                  "debugInfoSize": "1.5MB" })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(pp(*expected), pp(module));

  // Test without optional values.
  module.path.clear();
  module.isOptimized = false;
  module.isUserCode = false;
  module.version.clear();
  module.symbolStatus.clear();
  module.symbolFilePath.clear();
  module.dateTimeStamp.clear();
  module.addressRange.clear();
  module.debugInfoSizeBytes = 0;
  EXPECT_NE(pp(*expected), pp(module));

  Expected<json::Value> expected_no_opt = json::parse(
      R"({
                  "id" : "AC805E8E-B6A4-CD92-4B05-5CFA7CE24AE8-8926C776",
                  "name": "libm.so.6"})");
  ASSERT_THAT_EXPECTED(expected_no_opt, llvm::Succeeded());
  EXPECT_EQ(pp(*expected_no_opt), pp(module));
}

TEST(ProtocolTypesTest, ModulesArguments) {
  ModulesArguments args;

  llvm::Expected<ModulesArguments> expected = parse<ModulesArguments>(R"({})");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(args.startModule, expected->startModule);
  EXPECT_EQ(args.moduleCount, expected->moduleCount);

  // Non Default values.
  args.startModule = 1;
  args.moduleCount = 2;
  llvm::Expected<ModulesArguments> expected_no_default =
      parse<ModulesArguments>(R"({ "startModule": 1, "moduleCount": 2})");
  ASSERT_THAT_EXPECTED(expected_no_default, llvm::Succeeded());
  EXPECT_EQ(args.startModule, expected_no_default->startModule);
  EXPECT_EQ(args.moduleCount, expected_no_default->moduleCount);
}

TEST(ProtocolTypesTest, ModulesResponseBody) {
  ModulesResponseBody response;
  Module module1;
  module1.id = "first id";
  module1.name = "first name";

  Module module2;
  module2.id = "second id";
  module2.name = "second name";
  response.modules = {std::move(module1), std::move(module2)};
  response.totalModules = 2;

  Expected<json::Value> expected = json::parse(
      R"({
                  "modules": [
                    { "id": "first id", "name": "first name"},
                    { "id": "second id", "name": "second name"}
                  ],
                  "totalModules": 2 })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(pp(*expected), pp(response));
}

TEST(ProtocolTypesTest, VariablePresentationHint) {
  VariablePresentationHint hint;
  hint.kind = "kind";
  hint.attributes = {"a", "b", "c"};
  hint.visibility = "public";
  hint.lazy = true;

  const StringRef json = R"({
  "attributes": [
    "a",
    "b",
    "c"
  ],
  "kind": "kind",
  "lazy": true,
  "visibility": "public"
})";

  EXPECT_EQ(pp(Value(hint)), json);
  EXPECT_THAT_EXPECTED(json::parse(json), HasValue(Value(hint)));
}

TEST(ProtocolTypesTest, Variable) {
  Variable var;
  var.name = "var1";
  var.variablesReference = 42;
  var.value = "value";
  var.type = "type";

  VariablePresentationHint hint;
  hint.kind = "kind";
  var.presentationHint = std::move(hint);
  var.evaluateName = "my_name";
  var.namedVariables = 7;
  var.indexedVariables = 7;
  var.memoryReference = 291u;
  var.declarationLocationReference = 24;
  var.valueLocationReference = 100;

  const StringRef json = R"({
  "declarationLocationReference": 24,
  "evaluateName": "my_name",
  "indexedVariables": 7,
  "memoryReference": "0x123",
  "name": "var1",
  "namedVariables": 7,
  "presentationHint": {
    "kind": "kind"
  },
  "type": "type",
  "value": "value",
  "valueLocationReference": 100,
  "variablesReference": 42
})";

  EXPECT_EQ(pp(Value(var)), json);
  EXPECT_THAT_EXPECTED(json::parse(json), HasValue(Value(var)));
}

TEST(ProtocolTypesTest, VariablesArguments) {
  llvm::Expected<VariablesArguments> expected = parse<VariablesArguments>(R"({
    "variablesReference": 42,
    "filter": "indexed",
    "start": 10,
    "count": 5,
    "format": {
      "hex": true
    }
  })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(expected->variablesReference, 42u);
  EXPECT_EQ(expected->filter, VariablesArguments::eVariablesFilterIndexed);
  EXPECT_EQ(expected->start, 10u);
  EXPECT_EQ(expected->count, 5u);
  EXPECT_EQ(expected->format->hex, true);

  EXPECT_THAT_EXPECTED(
      parse<VariablesArguments>(R"({})"),
      FailedWithMessage("missing value at (root).variablesReference"));
  EXPECT_THAT_EXPECTED(
      parse<VariablesArguments>(
          R"({"variablesReference": 42, "filter": "my-filter"})"),
      FailedWithMessage(
          "unexpected value, expected 'named' or 'indexed' at (root).filter"));
}

TEST(ProtocolTypesTest, VariablesResponseBody) {
  Variable var1;
  var1.name = "var1";
  var1.variablesReference = 42;
  var1.value = "<var1-value>";

  Variable var2;
  var2.name = "var2";
  var2.variablesReference = 3;
  var2.value = "<var2-value>";

  VariablesResponseBody response{{var1, var2}};

  Expected<json::Value> expected = json::parse(R"({
      "variables": [
        {
          "name": "var1",
          "value": "<var1-value>",
          "variablesReference": 42
        },
        {
          "name": "var2",
          "value": "<var2-value>",
          "variablesReference": 3
        }
      ]
    })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(pp(*expected), pp(response));
}

TEST(ProtocolTypesTest, CompletionItem) {
  CompletionItem item;
  item.label = "label";
  item.text = "text";
  item.sortText = "sortText";
  item.detail = "detail";
  item.type = eCompletionItemTypeConstructor;
  item.start = 1;
  item.length = 3;
  item.selectionStart = 4;
  item.selectionLength = 8;

  const StringRef json = R"({
  "detail": "detail",
  "label": "label",
  "length": 3,
  "selectionLength": 8,
  "selectionStart": 4,
  "sortText": "sortText",
  "start": 1,
  "text": "text",
  "type": "constructor"
})";

  EXPECT_EQ(pp(Value(item)), json);
  EXPECT_THAT_EXPECTED(json::parse(json), HasValue(Value(item)));
}

TEST(ProtocolTypesTest, CompletionsArguments) {
  llvm::Expected<CompletionsArguments> expected =
      parse<CompletionsArguments>(R"({
    "column": 8,
    "frameId": 7,
    "line": 9,
    "text": "abc"
  })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(expected->frameId, 7u);
  EXPECT_EQ(expected->text, "abc");
  EXPECT_EQ(expected->column, 8);
  EXPECT_EQ(expected->line, 9);

  // Check required keys.
  EXPECT_THAT_EXPECTED(parse<CompletionsArguments>(R"({})"),
                       FailedWithMessage("missing value at (root).text"));
  EXPECT_THAT_EXPECTED(parse<CompletionsArguments>(R"({"text":"abc"})"),
                       FailedWithMessage("missing value at (root).column"));
}

TEST(ProtocolTypesTest, CompletionsResponseBody) {
  CompletionItem item;
  item.label = "label";
  item.text = "text";
  item.detail = "detail";
  CompletionsResponseBody response{{item}};

  Expected<json::Value> expected = json::parse(R"({
      "targets": [
        {
          "detail": "detail",
          "label": "label",
          "text": "text"
        }
      ]
    })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(pp(*expected), pp(response));
}
