//===-- ProtocolTypesTest.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolTypes.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

template <typename T> static llvm::Expected<T> roundtrip(const T &input) {
  llvm::json::Value value = toJSON(input);
  llvm::json::Path::Root root;
  T output;
  if (!fromJSON(value, output, root))
    return root.getError();
  return output;
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
      roundtrip(filter);
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
  source.presentationHint = ePresentationHintEmphasize;

  llvm::Expected<Source> deserialized_source = roundtrip(source);
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

  llvm::Expected<ColumnDescriptor> deserialized_column = roundtrip(column);
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

  llvm::Expected<BreakpointMode> deserialized_mode = roundtrip(mode);
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
      Source{"test.cpp", "/path/to/test.cpp", 123, ePresentationHintNormal};
  breakpoint.line = 10;
  breakpoint.column = 5;
  breakpoint.endLine = 15;
  breakpoint.endColumn = 10;
  breakpoint.instructionReference = "0x12345678";
  breakpoint.offset = 4;
  breakpoint.reason = BreakpointReason::eBreakpointReasonPending;

  llvm::Expected<Breakpoint> deserialized_breakpoint = roundtrip(breakpoint);
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
      roundtrip(source_breakpoint);
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
      roundtrip(function_breakpoint);
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
      roundtrip(data_breakpoint_info);
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
