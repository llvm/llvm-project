//===-- ProtocolRequestsTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "TestingSupport/TestUtilities.h"
#include "llvm/Testing/Support/Error.h"
#include <gtest/gtest.h>

using namespace llvm;
using namespace lldb_dap::protocol;
using lldb_private::PrettyPrint;
using llvm::json::parse;

TEST(ProtocolRequestsTest, ExceptionInfoArguments) {
  llvm::Expected<ExceptionInfoArguments> expected =
      parse<ExceptionInfoArguments>(R"({
        "threadId": 3434
        })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(expected->threadId, 3434U);

  // Check required keys;
  EXPECT_THAT_EXPECTED(parse<ExceptionInfoArguments>(R"({})"),
                       FailedWithMessage("missing value at (root).threadId"));

  EXPECT_THAT_EXPECTED(parse<ExceptionInfoArguments>(R"({"id": 10})"),
                       FailedWithMessage("missing value at (root).threadId"));
}

TEST(ProtocolRequestsTest, ExceptionInfoResponseBody) {
  ExceptionInfoResponseBody body;
  body.exceptionId = "signal";
  body.breakMode = eExceptionBreakModeAlways;

  // Check required keys.
  Expected<json::Value> expected = parse(
      R"({
    "exceptionId": "signal",
    "breakMode": "always"
    })");

  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected), PrettyPrint(body));

  // Check optional keys.
  body.description = "SIGNAL SIGWINCH";
  body.breakMode = eExceptionBreakModeNever;
  body.details = ExceptionDetails{};
  body.details->message = "some message";

  Expected<json::Value> expected_opt = parse(
      R"({
    "exceptionId": "signal",
    "description": "SIGNAL SIGWINCH",
    "breakMode": "never",
    "details": {
      "message": "some message"
    }
  })");

  ASSERT_THAT_EXPECTED(expected_opt, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected_opt), PrettyPrint(body));
}

TEST(ProtocolRequestsTest, EvaluateArguments) {
  llvm::Expected<EvaluateArguments> expected = parse<EvaluateArguments>(R"({
    "expression": "hello world",
    "context": "repl"
  })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(expected->expression, "hello world");
  EXPECT_EQ(expected->context, eEvaluateContextRepl);

  // Check required keys.
  EXPECT_THAT_EXPECTED(parse<EvaluateArguments>(R"({})"),
                       FailedWithMessage("missing value at (root).expression"));
}

TEST(ProtocolRequestsTest, EvaluateResponseBody) {
  EvaluateResponseBody body;
  body.result = "hello world";
  body.variablesReference = 7;

  // Check required keys.
  Expected<json::Value> expected = parse(R"({
    "result": "hello world",
    "variablesReference": 7
  })");

  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected), PrettyPrint(body));

  // Check optional keys.
  body.result = "'abc'";
  body.type = "string";
  body.variablesReference = 42;
  body.namedVariables = 1;
  body.indexedVariables = 2;
  body.memoryReference = "0x123";
  body.valueLocationReference = 22;

  Expected<json::Value> expected_opt = parse(R"({
    "result": "'abc'",
    "type": "string",
    "variablesReference": 42,
    "namedVariables": 1,
    "indexedVariables": 2,
    "memoryReference": "0x123",
    "valueLocationReference": 22
  })");

  ASSERT_THAT_EXPECTED(expected_opt, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected_opt), PrettyPrint(body));
}

TEST(ProtocolRequestsTest, InitializeRequestArguments) {
  llvm::Expected<InitializeRequestArguments> expected =
      parse<InitializeRequestArguments>(R"({"adapterID": "myid"})");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(expected->adapterID, "myid");

  // Check optional keys.
  expected = parse<InitializeRequestArguments>(R"({
    "adapterID": "myid",
    "clientID": "myclientid",
    "clientName": "lldb-dap-unit-tests",
    "locale": "en-US",
    "linesStartAt1": true,
    "columnsStartAt1": true,
    "pathFormat": "uri",
    "supportsVariableType": true,
    "supportsVariablePaging": true,
    "supportsRunInTerminalRequest": true,
    "supportsMemoryReferences": true,
    "supportsProgressReporting": true,
    "supportsInvalidatedEvent": true,
    "supportsMemoryEvent": true,
    "supportsArgsCanBeInterpretedByShell": true,
    "supportsStartDebuggingRequest": true,
    "supportsANSIStyling": true
  })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(expected->adapterID, "myid");
  EXPECT_EQ(expected->clientID, "myclientid");
  EXPECT_EQ(expected->clientName, "lldb-dap-unit-tests");
  EXPECT_EQ(expected->locale, "en-US");
  EXPECT_EQ(expected->linesStartAt1, true);
  EXPECT_EQ(expected->columnsStartAt1, true);
  EXPECT_EQ(expected->pathFormat, ePathFormatURI);
  EXPECT_EQ(expected->supportedFeatures.contains(eClientFeatureVariableType),
            true);
  EXPECT_EQ(
      expected->supportedFeatures.contains(eClientFeatureRunInTerminalRequest),
      true);
  EXPECT_EQ(
      expected->supportedFeatures.contains(eClientFeatureMemoryReferences),
      true);
  EXPECT_EQ(
      expected->supportedFeatures.contains(eClientFeatureProgressReporting),
      true);
  EXPECT_EQ(
      expected->supportedFeatures.contains(eClientFeatureInvalidatedEvent),
      true);
  EXPECT_EQ(expected->supportedFeatures.contains(eClientFeatureMemoryEvent),
            true);
  EXPECT_EQ(expected->supportedFeatures.contains(
                eClientFeatureArgsCanBeInterpretedByShell),
            true);
  EXPECT_EQ(
      expected->supportedFeatures.contains(eClientFeatureStartDebuggingRequest),
      true);
  EXPECT_EQ(expected->supportedFeatures.contains(eClientFeatureANSIStyling),
            true);

  // Check required keys.
  EXPECT_THAT_EXPECTED(parse<InitializeRequestArguments>(R"({})"),
                       FailedWithMessage("missing value at (root).adapterID"));
}

TEST(ProtocolRequestsTest, PauseRequestArguments) {
  llvm::Expected<PauseArguments> expected =
      parse<PauseArguments>(R"({"threadId": 123})");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(expected->threadId, 123U);

  // Check required keys.
  EXPECT_THAT_EXPECTED(parse<PauseArguments>(R"({})"),
                       FailedWithMessage("missing value at (root).threadId"));
}

TEST(ProtocolRequestsTest, LocationsArguments) {
  llvm::Expected<LocationsArguments> expected =
      parse<LocationsArguments>(R"({"locationReference": 123})");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(expected->locationReference, 123U);

  // Check required keys.
  EXPECT_THAT_EXPECTED(
      parse<LocationsArguments>(R"({})"),
      FailedWithMessage("missing value at (root).locationReference"));
}

TEST(ProtocolRequestsTest, LocationsResponseBody) {
  LocationsResponseBody body;
  body.source.sourceReference = 123;
  body.source.name = "test.cpp";
  body.line = 42;

  // Check required keys.
  Expected<json::Value> expected = parse(R"({
    "source": {
      "sourceReference": 123,
      "name": "test.cpp"
    },
    "line": 42
  })");

  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected), PrettyPrint(body));

  // Check optional keys.
  body.column = 2;
  body.endLine = 43;
  body.endColumn = 4;

  expected = parse(R"({
    "source": {
      "sourceReference": 123,
      "name": "test.cpp"
    },
    "line": 42,
    "column": 2,
    "endLine": 43,
    "endColumn": 4
  })");

  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected), PrettyPrint(body));
}

TEST(ProtocolRequestsTest, CompileUnitsArguments) {
  llvm::Expected<CompileUnitsArguments> expected =
      parse<CompileUnitsArguments>(R"({"moduleId": "42"})");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(expected->moduleId, "42");

  // Check required keys.
  EXPECT_THAT_EXPECTED(parse<CompileUnitsArguments>(R"({})"),
                       FailedWithMessage("missing value at (root).moduleId"));
}

TEST(ProtocolRequestsTest, CompileUnitsResponseBody) {
  CompileUnitsResponseBody body;
  body.compileUnits = {{"main.cpp"}, {"util.cpp"}};

  // Check required keys.
  Expected<json::Value> expected = parse(R"({
    "compileUnits": [
      {
        "compileUnitPath": "main.cpp"
      },
      {
        "compileUnitPath": "util.cpp"
      }
    ]
  })");

  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected), PrettyPrint(body));
}

TEST(ProtocolRequestsTest, TestGetTargetBreakpointsResponseBody) {
  Breakpoint breakpoint1;
  breakpoint1.id = 1;
  breakpoint1.verified = true;
  Breakpoint breakpoint2;
  breakpoint2.id = 2;
  breakpoint2.verified = false;
  breakpoint2.message = "Failed to set breakpoint";
  TestGetTargetBreakpointsResponseBody body;
  body.breakpoints = {breakpoint1, breakpoint2};

  // Check required keys.
  Expected<json::Value> expected = parse(R"({
    "breakpoints": [
      {
        "id": 1,
        "verified": true
      },
      {
        "id": 2,
        "verified": false,
        "message": "Failed to set breakpoint"
      }
    ]
  })");

  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected), PrettyPrint(body));
}

TEST(ProtocolRequestsTest, RestartArguments) {
  llvm::Expected<RestartArguments> expected = parse<RestartArguments>(R"({})");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_TRUE(std::holds_alternative<std::monostate>(expected->arguments));

  // Check missed keys.
  expected = parse<RestartArguments>(R"({
    "arguments": {}
  })");
  EXPECT_THAT_EXPECTED(expected,
                       FailedWithMessage("failed to parse arguments, expected "
                                         "`launch` or `attach` arguments"));

  // Check launch arguments.
  expected = parse<RestartArguments>(R"({
    "arguments": {
      "program": "main.exe",
      "cwd": "/home/root"
    }
  })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  const LaunchRequestArguments *launch_args =
      std::get_if<LaunchRequestArguments>(&expected->arguments);
  EXPECT_NE(launch_args, nullptr);
  EXPECT_EQ(launch_args->configuration.program, "main.exe");
  EXPECT_EQ(launch_args->cwd, "/home/root");

  // Check attach arguments.
  expected = parse<RestartArguments>(R"({
    "arguments": {
      "pid": 123
    }
  })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  const AttachRequestArguments *attach_args =
      std::get_if<AttachRequestArguments>(&expected->arguments);
  EXPECT_NE(attach_args, nullptr);
  EXPECT_EQ(attach_args->pid, 123U);
}

TEST(ProtocolRequestsTest, StackTraceArguments) {
  llvm::Expected<StackTraceArguments> expected = parse<StackTraceArguments>(R"({
    "threadId": 42,
    "startFrame": 1,
    "levels": 10,
    "format": {
      "parameters": true,
      "line": true
    }
  })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(expected->threadId, 42U);
  EXPECT_EQ(expected->startFrame, 1U);
  EXPECT_EQ(expected->levels, 10U);
  EXPECT_EQ(expected->format->parameters, true);
  EXPECT_EQ(expected->format->line, true);

  // Check required keys.
  EXPECT_THAT_EXPECTED(parse<StackTraceArguments>(R"({})"),
                       FailedWithMessage("missing value at (root).threadId"));
}

TEST(ProtocolRequestsTest, StackTraceResponseBody) {
  StackFrame frame1;
  frame1.id = 1;
  frame1.name = "main";
  frame1.source = Source{};
  frame1.source->name = "main.cpp";
  frame1.source->sourceReference = 123;
  frame1.line = 23;
  frame1.column = 1;
  StackFrame frame2;
  frame2.id = 2;
  frame2.name = "test";
  frame2.presentationHint = StackFrame::ePresentationHintLabel;

  StackTraceResponseBody body;
  body.stackFrames = {frame1, frame2};
  body.totalFrames = 2;

  // Check required keys.
  Expected<json::Value> expected = parse(R"({
    "stackFrames": [
      {
        "id": 1,
        "name": "main",
        "source": {
          "name": "main.cpp",
          "sourceReference": 123
        },
        "line": 23,
        "column": 1
      },
      {
        "id": 2,
        "name": "test",
        "line": 0,
        "column": 0,
        "presentationHint": "label"
      }
    ],
    "totalFrames": 2
  })");

  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected), PrettyPrint(body));
}
