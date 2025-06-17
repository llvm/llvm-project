//===-- ProtocolBaseTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolBase.h"
#include "TestingSupport/TestUtilities.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
// using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;
using lldb_private::pp;
using llvm::json::parse;
using llvm::json::Value;

TEST(ProtocolBaseTest, Request) {
  // Validate toJSON
  EXPECT_EQ(pp(Request{/*seq=*/42, /*command=*/"hello_world",
                       /*arguments=*/std::nullopt}),
            R"({
  "command": "hello_world",
  "seq": 42,
  "type": "request"
})");

  // Validate fromJSON
  EXPECT_THAT_EXPECTED(
      parse<Request>(R"({"type":"request","seq":42,"command":"hello_world"})"),
      HasValue(Value(Request{/*seq=*/42, /*command=*/"hello_world",
                             /*arguments*/ std::nullopt})));

  // Validate parsing errors
  EXPECT_THAT_EXPECTED(parse<Request>(R"({})"),
                       FailedWithMessage("missing value at (root).type"));
  EXPECT_THAT_EXPECTED(parse<Request>(R"({"type":"request"})"),
                       FailedWithMessage("missing value at (root).command"));
}

TEST(ProtocolBaseTest, Response) {
  // Validate toJSON
  EXPECT_EQ(pp(Response{/*request_seq=*/42, /*command=*/"hello_world",
                        /*success=*/true,
                        /*message=*/std::nullopt, /*body=*/std::nullopt}),
            R"({
  "command": "hello_world",
  "request_seq": 42,
  "seq": 0,
  "success": true,
  "type": "response"
})");

  // Validate fromJSON
  EXPECT_THAT_EXPECTED(
      parse<Response>(
          R"({"type":"response","seq":0,"request_seq":42,"command":"hello_world","success":true})"),
      HasValue(
          Value(Response{/*seq=*/42, /*command=*/"hello_world",
                         /*success=*/true,
                         /*message*/ std::nullopt, /*body=*/std::nullopt})));

  // Validate parsing errors
  EXPECT_THAT_EXPECTED(parse<Response>(R"({})"),
                       FailedWithMessage("missing value at (root).type"));
  EXPECT_THAT_EXPECTED(parse<Response>(R"({"type":"response"})"),
                       FailedWithMessage("missing value at (root).seq"));
}

TEST(ProtocolBaseTest, Event) {
  // Validate toJSON
  EXPECT_EQ(pp(Event{/*event=*/"hello_world", /*body=*/std::nullopt}),
            R"({
  "event": "hello_world",
  "seq": 0,
  "type": "event"
})");

  // Validate fromJSON
  EXPECT_THAT_EXPECTED(
      parse<Event>(R"({"type":"event","seq":0,"event":"hello_world"})"),
      HasValue(Value(Event{/*command=*/"hello_world", /*body=*/std::nullopt})));

  // Validate parsing errors
  EXPECT_THAT_EXPECTED(parse<Response>(R"({})"),
                       FailedWithMessage("missing value at (root).type"));
}

TEST(ProtocolBaseTest, ErrorMessage) {
  // Validate toJSON
  EXPECT_EQ(pp(ErrorMessage{/*id=*/42,
                            /*format=*/"Hello world!",
                            /*variables=*/{{"name", "value"}},
                            /*sendTelemetry=*/true,
                            /*showUser=*/true,
                            /*url=*/"http://example.com/error/42",
                            /*urlLabel*/ "ErrorLabel"}),
            R"({
  "format": "Hello world!",
  "id": 42,
  "sendTelemetry": true,
  "showUser": true,
  "url": "http://example.com/error/42",
  "urlLabel": "ErrorLabel",
  "variables": {
    "name": "value"
  }
})");

  // Validate fromJSON
  EXPECT_THAT_EXPECTED(
      parse<ErrorMessage>(
          R"({"format":"Hello world!","id":42,"sendTelemetry":true,"showUser":true,"url":"http://example.com/error/42","urlLabel":"ErrorLabel","variables":{"name": "value"}})"),
      HasValue(Value(ErrorMessage{/*id=*/42,
                                  /*format=*/"Hello world!",
                                  /*variables=*/{{"name", "value"}},
                                  /*sendTelemetry=*/true,
                                  /*showUser=*/true,
                                  /*url=*/"http://example.com/error/42",
                                  /*urlLabel*/ "ErrorLabel"})));

  // Validate parsing errors
  EXPECT_THAT_EXPECTED(parse<ErrorMessage>(R"({})"),
                       FailedWithMessage("missing value at (root).id"));
  EXPECT_THAT_EXPECTED(parse<ErrorMessage>(R"({"id":42})"),
                       FailedWithMessage("missing value at (root).format"));
}
