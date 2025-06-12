//===-- ProtocolMCPTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Protocol/MCP/Protocol.h"
#include "TestingSupport/TestUtilities.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::mcp::protocol;

TEST(ProtocolMCPTest, Request) {
  Request request;
  request.id = 1;
  request.method = "foo";
  request.params = llvm::json::Object{{"key", "value"}};

  llvm::Expected<Request> deserialized_request = roundtripJSON(request);
  ASSERT_THAT_EXPECTED(deserialized_request, llvm::Succeeded());

  EXPECT_EQ(request.id, deserialized_request->id);
  EXPECT_EQ(request.method, deserialized_request->method);
  EXPECT_EQ(request.params, deserialized_request->params);
}

TEST(ProtocolMCPTest, Response) {
  Error error;
  error.code = 42;
  error.message = "error";
  error.data = "data";

  Response response;
  response.id = 1;
  response.result = llvm::json::Object{{"key", "value"}};
  response.error = error;

  llvm::Expected<Response> deserialized_response = roundtripJSON(response);
  ASSERT_THAT_EXPECTED(deserialized_response, llvm::Succeeded());

  EXPECT_EQ(response.id, deserialized_response->id);
  EXPECT_EQ(response.result, deserialized_response->result);
  EXPECT_EQ(response.error->code, deserialized_response->error->code);
  EXPECT_EQ(response.error->message, deserialized_response->error->message);
  EXPECT_EQ(response.error->data, deserialized_response->error->data);
}

TEST(ProtocolMCPTest, Notification) {
  Notification notification;
  notification.method = "notifyMethod";
  notification.params = llvm::json::Object{{"key", "value"}};

  llvm::Expected<Notification> deserialized_notification =
      roundtripJSON(notification);
  ASSERT_THAT_EXPECTED(deserialized_notification, llvm::Succeeded());

  EXPECT_EQ(notification.method, deserialized_notification->method);
  EXPECT_EQ(notification.params, deserialized_notification->params);
}

TEST(ProtocolMCPTest, ToolCapability) {
  ToolCapability tool_capability;
  tool_capability.listChanged = true;

  llvm::Expected<ToolCapability> deserialized_tool_capability =
      roundtripJSON(tool_capability);
  ASSERT_THAT_EXPECTED(deserialized_tool_capability, llvm::Succeeded());

  EXPECT_EQ(tool_capability.listChanged,
            deserialized_tool_capability->listChanged);
}

TEST(ProtocolMCPTest, Capabilities) {
  ToolCapability tool_capability;
  tool_capability.listChanged = true;

  Capabilities capabilities;
  capabilities.tools = tool_capability;

  llvm::Expected<Capabilities> deserialized_capabilities =
      roundtripJSON(capabilities);
  ASSERT_THAT_EXPECTED(deserialized_capabilities, llvm::Succeeded());

  EXPECT_EQ(capabilities.tools.listChanged,
            deserialized_capabilities->tools.listChanged);
}

TEST(ProtocolMCPTest, TextContent) {
  TextContent text_content;
  text_content.text = "Sample text";

  llvm::Expected<TextContent> deserialized_text_content =
      roundtripJSON(text_content);
  ASSERT_THAT_EXPECTED(deserialized_text_content, llvm::Succeeded());

  EXPECT_EQ(text_content.text, deserialized_text_content->text);
}

TEST(ProtocolMCPTest, TextResult) {
  TextContent text_content1;
  text_content1.text = "Text 1";

  TextContent text_content2;
  text_content2.text = "Text 2";

  TextResult text_result;
  text_result.content = {text_content1, text_content2};
  text_result.isError = true;

  llvm::Expected<TextResult> deserialized_text_result =
      roundtripJSON(text_result);
  ASSERT_THAT_EXPECTED(deserialized_text_result, llvm::Succeeded());

  EXPECT_EQ(text_result.isError, deserialized_text_result->isError);
  ASSERT_EQ(text_result.content.size(),
            deserialized_text_result->content.size());
  EXPECT_EQ(text_result.content[0].text,
            deserialized_text_result->content[0].text);
  EXPECT_EQ(text_result.content[1].text,
            deserialized_text_result->content[1].text);
}

TEST(ProtocolMCPTest, ToolAnnotations) {
  ToolAnnotations tool_annotations;
  tool_annotations.title = "Tool Title";
  tool_annotations.readOnlyHint = true;
  tool_annotations.destructiveHint = false;
  tool_annotations.idempotentHint = true;
  tool_annotations.openWorldHint = false;

  llvm::Expected<ToolAnnotations> deserialized_tool_annotations =
      roundtripJSON(tool_annotations);
  ASSERT_THAT_EXPECTED(deserialized_tool_annotations, llvm::Succeeded());

  EXPECT_EQ(tool_annotations.title, deserialized_tool_annotations->title);
  EXPECT_EQ(tool_annotations.readOnlyHint,
            deserialized_tool_annotations->readOnlyHint);
  EXPECT_EQ(tool_annotations.destructiveHint,
            deserialized_tool_annotations->destructiveHint);
  EXPECT_EQ(tool_annotations.idempotentHint,
            deserialized_tool_annotations->idempotentHint);
  EXPECT_EQ(tool_annotations.openWorldHint,
            deserialized_tool_annotations->openWorldHint);
}

TEST(ProtocolMCPTest, ToolDefinition) {
  ToolAnnotations tool_annotations;
  tool_annotations.title = "Tool Title";
  tool_annotations.readOnlyHint = true;

  ToolDefinition tool_definition;
  tool_definition.name = "ToolName";
  tool_definition.description = "Tool Description";
  tool_definition.inputSchema =
      llvm::json::Object{{"schemaKey", "schemaValue"}};
  tool_definition.annotations = tool_annotations;

  llvm::Expected<ToolDefinition> deserialized_tool_definition =
      roundtripJSON(tool_definition);
  ASSERT_THAT_EXPECTED(deserialized_tool_definition, llvm::Succeeded());

  EXPECT_EQ(tool_definition.name, deserialized_tool_definition->name);
  EXPECT_EQ(tool_definition.description,
            deserialized_tool_definition->description);
  EXPECT_EQ(tool_definition.inputSchema,
            deserialized_tool_definition->inputSchema);
  EXPECT_EQ(tool_definition.annotations->title,
            deserialized_tool_definition->annotations->title);
  EXPECT_EQ(tool_definition.annotations->readOnlyHint,
            deserialized_tool_definition->annotations->readOnlyHint);
}
