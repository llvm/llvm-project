//===-- ProtocolMCPTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/TestUtilities.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_protocol::mcp;

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
  Response response;
  response.id = 1;
  response.result = llvm::json::Object{{"key", "value"}};

  llvm::Expected<Response> deserialized_response = roundtripJSON(response);
  ASSERT_THAT_EXPECTED(deserialized_response, llvm::Succeeded());

  EXPECT_EQ(response.id, deserialized_response->id);
  EXPECT_EQ(response.result, deserialized_response->result);
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

TEST(ProtocolMCPTest, ToolDefinition) {
  ToolDefinition tool_definition;
  tool_definition.name = "ToolName";
  tool_definition.description = "Tool Description";
  tool_definition.inputSchema =
      llvm::json::Object{{"schemaKey", "schemaValue"}};

  llvm::Expected<ToolDefinition> deserialized_tool_definition =
      roundtripJSON(tool_definition);
  ASSERT_THAT_EXPECTED(deserialized_tool_definition, llvm::Succeeded());

  EXPECT_EQ(tool_definition.name, deserialized_tool_definition->name);
  EXPECT_EQ(tool_definition.description,
            deserialized_tool_definition->description);
  EXPECT_EQ(tool_definition.inputSchema,
            deserialized_tool_definition->inputSchema);
}

TEST(ProtocolMCPTest, MessageWithRequest) {
  Request request;
  request.id = 1;
  request.method = "test_method";
  request.params = llvm::json::Object{{"param", "value"}};

  Message message = request;

  llvm::Expected<Message> deserialized_message = roundtripJSON(message);
  ASSERT_THAT_EXPECTED(deserialized_message, llvm::Succeeded());

  ASSERT_TRUE(std::holds_alternative<Request>(*deserialized_message));
  const Request &deserialized_request =
      std::get<Request>(*deserialized_message);

  EXPECT_EQ(request, deserialized_request);
}

TEST(ProtocolMCPTest, MessageWithResponse) {
  Response response;
  response.id = 2;
  response.result = llvm::json::Object{{"result", "success"}};

  Message message = response;

  llvm::Expected<Message> deserialized_message = roundtripJSON(message);
  ASSERT_THAT_EXPECTED(deserialized_message, llvm::Succeeded());

  ASSERT_TRUE(std::holds_alternative<Response>(*deserialized_message));
  const Response &deserialized_response =
      std::get<Response>(*deserialized_message);

  EXPECT_EQ(response, deserialized_response);
}

TEST(ProtocolMCPTest, MessageWithNotification) {
  Notification notification;
  notification.method = "notification_method";
  notification.params = llvm::json::Object{{"notify", "data"}};

  Message message = notification;

  llvm::Expected<Message> deserialized_message = roundtripJSON(message);
  ASSERT_THAT_EXPECTED(deserialized_message, llvm::Succeeded());

  ASSERT_TRUE(std::holds_alternative<Notification>(*deserialized_message));
  const Notification &deserialized_notification =
      std::get<Notification>(*deserialized_message);

  EXPECT_EQ(notification, deserialized_notification);
}

TEST(ProtocolMCPTest, MessageWithErrorResponse) {
  Error error;
  error.code = -32603;
  error.message = "Internal error";

  Response error_response;
  error_response.id = 3;
  error_response.result = error;

  Message message = error_response;

  llvm::Expected<Message> deserialized_message = roundtripJSON(message);
  ASSERT_THAT_EXPECTED(deserialized_message, llvm::Succeeded());

  ASSERT_TRUE(std::holds_alternative<Response>(*deserialized_message));
  const Response &deserialized_error =
      std::get<Response>(*deserialized_message);

  EXPECT_EQ(error_response, deserialized_error);
}

TEST(ProtocolMCPTest, Resource) {
  Resource resource;
  resource.uri = "resource://example/test";
  resource.name = "Test Resource";
  resource.description = "A test resource for unit testing";
  resource.mimeType = "text/plain";

  llvm::Expected<Resource> deserialized_resource = roundtripJSON(resource);
  ASSERT_THAT_EXPECTED(deserialized_resource, llvm::Succeeded());

  EXPECT_EQ(resource.uri, deserialized_resource->uri);
  EXPECT_EQ(resource.name, deserialized_resource->name);
  EXPECT_EQ(resource.description, deserialized_resource->description);
  EXPECT_EQ(resource.mimeType, deserialized_resource->mimeType);
}

TEST(ProtocolMCPTest, ResourceWithoutOptionals) {
  Resource resource;
  resource.uri = "resource://example/minimal";
  resource.name = "Minimal Resource";

  llvm::Expected<Resource> deserialized_resource = roundtripJSON(resource);
  ASSERT_THAT_EXPECTED(deserialized_resource, llvm::Succeeded());

  EXPECT_EQ(resource.uri, deserialized_resource->uri);
  EXPECT_EQ(resource.name, deserialized_resource->name);
  EXPECT_TRUE(deserialized_resource->description.empty());
  EXPECT_TRUE(deserialized_resource->mimeType.empty());
}

TEST(ProtocolMCPTest, ResourceContents) {
  ResourceContents contents;
  contents.uri = "resource://example/content";
  contents.text = "This is the content of the resource";
  contents.mimeType = "text/plain";

  llvm::Expected<ResourceContents> deserialized_contents =
      roundtripJSON(contents);
  ASSERT_THAT_EXPECTED(deserialized_contents, llvm::Succeeded());

  EXPECT_EQ(contents.uri, deserialized_contents->uri);
  EXPECT_EQ(contents.text, deserialized_contents->text);
  EXPECT_EQ(contents.mimeType, deserialized_contents->mimeType);
}

TEST(ProtocolMCPTest, ResourceContentsWithoutMimeType) {
  ResourceContents contents;
  contents.uri = "resource://example/content-no-mime";
  contents.text = "Content without mime type specified";

  llvm::Expected<ResourceContents> deserialized_contents =
      roundtripJSON(contents);
  ASSERT_THAT_EXPECTED(deserialized_contents, llvm::Succeeded());

  EXPECT_EQ(contents.uri, deserialized_contents->uri);
  EXPECT_EQ(contents.text, deserialized_contents->text);
  EXPECT_TRUE(deserialized_contents->mimeType.empty());
}

TEST(ProtocolMCPTest, ResourceResult) {
  ResourceContents contents1;
  contents1.uri = "resource://example/content1";
  contents1.text = "First resource content";
  contents1.mimeType = "text/plain";

  ResourceContents contents2;
  contents2.uri = "resource://example/content2";
  contents2.text = "Second resource content";
  contents2.mimeType = "application/json";

  ResourceResult result;
  result.contents = {contents1, contents2};

  llvm::Expected<ResourceResult> deserialized_result = roundtripJSON(result);
  ASSERT_THAT_EXPECTED(deserialized_result, llvm::Succeeded());

  ASSERT_EQ(result.contents.size(), deserialized_result->contents.size());

  EXPECT_EQ(result.contents[0].uri, deserialized_result->contents[0].uri);
  EXPECT_EQ(result.contents[0].text, deserialized_result->contents[0].text);
  EXPECT_EQ(result.contents[0].mimeType,
            deserialized_result->contents[0].mimeType);

  EXPECT_EQ(result.contents[1].uri, deserialized_result->contents[1].uri);
  EXPECT_EQ(result.contents[1].text, deserialized_result->contents[1].text);
  EXPECT_EQ(result.contents[1].mimeType,
            deserialized_result->contents[1].mimeType);
}

TEST(ProtocolMCPTest, ResourceResultEmpty) {
  ResourceResult result;

  llvm::Expected<ResourceResult> deserialized_result = roundtripJSON(result);
  ASSERT_THAT_EXPECTED(deserialized_result, llvm::Succeeded());

  EXPECT_TRUE(deserialized_result->contents.empty());
}
