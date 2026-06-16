//===-- ProtocolMCPTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProtocolMCPTestUtilities.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_protocol::mcp;

// Flakey, see https://github.com/llvm/llvm-project/issues/152677.
#ifndef _WIN32

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

TEST(ProtocolMCPTest, ServerCapabilities) {
  ServerCapabilities capabilities;
  capabilities.supportsToolsList = true;

  llvm::Expected<ServerCapabilities> deserialized_capabilities =
      roundtripJSON(capabilities);
  ASSERT_THAT_EXPECTED(deserialized_capabilities, llvm::Succeeded());

  EXPECT_EQ(capabilities.supportsToolsList,
            deserialized_capabilities->supportsToolsList);
}

TEST(ProtocolMCPTest, TextContent) {
  TextContent text_content;
  text_content.text = "Sample text";

  llvm::Expected<TextContent> deserialized_text_content =
      roundtripJSON(text_content);
  ASSERT_THAT_EXPECTED(deserialized_text_content, llvm::Succeeded());

  EXPECT_EQ(text_content.text, deserialized_text_content->text);
}

TEST(ProtocolMCPTest, CallToolResult) {
  TextContent text_content1;
  text_content1.text = "Text 1";

  TextContent text_content2;
  text_content2.text = "Text 2";

  CallToolResult text_result;
  text_result.content = {text_content1, text_content2};
  text_result.isError = true;

  llvm::Expected<CallToolResult> deserialized_text_result =
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

TEST(ProtocolMCPTest, TextResourceContents) {
  TextResourceContents contents;
  contents.uri = "resource://example/content";
  contents.text = "This is the content of the resource";
  contents.mimeType = "text/plain";

  llvm::Expected<TextResourceContents> deserialized_contents =
      roundtripJSON(contents);
  ASSERT_THAT_EXPECTED(deserialized_contents, llvm::Succeeded());

  EXPECT_EQ(contents.uri, deserialized_contents->uri);
  EXPECT_EQ(contents.text, deserialized_contents->text);
  EXPECT_EQ(contents.mimeType, deserialized_contents->mimeType);
}

TEST(ProtocolMCPTest, TextResourceContentsWithoutMimeType) {
  TextResourceContents contents;
  contents.uri = "resource://example/content-no-mime";
  contents.text = "Content without mime type specified";

  llvm::Expected<TextResourceContents> deserialized_contents =
      roundtripJSON(contents);
  ASSERT_THAT_EXPECTED(deserialized_contents, llvm::Succeeded());

  EXPECT_EQ(contents.uri, deserialized_contents->uri);
  EXPECT_EQ(contents.text, deserialized_contents->text);
  EXPECT_TRUE(deserialized_contents->mimeType.empty());
}

TEST(ProtocolMCPTest, ReadResourceResult) {
  TextResourceContents contents1;
  contents1.uri = "resource://example/content1";
  contents1.text = "First resource content";
  contents1.mimeType = "text/plain";

  TextResourceContents contents2;
  contents2.uri = "resource://example/content2";
  contents2.text = "Second resource content";
  contents2.mimeType = "application/json";

  ReadResourceResult result;
  result.contents = {contents1, contents2};

  llvm::Expected<ReadResourceResult> deserialized_result =
      roundtripJSON(result);
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

TEST(ProtocolMCPTest, ReadResourceResultEmpty) {
  ReadResourceResult result;

  llvm::Expected<ReadResourceResult> deserialized_result =
      roundtripJSON(result);
  ASSERT_THAT_EXPECTED(deserialized_result, llvm::Succeeded());

  EXPECT_TRUE(deserialized_result->contents.empty());
}

TEST(ProtocolMCPTest, RequestWithStringId) {
  Request request;
  request.id = "request-1";
  request.method = "foo";

  llvm::Expected<Request> deserialized = roundtripJSON(request);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(request, *deserialized);
}

TEST(ProtocolMCPTest, RequestWithoutParams) {
  Request request;
  request.id = 7;
  request.method = "bar";

  llvm::Expected<Request> deserialized = roundtripJSON(request);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(request, *deserialized);
  EXPECT_FALSE(deserialized->params.has_value());
}

TEST(ProtocolMCPTest, RequestMissingId) {
  llvm::json::Value value =
      llvm::json::Object{{"jsonrpc", "2.0"}, {"method", "foo"}};
  Request request;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, request, root));
}

TEST(ProtocolMCPTest, RequestInvalidId) {
  llvm::json::Value value =
      llvm::json::Object{{"jsonrpc", "2.0"}, {"id", true}, {"method", "foo"}};
  Request request;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, request, root));
}

TEST(ProtocolMCPTest, ResponseWithStringId) {
  Response response;
  response.id = "resp-1";
  response.result = llvm::json::Value("ok");

  llvm::Expected<Response> deserialized = roundtripJSON(response);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(response, *deserialized);
}

TEST(ProtocolMCPTest, ResponseResultAndErrorMutuallyExclusive) {
  llvm::json::Value value = llvm::json::Object{
      {"jsonrpc", "2.0"},
      {"id", 1},
      {"result", 1},
      {"error", llvm::json::Object{{"code", 1}, {"message", "m"}}}};
  Response response;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, response, root));
}

TEST(ProtocolMCPTest, ResponseRequiresResultOrError) {
  llvm::json::Value value = llvm::json::Object{{"jsonrpc", "2.0"}, {"id", 1}};
  Response response;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, response, root));
}

TEST(ProtocolMCPTest, ResponseExpectsObject) {
  llvm::json::Value value(42);
  Response response;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, response, root));
}

TEST(ProtocolMCPTest, ResponseInvalidError) {
  llvm::json::Value value = llvm::json::Object{
      {"jsonrpc", "2.0"}, {"id", 1}, {"error", "not-an-object"}};
  Response response;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, response, root));
}

TEST(ProtocolMCPTest, ErrorWithData) {
  Error error;
  error.code = -32000;
  error.message = "boom";
  error.data = llvm::json::Object{{"detail", "stack"}};

  llvm::Expected<Error> deserialized = roundtripJSON(error);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(error, *deserialized);
  EXPECT_TRUE(deserialized->data.has_value());
}

TEST(ProtocolMCPTest, NotificationWithoutParams) {
  Notification notification;
  notification.method = "ping";

  llvm::Expected<Notification> deserialized = roundtripJSON(notification);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(notification, *deserialized);
  EXPECT_FALSE(deserialized->params.has_value());
}

TEST(ProtocolMCPTest, NotificationExpectsObject) {
  llvm::json::Value value(42);
  Notification notification;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, notification, root));
}

TEST(ProtocolMCPTest, NotificationMissingMethod) {
  llvm::json::Value value = llvm::json::Object{{"jsonrpc", "2.0"}};
  Notification notification;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, notification, root));
}

TEST(ProtocolMCPTest, ToolDefinitionMinimal) {
  ToolDefinition tool_definition;
  tool_definition.name = "tool";

  llvm::Expected<ToolDefinition> deserialized = roundtripJSON(tool_definition);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(tool_definition.name, deserialized->name);
  EXPECT_TRUE(deserialized->description.empty());
  EXPECT_FALSE(deserialized->inputSchema.has_value());
}

TEST(ProtocolMCPTest, ToolDefinitionMissingName) {
  llvm::json::Value value = llvm::json::Object{{"description", "d"}};
  ToolDefinition tool_definition;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, tool_definition, root));
}

TEST(ProtocolMCPTest, MessageExpectsObject) {
  llvm::json::Value value(42);
  Message message;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, message, root));
}

TEST(ProtocolMCPTest, MessageRequiresJSONRPC) {
  llvm::json::Value value = llvm::json::Object{{"id", 1}, {"method", "m"}};
  Message message;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, message, root));
}

TEST(ProtocolMCPTest, MessageUnsupportedJSONRPCVersion) {
  llvm::json::Value value =
      llvm::json::Object{{"jsonrpc", "1.0"}, {"id", 1}, {"method", "m"}};
  Message message;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, message, root));
}

TEST(ProtocolMCPTest, MessageUnrecognized) {
  llvm::json::Value value = llvm::json::Object{{"jsonrpc", "2.0"}, {"id", 1}};
  Message message;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, message, root));
}

TEST(ProtocolMCPTest, MessageInvalidNotification) {
  // No "id" routes to a Notification, but a missing "method" is invalid.
  llvm::json::Value value = llvm::json::Object{{"jsonrpc", "2.0"}};
  Message message;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, message, root));
}

TEST(ProtocolMCPTest, MessageInvalidRequest) {
  // Routed to a Request (it has a "method"), but the id is invalid.
  llvm::json::Value value =
      llvm::json::Object{{"jsonrpc", "2.0"}, {"id", true}, {"method", "m"}};
  Message message;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, message, root));
}

TEST(ProtocolMCPTest, MessageInvalidResponse) {
  // Routed to a Response (it has "result"/"error" but no "method"), but
  // 'result' and 'error' are mutually exclusive.
  llvm::json::Value value = llvm::json::Object{
      {"jsonrpc", "2.0"},
      {"id", 1},
      {"result", 1},
      {"error", llvm::json::Object{{"code", 1}, {"message", "m"}}}};
  Message message;
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, message, root));
}

TEST(ProtocolMCPTest, ImplementationWithTitle) {
  Implementation impl;
  impl.name = "lldb-mcp";
  impl.version = "0.1.0";
  impl.title = "LLDB MCP";

  llvm::Expected<Implementation> deserialized = roundtripJSON(impl);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(impl.name, deserialized->name);
  EXPECT_EQ(impl.version, deserialized->version);
  EXPECT_EQ(impl.title, deserialized->title);
}

TEST(ProtocolMCPTest, ImplementationWithoutTitle) {
  Implementation impl;
  impl.name = "lldb-mcp";
  impl.version = "0.1.0";

  llvm::Expected<Implementation> deserialized = roundtripJSON(impl);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(impl.name, deserialized->name);
  EXPECT_TRUE(deserialized->title.empty());
}

TEST(ProtocolMCPTest, ServerCapabilitiesAllFields) {
  ServerCapabilities caps;
  caps.supportsToolsList = true;
  caps.supportsResourcesList = true;
  caps.supportsResourcesSubscribe = true;
  caps.supportsCompletions = true;
  caps.supportsLogging = true;

  llvm::json::Value value = toJSON(caps);
  const llvm::json::Object *obj = value.getAsObject();
  ASSERT_NE(obj, nullptr);
  EXPECT_NE(obj->get("tools"), nullptr);
  EXPECT_NE(obj->get("completions"), nullptr);
  EXPECT_NE(obj->get("logging"), nullptr);

  const llvm::json::Object *resources = obj->getObject("resources");
  ASSERT_NE(resources, nullptr);
  EXPECT_EQ(resources->getBoolean("listChanged"), true);
  EXPECT_EQ(resources->getBoolean("subscribe"), true);
}

TEST(ProtocolMCPTest, ServerCapabilitiesSubscribeOnly) {
  ServerCapabilities caps;
  caps.supportsResourcesSubscribe = true;

  llvm::json::Value value = toJSON(caps);
  const llvm::json::Object *resources =
      value.getAsObject()->getObject("resources");
  ASSERT_NE(resources, nullptr);
  EXPECT_EQ(resources->getBoolean("subscribe"), true);
  EXPECT_EQ(resources->get("listChanged"), nullptr);
}

TEST(ProtocolMCPTest, ServerCapabilitiesFromJSONWithoutTools) {
  ServerCapabilities caps;
  llvm::json::Value value = llvm::json::Object{};
  llvm::json::Path::Root root;
  ASSERT_TRUE(fromJSON(value, caps, root));
  EXPECT_FALSE(caps.supportsToolsList);
}

TEST(ProtocolMCPTest, ServerCapabilitiesFromJSONExpectsObject) {
  ServerCapabilities caps;
  llvm::json::Value value(42);
  llvm::json::Path::Root root;
  EXPECT_FALSE(fromJSON(value, caps, root));
}

TEST(ProtocolMCPTest, ClientCapabilities) {
  ClientCapabilities caps;
  EXPECT_EQ(toJSON(caps), llvm::json::Value(llvm::json::Object{}));

  llvm::json::Value value = llvm::json::Object{};
  llvm::json::Path::Root root;
  EXPECT_TRUE(fromJSON(value, caps, root));
}

TEST(ProtocolMCPTest, InitializeParams) {
  InitializeParams params;
  params.protocolVersion = "2024-11-05";
  params.clientInfo.name = "client";
  params.clientInfo.version = "1.0";

  llvm::Expected<InitializeParams> deserialized = roundtripJSON(params);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(params.protocolVersion, deserialized->protocolVersion);
  EXPECT_EQ(params.clientInfo.name, deserialized->clientInfo.name);
  EXPECT_EQ(params.clientInfo.version, deserialized->clientInfo.version);
}

TEST(ProtocolMCPTest, InitializeResultWithInstructions) {
  InitializeResult result;
  result.protocolVersion = "2024-11-05";
  result.capabilities.supportsToolsList = true;
  result.serverInfo.name = "lldb-mcp";
  result.serverInfo.version = "0.1.0";
  result.instructions = "Use the tools wisely.";

  llvm::Expected<InitializeResult> deserialized = roundtripJSON(result);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(result.protocolVersion, deserialized->protocolVersion);
  EXPECT_EQ(result.instructions, deserialized->instructions);
  EXPECT_TRUE(deserialized->capabilities.supportsToolsList);
}

TEST(ProtocolMCPTest, InitializeResultWithoutInstructions) {
  InitializeResult result;
  result.protocolVersion = "2024-11-05";
  result.serverInfo.name = "lldb-mcp";
  result.serverInfo.version = "0.1.0";

  llvm::Expected<InitializeResult> deserialized = roundtripJSON(result);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_TRUE(deserialized->instructions.empty());
}

TEST(ProtocolMCPTest, ListToolsResult) {
  ToolDefinition tool_definition;
  tool_definition.name = "a";
  tool_definition.description = "d";

  ListToolsResult result;
  result.tools = {tool_definition};

  llvm::Expected<ListToolsResult> deserialized = roundtripJSON(result);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  ASSERT_EQ(deserialized->tools.size(), 1u);
  EXPECT_EQ(deserialized->tools[0].name, "a");
}

TEST(ProtocolMCPTest, CallToolResultStructuredContent) {
  CallToolResult result;
  result.content = {TextContent{"text"}};
  result.structuredContent = llvm::json::Object{{"k", "v"}};

  llvm::Expected<CallToolResult> deserialized = roundtripJSON(result);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  ASSERT_TRUE(deserialized->structuredContent.has_value());
  ASSERT_EQ(deserialized->content.size(), 1u);
  EXPECT_EQ(deserialized->content[0].text, "text");
}

TEST(ProtocolMCPTest, CallToolParamsWithArguments) {
  CallToolParams params;
  params.name = "tool";
  params.arguments = llvm::json::Object{{"a", 1}};

  llvm::Expected<CallToolParams> deserialized = roundtripJSON(params);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(deserialized->name, "tool");
  EXPECT_TRUE(deserialized->arguments.has_value());
}

TEST(ProtocolMCPTest, CallToolParamsWithoutArguments) {
  CallToolParams params;
  params.name = "tool";

  llvm::Expected<CallToolParams> deserialized = roundtripJSON(params);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(deserialized->name, "tool");
  EXPECT_FALSE(deserialized->arguments.has_value());
}

TEST(ProtocolMCPTest, ReadResourceParams) {
  ReadResourceParams params;
  params.uri = "lldb://debugger/0";

  llvm::Expected<ReadResourceParams> deserialized = roundtripJSON(params);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  EXPECT_EQ(deserialized->uri, params.uri);
}

TEST(ProtocolMCPTest, ListResourcesResult) {
  Resource resource;
  resource.uri = "lldb://x";
  resource.name = "x";

  ListResourcesResult result;
  result.resources = {resource};

  llvm::Expected<ListResourcesResult> deserialized = roundtripJSON(result);
  ASSERT_THAT_EXPECTED(deserialized, llvm::Succeeded());
  ASSERT_EQ(deserialized->resources.size(), 1u);
  EXPECT_EQ(deserialized->resources[0].uri, "lldb://x");
}

TEST(ProtocolMCPTest, Void) {
  EXPECT_EQ(toJSON(Void{}), llvm::json::Value(llvm::json::Object{}));

  Void value;
  llvm::json::Value json = llvm::json::Object{};
  llvm::json::Path::Root root;
  EXPECT_TRUE(fromJSON(json, value, root));
}

#endif
