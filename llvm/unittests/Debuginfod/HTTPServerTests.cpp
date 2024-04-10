//===-- llvm/unittest/Support/HTTPServer.cpp - unit tests -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/Debuginfod/HTTPClient.h"
#include "llvm/Debuginfod/HTTPServer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

#ifdef LLVM_ENABLE_HTTPLIB

TEST(HTTPServer, IsAvailable) { EXPECT_TRUE(HTTPServer::isAvailable()); }

HTTPResponse Response = {200u, "text/plain", "hello, world\n"};
std::string UrlPathPattern = R"(/(.*))";
std::string InvalidUrlPathPattern = R"(/(.*)";

HTTPRequestHandler Handler = [](HTTPServerRequest &Request) {
  Request.setResponse(Response);
};

HTTPRequestHandler DelayHandler = [](HTTPServerRequest &Request) {
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  Request.setResponse(Response);
};

HTTPRequestHandler StreamingHandler = [](HTTPServerRequest &Request) {
  Request.setResponse({200, "text/plain", Response.Body.size(),
                       [=](size_t Offset, size_t Length) -> StringRef {
                         return Response.Body.substr(Offset, Length);
                       }});
};

TEST(HTTPServer, InvalidUrlPath) {
  // test that we can bind to any address
  HTTPServer Server;
  EXPECT_THAT_ERROR(Server.get(InvalidUrlPathPattern, Handler),
                    Failed<StringError>());
  EXPECT_THAT_EXPECTED(Server.bind(), Succeeded());
}

TEST(HTTPServer, bind) {
  // test that we can bind to any address
  HTTPServer Server;
  EXPECT_THAT_ERROR(Server.get(UrlPathPattern, Handler), Succeeded());
  EXPECT_THAT_EXPECTED(Server.bind(), Succeeded());
}

TEST(HTTPServer, ListenBeforeBind) {
  // test that we can bind to any address
  HTTPServer Server;
  EXPECT_THAT_ERROR(Server.get(UrlPathPattern, Handler), Succeeded());
  EXPECT_THAT_ERROR(Server.listen(), Failed<StringError>());
}

#ifdef LLVM_ENABLE_CURL
// Test the client and server against each other.

// Test fixture to initialize and teardown the HTTP client for each
// client-server test
class HTTPClientServerTest : public ::testing::Test {
protected:
  void SetUp() override { HTTPClient::initialize(); }
  void TearDown() override { HTTPClient::cleanup(); }
};

/// A simple handler which writes returned data to a string.
struct StringHTTPResponseHandler final : public HTTPResponseHandler {
  std::string ResponseBody = "";
  /// These callbacks store the body and status code in an HTTPResponseBuffer
  /// allocated based on Content-Length. The Content-Length header must be
  /// handled by handleHeaderLine before any calls to handleBodyChunk.
  Error handleBodyChunk(StringRef BodyChunk) override {
    ResponseBody = ResponseBody + BodyChunk.str();
    return Error::success();
  }
};

TEST_F(HTTPClientServerTest, Hello) {
  HTTPServer Server;
  EXPECT_THAT_ERROR(Server.get(UrlPathPattern, Handler), Succeeded());
  Expected<unsigned> PortOrErr = Server.bind();
  EXPECT_THAT_EXPECTED(PortOrErr, Succeeded());
  unsigned Port = *PortOrErr;
  DefaultThreadPool Pool(hardware_concurrency(1));
  Pool.async([&]() { EXPECT_THAT_ERROR(Server.listen(), Succeeded()); });
  std::string Url = "http://localhost:" + utostr(Port);
  HTTPRequest Request(Url);
  StringHTTPResponseHandler Handler;
  HTTPClient Client;
  EXPECT_THAT_ERROR(Client.perform(Request, Handler), Succeeded());
  EXPECT_EQ(Handler.ResponseBody, Response.Body);
  EXPECT_EQ(Client.responseCode(), Response.Code);
  Server.stop();
}

TEST_F(HTTPClientServerTest, LambdaHandlerHello) {
  HTTPServer Server;
  HTTPResponse LambdaResponse = {200u, "text/plain",
                                 "hello, world from a lambda\n"};
  EXPECT_THAT_ERROR(Server.get(UrlPathPattern,
                               [LambdaResponse](HTTPServerRequest &Request) {
                                 Request.setResponse(LambdaResponse);
                               }),
                    Succeeded());
  Expected<unsigned> PortOrErr = Server.bind();
  EXPECT_THAT_EXPECTED(PortOrErr, Succeeded());
  unsigned Port = *PortOrErr;
  DefaultThreadPool Pool(hardware_concurrency(1));
  Pool.async([&]() { EXPECT_THAT_ERROR(Server.listen(), Succeeded()); });
  std::string Url = "http://localhost:" + utostr(Port);
  HTTPRequest Request(Url);
  StringHTTPResponseHandler Handler;
  HTTPClient Client;
  EXPECT_THAT_ERROR(Client.perform(Request, Handler), Succeeded());
  EXPECT_EQ(Handler.ResponseBody, LambdaResponse.Body);
  EXPECT_EQ(Client.responseCode(), LambdaResponse.Code);
  Server.stop();
}

// Test the streaming response.
TEST_F(HTTPClientServerTest, StreamingHello) {
  HTTPServer Server;
  EXPECT_THAT_ERROR(Server.get(UrlPathPattern, StreamingHandler), Succeeded());
  Expected<unsigned> PortOrErr = Server.bind();
  EXPECT_THAT_EXPECTED(PortOrErr, Succeeded());
  unsigned Port = *PortOrErr;
  DefaultThreadPool Pool(hardware_concurrency(1));
  Pool.async([&]() { EXPECT_THAT_ERROR(Server.listen(), Succeeded()); });
  std::string Url = "http://localhost:" + utostr(Port);
  HTTPRequest Request(Url);
  StringHTTPResponseHandler Handler;
  HTTPClient Client;
  EXPECT_THAT_ERROR(Client.perform(Request, Handler), Succeeded());
  EXPECT_EQ(Handler.ResponseBody, Response.Body);
  EXPECT_EQ(Client.responseCode(), Response.Code);
  Server.stop();
}

// Writes a temporary file and streams it back using streamFile.
HTTPRequestHandler TempFileStreamingHandler = [](HTTPServerRequest Request) {
  int FD;
  SmallString<64> TempFilePath;
  sys::fs::createTemporaryFile("http-stream-file-test", "temp", FD,
                               TempFilePath);
  raw_fd_ostream OS(FD, true, /*unbuffered=*/true);
  OS << Response.Body;
  OS.close();
  streamFile(Request, TempFilePath);
};

// Test streaming back chunks of a file.
TEST_F(HTTPClientServerTest, StreamingFileResponse) {
  HTTPServer Server;
  EXPECT_THAT_ERROR(Server.get(UrlPathPattern, TempFileStreamingHandler),
                    Succeeded());
  Expected<unsigned> PortOrErr = Server.bind();
  EXPECT_THAT_EXPECTED(PortOrErr, Succeeded());
  unsigned Port = *PortOrErr;
  DefaultThreadPool Pool(hardware_concurrency(1));
  Pool.async([&]() { EXPECT_THAT_ERROR(Server.listen(), Succeeded()); });
  std::string Url = "http://localhost:" + utostr(Port);
  HTTPRequest Request(Url);
  StringHTTPResponseHandler Handler;
  HTTPClient Client;
  EXPECT_THAT_ERROR(Client.perform(Request, Handler), Succeeded());
  EXPECT_EQ(Handler.ResponseBody, Response.Body);
  EXPECT_EQ(Client.responseCode(), Response.Code);
  Server.stop();
}

// Deletes the temporary file before streaming it back, should give a 404 not
// found status code.
HTTPRequestHandler MissingTempFileStreamingHandler =
    [](HTTPServerRequest Request) {
      int FD;
      SmallString<64> TempFilePath;
      sys::fs::createTemporaryFile("http-stream-file-test", "temp", FD,
                                   TempFilePath);
      raw_fd_ostream OS(FD, true, /*unbuffered=*/true);
      OS << Response.Body;
      OS.close();
      // delete the file
      sys::fs::remove(TempFilePath);
      streamFile(Request, TempFilePath);
    };

// Streaming a missing file should give a 404.
TEST_F(HTTPClientServerTest, StreamingMissingFileResponse) {
  HTTPServer Server;
  EXPECT_THAT_ERROR(Server.get(UrlPathPattern, MissingTempFileStreamingHandler),
                    Succeeded());
  Expected<unsigned> PortOrErr = Server.bind();
  EXPECT_THAT_EXPECTED(PortOrErr, Succeeded());
  unsigned Port = *PortOrErr;
  DefaultThreadPool Pool(hardware_concurrency(1));
  Pool.async([&]() { EXPECT_THAT_ERROR(Server.listen(), Succeeded()); });
  std::string Url = "http://localhost:" + utostr(Port);
  HTTPRequest Request(Url);
  StringHTTPResponseHandler Handler;
  HTTPClient Client;
  EXPECT_THAT_ERROR(Client.perform(Request, Handler), Succeeded());
  EXPECT_EQ(Client.responseCode(), 404u);
  Server.stop();
}

TEST_F(HTTPClientServerTest, ClientTimeout) {
  HTTPServer Server;
  EXPECT_THAT_ERROR(Server.get(UrlPathPattern, DelayHandler), Succeeded());
  Expected<unsigned> PortOrErr = Server.bind();
  EXPECT_THAT_EXPECTED(PortOrErr, Succeeded());
  unsigned Port = *PortOrErr;
  DefaultThreadPool Pool(hardware_concurrency(1));
  Pool.async([&]() { EXPECT_THAT_ERROR(Server.listen(), Succeeded()); });
  std::string Url = "http://localhost:" + utostr(Port);
  HTTPClient Client;
  // Timeout below 50ms, request should fail
  Client.setTimeout(std::chrono::milliseconds(40));
  HTTPRequest Request(Url);
  StringHTTPResponseHandler Handler;
  EXPECT_THAT_ERROR(Client.perform(Request, Handler), Failed<StringError>());
  Server.stop();
}

// Check that Url paths are dispatched to the first matching handler and provide
// the correct path pattern match components.
TEST_F(HTTPClientServerTest, PathMatching) {
  HTTPServer Server;

  EXPECT_THAT_ERROR(
      Server.get(R"(/abc/(.*)/(.*))",
                 [&](HTTPServerRequest &Request) {
                   EXPECT_EQ(Request.UrlPath, "/abc/1/2");
                   ASSERT_THAT(Request.UrlPathMatches,
                               testing::ElementsAre("1", "2"));
                   Request.setResponse({200u, "text/plain", Request.UrlPath});
                 }),
      Succeeded());
  EXPECT_THAT_ERROR(Server.get(UrlPathPattern,
                               [&](HTTPServerRequest &Request) {
                                 llvm_unreachable(
                                     "Should not reach this handler");
                                 Handler(Request);
                               }),
                    Succeeded());

  Expected<unsigned> PortOrErr = Server.bind();
  EXPECT_THAT_EXPECTED(PortOrErr, Succeeded());
  unsigned Port = *PortOrErr;
  DefaultThreadPool Pool(hardware_concurrency(1));
  Pool.async([&]() { EXPECT_THAT_ERROR(Server.listen(), Succeeded()); });
  std::string Url = "http://localhost:" + utostr(Port) + "/abc/1/2";
  HTTPRequest Request(Url);
  StringHTTPResponseHandler Handler;
  HTTPClient Client;
  EXPECT_THAT_ERROR(Client.perform(Request, Handler), Succeeded());
  EXPECT_EQ(Handler.ResponseBody, "/abc/1/2");
  EXPECT_EQ(Client.responseCode(), 200u);
  Server.stop();
}

TEST_F(HTTPClientServerTest, FirstPathMatched) {
  HTTPServer Server;

  EXPECT_THAT_ERROR(
      Server.get(UrlPathPattern,
                 [&](HTTPServerRequest Request) { Handler(Request); }),
      Succeeded());

  EXPECT_THAT_ERROR(
      Server.get(R"(/abc/(.*)/(.*))",
                 [&](HTTPServerRequest Request) {
                   EXPECT_EQ(Request.UrlPathMatches.size(), 2u);
                   llvm_unreachable("Should not reach this handler");
                   Request.setResponse({200u, "text/plain", Request.UrlPath});
                 }),
      Succeeded());

  Expected<unsigned> PortOrErr = Server.bind();
  EXPECT_THAT_EXPECTED(PortOrErr, Succeeded());
  unsigned Port = *PortOrErr;
  DefaultThreadPool Pool(hardware_concurrency(1));
  Pool.async([&]() { EXPECT_THAT_ERROR(Server.listen(), Succeeded()); });
  std::string Url = "http://localhost:" + utostr(Port) + "/abc/1/2";
  HTTPRequest Request(Url);
  StringHTTPResponseHandler Handler;
  HTTPClient Client;
  EXPECT_THAT_ERROR(Client.perform(Request, Handler), Succeeded());
  EXPECT_EQ(Handler.ResponseBody, Response.Body);
  EXPECT_EQ(Client.responseCode(), Response.Code);
  Server.stop();
}

#endif

#else

TEST(HTTPServer, IsAvailable) { EXPECT_FALSE(HTTPServer::isAvailable()); }

#endif // LLVM_ENABLE_HTTPLIB
