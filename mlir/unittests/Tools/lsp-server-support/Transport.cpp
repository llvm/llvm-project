//===- Transport.cpp - LSP JSON transport unit tests ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/lsp-server-support/Transport.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "llvm/Support/FileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::lsp;
using namespace testing;

namespace {

TEST(TransportTest, SendReply) {
  std::string out;
  llvm::raw_string_ostream os(out);
  JSONTransport transport(nullptr, os);
  MessageHandler handler(transport);

  transport.reply(1989, nullptr);
  EXPECT_THAT(out, HasSubstr("\"id\":1989"));
  EXPECT_THAT(out, HasSubstr("\"result\":null"));
}

class TransportInputTest : public Test {
  llvm::SmallVector<char> inputPath;
  std::FILE *in = nullptr;
  std::string output = "";
  llvm::raw_string_ostream os;
  std::optional<JSONTransport> transport = std::nullopt;
  std::optional<MessageHandler> messageHandler = std::nullopt;

protected:
  TransportInputTest() : os(output) {}

  void SetUp() override {
    std::error_code ec =
        llvm::sys::fs::createTemporaryFile("lsp-unittest", "json", inputPath);
    ASSERT_FALSE(ec) << "Could not create temporary file: " << ec.message();

    in = std::fopen(inputPath.data(), "r");
    ASSERT_TRUE(in) << "Could not open temporary file: "
                    << std::strerror(errno);
    transport.emplace(in, os, JSONStreamStyle::Delimited);
    messageHandler.emplace(*transport);
  }

  void TearDown() override {
    EXPECT_EQ(std::fclose(in), 0)
        << "Could not close temporary file FD: " << std::strerror(errno);
    std::error_code ec =
        llvm::sys::fs::remove(inputPath, /*IgnoreNonExisting=*/false);
    EXPECT_FALSE(ec) << "Could not remove temporary file '" << inputPath.data()
                     << "': " << ec.message();
  }

  void writeInput(StringRef buffer) {
    std::error_code ec;
    llvm::raw_fd_ostream os(inputPath.data(), ec);
    ASSERT_FALSE(ec) << "Could not write to '" << inputPath.data()
                     << "': " << ec.message();
    os << buffer;
    os.close();
  }

  StringRef getOutput() const { return output; }
  MessageHandler &getMessageHandler() { return *messageHandler; }

  void runTransport() {
    bool gotEOF = false;
    llvm::Error err = llvm::handleErrors(
        transport->run(*messageHandler), [&](const llvm::ECError &ecErr) {
          gotEOF = ecErr.convertToErrorCode() == std::errc::io_error;
        });
    llvm::consumeError(std::move(err));
    EXPECT_TRUE(gotEOF);
  }
};

TEST_F(TransportInputTest, RequestWithInvalidParams) {
  struct Handler {
    void onMethod(const TextDocumentItem &params,
                  mlir::lsp::Callback<TextDocumentIdentifier> callback) {}
  } handler;
  getMessageHandler().method("invalid-params-request", &handler,
                             &Handler::onMethod);

  writeInput("{\"jsonrpc\":\"2.0\",\"id\":92,"
             "\"method\":\"invalid-params-request\",\"params\":{}}\n");
  runTransport();
  EXPECT_THAT(getOutput(), HasSubstr("error"));
  EXPECT_THAT(getOutput(), HasSubstr("missing value at (root).uri"));
}

TEST_F(TransportInputTest, NotificationWithInvalidParams) {
  // JSON parsing errors are only reported via error logging. As a result, this
  // test can't make any expectations -- but it prints the output anyway, by way
  // of demonstration.
  Logger::setLogLevel(Logger::Level::Error);

  struct Handler {
    void onNotification(const TextDocumentItem &params) {}
  } handler;
  getMessageHandler().notification("invalid-params-notification", &handler,
                                   &Handler::onNotification);

  writeInput("{\"jsonrpc\":\"2.0\",\"method\":\"invalid-params-notification\","
             "\"params\":{}}\n");
  runTransport();
}

TEST_F(TransportInputTest, MethodNotFound) {
  writeInput("{\"jsonrpc\":\"2.0\",\"id\":29,\"method\":\"ack\"}\n");
  runTransport();
  EXPECT_THAT(getOutput(), HasSubstr("\"id\":29"));
  EXPECT_THAT(getOutput(), HasSubstr("\"error\""));
  EXPECT_THAT(getOutput(), HasSubstr("\"message\":\"method not found: ack\""));
}

TEST_F(TransportInputTest, OutgoingNotification) {
  auto notifyFn = getMessageHandler().outgoingNotification<CompletionList>(
      "outgoing-notification");
  notifyFn(CompletionList{});
  EXPECT_THAT(getOutput(), HasSubstr("\"method\":\"outgoing-notification\""));
}
} // namespace
