//===- Transport.cpp - LSP JSON transport unit tests ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/ADT/ScopeExit.h"
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

TEST(TransportTest, MethodNotFound) {
  auto tempOr = llvm::sys::fs::TempFile::create("lsp-unittest-%%%%%%.json");
  ASSERT_TRUE((bool)tempOr);
  auto discardTemp =
      llvm::make_scope_exit([&]() { ASSERT_FALSE((bool)tempOr->discard()); });

  {
    std::error_code ec;
    llvm::raw_fd_ostream os(tempOr->TmpName, ec);
    ASSERT_FALSE(ec);
    os << "{\"jsonrpc\":\"2.0\",\"id\":29,\"method\":\"ack\"}\n";
    os.close();
  }

  std::string out;
  llvm::raw_string_ostream os(out);
  std::FILE *in = std::fopen(tempOr->TmpName.c_str(), "r");
  auto closeIn = llvm::make_scope_exit([&]() { std::fclose(in); });

  JSONTransport transport(in, os, JSONStreamStyle::Delimited);
  MessageHandler handler(transport);

  bool gotEOF = false;
  llvm::Error err = llvm::handleErrors(
      transport.run(handler), [&](const llvm::ECError &ecErr) {
        gotEOF = ecErr.convertToErrorCode() == std::errc::io_error;
      });
  llvm::consumeError(std::move(err));
  EXPECT_TRUE(gotEOF);
  EXPECT_THAT(out, HasSubstr("\"id\":29"));
  EXPECT_THAT(out, HasSubstr("\"error\""));
  EXPECT_THAT(out, HasSubstr("\"message\":\"method not found: ack\""));
}
} // namespace
