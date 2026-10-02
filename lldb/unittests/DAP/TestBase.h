//===-- TestBase.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "DAPLog.h"
#include "Handler/RequestHandler.h"
#include "Handler/ResponseHandler.h"
#include "Protocol/ProtocolBase.h"
#include "TestingSupport/Host/JSONTransportTestUtilities.h"
#include "TestingSupport/SubsystemRAII.h"
#include "Transport.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <optional>

/// Helpers for gtest printing.
namespace lldb_dap::protocol {

inline void PrintTo(const Request &req, std::ostream *os) {
  *os << llvm::formatv("{0}", toJSON(req)).str();
}

inline void PrintTo(const Response &resp, std::ostream *os) {
  *os << llvm::formatv("{0}", toJSON(resp)).str();
}

inline void PrintTo(const Event &evt, std::ostream *os) {
  *os << llvm::formatv("{0}", toJSON(evt)).str();
}

inline void PrintTo(const Message &message, std::ostream *os) {
  return std::visit([os](auto &&message) { return PrintTo(message, os); },
                    message);
}

} // namespace lldb_dap::protocol

namespace lldb_dap_tests {

using TestDAPTransport = TestTransport<lldb_dap::ProtocolDescriptor>;

/// A base class for tests that need transport configured for communicating DAP
/// messages.
class TransportBase : public testing::Test {
protected:
  lldb_private::SubsystemRAII<lldb_private::FileSystem, lldb_private::HostInfo>
      subsystems;
  lldb_private::MainLoop loop;

  std::unique_ptr<lldb_dap::Log> log;
  lldb_dap::Log::Mutex log_mutex;

  std::unique_ptr<TestDAPTransport> to_client;
  MockMessageHandler<lldb_dap::ProtocolDescriptor> client;

  std::unique_ptr<TestDAPTransport> to_server;
  std::unique_ptr<lldb_dap::DAP> dap;

  void SetUp() override;

  void Run();
};

/// A matcher for a DAP event.
template <typename EventMatcher, typename BodyMatcher>
inline testing::Matcher<const lldb_dap::protocol::Event &>
IsEvent(const EventMatcher &event_matcher, const BodyMatcher &body_matcher) {
  return testing::AllOf(
      testing::Field(&lldb_dap::protocol::Event::event, event_matcher),
      testing::Field(&lldb_dap::protocol::Event::body, body_matcher));
}

template <typename EventMatcher>
inline testing::Matcher<const lldb_dap::protocol::Event &>
IsEvent(const EventMatcher &event_matcher) {
  return testing::AllOf(
      testing::Field(&lldb_dap::protocol::Event::event, event_matcher),
      testing::Field(&lldb_dap::protocol::Event::body, std::nullopt));
}

/// Matches an "output" event.
inline auto Output(llvm::StringRef o, llvm::StringRef cat = "console") {
  return IsEvent("output",
                 testing::Optional(llvm::json::Value(
                     llvm::json::Object{{"category", cat}, {"output", o}})));
}

/// Matches a response to the request with sequence number `seq`.
MATCHER_P(HasSeq, seq, "request_seq is " + testing::PrintToString(seq)) {
  if (seq <= 0)
    return false;
  return arg.request_seq == (lldb_dap::protocol::Id)seq;
}

/// Matches a response to `command`.
MATCHER_P(HasCommand, command,
          "command is " + testing::PrintToString(command)) {
  return arg.command == command;
}

/// Matches a successful response that also matches all of `fields`.
template <typename... Fields>
inline testing::Matcher<const lldb_dap::protocol::Response &>
SuccessResponse(Fields &&...fields) {
  return testing::AllOf(
      testing::Field("success", &lldb_dap::protocol::Response::success, true),
      std::forward<Fields>(fields)...);
}

/// Matches a 'cancelled' response that also matches all of `fields`.
template <typename... Fields>
inline testing::Matcher<const lldb_dap::protocol::Response &>
CancelledResponse(Fields &&...fields) {
  return testing::AllOf(
      testing::Field("success", &lldb_dap::protocol::Response::success, false),
      testing::Field(
          "message", &lldb_dap::protocol::Response::message,
          testing::Optional(
              testing::VariantWith<lldb_dap::protocol::ResponseMessage>(
                  lldb_dap::protocol::eResponseMessageCancelled))),
      std::forward<Fields>(fields)...);
}

/// A base class for tests that interact with a `lldb_dap::DAP` instance.
class DAPTestBase : public TransportBase {
protected:
  std::optional<llvm::sys::fs::TempFile> core;
  std::optional<llvm::sys::fs::TempFile> binary;

  static constexpr llvm::StringLiteral k_linux_binary = "linux-x86_64.out.yaml";
  static constexpr llvm::StringLiteral k_linux_core = "linux-x86_64.core.yaml";

  static void SetUpTestSuite();
  static void TearDownTestSuite();
  void SetUp() override;
  void TearDown() override;

  bool GetDebuggerSupportsTarget(llvm::StringRef platform);
  void CreateDebugger();
  void LoadCore();
};

} // namespace lldb_dap_tests
