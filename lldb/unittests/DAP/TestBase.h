//===-- TestBase.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Protocol/ProtocolBase.h"
#include "TestingSupport/Host/JSONTransportTestUtilities.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace lldb_dap_tests {

class TestTransport final
    : public lldb_private::Transport<lldb_dap::protocol::Request,
                                     lldb_dap::protocol::Response,
                                     lldb_dap::protocol::Event> {
public:
  using Message = lldb_private::Transport<lldb_dap::protocol::Request,
                                          lldb_dap::protocol::Response,
                                          lldb_dap::protocol::Event>::Message;

  TestTransport(lldb_private::MainLoop &loop, MessageHandler &handler)
      : m_loop(loop), m_handler(handler) {}

  llvm::Error Send(const lldb_dap::protocol::Event &e) override {
    m_loop.AddPendingCallback([this, e](lldb_private::MainLoopBase &) {
      this->m_handler.Received(e);
    });
    return llvm::Error::success();
  }

  llvm::Error Send(const lldb_dap::protocol::Request &r) override {
    m_loop.AddPendingCallback([this, r](lldb_private::MainLoopBase &) {
      this->m_handler.Received(r);
    });
    return llvm::Error::success();
  }

  llvm::Error Send(const lldb_dap::protocol::Response &r) override {
    m_loop.AddPendingCallback([this, r](lldb_private::MainLoopBase &) {
      this->m_handler.Received(r);
    });
    return llvm::Error::success();
  }

  llvm::Expected<lldb_private::MainLoop::ReadHandleUP>
  RegisterMessageHandler(lldb_private::MainLoop &loop,
                         MessageHandler &handler) override;

  void Log(llvm::StringRef message) override {
    log_messages.emplace_back(message);
  }

  std::vector<std::string> log_messages;

private:
  lldb_private::MainLoop &m_loop;
  MessageHandler &m_handler;
  lldb::FileSP m_dummy_file;
};

/// A base class for tests that need transport configured for communicating DAP
/// messages.
class TransportBase : public testing::Test {
protected:
  lldb_private::SubsystemRAII<lldb_private::FileSystem, lldb_private::HostInfo>
      subsystems;
  lldb_private::MainLoop loop;
  std::unique_ptr<TestTransport> transport;
  MockMessageHandler<lldb_dap::protocol::Request, lldb_dap::protocol::Response,
                     lldb_dap::protocol::Event>
      client;

  void SetUp() override {
    transport = std::make_unique<TestTransport>(loop, client);
  }
};

/// A matcher for a DAP event.
template <typename M1, typename M2>
inline testing::Matcher<const lldb_dap::protocol::Event &>
IsEvent(const M1 &m1, const M2 &m2) {
  return testing::AllOf(testing::Field(&lldb_dap::protocol::Event::event, m1),
                        testing::Field(&lldb_dap::protocol::Event::body, m2));
}

/// Matches an "output" event.
inline auto Output(llvm::StringRef o, llvm::StringRef cat = "console") {
  return IsEvent("output",
                 testing::Optional(llvm::json::Value(
                     llvm::json::Object{{"category", cat}, {"output", o}})));
}

/// A base class for tests that interact with a `lldb_dap::DAP` instance.
class DAPTestBase : public TransportBase {
protected:
  std::unique_ptr<lldb_dap::Log> log;
  std::unique_ptr<lldb_dap::DAP> dap;
  std::optional<llvm::sys::fs::TempFile> core;
  std::optional<llvm::sys::fs::TempFile> binary;

  static constexpr llvm::StringLiteral k_linux_binary = "linux-x86_64.out.yaml";
  static constexpr llvm::StringLiteral k_linux_core = "linux-x86_64.core.yaml";

  static void SetUpTestSuite();
  static void TeatUpTestSuite();
  void SetUp() override;
  void TearDown() override;

  bool GetDebuggerSupportsTarget(llvm::StringRef platform);
  void CreateDebugger();
  void LoadCore();

  void RunOnce() {
    loop.AddPendingCallback(
        [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });
    ASSERT_THAT_ERROR(dap->Loop(), llvm::Succeeded());
  }
};

} // namespace lldb_dap_tests
