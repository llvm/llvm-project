//===-- TestBase.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Protocol/ProtocolBase.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

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

  void Event(const lldb_dap::protocol::Event &e) override {
    m_loop.AddPendingCallback([this, e](lldb_private::MainLoopBase &) {
      this->m_handler.OnEvent(e);
    });
  }

  void Request(const lldb_dap::protocol::Request &r) override {
    m_loop.AddPendingCallback([this, r](lldb_private::MainLoopBase &) {
      this->m_handler.OnRequest(r);
    });
  }

  void Response(const lldb_dap::protocol::Response &r) override {
    m_loop.AddPendingCallback([this, r](lldb_private::MainLoopBase &) {
      this->m_handler.OnResponse(r);
    });
  }

  llvm::Error Run(lldb_private::MainLoop &loop, MessageHandler &) override {
    return loop.Run().takeError();
  }

  void Log(llvm::StringRef message) override {
    log_messages.emplace_back(message);
  }

  std::vector<std::string> log_messages;

private:
  lldb_private::MainLoop &m_loop;
  MessageHandler &m_handler;
};

/// A base class for tests that need transport configured for communicating DAP
/// messages.
class TransportBase : public testing::Test,
                      public TestTransport::MessageHandler {
protected:
  std::vector<TestTransport::Message> from_dap;
  lldb_private::MainLoop loop;
  std::unique_ptr<TestTransport> transport;

  void SetUp() override {
    transport = std::make_unique<TestTransport>(loop, *this);
  }

  void OnEvent(const lldb_dap::protocol::Event &e) override {
    from_dap.emplace_back(e);
  }
  void OnRequest(const lldb_dap::protocol::Request &r) override {
    from_dap.emplace_back(r);
  }
  void OnResponse(const lldb_dap::protocol::Response &r) override {
    from_dap.emplace_back(r);
  }
};

/// Matches an "output" event.
inline auto OutputMatcher(const llvm::StringRef output,
                          const llvm::StringRef category = "console") {
  return testing::VariantWith<lldb_dap::protocol::Event>(testing::FieldsAre(
      /*event=*/"output", /*body=*/testing::Optional<llvm::json::Value>(
          llvm::json::Object{{"category", category}, {"output", output}})));
}

/// A base class for tests that interact with a `lldb_dap::DAP` instance.
class DAPTestBase : public TransportBase {
protected:
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

  /// Closes the DAP output pipe and returns the remaining protocol messages in
  /// the buffer.
  // std::vector<lldb_dap::protocol::Message> DrainOutput();
  void RunOnce() {
    loop.AddPendingCallback(
        [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });
    ASSERT_THAT_ERROR(dap->Loop(), llvm::Succeeded());
  }
};

} // namespace lldb_dap_tests
