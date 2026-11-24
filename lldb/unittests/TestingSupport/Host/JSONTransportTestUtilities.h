//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_TESTINGSUPPORT_HOST_JSONTRANSPORTTESTUTILITIES_H
#define LLDB_UNITTESTS_TESTINGSUPPORT_HOST_JSONTRANSPORTTESTUTILITIES_H

#include "lldb/Host/FileSystem.h"
#include "lldb/Host/JSONTransport.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Utility/FileSpec.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <memory>
#include <utility>

template <typename Proto>
class TestTransport final
    : public lldb_private::transport::JSONTransport<Proto> {
public:
  using MessageHandler =
      typename lldb_private::transport::JSONTransport<Proto>::MessageHandler;

  static std::pair<std::unique_ptr<TestTransport<Proto>>,
                   std::unique_ptr<TestTransport<Proto>>>
  createPair(lldb_private::MainLoop &loop) {
    std::unique_ptr<TestTransport<Proto>> transports[2] = {
        std::make_unique<TestTransport<Proto>>(loop),
        std::make_unique<TestTransport<Proto>>(loop)};
    return std::make_pair(std::move(transports[0]), std::move(transports[1]));
  }

  /// Creates a cross-wired pair, where a message sent on one transport is
  /// delivered to the other's registered handler. This mirrors a real
  /// connected transport pair (bytes written to one end arrive at the other),
  /// unlike createPair() where a sent message is delivered back to the sender's
  /// own handler.
  static std::pair<std::unique_ptr<TestTransport<Proto>>,
                   std::unique_ptr<TestTransport<Proto>>>
  createConnectedPair(lldb_private::MainLoop &loop) {
    auto a = std::make_unique<TestTransport<Proto>>(loop);
    auto b = std::make_unique<TestTransport<Proto>>(loop);
    a->m_peer = b.get();
    b->m_peer = a.get();
    return std::make_pair(std::move(a), std::move(b));
  }

  explicit TestTransport(lldb_private::MainLoop &loop) : m_loop(loop) {}

  llvm::Error Send(const typename Proto::Evt &evt) override {
    EXPECT_TRUE(Target()) << "Send called before RegisterMessageHandler";
    m_loop.AddPendingCallback([this, evt](lldb_private::MainLoopBase &) {
      if (MessageHandler *target = Target())
        target->Received(evt);
    });
    return llvm::Error::success();
  }

  llvm::Error Send(const typename Proto::Req &req) override {
    EXPECT_TRUE(Target()) << "Send called before RegisterMessageHandler";
    m_loop.AddPendingCallback([this, req](lldb_private::MainLoopBase &) {
      if (MessageHandler *target = Target())
        target->Received(req);
    });
    return llvm::Error::success();
  }

  llvm::Error Send(const typename Proto::Resp &resp) override {
    EXPECT_TRUE(Target()) << "Send called before RegisterMessageHandler";
    m_loop.AddPendingCallback([this, resp](lldb_private::MainLoopBase &) {
      if (MessageHandler *target = Target())
        target->Received(resp);
    });
    return llvm::Error::success();
  }

  llvm::Error RegisterMessageHandler(MessageHandler &handler) override {
    if (m_register_should_fail)
      return llvm::createStringError("RegisterMessageHandler failed");
    if (!m_handler)
      m_handler = &handler;
    return llvm::Error::success();
  }

  /// Makes the next RegisterMessageHandler call fail, to exercise error paths.
  void SetRegisterMessageHandlerShouldFail(bool fail) {
    m_register_should_fail = fail;
  }

  /// Drives the registered handler's error callback, as the real transport
  /// would on a read or parse failure.
  void SimulateError(llvm::Error error) {
    EXPECT_TRUE(m_handler)
        << "SimulateError called before RegisterMessageHandler";
    m_handler->OnError(std::move(error));
  }

  /// Drives the registered handler's close callback, as the real transport
  /// would on EOF. Mirrors IOTransport::OnRead: the handler may destroy this
  /// transport, so members must not be accessed after this returns.
  void SimulateClosed() {
    EXPECT_TRUE(m_handler)
        << "SimulateClosed called before RegisterMessageHandler";
    m_handler->OnClosed();
  }

protected:
  void Log(llvm::StringRef message) override {};

private:
  /// The handler a sent message is delivered to: the peer's when cross-wired
  /// (see createConnectedPair), otherwise this transport's own handler.
  MessageHandler *Target() { return m_peer ? m_peer->m_handler : m_handler; }

  lldb_private::MainLoop &m_loop;
  MessageHandler *m_handler = nullptr;
  TestTransport<Proto> *m_peer = nullptr;
  bool m_register_should_fail = false;
};

template <typename Proto>
class MockMessageHandler final
    : public lldb_private::transport::JSONTransport<Proto>::MessageHandler {
public:
  MOCK_METHOD(void, Received, (const typename Proto::Req &), (override));
  MOCK_METHOD(void, Received, (const typename Proto::Resp &), (override));
  MOCK_METHOD(void, Received, (const typename Proto::Evt &), (override));
  MOCK_METHOD(void, OnError, (llvm::Error), (override));
  MOCK_METHOD(void, OnClosed, (), (override));
};

#endif
