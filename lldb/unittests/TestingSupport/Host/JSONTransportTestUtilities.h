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

template <typename Id, typename Req, typename Resp, typename Evt>
class TestTransport final
    : public lldb_private::JSONTransport<Id, Req, Resp, Evt> {
public:
  using MessageHandler =
      typename lldb_private::JSONTransport<Id, Req, Resp, Evt>::MessageHandler;

  static std::pair<std::unique_ptr<TestTransport<Id, Req, Resp, Evt>>,
                   std::unique_ptr<TestTransport<Id, Req, Resp, Evt>>>
  createPair() {
    std::unique_ptr<TestTransport<Id, Req, Resp, Evt>> transports[2] = {
        std::make_unique<TestTransport<Id, Req, Resp, Evt>>(),
        std::make_unique<TestTransport<Id, Req, Resp, Evt>>()};
    return std::make_pair(std::move(transports[0]), std::move(transports[1]));
  }

  explicit TestTransport() {
    llvm::Expected<lldb::FileUP> dummy_file =
        lldb_private::FileSystem::Instance().Open(
            lldb_private::FileSpec(lldb_private::FileSystem::DEV_NULL),
            lldb_private::File::eOpenOptionReadWrite);
    EXPECT_THAT_EXPECTED(dummy_file, llvm::Succeeded());
    m_dummy_file = std::move(*dummy_file);
  }

  llvm::Error Send(const Evt &evt) override {
    EXPECT_TRUE(m_loop && m_handler)
        << "Send called before RegisterMessageHandler";
    m_loop->AddPendingCallback([this, evt](lldb_private::MainLoopBase &) {
      m_handler->Received(evt);
    });
    return llvm::Error::success();
  }

  llvm::Error Send(const Req &req) override {
    EXPECT_TRUE(m_loop && m_handler)
        << "Send called before RegisterMessageHandler";
    m_loop->AddPendingCallback([this, req](lldb_private::MainLoopBase &) {
      m_handler->Received(req);
    });
    return llvm::Error::success();
  }

  llvm::Error Send(const Resp &resp) override {
    EXPECT_TRUE(m_loop && m_handler)
        << "Send called before RegisterMessageHandler";
    m_loop->AddPendingCallback([this, resp](lldb_private::MainLoopBase &) {
      m_handler->Received(resp);
    });
    return llvm::Error::success();
  }

  llvm::Expected<lldb_private::MainLoop::ReadHandleUP>
  RegisterMessageHandler(lldb_private::MainLoop &loop,
                         MessageHandler &handler) override {
    if (!m_loop)
      m_loop = &loop;
    if (!m_handler)
      m_handler = &handler;
    lldb_private::Status status;
    auto handle = loop.RegisterReadObject(
        m_dummy_file, [](lldb_private::MainLoopBase &) {}, status);
    if (status.Fail())
      return status.takeError();
    return handle;
  }

protected:
  void Log(llvm::StringRef message) override {};

private:
  lldb_private::MainLoop *m_loop = nullptr;
  MessageHandler *m_handler = nullptr;
  // Dummy file for registering with the MainLoop.
  lldb::FileSP m_dummy_file = nullptr;
};

template <typename Id, typename Req, typename Resp, typename Evt>
class MockMessageHandler final
    : public lldb_private::JSONTransport<Id, Req, Resp, Evt>::MessageHandler {
public:
  MOCK_METHOD(void, Received, (const Req &), (override));
  MOCK_METHOD(void, Received, (const Resp &), (override));
  MOCK_METHOD(void, Received, (const Evt &), (override));
  MOCK_METHOD(void, OnError, (llvm::Error), (override));
  MOCK_METHOD(void, OnClosed, (), (override));
};

#endif
