//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_TESTINGSUPPORT_HOST_NATIVEPROCESSTESTUTILS_H
#define LLDB_UNITTESTS_TESTINGSUPPORT_HOST_NATIVEPROCESSTESTUTILS_H

#include "lldb/Host/JSONTransport.h"
#include "gmock/gmock.h"

template <typename Req, typename Resp, typename Evt>
class MockMessageHandler final
    : public lldb_private::Transport<Req, Resp, Evt>::MessageHandler {
public:
  MOCK_METHOD(void, Received, (const Evt &), (override));
  MOCK_METHOD(void, Received, (const Req &), (override));
  MOCK_METHOD(void, Received, (const Resp &), (override));
  MOCK_METHOD(void, OnError, (llvm::Error), (override));
  MOCK_METHOD(void, OnClosed, (), (override));
};

#endif
