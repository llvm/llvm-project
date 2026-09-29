//===-- DAPTest.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Protocol/ProtocolBase.h"
#include "TestBase.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

using namespace lldb_dap;
using namespace lldb_dap_tests;
using namespace lldb_dap::protocol;
using namespace testing;

class DAPTest : public TransportBase {};

TEST_F(DAPTest, SendProtocolMessages) {
  dap->Send(Event{/*event=*/"my-event", /*body=*/std::nullopt});
  EXPECT_CALL(client, Received(IsEvent("my-event")));
  Run();
}
