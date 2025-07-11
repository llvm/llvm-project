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
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap_tests;
using namespace lldb_dap::protocol;

class DAPTest : public TransportBase {};

TEST_F(DAPTest, SendProtocolMessages) {
  DAP dap{
      /*log=*/nullptr,
      /*default_repl_mode=*/ReplMode::Auto,
      /*pre_init_commands=*/{},
      /*transport=*/*to_dap,
  };
  dap.Send(Event{/*event=*/"my-event", /*body=*/std::nullopt});
  RunOnce<protocol::Message>([&](llvm::Expected<protocol::Message> message) {
    ASSERT_THAT_EXPECTED(
        message, HasValue(testing::VariantWith<Event>(testing::FieldsAre(
                     /*event=*/"my-event", /*body=*/std::nullopt))));
  });
}
