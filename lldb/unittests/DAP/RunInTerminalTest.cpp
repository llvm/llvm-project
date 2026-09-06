//===-- RunInTerminalTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RunInTerminal.h"
#include "gtest/gtest.h"
#include "llvm/Testing/Support/Error.h"
#include <thread>

using namespace lldb_dap;
using namespace llvm;

TEST(RunInTerminalTest, ErrorRoundTrip) {
  Expected<std::shared_ptr<FifoFile>> fifo = CreateRunInTerminalCommFile();
  ASSERT_THAT_EXPECTED(fifo, Succeeded());

  RunInTerminalLauncherCommChannel launcher((*fifo)->GetPath());
  (*fifo)->Connect();
  RunInTerminalDebugAdapterCommChannel adapter(*fifo);

  std::thread sender([&launcher]() { launcher.NotifyError("boom"); });
  EXPECT_EQ(adapter.GetLauncherError(), "boom");
  sender.join();
}
