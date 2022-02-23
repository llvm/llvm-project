//===-- ThreadLauncherTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/ThreadLauncher.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <future>

using namespace lldb_private;

TEST(ThreadLauncherTest, LaunchThread) {
  std::promise<int> promise;
  std::future<int> future = promise.get_future();
  llvm::Expected<HostThread> thread =
      ThreadLauncher::LaunchThread("test", [&promise] {
        promise.set_value(47);
        return (lldb::thread_result_t)47;
      });
  ASSERT_THAT_EXPECTED(thread, llvm::Succeeded());
  EXPECT_EQ(future.get(), 47);
  lldb::thread_result_t result;
  thread->Join(&result);
  EXPECT_EQ(result, (lldb::thread_result_t)47);
}
