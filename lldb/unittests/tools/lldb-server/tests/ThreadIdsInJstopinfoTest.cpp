//===-- ThreadsInJstopinfoTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestBase.h"
#include "TestClient.h"
#include "lldb/Utility/DataExtractor.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>

using namespace llgs_tests;
using namespace lldb_private;
using namespace llvm;
using namespace lldb;
using namespace testing;

#ifdef __NetBSD__
#define SKIP_ON_NETBSD(x) DISABLED_ ## x
#else
#define SKIP_ON_NETBSD(x) x
#endif

TEST_F(StandardStartupTest, SKIP_ON_NETBSD(TestStopReplyContainsThreadPcs)) {
  // This inferior spawns 4 threads, then forces a break.
  ASSERT_THAT_ERROR(
      Client->SetInferior({getInferiorPath("thread_inferior"), "4"}),
      Succeeded());

  ASSERT_THAT_ERROR(Client->ListThreadsInStopReply(), Succeeded());
  ASSERT_THAT_ERROR(Client->ContinueAll(), Succeeded());
  unsigned int pc_reg = Client->GetPcRegisterId();
  ASSERT_NE(pc_reg, UINT_MAX);

  auto jthreads_info = Client->GetJThreadsInfo();
  ASSERT_THAT_EXPECTED(jthreads_info, Succeeded());

  auto stop_reply = Client->GetLatestStopReplyAs<StopReplyStop>();
  ASSERT_THAT_EXPECTED(stop_reply, Succeeded());
  auto stop_reply_pcs = stop_reply->getThreadPcs();
  auto thread_infos = jthreads_info->GetThreadInfos();
  ASSERT_EQ(stop_reply_pcs.size(), thread_infos.size())
      << "Thread count mismatch.";

  for (auto stop_reply_pc : stop_reply_pcs) {
    unsigned long tid = stop_reply_pc.first;
    ASSERT_TRUE(thread_infos.find(tid) != thread_infos.end())
        << "Thread ID: " << tid << " not in JThreadsInfo.";
    EXPECT_THAT(thread_infos[tid].ReadRegister(pc_reg),
                Pointee(Eq(stop_reply_pc.second)));
  }
}
