//===-- ThreadsInJstopinfoTest.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

TEST_F(StandardStartupTest, TestStopReplyContainsThreadPcs) {
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
