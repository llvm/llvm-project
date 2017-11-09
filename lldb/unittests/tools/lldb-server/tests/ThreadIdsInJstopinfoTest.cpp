//===-- ThreadsInJstopinfoTest.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestClient.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <string>

using namespace llgs_tests;

class ThreadsInJstopinfoTest : public ::testing::Test {
protected:
  virtual void SetUp() { TestClient::Initialize(); }
};

TEST_F(ThreadsInJstopinfoTest, TestStopReplyContainsThreadPcsLlgs) {
  std::vector<std::string> inferior_args;
  // This inferior spawns N threads, then forces a break.
  inferior_args.push_back(THREAD_INFERIOR);
  inferior_args.push_back("4");

  auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();

  TestClient client(test_info->name(), test_info->test_case_name());
  ASSERT_THAT_ERROR(client.StartDebugger(), llvm::Succeeded());
  ASSERT_THAT_ERROR(client.SetInferior(inferior_args), llvm::Succeeded());
  ASSERT_THAT_ERROR(client.ListThreadsInStopReply(), llvm::Succeeded());
  ASSERT_THAT_ERROR(client.ContinueAll(), llvm::Succeeded());
  unsigned int pc_reg = client.GetPcRegisterId();
  ASSERT_NE(pc_reg, UINT_MAX);

  auto jthreads_info = client.GetJThreadsInfo();
  ASSERT_TRUE(jthreads_info);

  auto stop_reply = client.GetLatestStopReply();
  auto stop_reply_pcs = stop_reply.GetThreadPcs();
  auto thread_infos = jthreads_info->GetThreadInfos();
  ASSERT_EQ(stop_reply_pcs.size(), thread_infos.size())
      << "Thread count mismatch.";

  for (auto stop_reply_pc : stop_reply_pcs) {
    unsigned long tid = stop_reply_pc.first;
    ASSERT_TRUE(thread_infos.find(tid) != thread_infos.end())
        << "Thread ID: " << tid << " not in JThreadsInfo.";
    auto pc_value = thread_infos[tid].ReadRegisterAsUint64(pc_reg);
    ASSERT_THAT_EXPECTED(pc_value, llvm::Succeeded());
    ASSERT_EQ(stop_reply_pcs[tid], *pc_value)
        << "Mismatched PC for thread: " << tid;
  }

  ASSERT_THAT_ERROR(client.StopDebugger(), llvm::Succeeded());
}
