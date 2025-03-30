//===-- HostTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/ProcessInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <future>

using namespace lldb_private;
using namespace llvm;

// From TestMain.cpp.
extern const char *TestMainArgv0;

static cl::opt<uint64_t> test_arg("test-arg");

TEST(Host, WaitStatusFormat) {
  EXPECT_EQ("W01", formatv("{0:g}", WaitStatus{WaitStatus::Exit, 1}).str());
  EXPECT_EQ("X02", formatv("{0:g}", WaitStatus{WaitStatus::Signal, 2}).str());
  EXPECT_EQ("S03", formatv("{0:g}", WaitStatus{WaitStatus::Stop, 3}).str());
  EXPECT_EQ("Exited with status 4",
            formatv("{0}", WaitStatus{WaitStatus::Exit, 4}).str());
}

TEST(Host, GetEnvironment) {
  putenv(const_cast<char *>("LLDB_TEST_ENVIRONMENT_VAR=Host::GetEnvironment"));
  ASSERT_EQ("Host::GetEnvironment",
            Host::GetEnvironment().lookup("LLDB_TEST_ENVIRONMENT_VAR"));
}

TEST(Host, ProcessInstanceInfoCumulativeUserTimeIsValid) {
  ProcessInstanceInfo info;
  info.SetCumulativeUserTime(ProcessInstanceInfo::timespec{0, 0});
  EXPECT_FALSE(info.CumulativeUserTimeIsValid());
  info.SetCumulativeUserTime(ProcessInstanceInfo::timespec{0, 1});
  EXPECT_TRUE(info.CumulativeUserTimeIsValid());
  info.SetCumulativeUserTime(ProcessInstanceInfo::timespec{1, 0});
  EXPECT_TRUE(info.CumulativeUserTimeIsValid());
}

TEST(Host, ProcessInstanceInfoCumulativeSystemTimeIsValid) {
  ProcessInstanceInfo info;
  info.SetCumulativeSystemTime(ProcessInstanceInfo::timespec{0, 0});
  EXPECT_FALSE(info.CumulativeSystemTimeIsValid());
  info.SetCumulativeSystemTime(ProcessInstanceInfo::timespec{0, 1});
  EXPECT_TRUE(info.CumulativeSystemTimeIsValid());
  info.SetCumulativeSystemTime(ProcessInstanceInfo::timespec{1, 0});
  EXPECT_TRUE(info.CumulativeSystemTimeIsValid());
}

TEST(Host, LaunchProcessSetsArgv0) {
  SubsystemRAII<FileSystem> subsystems;

  static constexpr StringLiteral TestArgv0 = "HelloArgv0";
  if (test_arg != 0) {
    // In subprocess
    if (TestMainArgv0 != TestArgv0) {
      errs() << formatv("Got '{0}' for argv[0]\n", TestMainArgv0);
      exit(1);
    }
    exit(0);
  }

  ProcessLaunchInfo info;
  info.SetExecutableFile(
      FileSpec(llvm::sys::fs::getMainExecutable(TestMainArgv0, &test_arg)),
      /*add_exe_file_as_first_arg=*/false);
  info.GetArguments().AppendArgument("HelloArgv0");
  info.GetArguments().AppendArgument(
      "--gtest_filter=Host.LaunchProcessSetsArgv0");
  info.GetArguments().AppendArgument("--test-arg=47");
  std::promise<int> exit_status;
  info.SetMonitorProcessCallback([&](lldb::pid_t pid, int signal, int status) {
    exit_status.set_value(status);
  });
  ASSERT_THAT_ERROR(Host::LaunchProcess(info).takeError(), Succeeded());
  ASSERT_THAT(exit_status.get_future().get(), 0);
}
