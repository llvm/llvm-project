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
#include "lldb/Host/Pipe.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/ProcessInfo.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

// From TestMain.cpp.
extern const char *TestMainArgv0;

using namespace lldb_private;
using namespace llvm;

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

#ifdef LLVM_ON_UNIX
TEST(Host, LaunchProcessDuplicatesHandle) {
  static constexpr llvm::StringLiteral test_msg("Hello subprocess!");

  SubsystemRAII<FileSystem> subsystems;

  if (test_arg) {
    Pipe pipe(LLDB_INVALID_PIPE, test_arg);
    size_t bytes_written;
    if (pipe.WriteWithTimeout(test_msg.data(), test_msg.size(),
                              std::chrono::microseconds(0), bytes_written)
            .Success() &&
        bytes_written == test_msg.size())
      exit(0);
    exit(1);
  }
  Pipe pipe;
  ASSERT_THAT_ERROR(pipe.CreateNew(/*child_process_inherit=*/false).takeError(),
                    llvm::Succeeded());
  ProcessLaunchInfo info;
  info.SetExecutableFile(FileSpec(TestMainArgv0),
                         /*add_exe_file_as_first_arg=*/true);
  info.GetArguments().AppendArgument(
      "--gtest_filter=Host.LaunchProcessDuplicatesHandle");
  info.GetArguments().AppendArgument(
      ("--test-arg=" + llvm::Twine::utohexstr(pipe.GetWritePipe())).str());
  info.AppendDuplicateFileAction(pipe.GetWritePipe(), pipe.GetWritePipe());
  info.SetMonitorProcessCallback(&ProcessLaunchInfo::NoOpMonitorCallback);
  ASSERT_THAT_ERROR(Host::LaunchProcess(info).takeError(), llvm::Succeeded());
  pipe.CloseWriteFileDescriptor();

  char msg[100];
  size_t bytes_read;
  ASSERT_THAT_ERROR(pipe.ReadWithTimeout(msg, sizeof(msg),
                                         std::chrono::seconds(10), bytes_read)
                        .takeError(),
                    llvm::Succeeded());
  ASSERT_EQ(llvm::StringRef(msg, bytes_read), test_msg);
}
#endif
