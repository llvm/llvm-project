//===-- MinidumpTypesTest.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Project includes
#include "Plugins/Process/minidump/MinidumpParser.h"
#include "Plugins/Process/minidump/MinidumpTypes.h"

// Other libraries and framework includes
#include "gtest/gtest.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Host/FileSpec.h"

#include "llvm/ADT/Optional.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

// C includes

// C++ includes
#include <memory>

extern const char *TestMainArgv0;

using namespace lldb_private;
using namespace minidump;

class MinidumpParserTest : public testing::Test {
public:
  void SetUp() override {
    llvm::StringRef dmp_folder = llvm::sys::path::parent_path(TestMainArgv0);
    inputs_folder = dmp_folder;
    llvm::sys::path::append(inputs_folder, "Inputs");
  }

  void SetUpData(const char *minidump_filename, size_t load_size = SIZE_MAX) {
    llvm::SmallString<128> filename = inputs_folder;
    llvm::sys::path::append(filename, minidump_filename);
    FileSpec minidump_file(filename.c_str(), false);
    lldb::DataBufferSP data_sp(
        minidump_file.MemoryMapFileContents(0, load_size));
    llvm::Optional<MinidumpParser> optional_parser =
        MinidumpParser::Create(data_sp);
    ASSERT_TRUE(optional_parser.hasValue());
    parser.reset(new MinidumpParser(optional_parser.getValue()));
    ASSERT_GT(parser->GetByteSize(), 0UL);
  }

  llvm::SmallString<128> inputs_folder;
  std::unique_ptr<MinidumpParser> parser;
};

TEST_F(MinidumpParserTest, GetThreads) {
  SetUpData("linux-x86_64.dmp");
  llvm::ArrayRef<MinidumpThread> thread_list;

  thread_list = parser->GetThreads();
  ASSERT_EQ(1UL, thread_list.size());

  const MinidumpThread thread = thread_list[0];
  ASSERT_EQ(16001UL, thread.thread_id);
}

TEST_F(MinidumpParserTest, GetThreadsTruncatedFile) {
  SetUpData("linux-x86_64.dmp", 200);
  llvm::ArrayRef<MinidumpThread> thread_list;

  thread_list = parser->GetThreads();
  ASSERT_EQ(0UL, thread_list.size());
}

TEST_F(MinidumpParserTest, GetArchitecture) {
  SetUpData("linux-x86_64.dmp");
  ASSERT_EQ(llvm::Triple::ArchType::x86_64,
            parser->GetArchitecture().GetMachine());
  ASSERT_EQ(llvm::Triple::OSType::Linux,
            parser->GetArchitecture().GetTriple().getOS());
}

TEST_F(MinidumpParserTest, GetMiscInfo) {
  SetUpData("linux-x86_64.dmp");
  const MinidumpMiscInfo *misc_info = parser->GetMiscInfo();
  ASSERT_EQ(nullptr, misc_info);
}

TEST_F(MinidumpParserTest, GetLinuxProcStatus) {
  SetUpData("linux-x86_64.dmp");
  llvm::Optional<LinuxProcStatus> proc_status = parser->GetLinuxProcStatus();
  ASSERT_TRUE(proc_status.hasValue());
  lldb::pid_t pid = proc_status->GetPid();
  ASSERT_EQ(16001UL, pid);
}

TEST_F(MinidumpParserTest, GetPid) {
  SetUpData("linux-x86_64.dmp");
  llvm::Optional<lldb::pid_t> pid = parser->GetPid();
  ASSERT_TRUE(pid.hasValue());
  ASSERT_EQ(16001UL, pid.getValue());
}

TEST_F(MinidumpParserTest, GetModuleList) {
  SetUpData("linux-x86_64.dmp");
  llvm::ArrayRef<MinidumpModule> modules = parser->GetModuleList();
  ASSERT_EQ(8UL, modules.size());
  std::string module_names[8] = {
      "/usr/local/google/home/dvlahovski/projects/test_breakpad/a.out",
      "/lib/x86_64-linux-gnu/libm-2.19.so",
      "/lib/x86_64-linux-gnu/libc-2.19.so",
      "/lib/x86_64-linux-gnu/libgcc_s.so.1",
      "/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.19",
      "/lib/x86_64-linux-gnu/libpthread-2.19.so",
      "/lib/x86_64-linux-gnu/ld-2.19.so",
      "linux-gate.so",
  };

  for (int i = 0; i < 8; ++i) {
    llvm::Optional<std::string> name =
        parser->GetMinidumpString(modules[i].module_name_rva);
    ASSERT_TRUE(name.hasValue());
    ASSERT_EQ(module_names[i], name.getValue());
  }
}

TEST_F(MinidumpParserTest, GetExceptionStream) {
  SetUpData("linux-x86_64.dmp");
  const MinidumpExceptionStream *exception_stream =
      parser->GetExceptionStream();
  ASSERT_NE(nullptr, exception_stream);
  ASSERT_EQ(11UL, exception_stream->exception_record.exception_code);
}

// Windows Minidump tests
// fizzbuzz_no_heap.dmp is copied from the WinMiniDump tests
TEST_F(MinidumpParserTest, GetArchitectureWindows) {
  SetUpData("fizzbuzz_no_heap.dmp");
  ASSERT_EQ(llvm::Triple::ArchType::x86,
            parser->GetArchitecture().GetMachine());
  ASSERT_EQ(llvm::Triple::OSType::Win32,
            parser->GetArchitecture().GetTriple().getOS());
}

TEST_F(MinidumpParserTest, GetLinuxProcStatusWindows) {
  SetUpData("fizzbuzz_no_heap.dmp");
  llvm::Optional<LinuxProcStatus> proc_status = parser->GetLinuxProcStatus();
  ASSERT_FALSE(proc_status.hasValue());
}

TEST_F(MinidumpParserTest, GetMiscInfoWindows) {
  SetUpData("fizzbuzz_no_heap.dmp");
  const MinidumpMiscInfo *misc_info = parser->GetMiscInfo();
  ASSERT_NE(nullptr, misc_info);
  llvm::Optional<lldb::pid_t> pid = misc_info->GetPid();
  ASSERT_TRUE(pid.hasValue());
  ASSERT_EQ(4440UL, pid.getValue());
}

TEST_F(MinidumpParserTest, GetPidWindows) {
  SetUpData("fizzbuzz_no_heap.dmp");
  llvm::Optional<lldb::pid_t> pid = parser->GetPid();
  ASSERT_TRUE(pid.hasValue());
  ASSERT_EQ(4440UL, pid.getValue());
}
