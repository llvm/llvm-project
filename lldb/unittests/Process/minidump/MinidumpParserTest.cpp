//===-- MinidumpTypesTest.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Project includes
#include "Plugins/Process/Utility/RegisterContextLinux_x86_64.h"
#include "Plugins/Process/minidump/MinidumpParser.h"
#include "Plugins/Process/minidump/MinidumpTypes.h"
#include "Plugins/Process/minidump/RegisterContextMinidump_x86_64.h"

// Other libraries and framework includes
#include "gtest/gtest.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Host/FileSpec.h"

#include "llvm/ADT/ArrayRef.h"
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
    ASSERT_GT(parser->GetData().size(), 0UL);
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

// Register stuff
// TODO probably split register stuff tests into different file?
#define REG_VAL(x) *(reinterpret_cast<uint64_t *>(x))

TEST_F(MinidumpParserTest, ConvertRegisterContext) {
  SetUpData("linux-x86_64.dmp");
  llvm::ArrayRef<MinidumpThread> thread_list = parser->GetThreads();
  const MinidumpThread thread = thread_list[0];
  llvm::ArrayRef<uint8_t> registers(parser->GetData().data() +
                                        thread.thread_context.rva,
                                    thread.thread_context.data_size);

  ArchSpec arch = parser->GetArchitecture();
  RegisterInfoInterface *reg_interface = new RegisterContextLinux_x86_64(arch);
  lldb::DataBufferSP buf =
      ConvertMinidumpContextToRegIface(registers, reg_interface);
  ASSERT_EQ(reg_interface->GetGPRSize(), buf->GetByteSize());

  const RegisterInfo *reg_info = reg_interface->GetRegisterInfo();

  std::map<uint64_t, uint64_t> reg_values;

  // clang-format off
  reg_values[lldb_rax_x86_64]    =  0x0000000000000000;
  reg_values[lldb_rbx_x86_64]    =  0x0000000000000000;
  reg_values[lldb_rcx_x86_64]    =  0x0000000000000010;
  reg_values[lldb_rdx_x86_64]    =  0x0000000000000000;
  reg_values[lldb_rdi_x86_64]    =  0x00007ffceb349cf0;
  reg_values[lldb_rsi_x86_64]    =  0x0000000000000000;
  reg_values[lldb_rbp_x86_64]    =  0x00007ffceb34a210;
  reg_values[lldb_rsp_x86_64]    =  0x00007ffceb34a210;
  reg_values[lldb_r8_x86_64]     =  0x00007fe9bc1aa9c0;
  reg_values[lldb_r9_x86_64]     =  0x0000000000000000;
  reg_values[lldb_r10_x86_64]    =  0x00007fe9bc3f16a0;
  reg_values[lldb_r11_x86_64]    =  0x0000000000000246;
  reg_values[lldb_r12_x86_64]    =  0x0000000000401c92;
  reg_values[lldb_r13_x86_64]    =  0x00007ffceb34a430;
  reg_values[lldb_r14_x86_64]    =  0x0000000000000000;
  reg_values[lldb_r15_x86_64]    =  0x0000000000000000;
  reg_values[lldb_rip_x86_64]    =  0x0000000000401dc6;
  reg_values[lldb_rflags_x86_64] =  0x0000000000010206;
  reg_values[lldb_cs_x86_64]     =  0x0000000000000033;
  reg_values[lldb_fs_x86_64]     =  0x0000000000000000;
  reg_values[lldb_gs_x86_64]     =  0x0000000000000000;
  reg_values[lldb_ss_x86_64]     =  0x0000000000000000;
  reg_values[lldb_ds_x86_64]     =  0x0000000000000000;
  reg_values[lldb_es_x86_64]     =  0x0000000000000000;
  // clang-format on

  for (uint32_t reg_index = 0; reg_index < reg_interface->GetRegisterCount();
       ++reg_index) {
    if (reg_values.find(reg_index) != reg_values.end()) {
      EXPECT_EQ(reg_values[reg_index],
                REG_VAL(buf->GetBytes() + reg_info[reg_index].byte_offset));
    }
  }
}
