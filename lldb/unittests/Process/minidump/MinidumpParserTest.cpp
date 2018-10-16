//===-- MinidumpTypesTest.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Project includes
#include "Plugins/Process/Utility/RegisterContextLinux_i386.h"
#include "Plugins/Process/Utility/RegisterContextLinux_x86_64.h"
#include "Plugins/Process/minidump/MinidumpParser.h"
#include "Plugins/Process/minidump/MinidumpTypes.h"
#include "Plugins/Process/minidump/RegisterContextMinidump_x86_32.h"
#include "Plugins/Process/minidump/RegisterContextMinidump_x86_64.h"

#include "TestingSupport/TestUtilities.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBufferLLVM.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/FileSpec.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

// C includes

// C++ includes
#include <memory>

using namespace lldb_private;
using namespace minidump;

class MinidumpParserTest : public testing::Test {
public:
  void SetUpData(const char *minidump_filename) {
    std::string filename = GetInputFilePath(minidump_filename);
    auto BufferPtr = DataBufferLLVM::CreateSliceFromPath(filename, -1, 0);
    ASSERT_NE(BufferPtr, nullptr);
    llvm::Optional<MinidumpParser> optional_parser =
        MinidumpParser::Create(BufferPtr);
    ASSERT_TRUE(optional_parser.hasValue());
    parser.reset(new MinidumpParser(optional_parser.getValue()));
    ASSERT_GT(parser->GetData().size(), 0UL);
    auto result = parser->Initialize();
    ASSERT_TRUE(result.Success()) << result.AsCString();
  }

  void InvalidMinidump(const char *minidump_filename, uint64_t load_size) {
    std::string filename = GetInputFilePath(minidump_filename);
    auto BufferPtr =
        DataBufferLLVM::CreateSliceFromPath(filename, load_size, 0);
    ASSERT_NE(BufferPtr, nullptr);

    llvm::Optional<MinidumpParser> optional_parser =
        MinidumpParser::Create(BufferPtr);
    ASSERT_TRUE(optional_parser.hasValue());
    parser.reset(new MinidumpParser(optional_parser.getValue()));
    ASSERT_GT(parser->GetData().size(), 0UL);
    auto result = parser->Initialize();
    ASSERT_TRUE(result.Fail());
  }

  std::unique_ptr<MinidumpParser> parser;
};

TEST_F(MinidumpParserTest, GetThreadsAndGetThreadContext) {
  SetUpData("linux-x86_64.dmp");
  llvm::ArrayRef<MinidumpThread> thread_list;

  thread_list = parser->GetThreads();
  ASSERT_EQ(1UL, thread_list.size());

  const MinidumpThread thread = thread_list[0];

  EXPECT_EQ(16001UL, thread.thread_id);

  llvm::ArrayRef<uint8_t> context = parser->GetThreadContext(thread);
  EXPECT_EQ(1232UL, context.size());
}

TEST_F(MinidumpParserTest, GetThreadListNotPadded) {
  // Verify that we can load a thread list that doesn't have 4 bytes of padding
  // after the thread count.
  SetUpData("thread-list-not-padded.dmp");
  llvm::ArrayRef<MinidumpThread> thread_list;
  
  thread_list = parser->GetThreads();
  ASSERT_EQ(2UL, thread_list.size());
  EXPECT_EQ(0x11223344UL, thread_list[0].thread_id);
  EXPECT_EQ(0x55667788UL, thread_list[1].thread_id);
}

TEST_F(MinidumpParserTest, GetThreadListPadded) {
  // Verify that we can load a thread list that has 4 bytes of padding
  // after the thread count as found in breakpad minidump files.
  SetUpData("thread-list-padded.dmp");
  auto thread_list = parser->GetThreads();
  ASSERT_EQ(2UL, thread_list.size());
  EXPECT_EQ(0x11223344UL, thread_list[0].thread_id);
  EXPECT_EQ(0x55667788UL, thread_list[1].thread_id);
}

TEST_F(MinidumpParserTest, GetModuleListNotPadded) {
  // Verify that we can load a module list that doesn't have 4 bytes of padding
  // after the module count.
  SetUpData("module-list-not-padded.dmp");
  auto module_list = parser->GetModuleList();
  ASSERT_EQ(2UL, module_list.size());
  EXPECT_EQ(0x1000UL, module_list[0].base_of_image);
  EXPECT_EQ(0x2000UL, module_list[0].size_of_image);
  EXPECT_EQ(0x5000UL, module_list[1].base_of_image);
  EXPECT_EQ(0x3000UL, module_list[1].size_of_image);
}

TEST_F(MinidumpParserTest, GetModuleListPadded) {
  // Verify that we can load a module list that has 4 bytes of padding
  // after the module count as found in breakpad minidump files.
  SetUpData("module-list-padded.dmp");
  auto module_list = parser->GetModuleList();
  ASSERT_EQ(2UL, module_list.size());
  EXPECT_EQ(0x1000UL, module_list[0].base_of_image);
  EXPECT_EQ(0x2000UL, module_list[0].size_of_image);
  EXPECT_EQ(0x5000UL, module_list[1].base_of_image);
  EXPECT_EQ(0x3000UL, module_list[1].size_of_image);
}

TEST_F(MinidumpParserTest, GetMemoryListNotPadded) {
  // Verify that we can load a memory list that doesn't have 4 bytes of padding
  // after the memory range count.
  SetUpData("memory-list-not-padded.dmp");
  auto mem = parser->FindMemoryRange(0x8000);
  ASSERT_TRUE(mem.hasValue());
  EXPECT_EQ((lldb::addr_t)0x8000, mem->start);
  mem = parser->FindMemoryRange(0x8010);
  ASSERT_TRUE(mem.hasValue());
  EXPECT_EQ((lldb::addr_t)0x8010, mem->start);
}

TEST_F(MinidumpParserTest, GetMemoryListPadded) {
  // Verify that we can load a memory list that has 4 bytes of padding
  // after the memory range count as found in breakpad minidump files.
  SetUpData("memory-list-padded.dmp");
  auto mem = parser->FindMemoryRange(0x8000);
  ASSERT_TRUE(mem.hasValue());
  EXPECT_EQ((lldb::addr_t)0x8000, mem->start);
  mem = parser->FindMemoryRange(0x8010);
  ASSERT_TRUE(mem.hasValue());
  EXPECT_EQ((lldb::addr_t)0x8010, mem->start);
}

TEST_F(MinidumpParserTest, TruncatedMinidumps) {
  InvalidMinidump("linux-x86_64.dmp", 32);
  InvalidMinidump("linux-x86_64.dmp", 100);
  InvalidMinidump("linux-x86_64.dmp", 20 * 1024);
}

TEST_F(MinidumpParserTest, IllFormedMinidumps) {
  InvalidMinidump("bad_duplicate_streams.dmp", -1);
  InvalidMinidump("bad_overlapping_streams.dmp", -1);
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
    EXPECT_EQ(module_names[i], name.getValue());
  }
}

TEST_F(MinidumpParserTest, GetFilteredModuleList) {
  SetUpData("linux-x86_64_not_crashed.dmp");
  llvm::ArrayRef<MinidumpModule> modules = parser->GetModuleList();
  std::vector<const MinidumpModule *> filtered_modules =
      parser->GetFilteredModuleList();
  EXPECT_EQ(10UL, modules.size());
  EXPECT_EQ(9UL, filtered_modules.size());
  // EXPECT_GT(modules.size(), filtered_modules.size());
  bool found = false;
  for (size_t i = 0; i < filtered_modules.size(); ++i) {
    llvm::Optional<std::string> name =
        parser->GetMinidumpString(filtered_modules[i]->module_name_rva);
    ASSERT_TRUE(name.hasValue());
    if (name.getValue() == "/tmp/test/linux-x86_64_not_crashed") {
      ASSERT_FALSE(found) << "There should be only one module with this name "
                             "in the filtered module list";
      found = true;
      ASSERT_EQ(0x400000UL, filtered_modules[i]->base_of_image);
    }
  }
}

TEST_F(MinidumpParserTest, GetExceptionStream) {
  SetUpData("linux-x86_64.dmp");
  const MinidumpExceptionStream *exception_stream =
      parser->GetExceptionStream();
  ASSERT_NE(nullptr, exception_stream);
  ASSERT_EQ(11UL, exception_stream->exception_record.exception_code);
}

void check_mem_range_exists(std::unique_ptr<MinidumpParser> &parser,
                            const uint64_t range_start,
                            const uint64_t range_size) {
  llvm::Optional<minidump::Range> range = parser->FindMemoryRange(range_start);
  ASSERT_TRUE(range.hasValue()) << "There is no range containing this address";
  EXPECT_EQ(range_start, range->start);
  EXPECT_EQ(range_start + range_size, range->start + range->range_ref.size());
}

TEST_F(MinidumpParserTest, FindMemoryRange) {
  SetUpData("linux-x86_64.dmp");
  // There are two memory ranges in the file (size is in bytes, decimal):
  // 1) 0x401d46 256
  // 2) 0x7ffceb34a000 12288
  EXPECT_FALSE(parser->FindMemoryRange(0x00).hasValue());
  EXPECT_FALSE(parser->FindMemoryRange(0x2a).hasValue());

  check_mem_range_exists(parser, 0x401d46, 256);
  EXPECT_FALSE(parser->FindMemoryRange(0x401d46 + 256).hasValue());

  check_mem_range_exists(parser, 0x7ffceb34a000, 12288);
  EXPECT_FALSE(parser->FindMemoryRange(0x7ffceb34a000 + 12288).hasValue());
}

TEST_F(MinidumpParserTest, GetMemory) {
  SetUpData("linux-x86_64.dmp");

  EXPECT_EQ(128UL, parser->GetMemory(0x401d46, 128).size());
  EXPECT_EQ(256UL, parser->GetMemory(0x401d46, 512).size());

  EXPECT_EQ(12288UL, parser->GetMemory(0x7ffceb34a000, 12288).size());
  EXPECT_EQ(1024UL, parser->GetMemory(0x7ffceb34a000, 1024).size());

  EXPECT_TRUE(parser->GetMemory(0x500000, 512).empty());
}

TEST_F(MinidumpParserTest, FindMemoryRangeWithFullMemoryMinidump) {
  SetUpData("fizzbuzz_wow64.dmp");

  // There are a lot of ranges in the file, just testing with some of them
  EXPECT_FALSE(parser->FindMemoryRange(0x00).hasValue());
  EXPECT_FALSE(parser->FindMemoryRange(0x2a).hasValue());
  check_mem_range_exists(parser, 0x10000, 65536); // first range
  check_mem_range_exists(parser, 0x40000, 4096);
  EXPECT_FALSE(parser->FindMemoryRange(0x40000 + 4096).hasValue());
  check_mem_range_exists(parser, 0x77c12000, 8192);
  check_mem_range_exists(parser, 0x7ffe0000, 4096); // last range
  EXPECT_FALSE(parser->FindMemoryRange(0x7ffe0000 + 4096).hasValue());
}

void check_region_info(std::unique_ptr<MinidumpParser> &parser,
                       const uint64_t addr, MemoryRegionInfo::OptionalBool read,
                       MemoryRegionInfo::OptionalBool write,
                       MemoryRegionInfo::OptionalBool exec) {
  auto range_info = parser->GetMemoryRegionInfo(addr);
  ASSERT_TRUE(range_info.hasValue());
  EXPECT_EQ(read, range_info->GetReadable());
  EXPECT_EQ(write, range_info->GetWritable());
  EXPECT_EQ(exec, range_info->GetExecutable());
}

TEST_F(MinidumpParserTest, GetMemoryRegionInfo) {
  SetUpData("fizzbuzz_wow64.dmp");

  const auto yes = MemoryRegionInfo::eYes;
  const auto no = MemoryRegionInfo::eNo;

  check_region_info(parser, 0x00000, no, no, no);
  check_region_info(parser, 0x10000, yes, yes, no);
  check_region_info(parser, 0x20000, yes, yes, no);
  check_region_info(parser, 0x30000, yes, yes, no);
  check_region_info(parser, 0x31000, no, no, no);
  check_region_info(parser, 0x40000, yes, no, no);
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

// wow64
TEST_F(MinidumpParserTest, GetPidWow64) {
  SetUpData("fizzbuzz_wow64.dmp");
  llvm::Optional<lldb::pid_t> pid = parser->GetPid();
  ASSERT_TRUE(pid.hasValue());
  ASSERT_EQ(7836UL, pid.getValue());
}

TEST_F(MinidumpParserTest, GetModuleListWow64) {
  SetUpData("fizzbuzz_wow64.dmp");
  llvm::ArrayRef<MinidumpModule> modules = parser->GetModuleList();
  ASSERT_EQ(16UL, modules.size());
  std::string module_names[16] = {
      R"(D:\src\llvm\llvm\tools\lldb\packages\Python\lldbsuite\test\functionalities\postmortem\wow64_minidump\fizzbuzz.exe)",
      R"(C:\Windows\System32\ntdll.dll)",
      R"(C:\Windows\System32\wow64.dll)",
      R"(C:\Windows\System32\wow64win.dll)",
      R"(C:\Windows\System32\wow64cpu.dll)",
      R"(D:\src\llvm\llvm\tools\lldb\packages\Python\lldbsuite\test\functionalities\postmortem\wow64_minidump\fizzbuzz.exe)",
      R"(C:\Windows\SysWOW64\ntdll.dll)",
      R"(C:\Windows\SysWOW64\kernel32.dll)",
      R"(C:\Windows\SysWOW64\KERNELBASE.dll)",
      R"(C:\Windows\SysWOW64\advapi32.dll)",
      R"(C:\Windows\SysWOW64\msvcrt.dll)",
      R"(C:\Windows\SysWOW64\sechost.dll)",
      R"(C:\Windows\SysWOW64\rpcrt4.dll)",
      R"(C:\Windows\SysWOW64\sspicli.dll)",
      R"(C:\Windows\SysWOW64\CRYPTBASE.dll)",
      R"(C:\Windows\System32\api-ms-win-core-synch-l1-2-0.DLL)",
  };

  for (int i = 0; i < 16; ++i) {
    llvm::Optional<std::string> name =
        parser->GetMinidumpString(modules[i].module_name_rva);
    ASSERT_TRUE(name.hasValue());
    EXPECT_EQ(module_names[i], name.getValue());
  }
}

// Register tests
#define REG_VAL32(x) *(reinterpret_cast<uint32_t *>(x))
#define REG_VAL64(x) *(reinterpret_cast<uint64_t *>(x))

TEST_F(MinidumpParserTest, ConvertMinidumpContext_x86_32) {
  SetUpData("linux-i386.dmp");
  llvm::ArrayRef<MinidumpThread> thread_list = parser->GetThreads();
  const MinidumpThread thread = thread_list[0];
  llvm::ArrayRef<uint8_t> registers(parser->GetThreadContext(thread));

  ArchSpec arch = parser->GetArchitecture();
  auto reg_interface = llvm::make_unique<RegisterContextLinux_i386>(arch);
  lldb::DataBufferSP buf =
      ConvertMinidumpContext_x86_32(registers, reg_interface.get());
  ASSERT_EQ(reg_interface->GetGPRSize(), buf->GetByteSize());

  const RegisterInfo *reg_info = reg_interface->GetRegisterInfo();

  std::map<uint64_t, uint32_t> reg_values;

  reg_values[lldb_eax_i386] = 0x00000000;
  reg_values[lldb_ebx_i386] = 0xf7778000;
  reg_values[lldb_ecx_i386] = 0x00000001;
  reg_values[lldb_edx_i386] = 0xff9dd4a3;
  reg_values[lldb_edi_i386] = 0x080482a8;
  reg_values[lldb_esi_i386] = 0xff9dd55c;
  reg_values[lldb_ebp_i386] = 0xff9dd53c;
  reg_values[lldb_esp_i386] = 0xff9dd52c;
  reg_values[lldb_eip_i386] = 0x080482a0;
  reg_values[lldb_eflags_i386] = 0x00010282;
  reg_values[lldb_cs_i386] = 0x00000023;
  reg_values[lldb_fs_i386] = 0x00000000;
  reg_values[lldb_gs_i386] = 0x00000063;
  reg_values[lldb_ss_i386] = 0x0000002b;
  reg_values[lldb_ds_i386] = 0x0000002b;
  reg_values[lldb_es_i386] = 0x0000002b;

  for (uint32_t reg_index = 0; reg_index < reg_interface->GetRegisterCount();
       ++reg_index) {
    if (reg_values.find(reg_index) != reg_values.end()) {
      EXPECT_EQ(reg_values[reg_index],
                REG_VAL32(buf->GetBytes() + reg_info[reg_index].byte_offset));
    }
  }
}

TEST_F(MinidumpParserTest, ConvertMinidumpContext_x86_64) {
  SetUpData("linux-x86_64.dmp");
  llvm::ArrayRef<MinidumpThread> thread_list = parser->GetThreads();
  const MinidumpThread thread = thread_list[0];
  llvm::ArrayRef<uint8_t> registers(parser->GetThreadContext(thread));

  ArchSpec arch = parser->GetArchitecture();
  auto reg_interface = llvm::make_unique<RegisterContextLinux_x86_64>(arch);
  lldb::DataBufferSP buf =
      ConvertMinidumpContext_x86_64(registers, reg_interface.get());
  ASSERT_EQ(reg_interface->GetGPRSize(), buf->GetByteSize());

  const RegisterInfo *reg_info = reg_interface->GetRegisterInfo();

  std::map<uint64_t, uint64_t> reg_values;

  reg_values[lldb_rax_x86_64] = 0x0000000000000000;
  reg_values[lldb_rbx_x86_64] = 0x0000000000000000;
  reg_values[lldb_rcx_x86_64] = 0x0000000000000010;
  reg_values[lldb_rdx_x86_64] = 0x0000000000000000;
  reg_values[lldb_rdi_x86_64] = 0x00007ffceb349cf0;
  reg_values[lldb_rsi_x86_64] = 0x0000000000000000;
  reg_values[lldb_rbp_x86_64] = 0x00007ffceb34a210;
  reg_values[lldb_rsp_x86_64] = 0x00007ffceb34a210;
  reg_values[lldb_r8_x86_64] = 0x00007fe9bc1aa9c0;
  reg_values[lldb_r9_x86_64] = 0x0000000000000000;
  reg_values[lldb_r10_x86_64] = 0x00007fe9bc3f16a0;
  reg_values[lldb_r11_x86_64] = 0x0000000000000246;
  reg_values[lldb_r12_x86_64] = 0x0000000000401c92;
  reg_values[lldb_r13_x86_64] = 0x00007ffceb34a430;
  reg_values[lldb_r14_x86_64] = 0x0000000000000000;
  reg_values[lldb_r15_x86_64] = 0x0000000000000000;
  reg_values[lldb_rip_x86_64] = 0x0000000000401dc6;
  reg_values[lldb_rflags_x86_64] = 0x0000000000010206;
  reg_values[lldb_cs_x86_64] = 0x0000000000000033;
  reg_values[lldb_fs_x86_64] = 0x0000000000000000;
  reg_values[lldb_gs_x86_64] = 0x0000000000000000;
  reg_values[lldb_ss_x86_64] = 0x0000000000000000;
  reg_values[lldb_ds_x86_64] = 0x0000000000000000;
  reg_values[lldb_es_x86_64] = 0x0000000000000000;

  for (uint32_t reg_index = 0; reg_index < reg_interface->GetRegisterCount();
       ++reg_index) {
    if (reg_values.find(reg_index) != reg_values.end()) {
      EXPECT_EQ(reg_values[reg_index],
                REG_VAL64(buf->GetBytes() + reg_info[reg_index].byte_offset));
    }
  }
}

TEST_F(MinidumpParserTest, ConvertMinidumpContext_x86_32_wow64) {
  SetUpData("fizzbuzz_wow64.dmp");
  llvm::ArrayRef<MinidumpThread> thread_list = parser->GetThreads();
  const MinidumpThread thread = thread_list[0];
  llvm::ArrayRef<uint8_t> registers(parser->GetThreadContextWow64(thread));

  ArchSpec arch = parser->GetArchitecture();
  auto reg_interface = llvm::make_unique<RegisterContextLinux_i386>(arch);
  lldb::DataBufferSP buf =
      ConvertMinidumpContext_x86_32(registers, reg_interface.get());
  ASSERT_EQ(reg_interface->GetGPRSize(), buf->GetByteSize());

  const RegisterInfo *reg_info = reg_interface->GetRegisterInfo();

  std::map<uint64_t, uint32_t> reg_values;

  reg_values[lldb_eax_i386] = 0x00000000;
  reg_values[lldb_ebx_i386] = 0x0037f608;
  reg_values[lldb_ecx_i386] = 0x00e61578;
  reg_values[lldb_edx_i386] = 0x00000008;
  reg_values[lldb_edi_i386] = 0x00000000;
  reg_values[lldb_esi_i386] = 0x00000002;
  reg_values[lldb_ebp_i386] = 0x0037f654;
  reg_values[lldb_esp_i386] = 0x0037f5b8;
  reg_values[lldb_eip_i386] = 0x77ce01fd;
  reg_values[lldb_eflags_i386] = 0x00000246;
  reg_values[lldb_cs_i386] = 0x00000023;
  reg_values[lldb_fs_i386] = 0x00000053;
  reg_values[lldb_gs_i386] = 0x0000002b;
  reg_values[lldb_ss_i386] = 0x0000002b;
  reg_values[lldb_ds_i386] = 0x0000002b;
  reg_values[lldb_es_i386] = 0x0000002b;

  for (uint32_t reg_index = 0; reg_index < reg_interface->GetRegisterCount();
       ++reg_index) {
    if (reg_values.find(reg_index) != reg_values.end()) {
      EXPECT_EQ(reg_values[reg_index],
                REG_VAL32(buf->GetBytes() + reg_info[reg_index].byte_offset));
    }
  }
}
