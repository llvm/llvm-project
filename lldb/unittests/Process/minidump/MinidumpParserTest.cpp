//===-- MinidumpTypesTest.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/minidump/MinidumpParser.h"
#include "Plugins/Process/minidump/MinidumpTypes.h"
#include "Plugins/Process/minidump/RegisterContextMinidump_x86_32.h"
#include "Plugins/Process/minidump/RegisterContextMinidump_x86_64.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/FileSpec.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ObjectYAML/MinidumpYAML.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

// C includes

// C++ includes
#include <memory>

using namespace lldb_private;
using namespace minidump;

class MinidumpParserTest : public testing::Test {
public:
  void SetUp() override { FileSystem::Initialize(); }

  void TearDown() override { FileSystem::Terminate(); }

  void SetUpData(const char *minidump_filename) {
    std::string filename = GetInputFilePath(minidump_filename);
    auto BufferPtr = FileSystem::Instance().CreateDataBuffer(filename, -1, 0);
    ASSERT_NE(BufferPtr, nullptr);
    llvm::Expected<MinidumpParser> expected_parser =
        MinidumpParser::Create(BufferPtr);
    ASSERT_THAT_EXPECTED(expected_parser, llvm::Succeeded());
    parser = std::move(*expected_parser);
    ASSERT_GT(parser->GetData().size(), 0UL);
  }

  llvm::Error SetUpFromYaml(llvm::StringRef yaml) {
    std::string data;
    llvm::raw_string_ostream os(data);
    if (llvm::Error E = llvm::MinidumpYAML::writeAsBinary(yaml, os))
      return E;

    os.flush();
    auto data_buffer_sp =
        std::make_shared<DataBufferHeap>(data.data(), data.size());
    auto expected_parser = MinidumpParser::Create(std::move(data_buffer_sp));
    if (!expected_parser)
      return expected_parser.takeError();
    parser = std::move(*expected_parser);
    return llvm::Error::success();
  }

  llvm::Optional<MinidumpParser> parser;
};

TEST_F(MinidumpParserTest, InvalidMinidump) {
  std::string duplicate_streams;
  llvm::raw_string_ostream os(duplicate_streams);
  ASSERT_THAT_ERROR(llvm::MinidumpYAML::writeAsBinary(R"(
--- !minidump
Streams:
  - Type:            LinuxAuxv
    Content:         DEADBEEFBAADF00D
  - Type:            LinuxAuxv
    Content:         DEADBEEFBAADF00D
  )",
                                                      os),
                    llvm::Succeeded());
  os.flush();
  auto data_buffer_sp = std::make_shared<DataBufferHeap>(
      duplicate_streams.data(), duplicate_streams.size());
  ASSERT_THAT_EXPECTED(MinidumpParser::Create(data_buffer_sp), llvm::Failed());
}

TEST_F(MinidumpParserTest, GetThreadsAndGetThreadContext) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            ThreadList
    Threads:
      - Thread Id:       0x00003E81
        Stack:
          Start of Memory Range: 0x00007FFCEB34A000
          Content:         C84D04BCE97F00
        Context:         00000000000000
...
)"),
                    llvm::Succeeded());
  llvm::ArrayRef<minidump::Thread> thread_list;

  thread_list = parser->GetThreads();
  ASSERT_EQ(1UL, thread_list.size());

  const minidump::Thread &thread = thread_list[0];

  EXPECT_EQ(0x3e81u, thread.ThreadId);

  llvm::ArrayRef<uint8_t> context = parser->GetThreadContext(thread);
  EXPECT_EQ(7u, context.size());
}

TEST_F(MinidumpParserTest, GetArchitecture) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            SystemInfo
    Processor Arch:  AMD64
    Processor Level: 6
    Processor Revision: 16130
    Number of Processors: 1
    Platform ID:     Linux
    CPU:
      Vendor ID:       GenuineIntel
      Version Info:    0x00000000
      Feature Info:    0x00000000
...
)"),
                    llvm::Succeeded());
  ASSERT_EQ(llvm::Triple::ArchType::x86_64,
            parser->GetArchitecture().GetMachine());
  ASSERT_EQ(llvm::Triple::OSType::Linux,
            parser->GetArchitecture().GetTriple().getOS());
}

TEST_F(MinidumpParserTest, GetMiscInfo_no_stream) {
  // Test that GetMiscInfo returns nullptr when the minidump does not contain
  // this stream.
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
...
)"),
                    llvm::Succeeded());
  EXPECT_EQ(nullptr, parser->GetMiscInfo());
}

TEST_F(MinidumpParserTest, GetLinuxProcStatus) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            SystemInfo
    Processor Arch:  AMD64
    Processor Level: 6
    Processor Revision: 16130
    Number of Processors: 1
    Platform ID:     Linux
    CSD Version:     'Linux 3.13.0-91-generic'
    CPU:
      Vendor ID:       GenuineIntel
      Version Info:    0x00000000
      Feature Info:    0x00000000
  - Type:            LinuxProcStatus
    Text:             |
      Name:	a.out
      State:	t (tracing stop)
      Tgid:	16001
      Ngid:	0
      Pid:	16001
      PPid:	13243
      TracerPid:	16002
      Uid:	404696	404696	404696	404696
      Gid:	5762	5762	5762	5762
...
)"),
                    llvm::Succeeded());
  llvm::Optional<LinuxProcStatus> proc_status = parser->GetLinuxProcStatus();
  ASSERT_TRUE(proc_status.hasValue());
  lldb::pid_t pid = proc_status->GetPid();
  ASSERT_EQ(16001UL, pid);
}

TEST_F(MinidumpParserTest, GetPid) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            SystemInfo
    Processor Arch:  AMD64
    Processor Level: 6
    Processor Revision: 16130
    Number of Processors: 1
    Platform ID:     Linux
    CSD Version:     'Linux 3.13.0-91-generic'
    CPU:
      Vendor ID:       GenuineIntel
      Version Info:    0x00000000
      Feature Info:    0x00000000
  - Type:            LinuxProcStatus
    Text:             |
      Name:	a.out
      State:	t (tracing stop)
      Tgid:	16001
      Ngid:	0
      Pid:	16001
      PPid:	13243
      TracerPid:	16002
      Uid:	404696	404696	404696	404696
      Gid:	5762	5762	5762	5762
...
)"),
                    llvm::Succeeded());
  llvm::Optional<lldb::pid_t> pid = parser->GetPid();
  ASSERT_TRUE(pid.hasValue());
  ASSERT_EQ(16001UL, pid.getValue());
}

TEST_F(MinidumpParserTest, GetFilteredModuleList) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            ModuleList
    Modules:
      - Base of Image:   0x0000000000400000
        Size of Image:   0x00001000
        Module Name:     '/tmp/test/linux-x86_64_not_crashed'
        CodeView Record: 4C4570426CCF3F60FFA7CC4B86AE8FF44DB2576A68983611
      - Base of Image:   0x0000000000600000
        Size of Image:   0x00002000
        Module Name:     '/tmp/test/linux-x86_64_not_crashed'
        CodeView Record: 4C4570426CCF3F60FFA7CC4B86AE8FF44DB2576A68983611
...
)"),
                    llvm::Succeeded());
  llvm::ArrayRef<minidump::Module> modules = parser->GetModuleList();
  std::vector<const minidump::Module *> filtered_modules =
      parser->GetFilteredModuleList();
  EXPECT_EQ(2u, modules.size());
  ASSERT_EQ(1u, filtered_modules.size());
  const minidump::Module &M = *filtered_modules[0];
  EXPECT_THAT_EXPECTED(parser->GetMinidumpFile().getString(M.ModuleNameRVA),
                       llvm::HasValue("/tmp/test/linux-x86_64_not_crashed"));
}

TEST_F(MinidumpParserTest, GetExceptionStream) {
  SetUpData("linux-x86_64.dmp");
  const MinidumpExceptionStream *exception_stream =
      parser->GetExceptionStream();
  ASSERT_NE(nullptr, exception_stream);
  ASSERT_EQ(11UL, exception_stream->exception_record.exception_code);
}

void check_mem_range_exists(MinidumpParser &parser, const uint64_t range_start,
                            const uint64_t range_size) {
  llvm::Optional<minidump::Range> range = parser.FindMemoryRange(range_start);
  ASSERT_TRUE(range.hasValue()) << "There is no range containing this address";
  EXPECT_EQ(range_start, range->start);
  EXPECT_EQ(range_start + range_size, range->start + range->range_ref.size());
}

TEST_F(MinidumpParserTest, FindMemoryRange) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            MemoryList
    Memory Ranges:
      - Start of Memory Range: 0x00007FFCEB34A000
        Content:         C84D04BCE9
      - Start of Memory Range: 0x0000000000401D46
        Content:         5421
...
)"),
                    llvm::Succeeded());
  EXPECT_EQ(llvm::None, parser->FindMemoryRange(0x00));
  EXPECT_EQ(llvm::None, parser->FindMemoryRange(0x2a));
  EXPECT_EQ((minidump::Range{0x401d46, llvm::ArrayRef<uint8_t>{0x54, 0x21}}),
            parser->FindMemoryRange(0x401d46));
  EXPECT_EQ(llvm::None, parser->FindMemoryRange(0x401d46 + 2));

  EXPECT_EQ(
      (minidump::Range{0x7ffceb34a000,
                       llvm::ArrayRef<uint8_t>{0xc8, 0x4d, 0x04, 0xbc, 0xe9}}),
      parser->FindMemoryRange(0x7ffceb34a000 + 2));
  EXPECT_EQ(llvm::None, parser->FindMemoryRange(0x7ffceb34a000 + 5));
}

TEST_F(MinidumpParserTest, GetMemory) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            MemoryList
    Memory Ranges:
      - Start of Memory Range: 0x00007FFCEB34A000
        Content:         C84D04BCE9
      - Start of Memory Range: 0x0000000000401D46
        Content:         5421
...
)"),
                    llvm::Succeeded());

  EXPECT_EQ((llvm::ArrayRef<uint8_t>{0x54}), parser->GetMemory(0x401d46, 1));
  EXPECT_EQ((llvm::ArrayRef<uint8_t>{0x54, 0x21}),
            parser->GetMemory(0x401d46, 4));

  EXPECT_EQ((llvm::ArrayRef<uint8_t>{0xc8, 0x4d, 0x04, 0xbc, 0xe9}),
            parser->GetMemory(0x7ffceb34a000, 5));
  EXPECT_EQ((llvm::ArrayRef<uint8_t>{0xc8, 0x4d, 0x04}),
            parser->GetMemory(0x7ffceb34a000, 3));

  EXPECT_EQ(llvm::ArrayRef<uint8_t>(), parser->GetMemory(0x500000, 512));
}

TEST_F(MinidumpParserTest, FindMemoryRangeWithFullMemoryMinidump) {
  SetUpData("fizzbuzz_wow64.dmp");

  // There are a lot of ranges in the file, just testing with some of them
  EXPECT_FALSE(parser->FindMemoryRange(0x00).hasValue());
  EXPECT_FALSE(parser->FindMemoryRange(0x2a).hasValue());
  check_mem_range_exists(*parser, 0x10000, 65536); // first range
  check_mem_range_exists(*parser, 0x40000, 4096);
  EXPECT_FALSE(parser->FindMemoryRange(0x40000 + 4096).hasValue());
  check_mem_range_exists(*parser, 0x77c12000, 8192);
  check_mem_range_exists(*parser, 0x7ffe0000, 4096); // last range
  EXPECT_FALSE(parser->FindMemoryRange(0x7ffe0000 + 4096).hasValue());
}

void check_region(MinidumpParser &parser, lldb::addr_t addr, lldb::addr_t start,
                  lldb::addr_t end, MemoryRegionInfo::OptionalBool read,
                  MemoryRegionInfo::OptionalBool write,
                  MemoryRegionInfo::OptionalBool exec,
                  MemoryRegionInfo::OptionalBool mapped,
                  ConstString name = ConstString()) {
  auto range_info = parser.GetMemoryRegionInfo(addr);
  EXPECT_EQ(start, range_info.GetRange().GetRangeBase());
  EXPECT_EQ(end, range_info.GetRange().GetRangeEnd());
  EXPECT_EQ(read, range_info.GetReadable());
  EXPECT_EQ(write, range_info.GetWritable());
  EXPECT_EQ(exec, range_info.GetExecutable());
  EXPECT_EQ(mapped, range_info.GetMapped());
  EXPECT_EQ(name, range_info.GetName());
}

// Same as above function where addr == start
void check_region(MinidumpParser &parser, lldb::addr_t start, lldb::addr_t end,
                  MemoryRegionInfo::OptionalBool read,
                  MemoryRegionInfo::OptionalBool write,
                  MemoryRegionInfo::OptionalBool exec,
                  MemoryRegionInfo::OptionalBool mapped,
                  ConstString name = ConstString()) {
  check_region(parser, start, start, end, read, write, exec, mapped, name);
}


constexpr auto yes = MemoryRegionInfo::eYes;
constexpr auto no = MemoryRegionInfo::eNo;
constexpr auto unknown = MemoryRegionInfo::eDontKnow;

TEST_F(MinidumpParserTest, GetMemoryRegionInfo) {
  SetUpData("fizzbuzz_wow64.dmp");

  check_region(*parser, 0x00000000, 0x00010000, no, no, no, no);
  check_region(*parser, 0x00010000, 0x00020000, yes, yes, no, yes);
  check_region(*parser, 0x00020000, 0x00030000, yes, yes, no, yes);
  check_region(*parser, 0x00030000, 0x00031000, yes, yes, no, yes);
  check_region(*parser, 0x00031000, 0x00040000, no, no, no, no);
  check_region(*parser, 0x00040000, 0x00041000, yes, no, no, yes);

  // Check addresses contained inside ranges
  check_region(*parser, 0x00000001, 0x00000000, 0x00010000, no, no, no, no);
  check_region(*parser, 0x0000ffff, 0x00000000, 0x00010000, no, no, no, no);
  check_region(*parser, 0x00010001, 0x00010000, 0x00020000, yes, yes, no, yes);
  check_region(*parser, 0x0001ffff, 0x00010000, 0x00020000, yes, yes, no, yes);

  // Test that an address after the last entry maps to rest of the memory space
  check_region(*parser, 0x7fff0000, 0x7fff0000, UINT64_MAX, no, no, no, no);
}

TEST_F(MinidumpParserTest, GetMemoryRegionInfoFromMemoryList) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            MemoryList
    Memory Ranges:
      - Start of Memory Range: 0x0000000000001000
        Content:         '31313131313131313131313131313131'
      - Start of Memory Range: 0x0000000000002000
        Content:         '3333333333333333333333333333333333333333333333333333333333333333'
...
)"),
                    llvm::Succeeded());
  // Test we can get memory regions from the MINIDUMP_MEMORY_LIST stream when
  // we don't have a MemoryInfoListStream.

  // Test addres before the first entry comes back with nothing mapped up
  // to first valid region info
  check_region(*parser, 0x00000000, 0x00001000, no, no, no, no);
  check_region(*parser, 0x00001000, 0x00001010, yes, unknown, unknown, yes);
  check_region(*parser, 0x00001010, 0x00002000, no, no, no, no);
  check_region(*parser, 0x00002000, 0x00002020, yes, unknown, unknown, yes);
  check_region(*parser, 0x00002020, UINT64_MAX, no, no, no, no);
}

TEST_F(MinidumpParserTest, GetMemoryRegionInfoFromMemory64List) {
  SetUpData("regions-memlist64.dmp");
  // Test we can get memory regions from the MINIDUMP_MEMORY64_LIST stream when
  // we don't have a MemoryInfoListStream.

  // Test addres before the first entry comes back with nothing mapped up
  // to first valid region info
  check_region(*parser, 0x00000000, 0x00001000, no, no, no, no);
  check_region(*parser, 0x00001000, 0x00001010, yes, unknown, unknown, yes);
  check_region(*parser, 0x00001010, 0x00002000, no, no, no, no);
  check_region(*parser, 0x00002000, 0x00002020, yes, unknown, unknown, yes);
  check_region(*parser, 0x00002020, UINT64_MAX, no, no, no, no);
}

TEST_F(MinidumpParserTest, GetMemoryRegionInfoLinuxMaps) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            LinuxMaps
    Text:             |
      400d9000-400db000 r-xp 00000000 b3:04 227        /system/bin/app_process
      400db000-400dc000 r--p 00001000 b3:04 227        /system/bin/app_process
      400dc000-400dd000 rw-p 00000000 00:00 0
      400dd000-400ec000 r-xp 00000000 b3:04 300        /system/bin/linker
      400ec000-400ed000 r--p 00000000 00:00 0
      400ed000-400ee000 r--p 0000f000 b3:04 300        /system/bin/linker
      400ee000-400ef000 rw-p 00010000 b3:04 300        /system/bin/linker
      400ef000-400fb000 rw-p 00000000 00:00 0
      400fb000-400fc000 r-xp 00000000 b3:04 1096       /system/lib/liblog.so
      400fc000-400fd000 rwxp 00001000 b3:04 1096       /system/lib/liblog.so
      400fd000-400ff000 r-xp 00002000 b3:04 1096       /system/lib/liblog.so
      400ff000-40100000 r--p 00003000 b3:04 1096       /system/lib/liblog.so
      40100000-40101000 rw-p 00004000 b3:04 1096       /system/lib/liblog.so
      40101000-40122000 r-xp 00000000 b3:04 955        /system/lib/libc.so
      40122000-40123000 rwxp 00021000 b3:04 955        /system/lib/libc.so
      40123000-40167000 r-xp 00022000 b3:04 955        /system/lib/libc.so
      40167000-40169000 r--p 00065000 b3:04 955        /system/lib/libc.so
      40169000-4016b000 rw-p 00067000 b3:04 955        /system/lib/libc.so
      4016b000-40176000 rw-p 00000000 00:00 0

...
)"),
                    llvm::Succeeded());
  // Test we can get memory regions from the linux /proc/<pid>/maps stream when
  // we don't have a MemoryInfoListStream.

  // Test addres before the first entry comes back with nothing mapped up
  // to first valid region info
  ConstString a("/system/bin/app_process");
  ConstString b("/system/bin/linker");
  ConstString c("/system/lib/liblog.so");
  ConstString d("/system/lib/libc.so");
  ConstString n;
  check_region(*parser, 0x00000000, 0x400d9000, no, no, no, no, n);
  check_region(*parser, 0x400d9000, 0x400db000, yes, no, yes, yes, a);
  check_region(*parser, 0x400db000, 0x400dc000, yes, no, no, yes, a);
  check_region(*parser, 0x400dc000, 0x400dd000, yes, yes, no, yes, n);
  check_region(*parser, 0x400dd000, 0x400ec000, yes, no, yes, yes, b);
  check_region(*parser, 0x400ec000, 0x400ed000, yes, no, no, yes, n);
  check_region(*parser, 0x400ed000, 0x400ee000, yes, no, no, yes, b);
  check_region(*parser, 0x400ee000, 0x400ef000, yes, yes, no, yes, b);
  check_region(*parser, 0x400ef000, 0x400fb000, yes, yes, no, yes, n);
  check_region(*parser, 0x400fb000, 0x400fc000, yes, no, yes, yes, c);
  check_region(*parser, 0x400fc000, 0x400fd000, yes, yes, yes, yes, c);
  check_region(*parser, 0x400fd000, 0x400ff000, yes, no, yes, yes, c);
  check_region(*parser, 0x400ff000, 0x40100000, yes, no, no, yes, c);
  check_region(*parser, 0x40100000, 0x40101000, yes, yes, no, yes, c);
  check_region(*parser, 0x40101000, 0x40122000, yes, no, yes, yes, d);
  check_region(*parser, 0x40122000, 0x40123000, yes, yes, yes, yes, d);
  check_region(*parser, 0x40123000, 0x40167000, yes, no, yes, yes, d);
  check_region(*parser, 0x40167000, 0x40169000, yes, no, no, yes, d);
  check_region(*parser, 0x40169000, 0x4016b000, yes, yes, no, yes, d);
  check_region(*parser, 0x4016b000, 0x40176000, yes, yes, no, yes, n);
  check_region(*parser, 0x40176000, UINT64_MAX, no, no, no, no, n);
}

// Windows Minidump tests
TEST_F(MinidumpParserTest, GetArchitectureWindows) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            SystemInfo
    Processor Arch:  X86
    Processor Level: 6
    Processor Revision: 15876
    Number of Processors: 32
    Product type:    1
    Major Version:   6
    Minor Version:   1
    Build Number:    7601
    Platform ID:     Win32NT
    CSD Version:     Service Pack 1
    Suite Mask:      0x0100
    CPU:
      Vendor ID:       GenuineIntel
      Version Info:    0x000306E4
      Feature Info:    0xBFEBFBFF
      AMD Extended Features: 0x771EEC80
...
)"),
                    llvm::Succeeded());
  ASSERT_EQ(llvm::Triple::ArchType::x86,
            parser->GetArchitecture().GetMachine());
  ASSERT_EQ(llvm::Triple::OSType::Win32,
            parser->GetArchitecture().GetTriple().getOS());
}

TEST_F(MinidumpParserTest, GetLinuxProcStatus_no_stream) {
  // Test that GetLinuxProcStatus returns nullptr when the minidump does not
  // contain this stream.
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
...
)"),
                    llvm::Succeeded());
  EXPECT_EQ(llvm::None, parser->GetLinuxProcStatus());
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

// Register tests
#define REG_VAL32(x) *(reinterpret_cast<uint32_t *>(x))
#define REG_VAL64(x) *(reinterpret_cast<uint64_t *>(x))

TEST_F(MinidumpParserTest, GetThreadContext_x86_32) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            ThreadList
    Threads:
      - Thread Id:       0x00026804
        Stack:
          Start of Memory Range: 0x00000000FF9DD000
          Content:         68D39DFF
        Context:         0F0001000000000000000000000000000000000000000000000000007F03FFFF0000FFFFFFFFFFFF09DC62F72300000088E36CF72B00FFFF00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000063000000000000002B0000002B000000A88204085CD59DFF008077F7A3D49DFF01000000000000003CD59DFFA082040823000000820201002CD59DFF2B0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
)"),
                    llvm::Succeeded());

  llvm::ArrayRef<minidump::Thread> thread_list = parser->GetThreads();
  const minidump::Thread &thread = thread_list[0];
  llvm::ArrayRef<uint8_t> registers(parser->GetThreadContext(thread));
  const MinidumpContext_x86_32 *context;
  EXPECT_TRUE(consumeObject(registers, context).Success());

  EXPECT_EQ(MinidumpContext_x86_32_Flags(uint32_t(context->context_flags)),
            MinidumpContext_x86_32_Flags::x86_32_Flag |
                MinidumpContext_x86_32_Flags::Full |
                MinidumpContext_x86_32_Flags::FloatingPoint);

  EXPECT_EQ(0x00000000u, context->eax);
  EXPECT_EQ(0xf7778000u, context->ebx);
  EXPECT_EQ(0x00000001u, context->ecx);
  EXPECT_EQ(0xff9dd4a3u, context->edx);
  EXPECT_EQ(0x080482a8u, context->edi);
  EXPECT_EQ(0xff9dd55cu, context->esi);
  EXPECT_EQ(0xff9dd53cu, context->ebp);
  EXPECT_EQ(0xff9dd52cu, context->esp);
  EXPECT_EQ(0x080482a0u, context->eip);
  EXPECT_EQ(0x00010282u, context->eflags);
  EXPECT_EQ(0x0023u, context->cs);
  EXPECT_EQ(0x0000u, context->fs);
  EXPECT_EQ(0x0063u, context->gs);
  EXPECT_EQ(0x002bu, context->ss);
  EXPECT_EQ(0x002bu, context->ds);
  EXPECT_EQ(0x002bu, context->es);
}

TEST_F(MinidumpParserTest, GetThreadContext_x86_64) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            ThreadList
    Threads:
      - Thread Id:       0x00003E81
        Stack:
          Start of Memory Range: 0x00007FFCEB34A000
          Content:         C84D04BCE97F00
        Context:         0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B0010000000000033000000000000000000000006020100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000010A234EBFC7F000010A234EBFC7F00000000000000000000F09C34EBFC7F0000C0A91ABCE97F00000000000000000000A0163FBCE97F00004602000000000000921C40000000000030A434EBFC7F000000000000000000000000000000000000C61D4000000000007F0300000000000000000000000000000000000000000000801F0000FFFF0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000FFFF00FFFFFFFFFFFFFF00FFFFFFFF25252525252525252525252525252525000000000000000000000000000000000000000000000000000000000000000000FFFF00FFFFFFFFFFFFFF00FFFFFFFF0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000FF00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
...
)"),
                    llvm::Succeeded());
  llvm::ArrayRef<minidump::Thread> thread_list = parser->GetThreads();
  const minidump::Thread &thread = thread_list[0];
  llvm::ArrayRef<uint8_t> registers(parser->GetThreadContext(thread));
  const MinidumpContext_x86_64 *context;
  EXPECT_TRUE(consumeObject(registers, context).Success());

  EXPECT_EQ(MinidumpContext_x86_64_Flags(uint32_t(context->context_flags)),
            MinidumpContext_x86_64_Flags::x86_64_Flag |
                MinidumpContext_x86_64_Flags::Control |
                MinidumpContext_x86_64_Flags::FloatingPoint |
                MinidumpContext_x86_64_Flags::Integer);
  EXPECT_EQ(0x0000000000000000u, context->rax);
  EXPECT_EQ(0x0000000000000000u, context->rbx);
  EXPECT_EQ(0x0000000000000010u, context->rcx);
  EXPECT_EQ(0x0000000000000000u, context->rdx);
  EXPECT_EQ(0x00007ffceb349cf0u, context->rdi);
  EXPECT_EQ(0x0000000000000000u, context->rsi);
  EXPECT_EQ(0x00007ffceb34a210u, context->rbp);
  EXPECT_EQ(0x00007ffceb34a210u, context->rsp);
  EXPECT_EQ(0x00007fe9bc1aa9c0u, context->r8);
  EXPECT_EQ(0x0000000000000000u, context->r9);
  EXPECT_EQ(0x00007fe9bc3f16a0u, context->r10);
  EXPECT_EQ(0x0000000000000246u, context->r11);
  EXPECT_EQ(0x0000000000401c92u, context->r12);
  EXPECT_EQ(0x00007ffceb34a430u, context->r13);
  EXPECT_EQ(0x0000000000000000u, context->r14);
  EXPECT_EQ(0x0000000000000000u, context->r15);
  EXPECT_EQ(0x0000000000401dc6u, context->rip);
  EXPECT_EQ(0x00010206u, context->eflags);
  EXPECT_EQ(0x0033u, context->cs);
  EXPECT_EQ(0x0000u, context->ss);
}

TEST_F(MinidumpParserTest, GetThreadContext_x86_32_wow64) {
  SetUpData("fizzbuzz_wow64.dmp");
  llvm::ArrayRef<minidump::Thread> thread_list = parser->GetThreads();
  const minidump::Thread &thread = thread_list[0];
  llvm::ArrayRef<uint8_t> registers(parser->GetThreadContextWow64(thread));
  const MinidumpContext_x86_32 *context;
  EXPECT_TRUE(consumeObject(registers, context).Success());

  EXPECT_EQ(MinidumpContext_x86_32_Flags(uint32_t(context->context_flags)),
            MinidumpContext_x86_32_Flags::x86_32_Flag |
                MinidumpContext_x86_32_Flags::Full |
                MinidumpContext_x86_32_Flags::FloatingPoint |
                MinidumpContext_x86_32_Flags::ExtendedRegisters);

  EXPECT_EQ(0x00000000u, context->eax);
  EXPECT_EQ(0x0037f608u, context->ebx);
  EXPECT_EQ(0x00e61578u, context->ecx);
  EXPECT_EQ(0x00000008u, context->edx);
  EXPECT_EQ(0x00000000u, context->edi);
  EXPECT_EQ(0x00000002u, context->esi);
  EXPECT_EQ(0x0037f654u, context->ebp);
  EXPECT_EQ(0x0037f5b8u, context->esp);
  EXPECT_EQ(0x77ce01fdu, context->eip);
  EXPECT_EQ(0x00000246u, context->eflags);
  EXPECT_EQ(0x0023u, context->cs);
  EXPECT_EQ(0x0053u, context->fs);
  EXPECT_EQ(0x002bu, context->gs);
  EXPECT_EQ(0x002bu, context->ss);
  EXPECT_EQ(0x002bu, context->ds);
  EXPECT_EQ(0x002bu, context->es);
}

TEST_F(MinidumpParserTest, MinidumpDuplicateModuleMinAddress) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            ModuleList
    Modules:
      - Base of Image:   0x0000000000002000
        Size of Image:   0x00001000
        Module Name:     '/tmp/a'
        CodeView Record: ''
      - Base of Image:   0x0000000000001000
        Size of Image:   0x00001000
        Module Name:     '/tmp/a'
        CodeView Record: ''
...
)"),
                    llvm::Succeeded());
  // If we have a module mentioned twice in the module list, the filtered
  // module list should contain the instance with the lowest BaseOfImage.
  std::vector<const minidump::Module *> filtered_modules =
      parser->GetFilteredModuleList();
  ASSERT_EQ(1u, filtered_modules.size());
  EXPECT_EQ(0x0000000000001000u, filtered_modules[0]->BaseOfImage);
}

TEST_F(MinidumpParserTest, MinidumpModuleOrder) {
  ASSERT_THAT_ERROR(SetUpFromYaml(R"(
--- !minidump
Streams:
  - Type:            ModuleList
    Modules:
      - Base of Image:   0x0000000000002000
        Size of Image:   0x00001000
        Module Name:     '/tmp/a'
        CodeView Record: ''
      - Base of Image:   0x0000000000001000
        Size of Image:   0x00001000
        Module Name:     '/tmp/b'
        CodeView Record: ''
...
)"),
                    llvm::Succeeded());
  // Test module filtering does not affect the overall module order.  Previous
  // versions of the MinidumpParser::GetFilteredModuleList() function would sort
  // all images by address and modify the order of the modules.
  std::vector<const minidump::Module *> filtered_modules =
      parser->GetFilteredModuleList();
  ASSERT_EQ(2u, filtered_modules.size());
  EXPECT_EQ(0x0000000000002000u, filtered_modules[0]->BaseOfImage);
  EXPECT_THAT_EXPECTED(
      parser->GetMinidumpFile().getString(filtered_modules[0]->ModuleNameRVA),
      llvm::HasValue("/tmp/a"));
  EXPECT_EQ(0x0000000000001000u, filtered_modules[1]->BaseOfImage);
  EXPECT_THAT_EXPECTED(
      parser->GetMinidumpFile().getString(filtered_modules[1]->ModuleNameRVA),
      llvm::HasValue("/tmp/b"));
}

