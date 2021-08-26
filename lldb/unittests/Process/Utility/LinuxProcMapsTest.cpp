//===-- LinuxProcMapsTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Plugins/Process/Utility/LinuxProcMaps.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/Status.h"
#include <tuple>

using namespace lldb_private;

typedef std::tuple<const char *, MemoryRegionInfos, const char *>
    LinuxProcMapsTestParams;

// Wrapper for convenience because Range is usually begin, size
static MemoryRegionInfo::RangeType make_range(lldb::addr_t begin,
                                              lldb::addr_t end) {
  MemoryRegionInfo::RangeType range(begin, 0);
  range.SetRangeEnd(end);
  return range;
}

class LinuxProcMapsTestFixture
    : public ::testing::TestWithParam<LinuxProcMapsTestParams> {
protected:
  Status error;
  std::string err_str;
  MemoryRegionInfos regions;
  LinuxMapCallback callback;

  void SetUp() override {
    callback = [this](llvm::Expected<MemoryRegionInfo> Info) {
      if (Info) {
        err_str.clear();
        regions.push_back(*Info);
        return true;
      }

      err_str = toString(Info.takeError());
      return false;
    };
  }

  void check_regions(LinuxProcMapsTestParams params) {
    EXPECT_THAT(std::get<1>(params), testing::ContainerEq(regions));
    ASSERT_EQ(std::get<2>(params), err_str);
  }
};

TEST_P(LinuxProcMapsTestFixture, ParseMapRegions) {
  auto params = GetParam();
  ParseLinuxMapRegions(std::get<0>(params), callback);
  check_regions(params);
}

// Note: ConstString("") != ConstString(nullptr)
// When a region has no name, it will have the latter in the MemoryRegionInfo
INSTANTIATE_TEST_SUITE_P(
    ProcMapTests, LinuxProcMapsTestFixture,
    ::testing::Values(
        // Nothing in nothing out
        std::make_tuple("", MemoryRegionInfos{}, ""),
        // Various formatting error conditions
        std::make_tuple("55a4512f7000/55a451b68000 rw-p 00000000 00:00 0",
                        MemoryRegionInfos{},
                        "malformed /proc/{pid}/maps entry, missing dash "
                        "between address range"),
        std::make_tuple("0-0 rw", MemoryRegionInfos{},
                        "malformed /proc/{pid}/maps entry, missing some "
                        "portion of permissions"),
        std::make_tuple("0-0 z--p 00000000 00:00 0", MemoryRegionInfos{},
                        "unexpected /proc/{pid}/maps read permission char"),
        std::make_tuple("0-0 rz-p 00000000 00:00 0", MemoryRegionInfos{},
                        "unexpected /proc/{pid}/maps write permission char"),
        std::make_tuple("0-0 rwzp 00000000 00:00 0", MemoryRegionInfos{},
                        "unexpected /proc/{pid}/maps exec permission char"),
        // Stops at first parsing error
        std::make_tuple(
            "0-1 rw-p 00000000 00:00 0 [abc]\n"
            "0-0 rwzp 00000000 00:00 0\n"
            "2-3 r-xp 00000000 00:00 0 [def]\n",
            MemoryRegionInfos{
                MemoryRegionInfo(make_range(0, 1), MemoryRegionInfo::eYes,
                                 MemoryRegionInfo::eYes, MemoryRegionInfo::eNo,
                                 MemoryRegionInfo::eYes, ConstString("[abc]"),
                                 MemoryRegionInfo::eDontKnow, 0,
                                 MemoryRegionInfo::eDontKnow,
                                 MemoryRegionInfo::eDontKnow),
            },
            "unexpected /proc/{pid}/maps exec permission char"),
        // Single entry
        std::make_tuple(
            "55a4512f7000-55a451b68000 rw-p 00000000 00:00 0    [heap]",
            MemoryRegionInfos{
                MemoryRegionInfo(
                    make_range(0x55a4512f7000, 0x55a451b68000),
                    MemoryRegionInfo::eYes, MemoryRegionInfo::eYes,
                    MemoryRegionInfo::eNo, MemoryRegionInfo::eYes,
                    ConstString("[heap]"), MemoryRegionInfo::eDontKnow, 0,
                    MemoryRegionInfo::eDontKnow, MemoryRegionInfo::eDontKnow),
            },
            ""),
        // Multiple entries
        std::make_tuple(
            "7fc090021000-7fc094000000 ---p 00000000 00:00 0\n"
            "ffffffffff600000-ffffffffff601000 r-xp 00000000 00:00 0 "
            "[vsyscall]",
            MemoryRegionInfos{
                MemoryRegionInfo(
                    make_range(0x7fc090021000, 0x7fc094000000),
                    MemoryRegionInfo::eNo, MemoryRegionInfo::eNo,
                    MemoryRegionInfo::eNo, MemoryRegionInfo::eYes,
                    ConstString(nullptr), MemoryRegionInfo::eDontKnow, 0,
                    MemoryRegionInfo::eDontKnow, MemoryRegionInfo::eDontKnow),
                MemoryRegionInfo(
                    make_range(0xffffffffff600000, 0xffffffffff601000),
                    MemoryRegionInfo::eYes, MemoryRegionInfo::eNo,
                    MemoryRegionInfo::eYes, MemoryRegionInfo::eYes,
                    ConstString("[vsyscall]"), MemoryRegionInfo::eDontKnow, 0,
                    MemoryRegionInfo::eDontKnow, MemoryRegionInfo::eDontKnow),
            },
            "")));

class LinuxProcSMapsTestFixture : public LinuxProcMapsTestFixture {};

INSTANTIATE_TEST_SUITE_P(
    ProcSMapTests, LinuxProcSMapsTestFixture,
    ::testing::Values(
        // Nothing in nothing out
        std::make_tuple("", MemoryRegionInfos{}, ""),
        // Uses the same parsing for first line, so same errors but referring to
        // smaps
        std::make_tuple("0/0 rw-p 00000000 00:00 0", MemoryRegionInfos{},
                        "malformed /proc/{pid}/smaps entry, missing dash "
                        "between address range"),
        // Stop parsing at first error
        std::make_tuple(
            "1111-2222 rw-p 00000000 00:00 0 [foo]\n"
            "0/0 rw-p 00000000 00:00 0",
            MemoryRegionInfos{
                MemoryRegionInfo(
                    make_range(0x1111, 0x2222), MemoryRegionInfo::eYes,
                    MemoryRegionInfo::eYes, MemoryRegionInfo::eNo,
                    MemoryRegionInfo::eYes, ConstString("[foo]"),
                    MemoryRegionInfo::eDontKnow, 0, MemoryRegionInfo::eDontKnow,
                    MemoryRegionInfo::eDontKnow),
            },
            "malformed /proc/{pid}/smaps entry, missing dash between address "
            "range"),
        // Property line without a region is an error
        std::make_tuple("Referenced:         2188 kB\n"
                        "1111-2222 rw-p 00000000 00:00 0    [foo]\n"
                        "3333-4444 rw-p 00000000 00:00 0    [bar]\n",
                        MemoryRegionInfos{},
                        "Found a property line without a corresponding mapping "
                        "in /proc/{pid}/smaps"),
        // Single region parses, has no flags
        std::make_tuple(
            "1111-2222 rw-p 00000000 00:00 0    [foo]",
            MemoryRegionInfos{
                MemoryRegionInfo(
                    make_range(0x1111, 0x2222), MemoryRegionInfo::eYes,
                    MemoryRegionInfo::eYes, MemoryRegionInfo::eNo,
                    MemoryRegionInfo::eYes, ConstString("[foo]"),
                    MemoryRegionInfo::eDontKnow, 0, MemoryRegionInfo::eDontKnow,
                    MemoryRegionInfo::eDontKnow),
            },
            ""),
        // Single region with flags, other lines ignored
        std::make_tuple(
            "1111-2222 rw-p 00000000 00:00 0    [foo]\n"
            "Referenced:         2188 kB\n"
            "AnonHugePages:         0 kB\n"
            "VmFlags: mt",
            MemoryRegionInfos{
                MemoryRegionInfo(
                    make_range(0x1111, 0x2222), MemoryRegionInfo::eYes,
                    MemoryRegionInfo::eYes, MemoryRegionInfo::eNo,
                    MemoryRegionInfo::eYes, ConstString("[foo]"),
                    MemoryRegionInfo::eDontKnow, 0, MemoryRegionInfo::eYes,
                    MemoryRegionInfo::eDontKnow),
            },
            ""),
        // Whitespace ignored
        std::make_tuple(
            "0-0 rw-p 00000000 00:00 0\n"
            "VmFlags:      mt      ",
            MemoryRegionInfos{
                MemoryRegionInfo(make_range(0, 0), MemoryRegionInfo::eYes,
                                 MemoryRegionInfo::eYes, MemoryRegionInfo::eNo,
                                 MemoryRegionInfo::eYes, ConstString(nullptr),
                                 MemoryRegionInfo::eDontKnow, 0,
                                 MemoryRegionInfo::eYes,
                                 MemoryRegionInfo::eDontKnow),
            },
            ""),
        // VmFlags line means it has flag info, but nothing is set
        std::make_tuple(
            "0-0 rw-p 00000000 00:00 0\n"
            "VmFlags:         ",
            MemoryRegionInfos{
                MemoryRegionInfo(make_range(0, 0), MemoryRegionInfo::eYes,
                                 MemoryRegionInfo::eYes, MemoryRegionInfo::eNo,
                                 MemoryRegionInfo::eYes, ConstString(nullptr),
                                 MemoryRegionInfo::eDontKnow, 0,
                                 MemoryRegionInfo::eNo,
                                 MemoryRegionInfo::eDontKnow),
            },
            ""),
        // Handle some pages not having a flags line
        std::make_tuple(
            "1111-2222 rw-p 00000000 00:00 0    [foo]\n"
            "Referenced:         2188 kB\n"
            "AnonHugePages:         0 kB\n"
            "3333-4444 r-xp 00000000 00:00 0    [bar]\n"
            "VmFlags: mt",
            MemoryRegionInfos{
                MemoryRegionInfo(
                    make_range(0x1111, 0x2222), MemoryRegionInfo::eYes,
                    MemoryRegionInfo::eYes, MemoryRegionInfo::eNo,
                    MemoryRegionInfo::eYes, ConstString("[foo]"),
                    MemoryRegionInfo::eDontKnow, 0, MemoryRegionInfo::eDontKnow,
                    MemoryRegionInfo::eDontKnow),
                MemoryRegionInfo(
                    make_range(0x3333, 0x4444), MemoryRegionInfo::eYes,
                    MemoryRegionInfo::eNo, MemoryRegionInfo::eYes,
                    MemoryRegionInfo::eYes, ConstString("[bar]"),
                    MemoryRegionInfo::eDontKnow, 0, MemoryRegionInfo::eYes,
                    MemoryRegionInfo::eDontKnow),
            },
            ""),
        // Handle no pages having a flags line (older kernels)
        std::make_tuple(
            "1111-2222 rw-p 00000000 00:00 0\n"
            "Referenced:         2188 kB\n"
            "AnonHugePages:         0 kB\n"
            "3333-4444 r-xp 00000000 00:00 0\n"
            "KernelPageSize:        4 kB\n"
            "MMUPageSize:           4 kB\n",
            MemoryRegionInfos{
                MemoryRegionInfo(
                    make_range(0x1111, 0x2222), MemoryRegionInfo::eYes,
                    MemoryRegionInfo::eYes, MemoryRegionInfo::eNo,
                    MemoryRegionInfo::eYes, ConstString(nullptr),
                    MemoryRegionInfo::eDontKnow, 0, MemoryRegionInfo::eDontKnow,
                    MemoryRegionInfo::eDontKnow),
                MemoryRegionInfo(
                    make_range(0x3333, 0x4444), MemoryRegionInfo::eYes,
                    MemoryRegionInfo::eNo, MemoryRegionInfo::eYes,
                    MemoryRegionInfo::eYes, ConstString(nullptr),
                    MemoryRegionInfo::eDontKnow, 0, MemoryRegionInfo::eDontKnow,
                    MemoryRegionInfo::eDontKnow),
            },
            "")));

TEST_P(LinuxProcSMapsTestFixture, ParseSMapRegions) {
  auto params = GetParam();
  ParseLinuxSMapRegions(std::get<0>(params), callback);
  check_regions(params);
}
