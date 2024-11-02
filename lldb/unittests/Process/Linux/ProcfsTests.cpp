//===-- ProcfsTests.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Procfs.h"

#include "lldb/Host/linux/Support.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace process_linux;
using namespace llvm;

TEST(Perf, HardcodedLogicalCoreIDs) {
  Expected<std::vector<lldb::core_id_t>> core_ids =
      GetAvailableLogicalCoreIDs(R"(processor       : 13
vendor_id       : GenuineIntel
cpu family      : 6
model           : 85
model name      : Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
stepping        : 4
microcode       : 0x2000065
cpu MHz         : 2886.370
cache size      : 28160 KB
physical id     : 1
siblings        : 40
core id         : 19
cpu cores       : 20
apicid          : 103
initial apicid  : 103
fpu             : yes
fpu_exception   : yes
cpuid level     : 22
power management:

processor       : 24
vendor_id       : GenuineIntel
cpu family      : 6
model           : 85
model name      : Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
stepping        : 4
microcode       : 0x2000065
cpu MHz         : 2768.494
cache size      : 28160 KB
physical id     : 1
siblings        : 40
core id         : 20
cpu cores       : 20
apicid          : 105
power management:

processor       : 35
vendor_id       : GenuineIntel
cpu family      : 6
model           : 85
model name      : Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
stepping        : 4
microcode       : 0x2000065
cpu MHz         : 2884.703
cache size      : 28160 KB
physical id     : 1
siblings        : 40
core id         : 24
cpu cores       : 20
apicid          : 113

processor       : 79
vendor_id       : GenuineIntel
cpu family      : 6
model           : 85
model name      : Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
stepping        : 4
microcode       : 0x2000065
cpu MHz         : 3073.955
cache size      : 28160 KB
physical id     : 1
siblings        : 40
core id         : 28
cpu cores       : 20
apicid          : 121
power management:
)");

  ASSERT_TRUE((bool)core_ids);
  ASSERT_THAT(*core_ids, ::testing::ElementsAre(13, 24, 35, 79));
}

TEST(Perf, RealLogicalCoreIDs) {
  // We first check we can read /proc/cpuinfo
  auto buffer_or_error = errorOrToExpected(getProcFile("cpuinfo"));
  if (!buffer_or_error)
    GTEST_SKIP() << toString(buffer_or_error.takeError());

  // At this point we shouldn't fail parsing the core ids
  Expected<ArrayRef<lldb::core_id_t>> core_ids = GetAvailableLogicalCoreIDs();
  ASSERT_TRUE((bool)core_ids);
  ASSERT_GT((int)core_ids->size(), 0) << "We must see at least one core";
}
