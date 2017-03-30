//========- unittests/Support/Host.cpp - Host.cpp tests --------------========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Host.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"

#include "gtest/gtest.h"

using namespace llvm;

class HostTest : public testing::Test {
  Triple Host;

protected:
  bool isSupportedArchAndOS() {
    // Initially this is only testing detection of the number of
    // physical cores, which is currently only supported/tested for
    // x86_64 Linux and Darwin.
    return (Host.getArch() == Triple::x86_64 &&
            (Host.isOSDarwin() || Host.getOS() == Triple::Linux));
  }

  HostTest() : Host(Triple::normalize(sys::getProcessTriple())) {}
};

TEST_F(HostTest, NumPhysicalCores) {
  int Num = sys::getHostNumPhysicalCores();

  if (isSupportedArchAndOS())
    ASSERT_GT(Num, 0);
  else
    ASSERT_EQ(Num, -1);
}

TEST(getLinuxHostCPUName, ARM) {
  StringRef CortexA9ProcCpuinfo = R"(
processor       : 0
model name      : ARMv7 Processor rev 10 (v7l)
BogoMIPS        : 1393.66
Features        : half thumb fastmult vfp edsp thumbee neon vfpv3 tls vfpd32
CPU implementer : 0x41
CPU architecture: 7
CPU variant     : 0x2
CPU part        : 0xc09
CPU revision    : 10

processor       : 1
model name      : ARMv7 Processor rev 10 (v7l)
BogoMIPS        : 1393.66
Features        : half thumb fastmult vfp edsp thumbee neon vfpv3 tls vfpd32
CPU implementer : 0x41
CPU architecture: 7
CPU variant     : 0x2
CPU part        : 0xc09
CPU revision    : 10

Hardware        : Generic OMAP4 (Flattened Device Tree)
Revision        : 0000
Serial          : 0000000000000000
)";

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(CortexA9ProcCpuinfo),
            "cortex-a9");
  EXPECT_EQ(sys::detail::getHostCPUNameForARM("CPU implementer : 0x41\n"
                                              "CPU part        : 0xc0f"),
            "cortex-a15");
  // Verify that both CPU implementer and CPU part are checked:
  EXPECT_EQ(sys::detail::getHostCPUNameForARM("CPU implementer : 0x40\n"
                                              "CPU part        : 0xc0f"),
            "generic");
  EXPECT_EQ(sys::detail::getHostCPUNameForARM("CPU implementer : 0x51\n"
                                              "CPU part        : 0x06f"),
            "krait");
}
