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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "gtest/gtest.h"

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    SmallString<128> MessageStorage;                                           \
    raw_svector_ostream Message(MessageStorage);                               \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }

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

TEST(getLinuxHostCPUName, AArch64) {
  EXPECT_EQ(sys::detail::getHostCPUNameForARM("CPU implementer : 0x41\n"
                                              "CPU part        : 0xd03"),
            "cortex-a53");
  // Verify that both CPU implementer and CPU part are checked:
  EXPECT_EQ(sys::detail::getHostCPUNameForARM("CPU implementer : 0x40\n"
                                              "CPU part        : 0xd03"),
            "generic");
  EXPECT_EQ(sys::detail::getHostCPUNameForARM("CPU implementer : 0x51\n"
                                              "CPU part        : 0x201"),
            "kryo");
  EXPECT_EQ(sys::detail::getHostCPUNameForARM("CPU implementer : 0x51\n"
                                              "CPU part        : 0x800"),
            "cortex-a73");
  EXPECT_EQ(sys::detail::getHostCPUNameForARM("CPU implementer : 0x51\n"
                                              "CPU part        : 0x801"),
            "cortex-a73");
  EXPECT_EQ(sys::detail::getHostCPUNameForARM("CPU implementer : 0x51\n"
                                              "CPU part        : 0xc00"),
            "falkor");
  EXPECT_EQ(sys::detail::getHostCPUNameForARM("CPU implementer : 0x51\n"
                                              "CPU part        : 0xc01"),
            "saphira");

  // MSM8992/4 weirdness
  StringRef MSM8992ProcCpuInfo = R"(
Processor       : AArch64 Processor rev 3 (aarch64)
processor       : 0
processor       : 1
processor       : 2
processor       : 3
processor       : 4
processor       : 5
Features        : fp asimd evtstrm aes pmull sha1 sha2 crc32
CPU implementer : 0x41
CPU architecture: 8
CPU variant     : 0x0
CPU part        : 0xd03
CPU revision    : 3

Hardware        : Qualcomm Technologies, Inc MSM8992
)";

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(MSM8992ProcCpuInfo),
            "cortex-a53");

  // Exynos big.LITTLE weirdness
  const std::string ExynosProcCpuInfo = R"(
processor       : 0
Features        : fp asimd evtstrm aes pmull sha1 sha2 crc32
CPU implementer : 0x41
CPU architecture: 8
CPU variant     : 0x0
CPU part        : 0xd03

processor       : 1
Features        : fp asimd evtstrm aes pmull sha1 sha2 crc32
CPU implementer : 0x53
CPU architecture: 8
)";

  // Verify default for Exynos.
  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ExynosProcCpuInfo +
                                              "CPU variant     : 0xc\n"
                                              "CPU part        : 0xafe"),
            "exynos-m1");
  // Verify Exynos M1.
  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ExynosProcCpuInfo +
                                              "CPU variant     : 0x1\n"
                                              "CPU part        : 0x001"),
            "exynos-m1");
  // Verify Exynos M2.
  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ExynosProcCpuInfo +
                                              "CPU variant     : 0x4\n"
                                              "CPU part        : 0x001"),
            "exynos-m2");
}

#if defined(__APPLE__)
TEST_F(HostTest, getMacOSHostVersion) {
  using namespace llvm::sys;
  llvm::Triple HostTriple(getProcessTriple());
  if (!HostTriple.isMacOSX())
    return;

  SmallString<128> TestDirectory;
  ASSERT_NO_ERROR(fs::createUniqueDirectory("host_test", TestDirectory));
  SmallString<128> OutputFile(TestDirectory);
  path::append(OutputFile, "out");

  const char *SwVersPath = "/usr/bin/sw_vers";
  const char *argv[] = {SwVersPath, "-productVersion", nullptr};
  StringRef OutputPath = OutputFile.str();
  const Optional<StringRef> Redirects[] = {/*STDIN=*/None,
                                           /*STDOUT=*/OutputPath,
                                           /*STDERR=*/None};
  int RetCode = ExecuteAndWait(SwVersPath, argv, /*env=*/nullptr, Redirects);
  ASSERT_EQ(0, RetCode);

  int FD = 0;
  ASSERT_NO_ERROR(fs::openFileForRead(OutputPath, FD));
  off_t Size = ::lseek(FD, 0, SEEK_END);
  ASSERT_NE(-1, Size);
  ::lseek(FD, 0, SEEK_SET);
  std::unique_ptr<char[]> Buffer = llvm::make_unique<char[]>(Size);
  ASSERT_EQ(::read(FD, Buffer.get(), Size), Size);
  ::close(FD);

  // Ensure that the two versions match.
  StringRef SystemVersion(Buffer.get(), Size);
  unsigned SystemMajor, SystemMinor, SystemMicro;
  ASSERT_EQ(llvm::Triple((Twine("x86_64-apple-macos") + SystemVersion))
                .getMacOSXVersion(SystemMajor, SystemMinor, SystemMicro),
            true);
  unsigned HostMajor, HostMinor, HostMicro;
  ASSERT_EQ(HostTriple.getMacOSXVersion(HostMajor, HostMinor, HostMicro), true);

  // Don't compare the 'Micro' version, as it's always '0' for the 'Darwin'
  // triples.
  ASSERT_EQ(std::tie(SystemMajor, SystemMinor), std::tie(HostMajor, HostMinor));

  ASSERT_NO_ERROR(fs::remove(OutputPath));
  ASSERT_NO_ERROR(fs::remove(TestDirectory.str()));
}
#endif
