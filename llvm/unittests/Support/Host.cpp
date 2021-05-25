//========- unittests/Support/Host.cpp - Host.cpp tests --------------========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Host.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Threading.h"

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
    // physical cores, which is currently only supported/tested on
    // some systems.
    return (Host.isOSWindows() && llvm_is_multithreaded()) ||
           (Host.isX86() && (Host.isOSDarwin() || Host.isOSLinux())) ||
           (Host.isPPC64() && Host.isOSLinux()) ||
           (Host.isSystemZ() && (Host.isOSLinux() || Host.isOSzOS()));
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

  EXPECT_EQ(sys::detail::getHostCPUNameForARM("CPU implementer : 0x41\n"
                                              "CPU part        : 0xd0c"),
            "neoverse-n1");
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
CPU part        : 0xd05

processor       : 1
Features        : fp asimd evtstrm aes pmull sha1 sha2 crc32
CPU implementer : 0x53
CPU architecture: 8
)";

  // Verify default for Exynos.
  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ExynosProcCpuInfo +
                                              "CPU variant     : 0xc\n"
                                              "CPU part        : 0xafe"),
            "exynos-m3");
  // Verify Exynos M3.
  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ExynosProcCpuInfo +
                                              "CPU variant     : 0x1\n"
                                              "CPU part        : 0x002"),
            "exynos-m3");
  // Verify Exynos M4.
  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ExynosProcCpuInfo +
                                              "CPU variant     : 0x1\n"
                                              "CPU part        : 0x003"),
            "exynos-m4");

  const std::string ThunderX2T99ProcCpuInfo = R"(
processor	: 0
BogoMIPS	: 400.00
Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics
CPU implementer	: 0x43
CPU architecture: 8
CPU variant	: 0x1
CPU part	: 0x0af
)";

  // Verify different versions of ThunderX2T99.
  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ThunderX2T99ProcCpuInfo +
                                              "CPU implementer	: 0x42\n"
                                              "CPU part	: 0x516"),
            "thunderx2t99");

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ThunderX2T99ProcCpuInfo +
                                              "CPU implementer	: 0x42\n"
                                              "CPU part	: 0x0516"),
            "thunderx2t99");

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ThunderX2T99ProcCpuInfo +
                                              "CPU implementer	: 0x43\n"
                                              "CPU part	: 0x516"),
            "thunderx2t99");

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ThunderX2T99ProcCpuInfo +
                                              "CPU implementer	: 0x43\n"
                                              "CPU part	: 0x0516"),
            "thunderx2t99");

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ThunderX2T99ProcCpuInfo +
                                              "CPU implementer	: 0x42\n"
                                              "CPU part	: 0xaf"),
            "thunderx2t99");

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ThunderX2T99ProcCpuInfo +
                                              "CPU implementer	: 0x42\n"
                                              "CPU part	: 0x0af"),
            "thunderx2t99");

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ThunderX2T99ProcCpuInfo +
                                              "CPU implementer	: 0x43\n"
                                              "CPU part	: 0xaf"),
            "thunderx2t99");

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ThunderX2T99ProcCpuInfo +
                                              "CPU implementer	: 0x43\n"
                                              "CPU part	: 0x0af"),
            "thunderx2t99");

  // Verify ThunderXT88.
  const std::string ThunderXT88ProcCpuInfo = R"(
processor	: 0
BogoMIPS	: 200.00
Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32
CPU implementer	: 0x43
CPU architecture: 8
CPU variant	: 0x1
CPU part	: 0x0a1
)";

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ThunderXT88ProcCpuInfo +
                                              "CPU implementer	: 0x43\n"
                                              "CPU part	: 0x0a1"),
            "thunderxt88");

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(ThunderXT88ProcCpuInfo +
                                              "CPU implementer	: 0x43\n"
                                              "CPU part	: 0xa1"),
            "thunderxt88");

  // Verify HiSilicon processors.
  EXPECT_EQ(sys::detail::getHostCPUNameForARM("CPU implementer : 0x48\n"
                                              "CPU part        : 0xd01"),
            "tsv110");

  // Verify A64FX.
  const std::string A64FXProcCpuInfo = R"(
processor       : 0
BogoMIPS        : 200.00
Features        : fp asimd evtstrm sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm fcma dcpop sve
CPU implementer : 0x46
CPU architecture: 8
CPU variant     : 0x1
CPU part        : 0x001
)";

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(A64FXProcCpuInfo), "a64fx");

  // Verify Nvidia Carmel.
  const std::string CarmelProcCpuInfo = R"(
processor       : 0
model name      : ARMv8 Processor rev 0 (v8l)
BogoMIPS        : 62.50
Features        : fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm dcpop
CPU implementer : 0x4e
CPU architecture: 8
CPU variant     : 0x0
CPU part        : 0x004
CPU revision    : 0
)";

  EXPECT_EQ(sys::detail::getHostCPUNameForARM(CarmelProcCpuInfo), "carmel");

  // Snapdragon mixed implementer quirk
  const std::string Snapdragon865ProcCPUInfo = R"(
processor       : 0
BogoMIPS        : 38.40
Features        : fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
CPU implementer : 0x51
CPU architecture: 8
CPU variant     : 0xd
CPU part        : 0x805
CPU revision    : 14
processor       : 1
processor       : 2
processor       : 3
processor       : 4
processor       : 5
processor       : 6
BogoMIPS        : 38.40
Features        : fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
CPU implementer : 0x41
CPU architecture: 8
CPU variant     : 0x1
CPU part        : 0xd0d
CPU revision    : 0
)";
  EXPECT_EQ(sys::detail::getHostCPUNameForARM(Snapdragon865ProcCPUInfo), "cortex-a77");
}

TEST(getLinuxHostCPUName, s390x) {
  SmallVector<std::string> ModelIDs(
      {"8561", "3906", "2964", "2827", "2817", "7"});
  SmallVector<std::string> VectorSupport({"", "vx"});
  SmallVector<StringRef> ExpectedCPUs;

  // Model Id: 8561
  ExpectedCPUs.push_back("zEC12");
  ExpectedCPUs.push_back("z15");

  // Model Id: 3906
  ExpectedCPUs.push_back("zEC12");
  ExpectedCPUs.push_back("z14");

  // Model Id: 2964
  ExpectedCPUs.push_back("zEC12");
  ExpectedCPUs.push_back("z13");

  // Model Id: 2827
  ExpectedCPUs.push_back("zEC12");
  ExpectedCPUs.push_back("zEC12");

  // Model Id: 2817
  ExpectedCPUs.push_back("z196");
  ExpectedCPUs.push_back("z196");

  // Model Id: 7
  ExpectedCPUs.push_back("generic");
  ExpectedCPUs.push_back("generic");

  const std::string DummyBaseVectorInfo =
      "features : esan3 zarch stfle msa ldisp eimm dfp edat etf3eh highgprs "
      "te ";
  const std::string DummyBaseMachineInfo =
      "processor 0: version = FF,  identification = 059C88,  machine = ";

  int CheckIndex = 0;
  for (size_t I = 0; I < ModelIDs.size(); I++) {
    for (size_t J = 0; J < VectorSupport.size(); J++) {
      const std::string DummyCPUInfo = DummyBaseVectorInfo + VectorSupport[J] +
                                       "\n" + DummyBaseMachineInfo +
                                       ModelIDs[I];
      EXPECT_EQ(sys::detail::getHostCPUNameForS390x(DummyCPUInfo),
                ExpectedCPUs[CheckIndex++]);
    }
  }
}

#if defined(__APPLE__) || defined(_AIX)
static bool runAndGetCommandOutput(
    const char *ExePath, ArrayRef<llvm::StringRef> argv,
    std::unique_ptr<char[]> &Buffer, off_t &Size) {
  bool Success = false;
  [ExePath, argv, &Buffer, &Size, &Success] {
    using namespace llvm::sys;
    SmallString<128> TestDirectory;
    ASSERT_NO_ERROR(fs::createUniqueDirectory("host_test", TestDirectory));

    SmallString<128> OutputFile(TestDirectory);
    path::append(OutputFile, "out");
    StringRef OutputPath = OutputFile.str();

    const Optional<StringRef> Redirects[] = {
        /*STDIN=*/None, /*STDOUT=*/OutputPath, /*STDERR=*/None};
    int RetCode = ExecuteAndWait(ExePath, argv, /*env=*/llvm::None, Redirects);
    ASSERT_EQ(0, RetCode);

    int FD = 0;
    ASSERT_NO_ERROR(fs::openFileForRead(OutputPath, FD));
    Size = ::lseek(FD, 0, SEEK_END);
    ASSERT_NE(-1, Size);
    ::lseek(FD, 0, SEEK_SET);
    Buffer = std::make_unique<char[]>(Size);
    ASSERT_EQ(::read(FD, Buffer.get(), Size), Size);
    ::close(FD);

    ASSERT_NO_ERROR(fs::remove(OutputPath));
    ASSERT_NO_ERROR(fs::remove(TestDirectory.str()));
    Success = true;
  }();
  return Success;
}

TEST_F(HostTest, DummyRunAndGetCommandOutputUse) {
  // Suppress defined-but-not-used warnings when the tests using the helper are
  // disabled.
  (void) runAndGetCommandOutput;
}
#endif

#if defined(__APPLE__)
TEST_F(HostTest, getMacOSHostVersion) {
  using namespace llvm::sys;
  llvm::Triple HostTriple(getProcessTriple());
  if (!HostTriple.isMacOSX())
    return;

  const char *SwVersPath = "/usr/bin/sw_vers";
  StringRef argv[] = {SwVersPath, "-productVersion"};
  std::unique_ptr<char[]> Buffer;
  off_t Size;
  ASSERT_EQ(runAndGetCommandOutput(SwVersPath, argv, Buffer, Size), true);
  StringRef SystemVersion(Buffer.get(), Size);

  // Ensure that the two versions match.
  unsigned SystemMajor, SystemMinor, SystemMicro;
  ASSERT_EQ(llvm::Triple((Twine("x86_64-apple-macos") + SystemVersion))
                .getMacOSXVersion(SystemMajor, SystemMinor, SystemMicro),
            true);
  unsigned HostMajor, HostMinor, HostMicro;
  ASSERT_EQ(HostTriple.getMacOSXVersion(HostMajor, HostMinor, HostMicro), true);

  if (SystemMajor > 10) {
    // Don't compare the 'Minor' and 'Micro' versions, as they're always '0' for
    // the 'Darwin' triples on 11.x.
    ASSERT_EQ(SystemMajor, HostMajor);
  } else {
    // Don't compare the 'Micro' version, as it's always '0' for the 'Darwin'
    // triples.
    ASSERT_EQ(std::tie(SystemMajor, SystemMinor), std::tie(HostMajor, HostMinor));
  }
}
#endif

#if defined(_AIX)
TEST_F(HostTest, AIXVersionDetect) {
  using namespace llvm::sys;

  llvm::Triple HostTriple(getProcessTriple());
  ASSERT_EQ(HostTriple.getOS(), Triple::AIX);

  llvm::Triple ConfiguredHostTriple(LLVM_HOST_TRIPLE);
  ASSERT_EQ(ConfiguredHostTriple.getOS(), Triple::AIX);

  const char *ExePath = "/usr/bin/oslevel";
  StringRef argv[] = {ExePath};
  std::unique_ptr<char[]> Buffer;
  off_t Size;
  ASSERT_EQ(runAndGetCommandOutput(ExePath, argv, Buffer, Size), true);
  StringRef SystemVersion(Buffer.get(), Size);

  unsigned SystemMajor, SystemMinor, SystemMicro;
  llvm::Triple((Twine("powerpc-ibm-aix") + SystemVersion))
      .getOSVersion(SystemMajor, SystemMinor, SystemMicro);

  // Ensure that the host triple version (major) and release (minor) numbers,
  // unless explicitly configured, match with those of the current system.
  if (!ConfiguredHostTriple.getOSMajorVersion()) {
    unsigned HostMajor, HostMinor, HostMicro;
    HostTriple.getOSVersion(HostMajor, HostMinor, HostMicro);
    ASSERT_EQ(std::tie(SystemMajor, SystemMinor),
              std::tie(HostMajor, HostMinor));
  }

  llvm::Triple TargetTriple(getDefaultTargetTriple());
  if (TargetTriple.getOS() != Triple::AIX)
    return;

  // Ensure that the target triple version (major) and release (minor) numbers
  // match with those of the current system.
  llvm::Triple ConfiguredTargetTriple(LLVM_DEFAULT_TARGET_TRIPLE);
  if (ConfiguredTargetTriple.getOSMajorVersion())
    return; // The version was configured explicitly; skip.

  unsigned TargetMajor, TargetMinor, TargetMicro;
  TargetTriple.getOSVersion(TargetMajor, TargetMinor, TargetMicro);
  ASSERT_EQ(std::tie(SystemMajor, SystemMinor),
            std::tie(TargetMajor, TargetMinor));
}

TEST_F(HostTest, AIXHostCPUDetect) {
  // Return a value based on the current processor implementation mode.
  const char *ExePath = "/usr/sbin/getsystype";
  StringRef argv[] = {ExePath, "-i"};
  std::unique_ptr<char[]> Buffer;
  off_t Size;
  ASSERT_EQ(runAndGetCommandOutput(ExePath, argv, Buffer, Size), true);
  StringRef CPU(Buffer.get(), Size);
  StringRef MCPU = StringSwitch<const char *>(CPU)
                       .Case("POWER 4\n", "pwr4")
                       .Case("POWER 5\n", "pwr5")
                       .Case("POWER 6\n", "pwr6")
                       .Case("POWER 7\n", "pwr7")
                       .Case("POWER 8\n", "pwr8")
                       .Case("POWER 9\n", "pwr9")
                       .Case("POWER 10\n", "pwr10")
                       .Default("unknown");

  StringRef HostCPU = sys::getHostCPUName();

  // Just do the comparison on the base implementation mode.
  if (HostCPU == "970")
    HostCPU = StringRef("pwr4");
  else
    HostCPU = HostCPU.rtrim('x');

  EXPECT_EQ(HostCPU, MCPU);
}
#endif
