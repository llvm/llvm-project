//===- MinidumpYAMLTest.cpp - Tests for Minidump<->YAML code --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/MinidumpYAML.h"
#include "llvm/Object/Minidump.h"
#include "llvm/ObjectYAML/ObjectYAML.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::minidump;

static Expected<std::unique_ptr<object::MinidumpFile>>
toBinary(SmallVectorImpl<char> &Storage, StringRef Yaml) {
  Storage.clear();
  raw_svector_ostream OS(Storage);
  if (Error E = MinidumpYAML::writeAsBinary(Yaml, OS))
    return std::move(E);

  return object::MinidumpFile::create(MemoryBufferRef(OS.str(), "Binary"));
}

TEST(MinidumpYAML, Basic) {
  SmallString<0> Storage;
  auto ExpectedFile = toBinary(Storage, R"(
--- !minidump
Streams:
  - Type:            SystemInfo
    Processor Arch:  ARM64
    Platform ID:     Linux
    CPU:
      CPUID:           0x05060708
  - Type:            LinuxMaps
    Text:             |
      400d9000-400db000 r-xp 00000000 b3:04 227        /system/bin/app_process
      400db000-400dc000 r--p 00001000 b3:04 227        /system/bin/app_process

  - Type:            LinuxAuxv
    Content:         DEADBEEFBAADF00D)");
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  object::MinidumpFile &File = **ExpectedFile;

  ASSERT_EQ(3u, File.streams().size());

  EXPECT_EQ(StreamType::SystemInfo, File.streams()[0].Type);
  auto ExpectedSysInfo = File.getSystemInfo();
  ASSERT_THAT_EXPECTED(ExpectedSysInfo, Succeeded());
  const SystemInfo &SysInfo = *ExpectedSysInfo;
  EXPECT_EQ(ProcessorArchitecture::ARM64, SysInfo.ProcessorArch);
  EXPECT_EQ(OSPlatform::Linux, SysInfo.PlatformId);
  EXPECT_EQ(0x05060708u, SysInfo.CPU.Arm.CPUID);

  EXPECT_EQ(StreamType::LinuxMaps, File.streams()[1].Type);
  EXPECT_EQ("400d9000-400db000 r-xp 00000000 b3:04 227        "
            "/system/bin/app_process\n"
            "400db000-400dc000 r--p 00001000 b3:04 227        "
            "/system/bin/app_process\n",
            toStringRef(*File.getRawStream(StreamType::LinuxMaps)));

  EXPECT_EQ(StreamType::LinuxAuxv, File.streams()[2].Type);
  EXPECT_EQ((ArrayRef<uint8_t>{0xDE, 0xAD, 0xBE, 0xEF, 0xBA, 0xAD, 0xF0, 0x0D}),
            File.getRawStream(StreamType::LinuxAuxv));
}

TEST(MinidumpYAML, RawContent) {
  SmallString<0> Storage;
  auto ExpectedFile = toBinary(Storage, R"(
--- !minidump
Streams:
  - Type:            LinuxAuxv
    Size:            9
    Content:         DEADBEEFBAADF00D)");
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  object::MinidumpFile &File = **ExpectedFile;

  EXPECT_EQ(
      (ArrayRef<uint8_t>{0xDE, 0xAD, 0xBE, 0xEF, 0xBA, 0xAD, 0xF0, 0x0D, 0x00}),
      File.getRawStream(StreamType::LinuxAuxv));
}

TEST(MinidumpYAML, X86SystemInfo) {
  SmallString<0> Storage;
  auto ExpectedFile = toBinary(Storage, R"(
--- !minidump
Streams:
  - Type:            SystemInfo
    Processor Arch:  X86
    Platform ID:     Linux
    CPU:
      Vendor ID:       LLVMLLVMLLVM
      Version Info:    0x01020304
      Feature Info:    0x05060708
      AMD Extended Features: 0x09000102)");
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  object::MinidumpFile &File = **ExpectedFile;

  ASSERT_EQ(1u, File.streams().size());

  auto ExpectedSysInfo = File.getSystemInfo();
  ASSERT_THAT_EXPECTED(ExpectedSysInfo, Succeeded());
  const SystemInfo &SysInfo = *ExpectedSysInfo;
  EXPECT_EQ(ProcessorArchitecture::X86, SysInfo.ProcessorArch);
  EXPECT_EQ(OSPlatform::Linux, SysInfo.PlatformId);
  EXPECT_EQ("LLVMLLVMLLVM", StringRef(SysInfo.CPU.X86.VendorID,
                                      sizeof(SysInfo.CPU.X86.VendorID)));
  EXPECT_EQ(0x01020304u, SysInfo.CPU.X86.VersionInfo);
  EXPECT_EQ(0x05060708u, SysInfo.CPU.X86.FeatureInfo);
  EXPECT_EQ(0x09000102u, SysInfo.CPU.X86.AMDExtendedFeatures);
}

TEST(MinidumpYAML, OtherSystemInfo) {
  SmallString<0> Storage;
  auto ExpectedFile = toBinary(Storage, R"(
--- !minidump
Streams:
  - Type:            SystemInfo
    Processor Arch:  PPC
    Platform ID:     Linux
    CPU:
      Features:        000102030405060708090a0b0c0d0e0f)");
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  object::MinidumpFile &File = **ExpectedFile;

  ASSERT_EQ(1u, File.streams().size());

  auto ExpectedSysInfo = File.getSystemInfo();
  ASSERT_THAT_EXPECTED(ExpectedSysInfo, Succeeded());
  const SystemInfo &SysInfo = *ExpectedSysInfo;
  EXPECT_EQ(ProcessorArchitecture::PPC, SysInfo.ProcessorArch);
  EXPECT_EQ(OSPlatform::Linux, SysInfo.PlatformId);
  EXPECT_EQ(
      (ArrayRef<uint8_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
      makeArrayRef(SysInfo.CPU.Other.ProcessorFeatures));
}
