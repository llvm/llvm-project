//===- MinidumpYAMLTest.cpp - Tests for Minidump<->YAML code --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Minidump.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::minidump;

static Expected<std::unique_ptr<object::MinidumpFile>>
toBinary(SmallVectorImpl<char> &Storage, StringRef Yaml) {
  Storage.clear();
  raw_svector_ostream OS(Storage);
  yaml::Input YIn(Yaml);
  if (!yaml::convertYAML(YIn, OS, [](const Twine &Msg) {}))
    return createStringError(std::errc::invalid_argument,
                             "unable to convert YAML");

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
      ArrayRef(SysInfo.CPU.Other.ProcessorFeatures));
}

// Test that we can parse a normal-looking ExceptionStream.
TEST(MinidumpYAML, ExceptionStream) {
  SmallString<0> Storage;
  auto ExpectedFile = toBinary(Storage, R"(
--- !minidump
Streams:
  - Type:            Exception
    Thread ID:  0x7
    Exception Record:
      Exception Code:  0x23
      Exception Flags: 0x5
      Exception Record: 0x0102030405060708
      Exception Address: 0x0a0b0c0d0e0f1011
      Number of Parameters: 2
      Parameter 0: 0x22
      Parameter 1: 0x24
    Thread Context:  3DeadBeefDefacedABadCafe)");
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  object::MinidumpFile &File = **ExpectedFile;

  ASSERT_EQ(1u, File.streams().size());

  auto ExceptionIterator = File.getExceptionStreams().begin();

  Expected<const ExceptionStream &> ExpectedStream = *ExceptionIterator;

  ASSERT_THAT_EXPECTED(ExpectedStream, Succeeded());

  const minidump::ExceptionStream &Stream = *ExpectedStream;
  EXPECT_EQ(0x7u, Stream.ThreadId);
  const minidump::Exception &Exception = Stream.ExceptionRecord;
  EXPECT_EQ(0x23u, Exception.ExceptionCode);
  EXPECT_EQ(0x5u, Exception.ExceptionFlags);
  EXPECT_EQ(0x0102030405060708u, Exception.ExceptionRecord);
  EXPECT_EQ(0x0a0b0c0d0e0f1011u, Exception.ExceptionAddress);
  EXPECT_EQ(2u, Exception.NumberParameters);
  EXPECT_EQ(0x22u, Exception.ExceptionInformation[0]);
  EXPECT_EQ(0x24u, Exception.ExceptionInformation[1]);

  Expected<ArrayRef<uint8_t>> ExpectedContext =
      File.getRawData(Stream.ThreadContext);
  ASSERT_THAT_EXPECTED(ExpectedContext, Succeeded());
  EXPECT_EQ((ArrayRef<uint8_t>{0x3d, 0xea, 0xdb, 0xee, 0xfd, 0xef, 0xac, 0xed,
                               0xab, 0xad, 0xca, 0xfe}),
            *ExpectedContext);
}

// Test that we can parse an exception stream with no ExceptionInformation.
TEST(MinidumpYAML, ExceptionStream_NoParameters) {
  SmallString<0> Storage;
  auto ExpectedFile = toBinary(Storage, R"(
--- !minidump
Streams:
  - Type:            Exception
    Thread ID:  0x7
    Exception Record:
      Exception Code:  0x23
      Exception Flags: 0x5
      Exception Record: 0x0102030405060708
      Exception Address: 0x0a0b0c0d0e0f1011
    Thread Context:  3DeadBeefDefacedABadCafe)");
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  object::MinidumpFile &File = **ExpectedFile;

  ASSERT_EQ(1u, File.streams().size());

  auto ExceptionIterator = File.getExceptionStreams().begin();

  Expected<const ExceptionStream &> ExpectedStream = *ExceptionIterator;
  ASSERT_THAT_EXPECTED(ExpectedStream, Succeeded());

  const minidump::ExceptionStream &Stream = *ExpectedStream;
  EXPECT_EQ(0x7u, Stream.ThreadId);
  const minidump::Exception &Exception = Stream.ExceptionRecord;
  EXPECT_EQ(0x23u, Exception.ExceptionCode);
  EXPECT_EQ(0x5u, Exception.ExceptionFlags);
  EXPECT_EQ(0x0102030405060708u, Exception.ExceptionRecord);
  EXPECT_EQ(0x0a0b0c0d0e0f1011u, Exception.ExceptionAddress);
  EXPECT_EQ(0u, Exception.NumberParameters);

  Expected<ArrayRef<uint8_t>> ExpectedContext =
      File.getRawData(Stream.ThreadContext);
  ASSERT_THAT_EXPECTED(ExpectedContext, Succeeded());
  EXPECT_EQ((ArrayRef<uint8_t>{0x3d, 0xea, 0xdb, 0xee, 0xfd, 0xef, 0xac, 0xed,
                               0xab, 0xad, 0xca, 0xfe}),
            *ExpectedContext);
}

// Test that we can parse an ExceptionStream where the stated number of
// parameters is greater than the actual size of the ExceptionInformation
// array.
TEST(MinidumpYAML, ExceptionStream_TooManyParameters) {
  SmallString<0> Storage;
  auto ExpectedFile = toBinary(Storage, R"(
--- !minidump
Streams:
  - Type:            Exception
    Thread ID:  0x8
    Exception Record:
      Exception Code: 0
      Number of Parameters: 16
      Parameter 0: 0x0
      Parameter 1: 0xff
      Parameter 2: 0xee
      Parameter 3: 0xdd
      Parameter 4: 0xcc
      Parameter 5: 0xbb
      Parameter 6: 0xaa
      Parameter 7: 0x99
      Parameter 8: 0x88
      Parameter 9: 0x77
      Parameter 10: 0x66
      Parameter 11: 0x55
      Parameter 12: 0x44
      Parameter 13: 0x33
      Parameter 14: 0x22
    Thread Context:  3DeadBeefDefacedABadCafe)");
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  object::MinidumpFile &File = **ExpectedFile;

  ASSERT_EQ(1u, File.streams().size());

  auto ExceptionIterator = File.getExceptionStreams().begin();

  Expected<const ExceptionStream &> ExpectedStream = *ExceptionIterator;

  ASSERT_THAT_EXPECTED(ExpectedStream, Succeeded());

  const minidump::ExceptionStream &Stream = *ExpectedStream;
  EXPECT_EQ(0x8u, Stream.ThreadId);
  const minidump::Exception &Exception = Stream.ExceptionRecord;
  EXPECT_EQ(0x0u, Exception.ExceptionCode);
  EXPECT_EQ(0x0u, Exception.ExceptionFlags);
  EXPECT_EQ(0x00u, Exception.ExceptionRecord);
  EXPECT_EQ(0x0u, Exception.ExceptionAddress);
  EXPECT_EQ(16u, Exception.NumberParameters);
  EXPECT_EQ(0x0u, Exception.ExceptionInformation[0]);
  for (int Index = 1; Index < 15; ++Index) {
    EXPECT_EQ(0x110u - Index * 0x11, Exception.ExceptionInformation[Index]);
  }

  Expected<ArrayRef<uint8_t>> ExpectedContext =
      File.getRawData(Stream.ThreadContext);
  ASSERT_THAT_EXPECTED(ExpectedContext, Succeeded());
  EXPECT_EQ((ArrayRef<uint8_t>{0x3d, 0xea, 0xdb, 0xee, 0xfd, 0xef, 0xac, 0xed,
                               0xab, 0xad, 0xca, 0xfe}),
            *ExpectedContext);
}

// Test that we can parse an ExceptionStream where the number of
// ExceptionInformation parameters provided is greater than the
// specified Number of Parameters.
TEST(MinidumpYAML, ExceptionStream_ExtraParameter) {
  SmallString<0> Storage;
  auto ExpectedFile = toBinary(Storage, R"(
--- !minidump
Streams:
  - Type:            Exception
    Thread ID:  0x7
    Exception Record:
      Exception Code:  0x23
      Exception Flags: 0x5
      Exception Record: 0x0102030405060708
      Exception Address: 0x0a0b0c0d0e0f1011
      Number of Parameters: 2
      Parameter 0: 0x99
      Parameter 1: 0x23
      Parameter 2: 0x42
    Thread Context:  3DeadBeefDefacedABadCafe)");
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  object::MinidumpFile &File = **ExpectedFile;

  ASSERT_EQ(1u, File.streams().size());

  auto ExceptionIterator = File.getExceptionStreams().begin();

  Expected<const ExceptionStream &> ExpectedStream = *ExceptionIterator;

  ASSERT_THAT_EXPECTED(ExpectedStream, Succeeded());

  const minidump::ExceptionStream &Stream = *ExpectedStream;
  EXPECT_EQ(0x7u, Stream.ThreadId);
  const minidump::Exception &Exception = Stream.ExceptionRecord;
  EXPECT_EQ(0x23u, Exception.ExceptionCode);
  EXPECT_EQ(0x5u, Exception.ExceptionFlags);
  EXPECT_EQ(0x0102030405060708u, Exception.ExceptionRecord);
  EXPECT_EQ(0x0a0b0c0d0e0f1011u, Exception.ExceptionAddress);
  EXPECT_EQ(2u, Exception.NumberParameters);
  EXPECT_EQ(0x99u, Exception.ExceptionInformation[0]);
  EXPECT_EQ(0x23u, Exception.ExceptionInformation[1]);
  EXPECT_EQ(0x42u, Exception.ExceptionInformation[2]);

  Expected<ArrayRef<uint8_t>> ExpectedContext =
      File.getRawData(Stream.ThreadContext);
  ASSERT_THAT_EXPECTED(ExpectedContext, Succeeded());
  EXPECT_EQ((ArrayRef<uint8_t>{0x3d, 0xea, 0xdb, 0xee, 0xfd, 0xef, 0xac, 0xed,
                               0xab, 0xad, 0xca, 0xfe}),
            *ExpectedContext);
}

TEST(MinidumpYAML, MemoryRegion_64bit) {
  SmallString<0> Storage;
  auto ExpectedFile = toBinary(Storage, R"(
--- !minidump
Streams:
  - Type:            Memory64List
    Memory Ranges:
      - Start of Memory Range: 0x7FFFFFCF0818283
        Content:               '68656c6c6f'
      - Start of Memory Range: 0x7FFFFFFF0818283
        Content:               '776f726c64'
        )");

  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  object::MinidumpFile &File = **ExpectedFile;

  ASSERT_THAT(File.streams().size(), 1u);

  Error Err = Error::success();
  iterator_range<object::MinidumpFile::FallibleMemory64Iterator> MemoryList =
      File.getMemory64List(Err);

  ASSERT_THAT_ERROR(std::move(Err), Succeeded());
  auto Iterator = MemoryList.begin();

  auto DescOnePair = *Iterator;
  const minidump::MemoryDescriptor_64 &DescOne = DescOnePair.first;
  ASSERT_THAT(DescOne.StartOfMemoryRange, 0x7FFFFFCF0818283u);
  ASSERT_THAT(DescOne.DataSize, 5u);
  ++Iterator;
  ASSERT_THAT_ERROR(std::move(Err), Succeeded());

  auto DescTwoPair = *Iterator;
  const minidump::MemoryDescriptor_64 &DescTwo = DescTwoPair.first;
  ASSERT_THAT(DescTwo.StartOfMemoryRange, 0x7FFFFFFF0818283u);
  ASSERT_THAT(DescTwo.DataSize, 5u);
  ++Iterator;
  ASSERT_THAT_ERROR(std::move(Err), Succeeded());

  const std::optional<ArrayRef<uint8_t>> ExpectedContent =
      File.getRawStream(StreamType::Memory64List);
  ASSERT_TRUE(ExpectedContent);
  const size_t ExpectedStreamSize =
      sizeof(Memory64ListHeader) + (sizeof(MemoryDescriptor_64) * 2);
  ASSERT_THAT(ExpectedContent->size(), ExpectedStreamSize);

  Expected<minidump::Memory64ListHeader> ExpectedHeader =
      File.getMemoryList64Header();
  ASSERT_THAT_EXPECTED(ExpectedHeader, Succeeded());
  ASSERT_THAT(ExpectedHeader->BaseRVA, 92u);

  Expected<ArrayRef<uint8_t>> DescOneExpectedContentSlice = DescOnePair.second;
  ASSERT_THAT_EXPECTED(DescOneExpectedContentSlice, Succeeded());
  ASSERT_THAT(DescOneExpectedContentSlice->size(), 5u);
  ASSERT_THAT(*DescOneExpectedContentSlice, arrayRefFromStringRef("hello"));

  Expected<ArrayRef<uint8_t>> DescTwoExpectedContentSlice = DescTwoPair.second;
  ASSERT_THAT_EXPECTED(DescTwoExpectedContentSlice, Succeeded());
  ASSERT_THAT(*DescTwoExpectedContentSlice, arrayRefFromStringRef("world"));

  ASSERT_EQ(Iterator, MemoryList.end());
}

// Test that we can parse multiple exception streams.
TEST(MinidumpYAML, ExceptionStream_MultipleExceptions) {
  SmallString<0> Storage;
  auto ExpectedFile = toBinary(Storage, R"(
--- !minidump
Streams:
  - Type:            Exception
    Thread ID:  0x7
    Exception Record:
      Exception Code:  0x23
      Exception Flags: 0x5
      Exception Record: 0x0102030405060708
      Exception Address: 0x0a0b0c0d0e0f1011
      Number of Parameters: 2
      Parameter 0: 0x99
      Parameter 1: 0x23
      Parameter 2: 0x42
    Thread Context:  3DeadBeefDefacedABadCafe
  - Type:            Exception
    Thread ID:  0x5
    Exception Record:
      Exception Code:  0x23
      Exception Flags: 0x5
      Exception Record: 0x0102030405060708
      Exception Address: 0x0a0b0c0d0e0f1011
    Thread Context:  3DeadBeefDefacedABadCafe)");

  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  object::MinidumpFile &File = **ExpectedFile;

  ASSERT_EQ(2u, File.streams().size());

  size_t count = 0;
  for (auto exception_stream : File.getExceptionStreams()) {
    count++;
    ASSERT_THAT_EXPECTED(exception_stream, Succeeded());
    ASSERT_THAT(0x23u, exception_stream->ExceptionRecord.ExceptionCode);
  }

  ASSERT_THAT(2u, count);
}
