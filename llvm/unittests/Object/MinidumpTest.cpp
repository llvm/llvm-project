//===- MinidumpTest.cpp - Tests for Minidump.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Minidump.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;
using namespace minidump;

static Expected<std::unique_ptr<MinidumpFile>> create(ArrayRef<uint8_t> Data) {
  return MinidumpFile::create(
      MemoryBufferRef(toStringRef(Data), "Test buffer"));
}

TEST(MinidumpFile, BasicInterface) {
  std::vector<uint8_t> Data{                        // Header
                            'M', 'D', 'M', 'P',     // Signature
                            0x93, 0xa7, 0, 0,       // Version
                            1, 0, 0, 0,             // NumberOfStreams,
                            0x20, 0, 0, 0,          // StreamDirectoryRVA
                            0, 1, 2, 3, 4, 5, 6, 7, // Checksum, TimeDateStamp
                            8, 9, 0, 1, 2, 3, 4, 5, // Flags
                                                    // Stream Directory
                            3, 0, 0x67, 0x47, 7, 0, 0, 0, // Type, DataSize,
                            0x2c, 0, 0, 0,                // RVA
                                                          // Stream
                            'C', 'P', 'U', 'I', 'N', 'F', 'O'};
  // A very simple minidump file which contains just a single stream.
  auto ExpectedFile = create(Data);
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  const MinidumpFile &File = **ExpectedFile;
  const Header &H = File.header();
  EXPECT_EQ(Header::MagicSignature, H.Signature);
  EXPECT_EQ(Header::MagicVersion, H.Version);
  EXPECT_EQ(1u, H.NumberOfStreams);
  EXPECT_EQ(0x20u, H.StreamDirectoryRVA);
  EXPECT_EQ(0x03020100u, H.Checksum);
  EXPECT_EQ(0x07060504u, H.TimeDateStamp);
  EXPECT_EQ(uint64_t(0x0504030201000908), H.Flags);

  ASSERT_EQ(1u, File.streams().size());
  const Directory &Stream0 = File.streams()[0];
  EXPECT_EQ(StreamType::LinuxCPUInfo, Stream0.Type);
  EXPECT_EQ(7u, Stream0.Location.DataSize);
  EXPECT_EQ(0x2cu, Stream0.Location.RVA);

  EXPECT_EQ("CPUINFO", toStringRef(File.getRawStream(Stream0)));
  EXPECT_EQ("CPUINFO",
            toStringRef(*File.getRawStream(StreamType::LinuxCPUInfo)));

  EXPECT_THAT_EXPECTED(File.getSystemInfo(), Failed<BinaryError>());
}

// Use the input from the previous test, but corrupt it in various ways
TEST(MinidumpFile, create_ErrorCases) {
  std::vector<uint8_t> FileTooShort{'M', 'D', 'M', 'P'};
  EXPECT_THAT_EXPECTED(create(FileTooShort), Failed<BinaryError>());

  std::vector<uint8_t> WrongSignature{
      // Header
      '!', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      0x20, 0, 0, 0,                        // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      8, 9, 0, 1, 2, 3, 4, 5,               // Flags
                                            // Stream Directory
      3, 0, 0x67, 0x47, 7, 0, 0, 0,         // Type, DataSize,
      0x2c, 0, 0, 0,                        // RVA
                                            // Stream
      'C', 'P', 'U', 'I', 'N', 'F', 'O'};
  EXPECT_THAT_EXPECTED(create(WrongSignature), Failed<BinaryError>());

  std::vector<uint8_t> WrongVersion{
      // Header
      'M', 'D', 'M', 'P', 0x39, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      0x20, 0, 0, 0,                        // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      8, 9, 0, 1, 2, 3, 4, 5,               // Flags
                                            // Stream Directory
      3, 0, 0x67, 0x47, 7, 0, 0, 0,         // Type, DataSize,
      0x2c, 0, 0, 0,                        // RVA
                                            // Stream
      'C', 'P', 'U', 'I', 'N', 'F', 'O'};
  EXPECT_THAT_EXPECTED(create(WrongVersion), Failed<BinaryError>());

  std::vector<uint8_t> DirectoryAfterEOF{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      0x20, 1, 0, 0,                        // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      8, 9, 0, 1, 2, 3, 4, 5,               // Flags
                                            // Stream Directory
      3, 0, 0x67, 0x47, 7, 0, 0, 0,         // Type, DataSize,
      0x2c, 0, 0, 0,                        // RVA
                                            // Stream
      'C', 'P', 'U', 'I', 'N', 'F', 'O'};
  EXPECT_THAT_EXPECTED(create(DirectoryAfterEOF), Failed<BinaryError>());

  std::vector<uint8_t> TruncatedDirectory{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 1, 0, 0,                           // NumberOfStreams,
      0x20, 0, 0, 0,                        // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      8, 9, 0, 1, 2, 3, 4, 5,               // Flags
                                            // Stream Directory
      3, 0, 0x67, 0x47, 7, 0, 0, 0,         // Type, DataSize,
      0x2c, 0, 0, 0,                        // RVA
                                            // Stream
      'C', 'P', 'U', 'I', 'N', 'F', 'O'};
  EXPECT_THAT_EXPECTED(create(TruncatedDirectory), Failed<BinaryError>());

  std::vector<uint8_t> Stream0AfterEOF{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      0x20, 0, 0, 0,                        // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      8, 9, 0, 1, 2, 3, 4, 5,               // Flags
                                            // Stream Directory
      3, 0, 0x67, 0x47, 7, 0, 0, 0,         // Type, DataSize,
      0x2c, 1, 0, 0,                        // RVA
                                            // Stream
      'C', 'P', 'U', 'I', 'N', 'F', 'O'};
  EXPECT_THAT_EXPECTED(create(Stream0AfterEOF), Failed<BinaryError>());

  std::vector<uint8_t> Stream0Truncated{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      0x20, 0, 0, 0,                        // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      8, 9, 0, 1, 2, 3, 4, 5,               // Flags
                                            // Stream Directory
      3, 0, 0x67, 0x47, 8, 0, 0, 0,         // Type, DataSize,
      0x2c, 0, 0, 0,                        // RVA
                                            // Stream
      'C', 'P', 'U', 'I', 'N', 'F', 'O'};
  EXPECT_THAT_EXPECTED(create(Stream0Truncated), Failed<BinaryError>());

  std::vector<uint8_t> DuplicateStream{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      2, 0, 0, 0,                           // NumberOfStreams,
      0x20, 0, 0, 0,                        // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      8, 9, 0, 1, 2, 3, 4, 5,               // Flags
                                            // Stream Directory
      3, 0, 0x67, 0x47, 7, 0, 0, 0,         // Type, DataSize,
      0x40, 0, 0, 0,                        // RVA
                                            // Stream
      3, 0, 0x67, 0x47, 7, 0, 0, 0,         // Type, DataSize,
      0x40, 0, 0, 0,                        // RVA
                                            // Stream
      'C', 'P', 'U', 'I', 'N', 'F', 'O'};
  EXPECT_THAT_EXPECTED(create(DuplicateStream), Failed<BinaryError>());

  std::vector<uint8_t> DenseMapInfoConflict{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      0x20, 0, 0, 0,                        // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      8, 9, 0, 1, 2, 3, 4, 5,               // Flags
                                            // Stream Directory
      0xff, 0xff, 0xff, 0xff, 7, 0, 0, 0,   // Type, DataSize,
      0x2c, 0, 0, 0,                        // RVA
                                            // Stream
      'C', 'P', 'U', 'I', 'N', 'F', 'O'};
  EXPECT_THAT_EXPECTED(create(DenseMapInfoConflict), Failed<BinaryError>());
}

TEST(MinidumpFile, IngoresDummyStreams) {
  std::vector<uint8_t> TwoDummyStreams{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      2, 0, 0, 0,                           // NumberOfStreams,
      0x20, 0, 0, 0,                        // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      8, 9, 0, 1, 2, 3, 4, 5,               // Flags
                                            // Stream Directory
      0, 0, 0, 0, 0, 0, 0, 0,               // Type, DataSize,
      0x20, 0, 0, 0,                        // RVA
      0, 0, 0, 0, 0, 0, 0, 0,               // Type, DataSize,
      0x20, 0, 0, 0,                        // RVA
  };
  auto ExpectedFile = create(TwoDummyStreams);
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  const MinidumpFile &File = **ExpectedFile;
  ASSERT_EQ(2u, File.streams().size());
  EXPECT_EQ(StreamType::Unused, File.streams()[0].Type);
  EXPECT_EQ(StreamType::Unused, File.streams()[1].Type);
  EXPECT_EQ(std::nullopt, File.getRawStream(StreamType::Unused));
}

TEST(MinidumpFile, getSystemInfo) {
  std::vector<uint8_t> Data{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      0x20, 0, 0, 0,                        // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      8, 9, 0, 1, 2, 3, 4, 5,               // Flags
                                            // Stream Directory
      7, 0, 0, 0, 56, 0, 0, 0,              // Type, DataSize,
      0x2c, 0, 0, 0,                        // RVA
                                            // SystemInfo
      0, 0, 1, 2,                           // ProcessorArch, ProcessorLevel
      3, 4, 5, 6, // ProcessorRevision, NumberOfProcessors, ProductType
      7, 8, 9, 0, 1, 2, 3, 4, // MajorVersion, MinorVersion
      5, 6, 7, 8, 2, 0, 0, 0, // BuildNumber, PlatformId
      1, 2, 3, 4, 5, 6, 7, 8, // CSDVersionRVA, SuiteMask, Reserved
      'L', 'L', 'V', 'M', 'L', 'L', 'V', 'M', 'L', 'L', 'V', 'M', // VendorID
      1, 2, 3, 4, 5, 6, 7, 8, // VersionInfo, FeatureInfo
      9, 0, 1, 2,             // AMDExtendedFeatures
  };
  auto ExpectedFile = create(Data);
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  const MinidumpFile &File = **ExpectedFile;

  auto ExpectedInfo = File.getSystemInfo();
  ASSERT_THAT_EXPECTED(ExpectedInfo, Succeeded());
  const SystemInfo &Info = *ExpectedInfo;
  EXPECT_EQ(ProcessorArchitecture::X86, Info.ProcessorArch);
  EXPECT_EQ(0x0201, Info.ProcessorLevel);
  EXPECT_EQ(0x0403, Info.ProcessorRevision);
  EXPECT_EQ(5, Info.NumberOfProcessors);
  EXPECT_EQ(6, Info.ProductType);
  EXPECT_EQ(0x00090807u, Info.MajorVersion);
  EXPECT_EQ(0x04030201u, Info.MinorVersion);
  EXPECT_EQ(0x08070605u, Info.BuildNumber);
  EXPECT_EQ(OSPlatform::Win32NT, Info.PlatformId);
  EXPECT_EQ(0x04030201u, Info.CSDVersionRVA);
  EXPECT_EQ(0x0605u, Info.SuiteMask);
  EXPECT_EQ(0x0807u, Info.Reserved);
  EXPECT_EQ("LLVMLLVMLLVM", llvm::StringRef(Info.CPU.X86.VendorID,
                                            sizeof(Info.CPU.X86.VendorID)));
  EXPECT_EQ(0x04030201u, Info.CPU.X86.VersionInfo);
  EXPECT_EQ(0x08070605u, Info.CPU.X86.FeatureInfo);
  EXPECT_EQ(0x02010009u, Info.CPU.X86.AMDExtendedFeatures);
}

TEST(MinidumpFile, getString) {
  std::vector<uint8_t> ManyStrings{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      2, 0, 0, 0,                           // NumberOfStreams,
      0x20, 0, 0, 0,                        // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      8, 9, 0, 1, 2, 3, 4, 5,               // Flags
                                            // Stream Directory
      0, 0, 0, 0, 0, 0, 0, 0,               // Type, DataSize,
      0x20, 0, 0, 0,                        // RVA
      1, 0, 0, 0, 0, 0,                     // String1 - odd length
      0, 0, 1, 0, 0, 0,                     // String2 - too long
      2, 0, 0, 0, 0, 0xd8,                  // String3 - invalid utf16
      0, 0, 0, 0, 0, 0,                     // String4 - ""
      2, 0, 0, 0, 'a', 0,                   // String5 - "a"
      0,                                    // Mis-align next string
      2, 0, 0, 0, 'a', 0,                   // String6 - "a"

  };
  auto ExpectedFile = create(ManyStrings);
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  const MinidumpFile &File = **ExpectedFile;
  EXPECT_THAT_EXPECTED(File.getString(44), Failed<BinaryError>());
  EXPECT_THAT_EXPECTED(File.getString(50), Failed<BinaryError>());
  EXPECT_THAT_EXPECTED(File.getString(56), Failed<BinaryError>());
  EXPECT_THAT_EXPECTED(File.getString(62), HasValue(""));
  EXPECT_THAT_EXPECTED(File.getString(68), HasValue("a"));
  EXPECT_THAT_EXPECTED(File.getString(75), HasValue("a"));

  // Check the case when the size field does not fit into the remaining data.
  EXPECT_THAT_EXPECTED(File.getString(ManyStrings.size() - 2),
                       Failed<BinaryError>());
}

TEST(MinidumpFile, getModuleList) {
  std::vector<uint8_t> OneModule{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      4, 0, 0, 0, 112, 0, 0, 0,             // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // ModuleList
      1, 0, 0, 0,             // NumberOfModules
      1, 2, 3, 4, 5, 6, 7, 8, // BaseOfImage
      9, 0, 1, 2, 3, 4, 5, 6, // SizeOfImage, Checksum
      7, 8, 9, 0, 1, 2, 3, 4, // TimeDateStamp, ModuleNameRVA
      0, 0, 0, 0, 0, 0, 0, 0, // Signature, StructVersion
      0, 0, 0, 0, 0, 0, 0, 0, // FileVersion
      0, 0, 0, 0, 0, 0, 0, 0, // ProductVersion
      0, 0, 0, 0, 0, 0, 0, 0, // FileFlagsMask, FileFlags
      0, 0, 0, 0,             // FileOS
      0, 0, 0, 0, 0, 0, 0, 0, // FileType, FileSubType
      0, 0, 0, 0, 0, 0, 0, 0, // FileDate
      1, 2, 3, 4, 5, 6, 7, 8, // CvRecord
      9, 0, 1, 2, 3, 4, 5, 6, // MiscRecord
      7, 8, 9, 0, 1, 2, 3, 4, // Reserved0
      5, 6, 7, 8, 9, 0, 1, 2, // Reserved1
  };
  // Same as before, but with a padded module list.
  std::vector<uint8_t> PaddedModule{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      4, 0, 0, 0, 116, 0, 0, 0,             // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // ModuleList
      1, 0, 0, 0,             // NumberOfModules
      0, 0, 0, 0,             // Padding
      1, 2, 3, 4, 5, 6, 7, 8, // BaseOfImage
      9, 0, 1, 2, 3, 4, 5, 6, // SizeOfImage, Checksum
      7, 8, 9, 0, 1, 2, 3, 4, // TimeDateStamp, ModuleNameRVA
      0, 0, 0, 0, 0, 0, 0, 0, // Signature, StructVersion
      0, 0, 0, 0, 0, 0, 0, 0, // FileVersion
      0, 0, 0, 0, 0, 0, 0, 0, // ProductVersion
      0, 0, 0, 0, 0, 0, 0, 0, // FileFlagsMask, FileFlags
      0, 0, 0, 0,             // FileOS
      0, 0, 0, 0, 0, 0, 0, 0, // FileType, FileSubType
      0, 0, 0, 0, 0, 0, 0, 0, // FileDate
      1, 2, 3, 4, 5, 6, 7, 8, // CvRecord
      9, 0, 1, 2, 3, 4, 5, 6, // MiscRecord
      7, 8, 9, 0, 1, 2, 3, 4, // Reserved0
      5, 6, 7, 8, 9, 0, 1, 2, // Reserved1
  };

  for (ArrayRef<uint8_t> Data : {OneModule, PaddedModule}) {
    auto ExpectedFile = create(Data);
    ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
    const MinidumpFile &File = **ExpectedFile;
    Expected<ArrayRef<Module>> ExpectedModule = File.getModuleList();
    ASSERT_THAT_EXPECTED(ExpectedModule, Succeeded());
    ASSERT_EQ(1u, ExpectedModule->size());
    const Module &M = ExpectedModule.get()[0];
    EXPECT_EQ(0x0807060504030201u, M.BaseOfImage);
    EXPECT_EQ(0x02010009u, M.SizeOfImage);
    EXPECT_EQ(0x06050403u, M.Checksum);
    EXPECT_EQ(0x00090807u, M.TimeDateStamp);
    EXPECT_EQ(0x04030201u, M.ModuleNameRVA);
    EXPECT_EQ(0x04030201u, M.CvRecord.DataSize);
    EXPECT_EQ(0x08070605u, M.CvRecord.RVA);
    EXPECT_EQ(0x02010009u, M.MiscRecord.DataSize);
    EXPECT_EQ(0x06050403u, M.MiscRecord.RVA);
    EXPECT_EQ(0x0403020100090807u, M.Reserved0);
    EXPECT_EQ(0x0201000908070605u, M.Reserved1);
  }

  std::vector<uint8_t> StreamTooShort{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      4, 0, 0, 0, 111, 0, 0, 0,             // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // ModuleList
      1, 0, 0, 0,             // NumberOfModules
      1, 2, 3, 4, 5, 6, 7, 8, // BaseOfImage
      9, 0, 1, 2, 3, 4, 5, 6, // SizeOfImage, Checksum
      7, 8, 9, 0, 1, 2, 3, 4, // TimeDateStamp, ModuleNameRVA
      0, 0, 0, 0, 0, 0, 0, 0, // Signature, StructVersion
      0, 0, 0, 0, 0, 0, 0, 0, // FileVersion
      0, 0, 0, 0, 0, 0, 0, 0, // ProductVersion
      0, 0, 0, 0, 0, 0, 0, 0, // FileFlagsMask, FileFlags
      0, 0, 0, 0,             // FileOS
      0, 0, 0, 0, 0, 0, 0, 0, // FileType, FileSubType
      0, 0, 0, 0, 0, 0, 0, 0, // FileDate
      1, 2, 3, 4, 5, 6, 7, 8, // CvRecord
      9, 0, 1, 2, 3, 4, 5, 6, // MiscRecord
      7, 8, 9, 0, 1, 2, 3, 4, // Reserved0
      5, 6, 7, 8, 9, 0, 1, 2, // Reserved1
  };
  auto ExpectedFile = create(StreamTooShort);
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  const MinidumpFile &File = **ExpectedFile;
  EXPECT_THAT_EXPECTED(File.getModuleList(), Failed<BinaryError>());
}

TEST(MinidumpFile, getThreadList) {
  std::vector<uint8_t> OneThread{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      3, 0, 0, 0, 52, 0, 0, 0,              // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // ThreadList
      1, 0, 0, 0,             // NumberOfThreads
      1, 2, 3, 4, 5, 6, 7, 8, // ThreadId, SuspendCount
      9, 0, 1, 2, 3, 4, 5, 6, // PriorityClass, Priority
      7, 8, 9, 0, 1, 2, 3, 4, // EnvironmentBlock
      // Stack
      5, 6, 7, 8, 9, 0, 1, 2, // StartOfMemoryRange
      3, 4, 5, 6, 7, 8, 9, 0, // DataSize, RVA
      // Context
      1, 2, 3, 4, 5, 6, 7, 8, // DataSize, RVA
  };
  // Same as before, but with a padded thread list.
  std::vector<uint8_t> PaddedThread{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      3, 0, 0, 0, 56, 0, 0, 0,              // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // ThreadList
      1, 0, 0, 0,             // NumberOfThreads
      0, 0, 0, 0,             // Padding
      1, 2, 3, 4, 5, 6, 7, 8, // ThreadId, SuspendCount
      9, 0, 1, 2, 3, 4, 5, 6, // PriorityClass, Priority
      7, 8, 9, 0, 1, 2, 3, 4, // EnvironmentBlock
      // Stack
      5, 6, 7, 8, 9, 0, 1, 2, // StartOfMemoryRange
      3, 4, 5, 6, 7, 8, 9, 0, // DataSize, RVA
      // Context
      1, 2, 3, 4, 5, 6, 7, 8, // DataSize, RVA
  };

  for (ArrayRef<uint8_t> Data : {OneThread, PaddedThread}) {
    auto ExpectedFile = create(Data);
    ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
    const MinidumpFile &File = **ExpectedFile;
    Expected<ArrayRef<Thread>> ExpectedThread = File.getThreadList();
    ASSERT_THAT_EXPECTED(ExpectedThread, Succeeded());
    ASSERT_EQ(1u, ExpectedThread->size());
    const Thread &T = ExpectedThread.get()[0];
    EXPECT_EQ(0x04030201u, T.ThreadId);
    EXPECT_EQ(0x08070605u, T.SuspendCount);
    EXPECT_EQ(0x02010009u, T.PriorityClass);
    EXPECT_EQ(0x06050403u, T.Priority);
    EXPECT_EQ(0x0403020100090807u, T.EnvironmentBlock);
    EXPECT_EQ(0x0201000908070605u, T.Stack.StartOfMemoryRange);
    EXPECT_EQ(0x06050403u, T.Stack.Memory.DataSize);
    EXPECT_EQ(0x00090807u, T.Stack.Memory.RVA);
    EXPECT_EQ(0x04030201u, T.Context.DataSize);
    EXPECT_EQ(0x08070605u, T.Context.RVA);
  }
}

TEST(MinidumpFile, getMemoryList) {
  std::vector<uint8_t> OneRange{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      5, 0, 0, 0, 20, 0, 0, 0,              // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // MemoryDescriptor
      1, 0, 0, 0,             // NumberOfMemoryRanges
      5, 6, 7, 8, 9, 0, 1, 2, // StartOfMemoryRange
      3, 4, 5, 6, 7, 8, 9, 0, // DataSize, RVA
  };
  // Same as before, but with a padded memory list.
  std::vector<uint8_t> PaddedRange{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      5, 0, 0, 0, 24, 0, 0, 0,              // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // MemoryDescriptor
      1, 0, 0, 0,             // NumberOfMemoryRanges
      0, 0, 0, 0,             // Padding
      5, 6, 7, 8, 9, 0, 1, 2, // StartOfMemoryRange
      3, 4, 5, 6, 7, 8, 9, 0, // DataSize, RVA
  };

  for (ArrayRef<uint8_t> Data : {OneRange, PaddedRange}) {
    auto ExpectedFile = create(Data);
    ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
    const MinidumpFile &File = **ExpectedFile;
    Expected<ArrayRef<MemoryDescriptor>> ExpectedRanges = File.getMemoryList();
    ASSERT_THAT_EXPECTED(ExpectedRanges, Succeeded());
    ASSERT_EQ(1u, ExpectedRanges->size());
    const MemoryDescriptor &MD = ExpectedRanges.get()[0];
    EXPECT_EQ(0x0201000908070605u, MD.StartOfMemoryRange);
    EXPECT_EQ(0x06050403u, MD.Memory.DataSize);
    EXPECT_EQ(0x00090807u, MD.Memory.RVA);
  }
}

TEST(MinidumpFile, getMemoryInfoList) {
  std::vector<uint8_t> OneEntry{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      16, 0, 0, 0, 64, 0, 0, 0,             // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // MemoryInfoListHeader
      16, 0, 0, 0, 48, 0, 0, 0, // SizeOfHeader, SizeOfEntry
      1, 0, 0, 0, 0, 0, 0, 0,   // NumberOfEntries
      // MemoryInfo
      0, 1, 2, 3, 4, 5, 6, 7,   // BaseAddress
      8, 9, 0, 1, 2, 3, 4, 5,   // AllocationBase
      16, 0, 0, 0, 6, 7, 8, 9,  // AllocationProtect, Reserved0
      0, 1, 2, 3, 4, 5, 6, 7,   // RegionSize
      0, 16, 0, 0, 32, 0, 0, 0, // State, Protect
      0, 0, 2, 0, 8, 9, 0, 1,   // Type, Reserved1
  };

  // Same as before, but the list header is larger.
  std::vector<uint8_t> BiggerHeader{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      16, 0, 0, 0, 68, 0, 0, 0,             // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // MemoryInfoListHeader
      20, 0, 0, 0, 48, 0, 0, 0, // SizeOfHeader, SizeOfEntry
      1, 0, 0, 0, 0, 0, 0, 0,   // NumberOfEntries
      0, 0, 0, 0,               // ???
      // MemoryInfo
      0, 1, 2, 3, 4, 5, 6, 7,   // BaseAddress
      8, 9, 0, 1, 2, 3, 4, 5,   // AllocationBase
      16, 0, 0, 0, 6, 7, 8, 9,  // AllocationProtect, Reserved0
      0, 1, 2, 3, 4, 5, 6, 7,   // RegionSize
      0, 16, 0, 0, 32, 0, 0, 0, // State, Protect
      0, 0, 2, 0, 8, 9, 0, 1,   // Type, Reserved1
  };

  // Same as before, but the entry is larger.
  std::vector<uint8_t> BiggerEntry{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      16, 0, 0, 0, 68, 0, 0, 0,             // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // MemoryInfoListHeader
      16, 0, 0, 0, 52, 0, 0, 0, // SizeOfHeader, SizeOfEntry
      1, 0, 0, 0, 0, 0, 0, 0,   // NumberOfEntries
      // MemoryInfo
      0, 1, 2, 3, 4, 5, 6, 7,   // BaseAddress
      8, 9, 0, 1, 2, 3, 4, 5,   // AllocationBase
      16, 0, 0, 0, 6, 7, 8, 9,  // AllocationProtect, Reserved0
      0, 1, 2, 3, 4, 5, 6, 7,   // RegionSize
      0, 16, 0, 0, 32, 0, 0, 0, // State, Protect
      0, 0, 2, 0, 8, 9, 0, 1,   // Type, Reserved1
      0, 0, 0, 0,               // ???
  };

  for (ArrayRef<uint8_t> Data : {OneEntry, BiggerHeader, BiggerEntry}) {
    auto ExpectedFile = create(Data);
    ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
    const MinidumpFile &File = **ExpectedFile;
    auto ExpectedInfo = File.getMemoryInfoList();
    ASSERT_THAT_EXPECTED(ExpectedInfo, Succeeded());
    ASSERT_EQ(1, std::distance(ExpectedInfo->begin(), ExpectedInfo->end()));
    const MemoryInfo &Info = *ExpectedInfo.get().begin();
    EXPECT_EQ(0x0706050403020100u, Info.BaseAddress);
    EXPECT_EQ(0x0504030201000908u, Info.AllocationBase);
    EXPECT_EQ(MemoryProtection::Execute, Info.AllocationProtect);
    EXPECT_EQ(0x09080706u, Info.Reserved0);
    EXPECT_EQ(0x0706050403020100u, Info.RegionSize);
    EXPECT_EQ(MemoryState::Commit, Info.State);
    EXPECT_EQ(MemoryProtection::ExecuteRead, Info.Protect);
    EXPECT_EQ(MemoryType::Private, Info.Type);
    EXPECT_EQ(0x01000908u, Info.Reserved1);
  }

  // Header does not fit into the stream.
  std::vector<uint8_t> HeaderTooBig{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      16, 0, 0, 0, 15, 0, 0, 0,             // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // MemoryInfoListHeader
      16, 0, 0, 0, 48, 0, 0, 0, // SizeOfHeader, SizeOfEntry
      1, 0, 0, 0, 0, 0, 0,      // ???
  };
  Expected<std::unique_ptr<MinidumpFile>> File = create(HeaderTooBig);
  ASSERT_THAT_EXPECTED(File, Succeeded());
  EXPECT_THAT_EXPECTED(File.get()->getMemoryInfoList(), Failed<BinaryError>());

  // Header fits into the stream, but it is too small to contain the required
  // entries.
  std::vector<uint8_t> HeaderTooSmall{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      16, 0, 0, 0, 15, 0, 0, 0,             // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // MemoryInfoListHeader
      15, 0, 0, 0, 48, 0, 0, 0, // SizeOfHeader, SizeOfEntry
      1, 0, 0, 0, 0, 0, 0,      // ???
  };
  File = create(HeaderTooSmall);
  ASSERT_THAT_EXPECTED(File, Succeeded());
  EXPECT_THAT_EXPECTED(File.get()->getMemoryInfoList(), Failed<BinaryError>());

  std::vector<uint8_t> EntryTooBig{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      16, 0, 0, 0, 64, 0, 0, 0,             // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // MemoryInfoListHeader
      16, 0, 0, 0, 49, 0, 0, 0, // SizeOfHeader, SizeOfEntry
      1, 0, 0, 0, 0, 0, 0, 0,   // NumberOfEntries
      // MemoryInfo
      0, 1, 2, 3, 4, 5, 6, 7,   // BaseAddress
      8, 9, 0, 1, 2, 3, 4, 5,   // AllocationBase
      16, 0, 0, 0, 6, 7, 8, 9,  // AllocationProtect, Reserved0
      0, 1, 2, 3, 4, 5, 6, 7,   // RegionSize
      0, 16, 0, 0, 32, 0, 0, 0, // State, Protect
      0, 0, 2, 0, 8, 9, 0, 1,   // Type, Reserved1
  };
  File = create(EntryTooBig);
  ASSERT_THAT_EXPECTED(File, Succeeded());
  EXPECT_THAT_EXPECTED(File.get()->getMemoryInfoList(), Failed<BinaryError>());

  std::vector<uint8_t> ThreeEntries{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      32, 0, 0, 0,                          // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      0, 0, 0, 0, 0, 0, 0, 0,               // Flags
                                            // Stream Directory
      16, 0, 0, 0, 160, 0, 0, 0,            // Type, DataSize,
      44, 0, 0, 0,                          // RVA
      // MemoryInfoListHeader
      16, 0, 0, 0, 48, 0, 0, 0, // SizeOfHeader, SizeOfEntry
      3, 0, 0, 0, 0, 0, 0, 0,   // NumberOfEntries
      // MemoryInfo
      0, 1, 2, 3, 0, 0, 0, 0, // BaseAddress
      0, 0, 0, 0, 0, 0, 0, 0, // AllocationBase
      0, 0, 0, 0, 0, 0, 0, 0, // AllocationProtect, Reserved0
      0, 0, 0, 0, 0, 0, 0, 0, // RegionSize
      0, 0, 0, 0, 0, 0, 0, 0, // State, Protect
      0, 0, 0, 0, 0, 0, 0, 0, // Type, Reserved1
      0, 0, 4, 5, 6, 7, 0, 0, // BaseAddress
      0, 0, 0, 0, 0, 0, 0, 0, // AllocationBase
      0, 0, 0, 0, 0, 0, 0, 0, // AllocationProtect, Reserved0
      0, 0, 0, 0, 0, 0, 0, 0, // RegionSize
      0, 0, 0, 0, 0, 0, 0, 0, // State, Protect
      0, 0, 0, 0, 0, 0, 0, 0, // Type, Reserved1
      0, 0, 0, 8, 9, 0, 1, 0, // BaseAddress
      0, 0, 0, 0, 0, 0, 0, 0, // AllocationBase
      0, 0, 0, 0, 0, 0, 0, 0, // AllocationProtect, Reserved0
      0, 0, 0, 0, 0, 0, 0, 0, // RegionSize
      0, 0, 0, 0, 0, 0, 0, 0, // State, Protect
      0, 0, 0, 0, 0, 0, 0, 0, // Type, Reserved1
  };
  File = create(ThreeEntries);
  ASSERT_THAT_EXPECTED(File, Succeeded());
  auto ExpectedInfo = File.get()->getMemoryInfoList();
  ASSERT_THAT_EXPECTED(ExpectedInfo, Succeeded());
  EXPECT_THAT(to_vector<3>(map_range(*ExpectedInfo,
                                     [](const MemoryInfo &Info) -> uint64_t {
                                       return Info.BaseAddress;
                                     })),
              testing::ElementsAre(0x0000000003020100u, 0x0000070605040000u,
                                   0x0001000908000000u));
}

TEST(MinidumpFile, getExceptionStreams) {
  std::vector<uint8_t> Data{
      // Header
      'M', 'D', 'M', 'P', 0x93, 0xa7, 0, 0, // Signature, Version
      1, 0, 0, 0,                           // NumberOfStreams,
      0x20, 0, 0, 0,                        // StreamDirectoryRVA
      0, 1, 2, 3, 4, 5, 6, 7,               // Checksum, TimeDateStamp
      8, 9, 0, 1, 2, 3, 4, 5,               // Flags
      // Stream Directory
      6, 0, 0, 0, 168, 0, 0, 0, // Type, DataSize,
      0x2c, 0, 0, 0,            // RVA
      // Exception Stream
      1, 2, 3, 4, // Thread ID
      0, 0, 0, 0, // Padding
      // Exception Record
      2, 3, 4, 2, 7, 8, 8, 9,  // Code, Flags
      3, 4, 5, 6, 7, 8, 9, 10, // Inner exception record address
      8, 7, 6, 5, 4, 3, 2, 1,  // Exception address
      4, 0, 0, 0, 0, 0, 0, 0,  // Parameter count, padding
      0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, // Parameter 0
      0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, // Parameter 1
      0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, // Parameter 2
      0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, // Parameter 3
      0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, // Parameter 4
      0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, // Parameter 5
      0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, // Parameter 6
      0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, // Parameter 7
      0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, // Parameter 8
      0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, // Parameter 9
      0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, // Parameter 10
      0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, // Parameter 11
      0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, // Parameter 12
      0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, // Parameter 13
      0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, // Parameter 14
      // Thread Context
      0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, // DataSize, RVA
  };
  auto ExpectedFile = create(Data);
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  const MinidumpFile &File = **ExpectedFile;

  auto ExceptionStreams = File.getExceptionStreams();
  ASSERT_NE(ExceptionStreams.begin(), ExceptionStreams.end());
  auto ExceptionIterator = ExceptionStreams.begin();
  Expected<const ExceptionStream &> ExpectedStream = *ExceptionIterator;
  ASSERT_THAT_EXPECTED(ExpectedStream, Succeeded());
  EXPECT_EQ(0x04030201u, ExpectedStream->ThreadId);
  const minidump::Exception &Exception = ExpectedStream->ExceptionRecord;
  EXPECT_EQ(0x02040302u, Exception.ExceptionCode);
  EXPECT_EQ(0x09080807u, Exception.ExceptionFlags);
  EXPECT_EQ(0x0a09080706050403u, Exception.ExceptionRecord);
  EXPECT_EQ(0x0102030405060708u, Exception.ExceptionAddress);
  EXPECT_EQ(4u, Exception.NumberParameters);
  for (uint64_t index = 0; index < Exception.MaxParameters; ++index) {
    EXPECT_EQ(0x1716151413121110u + index * 0x1010101010101010u,
              Exception.ExceptionInformation[index]);
  }
  EXPECT_EQ(0x84838281, ExpectedStream->ThreadContext.DataSize);
  EXPECT_EQ(0x88878685, ExpectedStream->ThreadContext.RVA);
  ++ExceptionIterator;
  ASSERT_EQ(ExceptionIterator, ExceptionStreams.end());
}
