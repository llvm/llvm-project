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
  EXPECT_EQ(None, File.getRawStream(StreamType::Unused));
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
