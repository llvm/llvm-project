//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectContainer/Universal-Mach-O/ObjectContainerUniversalMachO.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/FileSpec.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
class ObjectContainerUniversalMachOTest : public ::testing::Test {
  SubsystemRAII<FileSystem, ObjectContainerUniversalMachO> subsystems;
};
} // namespace

// Regression fixture: a universal (fat) Mach-O whose header claims
// nfat_arch = 0xFFFFFFFF while the file holds a single arch slice.  The arch
// loop in ObjectContainerUniversalMachO::ParseHeader used nfat_arch as its
// bound without checking it against the available data, so this header sent it
// spinning ~4.29 billion times.  GetModuleSpecifications must instead stop once
// the data is exhausted and return promptly.  Found by lldb-target-fuzzer.
TEST_F(ObjectContainerUniversalMachOTest, HugeNfatArch) {
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !fat-mach-o
FatHeader:
  magic:           0xCAFEBABF
  nfat_arch:       0xFFFFFFFF
FatArchs:
  - cputype:         0x01000007
    cpusubtype:      0x00000003
    offset:          0x0000000000004000
    size:            4
    align:           14
    reserved:        0x00000000
Slices:
  - !mach-o
    FileHeader:
      magic:           0xFEEDFACF
      cputype:         0x01000007
      cpusubtype:      0x00000003
      filetype:        0x00000002
      ncmds:           0
      sizeofcmds:      0
      flags:           0x00000000
      reserved:        0x00000000
    LoadCommands:    []
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  ModuleSpec Spec = ExpectedFile->moduleSpec();
  lldb::DataExtractorSP Data = Spec.GetExtractor();
  // Before the fix this loops ~0xFFFFFFFF times and never returns in practice;
  // reaching the assertion at all is the regression check.
  ModuleSpecList Specs = ObjectFile::GetModuleSpecifications(
      Spec.GetFileSpec(), Data, 0, Data->GetByteSize());
  EXPECT_EQ(Specs.GetSize(), 0u);
}

// Regression fixture: a universal (fat) Mach-O with a self-referential slice
// whose offset is 0.  ObjectContainerUniversalMachO::GetModuleSpecifications
// re-parses each slice by recursing into ObjectFile::GetModuleSpecifications at
// the slice offset; a slice at offset 0 produced a recursive call with
// identical arguments and recursed until the stack overflowed.  The
// non-advancing slice must be skipped, leaving nothing loadable.  Found by
// lldb-target-fuzzer.
TEST_F(ObjectContainerUniversalMachOTest, SliceOffsetZero) {
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !fat-mach-o
FatHeader:
  magic:           0xCAFEBABE
  nfat_arch:       1
FatArchs:
  - cputype:         0x00000007
    cpusubtype:      0x00000003
    offset:          0x00000000
    size:            0x00001000
    align:           12
Slices:
  - !mach-o
    FileHeader:
      magic:           0xFEEDFACF
      cputype:         0x00000007
      cpusubtype:      0x00000003
      filetype:        0x00000002
      ncmds:           0
      sizeofcmds:      0
      flags:           0x00000000
      reserved:        0x00000000
    LoadCommands:    []
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  llvm::Expected<llvm::sys::fs::TempFile> TmpFile =
      ExpectedFile->writeToTemporaryFile();
  ASSERT_THAT_EXPECTED(TmpFile, llvm::Succeeded());

  // Before the fix the offset-0 slice recurses on identical arguments and
  // overflows the stack; reaching the assertion at all is the regression
  // check.  The self-referential slice is now skipped, so no specs are found.
  ModuleSpecList Specs = ObjectFile::GetModuleSpecifications(
      FileSpec(TmpFile->TmpName), /*file_offset=*/0, /*file_size=*/0);
  EXPECT_EQ(Specs.GetSize(), 0u);

  ASSERT_THAT_ERROR(TmpFile->discard(), llvm::Succeeded());
}

// Regression fixture: a universal (fat) Mach-O whose header claims a huge
// nfat_arch (here 0xAFAFAFAF) but provides no fat_arch entries beyond the
// header bytes.  Found by lldb-target-fuzzer.
TEST_F(ObjectContainerUniversalMachOTest, NfatArchTruncatedSlices) {
  // Hand-crafted fat header: FAT_MAGIC_64 + nfat_arch=0xAFAFAFAF + 2 stray
  // payload bytes, not enough for even one fat_arch_64 entry (32 bytes).
  const uint8_t kData[] = {
      0xCA, 0xFE, 0xBA, 0xBF, // magic:     FAT_MAGIC_64 (big endian)
      0xAF, 0xAF, 0xAF, 0xAF, // nfat_arch: 0xAFAFAFAF (untrusted, huge)
      0xAF, 0xAF,             // truncated arch payload
  };
  lldb::DataBufferSP Buf =
      std::make_shared<DataBufferHeap>(kData, sizeof(kData));

  std::unique_ptr<lldb_private::ObjectContainer> Container(
      ObjectContainerUniversalMachO::CreateInstance(
          /*module_sp=*/nullptr, Buf, /*data_offset=*/0, /*file=*/nullptr,
          /*file_offset=*/0, /*length=*/sizeof(kData)));
  ASSERT_NE(Container.get(), nullptr);

  // m_fat_archs has zero elements, returns false.
  ArchSpec Arch;
  EXPECT_FALSE(Container->GetArchitectureAtIndex(0, Arch));
}
