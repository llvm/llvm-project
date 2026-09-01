//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectContainer/Universal-Mach-O/ObjectContainerUniversalMachO.h"
#include "Plugins/ObjectContainer/Mach-O-Fileset/ObjectContainerMachOFileset.h"
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/FileSpec.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <vector>

using namespace lldb_private;

namespace {
class ObjectContainerUniversalMachOTest : public ::testing::Test {
  SubsystemRAII<FileSystem, ObjectContainerUniversalMachO, ObjectFileMachO>
      subsystems;
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

// A fat Mach-O slice at offset 0 is self-referential.
TEST_F(ObjectContainerUniversalMachOTest, GetObjectFileSelfReferentialSlice) {
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !fat-mach-o
FatHeader:
  magic:           0xCAFEBABE
  nfat_arch:       1
FatArchs:
  - cputype:         0x01000007
    cpusubtype:      0x00000003
    offset:          0x00000000
    size:            0x00001000
    align:           12
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

  llvm::Expected<llvm::sys::fs::TempFile> TmpFile =
      ExpectedFile->writeToTemporaryFile();
  ASSERT_THAT_EXPECTED(TmpFile, llvm::Succeeded());

  ArchSpec Arch;
  Arch.SetArchitecture(eArchTypeMachO, 0x01000007, 0x00000003);
  lldb::ModuleSP Module =
      std::make_shared<lldb_private::Module>(FileSpec(TmpFile->TmpName), Arch);
  EXPECT_EQ(Module->GetObjectFile(), nullptr);

  ASSERT_THAT_ERROR(TmpFile->discard(), llvm::Succeeded());
}

// A fat Mach-O slice whose declared size is far larger than the file.
TEST_F(ObjectContainerUniversalMachOTest, GetObjectFileOversizedSlice) {
  std::vector<uint8_t> Data = {
      0xCA, 0xFE, 0xBA, 0xBF,                         // magic: FAT_MAGIC_64
      0x00, 0x00, 0x00, 0x01,                         // nfat_arch: 1
      0x01, 0x00, 0x00, 0x07,                         // cputype: X86_64
      0x00, 0x00, 0x00, 0x03,                         // cpusubtype: 3
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x28, // offset: 40
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, // size: 2^48 - 1
      0x00, 0x00, 0x00, 0x0C,                         // align: 12
      0x00, 0x00, 0x00, 0x00,                         // reserved: 0
  };
  // It is the plugin claiming this slice that goes on to request the declared
  // size above.
  llvm::MachO::mach_header_64 SliceHeader = {};
  SliceHeader.magic = llvm::MachO::MH_MAGIC_64;
  SliceHeader.cputype = llvm::MachO::CPU_TYPE_X86_64;
  SliceHeader.cpusubtype = 3;
  SliceHeader.filetype = llvm::MachO::MH_EXECUTE;
  const auto *SliceBytes = reinterpret_cast<const uint8_t *>(&SliceHeader);
  Data.insert(Data.end(), SliceBytes, SliceBytes + sizeof(SliceHeader));
  Data.resize(40 + 512, 0);

  llvm::Expected<llvm::sys::fs::TempFile> TmpFile =
      llvm::sys::fs::TempFile::create("temp%%%%%%%%%%%%%%%%");
  ASSERT_THAT_EXPECTED(TmpFile, llvm::Succeeded());
  llvm::raw_fd_ostream(TmpFile->FD, /*shouldClose=*/false) << llvm::StringRef(
      reinterpret_cast<const char *>(Data.data()), Data.size());

  ArchSpec Arch;
  Arch.SetArchitecture(eArchTypeMachO, 0x01000007, 0x00000003);
  lldb::ModuleSP Module =
      std::make_shared<lldb_private::Module>(FileSpec(TmpFile->TmpName), Arch);
  ObjectFile *Obj = Module->GetObjectFile();
  ASSERT_THAT_ERROR(TmpFile->discard(), llvm::Succeeded());
  ASSERT_NE(Obj, nullptr);
  EXPECT_EQ(Obj->GetByteSize(), 512u); // Clamped to the bytes available.
}

// Regression fixture: a Mach-O fileset whose single load command has
// cmdsize = 0.  With ncmds set near INT_MAX the function hangs.  The
// fix breaks out of the loop as soon as
// cmdsize < sizeof(load_command).  Found by lldb-target-fuzzer.
namespace {
class ObjectContainerMachOFilesetTest : public ::testing::Test {
  SubsystemRAII<FileSystem, ObjectContainerMachOFileset> subsystems;
};
} // namespace

TEST_F(ObjectContainerMachOFilesetTest, ZeroCmdSize) {
  // Minimal little-endian x86_64 Mach-O fileset: mach_header_64 (32 bytes)
  // followed by a single load_command with cmdsize = 0.  ncmds is set to
  // 0x7FFFFFFF so that without the fix ParseFileset spins ~2 billion times and
  // never returns in practice; with the fix it breaks on the first iteration.
  // Reaching the assertion below is the regression check.
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x01000007
  cpusubtype:      0x80000003
  filetype:        0x0000000C
  ncmds:           0x7FFFFFFF
  sizeofcmds:      8
  flags:           0x00000000
  reserved:        0x00000000
LoadCommands:
  - cmd:             LC_THREAD
    cmdsize:         0
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  ModuleSpec Spec = ExpectedFile->moduleSpec();
  lldb::DataExtractorSP DataSP = Spec.GetExtractor();
  // Before the fix ParseFileset loops ~0x7FFFFFFF times and never returns.
  (void)ObjectContainerMachOFileset::GetModuleSpecifications(
      FileSpec(), DataSP, 0, DataSP->GetByteSize());
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
