//===-- NativeProcessELFTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/Host/NativeProcessTestUtils.h"

#include "Plugins/Process/POSIX/NativeProcessELF.h"
#include "Plugins/Process/Utility/AuxVector.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/DataExtractor.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/MemoryBuffer.h"

#include "gmock/gmock.h"

using namespace lldb_private;
using namespace lldb;
using namespace testing;

namespace {
class MockProcessELF : public MockProcess<NativeProcessELF> {
public:
  using MockProcess::MockProcess;
  using NativeProcessELF::GetAuxValue;
  using NativeProcessELF::GetELFImageInfoAddress;
};

std::unique_ptr<llvm::MemoryBuffer> CreateAuxvData(
    MockProcessELF &process,
    llvm::ArrayRef<std::pair<AuxVector::EntryType, uint32_t>> auxv_data) {
  DataEncoder encoder(process.GetByteOrder(), process.GetAddressByteSize());
  for (auto &pair : auxv_data) {
    encoder.AppendAddress(pair.first);
    encoder.AppendAddress(pair.second);
  }
  return llvm::MemoryBuffer::getMemBufferCopy(
      llvm::toStringRef(encoder.GetData()), "");
}

} // namespace

TEST(NativeProcessELFTest, GetAuxValue) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcessELF process(DummyDelegate, ArchSpec("i386-pc-linux"));

  uint64_t phdr_addr = 0x42;
  auto auxv_buffer = CreateAuxvData(
      process, {std::make_pair(AuxVector::AUXV_AT_PHDR, phdr_addr)});
  EXPECT_CALL(process, GetAuxvData())
      .WillOnce(Return(ByMove(std::move(auxv_buffer))));

  ASSERT_EQ(phdr_addr, process.GetAuxValue(AuxVector::AUXV_AT_PHDR));
}

TEST(NativeProcessELFTest, GetELFImageInfoAddress) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcessELF process(DummyDelegate, ArchSpec("i386-pc-linux"));

  uint32_t load_base = 0x1000;
  uint32_t info_addr = 0x3741;
  uint32_t phdr_addr = load_base + sizeof(llvm::ELF::Elf32_Ehdr);

  auto auxv_buffer = CreateAuxvData(
      process,
      {std::make_pair(AuxVector::AUXV_AT_PHDR, phdr_addr),
       std::make_pair(AuxVector::AUXV_AT_PHENT, sizeof(llvm::ELF::Elf32_Phdr)),
       std::make_pair(AuxVector::AUXV_AT_PHNUM, 2)});
  EXPECT_CALL(process, GetAuxvData())
      .WillOnce(Return(ByMove(std::move(auxv_buffer))));

  // We're going to set up a fake memory with 2 program headers and 1 entry in
  // the dynamic section. For simplicity sake they will be contiguous in memory.
  struct MemoryContents {
    llvm::ELF::Elf32_Phdr phdr_load;
    llvm::ELF::Elf32_Phdr phdr_dynamic;
    llvm::ELF::Elf32_Dyn dyn_debug;
  } MC;
  // Setup the 2 program header entries
  MC.phdr_load.p_type = llvm::ELF::PT_PHDR;
  MC.phdr_load.p_vaddr = phdr_addr - load_base;

  MC.phdr_dynamic.p_type = llvm::ELF::PT_DYNAMIC;
  MC.phdr_dynamic.p_vaddr =
      (phdr_addr + 2 * sizeof(llvm::ELF::Elf32_Phdr)) - load_base;
  MC.phdr_dynamic.p_memsz = sizeof(llvm::ELF::Elf32_Dyn);

  // Setup the single entry in the .dynamic section
  MC.dyn_debug.d_tag = llvm::ELF::DT_DEBUG;
  MC.dyn_debug.d_un.d_ptr = info_addr;

  FakeMemory M(&MC, sizeof(MC), phdr_addr);
  EXPECT_CALL(process, ReadMemory(_, _))
      .WillRepeatedly(Invoke(&M, &FakeMemory::Read));

  lldb::addr_t elf_info_addr = process.GetELFImageInfoAddress<
      llvm::ELF::Elf32_Ehdr, llvm::ELF::Elf32_Phdr, llvm::ELF::Elf32_Dyn>();

  // Read the address at the elf_info_addr location to make sure we're reading
  // the correct one.
  lldb::offset_t info_addr_offset = elf_info_addr - phdr_addr;
  DataExtractor mem_extractor(&MC, sizeof(MC), process.GetByteOrder(),
                              process.GetAddressByteSize());
  ASSERT_EQ(mem_extractor.GetAddress(&info_addr_offset), info_addr);
}

TEST(NativeProcessELFTest, GetELFImageInfoAddress_NoDebugEntry) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcessELF process(DummyDelegate, ArchSpec("i386-pc-linux"));

  uint32_t phdr_addr = sizeof(llvm::ELF::Elf32_Ehdr);

  auto auxv_buffer = CreateAuxvData(
      process,
      {std::make_pair(AuxVector::AUXV_AT_PHDR, phdr_addr),
       std::make_pair(AuxVector::AUXV_AT_PHENT, sizeof(llvm::ELF::Elf32_Phdr)),
       std::make_pair(AuxVector::AUXV_AT_PHNUM, 2)});
  EXPECT_CALL(process, GetAuxvData())
      .WillOnce(Return(ByMove(std::move(auxv_buffer))));

  // We're going to set up a fake memory with 2 program headers and 1 entry in
  // the dynamic section. For simplicity sake they will be contiguous in memory.
  struct MemoryContents {
    llvm::ELF::Elf32_Phdr phdr_load;
    llvm::ELF::Elf32_Phdr phdr_dynamic;
    llvm::ELF::Elf32_Dyn dyn_notdebug;
  } MC;
  // Setup the 2 program header entries
  MC.phdr_load.p_type = llvm::ELF::PT_PHDR;
  MC.phdr_load.p_vaddr = phdr_addr;

  MC.phdr_dynamic.p_type = llvm::ELF::PT_DYNAMIC;
  MC.phdr_dynamic.p_vaddr = (phdr_addr + 2 * sizeof(llvm::ELF::Elf32_Phdr));
  MC.phdr_dynamic.p_memsz = sizeof(llvm::ELF::Elf32_Dyn);

  // Setup the single entry in the .dynamic section
  MC.dyn_notdebug.d_tag = llvm::ELF::DT_NULL;

  FakeMemory M(&MC, sizeof(MC), phdr_addr);
  EXPECT_CALL(process, ReadMemory(_, _))
      .WillRepeatedly(Invoke(&M, &FakeMemory::Read));

  lldb::addr_t elf_info_addr = process.GetELFImageInfoAddress<
      llvm::ELF::Elf32_Ehdr, llvm::ELF::Elf32_Phdr, llvm::ELF::Elf32_Dyn>();

  ASSERT_EQ(elf_info_addr, LLDB_INVALID_ADDRESS);
}
