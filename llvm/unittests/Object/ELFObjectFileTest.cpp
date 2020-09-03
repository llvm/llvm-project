//===- ELFObjectFileTest.cpp - Tests for ELFObjectFile --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

template <class ELFT>
static Expected<ELFObjectFile<ELFT>> create(ArrayRef<uint8_t> Data) {
  return ELFObjectFile<ELFT>::create(
      MemoryBufferRef(toStringRef(Data), "Test buffer"));
}

// A class to initialize a buffer to represent an ELF object file.
struct DataForTest {
  std::vector<uint8_t> Data;

  template <typename T>
  std::vector<uint8_t> makeElfData(uint8_t Class, uint8_t Encoding,
                                   uint16_t Machine) {
    T Ehdr{}; // Zero-initialise the header.
    Ehdr.e_ident[ELF::EI_MAG0] = 0x7f;
    Ehdr.e_ident[ELF::EI_MAG1] = 'E';
    Ehdr.e_ident[ELF::EI_MAG2] = 'L';
    Ehdr.e_ident[ELF::EI_MAG3] = 'F';
    Ehdr.e_ident[ELF::EI_CLASS] = Class;
    Ehdr.e_ident[ELF::EI_DATA] = Encoding;
    Ehdr.e_ident[ELF::EI_VERSION] = 1;
    Ehdr.e_type = ELF::ET_REL;
    Ehdr.e_machine = Machine;
    Ehdr.e_version = 1;
    Ehdr.e_ehsize = sizeof(T);

    bool IsLittleEndian = Encoding == ELF::ELFDATA2LSB;
    if (sys::IsLittleEndianHost != IsLittleEndian) {
      sys::swapByteOrder(Ehdr.e_type);
      sys::swapByteOrder(Ehdr.e_machine);
      sys::swapByteOrder(Ehdr.e_version);
      sys::swapByteOrder(Ehdr.e_ehsize);
    }

    uint8_t *EhdrBytes = reinterpret_cast<uint8_t *>(&Ehdr);
    std::vector<uint8_t> Bytes;
    std::copy(EhdrBytes, EhdrBytes + sizeof(Ehdr), std::back_inserter(Bytes));
    return Bytes;
  }

  DataForTest(uint8_t Class, uint8_t Encoding, uint16_t Machine) {
    if (Class == ELF::ELFCLASS64)
      Data = makeElfData<ELF::Elf64_Ehdr>(Class, Encoding, Machine);
    else {
      assert(Class == ELF::ELFCLASS32);
      Data = makeElfData<ELF::Elf64_Ehdr>(Class, Encoding, Machine);
    }
  }
};

TEST(ELFObjectFileTest, MachineTestForVE) {
  DataForTest Data(ELF::ELFCLASS64, ELF::ELFDATA2LSB, ELF::EM_VE);
  auto ExpectedFile = create<ELF64LE>(Data.Data);
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  const ELFObjectFile<ELF64LE> &File = *ExpectedFile;
  EXPECT_EQ("elf64-ve", File.getFileFormatName());
  EXPECT_EQ(Triple::ve, File.getArch());
}

TEST(ELFObjectFileTest, MachineTestForX86_64) {
  DataForTest Data(ELF::ELFCLASS64, ELF::ELFDATA2LSB, ELF::EM_X86_64);
  auto ExpectedFile = create<ELF64LE>(Data.Data);
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  const ELFObjectFile<ELF64LE> &File = *ExpectedFile;
  EXPECT_EQ("elf64-x86-64", File.getFileFormatName());
  EXPECT_EQ(Triple::x86_64, File.getArch());
}

TEST(ELFObjectFileTest, MachineTestFor386) {
  DataForTest Data(ELF::ELFCLASS32, ELF::ELFDATA2LSB, ELF::EM_386);
  auto ExpectedFile = create<ELF32LE>(Data.Data);
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  const ELFObjectFile<ELF32LE> &File = *ExpectedFile;
  EXPECT_EQ("elf32-i386", File.getFileFormatName());
  EXPECT_EQ(Triple::x86, File.getArch());
}

TEST(ELFObjectFileTest, MachineTestForMIPS) {
  {
    DataForTest Data(ELF::ELFCLASS64, ELF::ELFDATA2LSB, ELF::EM_MIPS);
    auto ExpectedFile = create<ELF64LE>(Data.Data);
    ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
    const ELFObjectFile<ELF64LE> &File = *ExpectedFile;
    EXPECT_EQ("elf64-mips", File.getFileFormatName());
    EXPECT_EQ(Triple::mips64el, File.getArch());
  }
  {
    DataForTest Data(ELF::ELFCLASS64, ELF::ELFDATA2MSB, ELF::EM_MIPS);
    auto ExpectedFile = create<ELF64BE>(Data.Data);
    ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
    const ELFObjectFile<ELF64BE> &File = *ExpectedFile;
    EXPECT_EQ("elf64-mips", File.getFileFormatName());
    EXPECT_EQ(Triple::mips64, File.getArch());
  }
  {
    DataForTest Data(ELF::ELFCLASS32, ELF::ELFDATA2LSB, ELF::EM_MIPS);
    auto ExpectedFile = create<ELF32LE>(Data.Data);
    ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
    const ELFObjectFile<ELF32LE> &File = *ExpectedFile;
    EXPECT_EQ("elf32-mips", File.getFileFormatName());
    EXPECT_EQ(Triple::mipsel, File.getArch());
  }
  {
    DataForTest Data(ELF::ELFCLASS32, ELF::ELFDATA2MSB, ELF::EM_MIPS);
    auto ExpectedFile = create<ELF32BE>(Data.Data);
    ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
    const ELFObjectFile<ELF32BE> &File = *ExpectedFile;
    EXPECT_EQ("elf32-mips", File.getFileFormatName());
    EXPECT_EQ(Triple::mips, File.getArch());
  }
}
