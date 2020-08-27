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

namespace {

// A struct to initialize a buffer to represent an ELF object file.
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
      Data = makeElfData<ELF::Elf32_Ehdr>(Class, Encoding, Machine);
    }
  }
};

void checkFormatAndArch(const DataForTest &D, StringRef Fmt,
                        Triple::ArchType Arch) {
  Expected<std::unique_ptr<ObjectFile>> ELFObjOrErr =
      object::ObjectFile::createELFObjectFile(
          MemoryBufferRef(toStringRef(D.Data), "dummyELF"));
  ASSERT_THAT_EXPECTED(ELFObjOrErr, Succeeded());

  const ObjectFile &File = *(*ELFObjOrErr).get();
  EXPECT_EQ(Fmt, File.getFileFormatName());
  EXPECT_EQ(Arch, File.getArch());
}

} // namespace

TEST(ELFObjectFileTest, MachineTestForVE) {
  checkFormatAndArch({ELF::ELFCLASS64, ELF::ELFDATA2LSB, ELF::EM_VE},
                     "elf64-ve", Triple::ve);
}

TEST(ELFObjectFileTest, MachineTestForX86_64) {
  checkFormatAndArch({ELF::ELFCLASS64, ELF::ELFDATA2LSB, ELF::EM_X86_64},
                     "elf64-x86-64", Triple::x86_64);
}

TEST(ELFObjectFileTest, MachineTestFor386) {
  checkFormatAndArch({ELF::ELFCLASS32, ELF::ELFDATA2LSB, ELF::EM_386},
                     "elf32-i386", Triple::x86);
}

TEST(ELFObjectFileTest, MachineTestForMIPS) {
  checkFormatAndArch({ELF::ELFCLASS64, ELF::ELFDATA2LSB, ELF::EM_MIPS},
                     "elf64-mips", Triple::mips64el);

  checkFormatAndArch({ELF::ELFCLASS64, ELF::ELFDATA2MSB, ELF::EM_MIPS},
                     "elf64-mips", Triple::mips64);

  checkFormatAndArch({ELF::ELFCLASS32, ELF::ELFDATA2LSB, ELF::EM_MIPS},
                     "elf32-mips", Triple::mipsel);

  checkFormatAndArch({ELF::ELFCLASS32, ELF::ELFDATA2MSB, ELF::EM_MIPS},
                     "elf32-mips", Triple::mips);
}
