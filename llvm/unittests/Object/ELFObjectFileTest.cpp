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

std::array<DataForTest, 4> generateData(uint16_t Machine) {
  return {DataForTest(ELF::ELFCLASS32, ELF::ELFDATA2LSB, Machine),
          DataForTest(ELF::ELFCLASS32, ELF::ELFDATA2MSB, Machine),
          DataForTest(ELF::ELFCLASS64, ELF::ELFDATA2LSB, Machine),
          DataForTest(ELF::ELFCLASS64, ELF::ELFDATA2MSB, Machine)};
}

} // namespace

TEST(ELFObjectFileTest, MachineTestForNoneOrUnused) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-unknown", "elf64-unknown"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_NONE))
    checkFormatAndArch(D, Formats[I++], Triple::UnknownArch);

  // Test an arbitrary unused EM_* value (255).
  I = 0;
  for (const DataForTest &D : generateData(255))
    checkFormatAndArch(D, Formats[I++], Triple::UnknownArch);
}

TEST(ELFObjectFileTest, MachineTestForVE) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-ve", "elf64-ve"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_VE))
    checkFormatAndArch(D, Formats[I++], Triple::ve);
}

TEST(ELFObjectFileTest, MachineTestForX86_64) {
  std::array<StringRef, 4> Formats = {"elf32-x86-64", "elf32-x86-64",
                                      "elf64-x86-64", "elf64-x86-64"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_X86_64))
    checkFormatAndArch(D, Formats[I++], Triple::x86_64);
}

TEST(ELFObjectFileTest, MachineTestFor386) {
  std::array<StringRef, 4> Formats = {"elf32-i386", "elf32-i386", "elf64-i386",
                                      "elf64-i386"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_386))
    checkFormatAndArch(D, Formats[I++], Triple::x86);
}

TEST(ELFObjectFileTest, MachineTestForMIPS) {
  std::array<StringRef, 4> Formats = {"elf32-mips", "elf32-mips", "elf64-mips",
                                      "elf64-mips"};
  std::array<Triple::ArchType, 4> Archs = {Triple::mipsel, Triple::mips,
                                           Triple::mips64el, Triple::mips64};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_MIPS)) {
    checkFormatAndArch(D, Formats[I], Archs[I]);
    ++I;
  }
}

TEST(ELFObjectFileTest, MachineTestForAMDGPU) {
  std::array<StringRef, 4> Formats = {"elf32-amdgpu", "elf32-amdgpu",
                                      "elf64-amdgpu", "elf64-amdgpu"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_AMDGPU))
    checkFormatAndArch(D, Formats[I++], Triple::UnknownArch);
}

TEST(ELFObjectFileTest, MachineTestForIAMCU) {
  std::array<StringRef, 4> Formats = {"elf32-iamcu", "elf32-iamcu",
                                      "elf64-unknown", "elf64-unknown"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_IAMCU))
    checkFormatAndArch(D, Formats[I++], Triple::x86);
}

TEST(ELFObjectFileTest, MachineTestForAARCH64) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-littleaarch64",
                                      "elf64-bigaarch64"};
  std::array<Triple::ArchType, 4> Archs = {Triple::aarch64, Triple::aarch64_be,
                                           Triple::aarch64, Triple::aarch64_be};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_AARCH64)) {
    checkFormatAndArch(D, Formats[I], Archs[I]);
    ++I;
  }
}

TEST(ELFObjectFileTest, MachineTestForPPC64) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-powerpcle", "elf64-powerpc"};
  std::array<Triple::ArchType, 4> Archs = {Triple::ppc64le, Triple::ppc64,
                                           Triple::ppc64le, Triple::ppc64};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_PPC64)) {
    checkFormatAndArch(D, Formats[I], Archs[I]);
    ++I;
  }
}

TEST(ELFObjectFileTest, MachineTestForPPC) {
  std::array<StringRef, 4> Formats = {"elf32-powerpc", "elf32-powerpc",
                                      "elf64-unknown", "elf64-unknown"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_PPC))
    checkFormatAndArch(D, Formats[I++], Triple::ppc);
}

TEST(ELFObjectFileTest, MachineTestForRISCV) {
  std::array<StringRef, 4> Formats = {"elf32-littleriscv", "elf32-littleriscv",
                                      "elf64-littleriscv", "elf64-littleriscv"};
  std::array<Triple::ArchType, 4> Archs = {Triple::riscv32, Triple::riscv32,
                                           Triple::riscv64, Triple::riscv64};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_RISCV)) {
    checkFormatAndArch(D, Formats[I], Archs[I]);
    ++I;
  }
}

TEST(ELFObjectFileTest, MachineTestForARM) {
  std::array<StringRef, 4> Formats = {"elf32-littlearm", "elf32-bigarm",
                                      "elf64-unknown", "elf64-unknown"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_ARM))
    checkFormatAndArch(D, Formats[I++], Triple::arm);
}

TEST(ELFObjectFileTest, MachineTestForS390) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-s390", "elf64-s390"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_S390))
    checkFormatAndArch(D, Formats[I++], Triple::systemz);
}

TEST(ELFObjectFileTest, MachineTestForSPARCV9) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-sparc", "elf64-sparc"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_SPARCV9))
    checkFormatAndArch(D, Formats[I++], Triple::sparcv9);
}

TEST(ELFObjectFileTest, MachineTestForSPARC) {
  std::array<StringRef, 4> Formats = {"elf32-sparc", "elf32-sparc",
                                      "elf64-unknown", "elf64-unknown"};
  std::array<Triple::ArchType, 4> Archs = {Triple::sparcel, Triple::sparc,
                                           Triple::sparcel, Triple::sparc};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_SPARC)) {
    checkFormatAndArch(D, Formats[I], Archs[I]);
    ++I;
  }
}

TEST(ELFObjectFileTest, MachineTestForSPARC32PLUS) {
  std::array<StringRef, 4> Formats = {"elf32-sparc", "elf32-sparc",
                                      "elf64-unknown", "elf64-unknown"};
  std::array<Triple::ArchType, 4> Archs = {Triple::sparcel, Triple::sparc,
                                           Triple::sparcel, Triple::sparc};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_SPARC32PLUS)) {
    checkFormatAndArch(D, Formats[I], Archs[I]);
    ++I;
  }
}

TEST(ELFObjectFileTest, MachineTestForBPF) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-bpf", "elf64-bpf"};
  std::array<Triple::ArchType, 4> Archs = {Triple::bpfel, Triple::bpfeb,
                                           Triple::bpfel, Triple::bpfeb};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_BPF)) {
    checkFormatAndArch(D, Formats[I], Archs[I]);
    ++I;
  }
}

TEST(ELFObjectFileTest, MachineTestForAVR) {
  std::array<StringRef, 4> Formats = {"elf32-avr", "elf32-avr", "elf64-unknown",
                                      "elf64-unknown"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_AVR))
    checkFormatAndArch(D, Formats[I++], Triple::avr);
}

TEST(ELFObjectFileTest, MachineTestForHEXAGON) {
  std::array<StringRef, 4> Formats = {"elf32-hexagon", "elf32-hexagon",
                                      "elf64-unknown", "elf64-unknown"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_HEXAGON))
    checkFormatAndArch(D, Formats[I++], Triple::hexagon);
}

TEST(ELFObjectFileTest, MachineTestForLANAI) {
  std::array<StringRef, 4> Formats = {"elf32-lanai", "elf32-lanai",
                                      "elf64-unknown", "elf64-unknown"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_LANAI))
    checkFormatAndArch(D, Formats[I++], Triple::lanai);
}

TEST(ELFObjectFileTest, MachineTestForMSP430) {
  std::array<StringRef, 4> Formats = {"elf32-msp430", "elf32-msp430",
                                      "elf64-unknown", "elf64-unknown"};
  size_t I = 0;
  for (const DataForTest &D : generateData(ELF::EM_MSP430))
    checkFormatAndArch(D, Formats[I++], Triple::msp430);
}
