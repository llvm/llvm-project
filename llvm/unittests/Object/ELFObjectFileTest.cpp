//===- ELFObjectFileTest.cpp - Tests for ELFObjectFile --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include "llvm/Support/thread.h"
#include "llvm/TargetParser/Host.h"

using namespace llvm;
using namespace llvm::object;

// Used to skip LLVM_BB_ADDR_MAP tests on windows platforms due to
// https://github.com/llvm/llvm-project/issues/60013.
bool IsHostWindows() {
  Triple Host(Triple::normalize(sys::getProcessTriple()));
  return Host.isOSWindows();
}

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
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_NONE)))
    checkFormatAndArch(Data, Formats[Idx], Triple::UnknownArch);

  // Test an arbitrary unused EM_* value (255).
  for (auto [Idx, Data] : enumerate(generateData(255)))
    checkFormatAndArch(Data, Formats[Idx], Triple::UnknownArch);
}

TEST(ELFObjectFileTest, MachineTestForVE) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-ve", "elf64-ve"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_VE)))
    checkFormatAndArch(Data, Formats[Idx], Triple::ve);
}

TEST(ELFObjectFileTest, MachineTestForX86_64) {
  std::array<StringRef, 4> Formats = {"elf32-x86-64", "elf32-x86-64",
                                      "elf64-x86-64", "elf64-x86-64"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_X86_64)))
    checkFormatAndArch(Data, Formats[Idx], Triple::x86_64);
}

TEST(ELFObjectFileTest, MachineTestFor386) {
  std::array<StringRef, 4> Formats = {"elf32-i386", "elf32-i386", "elf64-i386",
                                      "elf64-i386"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_386)))
    checkFormatAndArch(Data, Formats[Idx], Triple::x86);
}

TEST(ELFObjectFileTest, MachineTestForMIPS) {
  std::array<StringRef, 4> Formats = {"elf32-mips", "elf32-mips", "elf64-mips",
                                      "elf64-mips"};
  std::array<Triple::ArchType, 4> Archs = {Triple::mipsel, Triple::mips,
                                           Triple::mips64el, Triple::mips64};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_MIPS)))
    checkFormatAndArch(Data, Formats[Idx], Archs[Idx]);
}

TEST(ELFObjectFileTest, MachineTestForAMDGPU) {
  std::array<StringRef, 4> Formats = {"elf32-amdgpu", "elf32-amdgpu",
                                      "elf64-amdgpu", "elf64-amdgpu"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_AMDGPU)))
    checkFormatAndArch(Data, Formats[Idx], Triple::UnknownArch);
}

TEST(ELFObjectFileTest, MachineTestForIAMCU) {
  std::array<StringRef, 4> Formats = {"elf32-iamcu", "elf32-iamcu",
                                      "elf64-unknown", "elf64-unknown"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_IAMCU)))
    checkFormatAndArch(Data, Formats[Idx], Triple::x86);
}

TEST(ELFObjectFileTest, MachineTestForAARCH64) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-littleaarch64",
                                      "elf64-bigaarch64"};
  std::array<Triple::ArchType, 4> Archs = {Triple::aarch64, Triple::aarch64_be,
                                           Triple::aarch64, Triple::aarch64_be};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_AARCH64)))
    checkFormatAndArch(Data, Formats[Idx], Archs[Idx]);
}

TEST(ELFObjectFileTest, MachineTestForPPC64) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-powerpcle", "elf64-powerpc"};
  std::array<Triple::ArchType, 4> Archs = {Triple::ppc64le, Triple::ppc64,
                                           Triple::ppc64le, Triple::ppc64};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_PPC64)))
    checkFormatAndArch(Data, Formats[Idx], Archs[Idx]);
}

TEST(ELFObjectFileTest, MachineTestForPPC) {
  std::array<StringRef, 4> Formats = {"elf32-powerpcle", "elf32-powerpc",
                                      "elf64-unknown", "elf64-unknown"};
  std::array<Triple::ArchType, 4> Archs = {Triple::ppcle, Triple::ppc,
                                           Triple::ppcle, Triple::ppc};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_PPC)))
    checkFormatAndArch(Data, Formats[Idx], Archs[Idx]);
}

TEST(ELFObjectFileTest, MachineTestForRISCV) {
  std::array<StringRef, 4> Formats = {"elf32-littleriscv", "elf32-littleriscv",
                                      "elf64-littleriscv", "elf64-littleriscv"};
  std::array<Triple::ArchType, 4> Archs = {Triple::riscv32, Triple::riscv32,
                                           Triple::riscv64, Triple::riscv64};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_RISCV)))
    checkFormatAndArch(Data, Formats[Idx], Archs[Idx]);
}

TEST(ELFObjectFileTest, MachineTestForARM) {
  std::array<StringRef, 4> Formats = {"elf32-littlearm", "elf32-bigarm",
                                      "elf64-unknown", "elf64-unknown"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_ARM)))
    checkFormatAndArch(Data, Formats[Idx], Triple::arm);
}

TEST(ELFObjectFileTest, MachineTestForS390) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-s390", "elf64-s390"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_S390)))
    checkFormatAndArch(Data, Formats[Idx], Triple::systemz);
}

TEST(ELFObjectFileTest, MachineTestForSPARCV9) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-sparc", "elf64-sparc"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_SPARCV9)))
    checkFormatAndArch(Data, Formats[Idx], Triple::sparcv9);
}

TEST(ELFObjectFileTest, MachineTestForSPARC) {
  std::array<StringRef, 4> Formats = {"elf32-sparc", "elf32-sparc",
                                      "elf64-unknown", "elf64-unknown"};
  std::array<Triple::ArchType, 4> Archs = {Triple::sparcel, Triple::sparc,
                                           Triple::sparcel, Triple::sparc};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_SPARC)))
    checkFormatAndArch(Data, Formats[Idx], Archs[Idx]);
}

TEST(ELFObjectFileTest, MachineTestForSPARC32PLUS) {
  std::array<StringRef, 4> Formats = {"elf32-sparc", "elf32-sparc",
                                      "elf64-unknown", "elf64-unknown"};
  std::array<Triple::ArchType, 4> Archs = {Triple::sparcel, Triple::sparc,
                                           Triple::sparcel, Triple::sparc};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_SPARC32PLUS)))
    checkFormatAndArch(Data, Formats[Idx], Archs[Idx]);
}

TEST(ELFObjectFileTest, MachineTestForBPF) {
  std::array<StringRef, 4> Formats = {"elf32-unknown", "elf32-unknown",
                                      "elf64-bpf", "elf64-bpf"};
  std::array<Triple::ArchType, 4> Archs = {Triple::bpfel, Triple::bpfeb,
                                           Triple::bpfel, Triple::bpfeb};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_BPF)))
    checkFormatAndArch(Data, Formats[Idx], Archs[Idx]);
}

TEST(ELFObjectFileTest, MachineTestForAVR) {
  std::array<StringRef, 4> Formats = {"elf32-avr", "elf32-avr", "elf64-unknown",
                                      "elf64-unknown"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_AVR)))
    checkFormatAndArch(Data, Formats[Idx], Triple::avr);
}

TEST(ELFObjectFileTest, MachineTestForHEXAGON) {
  std::array<StringRef, 4> Formats = {"elf32-hexagon", "elf32-hexagon",
                                      "elf64-unknown", "elf64-unknown"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_HEXAGON)))
    checkFormatAndArch(Data, Formats[Idx], Triple::hexagon);
}

TEST(ELFObjectFileTest, MachineTestForLANAI) {
  std::array<StringRef, 4> Formats = {"elf32-lanai", "elf32-lanai",
                                      "elf64-unknown", "elf64-unknown"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_LANAI)))
    checkFormatAndArch(Data, Formats[Idx], Triple::lanai);
}

TEST(ELFObjectFileTest, MachineTestForMSP430) {
  std::array<StringRef, 4> Formats = {"elf32-msp430", "elf32-msp430",
                                      "elf64-unknown", "elf64-unknown"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_MSP430)))
    checkFormatAndArch(Data, Formats[Idx], Triple::msp430);
}

TEST(ELFObjectFileTest, MachineTestForLoongArch) {
  std::array<StringRef, 4> Formats = {"elf32-loongarch", "elf32-loongarch",
                                      "elf64-loongarch", "elf64-loongarch"};
  std::array<Triple::ArchType, 4> Archs = {
      Triple::loongarch32, Triple::loongarch32, Triple::loongarch64,
      Triple::loongarch64};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_LOONGARCH)))
    checkFormatAndArch(Data, Formats[Idx], Archs[Idx]);
}

TEST(ELFObjectFileTest, MachineTestForCSKY) {
  std::array<StringRef, 4> Formats = {"elf32-csky", "elf32-csky",
                                      "elf64-unknown", "elf64-unknown"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_CSKY)))
    checkFormatAndArch(Data, Formats[Idx], Triple::csky);
}

TEST(ELFObjectFileTest, MachineTestForXtensa) {
  std::array<StringRef, 4> Formats = {"elf32-xtensa", "elf32-xtensa",
                                      "elf64-unknown", "elf64-unknown"};
  for (auto [Idx, Data] : enumerate(generateData(ELF::EM_XTENSA)))
    checkFormatAndArch(Data, Formats[Idx], Triple::xtensa);
}

// ELF relative relocation type test.
TEST(ELFObjectFileTest, RelativeRelocationTypeTest) {
  EXPECT_EQ(ELF::R_CKCORE_RELATIVE, getELFRelativeRelocationType(ELF::EM_CSKY));
}

template <class ELFT>
static Expected<ELFObjectFile<ELFT>> toBinary(SmallVectorImpl<char> &Storage,
                                              StringRef Yaml) {
  raw_svector_ostream OS(Storage);
  yaml::Input YIn(Yaml);
  if (!yaml::convertYAML(YIn, OS, [](const Twine &Msg) {}))
    return createStringError(std::errc::invalid_argument,
                             "unable to convert YAML");
  return ELFObjectFile<ELFT>::create(MemoryBufferRef(OS.str(), "dummyELF"));
}

// Check we are able to create an ELFObjectFile even when the content of the
// SHT_SYMTAB_SHNDX section can't be read properly.
TEST(ELFObjectFileTest, InvalidSymtabShndxTest) {
  SmallString<0> Storage;
  Expected<ELFObjectFile<ELF64LE>> ExpectedFile = toBinary<ELF64LE>(Storage, R"(
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data:  ELFDATA2LSB
  Type:  ET_REL
Sections:
  - Name:    .symtab_shndx
    Type:    SHT_SYMTAB_SHNDX
    Entries: [ 0 ]
    ShSize: 0xFFFFFFFF
)");

  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
}

// Test that we are able to create an ELFObjectFile even when loadable segments
// are unsorted by virtual address.
// Test that ELFFile<ELFT>::toMappedAddr works properly in this case.

TEST(ELFObjectFileTest, InvalidLoadSegmentsOrderTest) {
  SmallString<0> Storage;
  Expected<ELFObjectFile<ELF64LE>> ExpectedFile = toBinary<ELF64LE>(Storage, R"(
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data:  ELFDATA2LSB
  Type:  ET_EXEC
Sections:
  - Name:         .foo
    Type:         SHT_PROGBITS
    Address:      0x1000
    Offset:       0x3000
    ContentArray: [ 0x11 ]
  - Name:         .bar
    Type:         SHT_PROGBITS
    Address:      0x2000
    Offset:       0x4000
    ContentArray: [ 0x99 ]
ProgramHeaders:
  - Type:     PT_LOAD
    VAddr:    0x2000
    FirstSec: .bar
    LastSec:  .bar
  - Type:     PT_LOAD
    VAddr:    0x1000
    FirstSec: .foo
    LastSec:  .foo
)");

  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());

  std::string WarnString;
  auto ToMappedAddr = [&](uint64_t Addr) -> const uint8_t * {
    Expected<const uint8_t *> DataOrErr =
        ExpectedFile->getELFFile().toMappedAddr(Addr, [&](const Twine &Msg) {
          EXPECT_TRUE(WarnString.empty());
          WarnString = Msg.str();
          return Error::success();
        });

    if (!DataOrErr) {
      ADD_FAILURE() << toString(DataOrErr.takeError());
      return nullptr;
    }

    EXPECT_TRUE(WarnString ==
                "loadable segments are unsorted by virtual address");
    WarnString = "";
    return *DataOrErr;
  };

  const uint8_t *Data = ToMappedAddr(0x1000);
  ASSERT_TRUE(Data);
  MemoryBufferRef Buf = ExpectedFile->getMemoryBufferRef();
  EXPECT_EQ((const char *)Data - Buf.getBufferStart(), 0x3000);
  EXPECT_EQ(Data[0], 0x11);

  Data = ToMappedAddr(0x2000);
  ASSERT_TRUE(Data);
  Buf = ExpectedFile->getMemoryBufferRef();
  EXPECT_EQ((const char *)Data - Buf.getBufferStart(), 0x4000);
  EXPECT_EQ(Data[0], 0x99);
}

// This is a test for API that is related to symbols.
// We check that errors are properly reported here.
TEST(ELFObjectFileTest, InvalidSymbolTest) {
  SmallString<0> Storage;
  Expected<ELFObjectFile<ELF64LE>> ElfOrErr = toBinary<ELF64LE>(Storage, R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_DYN
  Machine: EM_X86_64
Sections:
  - Name: .symtab
    Type: SHT_SYMTAB
)");

  ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());
  const ELFFile<ELF64LE> &Elf = ElfOrErr->getELFFile();
  const ELFObjectFile<ELF64LE> &Obj = *ElfOrErr;

  Expected<const typename ELF64LE::Shdr *> SymtabSecOrErr = Elf.getSection(1);
  ASSERT_THAT_EXPECTED(SymtabSecOrErr, Succeeded());
  ASSERT_EQ((*SymtabSecOrErr)->sh_type, ELF::SHT_SYMTAB);

  auto DoCheck = [&](unsigned BrokenSymIndex, const char *ErrMsg) {
    ELFSymbolRef BrokenSym = Obj.toSymbolRef(*SymtabSecOrErr, BrokenSymIndex);

    // 1) Check the behavior of ELFObjectFile<ELFT>::getSymbolName().
    //    SymbolRef::getName() calls it internally. We can't test it directly,
    //    because it is protected.
    EXPECT_THAT_ERROR(BrokenSym.getName().takeError(),
                      FailedWithMessage(ErrMsg));

    // 2) Check the behavior of ELFObjectFile<ELFT>::getSymbol().
    EXPECT_THAT_ERROR(Obj.getSymbol(BrokenSym.getRawDataRefImpl()).takeError(),
                      FailedWithMessage(ErrMsg));

    // 3) Check the behavior of ELFObjectFile<ELFT>::getSymbolSection().
    //    SymbolRef::getSection() calls it internally. We can't test it
    //    directly, because it is protected.
    EXPECT_THAT_ERROR(BrokenSym.getSection().takeError(),
                      FailedWithMessage(ErrMsg));

    // 4) Check the behavior of ELFObjectFile<ELFT>::getSymbolFlags().
    //    SymbolRef::getFlags() calls it internally. We can't test it directly,
    //    because it is protected.
    EXPECT_THAT_ERROR(BrokenSym.getFlags().takeError(),
                      FailedWithMessage(ErrMsg));

    // 5) Check the behavior of ELFObjectFile<ELFT>::getSymbolType().
    //    SymbolRef::getType() calls it internally. We can't test it directly,
    //    because it is protected.
    EXPECT_THAT_ERROR(BrokenSym.getType().takeError(),
                      FailedWithMessage(ErrMsg));

    // 6) Check the behavior of ELFObjectFile<ELFT>::getSymbolAddress().
    //    SymbolRef::getAddress() calls it internally. We can't test it
    //    directly, because it is protected.
    EXPECT_THAT_ERROR(BrokenSym.getAddress().takeError(),
                      FailedWithMessage(ErrMsg));

    // Finally, check the `ELFFile<ELFT>::getEntry` API. This is an underlying
    // method that generates errors for all cases above.
    EXPECT_THAT_EXPECTED(
        Elf.getEntry<typename ELF64LE::Sym>(**SymtabSecOrErr, 0), Succeeded());
    EXPECT_THAT_ERROR(
        Elf.getEntry<typename ELF64LE::Sym>(**SymtabSecOrErr, BrokenSymIndex)
            .takeError(),
        FailedWithMessage(ErrMsg));
  };

  // We create a symbol with an index that is too large to exist in the symbol
  // table.
  DoCheck(0x1, "can't read an entry at 0x18: it goes past the end of the "
               "section (0x18)");

  // We create a symbol with an index that is too large to exist in the object.
  DoCheck(0xFFFFFFFF, "can't read an entry at 0x17ffffffe8: it goes past the "
                      "end of the section (0x18)");
}

// Tests for error paths of the ELFFile::decodeBBAddrMap API.
TEST(ELFObjectFileTest, InvalidDecodeBBAddrMap) {
  if (IsHostWindows())
    GTEST_SKIP();
  StringRef CommonYamlString(R"(
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data:  ELFDATA2LSB
  Type:  ET_EXEC
Sections:
  - Type: SHT_LLVM_BB_ADDR_MAP
    Name: .llvm_bb_addr_map
    Entries:
      - Address: 0x11111
)");

  auto DoCheck = [&](StringRef YamlString, const char *ErrMsg) {
    SmallString<0> Storage;
    Expected<ELFObjectFile<ELF64LE>> ElfOrErr =
        toBinary<ELF64LE>(Storage, YamlString);
    ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());
    const ELFFile<ELF64LE> &Elf = ElfOrErr->getELFFile();

    Expected<const typename ELF64LE::Shdr *> BBAddrMapSecOrErr =
        Elf.getSection(1);
    ASSERT_THAT_EXPECTED(BBAddrMapSecOrErr, Succeeded());
    EXPECT_THAT_ERROR(Elf.decodeBBAddrMap(**BBAddrMapSecOrErr).takeError(),
                      FailedWithMessage(ErrMsg));
  };

  // Check that we can detect unsupported versions.
  SmallString<128> UnsupportedVersionYamlString(CommonYamlString);
  UnsupportedVersionYamlString += R"(
        Version: 3
        BBEntries:
          - AddressOffset: 0x0
            Size:          0x1
            Metadata:      0x2
)";

  DoCheck(UnsupportedVersionYamlString,
          "unsupported SHT_LLVM_BB_ADDR_MAP version: 3");

  SmallString<128> CommonVersionedYamlString(CommonYamlString);
  CommonVersionedYamlString += R"(
        Version: 2
        BBEntries:
          - ID:            1
            AddressOffset: 0x0
            Size:          0x1
            Metadata:      0x2
)";

  // Check that we can detect the malformed encoding when the section is
  // truncated.
  SmallString<128> TruncatedYamlString(CommonVersionedYamlString);
  TruncatedYamlString += R"(
    ShSize: 0xb
)";
  DoCheck(TruncatedYamlString, "unable to decode LEB128 at offset 0x0000000b: "
                               "malformed uleb128, extends past end");

  // Check that we can detect when the encoded BB entry fields exceed the UINT32
  // limit.
  SmallVector<SmallString<128>, 3> OverInt32LimitYamlStrings(
      3, CommonVersionedYamlString);
  OverInt32LimitYamlStrings[0] += R"(
          - ID:            1
            AddressOffset: 0x100000000
            Size:          0xFFFFFFFF
            Metadata:      0xFFFFFFFF
)";

  OverInt32LimitYamlStrings[1] += R"(
          - ID:            2
            AddressOffset: 0xFFFFFFFF
            Size:          0x100000000
            Metadata:      0xFFFFFFFF
)";

  OverInt32LimitYamlStrings[2] += R"(
          - ID:            3
            AddressOffset: 0xFFFFFFFF
            Size:          0xFFFFFFFF
            Metadata:      0x100000000
)";

  DoCheck(OverInt32LimitYamlStrings[0],
          "ULEB128 value at offset 0x10 exceeds UINT32_MAX (0x100000000)");
  DoCheck(OverInt32LimitYamlStrings[1],
          "ULEB128 value at offset 0x15 exceeds UINT32_MAX (0x100000000)");
  DoCheck(OverInt32LimitYamlStrings[2],
          "ULEB128 value at offset 0x1a exceeds UINT32_MAX (0x100000000)");

  // Check the proper error handling when the section has fields exceeding
  // UINT32 and is also truncated. This is for checking that we don't generate
  // unhandled errors.
  SmallVector<SmallString<128>, 3> OverInt32LimitAndTruncated(
      3, OverInt32LimitYamlStrings[1]);
  // Truncate before the end of the 5-byte field.
  OverInt32LimitAndTruncated[0] += R"(
    ShSize: 0x19
)";
  // Truncate at the end of the 5-byte field.
  OverInt32LimitAndTruncated[1] += R"(
    ShSize: 0x1a
)";
  // Truncate after the end of the 5-byte field.
  OverInt32LimitAndTruncated[2] += R"(
    ShSize: 0x1b
)";

  DoCheck(OverInt32LimitAndTruncated[0],
          "unable to decode LEB128 at offset 0x00000015: malformed uleb128, "
          "extends past end");
  DoCheck(OverInt32LimitAndTruncated[1],
          "ULEB128 value at offset 0x15 exceeds UINT32_MAX (0x100000000)");
  DoCheck(OverInt32LimitAndTruncated[2],
          "ULEB128 value at offset 0x15 exceeds UINT32_MAX (0x100000000)");

  // Check for proper error handling when the 'NumBlocks' field is overridden
  // with an out-of-range value.
  SmallString<128> OverLimitNumBlocks(CommonVersionedYamlString);
  OverLimitNumBlocks += R"(
        NumBlocks: 0x100000000
)";

  DoCheck(OverLimitNumBlocks,
          "ULEB128 value at offset 0xa exceeds UINT32_MAX (0x100000000)");
}

// Test for the ELFObjectFile::readBBAddrMap API.
TEST(ELFObjectFileTest, ReadBBAddrMap) {
  if (IsHostWindows())
    GTEST_SKIP();
  StringRef CommonYamlString(R"(
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data:  ELFDATA2LSB
  Type:  ET_EXEC
Sections:
  - Name: .llvm_bb_addr_map_1
    Type: SHT_LLVM_BB_ADDR_MAP
    Link: 1
    Entries:
      - Version: 2
        Address: 0x11111
        BBEntries:
          - ID:            1
            AddressOffset: 0x0
            Size:          0x1
            Metadata:      0x2
  - Name: .llvm_bb_addr_map_2
    Type: SHT_LLVM_BB_ADDR_MAP
    Link: 1
    Entries:
      - Version: 2
        Address: 0x22222
        BBEntries:
          - ID:            2
            AddressOffset: 0x0
            Size:          0x2
            Metadata:      0x4
  - Name: .llvm_bb_addr_map_3
    Type: SHT_LLVM_BB_ADDR_MAP
    Link: 2
    Entries:
      - Version: 1
        Address: 0x33333
        BBEntries:
          - ID:            0
            AddressOffset: 0x0
            Size:          0x3
            Metadata:      0x6
  - Name: .llvm_bb_addr_map_4
    Type: SHT_LLVM_BB_ADDR_MAP_V0
  # Link: 0 (by default, can be overriden)
    Entries:
      - Version: 0
        Address: 0x44444
        BBEntries:
          - ID:            0
            AddressOffset: 0x0
            Size:          0x4
            Metadata:      0x18
)");

  BBAddrMap E1(0x11111, {{1, 0x0, 0x1, {false, true, false, false, false}}});
  BBAddrMap E2(0x22222, {{2, 0x0, 0x2, {false, false, true, false, false}}});
  BBAddrMap E3(0x33333, {{0, 0x0, 0x3, {false, true, true, false, false}}});
  BBAddrMap E4(0x44444, {{0, 0x0, 0x4, {false, false, false, true, true}}});

  std::vector<BBAddrMap> Section0BBAddrMaps = {E4};
  std::vector<BBAddrMap> Section1BBAddrMaps = {E3};
  std::vector<BBAddrMap> Section2BBAddrMaps = {E1, E2};
  std::vector<BBAddrMap> AllBBAddrMaps = {E1, E2, E3, E4};

  auto DoCheckSucceeds = [&](StringRef YamlString,
                             std::optional<unsigned> TextSectionIndex,
                             std::vector<BBAddrMap> ExpectedResult) {
    SmallString<0> Storage;
    Expected<ELFObjectFile<ELF64LE>> ElfOrErr =
        toBinary<ELF64LE>(Storage, YamlString);
    ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());

    Expected<const typename ELF64LE::Shdr *> BBAddrMapSecOrErr =
        ElfOrErr->getELFFile().getSection(1);
    ASSERT_THAT_EXPECTED(BBAddrMapSecOrErr, Succeeded());
    auto BBAddrMaps = ElfOrErr->readBBAddrMap(TextSectionIndex);
    ASSERT_THAT_EXPECTED(BBAddrMaps, Succeeded());
    EXPECT_EQ(*BBAddrMaps, ExpectedResult);
  };

  auto DoCheckFails = [&](StringRef YamlString,
                          std::optional<unsigned> TextSectionIndex,
                          const char *ErrMsg) {
    SmallString<0> Storage;
    Expected<ELFObjectFile<ELF64LE>> ElfOrErr =
        toBinary<ELF64LE>(Storage, YamlString);
    ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());

    Expected<const typename ELF64LE::Shdr *> BBAddrMapSecOrErr =
        ElfOrErr->getELFFile().getSection(1);
    ASSERT_THAT_EXPECTED(BBAddrMapSecOrErr, Succeeded());
    EXPECT_THAT_ERROR(ElfOrErr->readBBAddrMap(TextSectionIndex).takeError(),
                      FailedWithMessage(ErrMsg));
  };

  // Check that we can retrieve the data in the normal case.
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/std::nullopt,
                  AllBBAddrMaps);
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/0, Section0BBAddrMaps);
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/2, Section1BBAddrMaps);
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/1, Section2BBAddrMaps);
  // Check that when no bb-address-map section is found for a text section,
  // we return an empty result.
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/3, {});

  // Check that we detect when a bb-addr-map section is linked to an invalid
  // (not present) section.
  SmallString<128> InvalidLinkedYamlString(CommonYamlString);
  InvalidLinkedYamlString += R"(
    Link: 10
)";

  DoCheckFails(InvalidLinkedYamlString, /*TextSectionIndex=*/4,
               "unable to get the linked-to section for "
               "SHT_LLVM_BB_ADDR_MAP_V0 section with index 4: invalid section "
               "index: 10");
  // Linked sections are not checked when we don't target a specific text
  // section.
  DoCheckSucceeds(InvalidLinkedYamlString, /*TextSectionIndex=*/std::nullopt,
                  AllBBAddrMaps);

  // Check that we can detect when bb-address-map decoding fails.
  SmallString<128> TruncatedYamlString(CommonYamlString);
  TruncatedYamlString += R"(
    ShSize: 0x8
)";

  DoCheckFails(TruncatedYamlString, /*TextSectionIndex=*/std::nullopt,
               "unable to read SHT_LLVM_BB_ADDR_MAP_V0 section with index 4: "
               "unable to decode LEB128 at offset 0x00000008: malformed "
               "uleb128, extends past end");
  // Check that we can read the other section's bb-address-maps which are
  // valid.
  DoCheckSucceeds(TruncatedYamlString, /*TextSectionIndex=*/2,
                  Section1BBAddrMaps);
}

// Tests for error paths of the ELFFile::decodeBBAddrMap with PGOAnalysisMap
// API.
TEST(ELFObjectFileTest, InvalidDecodePGOAnalysisMap) {
  if (IsHostWindows())
    GTEST_SKIP();
  StringRef CommonYamlString(R"(
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data:  ELFDATA2LSB
  Type:  ET_EXEC
Sections:
  - Type: SHT_LLVM_BB_ADDR_MAP
    Name: .llvm_bb_addr_map
    Entries:
      - Address: 0x11111
)");

  auto DoCheck = [&](StringRef YamlString, const char *ErrMsg) {
    SmallString<0> Storage;
    Expected<ELFObjectFile<ELF64LE>> ElfOrErr =
        toBinary<ELF64LE>(Storage, YamlString);
    ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());
    const ELFFile<ELF64LE> &Elf = ElfOrErr->getELFFile();

    Expected<const typename ELF64LE::Shdr *> BBAddrMapSecOrErr =
        Elf.getSection(1);
    ASSERT_THAT_EXPECTED(BBAddrMapSecOrErr, Succeeded());

    std::vector<PGOAnalysisMap> PGOAnalyses;
    EXPECT_THAT_ERROR(
        Elf.decodeBBAddrMap(**BBAddrMapSecOrErr, nullptr, &PGOAnalyses)
            .takeError(),
        FailedWithMessage(ErrMsg));
  };

  // Check that we can detect unsupported versions that are too old.
  SmallString<128> UnsupportedLowVersionYamlString(CommonYamlString);
  UnsupportedLowVersionYamlString += R"(
        Version: 1
        Feature: 0x4
        BBEntries:
          - AddressOffset: 0x0
            Size:          0x1
            Metadata:      0x2
)";

  DoCheck(UnsupportedLowVersionYamlString,
          "version should be >= 2 for SHT_LLVM_BB_ADDR_MAP when PGO features "
          "are enabled: version = 1 feature = 4");

  SmallString<128> CommonVersionedYamlString(CommonYamlString);
  CommonVersionedYamlString += R"(
        Version: 2
        BBEntries:
          - ID:            1
            AddressOffset: 0x0
            Size:          0x1
            Metadata:      0x2
)";

  // Check that we fail when function entry count is enabled but not provided.
  SmallString<128> MissingFuncEntryCount(CommonYamlString);
  MissingFuncEntryCount += R"(
        Version: 2
        Feature: 0x01
)";

  DoCheck(MissingFuncEntryCount,
          "unable to decode LEB128 at offset 0x0000000b: malformed uleb128, "
          "extends past end");

  // Check that we fail when basic block frequency is enabled but not provided.
  SmallString<128> MissingBBFreq(CommonYamlString);
  MissingBBFreq += R"(
        Version: 2
        Feature: 0x02
        BBEntries:
          - ID:            1
            AddressOffset: 0x0
            Size:          0x1
            Metadata:      0x2
)";

  DoCheck(MissingBBFreq, "unable to decode LEB128 at offset 0x0000000f: "
                         "malformed uleb128, extends past end");

  // Check that we fail when branch probability is enabled but not provided.
  SmallString<128> MissingBrProb(CommonYamlString);
  MissingBrProb += R"(
        Version: 2
        Feature: 0x04
        BBEntries:
          - ID:            1
            AddressOffset: 0x0
            Size:          0x1
            Metadata:      0x6
          - ID:            2
            AddressOffset: 0x1
            Size:          0x1
            Metadata:      0x2
          - ID:            3
            AddressOffset: 0x2
            Size:          0x1
            Metadata:      0x2
    PGOAnalyses:
      - PGOBBEntries:
         - Successors:
            - ID:          2
              BrProb:      0x80000000
            - ID:          3
              BrProb:      0x80000000
         - Successors:
            - ID:          3
              BrProb:      0xF0000000
)";

  DoCheck(MissingBrProb, "unable to decode LEB128 at offset 0x00000017: "
                         "malformed uleb128, extends past end");
}

// Test for the ELFObjectFile::readBBAddrMap API with PGOAnalysisMap.
TEST(ELFObjectFileTest, ReadPGOAnalysisMap) {
  if (IsHostWindows())
    GTEST_SKIP();
  StringRef CommonYamlString(R"(
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data:  ELFDATA2LSB
  Type:  ET_EXEC
Sections:
  - Name: .llvm_bb_addr_map_1
    Type: SHT_LLVM_BB_ADDR_MAP
    Link: 1
    Entries:
      - Version: 2
        Address: 0x11111
        Feature: 0x1
        BBEntries:
          - ID:            1
            AddressOffset: 0x0
            Size:          0x1
            Metadata:      0x2
    PGOAnalyses:
      - FuncEntryCount: 892
  - Name: .llvm_bb_addr_map_2
    Type: SHT_LLVM_BB_ADDR_MAP
    Link: 1
    Entries:
      - Version: 2
        Address: 0x22222
        Feature: 0x2
        BBEntries:
          - ID:            2
            AddressOffset: 0x0
            Size:          0x2
            Metadata:      0x4
    PGOAnalyses:
      - PGOBBEntries:
         - BBFreq:         343
  - Name: .llvm_bb_addr_map_3
    Type: SHT_LLVM_BB_ADDR_MAP
    Link: 2
    Entries:
      - Version: 2
        Address: 0x33333
        Feature: 0x4
        BBEntries:
          - ID:            0
            AddressOffset: 0x0
            Size:          0x3
            Metadata:      0x6
          - ID:            1
            AddressOffset: 0x0
            Size:          0x3
            Metadata:      0x4
          - ID:            2
            AddressOffset: 0x0
            Size:          0x3
            Metadata:      0x0
    PGOAnalyses:
      - PGOBBEntries:
         - Successors:
            - ID:          1
              BrProb:      0x11111111
            - ID:          2
              BrProb:      0xeeeeeeee
         - Successors:
            - ID:          2
              BrProb:      0xffffffff
         - Successors:     []
  - Name: .llvm_bb_addr_map_4
    Type: SHT_LLVM_BB_ADDR_MAP
  # Link: 0 (by default, can be overriden)
    Entries:
      - Version: 2
        Address: 0x44444
        Feature: 0x7
        BBEntries:
          - ID:            0
            AddressOffset: 0x0
            Size:          0x4
            Metadata:      0x18
          - ID:            1
            AddressOffset: 0x0
            Size:          0x4
            Metadata:      0x0
          - ID:            2
            AddressOffset: 0x0
            Size:          0x4
            Metadata:      0x0
          - ID:            3
            AddressOffset: 0x0
            Size:          0x4
            Metadata:      0x0
    PGOAnalyses:
      - FuncEntryCount: 1000
        PGOBBEntries:
          - BBFreq:         1000
            Successors:
            - ID:          1
              BrProb:      0x22222222
            - ID:          2
              BrProb:      0x33333333
            - ID:          3
              BrProb:      0xaaaaaaaa
          - BBFreq:         133
            Successors:
            - ID:          2
              BrProb:      0x11111111
            - ID:          3
              BrProb:      0xeeeeeeee
          - BBFreq:         18
            Successors:
            - ID:          3
              BrProb:      0xffffffff
          - BBFreq:         1000
            Successors:    []
)");

  BBAddrMap E1(0x11111, {{1, 0x0, 0x1, {false, true, false, false, false}}});
  PGOAnalysisMap P1 = {892, {{}}, {true, false, false}};
  BBAddrMap E2(0x22222, {{2, 0x0, 0x2, {false, false, true, false, false}}});
  PGOAnalysisMap P2 = {{}, {{BlockFrequency(343), {}}}, {false, true, false}};
  BBAddrMap E3(0x33333, {{0, 0x0, 0x3, {false, true, true, false, false}},
                         {1, 0x3, 0x3, {false, false, true, false, false}},
                         {2, 0x6, 0x3, {false, false, false, false, false}}});
  PGOAnalysisMap P3 = {{},
                       {{{},
                         {{1, BranchProbability::getRaw(0x1111'1111)},
                          {2, BranchProbability::getRaw(0xeeee'eeee)}}},
                        {{}, {{2, BranchProbability::getRaw(0xffff'ffff)}}},
                        {{}, {}}},
                       {false, false, true}};
  BBAddrMap E4(0x44444, {{0, 0x0, 0x4, {false, false, false, true, true}},
                         {1, 0x4, 0x4, {false, false, false, false, false}},
                         {2, 0x8, 0x4, {false, false, false, false, false}},
                         {3, 0xc, 0x4, {false, false, false, false, false}}});
  PGOAnalysisMap P4 = {
      1000,
      {{BlockFrequency(1000),
        {{1, BranchProbability::getRaw(0x2222'2222)},
         {2, BranchProbability::getRaw(0x3333'3333)},
         {3, BranchProbability::getRaw(0xaaaa'aaaa)}}},
       {BlockFrequency(133),
        {{2, BranchProbability::getRaw(0x1111'1111)},
         {3, BranchProbability::getRaw(0xeeee'eeee)}}},
       {BlockFrequency(18), {{3, BranchProbability::getRaw(0xffff'ffff)}}},
       {BlockFrequency(1000), {}}},
      {true, true, true}};

  std::vector<BBAddrMap> Section0BBAddrMaps = {E4};
  std::vector<BBAddrMap> Section1BBAddrMaps = {E3};
  std::vector<BBAddrMap> Section2BBAddrMaps = {E1, E2};
  std::vector<BBAddrMap> AllBBAddrMaps = {E1, E2, E3, E4};

  std::vector<PGOAnalysisMap> Section0PGOAnalysisMaps = {P4};
  std::vector<PGOAnalysisMap> Section1PGOAnalysisMaps = {P3};
  std::vector<PGOAnalysisMap> Section2PGOAnalysisMaps = {P1, P2};
  std::vector<PGOAnalysisMap> AllPGOAnalysisMaps = {P1, P2, P3, P4};

  auto DoCheckSucceeds =
      [&](StringRef YamlString, std::optional<unsigned> TextSectionIndex,
          std::vector<BBAddrMap> ExpectedResult,
          std::optional<std::vector<PGOAnalysisMap>> ExpectedPGO) {
        SmallString<0> Storage;
        Expected<ELFObjectFile<ELF64LE>> ElfOrErr =
            toBinary<ELF64LE>(Storage, YamlString);
        ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());

        Expected<const typename ELF64LE::Shdr *> BBAddrMapSecOrErr =
            ElfOrErr->getELFFile().getSection(1);
        ASSERT_THAT_EXPECTED(BBAddrMapSecOrErr, Succeeded());

        std::vector<PGOAnalysisMap> PGOAnalyses;
        auto BBAddrMaps = ElfOrErr->readBBAddrMap(
            TextSectionIndex, ExpectedPGO ? &PGOAnalyses : nullptr);
        ASSERT_THAT_EXPECTED(BBAddrMaps, Succeeded());
        EXPECT_EQ(*BBAddrMaps, ExpectedResult);
        if (ExpectedPGO) {
          EXPECT_EQ(BBAddrMaps->size(), PGOAnalyses.size());
          EXPECT_EQ(PGOAnalyses, *ExpectedPGO);
        }
      };

  auto DoCheckFails = [&](StringRef YamlString,
                          std::optional<unsigned> TextSectionIndex,
                          const char *ErrMsg) {
    SmallString<0> Storage;
    Expected<ELFObjectFile<ELF64LE>> ElfOrErr =
        toBinary<ELF64LE>(Storage, YamlString);
    ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());

    Expected<const typename ELF64LE::Shdr *> BBAddrMapSecOrErr =
        ElfOrErr->getELFFile().getSection(1);
    ASSERT_THAT_EXPECTED(BBAddrMapSecOrErr, Succeeded());
    std::vector<PGOAnalysisMap> PGOAnalyses;
    EXPECT_THAT_ERROR(
        ElfOrErr->readBBAddrMap(TextSectionIndex, &PGOAnalyses).takeError(),
        FailedWithMessage(ErrMsg));
  };

  // Check that we can retrieve the data in the normal case.
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/std::nullopt,
                  AllBBAddrMaps, std::nullopt);
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/0, Section0BBAddrMaps,
                  std::nullopt);
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/2, Section1BBAddrMaps,
                  std::nullopt);
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/1, Section2BBAddrMaps,
                  std::nullopt);

  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/std::nullopt,
                  AllBBAddrMaps, AllPGOAnalysisMaps);
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/0, Section0BBAddrMaps,
                  Section0PGOAnalysisMaps);
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/2, Section1BBAddrMaps,
                  Section1PGOAnalysisMaps);
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/1, Section2BBAddrMaps,
                  Section2PGOAnalysisMaps);
  // Check that when no bb-address-map section is found for a text section,
  // we return an empty result.
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/3, {}, std::nullopt);
  DoCheckSucceeds(CommonYamlString, /*TextSectionIndex=*/3, {},
                  std::vector<PGOAnalysisMap>{});

  // Check that we detect when a bb-addr-map section is linked to an invalid
  // (not present) section.
  SmallString<128> InvalidLinkedYamlString(CommonYamlString);
  InvalidLinkedYamlString += R"(
    Link: 10
)";

  DoCheckFails(InvalidLinkedYamlString, /*TextSectionIndex=*/4,
               "unable to get the linked-to section for "
               "SHT_LLVM_BB_ADDR_MAP section with index 4: invalid section "
               "index: 10");
  // Linked sections are not checked when we don't target a specific text
  // section.
  DoCheckSucceeds(InvalidLinkedYamlString, /*TextSectionIndex=*/std::nullopt,
                  AllBBAddrMaps, std::nullopt);
  DoCheckSucceeds(InvalidLinkedYamlString, /*TextSectionIndex=*/std::nullopt,
                  AllBBAddrMaps, AllPGOAnalysisMaps);

  // Check that we can detect when bb-address-map decoding fails.
  SmallString<128> TruncatedYamlString(CommonYamlString);
  TruncatedYamlString += R"(
    ShSize: 0xa
)";

  DoCheckFails(TruncatedYamlString, /*TextSectionIndex=*/std::nullopt,
               "unable to read SHT_LLVM_BB_ADDR_MAP section with index 4: "
               "unable to decode LEB128 at offset 0x0000000a: malformed "
               "uleb128, extends past end");
  // Check that we can read the other section's bb-address-maps which are
  // valid.
  DoCheckSucceeds(TruncatedYamlString, /*TextSectionIndex=*/2,
                  Section1BBAddrMaps, std::nullopt);
  DoCheckSucceeds(TruncatedYamlString, /*TextSectionIndex=*/2,
                  Section1BBAddrMaps, Section1PGOAnalysisMaps);
}

// Test for ObjectFile::getRelocatedSection: check that it returns a relocated
// section for executable and relocatable files.
TEST(ELFObjectFileTest, ExecutableWithRelocs) {
  StringRef HeaderString(R"(
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data:  ELFDATA2LSB
)");
  StringRef ContentsString(R"(
Sections:
  - Name:  .text
    Type:  SHT_PROGBITS
    Flags: [ SHF_ALLOC, SHF_EXECINSTR ]
  - Name:  .rela.text
    Type:  SHT_RELA
    Flags: [ SHF_INFO_LINK ]
    Info:  .text
)");

  auto DoCheck = [&](StringRef YamlString) {
    SmallString<0> Storage;
    Expected<ELFObjectFile<ELF64LE>> ElfOrErr =
        toBinary<ELF64LE>(Storage, YamlString);
    ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());
    const ELFObjectFile<ELF64LE> &Obj = *ElfOrErr;

    bool FoundRela;

    for (SectionRef Sec : Obj.sections()) {
      Expected<StringRef> SecNameOrErr = Sec.getName();
      ASSERT_THAT_EXPECTED(SecNameOrErr, Succeeded());
      StringRef SecName = *SecNameOrErr;
      if (SecName != ".rela.text")
        continue;
      FoundRela = true;
      Expected<section_iterator> RelSecOrErr = Sec.getRelocatedSection();
      ASSERT_THAT_EXPECTED(RelSecOrErr, Succeeded());
      section_iterator RelSec = *RelSecOrErr;
      ASSERT_NE(RelSec, Obj.section_end());
      Expected<StringRef> TextSecNameOrErr = RelSec->getName();
      ASSERT_THAT_EXPECTED(TextSecNameOrErr, Succeeded());
      StringRef TextSecName = *TextSecNameOrErr;
      EXPECT_EQ(TextSecName, ".text");
    }
    ASSERT_TRUE(FoundRela);
  };

  // Check ET_EXEC file (`ld --emit-relocs` use-case).
  SmallString<128> ExecFileYamlString(HeaderString);
  ExecFileYamlString += R"(
  Type:  ET_EXEC
)";
  ExecFileYamlString += ContentsString;
  DoCheck(ExecFileYamlString);

  // Check ET_REL file.
  SmallString<128> RelocatableFileYamlString(HeaderString);
  RelocatableFileYamlString += R"(
  Type:  ET_REL
)";
  RelocatableFileYamlString += ContentsString;
  DoCheck(RelocatableFileYamlString);
}

TEST(ELFObjectFileTest, GetSectionAndRelocations) {
  StringRef HeaderString(R"(
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data:  ELFDATA2LSB
  Type:  ET_EXEC
)");

  using Elf_Shdr = Elf_Shdr_Impl<ELF64LE>;

  auto DoCheckSucceeds = [&](StringRef ContentsString,
                             std::function<Expected<bool>(const Elf_Shdr &)>
                                 Matcher) {
    SmallString<0> Storage;
    SmallString<128> FullYamlString(HeaderString);
    FullYamlString += ContentsString;
    Expected<ELFObjectFile<ELF64LE>> ElfOrErr =
        toBinary<ELF64LE>(Storage, FullYamlString);
    ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());

    Expected<MapVector<const Elf_Shdr *, const Elf_Shdr *>> SecToRelocMapOrErr =
        ElfOrErr->getELFFile().getSectionAndRelocations(Matcher);
    ASSERT_THAT_EXPECTED(SecToRelocMapOrErr, Succeeded());

    // Basic verification to make sure we have the correct section types.
    for (auto const &[Sec, RelaSec] : *SecToRelocMapOrErr) {
      ASSERT_EQ(Sec->sh_type, ELF::SHT_PROGBITS);
      ASSERT_EQ(RelaSec->sh_type, ELF::SHT_RELA);
    }
  };

  auto DoCheckFails = [&](StringRef ContentsString,
                          std::function<Expected<bool>(const Elf_Shdr &)>
                              Matcher,
                          const char *ErrorMessage) {
    SmallString<0> Storage;
    SmallString<128> FullYamlString(HeaderString);
    FullYamlString += ContentsString;
    Expected<ELFObjectFile<ELF64LE>> ElfOrErr =
        toBinary<ELF64LE>(Storage, FullYamlString);
    ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());

    Expected<MapVector<const Elf_Shdr *, const Elf_Shdr *>> SecToRelocMapOrErr =
        ElfOrErr->getELFFile().getSectionAndRelocations(Matcher);
    ASSERT_THAT_ERROR(SecToRelocMapOrErr.takeError(),
                      FailedWithMessage(ErrorMessage));
  };

  auto DefaultMatcher = [](const Elf_Shdr &Sec) -> bool {
    return Sec.sh_type == ELF::SHT_PROGBITS;
  };

  StringRef TwoTextSections = R"(
Sections:
  - Name: .text
    Type: SHT_PROGBITS
    Flags: [ SHF_ALLOC, SHF_EXECINSTR ]
  - Name: .rela.text
    Type: SHT_RELA
    Flags: [ SHF_INFO_LINK ]
    Info: .text
  - Name: .text2
    Type: SHT_PROGBITS
    Flags: [ SHF_ALLOC, SHF_EXECINSTR ]
  - Name: .rela.text2
    Type: SHT_RELA
    Flags: [ SHF_INFO_LINK ]
    Info: .text2
)";
  DoCheckSucceeds(TwoTextSections, DefaultMatcher);

  StringRef OneTextSection = R"(
Sections:
  - Name: .text
    Type: SHT_PROGBITS
    Flags: [ SHF_ALLOC, SHF_EXECINSTR ]
)";

  auto ErroringMatcher = [](const Elf_Shdr &Sec) -> Expected<bool> {
    if(Sec.sh_type == ELF::SHT_PROGBITS)
      return createError("This was supposed to fail.");
    return false;
  };

  DoCheckFails(OneTextSection, ErroringMatcher,
               "This was supposed to fail.");

  StringRef MissingRelocatableContent = R"(
Sections:
  - Name: .rela.text
    Type: SHT_RELA
    Flags: [ SHF_INFO_LINK ]
    Info: 0xFF
)";

  DoCheckFails(MissingRelocatableContent, DefaultMatcher,
               "SHT_RELA section with index 1: failed to get a "
               "relocated section: invalid section index: 255");
}
