//===- ELFYAMLTest.cpp - Tests for ELFYAML.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

template <class ELFT>
static Expected<ELFObjectFile<ELFT>> toBinary(SmallVectorImpl<char> &Storage,
                                              StringRef Yaml) {
  Storage.clear();
  raw_svector_ostream OS(Storage);
  yaml::Input YIn(Yaml);
  if (!yaml::convertYAML(YIn, OS, [](const Twine &Msg) {}))
    return createStringError(std::errc::invalid_argument,
                             "unable to convert YAML");

  return ELFObjectFile<ELFT>::create(MemoryBufferRef(OS.str(), "Binary"));
}

TEST(ELFRelocationTypeTest, RelocationTestForVE) {
  SmallString<0> Storage;
  Expected<ELFObjectFile<ELF64LE>> ExpectedFile = toBinary<ELF64LE>(Storage, R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_REL
  Machine: EM_VE
Sections:
  - Name: .rela.text
    Type: SHT_RELA
    Relocations:
      - Type: R_VE_NONE
      - Type: R_VE_REFLONG
      - Type: R_VE_REFQUAD
      - Type: R_VE_SREL32
      - Type: R_VE_HI32
      - Type: R_VE_LO32
      - Type: R_VE_PC_HI32
      - Type: R_VE_PC_LO32
      - Type: R_VE_GOT32
      - Type: R_VE_GOT_HI32
      - Type: R_VE_GOT_LO32
      - Type: R_VE_GOTOFF32
      - Type: R_VE_GOTOFF_HI32
      - Type: R_VE_GOTOFF_LO32
      - Type: R_VE_PLT32
      - Type: R_VE_PLT_HI32
      - Type: R_VE_PLT_LO32
      - Type: R_VE_RELATIVE
      - Type: R_VE_GLOB_DAT
      - Type: R_VE_JUMP_SLOT
      - Type: R_VE_COPY
      - Type: R_VE_DTPMOD64
      - Type: R_VE_DTPOFF64
      - Type: R_VE_TLS_GD_HI32
      - Type: R_VE_TLS_GD_LO32
      - Type: R_VE_TPOFF_HI32
      - Type: R_VE_TPOFF_LO32
      - Type: R_VE_CALL_HI32
      - Type: R_VE_CALL_LO32)");
  ASSERT_THAT_EXPECTED(ExpectedFile, Succeeded());
  const ELFObjectFile<ELF64LE> &File = *ExpectedFile;
  EXPECT_EQ("elf64-ve", File.getFileFormatName());
  EXPECT_EQ(Triple::ve, File.getArch());

  // Test relocation types.
  for (SectionRef Sec : File.sections()) {
    Expected<StringRef> NameOrErr = Sec.getName();
    ASSERT_THAT_EXPECTED(NameOrErr, Succeeded());
    StringRef SectionName = *NameOrErr;
    if (SectionName != ".rela.text")
      continue;

    for (RelocationRef R : Sec.relocations()) {
      SmallString<32> RelTypeName;
      using namespace llvm::ELF;

#define NAME_CHECK(ID)                                                         \
  case ID:                                                                     \
    R.getTypeName(RelTypeName);                                                \
    EXPECT_EQ(#ID, RelTypeName);                                               \
    break

      switch (R.getType()) {
        NAME_CHECK(R_VE_NONE);
        NAME_CHECK(R_VE_REFLONG);
        NAME_CHECK(R_VE_REFQUAD);
        NAME_CHECK(R_VE_SREL32);
        NAME_CHECK(R_VE_HI32);
        NAME_CHECK(R_VE_LO32);
        NAME_CHECK(R_VE_PC_HI32);
        NAME_CHECK(R_VE_PC_LO32);
        NAME_CHECK(R_VE_GOT32);
        NAME_CHECK(R_VE_GOT_HI32);
        NAME_CHECK(R_VE_GOT_LO32);
        NAME_CHECK(R_VE_GOTOFF32);
        NAME_CHECK(R_VE_GOTOFF_HI32);
        NAME_CHECK(R_VE_GOTOFF_LO32);
        NAME_CHECK(R_VE_PLT32);
        NAME_CHECK(R_VE_PLT_HI32);
        NAME_CHECK(R_VE_PLT_LO32);
        NAME_CHECK(R_VE_RELATIVE);
        NAME_CHECK(R_VE_GLOB_DAT);
        NAME_CHECK(R_VE_JUMP_SLOT);
        NAME_CHECK(R_VE_COPY);
        NAME_CHECK(R_VE_DTPMOD64);
        NAME_CHECK(R_VE_DTPOFF64);
        NAME_CHECK(R_VE_TLS_GD_HI32);
        NAME_CHECK(R_VE_TLS_GD_LO32);
        NAME_CHECK(R_VE_TPOFF_HI32);
        NAME_CHECK(R_VE_TPOFF_LO32);
        NAME_CHECK(R_VE_CALL_HI32);
        NAME_CHECK(R_VE_CALL_LO32);
      default:
        FAIL() << "Found unexpected relocation type: " + Twine(R.getType());
        break;
      }
    }
  }
}
