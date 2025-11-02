//===- ELFTest.cpp - Tests for ELF.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

TEST(ELFTest, getELFRelocationTypeNameForVE) {
  EXPECT_EQ("R_VE_NONE", getELFRelocationTypeName(EM_VE, R_VE_NONE));
  EXPECT_EQ("R_VE_REFLONG", getELFRelocationTypeName(EM_VE, R_VE_REFLONG));
  EXPECT_EQ("R_VE_REFQUAD", getELFRelocationTypeName(EM_VE, R_VE_REFQUAD));
  EXPECT_EQ("R_VE_SREL32", getELFRelocationTypeName(EM_VE, R_VE_SREL32));
  EXPECT_EQ("R_VE_HI32", getELFRelocationTypeName(EM_VE, R_VE_HI32));
  EXPECT_EQ("R_VE_LO32", getELFRelocationTypeName(EM_VE, R_VE_LO32));
  EXPECT_EQ("R_VE_PC_HI32", getELFRelocationTypeName(EM_VE, R_VE_PC_HI32));
  EXPECT_EQ("R_VE_PC_LO32", getELFRelocationTypeName(EM_VE, R_VE_PC_LO32));
  EXPECT_EQ("R_VE_GOT32", getELFRelocationTypeName(EM_VE, R_VE_GOT32));
  EXPECT_EQ("R_VE_GOT_HI32", getELFRelocationTypeName(EM_VE, R_VE_GOT_HI32));
  EXPECT_EQ("R_VE_GOT_LO32", getELFRelocationTypeName(EM_VE, R_VE_GOT_LO32));
  EXPECT_EQ("R_VE_GOTOFF32", getELFRelocationTypeName(EM_VE, R_VE_GOTOFF32));
  EXPECT_EQ("R_VE_GOTOFF_HI32",
            getELFRelocationTypeName(EM_VE, R_VE_GOTOFF_HI32));
  EXPECT_EQ("R_VE_GOTOFF_LO32",
            getELFRelocationTypeName(EM_VE, R_VE_GOTOFF_LO32));
  EXPECT_EQ("R_VE_PLT32", getELFRelocationTypeName(EM_VE, R_VE_PLT32));
  EXPECT_EQ("R_VE_PLT_HI32", getELFRelocationTypeName(EM_VE, R_VE_PLT_HI32));
  EXPECT_EQ("R_VE_PLT_LO32", getELFRelocationTypeName(EM_VE, R_VE_PLT_LO32));
  EXPECT_EQ("R_VE_RELATIVE", getELFRelocationTypeName(EM_VE, R_VE_RELATIVE));
  EXPECT_EQ("R_VE_GLOB_DAT", getELFRelocationTypeName(EM_VE, R_VE_GLOB_DAT));
  EXPECT_EQ("R_VE_JUMP_SLOT", getELFRelocationTypeName(EM_VE, R_VE_JUMP_SLOT));
  EXPECT_EQ("R_VE_COPY", getELFRelocationTypeName(EM_VE, R_VE_COPY));
  EXPECT_EQ("R_VE_DTPMOD64", getELFRelocationTypeName(EM_VE, R_VE_DTPMOD64));
  EXPECT_EQ("R_VE_DTPOFF64", getELFRelocationTypeName(EM_VE, R_VE_DTPOFF64));
  EXPECT_EQ("R_VE_TLS_GD_HI32",
            getELFRelocationTypeName(EM_VE, R_VE_TLS_GD_HI32));
  EXPECT_EQ("R_VE_TLS_GD_LO32",
            getELFRelocationTypeName(EM_VE, R_VE_TLS_GD_LO32));
  EXPECT_EQ("R_VE_TPOFF_HI32",
            getELFRelocationTypeName(EM_VE, R_VE_TPOFF_HI32));
  EXPECT_EQ("R_VE_TPOFF_LO32",
            getELFRelocationTypeName(EM_VE, R_VE_TPOFF_LO32));
  EXPECT_EQ("R_VE_CALL_HI32", getELFRelocationTypeName(EM_VE, R_VE_CALL_HI32));
  EXPECT_EQ("R_VE_CALL_LO32", getELFRelocationTypeName(EM_VE, R_VE_CALL_LO32));
}

TEST(ELFTest, getELFRelocationTypeNameForLoongArch) {
  EXPECT_EQ("R_LARCH_NONE",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_NONE));
  EXPECT_EQ("R_LARCH_32", getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_32));
  EXPECT_EQ("R_LARCH_64", getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_64));
  EXPECT_EQ("R_LARCH_RELATIVE",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_RELATIVE));
  EXPECT_EQ("R_LARCH_COPY",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_COPY));
  EXPECT_EQ("R_LARCH_JUMP_SLOT",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_JUMP_SLOT));
  EXPECT_EQ("R_LARCH_TLS_DTPMOD32",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_DTPMOD32));
  EXPECT_EQ("R_LARCH_TLS_DTPMOD64",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_DTPMOD64));
  EXPECT_EQ("R_LARCH_TLS_DTPREL32",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_DTPREL32));
  EXPECT_EQ("R_LARCH_TLS_DTPREL64",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_DTPREL64));
  EXPECT_EQ("R_LARCH_TLS_TPREL32",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_TPREL32));
  EXPECT_EQ("R_LARCH_TLS_TPREL64",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_TPREL64));
  EXPECT_EQ("R_LARCH_IRELATIVE",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_IRELATIVE));

  EXPECT_EQ("R_LARCH_MARK_LA",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_MARK_LA));
  EXPECT_EQ("R_LARCH_MARK_PCREL",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_MARK_PCREL));
  EXPECT_EQ("R_LARCH_SOP_PUSH_PCREL",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_PUSH_PCREL));
  EXPECT_EQ("R_LARCH_SOP_PUSH_ABSOLUTE",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_PUSH_ABSOLUTE));
  EXPECT_EQ("R_LARCH_SOP_PUSH_DUP",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_PUSH_DUP));
  EXPECT_EQ("R_LARCH_SOP_PUSH_GPREL",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_PUSH_GPREL));
  EXPECT_EQ("R_LARCH_SOP_PUSH_TLS_TPREL",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_PUSH_TLS_TPREL));
  EXPECT_EQ("R_LARCH_SOP_PUSH_TLS_GOT",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_PUSH_TLS_GOT));
  EXPECT_EQ("R_LARCH_SOP_PUSH_TLS_GD",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_PUSH_TLS_GD));
  EXPECT_EQ("R_LARCH_SOP_PUSH_PLT_PCREL",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_PUSH_PLT_PCREL));
  EXPECT_EQ("R_LARCH_SOP_ASSERT",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_ASSERT));
  EXPECT_EQ("R_LARCH_SOP_NOT",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_NOT));
  EXPECT_EQ("R_LARCH_SOP_SUB",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_SUB));
  EXPECT_EQ("R_LARCH_SOP_SL",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_SL));
  EXPECT_EQ("R_LARCH_SOP_SR",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_SR));
  EXPECT_EQ("R_LARCH_SOP_ADD",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_ADD));
  EXPECT_EQ("R_LARCH_SOP_AND",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_AND));
  EXPECT_EQ("R_LARCH_SOP_IF_ELSE",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_IF_ELSE));
  EXPECT_EQ("R_LARCH_SOP_POP_32_S_10_5",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_POP_32_S_10_5));
  EXPECT_EQ("R_LARCH_SOP_POP_32_U_10_12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_POP_32_U_10_12));
  EXPECT_EQ("R_LARCH_SOP_POP_32_S_10_12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_POP_32_S_10_12));
  EXPECT_EQ("R_LARCH_SOP_POP_32_S_10_16",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_POP_32_S_10_16));
  EXPECT_EQ(
      "R_LARCH_SOP_POP_32_S_10_16_S2",
      getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_POP_32_S_10_16_S2));
  EXPECT_EQ("R_LARCH_SOP_POP_32_S_5_20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_POP_32_S_5_20));
  EXPECT_EQ("R_LARCH_SOP_POP_32_S_0_5_10_16_S2",
            getELFRelocationTypeName(EM_LOONGARCH,
                                     R_LARCH_SOP_POP_32_S_0_5_10_16_S2));
  EXPECT_EQ("R_LARCH_SOP_POP_32_S_0_10_10_16_S2",
            getELFRelocationTypeName(EM_LOONGARCH,
                                     R_LARCH_SOP_POP_32_S_0_10_10_16_S2));
  EXPECT_EQ("R_LARCH_SOP_POP_32_U",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SOP_POP_32_U));
  EXPECT_EQ("R_LARCH_ADD8",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_ADD8));
  EXPECT_EQ("R_LARCH_ADD16",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_ADD16));
  EXPECT_EQ("R_LARCH_ADD24",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_ADD24));
  EXPECT_EQ("R_LARCH_ADD32",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_ADD32));
  EXPECT_EQ("R_LARCH_ADD64",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_ADD64));
  EXPECT_EQ("R_LARCH_SUB8",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SUB8));
  EXPECT_EQ("R_LARCH_SUB16",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SUB16));
  EXPECT_EQ("R_LARCH_SUB24",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SUB24));
  EXPECT_EQ("R_LARCH_SUB32",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SUB32));
  EXPECT_EQ("R_LARCH_SUB64",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SUB64));
  EXPECT_EQ("R_LARCH_GNU_VTINHERIT",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_GNU_VTINHERIT));
  EXPECT_EQ("R_LARCH_GNU_VTENTRY",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_GNU_VTENTRY));
  EXPECT_EQ("R_LARCH_B16",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_B16));
  EXPECT_EQ("R_LARCH_B21",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_B21));
  EXPECT_EQ("R_LARCH_B26",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_B26));
  EXPECT_EQ("R_LARCH_ABS_HI20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_ABS_HI20));
  EXPECT_EQ("R_LARCH_ABS_LO12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_ABS_LO12));
  EXPECT_EQ("R_LARCH_ABS64_LO20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_ABS64_LO20));
  EXPECT_EQ("R_LARCH_ABS64_HI12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_ABS64_HI12));
  EXPECT_EQ("R_LARCH_PCALA_HI20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_PCALA_HI20));
  EXPECT_EQ("R_LARCH_PCALA_LO12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_PCALA_LO12));
  EXPECT_EQ("R_LARCH_PCALA64_LO20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_PCALA64_LO20));
  EXPECT_EQ("R_LARCH_PCALA64_HI12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_PCALA64_HI12));
  EXPECT_EQ("R_LARCH_GOT_PC_HI20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_GOT_PC_HI20));
  EXPECT_EQ("R_LARCH_GOT_PC_LO12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_GOT_PC_LO12));
  EXPECT_EQ("R_LARCH_GOT64_PC_LO20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_GOT64_PC_LO20));
  EXPECT_EQ("R_LARCH_GOT64_PC_HI12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_GOT64_PC_HI12));
  EXPECT_EQ("R_LARCH_GOT_HI20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_GOT_HI20));
  EXPECT_EQ("R_LARCH_GOT_LO12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_GOT_LO12));
  EXPECT_EQ("R_LARCH_GOT64_LO20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_GOT64_LO20));
  EXPECT_EQ("R_LARCH_GOT64_HI12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_GOT64_HI12));
  EXPECT_EQ("R_LARCH_TLS_LE_HI20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_LE_HI20));
  EXPECT_EQ("R_LARCH_TLS_LE_LO12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_LE_LO12));
  EXPECT_EQ("R_LARCH_TLS_LE64_LO20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_LE64_LO20));
  EXPECT_EQ("R_LARCH_TLS_LE64_HI12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_LE64_HI12));
  EXPECT_EQ("R_LARCH_TLS_IE_PC_HI20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_IE_PC_HI20));
  EXPECT_EQ("R_LARCH_TLS_IE_PC_LO12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_IE_PC_LO12));
  EXPECT_EQ("R_LARCH_TLS_IE64_PC_LO20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_IE64_PC_LO20));
  EXPECT_EQ("R_LARCH_TLS_IE64_PC_HI12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_IE64_PC_HI12));
  EXPECT_EQ("R_LARCH_TLS_IE_HI20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_IE_HI20));
  EXPECT_EQ("R_LARCH_TLS_IE_LO12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_IE_LO12));
  EXPECT_EQ("R_LARCH_TLS_IE64_LO20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_IE64_LO20));
  EXPECT_EQ("R_LARCH_TLS_IE64_HI12",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_IE64_HI12));
  EXPECT_EQ("R_LARCH_TLS_LD_PC_HI20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_LD_PC_HI20));
  EXPECT_EQ("R_LARCH_TLS_LD_HI20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_LD_HI20));
  EXPECT_EQ("R_LARCH_TLS_GD_PC_HI20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_GD_PC_HI20));
  EXPECT_EQ("R_LARCH_TLS_GD_HI20",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_TLS_GD_HI20));
  EXPECT_EQ("R_LARCH_32_PCREL",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_32_PCREL));
  EXPECT_EQ("R_LARCH_RELAX",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_RELAX));
  EXPECT_EQ("R_LARCH_ALIGN",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_ALIGN));
  EXPECT_EQ("R_LARCH_PCREL20_S2",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_PCREL20_S2));
  EXPECT_EQ("R_LARCH_ADD6",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_ADD6));
  EXPECT_EQ("R_LARCH_SUB6",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SUB6));
  EXPECT_EQ("R_LARCH_ADD_ULEB128",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_ADD_ULEB128));
  EXPECT_EQ("R_LARCH_SUB_ULEB128",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_SUB_ULEB128));
  EXPECT_EQ("R_LARCH_64_PCREL",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_64_PCREL));
  EXPECT_EQ("R_LARCH_CALL36",
            getELFRelocationTypeName(EM_LOONGARCH, R_LARCH_CALL36));
}

TEST(ELFTest, getELFRelativeRelocationType) {
  EXPECT_EQ(ELF::R_VE_RELATIVE, getELFRelativeRelocationType(EM_VE));
  EXPECT_EQ(ELF::R_LARCH_RELATIVE, getELFRelativeRelocationType(EM_LOONGARCH));
}

// This is a test for the DataRegion helper struct, defined in ELF.h header.
TEST(ELFTest, DataRegionTest) {
  std::vector<uint8_t> Data = {0, 1, 2};

  // Used to check that the operator[] works properly.
  auto CheckOperator = [&](DataRegion<uint8_t> &R) {
    for (size_t I = 0, E = Data.size(); I != E; ++I) {
      Expected<uint8_t> ValOrErr = R[I];
      ASSERT_THAT_EXPECTED(ValOrErr, Succeeded());
      EXPECT_EQ(*ValOrErr, I);
    }
  };

  // Check we can use the constructor that takes an ArrayRef<T>.
  DataRegion<uint8_t> Region(Data);

  CheckOperator(Region);
  const char *ErrMsg1 =
      "the index is greater than or equal to the number of entries (3)";
  EXPECT_THAT_ERROR(Region[3].takeError(), FailedWithMessage(ErrMsg1));
  EXPECT_THAT_ERROR(Region[4].takeError(), FailedWithMessage(ErrMsg1));

  // Check we can use the constructor that takes the data begin and the
  // data end pointers.
  Region = {Data.data(), Data.data() + Data.size()};

  CheckOperator(Region);
  const char *ErrMsg2 = "can't read past the end of the file";
  EXPECT_THAT_ERROR(Region[3].takeError(), FailedWithMessage(ErrMsg2));
  EXPECT_THAT_ERROR(Region[4].takeError(), FailedWithMessage(ErrMsg2));
}

// Test the sysV and the gnu hash functions, particularly with UTF-8 unicode.
// Use names long enough for the hash's recycling of the high bits to kick in.
// Explicitly encode the UTF-8 to avoid encoding transliterations.
TEST(ELFTest, Hash) {
  EXPECT_EQ(hashSysV("FooBarBazToto"), 0x5ec3e8fU);
  EXPECT_EQ(hashGnu("FooBarBazToto"), 0x5478be61U);

  // boom💥pants
  EXPECT_EQ(hashSysV("boom\xf0\x9f\x92\xa5pants"), 0x5a0cf53U);
  EXPECT_EQ(hashGnu("boom\xf0\x9f\x92\xa5pants"), 0xf5dda2deU);

  // woot!🧙 💑 🌈
  EXPECT_EQ(hashSysV("woot!\xf0\x9f\xa7\x99 \xf0\x9f\x92\x91 "
                     "\xf0\x9f\x8c\x88"), 0x3522e38U);
  EXPECT_EQ(hashGnu("woot!\xf0\x9f\xa7\x99 \xf0\x9f\x92\x91 "
                    "\xf0\x9f\x8c\x88"), 0xf7603f3U);

  // This string hashes to 0x100000000 in the originally formulated function,
  // when long is 64 bits -- but that was never the intent. The code was
  // presuming 32-bit long. Thus make sure that extra bit doesn't appear. 
  EXPECT_EQ(hashSysV("ZZZZZW9p"), 0U);
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

TEST(ELFObjectFileTest, ELFNoteIteratorOverflow) {
  using Elf_Shdr_Range = ELFFile<ELF64LE>::Elf_Shdr_Range;
  using Elf_Phdr_Range = ELFFile<ELF64LE>::Elf_Phdr_Range;

  SmallString<0> Storage;
  Expected<ELFObjectFile<ELF64LE>> ElfOrErr = toBinary<ELF64LE>(Storage, R"(
--- !ELF
FileHeader:
  Class:          ELFCLASS64
  Data:           ELFDATA2LSB
  Type:           ET_EXEC
  Machine:        EM_X86_64
ProgramHeaders:
  - Type:         PT_NOTE
    FileSize:     0xffffffffffffff88
    FirstSec:     .note.gnu.build-id
    LastSec:      .note.gnu.build-id
Sections:
  - Name:         .note.gnu.build-id
    Type:         SHT_NOTE
    AddressAlign: 0x04
    ShOffset:     0xffffffffffffff88
    Notes:
      - Name:     "GNU"
        Desc:     "abb50d82b6bdc861"
        Type:     3
)");
  ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());
  ELFFile<ELF64LE> Obj = ElfOrErr.get().getELFFile();

  auto CheckOverflow = [&](auto &&PhdrOrShdr, uint64_t Offset, uint64_t Size) {
    Error Err = Error::success();
    Obj.notes(PhdrOrShdr, Err);

    std::string ErrMessage;
    handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EI) {
      ErrMessage = EI.message();
    });

    EXPECT_EQ(ErrMessage, ("invalid offset (0x" + Twine::utohexstr(Offset) +
                           ") or size (0x" + Twine::utohexstr(Size) + ")")
                              .str());
  };

  Expected<Elf_Phdr_Range> PhdrsOrErr = Obj.program_headers();
  EXPECT_FALSE(!PhdrsOrErr);
  for (Elf_Phdr_Impl<ELF64LE> P : *PhdrsOrErr)
    if (P.p_type == ELF::PT_NOTE)
      CheckOverflow(P, P.p_offset, P.p_filesz);

  Expected<Elf_Shdr_Range> ShdrsOrErr = Obj.sections();
  EXPECT_FALSE(!ShdrsOrErr);
  for (Elf_Shdr_Impl<ELF64LE> S : *ShdrsOrErr)
    if (S.sh_type == ELF::SHT_NOTE)
      CheckOverflow(S, S.sh_offset, S.sh_size);
}
