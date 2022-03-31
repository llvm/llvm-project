//===- ELFTest.cpp - Tests for ELF.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELF.h"
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
}

TEST(ELFTest, getELFRelativeRelocationType) {
  EXPECT_EQ(ELF::R_VE_RELATIVE, getELFRelativeRelocationType(EM_VE));
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
