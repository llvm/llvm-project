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
