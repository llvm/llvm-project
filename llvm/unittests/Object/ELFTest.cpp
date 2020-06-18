//===- ELFTest.cpp - Tests for ELF.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELF.h"
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
  EXPECT_EQ(0U, getELFRelativeRelocationType(EM_VE));
}
