//===- llvm/unittest/CodeGen/DIETest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/DIE.h"
#include "TestAsmPrinter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using testing::_;
using testing::SaveArg;

namespace {

using DIETestParams =
    std::tuple<unsigned, dwarf::DwarfFormat, dwarf::Form, unsigned>;

class DIEFixtureBase : public testing::TestWithParam<DIETestParams> {
protected:
  void SetUp() override {
    unsigned Version;
    dwarf::DwarfFormat Format;
    std::tie(Version, Format, Form, Size) = GetParam();
    auto ExpectedTestPrinter =
        TestAsmPrinter::create("x86_64-pc-linux", Version, Format);
    ASSERT_THAT_EXPECTED(ExpectedTestPrinter, Succeeded());
    TestPrinter = std::move(ExpectedTestPrinter.get());
    if (!TestPrinter)
      GTEST_SKIP();
  }

  dwarf::Form Form;
  unsigned Size;
  std::unique_ptr<TestAsmPrinter> TestPrinter;
};

struct DIEExprFixture : public DIEFixtureBase {
  void SetUp() override {
    DIEFixtureBase::SetUp();
    if (!TestPrinter)
      return;

    Val = MCConstantExpr::create(42, TestPrinter->getCtx());
  }

  const MCExpr *Val = nullptr;
};

TEST_P(DIEExprFixture, SizeOf) {
  DIEExpr Tst(Val);
  EXPECT_EQ(Size, Tst.SizeOf(TestPrinter->getAP(), Form));
}

TEST_P(DIEExprFixture, EmitValue) {
  DIEExpr Tst(Val);
  EXPECT_CALL(TestPrinter->getMS(), emitValueImpl(Val, Size, _));
  Tst.emitValue(TestPrinter->getAP(), Form);
}

INSTANTIATE_TEST_SUITE_P(
    DIETestParams, DIEExprFixture,
    testing::Values(
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_data4, 4u},
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_data8, 8u},
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_sec_offset, 4u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_data4, 4u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_data8, 8u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_sec_offset, 8u}));

struct DIELabelFixture : public DIEFixtureBase {
  void SetUp() override {
    DIEFixtureBase::SetUp();
    if (!TestPrinter)
      return;

    Val = TestPrinter->getCtx().createTempSymbol();
  }

  const MCSymbol *Val = nullptr;
};

TEST_P(DIELabelFixture, SizeOf) {
  DIELabel Tst(Val);
  EXPECT_EQ(Size, Tst.SizeOf(TestPrinter->getAP(), Form));
}

TEST_P(DIELabelFixture, EmitValue) {
  DIELabel Tst(Val);

  const MCExpr *Arg0 = nullptr;
  EXPECT_CALL(TestPrinter->getMS(), emitValueImpl(_, Size, _))
      .WillOnce(SaveArg<0>(&Arg0));
  Tst.emitValue(TestPrinter->getAP(), Form);

  const MCSymbolRefExpr *ActualArg0 = dyn_cast_or_null<MCSymbolRefExpr>(Arg0);
  ASSERT_NE(ActualArg0, nullptr);
  EXPECT_EQ(&(ActualArg0->getSymbol()), Val);
}

INSTANTIATE_TEST_SUITE_P(
    DIETestParams, DIELabelFixture,
    testing::Values(
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_data4, 4u},
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_data8, 8u},
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_sec_offset, 4u},
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_strp, 4u},
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_addr, 8u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_data4, 4u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_data8, 8u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_sec_offset, 8u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_strp, 8u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_addr, 8u}));

struct DIEDeltaFixture : public DIEFixtureBase {
  void SetUp() override {
    DIEFixtureBase::SetUp();
    if (!TestPrinter)
      return;

    Hi = TestPrinter->getCtx().createTempSymbol();
    Lo = TestPrinter->getCtx().createTempSymbol();
  }

  const MCSymbol *Hi = nullptr;
  const MCSymbol *Lo = nullptr;
};

TEST_P(DIEDeltaFixture, SizeOf) {
  DIEDelta Tst(Hi, Lo);
  EXPECT_EQ(Size, Tst.SizeOf(TestPrinter->getAP(), Form));
}

TEST_P(DIEDeltaFixture, EmitValue) {
  DIEDelta Tst(Hi, Lo);
  EXPECT_CALL(TestPrinter->getMS(), emitAbsoluteSymbolDiff(Hi, Lo, Size));
  Tst.emitValue(TestPrinter->getAP(), Form);
}

INSTANTIATE_TEST_SUITE_P(
    DIETestParams, DIEDeltaFixture,
    testing::Values(
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_data4, 4u},
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_data8, 8u},
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_sec_offset, 4u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_data4, 4u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_data8, 8u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_sec_offset, 8u}));

struct DIELocListFixture : public DIEFixtureBase {
  void SetUp() override { DIEFixtureBase::SetUp(); }
};

TEST_P(DIELocListFixture, SizeOf) {
  DIELocList Tst(999);
  EXPECT_EQ(Size, Tst.SizeOf(TestPrinter->getAP(), Form));
}

INSTANTIATE_TEST_SUITE_P(
    DIETestParams, DIELocListFixture,
    testing::Values(
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_loclistx, 2u},
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_data4, 4u},
        DIETestParams{4, dwarf::DWARF32, dwarf::DW_FORM_sec_offset, 4u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_loclistx, 2u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_data8, 8u},
        DIETestParams{4, dwarf::DWARF64, dwarf::DW_FORM_sec_offset, 8u}));

} // end namespace
