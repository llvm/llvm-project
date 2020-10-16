//===- llvm/unittest/CodeGen/AsmPrinterDwarfTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestAsmPrinter.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using testing::_;
using testing::InSequence;
using testing::SaveArg;

namespace {

class AsmPrinterFixtureBase : public testing::Test {
  void setupTestPrinter(const std::string &TripleStr, unsigned DwarfVersion,
                        dwarf::DwarfFormat DwarfFormat) {
    auto ExpectedTestPrinter =
        TestAsmPrinter::create(TripleStr, DwarfVersion, DwarfFormat);
    ASSERT_THAT_EXPECTED(ExpectedTestPrinter, Succeeded());
    TestPrinter = std::move(ExpectedTestPrinter.get());
  }

protected:
  bool init(const std::string &TripleStr, unsigned DwarfVersion,
            dwarf::DwarfFormat DwarfFormat) {
    setupTestPrinter(TripleStr, DwarfVersion, DwarfFormat);
    return TestPrinter != nullptr;
  }

  std::unique_ptr<TestAsmPrinter> TestPrinter;
};

class AsmPrinterEmitDwarfSymbolReferenceTest : public AsmPrinterFixtureBase {
protected:
  bool init(const std::string &TripleStr, unsigned DwarfVersion,
            dwarf::DwarfFormat DwarfFormat) {
    if (!AsmPrinterFixtureBase::init(TripleStr, DwarfVersion, DwarfFormat))
      return false;

    // Create a symbol which will be emitted in the tests and associate it
    // with a section because that is required in some code paths.

    Val = TestPrinter->getCtx().createTempSymbol();
    Sec = TestPrinter->getCtx().getELFSection(".tst", ELF::SHT_PROGBITS, 0);
    SecBeginSymbol = Sec->getBeginSymbol();
    TestPrinter->getMS().SwitchSection(Sec);
    TestPrinter->getMS().emitLabel(Val);
    return true;
  }

  MCSymbol *Val = nullptr;
  MCSection *Sec = nullptr;
  MCSymbol *SecBeginSymbol = nullptr;
};

TEST_F(AsmPrinterEmitDwarfSymbolReferenceTest, COFF) {
  if (!init("x86_64-pc-windows", /*DwarfVersion=*/4, dwarf::DWARF32))
    return;

  EXPECT_CALL(TestPrinter->getMS(), EmitCOFFSecRel32(Val, 0));
  TestPrinter->getAP()->emitDwarfSymbolReference(Val, false);
}

TEST_F(AsmPrinterEmitDwarfSymbolReferenceTest, COFFForceOffset) {
  if (!init("x86_64-pc-windows", /*DwarfVersion=*/4, dwarf::DWARF32))
    return;

  EXPECT_CALL(TestPrinter->getMS(),
              emitAbsoluteSymbolDiff(Val, SecBeginSymbol, 4));
  TestPrinter->getAP()->emitDwarfSymbolReference(Val, true);
}

TEST_F(AsmPrinterEmitDwarfSymbolReferenceTest, ELFDWARF32) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    return;

  const MCExpr *Arg0 = nullptr;
  EXPECT_CALL(TestPrinter->getMS(), emitValueImpl(_, 4, _))
      .WillOnce(SaveArg<0>(&Arg0));
  TestPrinter->getAP()->emitDwarfSymbolReference(Val, false);

  const MCSymbolRefExpr *ActualArg0 = dyn_cast_or_null<MCSymbolRefExpr>(Arg0);
  ASSERT_NE(ActualArg0, nullptr);
  EXPECT_EQ(&(ActualArg0->getSymbol()), Val);
}

TEST_F(AsmPrinterEmitDwarfSymbolReferenceTest, ELFDWARF32ForceOffset) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    return;

  EXPECT_CALL(TestPrinter->getMS(),
              emitAbsoluteSymbolDiff(Val, SecBeginSymbol, 4));
  TestPrinter->getAP()->emitDwarfSymbolReference(Val, true);
}

TEST_F(AsmPrinterEmitDwarfSymbolReferenceTest, ELFDWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    return;

  const MCExpr *Arg0 = nullptr;
  EXPECT_CALL(TestPrinter->getMS(), emitValueImpl(_, 8, _))
      .WillOnce(SaveArg<0>(&Arg0));
  TestPrinter->getAP()->emitDwarfSymbolReference(Val, false);

  const MCSymbolRefExpr *ActualArg0 = dyn_cast_or_null<MCSymbolRefExpr>(Arg0);
  ASSERT_NE(ActualArg0, nullptr);
  EXPECT_EQ(&(ActualArg0->getSymbol()), Val);
}

TEST_F(AsmPrinterEmitDwarfSymbolReferenceTest, ELFDWARF64ForceOffset) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    return;

  EXPECT_CALL(TestPrinter->getMS(),
              emitAbsoluteSymbolDiff(Val, SecBeginSymbol, 8));
  TestPrinter->getAP()->emitDwarfSymbolReference(Val, true);
}

class AsmPrinterEmitDwarfStringOffsetTest : public AsmPrinterFixtureBase {
protected:
  bool init(const std::string &TripleStr, unsigned DwarfVersion,
            dwarf::DwarfFormat DwarfFormat) {
    if (!AsmPrinterFixtureBase::init(TripleStr, DwarfVersion, DwarfFormat))
      return false;

    Val.Index = DwarfStringPoolEntry::NotIndexed;
    Val.Symbol = TestPrinter->getCtx().createTempSymbol();
    Val.Offset = 42;
    return true;
  }

  DwarfStringPoolEntry Val;
};

TEST_F(AsmPrinterEmitDwarfStringOffsetTest, DWARF32) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    return;

  const MCExpr *Arg0 = nullptr;
  EXPECT_CALL(TestPrinter->getMS(), emitValueImpl(_, 4, _))
      .WillOnce(SaveArg<0>(&Arg0));
  TestPrinter->getAP()->emitDwarfStringOffset(Val);

  const MCSymbolRefExpr *ActualArg0 = dyn_cast_or_null<MCSymbolRefExpr>(Arg0);
  ASSERT_NE(ActualArg0, nullptr);
  EXPECT_EQ(&(ActualArg0->getSymbol()), Val.Symbol);
}

TEST_F(AsmPrinterEmitDwarfStringOffsetTest,
       DWARF32NoRelocationsAcrossSections) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    return;

  TestPrinter->setDwarfUsesRelocationsAcrossSections(false);
  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(Val.Offset, 4));
  TestPrinter->getAP()->emitDwarfStringOffset(Val);
}

TEST_F(AsmPrinterEmitDwarfStringOffsetTest, DWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    return;

  const MCExpr *Arg0 = nullptr;
  EXPECT_CALL(TestPrinter->getMS(), emitValueImpl(_, 8, _))
      .WillOnce(SaveArg<0>(&Arg0));
  TestPrinter->getAP()->emitDwarfStringOffset(Val);

  const MCSymbolRefExpr *ActualArg0 = dyn_cast_or_null<MCSymbolRefExpr>(Arg0);
  ASSERT_NE(ActualArg0, nullptr);
  EXPECT_EQ(&(ActualArg0->getSymbol()), Val.Symbol);
}

TEST_F(AsmPrinterEmitDwarfStringOffsetTest,
       DWARF64NoRelocationsAcrossSections) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    return;

  TestPrinter->setDwarfUsesRelocationsAcrossSections(false);
  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(Val.Offset, 8));
  TestPrinter->getAP()->emitDwarfStringOffset(Val);
}

class AsmPrinterEmitDwarfOffsetTest : public AsmPrinterFixtureBase {
protected:
  bool init(const std::string &TripleStr, unsigned DwarfVersion,
            dwarf::DwarfFormat DwarfFormat) {
    if (!AsmPrinterFixtureBase::init(TripleStr, DwarfVersion, DwarfFormat))
      return false;

    Label = TestPrinter->getCtx().createTempSymbol();
    return true;
  }

  MCSymbol *Label = nullptr;
  uint64_t Offset = 42;
};

TEST_F(AsmPrinterEmitDwarfOffsetTest, DWARF32) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    return;

  const MCExpr *Arg0 = nullptr;
  EXPECT_CALL(TestPrinter->getMS(), emitValueImpl(_, 4, _))
      .WillOnce(SaveArg<0>(&Arg0));
  TestPrinter->getAP()->emitDwarfOffset(Label, Offset);

  const MCBinaryExpr *ActualArg0 = dyn_cast_or_null<MCBinaryExpr>(Arg0);
  ASSERT_NE(ActualArg0, nullptr);
  EXPECT_EQ(ActualArg0->getOpcode(), MCBinaryExpr::Add);

  const MCSymbolRefExpr *ActualLHS =
      dyn_cast_or_null<MCSymbolRefExpr>(ActualArg0->getLHS());
  ASSERT_NE(ActualLHS, nullptr);
  EXPECT_EQ(&(ActualLHS->getSymbol()), Label);

  const MCConstantExpr *ActualRHS =
      dyn_cast_or_null<MCConstantExpr>(ActualArg0->getRHS());
  ASSERT_NE(ActualRHS, nullptr);
  EXPECT_EQ(static_cast<uint64_t>(ActualRHS->getValue()), Offset);
}

TEST_F(AsmPrinterEmitDwarfOffsetTest, DWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    return;

  const MCExpr *Arg0 = nullptr;
  EXPECT_CALL(TestPrinter->getMS(), emitValueImpl(_, 8, _))
      .WillOnce(SaveArg<0>(&Arg0));
  TestPrinter->getAP()->emitDwarfOffset(Label, Offset);

  const MCBinaryExpr *ActualArg0 = dyn_cast_or_null<MCBinaryExpr>(Arg0);
  ASSERT_NE(ActualArg0, nullptr);
  EXPECT_EQ(ActualArg0->getOpcode(), MCBinaryExpr::Add);

  const MCSymbolRefExpr *ActualLHS =
      dyn_cast_or_null<MCSymbolRefExpr>(ActualArg0->getLHS());
  ASSERT_NE(ActualLHS, nullptr);
  EXPECT_EQ(&(ActualLHS->getSymbol()), Label);

  const MCConstantExpr *ActualRHS =
      dyn_cast_or_null<MCConstantExpr>(ActualArg0->getRHS());
  ASSERT_NE(ActualRHS, nullptr);
  EXPECT_EQ(static_cast<uint64_t>(ActualRHS->getValue()), Offset);
}

class AsmPrinterEmitDwarfLengthOrOffsetTest : public AsmPrinterFixtureBase {
protected:
  uint64_t Val = 42;
};

TEST_F(AsmPrinterEmitDwarfLengthOrOffsetTest, DWARF32) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    return;

  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(Val, 4));
  TestPrinter->getAP()->emitDwarfLengthOrOffset(Val);
}

TEST_F(AsmPrinterEmitDwarfLengthOrOffsetTest, DWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    return;

  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(Val, 8));
  TestPrinter->getAP()->emitDwarfLengthOrOffset(Val);
}

class AsmPrinterGetUnitLengthFieldByteSizeTest : public AsmPrinterFixtureBase {
};

TEST_F(AsmPrinterGetUnitLengthFieldByteSizeTest, DWARF32) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    return;

  EXPECT_EQ(TestPrinter->getAP()->getUnitLengthFieldByteSize(), 4u);
}

TEST_F(AsmPrinterGetUnitLengthFieldByteSizeTest, DWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    return;

  EXPECT_EQ(TestPrinter->getAP()->getUnitLengthFieldByteSize(), 12u);
}

class AsmPrinterMaybeEmitDwarf64MarkTest : public AsmPrinterFixtureBase {};

TEST_F(AsmPrinterMaybeEmitDwarf64MarkTest, DWARF32) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    return;

  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(_, _)).Times(0);
  TestPrinter->getAP()->maybeEmitDwarf64Mark();
}

TEST_F(AsmPrinterMaybeEmitDwarf64MarkTest, DWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    return;

  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(dwarf::DW_LENGTH_DWARF64, 4));
  TestPrinter->getAP()->maybeEmitDwarf64Mark();
}

class AsmPrinterEmitDwarfUnitLengthAsIntTest : public AsmPrinterFixtureBase {
protected:
  uint64_t Val = 42;
};

TEST_F(AsmPrinterEmitDwarfUnitLengthAsIntTest, DWARF32) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    return;

  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(Val, 4));
  TestPrinter->getAP()->emitDwarfUnitLength(Val, "");
}

TEST_F(AsmPrinterEmitDwarfUnitLengthAsIntTest, DWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    return;

  InSequence S;
  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(dwarf::DW_LENGTH_DWARF64, 4));
  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(Val, 8));

  TestPrinter->getAP()->emitDwarfUnitLength(Val, "");
}

class AsmPrinterEmitDwarfUnitLengthAsHiLoDiffTest
    : public AsmPrinterFixtureBase {
protected:
  bool init(const std::string &TripleStr, unsigned DwarfVersion,
            dwarf::DwarfFormat DwarfFormat) {
    if (!AsmPrinterFixtureBase::init(TripleStr, DwarfVersion, DwarfFormat))
      return false;

    Hi = TestPrinter->getCtx().createTempSymbol();
    Lo = TestPrinter->getCtx().createTempSymbol();
    return true;
  }

  MCSymbol *Hi = nullptr;
  MCSymbol *Lo = nullptr;
};

TEST_F(AsmPrinterEmitDwarfUnitLengthAsHiLoDiffTest, DWARF32) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    return;

  EXPECT_CALL(TestPrinter->getMS(), emitAbsoluteSymbolDiff(Hi, Lo, 4));
  TestPrinter->getAP()->emitDwarfUnitLength(Hi, Lo, "");
}

TEST_F(AsmPrinterEmitDwarfUnitLengthAsHiLoDiffTest, DWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    return;

  InSequence S;
  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(dwarf::DW_LENGTH_DWARF64, 4));
  EXPECT_CALL(TestPrinter->getMS(), emitAbsoluteSymbolDiff(Hi, Lo, 8));

  TestPrinter->getAP()->emitDwarfUnitLength(Hi, Lo, "");
}

} // end namespace
