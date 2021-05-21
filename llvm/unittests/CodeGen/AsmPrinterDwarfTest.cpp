//===- llvm/unittest/CodeGen/AsmPrinterDwarfTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestAsmPrinter.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Target/TargetMachine.h"
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

    // AsmPrinter::emitDwarfSymbolReference(Label, true) gets the associated
    // section from `Label` to find its BeginSymbol.
    // Prepare the test symbol `Val` accordingly.

    Val = TestPrinter->getCtx().createTempSymbol();
    MCSection *Sec =
        TestPrinter->getCtx().getELFSection(".tst", ELF::SHT_PROGBITS, 0);
    SecBeginSymbol = Sec->getBeginSymbol();
    TestPrinter->getMS().SwitchSection(Sec);
    Val->setFragment(&Sec->getDummyFragment());

    return true;
  }

  MCSymbol *Val = nullptr;
  MCSymbol *SecBeginSymbol = nullptr;
};

TEST_F(AsmPrinterEmitDwarfSymbolReferenceTest, COFF) {
  if (!init("x86_64-pc-windows", /*DwarfVersion=*/4, dwarf::DWARF32))
    GTEST_SKIP();

  EXPECT_CALL(TestPrinter->getMS(), EmitCOFFSecRel32(Val, 0));
  TestPrinter->getAP()->emitDwarfSymbolReference(Val, false);
}

TEST_F(AsmPrinterEmitDwarfSymbolReferenceTest, COFFForceOffset) {
  if (!init("x86_64-pc-windows", /*DwarfVersion=*/4, dwarf::DWARF32))
    GTEST_SKIP();

  EXPECT_CALL(TestPrinter->getMS(),
              emitAbsoluteSymbolDiff(Val, SecBeginSymbol, 4));
  TestPrinter->getAP()->emitDwarfSymbolReference(Val, true);
}

TEST_F(AsmPrinterEmitDwarfSymbolReferenceTest, ELFDWARF32) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    GTEST_SKIP();

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
    GTEST_SKIP();

  EXPECT_CALL(TestPrinter->getMS(),
              emitAbsoluteSymbolDiff(Val, SecBeginSymbol, 4));
  TestPrinter->getAP()->emitDwarfSymbolReference(Val, true);
}

TEST_F(AsmPrinterEmitDwarfSymbolReferenceTest, ELFDWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    GTEST_SKIP();

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
    GTEST_SKIP();

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
    GTEST_SKIP();

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
    GTEST_SKIP();

  TestPrinter->setDwarfUsesRelocationsAcrossSections(false);
  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(Val.Offset, 4));
  TestPrinter->getAP()->emitDwarfStringOffset(Val);
}

TEST_F(AsmPrinterEmitDwarfStringOffsetTest, DWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    GTEST_SKIP();

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
    GTEST_SKIP();

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
    GTEST_SKIP();

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
    GTEST_SKIP();

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
    GTEST_SKIP();

  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(Val, 4));
  TestPrinter->getAP()->emitDwarfLengthOrOffset(Val);
}

TEST_F(AsmPrinterEmitDwarfLengthOrOffsetTest, DWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    GTEST_SKIP();

  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(Val, 8));
  TestPrinter->getAP()->emitDwarfLengthOrOffset(Val);
}

class AsmPrinterGetUnitLengthFieldByteSizeTest : public AsmPrinterFixtureBase {
};

TEST_F(AsmPrinterGetUnitLengthFieldByteSizeTest, DWARF32) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    GTEST_SKIP();

  EXPECT_EQ(TestPrinter->getAP()->getUnitLengthFieldByteSize(), 4u);
}

TEST_F(AsmPrinterGetUnitLengthFieldByteSizeTest, DWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    GTEST_SKIP();

  EXPECT_EQ(TestPrinter->getAP()->getUnitLengthFieldByteSize(), 12u);
}

class AsmPrinterEmitDwarfUnitLengthAsIntTest : public AsmPrinterFixtureBase {
protected:
  uint64_t Val = 42;
};

TEST_F(AsmPrinterEmitDwarfUnitLengthAsIntTest, DWARF32) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    GTEST_SKIP();

  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(Val, 4));
  TestPrinter->getAP()->emitDwarfUnitLength(Val, "");
}

TEST_F(AsmPrinterEmitDwarfUnitLengthAsIntTest, DWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    GTEST_SKIP();

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

    return true;
  }
};

TEST_F(AsmPrinterEmitDwarfUnitLengthAsHiLoDiffTest, DWARF32) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    GTEST_SKIP();

  InSequence S;
  const MCSymbol *Hi = nullptr;
  const MCSymbol *Lo = nullptr;
  EXPECT_CALL(TestPrinter->getMS(), emitAbsoluteSymbolDiff(_, _, 4))
      .WillOnce(DoAll(SaveArg<0>(&Hi), SaveArg<1>(&Lo)));
  MCSymbol *LTmp = nullptr;
  EXPECT_CALL(TestPrinter->getMS(), emitLabel(_, _))
      .WillOnce(SaveArg<0>(&LTmp));

  MCSymbol *HTmp = TestPrinter->getAP()->emitDwarfUnitLength("", "");
  EXPECT_NE(Lo, nullptr);
  EXPECT_EQ(Lo, LTmp);
  EXPECT_NE(Hi, nullptr);
  EXPECT_EQ(Hi, HTmp);
}

TEST_F(AsmPrinterEmitDwarfUnitLengthAsHiLoDiffTest, DWARF64) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF64))
    GTEST_SKIP();

  InSequence S;
  const MCSymbol *Hi = nullptr;
  const MCSymbol *Lo = nullptr;
  EXPECT_CALL(TestPrinter->getMS(), emitIntValue(dwarf::DW_LENGTH_DWARF64, 4));
  EXPECT_CALL(TestPrinter->getMS(), emitAbsoluteSymbolDiff(_, _, 8))
      .WillOnce(DoAll(SaveArg<0>(&Hi), SaveArg<1>(&Lo)));
  MCSymbol *LTmp = nullptr;
  EXPECT_CALL(TestPrinter->getMS(), emitLabel(_, _))
      .WillOnce(SaveArg<0>(&LTmp));

  MCSymbol *HTmp = TestPrinter->getAP()->emitDwarfUnitLength("", "");
  EXPECT_NE(Lo, nullptr);
  EXPECT_EQ(Lo, LTmp);
  EXPECT_NE(Hi, nullptr);
  EXPECT_EQ(Hi, HTmp);
}

class AsmPrinterHandlerTest : public AsmPrinterFixtureBase {
  class TestHandler : public AsmPrinterHandler {
    AsmPrinterHandlerTest &Test;

  public:
    TestHandler(AsmPrinterHandlerTest &Test) : Test(Test) {}
    virtual ~TestHandler() {}
    virtual void setSymbolSize(const MCSymbol *Sym, uint64_t Size) override {}
    virtual void beginModule(Module *M) override { Test.BeginCount++; }
    virtual void endModule() override { Test.EndCount++; }
    virtual void beginFunction(const MachineFunction *MF) override {}
    virtual void endFunction(const MachineFunction *MF) override {}
    virtual void beginInstruction(const MachineInstr *MI) override {}
    virtual void endInstruction() override {}
  };

protected:
  bool init(const std::string &TripleStr, unsigned DwarfVersion,
            dwarf::DwarfFormat DwarfFormat) {
    if (!AsmPrinterFixtureBase::init(TripleStr, DwarfVersion, DwarfFormat))
      return false;

    auto *AP = TestPrinter->getAP();
    AP->addAsmPrinterHandler(AsmPrinter::HandlerInfo(
        std::unique_ptr<AsmPrinterHandler>(new TestHandler(*this)),
        "TestTimerName", "TestTimerDesc", "TestGroupName", "TestGroupDesc"));
    LLVMTargetMachine *LLVMTM = static_cast<LLVMTargetMachine *>(&AP->TM);
    legacy::PassManager PM;
    PM.add(new MachineModuleInfoWrapperPass(LLVMTM));
    PM.add(TestPrinter->releaseAP()); // Takes ownership of destroying AP
    LLVMContext Context;
    std::unique_ptr<Module> M(new Module("TestModule", Context));
    M->setDataLayout(LLVMTM->createDataLayout());
    PM.run(*M);
    // Now check that we can run it twice.
    AP->addAsmPrinterHandler(AsmPrinter::HandlerInfo(
        std::unique_ptr<AsmPrinterHandler>(new TestHandler(*this)),
        "TestTimerName", "TestTimerDesc", "TestGroupName", "TestGroupDesc"));
    PM.run(*M);
    return true;
  }

  int BeginCount = 0;
  int EndCount = 0;
};

TEST_F(AsmPrinterHandlerTest, Basic) {
  if (!init("x86_64-pc-linux", /*DwarfVersion=*/4, dwarf::DWARF32))
    GTEST_SKIP();

  ASSERT_EQ(BeginCount, 3);
  ASSERT_EQ(EndCount, 3);
}

} // end namespace
