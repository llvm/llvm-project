//===- llvm/unittest/DebugInfo/DWARFExpressionCompactPrinterTest.cpp ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DwarfGenerator.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFExpressionPrinter.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFExpression.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace dwarf;

namespace {
class DWARFExpressionCompactPrinterTest : public ::testing::Test {
public:
  std::unique_ptr<MCRegisterInfo> MRI;

  DWARFExpressionCompactPrinterTest() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();

    std::string TripleName = "armv8a-linux-gnueabi";
    std::string ErrorStr;

    const Target *TheTarget =
        TargetRegistry::lookupTarget(TripleName, ErrorStr);

    if (!TheTarget)
      return;

    MRI.reset(TheTarget->createMCRegInfo(Triple(TripleName)));
  }

  void TestExprPrinter(ArrayRef<uint8_t> ExprData, StringRef Expected);
};
} // namespace

void DWARFExpressionCompactPrinterTest::TestExprPrinter(
    ArrayRef<uint8_t> ExprData, StringRef Expected) {
  // If we didn't build ARM, do not run the test.
  if (!MRI)
    GTEST_SKIP();

  // Print the expression, passing in the subprogram DIE, and check that the
  // result is as expected.
  std::string Result;
  raw_string_ostream OS(Result);
  DataExtractor DE(ExprData, true, 8);
  DWARFExpression Expr(DE, 8);

  auto GetRegName = [&](uint64_t DwarfRegNum, bool IsEH) -> StringRef {
    if (std::optional<MCRegister> LLVMRegNum =
            this->MRI->getLLVMRegNum(DwarfRegNum, IsEH))
      if (const char *RegName = this->MRI->getName(*LLVMRegNum))
        return llvm::StringRef(RegName);
    OS << "<unknown register " << DwarfRegNum << ">";
    return {};
  };

  printDwarfExpressionCompact(&Expr, OS, GetRegName);
  EXPECT_EQ(OS.str(), Expected);
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_reg0) {
  TestExprPrinter({DW_OP_reg0}, "R0");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_reg10) {
  TestExprPrinter({DW_OP_reg10}, "R10");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_regx) {
  TestExprPrinter({DW_OP_regx, 0x80, 0x02}, "D0");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_breg0) {
  TestExprPrinter({DW_OP_breg0, 0x04}, "[R0+4]");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_breg0_large_offset) {
  TestExprPrinter({DW_OP_breg0, 0x80, 0x02}, "[R0+256]");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_breg13) {
  TestExprPrinter({DW_OP_breg13, 0x10}, "[SP+16]");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_breg13_zero_offset) {
  TestExprPrinter({DW_OP_breg13, 0x00}, "[SP]");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_breg0_negative) {
  TestExprPrinter({DW_OP_breg0, 0x70}, "[R0-16]");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_bregx) {
  TestExprPrinter({DW_OP_bregx, 0x0d, 0x28}, "[SP+40]");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_stack_value) {
  TestExprPrinter({DW_OP_breg13, 0x04, DW_OP_stack_value}, "SP+4");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_entry_value) {
  TestExprPrinter({DW_OP_entry_value, 0x01, DW_OP_reg0, DW_OP_stack_value},
                  "entry(R0)");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_entry_value_mem) {
  TestExprPrinter(
      {DW_OP_entry_value, 0x02, DW_OP_breg13, 0x10, DW_OP_stack_value},
      "entry([SP+16])");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_nop) {
  TestExprPrinter({DW_OP_nop}, "<stack of size 0, expected 1>");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_LLVM_nop) {
  TestExprPrinter({DW_OP_LLVM_user, DW_OP_LLVM_nop},
                  "<stack of size 0, expected 1>");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_nop_OP_reg) {
  TestExprPrinter({DW_OP_nop, DW_OP_reg0}, "R0");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_LLVM_nop_OP_reg) {
  TestExprPrinter({DW_OP_LLVM_user, DW_OP_LLVM_nop, DW_OP_reg0}, "R0");
}
