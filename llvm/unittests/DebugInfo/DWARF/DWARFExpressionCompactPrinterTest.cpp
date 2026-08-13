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
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace dwarf;

namespace {
static void appendULEB128(SmallVectorImpl<uint8_t> &V, uint64_t Val) {
  uint8_t Buf[16];
  unsigned N = encodeULEB128(Val, Buf);
  V.append(Buf, Buf + N);
}

// Use a fixed map so these tests don't depend on which targets are built into
// the unit test binary. Unmapped values cover failed target lookups and
// ASCII-packed names.
static StringRef getTestRegisterName(uint64_t DwarfRegNum, bool) {
  switch (DwarfRegNum) {
  case 0:
    return "R0";
  case 10:
    return "R10";
  case 13:
    return "SP";
  case 256:
    return "D0";
  default:
    return {};
  }
}

static void expectCompactPrintFailureWithoutRegNames(ArrayRef<uint8_t> ExprData,
                                                     StringRef Expected) {
  std::string Result;
  raw_string_ostream OS(Result);
  DataExtractor DE(ExprData, true);
  DWARFExpression Expr(DE, 8);

  EXPECT_FALSE(printDwarfExpressionCompact(&Expr, OS));
  EXPECT_EQ(OS.str(), Expected);
}

class DWARFExpressionCompactPrinterTest : public ::testing::Test {
public:
  void TestExprPrinter(ArrayRef<uint8_t> ExprData, StringRef Expected);
  void TestExprPrinterFailure(ArrayRef<uint8_t> ExprData, StringRef Expected);
};
} // namespace

void DWARFExpressionCompactPrinterTest::TestExprPrinter(
    ArrayRef<uint8_t> ExprData, StringRef Expected) {
  std::string Result;
  raw_string_ostream OS(Result);
  DataExtractor DE(ExprData, true);
  DWARFExpression Expr(DE, 8);

  EXPECT_TRUE(printDwarfExpressionCompact(&Expr, OS, getTestRegisterName));
  EXPECT_EQ(OS.str(), Expected);
}

void DWARFExpressionCompactPrinterTest::TestExprPrinterFailure(
    ArrayRef<uint8_t> ExprData, StringRef Expected) {
  std::string Result;
  raw_string_ostream OS(Result);
  DataExtractor DE(ExprData, true);
  DWARFExpression Expr(DE, 8);

  EXPECT_FALSE(printDwarfExpressionCompact(&Expr, OS, getTestRegisterName));
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

// Register 100 has neither a target name nor an ASCII-packed name, so check
// that DW_OP_regx reports it as unknown and returns false.
TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_regx_unknown) {
  TestExprPrinterFailure({DW_OP_regx, 0x64}, "<unknown register 100>");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_breg0) {
  TestExprPrinter({DW_OP_breg0, 0x04}, "[R0+4]");
}

// With no register callback, the short register form must report register 0 as
// unknown and return false without calling GetNameForDWARFReg.
TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_reg0_no_callback) {
  expectCompactPrintFailureWithoutRegNames({DW_OP_reg0},
                                           "<unknown register 0>");
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

// DW_OP_bregx uses the same two lookups, so check that it reports register 100
// as unknown and returns false when neither finds a name.
TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_bregx_unknown) {
  TestExprPrinterFailure({DW_OP_bregx, 0x64, 0x00}, "<unknown register 100>");
}

// With no register callback, the short base-register form must also report
// register 0 as unknown and return false without calling GetNameForDWARFReg.
TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_breg0_no_callback) {
  expectCompactPrintFailureWithoutRegNames({DW_OP_breg0, 0x00},
                                           "<unknown register 0>");
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

// A failed register lookup inside DW_OP_entry_value must keep the nested
// diagnostic and fail the enclosing expression before printing entry(...).
TEST_F(DWARFExpressionCompactPrinterTest,
       Test_OP_entry_value_unknown_register) {
  TestExprPrinterFailure(
      {DW_OP_entry_value, 0x02, DW_OP_regx, 0x64, DW_OP_stack_value},
      "<unknown register 100>");
}

// Use an opcode the compact printer does not handle to check that
// DW_OP_entry_value keeps the nested diagnostic and propagates the failure.
TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_entry_value_unknown_op) {
  TestExprPrinterFailure(
      {DW_OP_entry_value, 0x02, DW_OP_const1u, 0x01, DW_OP_stack_value},
      "<unknown op DW_OP_const1u (8)>");
}

// DW_OP_nop leaves the stack empty, so compact printing must emit the
// stack-size diagnostic and return false.
TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_nop) {
  TestExprPrinterFailure({DW_OP_nop}, "<stack of size 0, expected 1>");
}

// DW_OP_LLVM_nop leaves the stack empty as well, so it must emit the same
// diagnostic and return false.
TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_LLVM_nop) {
  TestExprPrinterFailure({DW_OP_LLVM_user, DW_OP_LLVM_nop},
                         "<stack of size 0, expected 1>");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_nop_OP_reg) {
  TestExprPrinter({DW_OP_nop, DW_OP_reg0}, "R0");
}

TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_LLVM_nop_OP_reg) {
  TestExprPrinter({DW_OP_LLVM_user, DW_OP_LLVM_nop, DW_OP_reg0}, "R0");
}

// An unhandled DW_OP_LLVM_user subopcode must print both opcode names and
// values, then return false.
TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_LLVM_user_unknown_subop) {
  TestExprPrinterFailure({DW_OP_LLVM_user, DW_OP_LLVM_form_aspace_address},
                         "<unknown op DW_OP_LLVM_user (233) subop "
                         "DW_OP_LLVM_form_aspace_address (2)>");
}

// DW_OP_NVIDIA_mux carries an opaque 8-bit selector, so the compact printer
// cannot know its stack effect and must bail out naming the opcode.
TEST_F(DWARFExpressionCompactPrinterTest, Test_OP_NVIDIA_mux) {
  TestExprPrinterFailure({DW_OP_NVIDIA_mux, 0xa5},
                         "<unknown op DW_OP_NVIDIA_mux (235)>");
}

// The selector is a fixed 1-byte operand, so the full printer must consume
// exactly one byte and print its value.
TEST(NVIDIAMux, Full_DW_OP_NVIDIA_mux) {
  const uint8_t Enc[] = {DW_OP_NVIDIA_mux, 0xa5};

  std::string Result;
  raw_string_ostream OS(Result);
  DataExtractor DE(Enc, true);
  DWARFExpression Expr(DE, 8);

  DIDumpOptions DumpOpts;
  printDwarfExpression(&Expr, OS, DumpOpts, nullptr);

  EXPECT_EQ(OS.str(), "DW_OP_NVIDIA_mux 0xa5");
}

// A trailing operation must still decode, proving the selector advanced the
// offset by exactly one byte.
TEST(NVIDIAMux, Full_DW_OP_NVIDIA_mux_TrailingOp) {
  const uint8_t Enc[] = {DW_OP_NVIDIA_mux, 0xa5, DW_OP_stack_value};

  std::string Result;
  raw_string_ostream OS(Result);
  DataExtractor DE(Enc, true);
  DWARFExpression Expr(DE, 8);

  DIDumpOptions DumpOpts;
  printDwarfExpression(&Expr, OS, DumpOpts, nullptr);

  EXPECT_EQ(OS.str(), "DW_OP_NVIDIA_mux 0xa5, DW_OP_stack_value");
}

// NVPTX packs virtual register names into DWARF register numbers, so compact
// printing without a callback must recover the name and return true.
TEST(NVPTXPackedRegister, Compact_DW_OP_regx_NoMRI) {
  SmallVector<uint8_t, 16> Enc;
  Enc.push_back(DW_OP_regx);
  appendULEB128(Enc, 0x25726432u);

  std::string Result;
  raw_string_ostream OS(Result);
  DataExtractor DE(Enc, true);
  DWARFExpression Expr(DE, 8);

  EXPECT_TRUE(printDwarfExpressionCompact(&Expr, OS, nullptr));
  EXPECT_EQ(OS.str(), "%rd2");
}

TEST(NVPTXPackedRegister, Full_DW_OP_regx_NoMRI) {
  SmallVector<uint8_t, 16> Enc;
  Enc.push_back(DW_OP_regx);
  appendULEB128(Enc, 0x25726431u);

  std::string Result;
  raw_string_ostream OS(Result);
  DataExtractor DE(Enc, true);
  DWARFExpression Expr(DE, 8);

  DIDumpOptions DumpOpts;
  printDwarfExpression(&Expr, OS, DumpOpts, nullptr);

  EXPECT_EQ(OS.str(), "DW_OP_regx %rd1");
}

TEST(NVPTXPackedRegister, Full_DW_OP_regx_CallbackMiss) {
  SmallVector<uint8_t, 16> Enc;
  Enc.push_back(DW_OP_regx);
  appendULEB128(Enc, 0x25727332u);

  std::string Result;
  raw_string_ostream OS(Result);
  DataExtractor DE(Enc, true);
  DWARFExpression Expr(DE, 8);

  DIDumpOptions DumpOpts;
  // getTestRegisterName misses this packed value, so the full printer still
  // needs to recover the name from the register number.
  DumpOpts.GetNameForDWARFReg = getTestRegisterName;

  printDwarfExpression(&Expr, OS, DumpOpts, nullptr);

  EXPECT_EQ(OS.str(), "DW_OP_regx %rs2");
}
