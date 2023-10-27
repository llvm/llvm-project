//===-- PDBFPOProgramToDWARFExpressionTests.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/SymbolFile/NativePDB/PdbFPOProgramToDWARFExpression.h"

#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/StreamBuffer.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::npdb;

/// Valid programs tests

static void
CheckValidProgramTranslation(llvm::StringRef fpo_program,
                             llvm::StringRef target_register_name,
                             llvm::StringRef expected_dwarf_expression) {
  // program translation
  StreamBuffer<32> stream(Stream::eBinary, 4, eByteOrderLittle);
  ASSERT_TRUE(TranslateFPOProgramToDWARFExpression(
      fpo_program, target_register_name, llvm::Triple::x86, stream));

  // print dwarf expression to comparable textual representation
  llvm::DataExtractor extractor({stream.GetData(), stream.GetSize()},
                                /*IsLittleEndian=*/true, /*AddressSize=*/4);

  std::string result;
  llvm::raw_string_ostream os(result);
  llvm::DWARFExpression(extractor, /*AddressSize=*/4, llvm::dwarf::DWARF32)
      .print(os, llvm::DIDumpOptions(), nullptr);

  // actual check
  ASSERT_EQ(expected_dwarf_expression, os.str());
}

TEST(PDBFPOProgramToDWARFExpressionTests, SingleAssignmentRegisterRef) {
  CheckValidProgramTranslation("$T0 $ebp = ", "$T0", "DW_OP_breg6 +0");
}

TEST(PDBFPOProgramToDWARFExpressionTests, MultipleIndependentAssignments) {
  CheckValidProgramTranslation("$T1 1 = $T0 0 =", "$T0", "DW_OP_consts +0");
}

TEST(PDBFPOProgramToDWARFExpressionTests, MultipleDependentAssignments) {
  CheckValidProgramTranslation(
      "$T1 $ebp 4 + = $T0 $T1 8 - 128 @ = ", "$T0",
      "DW_OP_breg6 +0, DW_OP_consts +4, DW_OP_plus, DW_OP_consts +8, "
      "DW_OP_minus, DW_OP_consts +128, DW_OP_lit1, DW_OP_minus, DW_OP_not, "
      "DW_OP_and");
}

TEST(PDBFPOProgramToDWARFExpressionTests, DependencyChain) {
  CheckValidProgramTranslation("$T1 0 = $T0 $T1 = $ebp $T0 =", "$ebp",
                               "DW_OP_consts +0");
}

/// Invalid programs tests
static void
CheckInvalidProgramTranslation(llvm::StringRef fpo_program,
                               llvm::StringRef target_register_name) {
  // initial setup
  ArchSpec arch_spec("i686-pc-windows");
  llvm::Triple::ArchType arch_type = arch_spec.GetMachine();
  ByteOrder byte_order = arch_spec.GetByteOrder();
  uint32_t address_size = arch_spec.GetAddressByteSize();

  // program translation
  StreamBuffer<32> stream(Stream::eBinary, address_size, byte_order);
  EXPECT_FALSE(TranslateFPOProgramToDWARFExpression(
      fpo_program, target_register_name, arch_type, stream));
  EXPECT_EQ((size_t)0, stream.GetSize());
}

TEST(PDBFPOProgramToDWARFExpressionTests, InvalidAssignmentSingle) {
  CheckInvalidProgramTranslation("$T0 0", "$T0");
}

TEST(PDBFPOProgramToDWARFExpressionTests, InvalidAssignmentMultiple) {
  CheckInvalidProgramTranslation("$T1 0 = $T0 0", "$T0");
}

TEST(PDBFPOProgramToDWARFExpressionTests, UnknownOp) {
  CheckInvalidProgramTranslation("$T0 $ebp 0 & = ", "$T0");
}

TEST(PDBFPOProgramToDWARFExpressionTests, InvalidOpBinary) {
  CheckInvalidProgramTranslation("$T0 0 + = ", "$T0");
}

TEST(PDBFPOProgramToDWARFExpressionTests, InvalidOpUnary) {
  CheckInvalidProgramTranslation("$T0 ^ = ", "$T0");
}

TEST(PDBFPOProgramToDWARFExpressionTests, MissingTargetRegister) {
  CheckInvalidProgramTranslation("$T1 0 = ", "$T0");
}

TEST(PDBFPOProgramToDWARFExpressionTests, UnresolvedRegisterReference) {
  CheckInvalidProgramTranslation("$T0 $abc = ", "$T0");
}

TEST(PDBFPOProgramToDWARFExpressionTests,
     UnresolvedRegisterAssignmentReference) {
  CheckInvalidProgramTranslation("$T2 0 = $T0 $T1 = ", "$T0");
}

TEST(PDBFPOProgramToDWARFExpressionTests,
     UnresolvedCyclicRegisterAssignmentReference) {
  CheckInvalidProgramTranslation("$T1 $T0 = $T0 $T1 = ", "$T0");
}

TEST(PDBFPOProgramToDWARFExpressionTests,
     UnresolvedDependentCyclicRegisterAssignmentReference) {
  CheckInvalidProgramTranslation("$T1 $T0 = $T0 $T1 = $T2 $T1 =", "$T2");
}

TEST(PDBFPOProgramToDWARFExpressionTests, UnsupportedRASearch) {
  CheckInvalidProgramTranslation("$T0 .raSearch = ", "$T0");
}
