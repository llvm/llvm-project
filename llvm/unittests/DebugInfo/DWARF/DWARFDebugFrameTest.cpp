//===- llvm/unittest/DebugInfo/DWARFDebugFrameTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

dwarf::CIE createCIE(bool IsDWARF64, uint64_t Offset, uint64_t Length) {
  return dwarf::CIE(IsDWARF64, Offset, Length,
                    /*Version=*/3,
                    /*Augmentation=*/StringRef(),
                    /*AddressSize=*/8,
                    /*SegmentDescriptorSize=*/0,
                    /*CodeAlignmentFactor=*/1,
                    /*DataAlignmentFactor=*/-8,
                    /*ReturnAddressRegister=*/16,
                    /*AugmentationData=*/StringRef(),
                    /*FDEPointerEncoding=*/dwarf::DW_EH_PE_absptr,
                    /*LSDAPointerEncoding=*/dwarf::DW_EH_PE_omit,
                    /*Personality=*/std::nullopt,
                    /*PersonalityEnc=*/std::nullopt,
                    /*Arch=*/Triple::x86_64);
}

void expectDumpResult(const dwarf::CIE &TestCIE, bool IsEH,
                      StringRef ExpectedFirstLine) {
  std::string Output;
  raw_string_ostream OS(Output);
  auto DumpOpts = DIDumpOptions();
  DumpOpts.IsEH = IsEH;
  TestCIE.dump(OS, DumpOpts);
  OS.flush();
  StringRef FirstLine = StringRef(Output).split('\n').first;
  EXPECT_EQ(FirstLine, ExpectedFirstLine);
}

void expectDumpResult(const dwarf::FDE &TestFDE, bool IsEH,
                      StringRef ExpectedFirstLine) {
  std::string Output;
  raw_string_ostream OS(Output);
  auto DumpOpts = DIDumpOptions();
  DumpOpts.IsEH = IsEH;
  TestFDE.dump(OS, DumpOpts);
  OS.flush();
  StringRef FirstLine = StringRef(Output).split('\n').first;
  EXPECT_EQ(FirstLine, ExpectedFirstLine);
}

TEST(DWARFDebugFrame, DumpDWARF32CIE) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x1111abcd,
                                 /*Length=*/0x2222abcd);
  expectDumpResult(TestCIE, /*IsEH=*/false, "1111abcd 2222abcd ffffffff CIE");
}

TEST(DWARFDebugFrame, DumpDWARF64CIE) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/true,
                                 /*Offset=*/0x1111abcdabcd,
                                 /*Length=*/0x2222abcdabcd);
  expectDumpResult(TestCIE, /*IsEH=*/false,
                   "1111abcdabcd 00002222abcdabcd ffffffffffffffff CIE");
}

TEST(DWARFDebugFrame, DumpEHCIE) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x1000,
                                 /*Length=*/0x20);
  expectDumpResult(TestCIE, /*IsEH=*/true, "00001000 00000020 00000000 CIE");
}

TEST(DWARFDebugFrame, DumpEH64CIE) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/true,
                                 /*Offset=*/0x1000,
                                 /*Length=*/0x20);
  expectDumpResult(TestCIE, /*IsEH=*/true,
                   "00001000 0000000000000020 00000000 CIE");
}

TEST(DWARFDebugFrame, DumpDWARF64FDE) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/true,
                                 /*Offset=*/0x1111abcdabcd,
                                 /*Length=*/0x2222abcdabcd);
  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x5555abcdabcd,
                     /*AddressRange=*/0x111111111111,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);
  expectDumpResult(TestFDE, /*IsEH=*/false,
                   "3333abcdabcd 00004444abcdabcd 00001111abcdabcd FDE "
                   "cie=1111abcdabcd pc=5555abcdabcd...6666bcdebcde");
}

TEST(DWARFDebugFrame, DumpEH64FDE) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/true,
                                 /*Offset=*/0x1111ab9a000c,
                                 /*Length=*/0x20);
  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x1111abcdabcd,
                     /*Length=*/0x2222abcdabcd,
                     /*CIEPointer=*/0x33abcd,
                     /*InitialLocation=*/0x4444abcdabcd,
                     /*AddressRange=*/0x111111111111,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);
  expectDumpResult(TestFDE, /*IsEH=*/true,
                   "1111abcdabcd 00002222abcdabcd 0033abcd FDE "
                   "cie=1111ab9a000c pc=4444abcdabcd...5555bcdebcde");
}

static Error parseCFI(dwarf::CIE &C, ArrayRef<uint8_t> Instructions,
                      std::optional<uint64_t> Size = std::nullopt) {
  DWARFDataExtractor Data(Instructions, /*IsLittleEndian=*/true,
                          /*AddressSize=*/8);
  uint64_t Offset = 0;
  const uint64_t EndOffset = Size ? *Size : (uint64_t)Instructions.size();
  return C.cfis().parse(Data, &Offset, EndOffset);
}

static Error parseCFI(dwarf::FDE &FDE, ArrayRef<uint8_t> Instructions) {
  DWARFDataExtractor Data(Instructions, /*IsLittleEndian=*/true,
                          /*AddressSize=*/8);
  uint64_t Offset = 0;
  return FDE.cfis().parse(Data, &Offset, Instructions.size());
}

TEST(DWARFDebugFrame, InvalidCFIOpcodesTest) {
  llvm::DenseSet<uint8_t> ValidExtendedOpcodes = {
      dwarf::DW_CFA_nop,
      dwarf::DW_CFA_advance_loc,
      dwarf::DW_CFA_offset,
      dwarf::DW_CFA_restore,
      dwarf::DW_CFA_set_loc,
      dwarf::DW_CFA_advance_loc1,
      dwarf::DW_CFA_advance_loc2,
      dwarf::DW_CFA_advance_loc4,
      dwarf::DW_CFA_offset_extended,
      dwarf::DW_CFA_restore_extended,
      dwarf::DW_CFA_undefined,
      dwarf::DW_CFA_same_value,
      dwarf::DW_CFA_register,
      dwarf::DW_CFA_remember_state,
      dwarf::DW_CFA_restore_state,
      dwarf::DW_CFA_def_cfa,
      dwarf::DW_CFA_def_cfa_register,
      dwarf::DW_CFA_def_cfa_offset,
      dwarf::DW_CFA_def_cfa_expression,
      dwarf::DW_CFA_expression,
      dwarf::DW_CFA_offset_extended_sf,
      dwarf::DW_CFA_def_cfa_sf,
      dwarf::DW_CFA_def_cfa_offset_sf,
      dwarf::DW_CFA_LLVM_def_aspace_cfa,
      dwarf::DW_CFA_LLVM_def_aspace_cfa_sf,
      dwarf::DW_CFA_val_offset,
      dwarf::DW_CFA_val_offset_sf,
      dwarf::DW_CFA_val_expression,
      dwarf::DW_CFA_MIPS_advance_loc8,
      dwarf::DW_CFA_GNU_window_save,
      dwarf::DW_CFA_AARCH64_negate_ra_state,
      dwarf::DW_CFA_GNU_args_size};

  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  // See DWARF standard v3, section 7.23: low 6 bits are used to encode an
  // extended opcode.
  for (uint8_t Code = 0; Code <= 63; ++Code) {
    if (ValidExtendedOpcodes.count(Code))
      continue;

    EXPECT_THAT_ERROR(parseCFI(TestCIE, Code),
                      FailedWithMessage(("invalid extended CFI opcode 0x" +
                                         Twine::utohexstr(Code))
                                            .str()
                                            .c_str()));
  }
}

// Here we test how truncated Call Frame Instructions are parsed.
TEST(DWARFDebugFrame, ParseTruncatedCFITest) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  // Having an empty instructions list is fine.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {}), Succeeded());

  // Unable to read an opcode, because the instructions list is empty, but we
  // say to the parser that it is not.
  EXPECT_THAT_ERROR(
      parseCFI(TestCIE, {}, /*Size=*/1),
      FailedWithMessage(
          "unexpected end of data at offset 0x0 while reading [0x0, 0x1)"));

  // Unable to read a truncated DW_CFA_offset instruction.
  EXPECT_THAT_ERROR(
      parseCFI(TestCIE, {dwarf::DW_CFA_offset}),
      FailedWithMessage("unable to decode LEB128 at offset 0x00000001: "
                        "malformed uleb128, extends past end"));

  // Unable to read a truncated DW_CFA_set_loc instruction.
  EXPECT_THAT_ERROR(
      parseCFI(TestCIE, {dwarf::DW_CFA_set_loc}),
      FailedWithMessage(
          "unexpected end of data at offset 0x1 while reading [0x1, 0x9)"));

  // Unable to read a truncated DW_CFA_advance_loc1 instruction.
  EXPECT_THAT_ERROR(
      parseCFI(TestCIE, {dwarf::DW_CFA_advance_loc1}),
      FailedWithMessage(
          "unexpected end of data at offset 0x1 while reading [0x1, 0x2)"));

  // Unable to read a truncated DW_CFA_advance_loc2 instruction.
  EXPECT_THAT_ERROR(
      parseCFI(TestCIE, {dwarf::DW_CFA_advance_loc2}),
      FailedWithMessage(
          "unexpected end of data at offset 0x1 while reading [0x1, 0x3)"));

  // Unable to read a truncated DW_CFA_advance_loc4 instruction.
  EXPECT_THAT_ERROR(
      parseCFI(TestCIE, {dwarf::DW_CFA_advance_loc4}),
      FailedWithMessage(
          "unexpected end of data at offset 0x1 while reading [0x1, 0x5)"));

  // A test for an instruction with a single ULEB128 operand.
  auto CheckOp_ULEB128 = [&](uint8_t Inst) {
    EXPECT_THAT_ERROR(
        parseCFI(TestCIE, Inst),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000001: "
                          "malformed uleb128, extends past end"));
  };

  for (uint8_t Inst :
       {dwarf::DW_CFA_restore_extended, dwarf::DW_CFA_undefined,
        dwarf::DW_CFA_same_value, dwarf::DW_CFA_def_cfa_register,
        dwarf::DW_CFA_def_cfa_offset, dwarf::DW_CFA_GNU_args_size})
    CheckOp_ULEB128(Inst);

  // Unable to read a truncated DW_CFA_def_cfa_offset_sf instruction.
  EXPECT_THAT_ERROR(
      parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa_offset_sf}),
      FailedWithMessage("unable to decode LEB128 at offset 0x00000001: "
                        "malformed sleb128, extends past end"));

  // A test for an instruction with two ULEB128 operands.
  auto CheckOp_ULEB128_ULEB128 = [&](uint8_t Inst) {
    EXPECT_THAT_ERROR(
        parseCFI(TestCIE, Inst),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000001: "
                          "malformed uleb128, extends past end"));

    EXPECT_THAT_ERROR(
        parseCFI(TestCIE, {Inst, /*Op1=*/0}),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000002: "
                          "malformed uleb128, extends past end"));
  };

  for (uint8_t Inst : {dwarf::DW_CFA_offset_extended, dwarf::DW_CFA_register,
                       dwarf::DW_CFA_def_cfa, dwarf::DW_CFA_LLVM_def_aspace_cfa,
                       dwarf::DW_CFA_val_offset})
    CheckOp_ULEB128_ULEB128(Inst);

  // A test for an instruction with two operands: ULEB128, SLEB128.
  auto CheckOp_ULEB128_SLEB128 = [&](uint8_t Inst) {
    EXPECT_THAT_ERROR(
        parseCFI(TestCIE, Inst),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000001: "
                          "malformed uleb128, extends past end"));

    EXPECT_THAT_ERROR(
        parseCFI(TestCIE, {Inst, /*Op1=*/0}),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000002: "
                          "malformed sleb128, extends past end"));
  };

  for (uint8_t Inst :
       {dwarf::DW_CFA_offset_extended_sf, dwarf::DW_CFA_def_cfa_sf,
        dwarf::DW_CFA_LLVM_def_aspace_cfa_sf, dwarf::DW_CFA_val_offset_sf})
    CheckOp_ULEB128_SLEB128(Inst);

  // Unable to read a truncated DW_CFA_def_cfa_expression instruction.
  EXPECT_THAT_ERROR(
      parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa_expression}),
      FailedWithMessage("unable to decode LEB128 at offset 0x00000001: "
                        "malformed uleb128, extends past end"));
  EXPECT_THAT_ERROR(
      parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa_expression,
                         /*expression length=*/0x1}),
      FailedWithMessage(
          "unexpected end of data at offset 0x2 while reading [0x2, 0x3)"));
  // The DW_CFA_def_cfa_expression can contain a zero length expression.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa_expression,
                                       /*ExprLen=*/0}),
                    Succeeded());

  // A test for an instruction with three operands: ULEB128, expression length
  // (ULEB128) and expression bytes.
  auto CheckOp_ULEB128_Expr = [&](uint8_t Inst) {
    EXPECT_THAT_ERROR(
        parseCFI(TestCIE, {Inst}),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000001: "
                          "malformed uleb128, extends past end"));
    EXPECT_THAT_ERROR(
        parseCFI(TestCIE, {Inst, /*Op1=*/0}),
        FailedWithMessage("unable to decode LEB128 at offset 0x00000002: "
                          "malformed uleb128, extends past end"));
    // A zero length expression is fine
    EXPECT_THAT_ERROR(parseCFI(TestCIE, {Inst,
                                         /*Op1=*/0, /*ExprLen=*/0}),
                      Succeeded());
    EXPECT_THAT_ERROR(
        parseCFI(TestCIE, {Inst,
                           /*Op1=*/0, /*ExprLen=*/1}),
        FailedWithMessage(
            "unexpected end of data at offset 0x3 while reading [0x3, 0x4)"));
  };

  for (uint8_t Inst : {dwarf::DW_CFA_expression, dwarf::DW_CFA_val_expression})
    CheckOp_ULEB128_Expr(Inst);
}

void expectDumpResult(const dwarf::UnwindLocation &Loc,
                      StringRef ExpectedFirstLine) {
  std::string Output;
  raw_string_ostream OS(Output);
  OS << Loc;
  OS.flush();
  StringRef FirstLine = StringRef(Output).split('\n').first;
  EXPECT_EQ(FirstLine, ExpectedFirstLine);
}

TEST(DWARFDebugFrame, DumpUnwindLocations) {
  // Test constructing unwind locations and dumping each kind.
  constexpr int32_t PlusOff = 8;
  constexpr int32_t MinusOff = -8;
  constexpr uint8_t RegNum = 12;
  expectDumpResult(dwarf::UnwindLocation::createUnspecified(), "unspecified");
  expectDumpResult(dwarf::UnwindLocation::createUndefined(), "undefined");
  expectDumpResult(dwarf::UnwindLocation::createSame(), "same");
  expectDumpResult(dwarf::UnwindLocation::createIsCFAPlusOffset(PlusOff),
                   "CFA+8");
  expectDumpResult(dwarf::UnwindLocation::createIsCFAPlusOffset(MinusOff),
                   "CFA-8");
  expectDumpResult(dwarf::UnwindLocation::createAtCFAPlusOffset(PlusOff),
                   "[CFA+8]");
  expectDumpResult(dwarf::UnwindLocation::createAtCFAPlusOffset(MinusOff),
                   "[CFA-8]");

  expectDumpResult(
      dwarf::UnwindLocation::createIsRegisterPlusOffset(RegNum, PlusOff),
      "reg12+8");
  expectDumpResult(
      dwarf::UnwindLocation::createIsRegisterPlusOffset(RegNum, MinusOff),
      "reg12-8");
  expectDumpResult(
      dwarf::UnwindLocation::createAtRegisterPlusOffset(RegNum, PlusOff),
      "[reg12+8]");
  expectDumpResult(
      dwarf::UnwindLocation::createAtRegisterPlusOffset(RegNum, MinusOff),
      "[reg12-8]");
  expectDumpResult(dwarf::UnwindLocation::createIsConstant(12), "12");
  expectDumpResult(dwarf::UnwindLocation::createIsConstant(-32), "-32");
}

void expectDumpResult(const dwarf::RegisterLocations &Locs,
                      StringRef ExpectedFirstLine) {
  std::string Output;
  raw_string_ostream OS(Output);
  OS << Locs;
  OS.flush();
  StringRef FirstLine = StringRef(Output).split('\n').first;
  EXPECT_EQ(FirstLine, ExpectedFirstLine);
}

TEST(DWARFDebugFrame, RegisterLocations) {
  // Test the functionality of the RegisterLocations class.
  dwarf::RegisterLocations Locs;
  expectDumpResult(Locs, "");
  EXPECT_FALSE(Locs.hasLocations());
  // Set a register location for reg12 to unspecified and verify it dumps
  // correctly.
  Locs.setRegisterLocation(12, dwarf::UnwindLocation::createUnspecified());
  EXPECT_TRUE(Locs.hasLocations());
  expectDumpResult(Locs, "reg12=unspecified");

  // Replace the register location for reg12 to "same" and verify it dumps
  // correctly after it is modified
  Locs.setRegisterLocation(12, dwarf::UnwindLocation::createSame());
  EXPECT_TRUE(Locs.hasLocations());
  expectDumpResult(Locs, "reg12=same");

  // Remove the register location for reg12 verify it dumps correctly after it
  // is removed.
  Locs.removeRegisterLocation(12);
  EXPECT_FALSE(Locs.hasLocations());
  expectDumpResult(Locs, "");

  // Verify multiple registers added to the list dump correctly.
  auto Reg12Loc = dwarf::UnwindLocation::createAtCFAPlusOffset(4);
  auto Reg13Loc = dwarf::UnwindLocation::createAtCFAPlusOffset(8);
  auto Reg14Loc = dwarf::UnwindLocation::createSame();
  Locs.setRegisterLocation(12, Reg12Loc);
  Locs.setRegisterLocation(13, Reg13Loc);
  Locs.setRegisterLocation(14, Reg14Loc);
  EXPECT_TRUE(Locs.hasLocations());
  expectDumpResult(Locs, "reg12=[CFA+4], reg13=[CFA+8], reg14=same");

  // Verify RegisterLocations::getRegisterLocation() works as expected.
  std::optional<dwarf::UnwindLocation> OptionalLoc;
  OptionalLoc = Locs.getRegisterLocation(0);
  EXPECT_FALSE(OptionalLoc.has_value());

  OptionalLoc = Locs.getRegisterLocation(12);
  EXPECT_TRUE(OptionalLoc.has_value());
  EXPECT_EQ(*OptionalLoc, Reg12Loc);

  OptionalLoc = Locs.getRegisterLocation(13);
  EXPECT_TRUE(OptionalLoc.has_value());
  EXPECT_EQ(*OptionalLoc, Reg13Loc);

  OptionalLoc = Locs.getRegisterLocation(14);
  EXPECT_TRUE(OptionalLoc.has_value());
  EXPECT_EQ(*OptionalLoc, Reg14Loc);

  // Verify registers are correctly removed when multiple exist in the list.
  Locs.removeRegisterLocation(13);
  EXPECT_FALSE(Locs.getRegisterLocation(13).has_value());
  EXPECT_TRUE(Locs.hasLocations());
  expectDumpResult(Locs, "reg12=[CFA+4], reg14=same");
  Locs.removeRegisterLocation(14);
  EXPECT_FALSE(Locs.getRegisterLocation(14).has_value());
  EXPECT_TRUE(Locs.hasLocations());
  expectDumpResult(Locs, "reg12=[CFA+4]");
  Locs.removeRegisterLocation(12);
  EXPECT_FALSE(Locs.getRegisterLocation(12).has_value());
  EXPECT_FALSE(Locs.hasLocations());
  expectDumpResult(Locs, "");
}

// Test that empty rows are not added to UnwindTable when
// dwarf::CIE::CFIs or dwarf::FDE::CFIs is empty.
TEST(DWARFDebugFrame, UnwindTableEmptyRows) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  // Having an empty instructions list is fine.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {}), Succeeded());
  EXPECT_TRUE(TestCIE.cfis().empty());

  // Verify dwarf::UnwindTable::create() won't result in errors and
  // and empty rows are not added to CIE UnwindTable.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestCIE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const size_t ExpectedNumOfRows = 0;
  EXPECT_EQ(RowsOrErr->size(), ExpectedNumOfRows);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Having an empty instructions list is fine.
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {}), Succeeded());
  EXPECT_TRUE(TestFDE.cfis().empty());

  // Verify dwarf::UnwindTable::create() won't result in errors and
  // and empty rows are not added to FDE UnwindTable.
  RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  EXPECT_EQ(RowsOrErr->size(), ExpectedNumOfRows);
}

// Test that empty rows are not added to UnwindTable when dwarf::CIE::CFIs
// or dwarf::FDE::CFIs is not empty but has only DW_CFA_nop instructions.
TEST(DWARFDebugFrame, UnwindTableEmptyRows_NOPs) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  // Make a CIE that has only DW_CFA_nop instructions.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_nop}), Succeeded());
  EXPECT_TRUE(!TestCIE.cfis().empty());

  // Verify dwarf::UnwindTable::create() won't result in errors and
  // and empty rows are not added to CIE UnwindTable.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestCIE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const size_t ExpectedNumOfRows = 0;
  EXPECT_EQ(RowsOrErr->size(), ExpectedNumOfRows);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make an FDE that has only DW_CFA_nop instructions.
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {dwarf::DW_CFA_nop}), Succeeded());
  EXPECT_TRUE(!TestFDE.cfis().empty());

  // Verify dwarf::UnwindTable::create() won't result in errors and
  // and empty rows are not added to FDE UnwindTable.
  RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  EXPECT_EQ(RowsOrErr->size(), ExpectedNumOfRows);
}

TEST(DWARFDebugFrame, UnwindTableErrorNonAscendingFDERows) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition.
  constexpr uint8_t Reg = 12;
  constexpr uint8_t Offset = 32;
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, Reg, Offset}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that have valid
  // syntax, but will cause an error when we parse them into a UnwindTable.
  // Here we encode two DW_CFA_set_loc opcodes:
  //   DW_CFA_set_loc(0x1100)
  //   DW_CFA_set_loc(0x1000)
  // These opcodes cause a new row to be appended to the rows in a UnwindTable
  // and the resulting rows are not in ascending address order and should cause
  // a state machine error.
  EXPECT_THAT_ERROR(
      parseCFI(TestFDE, {dwarf::DW_CFA_set_loc, 0x00, 0x11, 0, 0, 0, 0, 0, 0,
                         dwarf::DW_CFA_set_loc, 0x00, 0x10, 0, 0, 0, 0, 0, 0}),
      Succeeded());

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(),
                    FailedWithMessage("DW_CFA_set_loc with adrress 0x1000 which"
                                      " must be greater than the current row "
                                      "address 0x1100"));
}

TEST(DWARFDebugFrame, UnwindTableError_DW_CFA_restore_state) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition.
  constexpr uint8_t Reg = 12;
  constexpr uint8_t Offset = 32;
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, Reg, Offset}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that have valid
  // syntax, but will cause an error when we parse them into a UnwindTable.
  // Here we encode a DW_CFA_restore_state opcode that was not preceded by a
  // DW_CFA_remember_state, and an error should be returned.
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {dwarf::DW_CFA_restore_state}),
                    Succeeded());

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(),
                    FailedWithMessage("DW_CFA_restore_state without a matching "
                                      "previous DW_CFA_remember_state"));
}

TEST(DWARFDebugFrame, UnwindTableError_DW_CFA_GNU_window_save) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition.
  constexpr uint8_t Reg = 12;
  constexpr uint8_t Offset = 32;
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, Reg, Offset}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that have valid
  // syntax, but will cause an error when we parse them into a UnwindTable.
  // Here we encode a DW_CFA_GNU_window_save that is not supported. I have not
  // found any documentation that describes what this does after some brief
  // searching.
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {dwarf::DW_CFA_GNU_window_save}),
                    Succeeded());

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(),
                    FailedWithMessage("DW_CFA opcode 0x2d is not supported for "
                                      "architecture x86_64"));
}

TEST(DWARFDebugFrame, UnwindTableError_DW_CFA_def_cfa_offset) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has an invalid CFA definition. We do this so we can try
  // and use a DW_CFA_def_cfa_register opcode in the FDE and get an appropriate
  // error back.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {}), Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that have valid
  // syntax, but will cause an error when we parse them into a UnwindTable.
  // Here we encode a DW_CFA_def_cfa_offset with a offset of 16, but our CIE
  // didn't define the CFA in terms of a register plus offset, so this should
  // cause an error.
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {dwarf::DW_CFA_def_cfa_offset, 16}),
                    Succeeded());

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(),
                    FailedWithMessage("DW_CFA_def_cfa_offset found when CFA "
                                      "rule was not RegPlusOffset"));
}

TEST(DWARFDebugFrame, UnwindTableDefCFAOffsetSFCFAError) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has an invalid CFA definition. We do this so we can try
  // and use a DW_CFA_def_cfa_offset_sf opcode in the FDE and get an
  // appropriate error back.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {}), Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that have valid
  // syntax, but will cause an error when we parse them into a UnwindTable.
  // Here we encode a DW_CFA_def_cfa_offset_sf with a offset of 4, but our CIE
  // didn't define the CFA in terms of a register plus offset, so this should
  // cause an error.
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {dwarf::DW_CFA_def_cfa_offset_sf, 4}),
                    Succeeded());

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(),
                    FailedWithMessage("DW_CFA_def_cfa_offset_sf found when CFA "
                                      "rule was not RegPlusOffset"));
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_def_cfa_register) {
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has only defines the CFA register with no offset. Some
  // architectures do this and we must ensure that we set the CFA value to be
  // equal to that register with no offset.
  constexpr uint8_t CFAReg = 12;
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa_register, CFAReg}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that have valid
  // syntax, but will cause an error when we parse them into a UnwindTable.
  // Here we encode a DW_CFA_def_cfa_register with a register number of 12, but
  // our CIE didn't define the CFA in terms of a register plus offset, so this
  // should cause an error.
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {}), Succeeded());

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 1u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getCFAValue(),
            dwarf::UnwindLocation::createIsRegisterPlusOffset(CFAReg, 0));
}

TEST(DWARFDebugFrame, UnwindTableRowPushingOpcodes) {
  // Test all opcodes that should end up pushing a UnwindRow into a UnwindTable.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  constexpr uint8_t CFAReg = 12;
  constexpr uint8_t CFAOffset = 32;
  constexpr uint8_t Reg = 13;
  constexpr uint8_t InReg = 14;

  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, CFAReg, CFAOffset,
                                       dwarf::DW_CFA_register, Reg, InReg}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that use all of the
  // row pushing opcodes. This will verify that all opcodes that should create
  // a row are correctly working. Each opcode will push a row prior to
  // advancing the address, and then a row will be automatically pushed at the
  // end of the parsing, so we should end up with 6 rows starting at address
  // 0x1000 (from the FDE) and incrementing each one by 4 * CodeAlignmentFactor
  // from the CIE.
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {dwarf::DW_CFA_advance_loc | 4,
                                       dwarf::DW_CFA_advance_loc1,
                                       4,
                                       dwarf::DW_CFA_advance_loc2,
                                       4,
                                       0,
                                       dwarf::DW_CFA_advance_loc4,
                                       4,
                                       0,
                                       0,
                                       0,
                                       dwarf::DW_CFA_set_loc,
                                       0x14,
                                       0x10,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0}),
                    Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs;
  VerifyLocs.setRegisterLocation(
      Reg, dwarf::UnwindLocation::createIsRegisterPlusOffset(InReg, 0));

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  ASSERT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 6u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs);
  EXPECT_EQ(Rows[1].getAddress(), 0x1004u);
  EXPECT_EQ(Rows[1].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[1].getRegisterLocations(), VerifyLocs);
  EXPECT_EQ(Rows[2].getAddress(), 0x1008u);
  EXPECT_EQ(Rows[2].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[2].getRegisterLocations(), VerifyLocs);
  EXPECT_EQ(Rows[3].getAddress(), 0x100cu);
  EXPECT_EQ(Rows[3].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[3].getRegisterLocations(), VerifyLocs);
  EXPECT_EQ(Rows[4].getAddress(), 0x1010u);
  EXPECT_EQ(Rows[4].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[4].getRegisterLocations(), VerifyLocs);
  EXPECT_EQ(Rows[5].getAddress(), 0x1014u);
  EXPECT_EQ(Rows[5].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[5].getRegisterLocations(), VerifyLocs);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_restore) {
  // Test that DW_CFA_restore works as expected when parsed in the state
  // machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  constexpr uint8_t CFAReg = 12;
  constexpr uint8_t CFAOffset = 32;
  constexpr uint8_t Reg = 13;
  constexpr uint8_t InReg = 14;
  constexpr int32_t RegCFAOffset = -8;

  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, CFAReg, CFAOffset,
                                       dwarf::DW_CFA_register, Reg, InReg}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that changes the rule
  // for register "Reg" to be [CFA-8], then push a row, and then restore the
  // register unwind rule for "Reg" using DW_CFA_restore. We should end up with
  // two rows:
  //   - one with Reg = [CFA-8]
  //   - one with Reg = InReg
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {dwarf::DW_CFA_offset | Reg, 1,
                                       dwarf::DW_CFA_advance_loc | 4,
                                       dwarf::DW_CFA_restore | Reg}),
                    Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs1;
  VerifyLocs1.setRegisterLocation(
      Reg, dwarf::UnwindLocation::createAtCFAPlusOffset(RegCFAOffset));

  dwarf::RegisterLocations VerifyLocs2;
  VerifyLocs2.setRegisterLocation(
      Reg, dwarf::UnwindLocation::createIsRegisterPlusOffset(InReg, 0));

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 2u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs1);
  EXPECT_EQ(Rows[1].getAddress(), 0x1004u);
  EXPECT_EQ(Rows[1].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[1].getRegisterLocations(), VerifyLocs2);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_restore_extended) {
  // Test that DW_CFA_restore works as expected when parsed in the state
  // machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  constexpr uint8_t CFAReg = 12;
  constexpr uint8_t CFAOffset = 32;
  constexpr uint8_t Reg = 13;
  constexpr uint8_t InReg = 14;
  constexpr int32_t RegCFAOffset = -8;

  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, CFAReg, CFAOffset,
                                       dwarf::DW_CFA_register, Reg, InReg}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that changes the rule
  // for register "Reg" to be [CFA-8], then push a row, and then restore the
  // register unwind rule for "Reg" using DW_CFA_restore_extended. We should
  // end up with two rows:
  //   - one with Reg = [CFA-8]
  //   - one with Reg = InReg
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {dwarf::DW_CFA_offset | Reg, 1,
                                       dwarf::DW_CFA_advance_loc | 4,
                                       dwarf::DW_CFA_restore_extended, Reg}),
                    Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs1;
  VerifyLocs1.setRegisterLocation(
      Reg, dwarf::UnwindLocation::createAtCFAPlusOffset(RegCFAOffset));

  dwarf::RegisterLocations VerifyLocs2;
  VerifyLocs2.setRegisterLocation(
      Reg, dwarf::UnwindLocation::createIsRegisterPlusOffset(InReg, 0));

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 2u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs1);
  EXPECT_EQ(Rows[1].getAddress(), 0x1004u);
  EXPECT_EQ(Rows[1].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[1].getRegisterLocations(), VerifyLocs2);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_offset) {
  // Test that DW_CFA_offset, DW_CFA_offset_extended and
  // DW_CFA_offset_extended_sf work as expected when parsed in the state
  // machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, 12, 32}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that changes the
  // unwind rules for the follwing registers:
  //   Reg1 = [CFA-8]
  //   Reg2 = [CFA-16]
  //   Reg3 = [CFA+8]
  constexpr uint8_t Reg1 = 14;
  constexpr uint8_t Reg2 = 15;
  constexpr uint8_t Reg3 = 16;
  constexpr uint8_t Neg1SLEB = 0x7f;
  EXPECT_THAT_ERROR(
      parseCFI(TestFDE,
               {dwarf::DW_CFA_offset | Reg1, 1, dwarf::DW_CFA_offset_extended,
                Reg2, 2, dwarf::DW_CFA_offset_extended_sf, Reg3, Neg1SLEB}),
      Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs;
  VerifyLocs.setRegisterLocation(
      Reg1, dwarf::UnwindLocation::createAtCFAPlusOffset(-8));
  VerifyLocs.setRegisterLocation(
      Reg2, dwarf::UnwindLocation::createAtCFAPlusOffset(-16));
  VerifyLocs.setRegisterLocation(
      Reg3, dwarf::UnwindLocation::createAtCFAPlusOffset(8));

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 1u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_val_offset) {
  // Test that DW_CFA_val_offset and DW_CFA_val_offset_sf work as expected when
  // parsed in the state machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, 12, 32}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that changes the
  // unwind rules for the follwing registers:
  //   Reg1 = [CFA-8]
  //   Reg2 = [CFA-16]
  //   Reg3 = [CFA+8]
  constexpr uint8_t Reg1 = 14;
  constexpr uint8_t Reg2 = 15;
  constexpr uint8_t Neg1SLEB = 0x7f;
  EXPECT_THAT_ERROR(
      parseCFI(TestFDE, {dwarf::DW_CFA_val_offset, Reg1, 1,
                         dwarf::DW_CFA_val_offset_sf, Reg2, Neg1SLEB}),
      Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs;
  VerifyLocs.setRegisterLocation(
      Reg1, dwarf::UnwindLocation::createIsCFAPlusOffset(-8));
  VerifyLocs.setRegisterLocation(
      Reg2, dwarf::UnwindLocation::createIsCFAPlusOffset(8));

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 1u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_nop) {
  // Test that DW_CFA_nop works as expected when parsed in the state machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, 12, 32}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that changes the
  // unwind rules for the follwing registers:
  //   Reg1 = [CFA-8]
  // The opcodes for setting Reg1 are preceded by a DW_CFA_nop.
  constexpr uint8_t Reg1 = 14;
  EXPECT_THAT_ERROR(
      parseCFI(TestFDE, {dwarf::DW_CFA_nop, dwarf::DW_CFA_offset | Reg1, 1}),
      Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs;
  VerifyLocs.setRegisterLocation(
      Reg1, dwarf::UnwindLocation::createAtCFAPlusOffset(-8));

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 1u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_remember_state) {
  // Test that DW_CFA_remember_state and DW_CFA_restore_state work as expected
  // when parsed in the state machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  constexpr uint8_t CFAOff1 = 32;
  constexpr uint8_t CFAOff2 = 16;
  constexpr uint8_t Reg1 = 14;
  constexpr uint8_t Reg2 = 15;
  constexpr uint8_t Reg3 = 16;

  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, 12, CFAOff1}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that encodes the
  // follwing rows:
  // 0x1000: CFA=reg12+CFAOff1: Reg1=[CFA-8]
  // 0x1004: CFA=reg12+CFAOff1: Reg1=[CFA-8] Reg2=[CFA-16]
  // 0x1008: CFA=reg12+CFAOff2: Reg1=[CFA-8] Reg2=[CFA-16] Reg3=[CFA-24]
  // 0x100C: CFA=reg12+CFAOff1: Reg1=[CFA-8] Reg2=[CFA-16]
  // 0x1010: CFA=reg12+CFAOff1: Reg1=[CFA-8]
  // This state machine will:
  //  - set Reg1 location
  //  - push a row (from DW_CFA_advance_loc)
  //  - remember the state
  //  - set Reg2 location
  //  - push a row (from DW_CFA_advance_loc)
  //  - remember the state
  //  - set CFA offset to CFAOff2
  //  - set Reg3 location
  //  - push a row (from DW_CFA_advance_loc)
  //  - remember the state where Reg1 and Reg2 were set
  //  - push a row (from DW_CFA_advance_loc)
  //  - remember the state where only Reg1 was set
  //  - push a row (automatically at the end of instruction parsing)
  // Then we verify that all registers are correct in all generated rows.
  EXPECT_THAT_ERROR(
      parseCFI(TestFDE,
               {dwarf::DW_CFA_offset | Reg1, 1, dwarf::DW_CFA_advance_loc | 4,
                dwarf::DW_CFA_remember_state, dwarf::DW_CFA_offset | Reg2, 2,
                dwarf::DW_CFA_advance_loc | 4, dwarf::DW_CFA_remember_state,
                dwarf::DW_CFA_def_cfa_offset, CFAOff2,
                dwarf::DW_CFA_offset | Reg3, 3, dwarf::DW_CFA_advance_loc | 4,
                dwarf::DW_CFA_restore_state, dwarf::DW_CFA_advance_loc | 4,
                dwarf::DW_CFA_restore_state}),
      Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs1;
  VerifyLocs1.setRegisterLocation(
      Reg1, dwarf::UnwindLocation::createAtCFAPlusOffset(-8));

  dwarf::RegisterLocations VerifyLocs2;
  VerifyLocs2.setRegisterLocation(
      Reg1, dwarf::UnwindLocation::createAtCFAPlusOffset(-8));
  VerifyLocs2.setRegisterLocation(
      Reg2, dwarf::UnwindLocation::createAtCFAPlusOffset(-16));

  dwarf::RegisterLocations VerifyLocs3;
  VerifyLocs3.setRegisterLocation(
      Reg1, dwarf::UnwindLocation::createAtCFAPlusOffset(-8));
  VerifyLocs3.setRegisterLocation(
      Reg2, dwarf::UnwindLocation::createAtCFAPlusOffset(-16));
  VerifyLocs3.setRegisterLocation(
      Reg3, dwarf::UnwindLocation::createAtCFAPlusOffset(-24));

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 5u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getCFAValue(),
            dwarf::UnwindLocation::createIsRegisterPlusOffset(12, CFAOff1));
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs1);

  EXPECT_EQ(Rows[1].getAddress(), 0x1004u);
  EXPECT_EQ(Rows[1].getCFAValue(),
            dwarf::UnwindLocation::createIsRegisterPlusOffset(12, CFAOff1));
  EXPECT_EQ(Rows[1].getRegisterLocations(), VerifyLocs2);

  EXPECT_EQ(Rows[2].getAddress(), 0x1008u);
  EXPECT_EQ(Rows[2].getCFAValue(),
            dwarf::UnwindLocation::createIsRegisterPlusOffset(12, CFAOff2));
  EXPECT_EQ(Rows[2].getRegisterLocations(), VerifyLocs3);

  EXPECT_EQ(Rows[3].getAddress(), 0x100Cu);
  EXPECT_EQ(Rows[3].getCFAValue(),
            dwarf::UnwindLocation::createIsRegisterPlusOffset(12, CFAOff1));
  EXPECT_EQ(Rows[3].getRegisterLocations(), VerifyLocs2);

  EXPECT_EQ(Rows[4].getAddress(), 0x1010u);
  EXPECT_EQ(Rows[4].getCFAValue(),
            dwarf::UnwindLocation::createIsRegisterPlusOffset(12, CFAOff1));
  EXPECT_EQ(Rows[4].getRegisterLocations(), VerifyLocs1);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_undefined) {
  // Test that DW_CFA_undefined works as expected when parsed in the state
  // machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, 12, 32}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that encodes the
  // follwing rows:
  // 0x1000: CFA=reg12+32: Reg1=undefined
  // Then we verify that all registers are correct in all generated rows.
  constexpr uint8_t Reg1 = 14;
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {dwarf::DW_CFA_undefined, Reg1}),
                    Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs;
  VerifyLocs.setRegisterLocation(Reg1,
                                 dwarf::UnwindLocation::createUndefined());

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 1u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_same_value) {
  // Test that DW_CFA_same_value works as expected when parsed in the state
  // machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, 12, 32}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that encodes the
  // follwing rows:
  // 0x1000: CFA=reg12+32: Reg1=same
  // Then we verify that all registers are correct in all generated rows.
  constexpr uint8_t Reg1 = 14;
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {dwarf::DW_CFA_same_value, Reg1}),
                    Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs;
  VerifyLocs.setRegisterLocation(Reg1, dwarf::UnwindLocation::createSame());

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 1u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_register) {
  // Test that DW_CFA_register works as expected when parsed in the state
  // machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, 12, 32}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that encodes the
  // follwing rows:
  // 0x1000: CFA=reg12+32: Reg1=same
  // Then we verify that all registers are correct in all generated rows.
  constexpr uint8_t Reg = 13;
  constexpr uint8_t InReg = 14;
  EXPECT_THAT_ERROR(parseCFI(TestFDE, {dwarf::DW_CFA_register, Reg, InReg}),
                    Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs;
  VerifyLocs.setRegisterLocation(
      Reg, dwarf::UnwindLocation::createIsRegisterPlusOffset(InReg, 0));

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 1u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_expression) {
  // Test that DW_CFA_expression works as expected when parsed in the state
  // machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, 12, 32}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that encodes the
  // follwing rows:
  // 0x1000: CFA=reg12+32: Reg1=DWARFExpr(DW_OP_reg12)
  // Then we verify that all registers are correct in all generated rows.
  constexpr uint8_t Reg = 13;
  constexpr uint8_t AddrSize = 8;
  std::vector<uint8_t> CFIBytes = {dwarf::DW_CFA_expression, Reg, 1,
                                   dwarf::DW_OP_reg12};

  EXPECT_THAT_ERROR(parseCFI(TestFDE, CFIBytes), Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs;

  std::vector<uint8_t> ExprBytes = {dwarf::DW_OP_reg12};
  DataExtractor ExprData(ExprBytes, true, AddrSize);
  DWARFExpression Expr(ExprData, AddrSize);
  VerifyLocs.setRegisterLocation(
      Reg, dwarf::UnwindLocation::createAtDWARFExpression(Expr));

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 1u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_val_expression) {
  // Test that DW_CFA_val_expression works as expected when parsed in the state
  // machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, 12, 32}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that encodes the
  // follwing rows:
  // 0x1000: CFA=reg12+32: Reg1=DWARFExpr(DW_OP_reg12)
  // Then we verify that all registers are correct in all generated rows.
  constexpr uint8_t Reg = 13;
  constexpr uint8_t AddrSize = 8;
  std::vector<uint8_t> CFIBytes = {dwarf::DW_CFA_val_expression, Reg, 1,
                                   dwarf::DW_OP_reg12};

  EXPECT_THAT_ERROR(parseCFI(TestFDE, CFIBytes), Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs;

  std::vector<uint8_t> ExprBytes = {dwarf::DW_OP_reg12};
  DataExtractor ExprData(ExprBytes, true, AddrSize);
  DWARFExpression Expr(ExprData, AddrSize);
  VerifyLocs.setRegisterLocation(
      Reg, dwarf::UnwindLocation::createIsDWARFExpression(Expr));

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 1u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_def_cfa) {
  // Test that DW_CFA_def_cfa, DW_CFA_def_cfa_sf, DW_CFA_def_cfa_register,
  // DW_CFA_def_cfa_offset, and DW_CFA_def_cfa_offset_sf works as expected when
  // parsed in the state machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  constexpr uint8_t CFAReg1 = 12;
  constexpr uint8_t CFAOff1 = 32;
  constexpr uint8_t CFAReg2 = 13;
  constexpr uint8_t CFAOff2 = 48;
  constexpr uint8_t Reg = 13;
  constexpr uint8_t InReg = 14;

  EXPECT_THAT_ERROR(parseCFI(TestCIE, {dwarf::DW_CFA_def_cfa, CFAReg1, CFAOff1,
                                       dwarf::DW_CFA_register, Reg, InReg}),
                    Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that use all of the
  // DW_CFA_def_cfa* opcodes. This will verify that all opcodes that should
  // create a row are correctly working.
  EXPECT_THAT_ERROR(
      parseCFI(
          TestFDE,
          {
              dwarf::DW_CFA_advance_loc | 4, dwarf::DW_CFA_def_cfa_register,
              CFAReg2, dwarf::DW_CFA_advance_loc | 4,
              dwarf::DW_CFA_def_cfa_offset, CFAOff2,
              dwarf::DW_CFA_advance_loc | 4, dwarf::DW_CFA_def_cfa_offset_sf,
              0x7c, // -4 SLEB to make offset = 32 (CFAOff1)
              dwarf::DW_CFA_advance_loc | 4, dwarf::DW_CFA_def_cfa_sf, CFAReg1,
              0x7a, // -6 SLEB to make CFA offset 48 (CFAOff2)
          }),
      Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs;
  VerifyLocs.setRegisterLocation(
      Reg, dwarf::UnwindLocation::createIsRegisterPlusOffset(InReg, 0));

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 5u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(
      Rows[0].getCFAValue(),
      dwarf::UnwindLocation::createIsRegisterPlusOffset(CFAReg1, CFAOff1));
  EXPECT_EQ(Rows[0].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs);

  EXPECT_EQ(Rows[1].getAddress(), 0x1004u);
  EXPECT_EQ(
      Rows[1].getCFAValue(),
      dwarf::UnwindLocation::createIsRegisterPlusOffset(CFAReg2, CFAOff1));
  EXPECT_EQ(Rows[1].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[1].getRegisterLocations(), VerifyLocs);

  EXPECT_EQ(Rows[2].getAddress(), 0x1008u);
  EXPECT_EQ(
      Rows[2].getCFAValue(),
      dwarf::UnwindLocation::createIsRegisterPlusOffset(CFAReg2, CFAOff2));
  EXPECT_EQ(Rows[2].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[2].getRegisterLocations(), VerifyLocs);

  EXPECT_EQ(Rows[3].getAddress(), 0x100cu);
  EXPECT_EQ(
      Rows[3].getCFAValue(),
      dwarf::UnwindLocation::createIsRegisterPlusOffset(CFAReg2, CFAOff1));
  EXPECT_EQ(Rows[3].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[3].getRegisterLocations(), VerifyLocs);

  EXPECT_EQ(Rows[4].getAddress(), 0x1010u);
  EXPECT_EQ(
      Rows[4].getCFAValue(),
      dwarf::UnwindLocation::createIsRegisterPlusOffset(CFAReg1, CFAOff2));
  EXPECT_EQ(Rows[4].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[4].getRegisterLocations(), VerifyLocs);
}

TEST(DWARFDebugFrame, UnwindTable_DW_CFA_LLVM_def_aspace_cfa) {
  // Test that DW_CFA_LLVM_def_aspace_cfa, DW_CFA_LLVM_def_aspace_cfa_sf,
  // DW_CFA_def_cfa_register, DW_CFA_def_cfa_offset, and
  // DW_CFA_def_cfa_offset_sf works as expected when parsed in the state
  // machine.
  dwarf::CIE TestCIE = createCIE(/*IsDWARF64=*/false,
                                 /*Offset=*/0x0,
                                 /*Length=*/0xff);

  dwarf::FDE TestFDE(/*IsDWARF64=*/true,
                     /*Offset=*/0x3333abcdabcd,
                     /*Length=*/0x4444abcdabcd,
                     /*CIEPointer=*/0x1111abcdabcd,
                     /*InitialLocation=*/0x1000,
                     /*AddressRange=*/0x1000,
                     /*Cie=*/&TestCIE,
                     /*LSDAAddress=*/std::nullopt,
                     /*Arch=*/Triple::x86_64);

  // Make a CIE that has a valid CFA definition and a single register unwind
  // rule for register that we will verify is in all of the pushed rows.
  constexpr uint8_t CFAReg1 = 12;
  constexpr uint8_t CFAOff1 = 32;
  constexpr uint8_t CFAReg2 = 13;
  constexpr uint8_t CFAOff2 = 48;
  constexpr uint8_t Reg = 13;
  constexpr uint8_t InReg = 14;
  constexpr uint8_t AddrSpace = 2;

  EXPECT_THAT_ERROR(
      parseCFI(TestCIE, {dwarf::DW_CFA_LLVM_def_aspace_cfa, CFAReg1, CFAOff1,
                         AddrSpace, dwarf::DW_CFA_register, Reg, InReg}),
      Succeeded());

  // Make a FDE with DWARF call frame instruction opcodes that use all of the
  // DW_CFA_def_cfa* opcodes. This will verify that all opcodes that should
  // create a row are correctly working.
  EXPECT_THAT_ERROR(
      parseCFI(
          TestFDE,
          {
              dwarf::DW_CFA_advance_loc | 4, dwarf::DW_CFA_def_cfa_register,
              CFAReg2, dwarf::DW_CFA_advance_loc | 4,
              dwarf::DW_CFA_def_cfa_offset, CFAOff2,
              dwarf::DW_CFA_advance_loc | 4, dwarf::DW_CFA_def_cfa_offset_sf,
              0x7c, // -4 SLEB to make offset = 32 (CFAOff1)
              dwarf::DW_CFA_advance_loc | 4, dwarf::DW_CFA_def_cfa_sf, CFAReg1,
              0x7a, // -6 SLEB to make CFA offset 48 (CFAOff2)
          }),
      Succeeded());

  // Create locations that we expect the UnwindRow objects to contain after
  // parsing the DWARF call frame instructions.
  dwarf::RegisterLocations VerifyLocs;
  VerifyLocs.setRegisterLocation(
      Reg, dwarf::UnwindLocation::createIsRegisterPlusOffset(InReg, 0));

  // Verify we catch state machine error.
  Expected<dwarf::UnwindTable> RowsOrErr = dwarf::UnwindTable::create(&TestFDE);
  EXPECT_THAT_ERROR(RowsOrErr.takeError(), Succeeded());
  const dwarf::UnwindTable &Rows = RowsOrErr.get();
  EXPECT_EQ(Rows.size(), 5u);
  EXPECT_EQ(Rows[0].getAddress(), 0x1000u);
  EXPECT_EQ(Rows[0].getCFAValue(),
            dwarf::UnwindLocation::createIsRegisterPlusOffset(CFAReg1, CFAOff1,
                                                              AddrSpace));
  EXPECT_EQ(Rows[0].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[0].getRegisterLocations(), VerifyLocs);

  EXPECT_EQ(Rows[1].getAddress(), 0x1004u);
  EXPECT_EQ(Rows[1].getCFAValue(),
            dwarf::UnwindLocation::createIsRegisterPlusOffset(CFAReg2, CFAOff1,
                                                              AddrSpace));
  EXPECT_EQ(Rows[1].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[1].getRegisterLocations(), VerifyLocs);

  EXPECT_EQ(Rows[2].getAddress(), 0x1008u);
  EXPECT_EQ(Rows[2].getCFAValue(),
            dwarf::UnwindLocation::createIsRegisterPlusOffset(CFAReg2, CFAOff2,
                                                              AddrSpace));
  EXPECT_EQ(Rows[2].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[2].getRegisterLocations(), VerifyLocs);

  EXPECT_EQ(Rows[3].getAddress(), 0x100cu);
  EXPECT_EQ(Rows[3].getCFAValue(),
            dwarf::UnwindLocation::createIsRegisterPlusOffset(CFAReg2, CFAOff1,
                                                              AddrSpace));
  EXPECT_EQ(Rows[3].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[3].getRegisterLocations(), VerifyLocs);

  EXPECT_EQ(Rows[4].getAddress(), 0x1010u);
  EXPECT_EQ(Rows[4].getCFAValue(),
            dwarf::UnwindLocation::createIsRegisterPlusOffset(CFAReg1, CFAOff2,
                                                              AddrSpace));
  EXPECT_EQ(Rows[4].getRegisterLocations().size(), 1u);
  EXPECT_EQ(Rows[4].getRegisterLocations(), VerifyLocs);
}

} // end anonymous namespace
