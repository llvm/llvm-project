//===- llvm/unittest/DebugInfo/DWARFDebugFrameTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
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
                    /*Personality=*/None,
                    /*PersonalityEnc=*/None,
                    /*Arch=*/Triple::x86_64);
}

void expectDumpResult(const dwarf::CIE &TestCIE, bool IsEH,
                      StringRef ExpectedFirstLine) {
  std::string Output;
  raw_string_ostream OS(Output);
  TestCIE.dump(OS, DIDumpOptions(), /*MRI=*/nullptr, IsEH);
  OS.flush();
  StringRef FirstLine = StringRef(Output).split('\n').first;
  EXPECT_EQ(FirstLine, ExpectedFirstLine);
}

void expectDumpResult(const dwarf::FDE &TestFDE, bool IsEH,
                      StringRef ExpectedFirstLine) {
  std::string Output;
  raw_string_ostream OS(Output);
  TestFDE.dump(OS, DIDumpOptions(), /*MRI=*/nullptr, IsEH);
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
                     /*LSDAAddress=*/None,
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
                     /*LSDAAddress=*/None,
                     /*Arch=*/Triple::x86_64);
  expectDumpResult(TestFDE, /*IsEH=*/true,
                   "1111abcdabcd 00002222abcdabcd 0033abcd FDE "
                   "cie=1111ab9a000c pc=4444abcdabcd...5555bcdebcde");
}

static Error parseCFI(dwarf::CIE &C, ArrayRef<uint8_t> Instructions,
                      Optional<uint64_t> Size = None) {
  DWARFDataExtractor Data(Instructions, /*IsLittleEndian=*/true,
                          /*AddressSize=*/8);
  uint64_t Offset = 0;
  const uint64_t EndOffset = Size ? *Size : (uint64_t)Instructions.size();
  return C.cfis().parse(Data, &Offset, EndOffset);
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
                       dwarf::DW_CFA_def_cfa, dwarf::DW_CFA_val_offset})
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

  for (uint8_t Inst : {dwarf::DW_CFA_offset_extended_sf,
                       dwarf::DW_CFA_def_cfa_sf, dwarf::DW_CFA_val_offset_sf})
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

} // end anonymous namespace
