//===-- DWARFExpressionTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/DWARFExpression.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

static llvm::Expected<Scalar> Evaluate(llvm::ArrayRef<uint8_t> expr,
                                       lldb::ModuleSP module_sp = {},
                                       DWARFUnit *unit = nullptr) {
  DataExtractor extractor(expr.data(), expr.size(), lldb::eByteOrderLittle,
                          /*addr_size*/ 4);
  Value result;
  Status status;
  if (!DWARFExpression::Evaluate(
          /*exe_ctx*/ nullptr, /*reg_ctx*/ nullptr, module_sp, extractor, unit,
          lldb::eRegisterKindLLDB,
          /*initial_value_ptr*/ nullptr,
          /*object_address_ptr*/ nullptr, result, &status))
    return status.ToError();

  switch (result.GetValueType()) {
  case Value::eValueTypeScalar:
    return result.GetScalar();
  case Value::eValueTypeHostAddress: {
    // Convert small buffers to scalars to simplify the tests.
    DataBufferHeap &buf = result.GetBuffer();
    if (buf.GetByteSize() <= 8) {
      uint64_t val = 0;
      memcpy(&val, buf.GetBytes(), buf.GetByteSize());
      return Scalar(llvm::APInt(buf.GetByteSize()*8, val, false));
    }
  }
    LLVM_FALLTHROUGH;
  default:
    return status.ToError();
  }
}

class DWARFExpressionTester : public YAMLModuleTester {
public:
  using YAMLModuleTester::YAMLModuleTester;
  llvm::Expected<Scalar> Eval(llvm::ArrayRef<uint8_t> expr) {
    return ::Evaluate(expr, m_module_sp, m_dwarf_unit.get());
  }
};

/// Unfortunately Scalar's operator==() is really picky.
static Scalar GetScalar(unsigned bits, uint64_t value, bool sign) {
  Scalar scalar;
  auto type = Scalar::GetBestTypeForBitSize(bits, sign);
  switch (type) {
  case Scalar::e_sint:
    scalar = Scalar((int)value);
    break;
  case Scalar::e_slong:
    scalar = Scalar((long)value);
    break;
  case Scalar::e_slonglong:
    scalar = Scalar((long long)value);
    break;
  case Scalar::e_uint:
    scalar = Scalar((unsigned int)value);
    break;
  case Scalar::e_ulong:
    scalar = Scalar((unsigned long)value);
    break;
  case Scalar::e_ulonglong:
    scalar = Scalar((unsigned long long)value);
    break;
  default:
    llvm_unreachable("not implemented");
  }
  scalar.TruncOrExtendTo(type, bits);
  if (sign)
    scalar.MakeSigned();
  else
    scalar.MakeUnsigned();
  return scalar;
}

TEST(DWARFExpression, DW_OP_pick) {
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit1, DW_OP_lit0, DW_OP_pick, 0}),
                       llvm::HasValue(0));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit1, DW_OP_lit0, DW_OP_pick, 1}),
                       llvm::HasValue(1));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit1, DW_OP_lit0, DW_OP_pick, 2}),
                       llvm::Failed());
}

TEST(DWARFExpression, DW_OP_convert) {
  /// Auxiliary debug info.
  const char *yamldata =
      "debug_abbrev:\n"
      "  - Code:            0x00000001\n"
      "    Tag:             DW_TAG_compile_unit\n"
      "    Children:        DW_CHILDREN_yes\n"
      "    Attributes:\n"
      "      - Attribute:       DW_AT_language\n"
      "        Form:            DW_FORM_data2\n"
      "  - Code:            0x00000002\n"
      "    Tag:             DW_TAG_base_type\n"
      "    Children:        DW_CHILDREN_no\n"
      "    Attributes:\n"
      "      - Attribute:       DW_AT_encoding\n"
      "        Form:            DW_FORM_data1\n"
      "      - Attribute:       DW_AT_byte_size\n"
      "        Form:            DW_FORM_data1\n"
      "debug_info:\n"
      "  - Length:\n"
      "      TotalLength:     0\n"
      "    Version:         4\n"
      "    AbbrOffset:      0\n"
      "    AddrSize:        8\n"
      "    Entries:\n"
      "      - AbbrCode:        0x00000001\n"
      "        Values:\n"
      "          - Value:           0x000000000000000C\n"
      // 0x0000000e:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000007\n" // DW_ATE_unsigned
      "          - Value:           0x0000000000000004\n"
      // 0x00000011:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000007\n" // DW_ATE_unsigned
      "          - Value:           0x0000000000000008\n"
      // 0x00000014:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000005\n" // DW_ATE_signed
      "          - Value:           0x0000000000000008\n"
      // 0x00000017:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000008\n" // DW_ATE_unsigned_char
      "          - Value:           0x0000000000000001\n"
      // 0x0000001a:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000006\n" // DW_ATE_signed_char
      "          - Value:           0x0000000000000001\n"
      // 0x0000001d:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x000000000000000b\n" // DW_ATE_numeric_string
      "          - Value:           0x0000000000000001\n"
      ""
      "      - AbbrCode:        0x00000000\n"
      "        Values:          []\n";
  uint8_t offs_uint32_t = 0x0000000e;
  uint8_t offs_uint64_t = 0x00000011;
  uint8_t offs_sint64_t = 0x00000014;
  uint8_t offs_uchar = 0x00000017;
  uint8_t offs_schar = 0x0000001a;

  DWARFExpressionTester t(yamldata, "i386-unknown-linux");
  ASSERT_TRUE((bool)t.GetDwarfUnit());

  // Constant is given as little-endian.
  bool is_signed = true;
  bool not_signed = false;

  //
  // Positive tests.
  //

  // Truncate to default unspecified (pointer-sized) type.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const8u, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, //
              DW_OP_convert, 0x00}),
      llvm::HasValue(GetScalar(32, 0x44332211, not_signed)));
  // Truncate to 32 bits.
  EXPECT_THAT_EXPECTED(t.Eval({DW_OP_const8u, //
                               0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,//
                               DW_OP_convert, offs_uint32_t}),
                       llvm::HasValue(GetScalar(32, 0x44332211, not_signed)));

  // Leave as is.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const8u, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, //
              DW_OP_convert, offs_uint64_t}),
      llvm::HasValue(GetScalar(64, 0x8877665544332211, not_signed)));

  // Sign-extend to 64 bits.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const4s, 0xcc, 0xdd, 0xee, 0xff, //
              DW_OP_convert, offs_sint64_t}),
      llvm::HasValue(GetScalar(64, 0xffffffffffeeddcc, is_signed)));

  // Truncate to 8 bits.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const4s, 'A', 'B', 'C', 'D', DW_OP_convert, offs_uchar}),
      llvm::HasValue(GetScalar(8, 'A', not_signed)));

  // Also truncate to 8 bits.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const4s, 'A', 'B', 'C', 'D', DW_OP_convert, offs_schar}),
      llvm::HasValue(GetScalar(8, 'A', is_signed)));

  //
  // Errors.
  //

  // No Module.
  EXPECT_THAT_ERROR(Evaluate({DW_OP_const1s, 'X', DW_OP_convert, 0x00}, nullptr,
                             t.GetDwarfUnit().get())
                        .takeError(),
                    llvm::Failed());

  // No DIE.
  EXPECT_THAT_ERROR(
      t.Eval({DW_OP_const1s, 'X', DW_OP_convert, 0x01}).takeError(),
      llvm::Failed());

  // Unsupported.
  EXPECT_THAT_ERROR(
      t.Eval({DW_OP_const1s, 'X', DW_OP_convert, 0x1d}).takeError(),
      llvm::Failed());
}

TEST(DWARFExpression, DW_OP_stack_value) {
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_stack_value}), llvm::Failed());
}

TEST(DWARFExpression, DW_OP_piece) {
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_const2u, 0x11, 0x22, DW_OP_piece, 2,
                                 DW_OP_const2u, 0x33, 0x44, DW_OP_piece, 2}),
                       llvm::HasValue(GetScalar(32, 0x44332211, true)));
  EXPECT_THAT_EXPECTED(
      Evaluate({DW_OP_piece, 1, DW_OP_const1u, 0xff, DW_OP_piece, 1}),
      // Note that the "00" should really be "undef", but we can't
      // represent that yet.
      llvm::HasValue(GetScalar(16, 0xff00, true)));
}

TEST(DWARFExpression, DW_OP_unknown) {
  EXPECT_THAT_EXPECTED(
      Evaluate({0xff}),
      llvm::FailedWithMessage(
          "Unhandled opcode DW_OP_unknown_ff in DWARFExpression"));
}
