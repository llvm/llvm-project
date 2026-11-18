//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Utility/ArchSpec.h"
#include "gtest/gtest.h"

#include "lldb/Register/Layouts/Common/RegisterLayout_x86.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_x86_64.h"
#include "llvm/Support/Endian.h"
#include "llvm/Testing/Support/Error.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

#include <array>
#include <cstddef>

using namespace lldb_private;
using llvm::Failed;
using llvm::Succeeded;
using FXSAVE = RegisterLayout_x86::FXSAVE;
using MMSReg = RegisterLayout_x86::MMSReg;

struct TagWordTestVector {
  uint16_t sw;
  uint16_t tw;
  uint8_t tw_abridged;
  int st_reg_num;
};

constexpr MMSReg st_from_comp(uint64_t mantissa, uint16_t sign_exp) {
  MMSReg ret = {};
  ret.comp.mantissa = mantissa;
  ret.comp.sign_exp = sign_exp;
  return ret;
}

const std::array<MMSReg, 8> st_regs = {
    st_from_comp(0x8000000000000000, 0x4000), // +2.0
    st_from_comp(0x3f00000000000000, 0x0000), // 1.654785e-4932
    st_from_comp(0x0000000000000000, 0x0000), // +0
    st_from_comp(0x0000000000000000, 0x8000), // -0
    st_from_comp(0x8000000000000000, 0x7fff), // +inf
    st_from_comp(0x8000000000000000, 0xffff), // -inf
    st_from_comp(0xc000000000000000, 0xffff), // nan
    st_from_comp(0x8000000000000000, 0xc000), // -2.0
};

const std::array<TagWordTestVector, 8> tag_word_test_vectors{
    TagWordTestVector{0x3800, 0x3fff, 0x80, 1},
    TagWordTestVector{0x3000, 0x2fff, 0xc0, 2},
    TagWordTestVector{0x2800, 0x27ff, 0xe0, 3},
    TagWordTestVector{0x2000, 0x25ff, 0xf0, 4},
    TagWordTestVector{0x1800, 0x25bf, 0xf8, 5},
    TagWordTestVector{0x1000, 0x25af, 0xfc, 6},
    TagWordTestVector{0x0800, 0x25ab, 0xfe, 7},
    TagWordTestVector{0x0000, 0x25a8, 0xff, 8},
};

TEST(RegisterLayout_x86TagWordTest, ReadsFullTagWordFromSavedState) {
  auto layout =
      RegisterLayoutFreeBSD_x86_64(ArchSpec("x86_64-unknown-freebsd"));
  const auto *tag = layout.GetDescription()->Find("ftag");
  ASSERT_NE(tag, nullptr);

  for (const auto &x : llvm::enumerate(tag_word_test_vectors)) {
    SCOPED_TRACE(llvm::formatv("tag_word_test_vectors[{0}]", x.index()));
    std::array<uint8_t, sizeof(FXSAVE)> bytes{};
    bytes[offsetof(FXSAVE, ftag)] = x.value().tw_abridged;
    llvm::support::endian::write16le(bytes.data() + offsetof(FXSAVE, fstat),
                                     x.value().sw);

    for (int i = 0; i < x.value().st_reg_num; ++i) {
      const auto &reg = st_regs[x.value().st_reg_num - i - 1];
      auto *slot = bytes.data() + offsetof(FXSAVE, stmm) + i * sizeof(MMSReg);
      llvm::support::endian::write64le(slot, reg.comp.mantissa);
      llvm::support::endian::write16le(slot + 8, reg.comp.sign_exp);
    }

    auto value =
        layout.Read(tag->id, RegisterData{RegisterLayout_x86::FP, bytes});
    ASSERT_THAT_EXPECTED(value, llvm::Succeeded());
    EXPECT_EQ(value->GetAsUInt16(), x.value().tw);
  }
}

TEST(RegisterLayout_x86TagWordTest, WritesAbridgedTagWordToSavedState) {
  auto layout =
      RegisterLayoutFreeBSD_x86_64(ArchSpec("x86_64-unknown-freebsd"));
  const auto *tag = layout.GetDescription()->Find("ftag");
  ASSERT_NE(tag, nullptr);

  for (const auto &x : llvm::enumerate(tag_word_test_vectors)) {
    SCOPED_TRACE(llvm::formatv("tag_word_test_vectors[{0}]", x.index()));
    std::array<uint8_t, sizeof(FXSAVE)> bytes{};
    bytes[offsetof(FXSAVE, ftag) + 1] = 0xa5;
    auto error =
        layout.Write(tag->id, RegisterValue(x.value().tw),
                     MutableRegisterData{RegisterLayout_x86::FP, bytes});
    ASSERT_THAT_ERROR(std::move(error), llvm::Succeeded());
    EXPECT_EQ(bytes[offsetof(FXSAVE, ftag)], x.value().tw_abridged);
    EXPECT_EQ(bytes[offsetof(FXSAVE, ftag) + 1], 0xa5);
  }
}

TEST(RegisterLayout_x86TagWordTest,
     TagWordConversionUsesTheOrdinaryLayoutPath) {
  auto layout =
      RegisterLayoutFreeBSD_x86_64(ArchSpec("x86_64-unknown-freebsd"));
  std::array<uint8_t, sizeof(FXSAVE)> fp{};
  fp[offsetof(FXSAVE, ftag)] = 0xff;
  fp[offsetof(FXSAVE, ftag) + 1] = 0xa5;
  const RegisterData read{RegisterLayout_x86::FP, fp};
  auto value = layout.Read(registers::x86_64::id::ftag, read);
  ASSERT_THAT_EXPECTED(value, Succeeded());
  EXPECT_EQ(value->GetAsUInt16(), 0x5555);
  auto banks = layout.GetRequiredBanks(registers::x86_64::id::ftag, true);
  ASSERT_THAT_EXPECTED(banks, Succeeded());
  EXPECT_EQ(*banks, (std::vector<RegisterBankID>{RegisterLayout_x86::FP}));

  MutableRegisterData write{RegisterLayout_x86::FP, fp};
  ASSERT_THAT_ERROR(layout.Write(registers::x86_64::id::ftag,
                                 RegisterValue(uint16_t(0xffff)), write),
                    Succeeded());
  EXPECT_EQ(fp[offsetof(FXSAVE, ftag)], 0);
  EXPECT_EQ(fp[offsetof(FXSAVE, ftag) + 1], 0xa5);
  EXPECT_THAT_EXPECTED(
      layout.Read(registers::x86_64::id::ftag,
                  RegisterData{RegisterLayout_x86::FP,
                               llvm::ArrayRef<uint8_t>(fp.data(), 16)}),
      Failed());
}

TEST(RegisterLayout_x86TagWordTest, CoreTagWordViewRemainsAbridged) {
  auto layout =
      RegisterLayoutFreeBSD_x86_64(ArchSpec("x86_64-unknown-freebsd"),
                                   RegisterLayout_x86::TagWordView::Abridged);
  std::array<uint8_t, sizeof(FXSAVE)> fp{};
  fp[offsetof(FXSAVE, ftag)] = 0xff;
  const RegisterData read{RegisterLayout_x86::FP, fp};

  auto value = layout.Read(registers::x86_64::id::ftag, read);
  ASSERT_THAT_EXPECTED(value, Succeeded());
  EXPECT_EQ(value->GetAsUInt16(), 0xff);
  EXPECT_EQ(layout.Find(registers::x86_64::id::ftag)->conversion, nullptr);
}
