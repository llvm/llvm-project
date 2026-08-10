//===-- RegisterContextTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/Process/Utility/RegisterContext_x86.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

#include <array>

using namespace lldb_private;

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

TEST(RegisterContext_x86Test, AbridgedToFullTagWord) {
  for (const auto &x : llvm::enumerate(tag_word_test_vectors)) {
    SCOPED_TRACE(llvm::formatv("tag_word_test_vectors[{0}]", x.index()));
    std::array<MMSReg, 8> test_regs;
    for (int i = 0; i < x.value().st_reg_num; ++i)
      test_regs[i] = st_regs[x.value().st_reg_num - i - 1];
    EXPECT_EQ(
        AbridgedToFullTagWord(x.value().tw_abridged, x.value().sw, test_regs),
        x.value().tw);
  }
}

TEST(RegisterContext_x86Test, FullToAbridgedTagWord) {
  for (const auto &x : llvm::enumerate(tag_word_test_vectors)) {
    SCOPED_TRACE(llvm::formatv("tag_word_test_vectors[{0}]", x.index()));
    EXPECT_EQ(FullToAbridgedTagWord(x.value().tw), x.value().tw_abridged);
  }
}
