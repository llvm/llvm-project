//===- ASanStackFrameLayoutTest.cpp - Tests for ComputeASanStackFrameLayout===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Utils/ASanStackFrameLayout.h"
#include "llvm/ADT/ArrayRef.h"
#include "gtest/gtest.h"
#include <sstream>

using namespace llvm;

static std::string
ShadowBytesToString(ArrayRef<uint8_t> ShadowBytes) {
  std::ostringstream os;
  for (size_t i = 0, n = ShadowBytes.size(); i < n; i++) {
    switch (ShadowBytes[i]) {
      case kAsanStackLeftRedzoneMagic:    os << "L"; break;
      case kAsanStackRightRedzoneMagic:   os << "R"; break;
      case kAsanStackMidRedzoneMagic:     os << "M"; break;
      case kAsanStackUseAfterScopeMagic:
        os << "S";
        break;
      default:                            os << (unsigned)ShadowBytes[i];
    }
  }
  return os.str();
}

// Use macro to preserve line information in EXPECT_EQ output.
#define TEST_LAYOUT(V, Granularity, MinHeaderSize, ExpectedDescr,              \
                    ExpectedShadow, ExpectedShadowAfterScope)                  \
  {                                                                            \
    SmallVector<ASanStackVariableDescription, 10> Vars = V;                    \
    ASanStackFrameLayout L =                                                   \
        ComputeASanStackFrameLayout(Vars, Granularity, MinHeaderSize);         \
    EXPECT_STREQ(ExpectedDescr,                                                \
                 ComputeASanStackFrameDescription(Vars).c_str());              \
    EXPECT_EQ(ExpectedShadow, ShadowBytesToString(GetShadowBytes(Vars, L)));   \
    EXPECT_EQ(ExpectedShadowAfterScope,                                        \
              ShadowBytesToString(GetShadowBytesAfterScope(Vars, L)));         \
  }

TEST(ASanStackFrameLayout, Test) {
#define VAR(name, size, lifetime, alignment, line)                             \
  ASanStackVariableDescription name##size##_##alignment = {                    \
    #name #size "_" #alignment,                                                \
    size,                                                                      \
    lifetime,                                                                  \
    alignment,                                                                 \
    0,                                                                         \
    0,                                                                         \
    line,                                                                      \
  }

  VAR(a, 1, 0, 1, 0);
  VAR(p, 1, 0, 32, 15);
  VAR(p, 1, 0, 256, 2700);
  VAR(a, 2, 0, 1, 0);
  VAR(a, 3, 0, 1, 0);
  VAR(a, 4, 0, 1, 0);
  VAR(a, 7, 0, 1, 0);
  VAR(a, 8, 8, 1, 0);
  VAR(a, 9, 0, 1, 0);
  VAR(a, 16, 16, 1, 0);
  VAR(a, 41, 9, 1, 7);
  VAR(a, 105, 103, 1, 0);
  VAR(a, 200, 97, 1, 0);

  TEST_LAYOUT({a1_1}, 8, 16, "1 16 1 4 a1_1", "LL1R", "LL1R");
  TEST_LAYOUT({a1_1}, 16, 16, "1 16 1 4 a1_1", "L1R", "L1R");
  TEST_LAYOUT({a1_1}, 32, 32, "1 32 1 4 a1_1", "L1R", "L1R");
  TEST_LAYOUT({a1_1}, 64, 64, "1 64 1 4 a1_1", "L1R", "L1R");
  TEST_LAYOUT({p1_32}, 8, 32, "1 32 1 8 p1_32:15", "LLLL1RRR", "LLLL1RRR");
  TEST_LAYOUT({p1_32}, 8, 64, "1 64 1 8 p1_32:15", "LLLLLLLL1RRRRRRR",
              "LLLLLLLL1RRRRRRR");

  TEST_LAYOUT({a1_1}, 8, 32, "1 32 1 4 a1_1", "LLLL1RRR", "LLLL1RRR");
  TEST_LAYOUT({a2_1}, 8, 32, "1 32 2 4 a2_1", "LLLL2RRR", "LLLL2RRR");
  TEST_LAYOUT({a3_1}, 8, 32, "1 32 3 4 a3_1", "LLLL3RRR", "LLLL3RRR");
  TEST_LAYOUT({a4_1}, 8, 32, "1 32 4 4 a4_1", "LLLL4RRR", "LLLL4RRR");
  TEST_LAYOUT({a7_1}, 8, 32, "1 32 7 4 a7_1", "LLLL7RRR", "LLLL7RRR");
  TEST_LAYOUT({a8_1}, 8, 32, "1 32 8 4 a8_1", "LLLL0RRR", "LLLLSRRR");
  TEST_LAYOUT({a9_1}, 8, 32, "1 32 9 4 a9_1", "LLLL01RR", "LLLL01RR");
  TEST_LAYOUT({a16_1}, 8, 32, "1 32 16 5 a16_1", "LLLL00RR", "LLLLSSRR");
  TEST_LAYOUT({p1_256}, 8, 32, "1 256 1 11 p1_256:2700",
              "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL1RRR",
              "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL1RRR");
  TEST_LAYOUT({a41_1}, 8, 32, "1 32 41 7 a41_1:7", "LLLL000001RRRRRR",
              "LLLLSS0001RRRRRR");
  TEST_LAYOUT({a105_1}, 8, 32, "1 32 105 6 a105_1", "LLLL00000000000001RRRRRR",
              "LLLLSSSSSSSSSSSSS1RRRRRR");

  {
    SmallVector<ASanStackVariableDescription, 10> t = {a1_1, p1_256};
    TEST_LAYOUT(t, 8, 32, "2 256 1 11 p1_256:2700 272 1 4 a1_1",
                "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL1M1R",
                "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL1M1R");
  }

  {
    SmallVector<ASanStackVariableDescription, 10> t = {a1_1, a16_1, a41_1};
    TEST_LAYOUT(t, 8, 32, "3 32 1 4 a1_1 48 16 5 a16_1 80 41 7 a41_1:7",
                "LLLL1M00MM000001RRRR", "LLLL1MSSMMSS0001RRRR");
  }

  TEST_LAYOUT({a2_1}, 32, 32, "1 32 2 4 a2_1", "L2R", "L2R");
  TEST_LAYOUT({a9_1}, 32, 32, "1 32 9 4 a9_1", "L9R", "L9R");
  TEST_LAYOUT({a16_1}, 32, 32, "1 32 16 5 a16_1", "L16R", "LSR");
  TEST_LAYOUT({p1_256}, 32, 32, "1 256 1 11 p1_256:2700",
              "LLLLLLLL1R", "LLLLLLLL1R");
  TEST_LAYOUT({a41_1}, 32, 32, "1 32 41 7 a41_1:7", "L09R",
              "LS9R");
  TEST_LAYOUT({a105_1}, 32, 32, "1 32 105 6 a105_1", "L0009R",
              "LSSSSR");
  TEST_LAYOUT({a200_1}, 32, 32, "1 32 200 6 a200_1", "L0000008RR",
              "LSSSS008RR");

  {
    SmallVector<ASanStackVariableDescription, 10> t = {a1_1, p1_256};
    TEST_LAYOUT(t, 32, 32, "2 256 1 11 p1_256:2700 320 1 4 a1_1",
                "LLLLLLLL1M1R", "LLLLLLLL1M1R");
  }

  {
    SmallVector<ASanStackVariableDescription, 10> t = {a1_1, a16_1, a41_1};
    TEST_LAYOUT(t, 32, 32, "3 32 1 4 a1_1 96 16 5 a16_1 160 41 7 a41_1:7",
                "L1M16M09R", "L1MSMS9R");
  }
#undef VAR
#undef TEST_LAYOUT
}
