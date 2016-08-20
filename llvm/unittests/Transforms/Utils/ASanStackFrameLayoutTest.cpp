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
      default:                            os << (unsigned)ShadowBytes[i];
    }
  }
  return os.str();
}

static void TestLayout(SmallVector<ASanStackVariableDescription, 10> Vars,
                       size_t Granularity, size_t MinHeaderSize,
                       const std::string &ExpectedDescr,
                       const std::string &ExpectedShadow) {
  ASanStackFrameLayout L;
  ComputeASanStackFrameLayout(Vars, Granularity, MinHeaderSize, &L);
  EXPECT_EQ(ExpectedDescr, L.DescriptionString);
  EXPECT_EQ(ExpectedShadow, ShadowBytesToString(L.ShadowBytes));
}

TEST(ASanStackFrameLayout, Test) {
#define VEC1(a) SmallVector<ASanStackVariableDescription, 10>(1, a)
#define VEC(a)                                                                 \
  SmallVector<ASanStackVariableDescription, 10>(a, a + sizeof(a) / sizeof(a[0]))

#define VAR(name, size, alignment)                                             \
  ASanStackVariableDescription name##size##_##alignment = {                    \
    #name #size "_" #alignment,                                                \
    size,                                                                      \
    alignment,                                                                 \
    0,                                                                         \
    0                                                                          \
  }

  VAR(a, 1, 1);
  VAR(p, 1, 32);
  VAR(p, 1, 256);
  VAR(a, 2, 1);
  VAR(a, 3, 1);
  VAR(a, 4, 1);
  VAR(a, 7, 1);
  VAR(a, 8, 1);
  VAR(a, 9, 1);
  VAR(a, 16, 1);
  VAR(a, 41, 1);
  VAR(a, 105, 1);

  TestLayout(VEC1(a1_1), 8, 16, "1 16 1 4 a1_1", "LL1R");
  TestLayout(VEC1(a1_1), 64, 64, "1 64 1 4 a1_1", "L1");
  TestLayout(VEC1(p1_32), 8, 32, "1 32 1 5 p1_32", "LLLL1RRR");
  TestLayout(VEC1(p1_32), 8, 64, "1 64 1 5 p1_32", "LLLLLLLL1RRRRRRR");

  TestLayout(VEC1(a1_1), 8, 32, "1 32 1 4 a1_1", "LLLL1RRR");
  TestLayout(VEC1(a2_1), 8, 32, "1 32 2 4 a2_1", "LLLL2RRR");
  TestLayout(VEC1(a3_1), 8, 32, "1 32 3 4 a3_1", "LLLL3RRR");
  TestLayout(VEC1(a4_1), 8, 32, "1 32 4 4 a4_1", "LLLL4RRR");
  TestLayout(VEC1(a7_1), 8, 32, "1 32 7 4 a7_1", "LLLL7RRR");
  TestLayout(VEC1(a8_1), 8, 32, "1 32 8 4 a8_1", "LLLL0RRR");
  TestLayout(VEC1(a9_1), 8, 32, "1 32 9 4 a9_1", "LLLL01RR");
  TestLayout(VEC1(a16_1), 8, 32, "1 32 16 5 a16_1", "LLLL00RR");
  TestLayout(VEC1(p1_256), 8, 32, "1 256 1 6 p1_256",
             "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL1RRR");
  TestLayout(VEC1(a41_1), 8, 32, "1 32 41 5 a41_1", "LLLL000001RRRRRR");
  TestLayout(VEC1(a105_1), 8, 32, "1 32 105 6 a105_1",
             "LLLL00000000000001RRRRRR");

  {
    ASanStackVariableDescription t[] = {a1_1, p1_256};
    TestLayout(VEC(t), 8, 32,
               "2 256 1 6 p1_256 272 1 4 a1_1",
               "LLLLLLLL" "LLLLLLLL" "LLLLLLLL" "LLLLLLLL" "1M1R");
  }

  {
    ASanStackVariableDescription t[] = {a1_1, a16_1, a41_1};
    TestLayout(VEC(t), 8, 32,
               "3 32 1 4 a1_1 48 16 5 a16_1 80 41 5 a41_1",
               "LLLL" "1M00" "MM00" "0001" "RRRR");
  }
#undef VEC1
#undef VEC
#undef VAR
}
