//=== - llvm/unittest/Support/AlignOfTest.cpp - Alignment utility tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef _MSC_VER
// Disable warnings about alignment-based structure padding.
// This must be above the includes to suppress warnings in included templates.
#pragma warning(disable:4324)
#endif

#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Compiler.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
// Disable warnings about questionable type definitions.
// We're testing that even questionable types work with the alignment utilities.
#ifdef _MSC_VER
#pragma warning(disable:4584)
#endif

// Suppress direct base '{anonymous}::S1' inaccessible in '{anonymous}::D9'
// due to ambiguity warning.
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Winaccessible-base"
#elif ((__GNUC__ * 100) + __GNUC_MINOR__) >= 402
// Pragma based warning suppression was introduced in GGC 4.2.  Additionally
// this warning is "enabled by default".  The warning still appears if -Wall is
// suppressed.  Apparently GCC suppresses it when -w is specifed, which is odd.
#pragma GCC diagnostic warning "-w"
#endif

// Define some fixed alignment types to use in these tests.
struct LLVM_ALIGNAS(1) A1 {};
struct LLVM_ALIGNAS(2) A2 {};
struct LLVM_ALIGNAS(4) A4 {};
struct LLVM_ALIGNAS(8) A8 {};

struct S1 {};
struct S2 { char a; };
struct S3 { int x; };
struct S4 { double y; };
struct S5 { A1 a1; A2 a2; A4 a4; A8 a8; };
struct S6 { double f(); };
struct D1 : S1 {};
struct D2 : S6 { float g(); };
struct D3 : S2 {};
struct D4 : S2 { int x; };
struct D5 : S3 { char c; };
struct D6 : S2, S3 {};
struct D7 : S1, S3 {};
struct D8 : S1, D4, D5 { double x[2]; };
struct D9 : S1, D1 { S1 s1; };
struct V1 { virtual ~V1(); };
struct V2 { int x; virtual ~V2(); };
struct V3 : V1 {
  ~V3() override;
};
struct V4 : virtual V2 { int y;
  ~V4() override;
};
struct V5 : V4, V3 { double z;
  ~V5() override;
};
struct V6 : S1 { virtual ~V6(); };
struct V7 : virtual V2, virtual V6 {
  ~V7() override;
};
struct V8 : V5, virtual V6, V7 { double zz;
  ~V8() override;
};

double S6::f() { return 0.0; }
float D2::g() { return 0.0f; }
V1::~V1() {}
V2::~V2() {}
V3::~V3() {}
V4::~V4() {}
V5::~V5() {}
V6::~V6() {}
V7::~V7() {}
V8::~V8() {}

template <typename M> struct T { M m; };

TEST(AlignOfTest, BasicAlignedArray) {
  EXPECT_LE(1u, alignof(AlignedCharArrayUnion<A1>));
  EXPECT_LE(2u, alignof(AlignedCharArrayUnion<A2>));
  EXPECT_LE(4u, alignof(AlignedCharArrayUnion<A4>));
  EXPECT_LE(8u, alignof(AlignedCharArrayUnion<A8>));

  EXPECT_LE(1u, sizeof(AlignedCharArrayUnion<A1>));
  EXPECT_LE(2u, sizeof(AlignedCharArrayUnion<A2>));
  EXPECT_LE(4u, sizeof(AlignedCharArrayUnion<A4>));
  EXPECT_LE(8u, sizeof(AlignedCharArrayUnion<A8>));

  EXPECT_EQ(1u, (alignof(AlignedCharArrayUnion<A1>)));
  EXPECT_EQ(2u, (alignof(AlignedCharArrayUnion<A1, A2>)));
  EXPECT_EQ(4u, (alignof(AlignedCharArrayUnion<A1, A2, A4>)));
  EXPECT_EQ(8u, (alignof(AlignedCharArrayUnion<A1, A2, A4, A8>)));

  EXPECT_EQ(1u, sizeof(AlignedCharArrayUnion<A1>));
  EXPECT_EQ(2u, sizeof(AlignedCharArrayUnion<A1, A2>));
  EXPECT_EQ(4u, sizeof(AlignedCharArrayUnion<A1, A2, A4>));
  EXPECT_EQ(8u, sizeof(AlignedCharArrayUnion<A1, A2, A4, A8>));

  EXPECT_EQ(1u, (alignof(AlignedCharArrayUnion<A1[1]>)));
  EXPECT_EQ(2u, (alignof(AlignedCharArrayUnion<A1[2], A2[1]>)));
  EXPECT_EQ(4u, (alignof(AlignedCharArrayUnion<A1[42], A2[55], A4[13]>)));
  EXPECT_EQ(8u, (alignof(AlignedCharArrayUnion<A1[2], A2[1], A4, A8>)));

  EXPECT_EQ(1u,  sizeof(AlignedCharArrayUnion<A1[1]>));
  EXPECT_EQ(2u,  sizeof(AlignedCharArrayUnion<A1[2], A2[1]>));
  EXPECT_EQ(4u,  sizeof(AlignedCharArrayUnion<A1[3], A2[2], A4>));
  EXPECT_EQ(16u, sizeof(AlignedCharArrayUnion<A1, A2[3],
                                              A4[3], A8>));

  // For other tests we simply assert that the alignment of the union mathes
  // that of the fundamental type and hope that we have any weird type
  // productions that would trigger bugs.
  EXPECT_EQ(alignof(T<char>), alignof(AlignedCharArrayUnion<char>));
  EXPECT_EQ(alignof(T<short>), alignof(AlignedCharArrayUnion<short>));
  EXPECT_EQ(alignof(T<int>), alignof(AlignedCharArrayUnion<int>));
  EXPECT_EQ(alignof(T<long>), alignof(AlignedCharArrayUnion<long>));
  EXPECT_EQ(alignof(T<long long>), alignof(AlignedCharArrayUnion<long long>));
  EXPECT_EQ(alignof(T<float>), alignof(AlignedCharArrayUnion<float>));
  EXPECT_EQ(alignof(T<double>), alignof(AlignedCharArrayUnion<double>));
  EXPECT_EQ(alignof(T<long double>),
            alignof(AlignedCharArrayUnion<long double>));
  EXPECT_EQ(alignof(T<void *>), alignof(AlignedCharArrayUnion<void *>));
  EXPECT_EQ(alignof(T<int *>), alignof(AlignedCharArrayUnion<int *>));
  EXPECT_EQ(alignof(T<double (*)(double)>),
            alignof(AlignedCharArrayUnion<double (*)(double)>));
  EXPECT_EQ(alignof(T<double (S6::*)()>),
            alignof(AlignedCharArrayUnion<double (S6::*)()>));
  EXPECT_EQ(alignof(S1), alignof(AlignedCharArrayUnion<S1>));
  EXPECT_EQ(alignof(S2), alignof(AlignedCharArrayUnion<S2>));
  EXPECT_EQ(alignof(S3), alignof(AlignedCharArrayUnion<S3>));
  EXPECT_EQ(alignof(S4), alignof(AlignedCharArrayUnion<S4>));
  EXPECT_EQ(alignof(S5), alignof(AlignedCharArrayUnion<S5>));
  EXPECT_EQ(alignof(S6), alignof(AlignedCharArrayUnion<S6>));
  EXPECT_EQ(alignof(D1), alignof(AlignedCharArrayUnion<D1>));
  EXPECT_EQ(alignof(D2), alignof(AlignedCharArrayUnion<D2>));
  EXPECT_EQ(alignof(D3), alignof(AlignedCharArrayUnion<D3>));
  EXPECT_EQ(alignof(D4), alignof(AlignedCharArrayUnion<D4>));
  EXPECT_EQ(alignof(D5), alignof(AlignedCharArrayUnion<D5>));
  EXPECT_EQ(alignof(D6), alignof(AlignedCharArrayUnion<D6>));
  EXPECT_EQ(alignof(D7), alignof(AlignedCharArrayUnion<D7>));
  EXPECT_EQ(alignof(D8), alignof(AlignedCharArrayUnion<D8>));
  EXPECT_EQ(alignof(D9), alignof(AlignedCharArrayUnion<D9>));
  EXPECT_EQ(alignof(V1), alignof(AlignedCharArrayUnion<V1>));
  EXPECT_EQ(alignof(V2), alignof(AlignedCharArrayUnion<V2>));
  EXPECT_EQ(alignof(V3), alignof(AlignedCharArrayUnion<V3>));
  EXPECT_EQ(alignof(V4), alignof(AlignedCharArrayUnion<V4>));
  EXPECT_EQ(alignof(V5), alignof(AlignedCharArrayUnion<V5>));
  EXPECT_EQ(alignof(V6), alignof(AlignedCharArrayUnion<V6>));
  EXPECT_EQ(alignof(V7), alignof(AlignedCharArrayUnion<V7>));

  // Some versions of MSVC get this wrong somewhat disturbingly. The failure
  // appears to be benign: alignof(V8) produces a preposterous value: 12
#ifndef _MSC_VER
  EXPECT_EQ(alignof(V8), alignof(AlignedCharArrayUnion<V8>));
#endif

  EXPECT_EQ(sizeof(char), sizeof(AlignedCharArrayUnion<char>));
  EXPECT_EQ(sizeof(char[1]), sizeof(AlignedCharArrayUnion<char[1]>));
  EXPECT_EQ(sizeof(char[2]), sizeof(AlignedCharArrayUnion<char[2]>));
  EXPECT_EQ(sizeof(char[3]), sizeof(AlignedCharArrayUnion<char[3]>));
  EXPECT_EQ(sizeof(char[4]), sizeof(AlignedCharArrayUnion<char[4]>));
  EXPECT_EQ(sizeof(char[5]), sizeof(AlignedCharArrayUnion<char[5]>));
  EXPECT_EQ(sizeof(char[8]), sizeof(AlignedCharArrayUnion<char[8]>));
  EXPECT_EQ(sizeof(char[13]), sizeof(AlignedCharArrayUnion<char[13]>));
  EXPECT_EQ(sizeof(char[16]), sizeof(AlignedCharArrayUnion<char[16]>));
  EXPECT_EQ(sizeof(char[21]), sizeof(AlignedCharArrayUnion<char[21]>));
  EXPECT_EQ(sizeof(char[32]), sizeof(AlignedCharArrayUnion<char[32]>));
  EXPECT_EQ(sizeof(short), sizeof(AlignedCharArrayUnion<short>));
  EXPECT_EQ(sizeof(int), sizeof(AlignedCharArrayUnion<int>));
  EXPECT_EQ(sizeof(long), sizeof(AlignedCharArrayUnion<long>));
  EXPECT_EQ(sizeof(long long),
            sizeof(AlignedCharArrayUnion<long long>));
  EXPECT_EQ(sizeof(float), sizeof(AlignedCharArrayUnion<float>));
  EXPECT_EQ(sizeof(double), sizeof(AlignedCharArrayUnion<double>));
  EXPECT_EQ(sizeof(long double),
            sizeof(AlignedCharArrayUnion<long double>));
  EXPECT_EQ(sizeof(void *), sizeof(AlignedCharArrayUnion<void *>));
  EXPECT_EQ(sizeof(int *), sizeof(AlignedCharArrayUnion<int *>));
  EXPECT_EQ(sizeof(double (*)(double)),
            sizeof(AlignedCharArrayUnion<double (*)(double)>));
  EXPECT_EQ(sizeof(double (S6::*)()),
            sizeof(AlignedCharArrayUnion<double (S6::*)()>));
  EXPECT_EQ(sizeof(S1), sizeof(AlignedCharArrayUnion<S1>));
  EXPECT_EQ(sizeof(S2), sizeof(AlignedCharArrayUnion<S2>));
  EXPECT_EQ(sizeof(S3), sizeof(AlignedCharArrayUnion<S3>));
  EXPECT_EQ(sizeof(S4), sizeof(AlignedCharArrayUnion<S4>));
  EXPECT_EQ(sizeof(S5), sizeof(AlignedCharArrayUnion<S5>));
  EXPECT_EQ(sizeof(S6), sizeof(AlignedCharArrayUnion<S6>));
  EXPECT_EQ(sizeof(D1), sizeof(AlignedCharArrayUnion<D1>));
  EXPECT_EQ(sizeof(D2), sizeof(AlignedCharArrayUnion<D2>));
  EXPECT_EQ(sizeof(D3), sizeof(AlignedCharArrayUnion<D3>));
  EXPECT_EQ(sizeof(D4), sizeof(AlignedCharArrayUnion<D4>));
  EXPECT_EQ(sizeof(D5), sizeof(AlignedCharArrayUnion<D5>));
  EXPECT_EQ(sizeof(D6), sizeof(AlignedCharArrayUnion<D6>));
  EXPECT_EQ(sizeof(D7), sizeof(AlignedCharArrayUnion<D7>));
  EXPECT_EQ(sizeof(D8), sizeof(AlignedCharArrayUnion<D8>));
  EXPECT_EQ(sizeof(D9), sizeof(AlignedCharArrayUnion<D9>));
  EXPECT_EQ(sizeof(D9[1]), sizeof(AlignedCharArrayUnion<D9[1]>));
  EXPECT_EQ(sizeof(D9[2]), sizeof(AlignedCharArrayUnion<D9[2]>));
  EXPECT_EQ(sizeof(D9[3]), sizeof(AlignedCharArrayUnion<D9[3]>));
  EXPECT_EQ(sizeof(D9[4]), sizeof(AlignedCharArrayUnion<D9[4]>));
  EXPECT_EQ(sizeof(D9[5]), sizeof(AlignedCharArrayUnion<D9[5]>));
  EXPECT_EQ(sizeof(D9[8]), sizeof(AlignedCharArrayUnion<D9[8]>));
  EXPECT_EQ(sizeof(D9[13]), sizeof(AlignedCharArrayUnion<D9[13]>));
  EXPECT_EQ(sizeof(D9[16]), sizeof(AlignedCharArrayUnion<D9[16]>));
  EXPECT_EQ(sizeof(D9[21]), sizeof(AlignedCharArrayUnion<D9[21]>));
  EXPECT_EQ(sizeof(D9[32]), sizeof(AlignedCharArrayUnion<D9[32]>));
  EXPECT_EQ(sizeof(V1), sizeof(AlignedCharArrayUnion<V1>));
  EXPECT_EQ(sizeof(V2), sizeof(AlignedCharArrayUnion<V2>));
  EXPECT_EQ(sizeof(V3), sizeof(AlignedCharArrayUnion<V3>));
  EXPECT_EQ(sizeof(V4), sizeof(AlignedCharArrayUnion<V4>));
  EXPECT_EQ(sizeof(V5), sizeof(AlignedCharArrayUnion<V5>));
  EXPECT_EQ(sizeof(V6), sizeof(AlignedCharArrayUnion<V6>));
  EXPECT_EQ(sizeof(V7), sizeof(AlignedCharArrayUnion<V7>));

  // Some versions of MSVC also get this wrong. The failure again appears to be
  // benign: sizeof(V8) is only 52 bytes, but our array reserves 56.
#ifndef _MSC_VER
  EXPECT_EQ(sizeof(V8), sizeof(AlignedCharArrayUnion<V8>));
#endif

  EXPECT_EQ(1u, (alignof(AlignedCharArray<1, 1>)));
  EXPECT_EQ(2u, (alignof(AlignedCharArray<2, 1>)));
  EXPECT_EQ(4u, (alignof(AlignedCharArray<4, 1>)));
  EXPECT_EQ(8u, (alignof(AlignedCharArray<8, 1>)));
  EXPECT_EQ(16u, (alignof(AlignedCharArray<16, 1>)));

  EXPECT_EQ(1u, sizeof(AlignedCharArray<1, 1>));
  EXPECT_EQ(7u, sizeof(AlignedCharArray<1, 7>));
  EXPECT_EQ(2u, sizeof(AlignedCharArray<2, 2>));
  EXPECT_EQ(16u, sizeof(AlignedCharArray<2, 16>));
}
} // end anonymous namespace
