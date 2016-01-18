//=== - llvm/unittest/Support/AlignOfTest.cpp - Alignment utility tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

// Ensure alignment is a compile-time constant.
char LLVM_ATTRIBUTE_UNUSED test_arr1
  [AlignOf<char>::Alignment > 0]
  [AlignOf<short>::Alignment > 0]
  [AlignOf<int>::Alignment > 0]
  [AlignOf<long>::Alignment > 0]
  [AlignOf<long long>::Alignment > 0]
  [AlignOf<float>::Alignment > 0]
  [AlignOf<double>::Alignment > 0]
  [AlignOf<long double>::Alignment > 0]
  [AlignOf<void *>::Alignment > 0]
  [AlignOf<int *>::Alignment > 0]
  [AlignOf<double (*)(double)>::Alignment > 0]
  [AlignOf<double (S6::*)()>::Alignment > 0];
char LLVM_ATTRIBUTE_UNUSED test_arr2
  [AlignOf<A1>::Alignment > 0]
  [AlignOf<A2>::Alignment > 0]
  [AlignOf<A4>::Alignment > 0]
  [AlignOf<A8>::Alignment > 0];
char LLVM_ATTRIBUTE_UNUSED test_arr3
  [AlignOf<S1>::Alignment > 0]
  [AlignOf<S2>::Alignment > 0]
  [AlignOf<S3>::Alignment > 0]
  [AlignOf<S4>::Alignment > 0]
  [AlignOf<S5>::Alignment > 0]
  [AlignOf<S6>::Alignment > 0];
char LLVM_ATTRIBUTE_UNUSED test_arr4
  [AlignOf<D1>::Alignment > 0]
  [AlignOf<D2>::Alignment > 0]
  [AlignOf<D3>::Alignment > 0]
  [AlignOf<D4>::Alignment > 0]
  [AlignOf<D5>::Alignment > 0]
  [AlignOf<D6>::Alignment > 0]
  [AlignOf<D7>::Alignment > 0]
  [AlignOf<D8>::Alignment > 0]
  [AlignOf<D9>::Alignment > 0];
char LLVM_ATTRIBUTE_UNUSED test_arr5
  [AlignOf<V1>::Alignment > 0]
  [AlignOf<V2>::Alignment > 0]
  [AlignOf<V3>::Alignment > 0]
  [AlignOf<V4>::Alignment > 0]
  [AlignOf<V5>::Alignment > 0]
  [AlignOf<V6>::Alignment > 0]
  [AlignOf<V7>::Alignment > 0]
  [AlignOf<V8>::Alignment > 0];

TEST(AlignOfTest, BasicAlignmentInvariants) {
  EXPECT_LE(1u, alignOf<A1>());
  EXPECT_LE(2u, alignOf<A2>());
  EXPECT_LE(4u, alignOf<A4>());
  EXPECT_LE(8u, alignOf<A8>());

  EXPECT_EQ(1u, alignOf<char>());
  EXPECT_LE(alignOf<char>(),   alignOf<short>());
  EXPECT_LE(alignOf<short>(),  alignOf<int>());
  EXPECT_LE(alignOf<int>(),    alignOf<long>());
  EXPECT_LE(alignOf<long>(),   alignOf<long long>());
  EXPECT_LE(alignOf<char>(),   alignOf<float>());
  EXPECT_LE(alignOf<float>(),  alignOf<double>());
  EXPECT_LE(alignOf<char>(),   alignOf<long double>());
  EXPECT_LE(alignOf<char>(),   alignOf<void *>());
  EXPECT_EQ(alignOf<void *>(), alignOf<int *>());
  EXPECT_LE(alignOf<char>(),   alignOf<S1>());
  EXPECT_LE(alignOf<S1>(),     alignOf<S2>());
  EXPECT_LE(alignOf<S1>(),     alignOf<S3>());
  EXPECT_LE(alignOf<S1>(),     alignOf<S4>());
  EXPECT_LE(alignOf<S1>(),     alignOf<S5>());
  EXPECT_LE(alignOf<S1>(),     alignOf<S6>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D1>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D2>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D3>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D4>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D5>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D6>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D7>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D8>());
  EXPECT_LE(alignOf<S1>(),     alignOf<D9>());
  EXPECT_LE(alignOf<S1>(),     alignOf<V1>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V2>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V3>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V4>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V5>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V6>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V7>());
  EXPECT_LE(alignOf<V1>(),     alignOf<V8>());
}

TEST(AlignOfTest, BasicAlignedArray) {
  EXPECT_LE(1u, alignOf<AlignedCharArrayUnion<A1> >());
  EXPECT_LE(2u, alignOf<AlignedCharArrayUnion<A2> >());
  EXPECT_LE(4u, alignOf<AlignedCharArrayUnion<A4> >());
  EXPECT_LE(8u, alignOf<AlignedCharArrayUnion<A8> >());

  EXPECT_LE(1u, sizeof(AlignedCharArrayUnion<A1>));
  EXPECT_LE(2u, sizeof(AlignedCharArrayUnion<A2>));
  EXPECT_LE(4u, sizeof(AlignedCharArrayUnion<A4>));
  EXPECT_LE(8u, sizeof(AlignedCharArrayUnion<A8>));

  EXPECT_EQ(1u, (alignOf<AlignedCharArrayUnion<A1> >()));
  EXPECT_EQ(2u, (alignOf<AlignedCharArrayUnion<A1, A2> >()));
  EXPECT_EQ(4u, (alignOf<AlignedCharArrayUnion<A1, A2, A4> >()));
  EXPECT_EQ(8u, (alignOf<AlignedCharArrayUnion<A1, A2, A4, A8> >()));

  EXPECT_EQ(1u, sizeof(AlignedCharArrayUnion<A1>));
  EXPECT_EQ(2u, sizeof(AlignedCharArrayUnion<A1, A2>));
  EXPECT_EQ(4u, sizeof(AlignedCharArrayUnion<A1, A2, A4>));
  EXPECT_EQ(8u, sizeof(AlignedCharArrayUnion<A1, A2, A4, A8>));

  EXPECT_EQ(1u, (alignOf<AlignedCharArrayUnion<A1[1]> >()));
  EXPECT_EQ(2u, (alignOf<AlignedCharArrayUnion<A1[2], A2[1]> >()));
  EXPECT_EQ(4u, (alignOf<AlignedCharArrayUnion<A1[42], A2[55],
                                               A4[13]> >()));
  EXPECT_EQ(8u, (alignOf<AlignedCharArrayUnion<A1[2], A2[1],
                                               A4, A8> >()));

  EXPECT_EQ(1u,  sizeof(AlignedCharArrayUnion<A1[1]>));
  EXPECT_EQ(2u,  sizeof(AlignedCharArrayUnion<A1[2], A2[1]>));
  EXPECT_EQ(4u,  sizeof(AlignedCharArrayUnion<A1[3], A2[2], A4>));
  EXPECT_EQ(16u, sizeof(AlignedCharArrayUnion<A1, A2[3],
                                              A4[3], A8>));

  // For other tests we simply assert that the alignment of the union mathes
  // that of the fundamental type and hope that we have any weird type
  // productions that would trigger bugs.
  EXPECT_EQ(alignOf<char>(), alignOf<AlignedCharArrayUnion<char> >());
  EXPECT_EQ(alignOf<short>(), alignOf<AlignedCharArrayUnion<short> >());
  EXPECT_EQ(alignOf<int>(), alignOf<AlignedCharArrayUnion<int> >());
  EXPECT_EQ(alignOf<long>(), alignOf<AlignedCharArrayUnion<long> >());
  EXPECT_EQ(alignOf<long long>(),
            alignOf<AlignedCharArrayUnion<long long> >());
  EXPECT_EQ(alignOf<float>(), alignOf<AlignedCharArrayUnion<float> >());
  EXPECT_EQ(alignOf<double>(), alignOf<AlignedCharArrayUnion<double> >());
  EXPECT_EQ(alignOf<long double>(),
            alignOf<AlignedCharArrayUnion<long double> >());
  EXPECT_EQ(alignOf<void *>(), alignOf<AlignedCharArrayUnion<void *> >());
  EXPECT_EQ(alignOf<int *>(), alignOf<AlignedCharArrayUnion<int *> >());
  EXPECT_EQ(alignOf<double (*)(double)>(),
            alignOf<AlignedCharArrayUnion<double (*)(double)> >());
  EXPECT_EQ(alignOf<double (S6::*)()>(),
            alignOf<AlignedCharArrayUnion<double (S6::*)()> >());
  EXPECT_EQ(alignOf<S1>(), alignOf<AlignedCharArrayUnion<S1> >());
  EXPECT_EQ(alignOf<S2>(), alignOf<AlignedCharArrayUnion<S2> >());
  EXPECT_EQ(alignOf<S3>(), alignOf<AlignedCharArrayUnion<S3> >());
  EXPECT_EQ(alignOf<S4>(), alignOf<AlignedCharArrayUnion<S4> >());
  EXPECT_EQ(alignOf<S5>(), alignOf<AlignedCharArrayUnion<S5> >());
  EXPECT_EQ(alignOf<S6>(), alignOf<AlignedCharArrayUnion<S6> >());
  EXPECT_EQ(alignOf<D1>(), alignOf<AlignedCharArrayUnion<D1> >());
  EXPECT_EQ(alignOf<D2>(), alignOf<AlignedCharArrayUnion<D2> >());
  EXPECT_EQ(alignOf<D3>(), alignOf<AlignedCharArrayUnion<D3> >());
  EXPECT_EQ(alignOf<D4>(), alignOf<AlignedCharArrayUnion<D4> >());
  EXPECT_EQ(alignOf<D5>(), alignOf<AlignedCharArrayUnion<D5> >());
  EXPECT_EQ(alignOf<D6>(), alignOf<AlignedCharArrayUnion<D6> >());
  EXPECT_EQ(alignOf<D7>(), alignOf<AlignedCharArrayUnion<D7> >());
  EXPECT_EQ(alignOf<D8>(), alignOf<AlignedCharArrayUnion<D8> >());
  EXPECT_EQ(alignOf<D9>(), alignOf<AlignedCharArrayUnion<D9> >());
  EXPECT_EQ(alignOf<V1>(), alignOf<AlignedCharArrayUnion<V1> >());
  EXPECT_EQ(alignOf<V2>(), alignOf<AlignedCharArrayUnion<V2> >());
  EXPECT_EQ(alignOf<V3>(), alignOf<AlignedCharArrayUnion<V3> >());
  EXPECT_EQ(alignOf<V4>(), alignOf<AlignedCharArrayUnion<V4> >());
  EXPECT_EQ(alignOf<V5>(), alignOf<AlignedCharArrayUnion<V5> >());
  EXPECT_EQ(alignOf<V6>(), alignOf<AlignedCharArrayUnion<V6> >());
  EXPECT_EQ(alignOf<V7>(), alignOf<AlignedCharArrayUnion<V7> >());

  // Some versions of MSVC get this wrong somewhat disturbingly. The failure
  // appears to be benign: alignOf<V8>() produces a preposterous value: 12
#ifndef _MSC_VER
  EXPECT_EQ(alignOf<V8>(), alignOf<AlignedCharArrayUnion<V8> >());
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

  EXPECT_EQ(1u, (alignOf<AlignedCharArray<1, 1> >()));
  EXPECT_EQ(2u, (alignOf<AlignedCharArray<2, 1> >()));
  EXPECT_EQ(4u, (alignOf<AlignedCharArray<4, 1> >()));
  EXPECT_EQ(8u, (alignOf<AlignedCharArray<8, 1> >()));
  EXPECT_EQ(16u, (alignOf<AlignedCharArray<16, 1> >()));

  EXPECT_EQ(1u, sizeof(AlignedCharArray<1, 1>));
  EXPECT_EQ(7u, sizeof(AlignedCharArray<1, 7>));
  EXPECT_EQ(2u, sizeof(AlignedCharArray<2, 2>));
  EXPECT_EQ(16u, sizeof(AlignedCharArray<2, 16>));
}
}
