//===- llvm/unittest/IR/WaymarkTest.cpp - Waymarking unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Waymarking.h"
#include "gtest/gtest.h"

using namespace llvm;

static const int N = 100;

static int tag(int *P) {
  return static_cast<int>(reinterpret_cast<uintptr_t>(P) &
                          uintptr_t(alignof(int *) - 1));
}

static int *ref(int *P) {
  return reinterpret_cast<int *>(reinterpret_cast<uintptr_t>(P) &
                                 ~uintptr_t(alignof(int *) - 1));
}

static int **createArray(int Len) {
  int **A = new int *[Len];
  for (int I = 0; I < Len; ++I)
    A[I] = new int(I);
  return A;
}

static void freeArray(int **A, int Len) {
  for (int I = 0; I < Len; ++I)
    delete ref(A[I]);
  delete[] A;
}

namespace {

TEST(WaymarkingTest, SingleHead) {
  const int N2 = 2 * N;
  int **A = createArray(N2);

  // Fill the first half of the array with waymarkings.
  fillWaymarks(A, A + N, 0);

  // Verify the values stored in the array are as expected, while we can deduce
  // the array's head from them.
  for (int I = 0; I < N; ++I) {
    int **P = followWaymarks(A + I);
    EXPECT_EQ(A, P);
    EXPECT_EQ(I, *ref(A[I]));
  }

  // Fill the rest of the waymarkings (continuing from where we stopped).
  fillWaymarks(A + N, A + N2, N);

  // Verify the values stored in the array are as expected, while we can deduce
  // the array's head from them.
  for (int I = 0; I < N2; ++I) {
    int **P = followWaymarks(A + I);
    EXPECT_EQ(A, P);
    EXPECT_EQ(I, *ref(A[I]));
  }

  freeArray(A, N2);
}

TEST(WaymarkingTest, MultiHead) {
  const int N2 = 2 * N;
  const int N3 = 3 * N;
  int **A = createArray(N3);

  // Seperate the array into 3 sections, of waymarkings.
  fillWaymarks(A, A + N, 0);
  fillWaymarks(A + N, A + N2, 0);
  fillWaymarks(A + N2, A + N3, 0);

  // Verify the values stored in the array are as expected, while we can deduce
  // the array section's head from them.
  for (int I = 0; I < N; ++I) {
    int **P = followWaymarks(A + I);
    EXPECT_EQ(A, P);
    EXPECT_EQ(I, *ref(A[I]));
  }
  for (int I = N; I < N2; ++I) {
    int **P = followWaymarks(A + I);
    EXPECT_EQ(A + N, P);
    EXPECT_EQ(I, *ref(A[I]));
  }
  for (int I = N2; I < N3; ++I) {
    int **P = followWaymarks(A + I);
    EXPECT_EQ(A + N2, P);
    EXPECT_EQ(I, *ref(A[I]));
  }

  freeArray(A, N3);
}

TEST(WaymarkingTest, Reset) {
  int **A = createArray(N);

  fillWaymarks(A, A + N, 0);

  const int N2 = N / 2;
  const int N3 = N / 3;
  const int N4 = N / 4;

  // Reset specific elements and check that the tag remains the same.
  int T2 = tag(A[N2]);
  delete ref(A[N2]);
  A[N2] = new int(-1);
  fillWaymarks(A + N2, A + N2 + 1, N2);
  EXPECT_EQ(T2, tag(A[N2]));

  int T3 = tag(A[N3]);
  delete ref(A[N3]);
  A[N3] = new int(-1);
  fillWaymarks(A + N3, A + N3 + 1, N3);
  EXPECT_EQ(T3, tag(A[N3]));

  int T4 = tag(A[N4]);
  delete ref(A[N4]);
  A[N4] = new int(-1);
  fillWaymarks(A + N4, A + N4 + 1, N4);
  EXPECT_EQ(T4, tag(A[N4]));

  // Verify the values stored in the array are as expected, while we can deduce
  // the array's head from them.
  for (int I = 0; I < N; ++I) {
    int **P = followWaymarks(A + I);
    EXPECT_EQ(A, P);
    switch (I) {
    case N2:
    case N3:
    case N4:
      EXPECT_EQ(-1, *ref(A[I]));
      break;

    default:
      EXPECT_EQ(I, *ref(A[I]));
      break;
    }
  }

  freeArray(A, N);
}

} // end anonymous namespace
