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

// Get the Waymarking Tag of the pointer.
static int tag(int *P) {
  return static_cast<int>(reinterpret_cast<uintptr_t>(P) &
                          uintptr_t(alignof(int *) - 1));
}

// Get the actual pointer, by stripping the Waymarking Tag.
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

// Verify the values stored in the array are as expected, and did not change due
// to fillWaymarks.
static void verifyArrayValues(int **A, int Begin, int End) {
  for (int I = Begin; I < End; ++I)
    EXPECT_EQ(I, *ref(A[I]));
}

static void verifyArrayValues(int **A, int Len) {
  verifyArrayValues(A, 0, Len);
}

// Verify that we can follow the waymarks to the array's head from each element
// of the array.
static void verifyFollowWaymarks(int **A, int Len) {
  for (int I = 0; I < Len; ++I) {
    int **P = followWaymarks(A + I);
    EXPECT_EQ(A, P);
  }
}

namespace {

// Test filling and following the waymarks of a single array.
TEST(WaymarkingTest, SingleHead) {
  const int N2 = 2 * N;
  int **volatile A = createArray(N2);

  // Fill the first half of the array with waymarks.
  fillWaymarks(A, A + N, 0);
  verifyArrayValues(A, N2);

  verifyFollowWaymarks(A, N);

  // Fill the rest of the waymarks (continuing from where we stopped).
  fillWaymarks(A + N, A + N2, N);
  verifyArrayValues(A, N2);

  verifyFollowWaymarks(A, N);

  freeArray(A, N2);
}

// Test filling and following the waymarks of an array split into several
// different sections of waymarks (treated just like separate arrays).
TEST(WaymarkingTest, MultiHead) {
  const int N2 = 2 * N;
  const int N3 = 3 * N;
  int **volatile A = createArray(N3);

  // Separate the array into 3 sections of waymarks.
  fillWaymarks(A, A + N, 0);
  fillWaymarks(A + N, A + N2, 0);
  fillWaymarks(A + N2, A + N3, 0);
  verifyArrayValues(A, N3);

  verifyFollowWaymarks(A, N);
  verifyFollowWaymarks(A + N, N2 - N);
  verifyFollowWaymarks(A + N2, N3 - N2);

  freeArray(A, N3);
}

// Test reseting (value and tag of) elements inside an array of waymarks.
TEST(WaymarkingTest, Reset) {
  int **volatile A = createArray(N);

  fillWaymarks(A, A + N, 0);
  verifyArrayValues(A, N);

  const int N2 = N / 2;
  const int N3 = N / 3;
  const int N4 = N / 4;

  // Reset specific elements and check that the tag remains the same.
  int T2 = tag(A[N2]);
  delete ref(A[N2]);
  A[N2] = new int(N2);
  fillWaymarks(A + N2, A + N2 + 1, N2);
  verifyArrayValues(A, N2, N2 + 1);
  EXPECT_EQ(T2, tag(A[N2]));

  int T3 = tag(A[N3]);
  delete ref(A[N3]);
  A[N3] = new int(N3);
  fillWaymarks(A + N3, A + N3 + 1, N3);
  verifyArrayValues(A, N3, N3 + 1);
  EXPECT_EQ(T3, tag(A[N3]));

  int T4 = tag(A[N4]);
  delete ref(A[N4]);
  A[N4] = new int(N4);
  fillWaymarks(A + N4, A + N4 + 1, N4);
  verifyArrayValues(A, N4, N4 + 1);
  EXPECT_EQ(T4, tag(A[N4]));

  verifyArrayValues(A, N);
  verifyFollowWaymarks(A, N);

  freeArray(A, N);
}

} // end anonymous namespace
