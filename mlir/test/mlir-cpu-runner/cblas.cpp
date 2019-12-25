//===- cblas.cpp - Simple Blas subset implementation ----------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple Blas subset implementation.
//
//===----------------------------------------------------------------------===//

#include "include/cblas.h"
#include <assert.h>

extern "C" float cblas_sdot(const int N, const float *X, const int incX,
                            const float *Y, const int incY) {
  float res = 0.0f;
  for (int i = 0; i < N; ++i)
    res += X[i * incX] * Y[i * incY];
  return res;
}

extern "C" void cblas_sgemm(const enum CBLAS_ORDER Order,
                            const enum CBLAS_TRANSPOSE TransA,
                            const enum CBLAS_TRANSPOSE TransB, const int M,
                            const int N, const int K, const float alpha,
                            const float *A, const int lda, const float *B,
                            const int ldb, const float beta, float *C,
                            const int ldc) {
  assert(Order == CBLAS_ORDER::CblasRowMajor);
  assert(TransA == CBLAS_TRANSPOSE::CblasNoTrans);
  assert(TransB == CBLAS_TRANSPOSE::CblasNoTrans);
  for (int m = 0; m < M; ++m) {
    auto *pA = A + m * lda;
    auto *pC = C + m * ldc;
    for (int n = 0; n < N; ++n) {
      float c = pC[n];
      float res = 0.0f;
      for (int k = 0; k < K; ++k) {
        auto *pB = B + k * ldb;
        res += pA[k] * pB[n];
      }
      pC[n] = alpha * c + beta * res;
    }
  }
}
