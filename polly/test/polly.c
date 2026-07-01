// Sanity test for Polly in Clang
// RUN: %clang %s -O2 -c -mllvm -polly -mllvm -polly-process-unprofitable -mllvm -print-pipeline-passes -o %t.o | FileCheck %s

// CHECK: ,polly,

void foo(int *A, int *B, int n) {
  for (int i = 0; i < n; ++i)
    A[i] += B[i];
}
