// RUN: llvm-mc -triple x86_64-unknown-unknown -mattr=avx512f -show-encoding %s > %t 2> %t.err
// RUN: FileCheck < %t %s
// RUN: FileCheck --check-prefix=CHECK-STDERR < %t.err %s

// CHECK: vgatherdps %xmm2, (%rdi,%xmm2,2), %xmm2
// CHECK-STDERR: warning: mask, index, and destination registers should be distinct
vgatherdps %xmm2, (%rdi,%xmm2,2), %xmm2

// CHECK: vpgatherdd (%r14,%zmm11,8), %zmm11 {%k1}
// CHECK-STDERR: warning: index and destination registers should be distinct
vpgatherdd (%r14, %zmm11,8), %zmm11 {%k1}

// CHECK: vpgatherqd (%r14,%zmm11,8), %ymm11 {%k1}
// CHECK-STDERR: warning: index and destination registers should be distinct
vpgatherqd (%r14, %zmm11,8), %ymm11 {%k1}

// CHECK: vpgatherdq (%r14,%ymm11,8), %zmm11 {%k1}
// CHECK-STDERR: warning: index and destination registers should be distinct
vpgatherdq (%r14, %ymm11,8), %zmm11 {%k1}
