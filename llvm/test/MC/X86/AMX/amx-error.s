// RUN: not llvm-mc -triple x86_64 %s 2>&1 | FileCheck %s

// CHECK: error: all tmm registers must be distinct
tcmmimfp16ps %tmm0, %tmm0, %tmm0

// CHECK: error: all tmm registers must be distinct
tcmmrlfp16ps %tmm1, %tmm0, %tmm1

// CHECK: error: all tmm registers must be distinct
tdpbf16ps %tmm2, %tmm2, %tmm0

// CHECK: error: all tmm registers must be distinct
tdpfp16ps %tmm3, %tmm0, %tmm0

// CHECK: error: all tmm registers must be distinct
tdpbssd %tmm0, %tmm0, %tmm0

// CHECK: error: all tmm registers must be distinct
tdpbsud %tmm1, %tmm0, %tmm1

// CHECK: error: all tmm registers must be distinct
tdpbusd %tmm2, %tmm2, %tmm0

// CHECK: error: all tmm registers must be distinct
tdpbuud %tmm3, %tmm0, %tmm0
