// RUN: llvm-mc -triple i386-- --show-encoding %s |\
// RUN:   FileCheck %s --check-prefixes=CHECK,ENCODING

// RUN: llvm-mc -triple i386-- -filetype=obj %s |\
// RUN:   llvm-objdump -d - | FileCheck %s

// CHECK-NOT: addr32
// CHECK: maskmovdqu %xmm1, %xmm0
// ENCODING:  encoding: [0x66,0x0f,0xf7,0xc1]
maskmovdqu %xmm1, %xmm0

// CHECK-NOT: addr32
// CHECK:  vmaskmovdqu %xmm1, %xmm0
// ENCODING:  encoding: [0xc5,0xf9,0xf7,0xc1]
vmaskmovdqu %xmm1, %xmm0
