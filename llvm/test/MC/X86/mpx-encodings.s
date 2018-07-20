// RUN: llvm-mc -triple x86_64-- --show-encoding %s |\
// RUN:   FileCheck %s --check-prefixes=CHECK,ENCODING

// RUN: llvm-mc -triple x86_64-- -filetype=obj %s |\
// RUN:   llvm-objdump -d - | FileCheck %s

// CHECK: bndmk (%rax), %bnd0
// ENCODING:  encoding: [0xf3,0x0f,0x1b,0x00]
bndmk (%rax), %bnd0

// CHECK: bndmk 1024(%rax), %bnd1
// ENCODING:  encoding: [0xf3,0x0f,0x1b,0x88,0x00,0x04,0x00,0x00]
bndmk 1024(%rax), %bnd1

// CHECK: bndmov  %bnd2, %bnd1
// ENCODING:  encoding: [0x66,0x0f,0x1a,0xca]
bndmov %bnd2, %bnd1

// CHECK: bndmov %bnd1, 1024(%r9)
// ENCODING:  encoding: [0x66,0x41,0x0f,0x1b,0x89,0x00,0x04,0x00,0x00]
bndmov %bnd1, 1024(%r9)

// CHECK: bndstx %bnd1, 1024(%rax)
// ENCODING:  encoding: [0x0f,0x1b,0x88,0x00,0x04,0x00,0x00]
bndstx %bnd1, 1024(%rax)

// CHECK: bndldx 1024(%r8), %bnd1
// ENCODING:  encoding: [0x41,0x0f,0x1a,0x88,0x00,0x04,0x00,0x00]
bndldx 1024(%r8), %bnd1

// CHECK: bndcl 121(%r10), %bnd1
// ENCODING:  encoding: [0xf3,0x41,0x0f,0x1a,0x4a,0x79]
bndcl 121(%r10), %bnd1

// CHECK: bndcn 121(%rcx), %bnd3
// ENCODING:  encoding: [0xf2,0x0f,0x1b,0x59,0x79]
bndcn 121(%rcx), %bnd3

// CHECK: bndcu %rdx, %bnd3
// ENCODING:  encoding: [0xf2,0x0f,0x1a,0xda]
bndcu %rdx, %bnd3
