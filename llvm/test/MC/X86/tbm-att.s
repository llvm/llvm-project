// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// bextri 32 reg
// CHECK: bextrl   $2814, %edi, %eax
// CHECK: encoding: [0x8f,0xea,0x78,0x10,0xc7,0xfe,0x0a,0x00,0x00]
          bextr   $2814, %edi, %eax

// bextri 32 mem
// CHECK: bextrl   $2814, (%rdi), %eax
// CHECK: encoding: [0x8f,0xea,0x78,0x10,0x07,0xfe,0x0a,0x00,0x00]
          bextr   $2814, (%rdi), %eax

// bextri 64 reg
// CHECK: bextrq   $2814, %rdi, %rax
// CHECK: encoding: [0x8f,0xea,0xf8,0x10,0xc7,0xfe,0x0a,0x00,0x00]
          bextr   $2814, %rdi, %rax

// bextri 64 mem
// CHECK: bextrq   $2814, (%rdi), %rax
// CHECK: encoding: [0x8f,0xea,0xf8,0x10,0x07,0xfe,0x0a,0x00,0x00]
          bextr   $2814, (%rdi), %rax

// blcfill 32 reg
// CHECK: blcfilll %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xcf]
          blcfill %edi, %eax

// blcfill 32 mem
// CHECK: blcfilll (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x0f]
          blcfill (%rdi), %eax

// blcfill 64 reg
// CHECK: blcfillq %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xcf]
          blcfill %rdi, %rax

// blcfill 64 mem
// CHECK: blcfillq (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0x0f]
          blcfill (%rdi), %rax

// blci   32 reg
// CHECK: blcil    %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x02,0xf7]
          blci    %edi, %eax

// blci   32 mem
// CHECK: blcil    (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x02,0x37]
          blci    (%rdi), %eax

// blci   64 reg
// CHECK: blciq    %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x02,0xf7]
          blci    %rdi, %rax

// blci   64 mem
// CHECK: blciq    (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x02,0x37]
          blci    (%rdi), %rax

// blcic  32 reg
// CHECK: blcicl   %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xef]
          blcic   %edi, %eax

// blcic  32 mem
// CHECK: blcicl   (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x2f]
          blcic   (%rdi), %eax

// blcic  64 reg
// CHECK: blcicq   %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xef]
          blcic   %rdi, %rax

// blcic  64 mem
// CHECK: blcicq   (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0x2f]
          blcic   (%rdi), %rax

// blcmsk 32 reg
// CHECK: blcmskl  %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x02,0xcf]
          blcmsk  %edi, %eax

// blcmsk 32 mem
// CHECK: blcmskl  (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x02,0x0f]
          blcmsk  (%rdi), %eax

// blcmsk 64 reg
// CHECK: blcmskq  %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x02,0xcf]
          blcmsk  %rdi, %rax

// blcmsk 64 mem
// CHECK: blcmskq  (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x02,0x0f]
          blcmsk  (%rdi), %rax

// blcs   32 reg
// CHECK: blcsl    %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xdf]
          blcs    %edi, %eax

// blcs   32 mem
// CHECK: blcsl    (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x1f]
          blcs    (%rdi), %eax

// blcs   64 reg
// CHECK: blcsq    %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xdf]
          blcs    %rdi, %rax

// blcs   64 mem
// CHECK: blcsq    (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0x1f]
          blcs    (%rdi), %rax

// blsfill 32 reg
// CHECK: blsfilll %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xd7]
          blsfill %edi, %eax

// blsfill 32 mem
// CHECK: blsfilll (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x17]
          blsfill (%rdi), %eax

// blsfill 64 reg
// CHECK: blsfillq %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xd7]
          blsfill %rdi, %rax

// blsfill 64 mem
// CHECK: blsfillq (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0x17]
          blsfill (%rdi), %rax

// blsic  32 reg
// CHECK: blsicl   %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xf7]
          blsic   %edi, %eax

// blsic  32 mem
// CHECK: blsicl   (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x37]
          blsic   (%rdi), %eax

// blsic  64 reg
// CHECK: blsicq   %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xf7]
          blsic   %rdi, %rax

// t1mskc 32 reg
// CHECK: t1mskcl  %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xff]
          t1mskc  %edi, %eax

// t1mskc 32 mem
// CHECK: t1mskcl  (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x3f]
          t1mskc  (%rdi), %eax

// t1mskc 64 reg
// CHECK: t1mskcq  %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xff]
          t1mskc  %rdi, %rax

// t1mskc 64 mem
// CHECK: t1mskcq  (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0x3f]
          t1mskc  (%rdi), %rax

// tzmsk  32 reg
// CHECK: tzmskl   %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xe7]
          tzmsk   %edi, %eax

// tzmsk  32 mem
// CHECK: tzmskl   (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x27]
          tzmsk   (%rdi), %eax

// tzmsk  64 reg
// CHECK: tzmskq   %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xe7]
          tzmsk   %rdi, %rax

// tzmsk  64 mem
// CHECK: tzmskq   (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0x27]
          tzmsk   (%rdi), %rax

// CHECK: encoding: [0x67,0xc4,0xe2,0x60,0xf7,0x07]
          bextr   %ebx, (%edi), %eax

// CHECK: encoding: [0x67,0x8f,0xea,0x78,0x10,0x07,A,A,A,A]
          bextr   $foo, (%edi), %eax
