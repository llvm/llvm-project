// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// bextri 32 reg
// CHECK: bextr   $2814, %edi, %eax
// CHECK: encoding: [0x8f,0xea,0x78,0x10,0xc7,0xfe,0x0a,0x00,0x00]
          bextr   $2814, %edi, %eax

// bextri 32 mem
// CHECK: bextr   $2814, (%rdi), %eax
// CHECK: encoding: [0x8f,0xea,0x78,0x10,0x07,0xfe,0x0a,0x00,0x00]
          bextr   $2814, (%rdi), %eax

// bextri 64 reg
// CHECK: bextr   $2814, %rdi, %rax
// CHECK: encoding: [0x8f,0xea,0xf8,0x10,0xc7,0xfe,0x0a,0x00,0x00]
          bextr   $2814, %rdi, %rax

// bextri 64 mem
// CHECK: bextr   $2814, (%rdi), %rax
// CHECK: encoding: [0x8f,0xea,0xf8,0x10,0x07,0xfe,0x0a,0x00,0x00]
          bextr   $2814, (%rdi), %rax

// blcfill 32 reg
// CHECK: blcfill %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xcf]
          blcfill %edi, %eax

// blcfill 32 mem
// CHECK: blcfill (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x0f]
          blcfill (%rdi), %eax

// blcfill 64 reg
// CHECK: blcfill %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xcf]
          blcfill %rdi, %rax

// blcfill 64 mem
// CHECK: blcfill (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0x0f]
          blcfill (%rdi), %rax

// blci   32 reg
// CHECK: blci    %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x02,0xf7]
          blci    %edi, %eax

// blci   32 mem
// CHECK: blci    (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x02,0x37]
          blci    (%rdi), %eax

// blci   64 reg
// CHECK: blci    %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x02,0xf7]
          blci    %rdi, %rax

// blci   64 mem
// CHECK: blci    (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x02,0x37]
          blci    (%rdi), %rax

// blcic  32 reg
// CHECK: blcic   %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xef]
          blcic   %edi, %eax

// blcic  32 mem
// CHECK: blcic   (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x2f]
          blcic   (%rdi), %eax

// blcic  64 reg
// CHECK: blcic   %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xef]
          blcic   %rdi, %rax

// blcic  64 mem
// CHECK: blcic   (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0x2f]
          blcic   (%rdi), %rax

// blcmsk 32 reg
// CHECK: blcmsk  %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x02,0xcf]
          blcmsk  %edi, %eax

// blcmsk 32 mem
// CHECK: blcmsk  (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x02,0x0f]
          blcmsk  (%rdi), %eax

// blcmsk 64 reg
// CHECK: blcmsk  %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x02,0xcf]
          blcmsk  %rdi, %rax

// blcmsk 64 mem
// CHECK: blcmsk  (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x02,0x0f]
          blcmsk  (%rdi), %rax

// blcs   32 reg
// CHECK: blcs    %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xdf]
          blcs    %edi, %eax

// blcs   32 mem
// CHECK: blcs    (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x1f]
          blcs    (%rdi), %eax

// blcs   64 reg
// CHECK: blcs    %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xdf]
          blcs    %rdi, %rax

// blcs   64 mem
// CHECK: blcs    (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0x1f]
          blcs    (%rdi), %rax

// blsfill 32 reg
// CHECK: blsfill %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xd7]
          blsfill %edi, %eax

// blsfill 32 mem
// CHECK: blsfill (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x17]
          blsfill (%rdi), %eax

// blsfill 64 reg
// CHECK: blsfill %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xd7]
          blsfill %rdi, %rax

// blsfill 64 mem
// CHECK: blsfill (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0x17]
          blsfill (%rdi), %rax

// blsic  32 reg
// CHECK: blsic   %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xf7]
          blsic   %edi, %eax

// blsic  32 mem
// CHECK: blsic   (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x37]
          blsic   (%rdi), %eax

// blsic  64 reg
// CHECK: blsic   %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xf7]
          blsic   %rdi, %rax

// t1mskc 32 reg
// CHECK: t1mskc  %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xff]
          t1mskc  %edi, %eax

// t1mskc 32 mem
// CHECK: t1mskc  (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x3f]
          t1mskc  (%rdi), %eax

// t1mskc 64 reg
// CHECK: t1mskc  %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xff]
          t1mskc  %rdi, %rax

// t1mskc 64 mem
// CHECK: t1mskc  (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0x3f]
          t1mskc  (%rdi), %rax

// tzmsk  32 reg
// CHECK: tzmsk   %edi, %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0xe7]
          tzmsk   %edi, %eax

// tzmsk  32 mem
// CHECK: tzmsk   (%rdi), %eax
// CHECK: encoding: [0x8f,0xe9,0x78,0x01,0x27]
          tzmsk   (%rdi), %eax

// tzmsk  64 reg
// CHECK: tzmsk   %rdi, %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0xe7]
          tzmsk   %rdi, %rax

// tzmsk  64 mem
// CHECK: tzmsk   (%rdi), %rax
// CHECK: encoding: [0x8f,0xe9,0xf8,0x01,0x27]
          tzmsk   (%rdi), %rax

// CHECK: encoding: [0x67,0xc4,0xe2,0x60,0xf7,0x07]
          bextr   %ebx, (%edi), %eax

// CHECK: encoding: [0x67,0x8f,0xea,0x78,0x10,0x07,A,A,A,A]
          bextr   $foo, (%edi), %eax
