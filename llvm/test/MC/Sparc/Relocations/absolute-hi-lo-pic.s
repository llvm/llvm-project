! RUN: llvm-mc %s -triple=sparcv9 --position-independent -filetype=obj | llvm-objdump -d - | FileCheck %s

! CHECK-LABEL: <abs_hi_lo>:
! CHECK-NEXT:  sethi 0x16a09e, %l5
! CHECK-NEXT:  or %l5, 0x199, %l5
! CHECK-NEXT:  sethi 0x3fb72e, %o0
! CHECK-NEXT:  xor %o0, 0x298, %o0

        .text
        .globl abs_hi_lo
        .type abs_hi_lo,@function
abs = 0xfedcba98
abs_hi_lo:
        sethi %hi(0x5a827999), %l5
        or %l5, %lo(0x5a827999), %l5
        sethi %hi(abs), %o0
        xor %o0, %lo(abs), %o0
