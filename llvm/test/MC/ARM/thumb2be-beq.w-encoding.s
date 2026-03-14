@ RUN: llvm-mc -triple=thumbv7-none-linux-gnueabi -show-encoding < %s | FileCheck %s --check-prefix=CHECK-LE
@ RUN: llvm-mc -triple=thumbebv7-none-linux-gnueabi -show-encoding < %s | FileCheck %s --check-prefix=CHECK-BE

beq.w   bar
@ CHECK-LE: beq.w	bar                     @ encoding: [A,0xf0'A',A,0x80'A']
@ CHECK-LE-NEXT:                                @   fixup A - offset: 0, value: bar, kind: fixup_t2_condbranch
@ CHECK-BE: beq.w	bar                     @ encoding: [0xf0'A',A,0x80'A',A]
@ CHECK-BE-NEXT:                                @   fixup A - offset: 0, value: bar, kind: fixup_t2_condbranch

