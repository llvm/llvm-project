@ RUN: llvm-mc -triple=thumbv7-none-linux-gnueabi -show-encoding < %s | FileCheck %s --check-prefix=CHECK-LE
@ RUN: llvm-mc -triple=thumbebv7-none-linux-gnueabi -show-encoding < %s | FileCheck %s --check-prefix=CHECK-BE

movt r9, :upper16:(_bar)
@ CHECK-LE: movt    r9, :upper16:_bar       @ encoding: [0xc0'A',0xf2'A',0b0000AAAA,0x09]
@ CHECK-LE-NEXT:                            @   fixup A - offset: 0, value: _bar, kind: fixup_t2_movt_hi16
@ CHECK-BE: movt    r9, :upper16:_bar       @ encoding: [0xf2,0b1100AAAA,0x09'A',A]
@ CHECK-BE-NEXT:                            @   fixup A - offset: 0, value: _bar, kind: fixup_t2_movt_hi16

