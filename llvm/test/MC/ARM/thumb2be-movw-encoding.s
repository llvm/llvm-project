@ RUN: llvm-mc -triple=thumbv7-none-linux-gnueabi -show-encoding < %s | FileCheck %s --check-prefix=CHECK-LE
@ RUN: llvm-mc -triple=thumbebv7-none-linux-gnueabi -show-encoding < %s | FileCheck %s --check-prefix=CHECK-BE

movw r9, :lower16:(_bar)
@ CHECK-LE: movw    r9, :lower16:_bar       @ encoding: [0x40'A',0xf2'A',0b0000AAAA,0x09]
@ CHECK-LE-NEXT:                            @   fixup A - offset: 0, value: _bar, kind: fixup_t2_movw_lo16
@ CHECK-BE: movw    r9, :lower16:_bar       @ encoding: [0xf2,0b0100AAAA,0x09'A',A]
@ CHECK-BE-NEXT:                            @   fixup A - offset: 0, value: _bar, kind: fixup_t2_movw_lo16

