; RUN: llc -mtriple=thumbv7m-macho -o - -show-mc-encoding %s | FileCheck %s
; RUN: llc -mtriple=thumbv6m-macho -o - -show-mc-encoding %s | FileCheck %s --check-prefix=CHECK-V6M
; RUN: llc -mtriple=armv6-macho -o - -show-mc-encoding %s | FileCheck %s --check-prefix=CHECK-ARM
define i32 @test_mov() minsize {
; CHECK-LABEL: test_mov:
; CHECK: movs r0, #255 @ encoding: [0xff,0x20]

  ret i32 255
}

define i32 @test_mov_mvn() minsize {
; CHECK-LABEL: test_mov_mvn:
; CHECK: mvn r0, #203 @ encoding: [0x6f,0xf0,0xcb,0x00]

; CHECK-V6M-LABEL: test_mov_mvn:
; CHECK-V6M: movs [[TMP:r[0-7]]], #203 @ encoding: [0xcb,0x20]
; CHECK-V6M: mvns r0, [[TMP]] @ encoding: [0xc0,0x43]

; CHECK-ARM-LABEL: test_mov_mvn:
; CHECK-ARM: mvn r0, #203 @ encoding: [0xcb,0x00,0xe0,0xe3]
  ret i32 4294967092
}

define i32 @test_mov_lsl() minsize {
; CHECK-LABEL: test_mov_lsl:
; CHECK: mov.w r0, #589824 @ encoding: [0x4f,0xf4,0x10,0x20]

; CHECK-V6M-LABEL: test_mov_lsl:
; CHECK-V6M: movs [[TMP:r[0-7]]], #9 @ encoding: [0x09,0x20]
; CHECK-V6M: lsls r0, [[TMP]], #16 @ encoding: [0x00,0x04]

; CHECK-ARM-LABEL: test_mov_lsl:
; CHECK-ARM: mov r0, #589824 @ encoding: [0x09,0x08,0xa0,0xe3]
  ret i32 589824
}

define i32 @test_movw() minsize {
; CHECK-LABEL: test_movw:
; CHECK: movw r0, #65535

; CHECK-V6M-LABEL: test_movw:
; CHECK-V6M: ldr r0, [[CONSTPOOL:LCPI[0-9]+_[0-9]+]] @ encoding: [A,0x48]
; CHECK-V6M: [[CONSTPOOL]]:
; CHECK-V6M-NEXT: .long 65535

; CHECK-ARM-LABEL: test_movw:
; CHECK-ARM: mov r0, #255 @ encoding: [0xff,0x00,0xa0,0xe3]
; CHECK-ARM: orr r0, r0, #65280 @ encoding: [0xff,0x0c,0x80,0xe3]
 ret i32 65535
}

define i32 @test_regress1() {
; CHECK-ARM-LABEL: test_regress1:
; CHECK-ARM: mov r0, #248 @ encoding: [0xf8,0x00,0xa0,0xe3]
; CHECK-ARM: orr r0, r0, #16252928 @ encoding: [0x3e,0x07,0x80,0xe3]
  ret i32 16253176
}
