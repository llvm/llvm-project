; RUN: llc -verify-machineinstrs -mtriple=x86_64-apple-macosx -show-mc-encoding -mattr=+avx512f < %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK64
; RUN: llc -verify-machineinstrs -mtriple=i386-apple-macosx -show-mc-encoding -mattr=+avx512f < %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK32

; Make sure we spill the high numbered zmm registers and K registers with the right encoding.
; CHECK-LABEL: foo
; CHECK: kmovq %k7, {{.+}}
; CHECK64:      encoding: [0xc4,0xe1,0xf8,0x91,0xbc,0x24,0x68,0x08,0x00,0x00]
; CHECK32:      encoding: [0xc4,0xe1,0xf8,0x91,0xbc,0x24,0x68,0x02,0x00,0x00]
; k6 is used as an anchor for the previous regexp.
; CHECK-NEXT: kmovq %k6

; CHECK64: movups %zmm31, {{.+}}
; CHECK64:      encoding: [0x62,0x61,0x7c,0x48,0x11,0xbc,0x24,0xe0,0x07,0x00,0x00] 
; zmm30 is used as an anchor for the previous regexp.
; CHECK64-NEXT: movups %zmm30

; CHECK32-NOT: zmm31
; CHECK32-NOT: zmm8
; CHECK32: movups %zmm7, {{.+}}
; CHECK32:      encoding: [0x62,0xf1,0x7c,0x48,0x11,0xbc,0x24,0xe0,0x01,0x00,0x00] 
; zmm6 is used as an anchor for the previous regexp.
; CHECK32-NEXT: movups %zmm6

; CHECK: call
; CHECK: iret

define x86_intrcc void @foo(i8* %frame) {
  call void @bar()
  ret void
}

declare void @bar()

