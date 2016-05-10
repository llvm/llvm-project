; RUN: llc -verify-machineinstrs -mtriple=x86_64-apple-macosx -show-mc-encoding -mattr=+avx512f < %s | FileCheck %s


; Make sure we spill the high numbered YMM registers with the right encoding.
; CHECK-LABEL: foo
; CHECK: movups %zmm31, {{.+}}
; CHECK:      encoding: [0x62,0x61,0x7c,0x48,0x11,0xbc,0x24,0x10,0x08,0x00,0x00] 
; zmm30 is used as an anchor for the previous regexp.
; CHECK-NEXT: movups %zmm30
; CHECK: call
; CHECK: iret

define x86_intrcc void @foo(i8* %frame) {
  call void @bar()
  ret void
}

declare void @bar()

