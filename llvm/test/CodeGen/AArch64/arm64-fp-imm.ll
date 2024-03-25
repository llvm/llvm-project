; RUN: llc < %s -mtriple=arm64-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=arm64-apple-darwin -global-isel | FileCheck %s

; CHECK: literal8
; CHECK: .quad 0x400921fb54442d18
define double @foo() {
; CHECK: _foo:
; CHECK: adrp x[[REG:[0-9]+]], lCPI0_0@PAGE
; CHECK: ldr  d0, [x[[REG]], lCPI0_0@PAGEOFF]
; CHECK-NEXT: ret
  ret double 0x400921FB54442D18
}

define float @bar() {
; CHECK: _bar:
; CHECK:  mov  [[REG:w[0-9]+]], #4059
; CHECK:  movk [[REG]], #16457, lsl #16
; CHECK:  fmov s0, [[REG]]
; CHECK-NEXT:  ret
  ret float 0x400921FB60000000
}

; CHECK: literal16
; CHECK: .quad 0
; CHECK: .quad 0
define fp128 @baz() {
; CHECK: _baz:
; CHECK:  adrp x[[REG:[0-9]+]], lCPI2_0@PAGE
; CHECK:  ldr  q0, [x[[REG]], lCPI2_0@PAGEOFF]
; CHECK-NEXT:  ret
  ret fp128 0xL00000000000000000000000000000000
}
