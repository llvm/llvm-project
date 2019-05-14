; RUN: llc < %s -mtriple=arm64-apple-darwin | FileCheck %s

; CHECK: literal8
; CHECK: .quad  4614256656552045848
define double @foo() optsize {
; CHECK: _foo:
; CHECK: adrp x[[REG:[0-9]+]], lCPI0_0@PAGE
; CHECK: ldr  d0, [x[[REG]], lCPI0_0@PAGEOFF]
; CHECK-NEXT: ret
  ret double 0x400921FB54442D18
}

; CHECK: literal8
; CHECK: .quad  137438953409
define double @foo2() optsize {
; CHECK: _foo2:
; CHECK: adrp x[[REG:[0-9]+]], lCPI1_0@PAGE
; CHECK: ldr  d0, [x[[REG]], lCPI1_0@PAGEOFF]
; CHECK-NEXT: ret
  ret double 0x1FFFFFFFC1
}

define float @bar() optsize {
; CHECK: _bar:
; CHECK: adrp x[[REG:[0-9]+]], lCPI2_0@PAGE
; CHECK: ldr  s0, [x[[REG]], lCPI2_0@PAGEOFF]
; CHECK-NEXT:  ret
  ret float 0x400921FB60000000
}

; CHECK: literal16
; CHECK: .quad 0
; CHECK: .quad 0
define fp128 @baz() optsize {
; CHECK: _baz:
; CHECK:  adrp x[[REG:[0-9]+]], lCPI3_0@PAGE
; CHECK:  ldr  q0, [x[[REG]], lCPI3_0@PAGEOFF]
; CHECK-NEXT:  ret
  ret fp128 0xL00000000000000000000000000000000
}
