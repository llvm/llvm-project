; RUN: llc < %s -mtriple=arm64-apple-ios-8.0.0 | FileCheck -check-prefix CHECK -check-prefix CHECK-DARWIN %s
; RUN: llc < %s -mtriple=aarch64-unknown-windows-msvc | FileCheck -check-prefix CHECK -check-prefix CHECK-WIN %s

declare void @standard_cc_func()
declare preserve_mostcc void @preserve_mostcc_func()

; Registers r9-r15 should be saved before the call of a function
; with a standard calling convention.
define preserve_mostcc void @preserve_mostcc1() nounwind {
entry:
;CHECK-LABEL: preserve_mostcc1
;CHECK-DARWIN-NOT:   stp
;CHECK-DARWIN-NOT:   str
;CHECK-DARWIN:       str     x15
;CHECK-DARWIN-NEXT:  stp     x14, x13,
;CHECK-DARWIN-NEXT:  stp     x12, x11,
;CHECK-DARWIN-NEXT:  stp     x10, x9,
;CHECK-WIN:       stp     x15, x14
;CHECK-WIN-NEXT:  stp     x13, x12,
;CHECK-WIN-NEXT:  stp     x11, x10,
;CHECK-WIN-NEXT:  stp     x9, x30
;CHECK:       bl      {{_?}}standard_cc_func
  call void @standard_cc_func()
;CHECK-DARWIN:       ldp     x10, x9,
;CHECK-DARWIN-NEXT:  ldp     x12, x11,
;CHECK-DARWIN-NEXT:  ldp     x14, x13,
;CHECK-DARWIN-NEXT:  ldr     x15
;CHECK-WIN:       ldp     x9, x30
;CHECK-WIN-NEXT:  ldp     x11, x10,
;CHECK-WIN-NEXT:  ldp     x13, x12,
;CHECK-WIN-NEXT:  ldp     x15, x14,
  ret void
}

; Registers r9-r15 don't need to be saved if one
; function with preserve_mostcc calling convention calls another
; function with preserve_mostcc calling convention, because the
; callee wil save these registers anyways.
define preserve_mostcc void @preserve_mostcc2() nounwind {
entry:
;CHECK-LABEL: preserve_mostcc2
;CHECK-NOT: x14
;CHECK-DARWIN:     stp     x29, x30,
;CHECK-WIN:     str     x30
;CHECK-NOT: x14
;CHECK:     bl      {{_?}}preserve_mostcc_func
  call preserve_mostcc void @preserve_mostcc_func()
  ret void
}

