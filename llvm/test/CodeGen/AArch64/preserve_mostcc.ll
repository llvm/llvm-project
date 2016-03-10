; RUN: llc < %s -mtriple=arm64-apple-ios-8.0.0 | FileCheck %s

declare void @standard_cc_func()
declare preserve_mostcc void @preserve_mostcc_func()

; Registers r9-r15 should be saved before the call of a function
; with a standard calling convention.
define preserve_mostcc void @preserve_mostcc1() nounwind {
entry:
;CHECK-LABEL: preserve_mostcc1
;CHECK-NOT:   stp
;CHECK-NOT:   str
;CHECK:       str     x15
;CHECK-NEXT:  stp     x14, x13,
;CHECK-NEXT:  stp     x12, x11,
;CHECK-NEXT:  stp     x10, x9,
;CHECK:       bl      _standard_cc_func
  call void @standard_cc_func()
;CHECK:       ldp     x10, x9,
;CHECK-NEXT:  ldp     x12, x11,
;CHECK-NEXT:  ldp     x14, x13,
;CHECK-NEXT:  ldr     x15
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
;CHECK:     stp     x29, x30,
;CHECK-NOT: x14
;CHECK:     bl      _preserve_mostcc_func
  call preserve_mostcc void @preserve_mostcc_func()
  ret void
}

