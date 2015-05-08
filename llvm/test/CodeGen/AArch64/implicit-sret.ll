; RUN: llc %s -o - -mtriple=arm64-apple-ios7.0 | FileCheck %s
;
; Handle implicit sret arguments that are generated on-the-fly during lowering.
; <rdar://19792160> Null pointer assertion in AArch64TargetLowering

; CHECK-LABEL: big_retval
; ... str or stp for the first 1024 bits
; CHECK: strb wzr, [x8, #128]
; CHECK: ret
define i1032 @big_retval() {
entry:
  ret i1032 0
}
