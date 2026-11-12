; RUN: llc -mtriple=thumbv6m-none-eabi < %s | FileCheck %s

; CHECK-LABEL: .globl nosan
; CHECK-NEXT:  .p2align 1
; CHECK-NEXT:  .type nosan,%function
; CHECK-NEXT:  .code 16
; CHECK-NEXT:  .thumb_func
; CHECK-NEXT:  nosan:
define dso_local void @nosan() nounwind {
  ret void
}

;; The alignment is at least 4 to avoid unaligned type hash loads when this
;; instrumented function is indirectly called.
; CHECK-LABEL: .globl foo
; CHECK-NEXT:  .p2align 2
; CHECK-NEXT:  .type foo,%function
; CHECK-NEXT:  .long 3238382334
; CHECK-NEXT:  .long 3170468932
; CHECK-NEXT:  .code 16
; CHECK-NEXT:  .thumb_func
; CHECK-NEXT:  foo:
define dso_local void @foo() nounwind !func_sanitize !0 {
  ret void
}

;; If "align" is smaller than 4 (required alignment from !func_sanitize), use 4.
; CHECK-LABEL: .globl align2
; CHECK-NEXT:  .p2align 2
define dso_local void @align2() nounwind align 2 !func_sanitize !0 {
  ret void
}

;; If "align" is larger than 4, use its value.
; CHECK-LABEL: .globl align8
; CHECK-NEXT:  .p2align 3
define dso_local void @align8() nounwind align 8 !func_sanitize !0 {
  ret void
}

!0 = !{i32 -1056584962, i32 -1124498364}
