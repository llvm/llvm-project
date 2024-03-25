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
; CHECK-LABEL: .globl f1
; CHECK-NEXT:  .p2align 2
; CHECK-NEXT:  .type f1,%function
; CHECK-NEXT:  .long 3170468932
; CHECK-NEXT:  .code 16
; CHECK-NEXT:  .thumb_func
; CHECK-NEXT:  f1:
define void @f1(ptr noundef %x) !kcfi_type !1 {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 -1124498364}
