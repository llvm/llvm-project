; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

; CHECK-LABEL: .type _Z3funv,@function
; CHECK-NEXT:    .word   3238382334  // 0xc105cafe
; CHECK-NEXT:    .word   .L__llvm_rtti_proxy-_Z3funv
; CHECK-NEXT:  _Z3funv:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ret

; CHECK:       .section .rodata,"a",@progbits
; CHECK-LABEL: .L__llvm_rtti_proxy:
; CHECK-NEXT:    .xword  _ZTIFvvE
; CHECK-NEXT:    .size   .L__llvm_rtti_proxy, 8

@_ZTIFvvE = linkonce_odr constant i32 1
@__llvm_rtti_proxy = private unnamed_addr constant ptr @_ZTIFvvE

define dso_local void @_Z3funv() nounwind !func_sanitize !0 {
  ret void
}

!0 = !{i32 3238382334, ptr @__llvm_rtti_proxy}
