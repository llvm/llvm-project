; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; CHECK:      .type _Z3funv,@function
; CHECK-NEXT:   .long   3238382334  # 0xc105cafe
; CHECK-NEXT:   .long   .L__llvm_rtti_proxy-_Z3funv
; CHECK-NEXT: _Z3funv:
; CHECK-NEXT:   .cfi_startproc
; CHECK-NEXT:   # %bb.0:
; CHECK-NEXT:   retq

@i = linkonce_odr constant i32 1
@__llvm_rtti_proxy = private unnamed_addr constant ptr @i

define dso_local void @_Z3funv() !func_sanitize !0 {
  ret void
}

!0 = !{i32 3238382334, ptr @__llvm_rtti_proxy}
