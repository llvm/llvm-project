; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; CHECK: _Z3funv:
; CHECK:         .cfi_startproc
; CHECK:         .long   846595819
; CHECK:         .long   .L__llvm_rtti_proxy-_Z3funv
; CHECK: .L__llvm_rtti_proxy:
; CHECK:         .quad   i
; CHECK:         .size   .L__llvm_rtti_proxy, 8

@i = linkonce_odr constant i32 1
@__llvm_rtti_proxy = private unnamed_addr constant i32* @i

define dso_local void @_Z3funv() !func_sanitize !0 {
  ret void
}

!0 = !{i32 846595819, i32** @__llvm_rtti_proxy}
