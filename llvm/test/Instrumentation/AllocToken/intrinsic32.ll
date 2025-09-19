; Test that the alloc-token pass lowers the intrinsic to a constant token ID.
;
; RUN: opt < %s -passes=alloc-token -alloc-token-mode=typehashpointersplit -alloc-token-max=2 -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-pc-linux-gnu"

declare i32 @llvm.alloc.token.id.i32(metadata)

; CHECK-LABEL: @test_intrinsic_lowering
define i32 @test_intrinsic_lowering() {
entry:
  ; CHECK-NOT: call i32 @llvm.alloc.token.id.i32
  ; CHECK: ret i32 0
  %token_no_ptr = call i32 @llvm.alloc.token.id.i32(metadata !0)
  ret i32 %token_no_ptr
}

; CHECK-LABEL: @test_intrinsic_lowering_ptr
define i32 @test_intrinsic_lowering_ptr() {
entry:
  ; CHECK-NOT: call i32 @llvm.alloc.token.id.i32
  ; CHECK: ret i32 1
  %token_with_ptr = call i32 @llvm.alloc.token.id.i32(metadata !1)
  ret i32 %token_with_ptr
}

!0 = !{!"NoPointerType", i1 false}
!1 = !{!"PointerType", i1 true}
