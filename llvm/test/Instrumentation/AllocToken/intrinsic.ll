; Test that the alloc-token pass lowers the intrinsic to a constant token ID.
;
; RUN: opt < %s -passes=alloc-token -alloc-token-mode=3 -alloc-token-max=2 -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i64 @llvm.alloc.token.id(metadata)

; CHECK-LABEL: @test_intrinsic_lowering
define i64 @test_intrinsic_lowering() {
entry:
  ; CHECK-NOT: call i64 @llvm.alloc.token.id
  ; CHECK: ret i64 0
  %token_no_ptr = call i64 @llvm.alloc.token.id(metadata !0)
  ret i64 %token_no_ptr
}

; CHECK-LABEL: @test_intrinsic_lowering_ptr
define i64 @test_intrinsic_lowering_ptr() {
entry:
  ; CHECK-NOT: call i64 @llvm.alloc.token.id
  ; CHECK: ret i64 1
  %token_with_ptr = call i64 @llvm.alloc.token.id(metadata !1)
  ret i64 %token_with_ptr
}

!0 = !{!"NoPointerType", i1 false}
!1 = !{!"PointerType", i1 true}
