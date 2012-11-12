; RUN: opt -codegenprepare %s -S -o - | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; CHECK: @test1
; objectsize should fold to a constant, which causes the branch to fold to an
; uncond branch. Next, we fold the control flow alltogether.
; rdar://8785296
define i32 @test1(i8* %ptr) nounwind ssp noredzone align 2 {
entry:
  %0 = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false)
  %1 = icmp ugt i64 %0, 3
  br i1 %1, label %T, label %trap

; CHECK: T:
; CHECK-NOT: br label %

trap:                                             ; preds = %0, %entry
  tail call void @llvm.trap() noreturn nounwind
  unreachable

T:
; CHECK: ret i32 4
  ret i32 4
}

declare i64 @llvm.objectsize.i64(i8*, i1) nounwind readonly

declare void @llvm.trap() nounwind
