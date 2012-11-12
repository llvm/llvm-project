; RUN: opt < %s -simplifycfg -S | FileCheck %s

; Test that a lifetime intrinsic doesn't prevent us from simplifying this.

; CHECK: foo
; CHECK: entry:
; CHECK-NOT: bb0:
; CHECK-NOT: bb1:
; CHECK: ret
define void @foo(i1 %x) {
entry:
  %a = alloca i8
  call void @llvm.lifetime.start(i64 -1, i8* %a) nounwind
  br i1 %x, label %bb0, label %bb1

bb0:
  call void @llvm.lifetime.end(i64 -1, i8* %a) nounwind
  br label %bb1

bb1:
  call void @f()
  ret void
}

declare void @f()

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind

declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind
