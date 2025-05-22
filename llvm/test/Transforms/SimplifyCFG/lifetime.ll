; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

; Test that a lifetime intrinsic isn't removed because that would change semantics

; CHECK: foo
; CHECK: entry:
; CHECK: bb0:
; CHECK: bb1:
; CHECK: ret
define void @foo(i1 %x) {
entry:
  %a = alloca i8
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a) nounwind
  br i1 %x, label %bb0, label %bb1

bb0:
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a) nounwind
  br label %bb1

bb1:
  call void @f()
  ret void
}

declare void @f()

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) nounwind

declare void @llvm.lifetime.end.p0(i64, ptr nocapture) nounwind
