; RUN: opt -S -passes=lower-conditional-store < %s | FileCheck %s
define void @foo(ptr %p, i64 %val, i1 %cond) {
  call void @llvm.conditional.store.i64.p0(i64 %val, ptr %p, i32 4, i1 %cond)
  call void @llvm.conditional.store.i64.p0(i64 %val, ptr %p, i32 4, i1 %cond)
  ret void
} 

declare void @llvm.conditional.store.i64.p0(i64, ptr nocapture, i32 immarg, i1)

; CHECK: define void @foo(ptr %p, i64 %val, i1 %cond) {
; CHECK:   br i1 %cond, label %1, label %2
; CHECK: 1: 
; CHECK:   store i64 %val, ptr %p, align 4
; CHECK:   br label %2
; CHECK: 2:
; CHECK:   br i1 %cond, label %3, label %4
; CHECK: 3:
; CHECK:   store i64 %val, ptr %p, align 4
; CHECK:   br label %4
; CHECK: 4:
; CHECK:   ret void
; CHECK: }