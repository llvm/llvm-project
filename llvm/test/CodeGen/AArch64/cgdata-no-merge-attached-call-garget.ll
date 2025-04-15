; This test verifies that two similar functions, f1 and f2, are not merged
; when their attached call targets differ, since these targets cannot be parameterized.

; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true < %s | FileCheck %s

; CHECK-NOT: _f1.Tgm
; CHECK-NOT: _f2.Tgm

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin"

define i64 @f1(ptr %0) {
  %2 = call ptr @g1(ptr %0, i32 0) minsize [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  tail call void (...) @llvm.objc.clang.arc.noop.use(ptr %2)
  %3 = call i64 @g2(ptr %2)
  tail call void @objc_release(ptr %2)
  %4 = tail call i64 @g3(i64 %3)
  ret i64 %4
}

define i64 @f2(ptr %0) {
  %2 = call ptr @g1(ptr %0, i32 0) minsize [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  tail call void (...) @llvm.objc.clang.arc.noop.use(ptr %2)
  %3 = call i64 @g2(ptr %2)
  tail call void @objc_release(ptr %2)
  %4 = tail call i64 @g3(i64 %3)
  ret i64 %4
}

declare ptr @g1(ptr, i32)
declare i64 @g2(ptr)
declare i64 @g3(i64)

declare void @llvm.objc.clang.arc.noop.use(...)
declare ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare void @objc_release(ptr)
