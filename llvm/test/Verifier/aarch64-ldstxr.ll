; RUN: not opt -passes=verify -S < %s 2>&1 | FileCheck %s

define void @f(ptr %p) {
; CHECK: Intrinsic requires elementtype attribute on first argument
  %a = call i64 @llvm.aarch64.ldxr.p0(ptr %p)
; CHECK: Intrinsic requires elementtype attribute on second argument
  %c = call i32 @llvm.aarch64.stxr.p0(i64 0, ptr %p)

; CHECK: Intrinsic requires elementtype attribute on first argument
  %a2 = call i64 @llvm.aarch64.ldaxr.p0(ptr %p)
; CHECK: Intrinsic requires elementtype attribute on second argument
  %c2 = call i32 @llvm.aarch64.stlxr.p0(i64 0, ptr %p)
  ret void
}

declare i64 @llvm.aarch64.ldxr.p0(ptr)
declare i64 @llvm.aarch64.ldaxr.p0(ptr)
declare i32 @llvm.aarch64.stxr.p0(i64, ptr)
declare i32 @llvm.aarch64.stlxr.p0(i64, ptr)
