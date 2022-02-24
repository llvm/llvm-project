; RUN: llvm-dis < %S/upgrade-aarch64-ldstxr.bc | FileCheck %s

define void @f(i32* %p) {
; CHECK: call i64 @llvm.aarch64.ldxr.p0i32(i32* elementtype(i32)
  %a = call i64 @llvm.aarch64.ldxr.p0i32(i32* %p)
; CHECK: call i32 @llvm.aarch64.stxr.p0i32(i64 0, i32* elementtype(i32)
  %c = call i32 @llvm.aarch64.stxr.p0i32(i64 0, i32* %p)

; CHECK: call i64 @llvm.aarch64.ldaxr.p0i32(i32* elementtype(i32)
  %a2 = call i64 @llvm.aarch64.ldaxr.p0i32(i32* %p)
; CHECK: call i32 @llvm.aarch64.stlxr.p0i32(i64 0, i32* elementtype(i32)
  %c2 = call i32 @llvm.aarch64.stlxr.p0i32(i64 0, i32* %p)
  ret void
}

declare i64 @llvm.aarch64.ldxr.p0i32(i32*)
declare i64 @llvm.aarch64.ldaxr.p0i32(i32*)
declare i32 @llvm.aarch64.stxr.p0i32(i64, i32*)
declare i32 @llvm.aarch64.stlxr.p0i32(i64, i32*)
