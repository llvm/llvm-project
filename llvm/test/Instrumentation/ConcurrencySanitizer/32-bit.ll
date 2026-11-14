; RUN: opt < %s -passes='csan' -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-a:0-n16:32"
target triple = "hexagon-unknown-linux-musl"

define void @access(ptr %src, ptr %dst, i32 %n) sanitize_concurrency {
; CHECK-LABEL: define void @access(
; CHECK: call void @__csan_read4(ptr %src, i32 0)
; CHECK: call void @__csan_write4(ptr %dst, i32 0)
; CHECK: call void @__csan_read_range(ptr %src, i32 %n, i32 0)
; CHECK: call void @__csan_write_range(ptr %dst, i32 %n, i32 0)
  %v = load i32, ptr %src, align 4
  store i32 %v, ptr %dst, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %dst, ptr %src, i32 %n, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1)
