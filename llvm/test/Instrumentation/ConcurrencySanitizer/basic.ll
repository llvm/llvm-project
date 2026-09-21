; RUN: opt < %s -passes='module(csan-module),function(csan)' -S -mtriple=x86_64-unknown-linux-gnu | FileCheck %s
; RUN: opt < %s -passes='module(csan-module),function(csan)' -S -mtriple=amdgcn-amd-amdhsa | FileCheck %s

define void @read_write(ptr %a, ptr %b) sanitize_concurrency {
entry:
  %v = load i32, ptr %a, align 4
  store i32 %v, ptr %b, align 4
  ret void
}
; CHECK-LABEL: @read_write(
; CHECK: call void @__csan_func_entry
; CHECK: call void @__csan_read4(ptr %a, i32 0)
; CHECK-NEXT: %v = load i32, ptr %a, align 4
; CHECK: call void @__csan_write4(ptr %b, i32 0)
; CHECK-NEXT: store i32 %v, ptr %b, align 4
; CHECK: call void @__csan_func_exit

define i16 @unaligned(ptr %a) sanitize_concurrency {
entry:
  %v = load i16, ptr %a, align 1
  ret i16 %v
}
; CHECK-LABEL: @unaligned(
; CHECK: call void @__csan_unaligned_read2(ptr %a, i32 0)
; CHECK-NEXT: %v = load i16, ptr %a, align 1

define void @memintrinsics(ptr %dst, ptr %src, i64 %n) sanitize_concurrency {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
  call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
  call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 %n, i1 false)
  ret void
}
; CHECK-LABEL: @memintrinsics(
; CHECK: call void @__csan_read_range(ptr %src, i64 %n, i32 0)
; CHECK: call void @__csan_write_range(ptr %dst, i64 %n, i32 0)
; CHECK: call void @llvm.memcpy
; CHECK: call void @llvm.memmove
; CHECK: call void @llvm.memset

; CHECK: define internal void @csan.module_ctor
; CHECK: call void @__csan_init()
; CHECK-NOT: @__tsan
; CHECK-NOT: @__csan_atomic8
; CHECK-NOT: @__csan_memcpy

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
