; RUN: opt < %s -msan-check-access-address=0 -msan-kernel=1 -S -passes=msan 2>&1 | FileCheck %s

; Memory intrinsics are rewritten into KMSAN runtime calls. Outside an
; instrumentation region they have to be left alone: emitting the call there
; instruments code the region markers explicitly excluded, and in noinstr kernel
; code it makes objtool report
;   "call to __msan_memcpy() leaves .noinstr.text section".

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

define void @mem_intrinsics(ptr %dst, ptr %src) disable_sanitizer_instrumentation {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 64, i1 false)
  call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 64, i1 false)
  call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 64, i1 false)
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 64, i1 false)
  call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 64, i1 false)
  call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 64, i1 false)
  call void @llvm.sanitizer.instrumentation.end()
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 64, i1 false)
  ret void
}

; CHECK-LABEL: @mem_intrinsics(
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 64, i1 false)
; CHECK: call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 64, i1 false)
; CHECK: call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 64, i1 false)
; CHECK-NOT: __msan_get_context_state
; CHECK: call ptr @__msan_memcpy(ptr %dst, ptr %src, i64 64)
; CHECK: call ptr @__msan_memmove(ptr %dst, ptr %src, i64 64)
; CHECK: call ptr @__msan_memset(ptr %dst, i32 0, i64 64)
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 64, i1 false)
; CHECK: ret void
