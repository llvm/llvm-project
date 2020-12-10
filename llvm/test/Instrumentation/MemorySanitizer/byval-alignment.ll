; Test that copy alignment for byval arguments is limited by param-tls slot alignment.

; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck  \
; RUN: %s
; RUN: opt < %s -msan -msan-check-access-address=0 -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i64, i64, i64, [8 x i8] }

; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 {{.*}} add {{.*}} ptrtoint {{.*}} @__msan_param_tls {{.*}} i64 8) {{.*}}, i8* align 8 {{.*}}, i64 32, i1 false)

define void @Caller() sanitize_memory {
entry:
  %agg.tmp = alloca %struct.S, align 16
  call void @Callee(i32 1, %struct.S* byval(%struct.S) align 16 %agg.tmp)
  ret void
}

declare void @Callee(i32, %struct.S* byval(%struct.S) align 16)
