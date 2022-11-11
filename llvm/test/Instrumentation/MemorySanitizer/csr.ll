; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck %s --implicit-check-not="call void @__msan_warning"
; RUN: opt < %s -msan-check-access-address=1 -S -passes=msan 2>&1 | FileCheck %s --check-prefix=ADDR --implicit-check-not="call void @__msan_warning"

; REQUIRES: x86-registered-target

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.x86.sse.stmxcsr(i8*)
declare void @llvm.x86.sse.ldmxcsr(i8*)

define void @getcsr(i32 *%p) sanitize_memory {
entry:
  %0 = bitcast i32* %p to i8*
  call void @llvm.x86.sse.stmxcsr(i8* %0)
  ret void
}

; CHECK-LABEL: @getcsr(
; CHECK: store i32 0, i32*
; CHECK: call void @llvm.x86.sse.stmxcsr(
; CHECK: ret void

; ADDR-LABEL: @getcsr(
; ADDR: %[[A:.*]] = load i64, i64* getelementptr inbounds {{.*}} @__msan_param_tls, i32 0, i32 0), align 8
; ADDR: %[[B:.*]] = icmp ne i64 %[[A]], 0
; ADDR: br i1 %[[B]], label {{.*}}, label
; ADDR: call void @__msan_warning_noreturn()
; ADDR: call void @llvm.x86.sse.stmxcsr(
; ADDR: ret void

; Function Attrs: nounwind uwtable
define void @setcsr(i32 *%p) sanitize_memory {
entry:
  %0 = bitcast i32* %p to i8*
  call void @llvm.x86.sse.ldmxcsr(i8* %0)
  ret void
}

; CHECK-LABEL: @setcsr(
; CHECK: %[[A:.*]] = load i32, i32* %{{.*}}, align 1
; CHECK: %[[B:.*]] = icmp ne i32 %[[A]], 0
; CHECK: br i1 %[[B]], label {{.*}}, label
; CHECK: call void @__msan_warning_noreturn()
; CHECK: call void @llvm.x86.sse.ldmxcsr(
; CHECK: ret void

; ADDR-LABEL: @setcsr(
; ADDR: %[[A:.*]] = load i64, i64* getelementptr inbounds {{.*}} @__msan_param_tls, i32 0, i32 0), align 8
; ADDR: %[[C:.*]] = load i32, i32*
; ADDR: %[[B:.*]] = icmp ne i64 %[[A]], 0
; ADDR: %[[D:.*]] = icmp ne i32 %[[C]], 0
; ADDR: %[[E:.*]] = or i1 %[[B]], %[[D]]
; ADDR: br i1 %[[E]], label {{.*}}, label
; ADDR: call void @__msan_warning_noreturn()
; ADDR: call void @llvm.x86.sse.ldmxcsr(
; ADDR: ret void
