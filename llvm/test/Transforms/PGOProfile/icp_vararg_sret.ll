; RUN: opt < %s -pgo-icall-prom -S | FileCheck %s
; RUN: opt < %s -passes=pgo-icall-prom -S | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @va_func(i32 %num, ...) {
entry:
  ret void
}

%struct = type { i32 }
@func_ptr = common global void (i32, %struct*)* null, align 8

define void @test() {
; Even though value profiling suggests @va_func is the call target, don't do
; call promotion because the sret argument is not compatible with the varargs
; function.
; CHECK-LABEL: @test
; CHECK-NOT: call void (i32, ...) @va_func
; CHECK: call void %tmp
; CHECK: ret void

  %s = alloca %struct
  %tmp = load void (i32, %struct*)*, void (i32, %struct*)** @func_ptr, align 8
  call void %tmp(i32 1, %struct* sret(%struct) %s), !prof !1
  ret void
}

!1 = !{!"VP", i32 0, i64 12345, i64 989055279648259519, i64 12345}
