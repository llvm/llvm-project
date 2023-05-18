; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -fast-isel=false < %s | FileCheck %s

; This tests correct handling of empty aggregate parameters and return values.
; An empty parameter passed by value does not consume a protocol register or
; a parameter save area doubleword.  An empty parameter passed by reference
; is treated as any other pointer parameter.  An empty aggregate return value
; is treated as any other aggregate return value, passed via address as a
; hidden parameter in GPR3.  In this example, GPR3 contains the return value
; address, GPR4 contains the address of e2, and e1 and e3 are not passed or
; received.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.empty = type {}

define void @callee(ptr noalias sret(%struct.empty) %agg.result, ptr byval(%struct.empty) %a1, ptr %a2, ptr byval(%struct.empty) %a3) nounwind {
entry:
  %a2.addr = alloca ptr, align 8
  store ptr %a2, ptr %a2.addr, align 8
  %0 = load ptr, ptr %a2.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %agg.result, ptr %0, i64 0, i1 false)
  ret void
}

; CHECK-LABEL: callee:
; CHECK: std 4,
; CHECK-NOT: std 5,
; CHECK-NOT: std 6,
; CHECK: blr

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

define void @caller(ptr noalias sret(%struct.empty) %agg.result) nounwind {
entry:
  %e1 = alloca %struct.empty, align 1
  %e2 = alloca %struct.empty, align 1
  %e3 = alloca %struct.empty, align 1
  call void @callee(ptr sret(%struct.empty) %agg.result, ptr byval(%struct.empty) %e1, ptr %e2, ptr byval(%struct.empty) %e3)
  ret void
}

; CHECK-LABEL: caller:
; CHECK: addi 4,
; CHECK-NOT: std 5,
; CHECK-NOT: std 6,
; CHECK: bl callee
