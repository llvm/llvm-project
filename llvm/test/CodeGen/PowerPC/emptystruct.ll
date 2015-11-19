; RUN: llc -mcpu=pwr7 -O0 -fast-isel=false < %s | FileCheck %s

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

define void @callee(%struct.empty* noalias sret %agg.result, %struct.empty* byval %a1, %struct.empty* %a2, %struct.empty* byval %a3) nounwind {
entry:
  %a2.addr = alloca %struct.empty*, align 8
  store %struct.empty* %a2, %struct.empty** %a2.addr, align 8
  %0 = load %struct.empty*, %struct.empty** %a2.addr, align 8
  %1 = bitcast %struct.empty* %agg.result to i8*
  %2 = bitcast %struct.empty* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 0, i1 false)
  ret void
}

; CHECK-LABEL: callee:
; CHECK: std 4,
; CHECK-NOT: std 5,
; CHECK-NOT: std 6,
; CHECK: blr

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

define void @caller(%struct.empty* noalias sret %agg.result) nounwind {
entry:
  %e1 = alloca %struct.empty, align 1
  %e2 = alloca %struct.empty, align 1
  %e3 = alloca %struct.empty, align 1
  call void @callee(%struct.empty* sret %agg.result, %struct.empty* byval %e1, %struct.empty* %e2, %struct.empty* byval %e3)
  ret void
}

; CHECK-LABEL: caller:
; CHECK: addi 4,
; CHECK-NOT: std 5,
; CHECK-NOT: std 6,
; CHECK: bl callee
