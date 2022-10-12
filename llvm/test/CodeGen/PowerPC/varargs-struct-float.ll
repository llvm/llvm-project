; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 < %s | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.Sf1 = type { float }

define void @foo(float inreg %s.coerce) nounwind {
entry:
  %s = alloca %struct.Sf1, align 4
  store float %s.coerce, ptr %s, align 1
  %0 = load float, ptr %s, align 1
  call void (i32, ...) @testvaSf1(i32 1, float inreg %0)
  ret void
}

; CHECK: stfs {{[0-9]+}}, 116(1)
; CHECK: lwz 4, 116(1)
; CHECK: bl

declare void @testvaSf1(i32, ...)
