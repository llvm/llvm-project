; RUN: llc -mcpu=pwr7 -O0 < %s | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.Sf1 = type { float }

define void @foo(float inreg %s.coerce) nounwind {
entry:
  %s = alloca %struct.Sf1, align 4
  %coerce.dive = getelementptr %struct.Sf1* %s, i32 0, i32 0
  store float %s.coerce, float* %coerce.dive, align 1
  %coerce.dive1 = getelementptr %struct.Sf1* %s, i32 0, i32 0
  %0 = load float* %coerce.dive1, align 1
  call void (i32, ...)* @testvaSf1(i32 1, float inreg %0)
  ret void
}

; CHECK: stfs {{[0-9]+}}, 60(1)
; CHECK: ld 4, 56(1)
; CHECK: bl

declare void @testvaSf1(i32, ...)
