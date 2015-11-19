; RUN: opt < %s -S -scalarrepl | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%struct.S = type { [2 x %struct.anon], double }
%struct.anon = type {}

; CHECK: @test()
; CHECK-NOT: alloca
; CHECK: ret double 1.0

define double @test() nounwind uwtable ssp {
entry:
  %retval = alloca %struct.S, align 8
  %ret = alloca %struct.S, align 8
  %b = getelementptr inbounds %struct.S, %struct.S* %ret, i32 0, i32 1
  store double 1.000000e+00, double* %b, align 8
  %0 = bitcast %struct.S* %retval to i8*
  %1 = bitcast %struct.S* %ret to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* %1, i64 8, i32 8, i1 false)
  %2 = bitcast %struct.S* %retval to double*
  %3 = load double, double* %2, align 1
  ret double %3
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind
