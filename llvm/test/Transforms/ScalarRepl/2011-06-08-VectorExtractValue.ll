; RUN: opt < %s -S -scalarrepl | FileCheck %s
; RUN: opt < %s -S -scalarrepl-ssa | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

%0 = type { <2 x float>, float }
%struct.PointC3 = type { %struct.array }
%struct.Point_3 = type { %struct.PointC3 }
%struct.array = type { [3 x float], [4 x i8] }

; CHECK: main
; CHECK-NOT: alloca
; CHECK: extractelement <2 x float> zeroinitializer, i32 0

define void @main() uwtable ssp {
entry:
  %ref.tmp2 = alloca %0, align 16
  %tmpcast = bitcast %0* %ref.tmp2 to %struct.Point_3*
  %0 = getelementptr %0* %ref.tmp2, i64 0, i32 0
  store <2 x float> zeroinitializer, <2 x float>* %0, align 16
  %1 = getelementptr inbounds %struct.Point_3* %tmpcast, i64 0, i32 0
  %base.i.i.i = getelementptr inbounds %struct.PointC3* %1, i64 0, i32 0
  %arrayidx.i.i.i.i = getelementptr inbounds %struct.array* %base.i.i.i, i64 0, i32 0, i64 0
  %tmp5.i.i = load float* %arrayidx.i.i.i.i, align 4
  ret void
}

; CHECK: test1
; CHECK-NOT: alloca
; CHECK: extractelement <2 x float> zeroinitializer, i32 0

define void @test1() uwtable ssp {
entry:
  %ref.tmp2 = alloca {<2 x float>, float}, align 16
  %tmpcast = bitcast {<2 x float>, float}* %ref.tmp2 to float*
  %0 = getelementptr {<2 x float>, float}* %ref.tmp2, i64 0, i32 0
  store <2 x float> zeroinitializer, <2 x float>* %0, align 16
  %tmp5.i.i = load float* %tmpcast, align 4
  ret void
}

; CHECK: test2
; CHECK-NOT: alloca
; CHECK: %[[A:[a-z0-9]*]] = extractelement <2 x float> zeroinitializer, i32 0
; CHECK: fadd float %[[A]], 1.000000e+00
; CHECK-NOT: insertelement
; CHECK-NOT: extractelement

define float @test2() uwtable ssp {
entry:
  %ref.tmp2 = alloca {<2 x float>, float}, align 16
  %tmpcast = bitcast {<2 x float>, float}* %ref.tmp2 to float*
  %tmpcast2 = getelementptr {<2 x float>, float}* %ref.tmp2, i64 0, i32 1
  %0 = getelementptr {<2 x float>, float}* %ref.tmp2, i64 0, i32 0
  store <2 x float> zeroinitializer, <2 x float>* %0, align 16
  store float 1.0, float* %tmpcast2, align 4
  %r1 = load float* %tmpcast, align 4
  %r2 = load float* %tmpcast2, align 4
  %r = fadd float %r1, %r2
  ret float %r
}

; CHECK: test3
; CHECK: %[[A:[a-z0-9]*]] = extractelement <2 x float> <float 2.000000e+00, float 3.000000e+00>, i32 1
; CHECK: ret float %[[A]]

define float @test3() {
entry:
  %ai = alloca { <2 x float>, <2 x float> }, align 8
  store { <2 x float>, <2 x float> } {<2 x float> <float 0.0, float 1.0>, <2 x float> <float 2.0, float 3.0>}, { <2 x float>, <2 x float> }* %ai, align 8
  %tmpcast = bitcast { <2 x float>, <2 x float> }* %ai to [4 x float]*
  %arrayidx = getelementptr inbounds [4 x float]* %tmpcast, i64 0, i64 3
  %f = load float* %arrayidx, align 4
  ret float %f
}
