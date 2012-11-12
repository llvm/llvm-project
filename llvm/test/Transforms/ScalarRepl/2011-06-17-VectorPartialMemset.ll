; RUN: opt < %s -scalarrepl -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

; CHECK: f
; CHECK-NOT: alloca
; CHECK: %[[A:[a-z0-9]*]] = and i128 undef, -16777216
; CHECK: %[[B:[a-z0-9]*]] = bitcast i128 %[[A]] to <4 x float>
; CHECK: %[[C:[a-z0-9]*]] = extractelement <4 x float> %[[B]], i32 0
; CHECK: ret float %[[C]]

define float @f() nounwind ssp {
entry:
  %a = alloca <4 x float>, align 16
  %p = bitcast <4 x float>* %a to i8*
  call void @llvm.memset.p0i8.i32(i8* %p, i8 0, i32 3, i32 16, i1 false)
  %vec = load <4 x float>* %a, align 8
  %val = extractelement <4 x float> %vec, i32 0
  ret float %val
}

; CHECK: g
; CHECK-NOT: alloca
; CHECK: and i128

define void @g() nounwind ssp {
entry:
  %a = alloca { <4 x float> }, align 16
  %p = bitcast { <4 x float> }* %a to i8*
  call void @llvm.memset.p0i8.i32(i8* %p, i8 0, i32 16, i32 16, i1 false)
  %q = bitcast { <4 x float> }* %a to [2 x <2 x float>]*
  %arrayidx = getelementptr inbounds [2 x <2 x float>]* %q, i32 0, i32 0
  store <2 x float> undef, <2 x float>* %arrayidx, align 8
  ret void
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind
