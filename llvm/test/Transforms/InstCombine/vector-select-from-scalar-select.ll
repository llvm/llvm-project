; RUN: opt -passes=instcombine -S < %s | FileCheck %s

define <2 x float> @src(ptr %arg0, ptr %arg1) {
  %v0 = load <2 x float>, ptr %arg1, align 4
  %v1 = load <2 x float>, ptr %arg0, align 4
  %v2 = extractelement <2 x float> %v0, i64 0
  %v3 = extractelement <2 x float> %v1, i64 0
  %v4 = fcmp olt <2 x float> %v0, %v1
  %v5 = extractelement <2 x i1> %v4, i64 0
  %v6 = select i1 %v5, float %v3, float %v2
  %v7 = extractelement <2 x float> %v0, i64 1
  %v8 = extractelement <2 x float> %v1, i64 1
  %v9 = fcmp olt float %v7, %v8
  %v10 = select i1 %v9, float %v8, float %v7
  %v11 = insertelement <2 x float> poison, float %v6, i64 0
  %v12 = insertelement <2 x float> %v11, float %v10, i64 1
  ret <2 x float> %v12
}

; CHECK-LABEL: @src(
; CHECK: %v0 = load <2 x float>
; CHECK: %v1 = load <2 x float>
; CHECK: %v4 = fcmp olt <2 x float> %v0, %v1
; CHECK: %res = select <2 x i1> %v4, <2 x float> %v1, <2 x float> %v0
; CHECK: ret <2 x float> %res