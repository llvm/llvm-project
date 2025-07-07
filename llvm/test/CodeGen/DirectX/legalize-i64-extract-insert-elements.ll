; RUN: opt -S -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

define noundef <4 x float> @float4_extract(<4 x float> noundef %a) {
entry:
  ; CHECK: [[ee0:%.*]] = extractelement <4 x float> %a, i32 0
  ; CHECK: [[ee1:%.*]] = extractelement <4 x float> %a, i32 1
  ; CHECK: [[ee2:%.*]] = extractelement <4 x float> %a, i32 2
  ; CHECK: [[ee3:%.*]] = extractelement <4 x float> %a, i32 3
  ; CHECK: insertelement <4 x float> poison, float [[ee0]], i32 0
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ee1]], i32 1
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ee2]], i32 2
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ee3]], i32 3

  %a.i0 = extractelement <4 x float> %a, i64 0
  %a.i1 = extractelement <4 x float> %a, i64 1
  %a.i2 = extractelement <4 x float> %a, i64 2
  %a.i3 = extractelement <4 x float> %a, i64 3
  
  %.upto0 = insertelement <4 x float> poison, float %a.i0, i64 0
  %.upto1 = insertelement <4 x float> %.upto0, float %a.i1, i64 1
  %.upto2 = insertelement <4 x float> %.upto1, float %a.i2, i64 2
  %0 = insertelement <4 x float> %.upto2, float %a.i3, i64 3
  ret <4 x float> %0
}
