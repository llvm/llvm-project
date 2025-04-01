; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for isinf are generated for float and half.

define noundef i1 @isinf_float(float noundef %a) {
entry:
  ; CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 9, float %{{.*}}) #[[#ATTR:]]
  %dx.isinf = call i1 @llvm.dx.isinf.f32(float %a)
  ret i1 %dx.isinf
}

define noundef i1 @isinf_half(half noundef %a) {
entry:
  ; CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 9, half %{{.*}}) #[[#ATTR]]
  %dx.isinf = call i1 @llvm.dx.isinf.f16(half %a)
  ret i1 %dx.isinf
}

; CHECK: attributes #[[#ATTR]] = {{{.*}} memory(none) {{.*}}}

declare i1 @llvm.dx.isinf.f16(half)
declare i1 @llvm.dx.isinf.f32(float)
