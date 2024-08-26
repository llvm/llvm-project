; RUN: opt -S  -dxil-intrinsic-expansion  -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,EXPCHECK
; RUN: opt -S  -dxil-op-lower  -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,DOPCHECK

; Make sure dxil operation function calls for log are generated.

define noundef float @log_float(float noundef %a) #0 {
entry:
; DOPCHECK: call float @dx.op.unary.f32(i32 23, float %{{.*}})
; EXPCHECK: call float @llvm.log2.f32(float %a)
; CHECK: fmul float 0x3FE62E4300000000, %{{.*}}
  %elt.log = call float @llvm.log.f32(float %a)
  ret float %elt.log
}

define noundef half @log_half(half noundef %a) #0 {
entry:
; DOPCHECK: call half @dx.op.unary.f16(i32 23, half %{{.*}})
; EXPCHECK: call half @llvm.log2.f16(half %a)
; CHECK: fmul half 0xH398C, %{{.*}}
  %elt.log = call half @llvm.log.f16(half %a)
  ret half %elt.log
}

declare half @llvm.log.f16(half)
declare float @llvm.log.f32(float)
