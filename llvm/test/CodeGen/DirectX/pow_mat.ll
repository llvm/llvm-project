; RUN: opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefix=EXPCHECK
; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefix=DOPCHECK

; Make sure correct dxil expansions for pow are generated for float and half.

define noundef <16 x half> @pow_half4x4(<16 x half> noundef %a, <16 x half> noundef %b) {
entry:
; Just Expansion, no scalarization or lowering:
; EXPCHECK: [[LOG2:%.+]] = call <16 x half> @llvm.log2.v16f16(<16 x half> %a)
; EXPCHECK: [[MUL:%.+]] = fmul <16 x half> [[LOG2]], %b
; EXPCHECK: [[EXP2:%.+]] = call <16 x half> @llvm.exp2.v16f16(<16 x half> [[MUL]])
; EXPCHECK: ret <16 x half> [[EXP2]]

; Scalarization occurs after expansion, so log2/exp2 scalarization is tested separately.
; Expansion, scalarization and lowering:
; Just make sure this expands to exactly 16 scalar DXIL log2 (OpCode=23) and 16 scalar DXIL exp2 (OpCode=21) calls.
; DOPCHECK-COUNT-16: call half @dx.op.unary.f16(i32 23, half %{{.*}})
; DOPCHECK-NOT: call half @dx.op.unary.f16(i32 23,
; DOPCHECK-COUNT-16: call half @dx.op.unary.f16(i32 21, half %{{.*}})
; DOPCHECK-NOT: call half @dx.op.unary.f16(i32 21,

  %elt.pow = call <16 x half> @llvm.pow.v16f16(<16 x half> %a, <16 x half> %b)
  ret <16 x half> %elt.pow
}

define noundef <16 x float> @pow_float4x4(<16 x float> noundef %a, <16 x float> noundef %b) {
entry:
; Just Expansion, no scalarization or lowering:
; EXPCHECK: [[LOG2:%.+]] = call <16 x float> @llvm.log2.v16f32(<16 x float> %a)
; EXPCHECK: [[MUL:%.+]] = fmul <16 x float> [[LOG2]], %b
; EXPCHECK: [[EXP2:%.+]] = call <16 x float> @llvm.exp2.v16f32(<16 x float> [[MUL]])
; EXPCHECK: ret <16 x float> [[EXP2]]

; Scalarization occurs after expansion, so log2/exp2 scalarization is tested separately.
; Expansion, scalarization and lowering:
; Just make sure this expands to exactly 16 scalar DXIL log2 (OpCode=23) and 16 scalar DXIL exp2 (OpCode=21) calls.
; DOPCHECK-COUNT-16: call float @dx.op.unary.f32(i32 23, float %{{.*}})
; DOPCHECK-NOT: call float @dx.op.unary.f32(i32 23,
; DOPCHECK-COUNT-16: call float @dx.op.unary.f32(i32 21, float %{{.*}})
; DOPCHECK-NOT: call float @dx.op.unary.f32(i32 21,

  %elt.pow = call <16 x float> @llvm.pow.v16f32(<16 x float> %a, <16 x float> %b)
  ret <16 x float> %elt.pow
}

declare <16 x half> @llvm.pow.v16f16(<16 x half>, <16 x half>)
declare <16 x float> @llvm.pow.v16f32(<16 x float>, <16 x float>)
