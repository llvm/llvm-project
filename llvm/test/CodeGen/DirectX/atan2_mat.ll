; RUN: opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefix=EXPCHECK
; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefix=DOPCHECK

; Make sure correct dxil expansions for atan2 are generated for float and half.

define noundef <16 x half> @atan2_half4x4(<16 x half> noundef %y, <16 x half> noundef %x) {
entry:
; Just Expansion, no scalarization or lowering:
; EXPCHECK: [[DIV:%.+]] = fdiv <16 x half> %y, %x
; EXPCHECK: [[ATAN:%.+]] = call <16 x half> @llvm.atan.v16f16(<16 x half> [[DIV]])
; EXPCHECK-DAG: [[ADD_PI:%.+]] = fadd <16 x half> [[ATAN]], splat (half
; EXPCHECK-DAG: [[SUB_PI:%.+]] = fsub <16 x half> [[ATAN]], splat (half
; EXPCHECK-DAG: [[X_LT_0:%.+]] = fcmp olt <16 x half> %x, zeroinitializer
; EXPCHECK-DAG: [[X_EQ_0:%.+]] = fcmp oeq <16 x half> %x, zeroinitializer
; EXPCHECK-DAG: [[Y_GE_0:%.+]] = fcmp oge <16 x half> %y, zeroinitializer
; EXPCHECK-DAG: [[Y_LT_0:%.+]] = fcmp olt <16 x half> %y, zeroinitializer
; EXPCHECK: [[XLT0_AND_YGE0:%.+]] = and <16 x i1> [[X_LT_0]], [[Y_GE_0]]
; EXPCHECK: [[SELECT_ADD_PI:%.+]] = select <16 x i1> [[XLT0_AND_YGE0]], <16 x half> [[ADD_PI]], <16 x half> [[ATAN]]
; EXPCHECK: [[XLT0_AND_YLT0:%.+]] = and <16 x i1> [[X_LT_0]], [[Y_LT_0]]
; EXPCHECK: [[SELECT_SUB_PI:%.+]] = select <16 x i1> [[XLT0_AND_YLT0]], <16 x half> [[SUB_PI]], <16 x half> [[SELECT_ADD_PI]]
; EXPCHECK: [[XEQ0_AND_YLT0:%.+]] = and <16 x i1> [[X_EQ_0]], [[Y_LT_0]]
; EXPCHECK: [[SELECT_NEGHPI:%.+]] = select <16 x i1> [[XEQ0_AND_YLT0]], <16 x half> splat (half {{.*}}), <16 x half> [[SELECT_SUB_PI]]
; EXPCHECK: [[XEQ0_AND_YGE0:%.+]] = and <16 x i1> [[X_EQ_0]], [[Y_GE_0]]
; EXPCHECK: [[SELECT_HPI:%.+]] = select <16 x i1> [[XEQ0_AND_YGE0]], <16 x half> splat (half {{.*}}), <16 x half> [[SELECT_NEGHPI]]
; EXPCHECK: ret <16 x half> [[SELECT_HPI]]

; Scalarization occurs after expansion, so atan scalarization is tested separately.
; Expansion, scalarization and lowering:
; Just make sure this expands to exactly 16 scalar DXIL atan (OpCode=17) calls.
; DOPCHECK-COUNT-16: call half @dx.op.unary.f16(i32 17, half %{{.*}})
; DOPCHECK-NOT: call half @dx.op.unary.f16(i32 17,

  %elt.atan2 = call <16 x half> @llvm.atan2.v16f16(<16 x half> %y, <16 x half> %x)
  ret <16 x half> %elt.atan2
}

define noundef <16 x float> @atan2_float4x4(<16 x float> noundef %y, <16 x float> noundef %x) {
entry:
; Just Expansion, no scalarization or lowering:
; EXPCHECK: [[DIV:%.+]] = fdiv <16 x float> %y, %x
; EXPCHECK: [[ATAN:%.+]] = call <16 x float> @llvm.atan.v16f32(<16 x float> [[DIV]])
; EXPCHECK-DAG: [[ADD_PI:%.+]] = fadd <16 x float> [[ATAN]], splat (float
; EXPCHECK-DAG: [[SUB_PI:%.+]] = fsub <16 x float> [[ATAN]], splat (float
; EXPCHECK-DAG: [[X_LT_0:%.+]] = fcmp olt <16 x float> %x, zeroinitializer
; EXPCHECK-DAG: [[X_EQ_0:%.+]] = fcmp oeq <16 x float> %x, zeroinitializer
; EXPCHECK-DAG: [[Y_GE_0:%.+]] = fcmp oge <16 x float> %y, zeroinitializer
; EXPCHECK-DAG: [[Y_LT_0:%.+]] = fcmp olt <16 x float> %y, zeroinitializer
; EXPCHECK: [[XLT0_AND_YGE0:%.+]] = and <16 x i1> [[X_LT_0]], [[Y_GE_0]]
; EXPCHECK: [[SELECT_ADD_PI:%.+]] = select <16 x i1> [[XLT0_AND_YGE0]], <16 x float> [[ADD_PI]], <16 x float> [[ATAN]]
; EXPCHECK: [[XLT0_AND_YLT0:%.+]] = and <16 x i1> [[X_LT_0]], [[Y_LT_0]]
; EXPCHECK: [[SELECT_SUB_PI:%.+]] = select <16 x i1> [[XLT0_AND_YLT0]], <16 x float> [[SUB_PI]], <16 x float> [[SELECT_ADD_PI]]
; EXPCHECK: [[XEQ0_AND_YLT0:%.+]] = and <16 x i1> [[X_EQ_0]], [[Y_LT_0]]
; EXPCHECK: [[SELECT_NEGHPI:%.+]] = select <16 x i1> [[XEQ0_AND_YLT0]], <16 x float> splat (float {{.*}}), <16 x float> [[SELECT_SUB_PI]]
; EXPCHECK: [[XEQ0_AND_YGE0:%.+]] = and <16 x i1> [[X_EQ_0]], [[Y_GE_0]]
; EXPCHECK: [[SELECT_HPI:%.+]] = select <16 x i1> [[XEQ0_AND_YGE0]], <16 x float> splat (float {{.*}}), <16 x float> [[SELECT_NEGHPI]]
; EXPCHECK: ret <16 x float> [[SELECT_HPI]]

; Scalarization occurs after expansion, so atan scalarization is tested separately.
; Expansion, scalarization and lowering:
; Just make sure this expands to exactly 16 scalar DXIL atan (OpCode=17) calls.
; DOPCHECK-COUNT-16: call float @dx.op.unary.f32(i32 17, float %{{.*}})
; DOPCHECK-NOT: call float @dx.op.unary.f32(i32 17,

  %elt.atan2 = call <16 x float> @llvm.atan2.v16f32(<16 x float> %y, <16 x float> %x)
  ret <16 x float> %elt.atan2
}

declare <16 x float> @llvm.atan2.v16f32(<16 x float>, <16 x float>)
declare <16 x half> @llvm.atan2.v16f16(<16 x half>, <16 x half>)
