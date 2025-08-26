; RUN: opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,EXPCHECK
; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,DOPCHECK

; Make sure correct dxil expansions for atan2 are generated for float and half.

define noundef float @atan2_float(float noundef %y, float noundef %x) {
entry:
; CHECK: [[DIV:%.+]] = fdiv float %y, %x
; EXPCHECK: [[ATAN:%.+]] = call float @llvm.atan.f32(float [[DIV]])
; DOPCHECK: [[ATAN:%.+]] = call float @dx.op.unary.f32(i32 17, float [[DIV]])
; CHECK-DAG: [[ADD_PI:%.+]] = fadd float [[ATAN]], 0x400921FB60000000
; CHECK-DAG: [[SUB_PI:%.+]] = fsub float [[ATAN]], 0x400921FB60000000
; CHECK-DAG: [[X_LT_0:%.+]] = fcmp olt float %x, 0.000000e+00
; CHECK-DAG: [[X_EQ_0:%.+]] = fcmp oeq float %x, 0.000000e+00 
; CHECK-DAG: [[Y_GE_0:%.+]] = fcmp oge float %y, 0.000000e+00 
; CHECK-DAG: [[Y_LT_0:%.+]] = fcmp olt float %y, 0.000000e+00
; CHECK: [[XLT0_AND_YGE0:%.+]] = and i1 [[X_LT_0]], [[Y_GE_0]]
; CHECK: [[SELECT_ADD_PI:%.+]] = select i1 [[XLT0_AND_YGE0]], float [[ADD_PI]], float [[ATAN]]
; CHECK: [[XLT0_AND_YLT0:%.+]] = and i1 [[X_LT_0]], [[Y_LT_0]]
; CHECK: [[SELECT_SUB_PI:%.+]] = select i1 [[XLT0_AND_YLT0]], float [[SUB_PI]], float [[SELECT_ADD_PI]]
; CHECK: [[XEQ0_AND_YLT0:%.+]] = and i1 [[X_EQ_0]], [[Y_LT_0]]
; CHECK: [[SELECT_NEGHPI:%.+]] = select i1 [[XEQ0_AND_YLT0]], float 0xBFF921FB60000000, float [[SELECT_SUB_PI]]
; CHECK: [[XEQ0_AND_YGE0:%.+]] = and i1 [[X_EQ_0]], [[Y_GE_0]]
; CHECK: [[SELECT_HPI:%.+]] = select i1 [[XEQ0_AND_YGE0]], float 0x3FF921FB60000000, float [[SELECT_NEGHPI]]
; CHECK: ret float [[SELECT_HPI]]
  %elt.atan2 = call float @llvm.atan2.f32(float %y, float %x)
  ret float %elt.atan2
}

define noundef half @atan2_half(half noundef %y, half noundef %x) {
entry:
; CHECK: [[DIV:%.+]] = fdiv half %y, %x
; EXPCHECK: [[ATAN:%.+]] = call half @llvm.atan.f16(half [[DIV]])
; DOPCHECK: [[ATAN:%.+]] = call half @dx.op.unary.f16(i32 17, half [[DIV]])
; CHECK-DAG: [[ADD_PI:%.+]] = fadd half [[ATAN]], 0xH4248
; CHECK-DAG: [[SUB_PI:%.+]] = fsub half [[ATAN]], 0xH4248
; CHECK-DAG: [[X_LT_0:%.+]] = fcmp olt half %x, 0xH0000
; CHECK-DAG: [[X_EQ_0:%.+]] = fcmp oeq half %x, 0xH0000 
; CHECK-DAG: [[Y_GE_0:%.+]] = fcmp oge half %y, 0xH0000 
; CHECK-DAG: [[Y_LT_0:%.+]] = fcmp olt half %y, 0xH0000
; CHECK: [[XLT0_AND_YGE0:%.+]] = and i1 [[X_LT_0]], [[Y_GE_0]]
; CHECK: [[SELECT_ADD_PI:%.+]] = select i1 [[XLT0_AND_YGE0]], half [[ADD_PI]], half [[ATAN]]
; CHECK: [[XLT0_AND_YLT0:%.+]] = and i1 [[X_LT_0]], [[Y_LT_0]]
; CHECK: [[SELECT_SUB_PI:%.+]] = select i1 [[XLT0_AND_YLT0]], half [[SUB_PI]], half [[SELECT_ADD_PI]]
; CHECK: [[XEQ0_AND_YLT0:%.+]] = and i1 [[X_EQ_0]], [[Y_LT_0]]
; CHECK: [[SELECT_NEGHPI:%.+]] = select i1 [[XEQ0_AND_YLT0]], half 0xHBE48, half [[SELECT_SUB_PI]]
; CHECK: [[XEQ0_AND_YGE0:%.+]] = and i1 [[X_EQ_0]], [[Y_GE_0]]
; CHECK: [[SELECT_HPI:%.+]] = select i1 [[XEQ0_AND_YGE0]], half 0xH3E48, half [[SELECT_NEGHPI]]
; CHECK: ret half [[SELECT_HPI]]
  %elt.atan2 = call half @llvm.atan2.f16(half %y, half %x)
  ret half %elt.atan2
}

define noundef <4 x float> @atan2_float4(<4 x float> noundef %y, <4 x float> noundef %x) {
entry:
; Just Expansion, no scalarization or lowering:
; EXPCHECK: [[DIV:%.+]] = fdiv <4 x float> %y, %x
; EXPCHECK: [[ATAN:%.+]] = call <4 x float> @llvm.atan.v4f32(<4 x float> [[DIV]])
; EXPCHECK-DAG: [[ADD_PI:%.+]] = fadd <4 x float> [[ATAN]], splat (float 0x400921FB60000000)
; EXPCHECK-DAG: [[SUB_PI:%.+]] = fsub <4 x float> [[ATAN]], splat (float 0x400921FB60000000)
; EXPCHECK-DAG: [[X_LT_0:%.+]] = fcmp olt <4 x float> %x, zeroinitializer
; EXPCHECK-DAG: [[X_EQ_0:%.+]] = fcmp oeq <4 x float> %x, zeroinitializer
; EXPCHECK-DAG: [[Y_GE_0:%.+]] = fcmp oge <4 x float> %y, zeroinitializer
; EXPCHECK-DAG: [[Y_LT_0:%.+]] = fcmp olt <4 x float> %y, zeroinitializer
; EXPCHECK: [[XLT0_AND_YGE0:%.+]] = and <4 x i1> [[X_LT_0]], [[Y_GE_0]]
; EXPCHECK: [[SELECT_ADD_PI:%.+]] = select <4 x i1> [[XLT0_AND_YGE0]], <4 x float> [[ADD_PI]], <4 x float> [[ATAN]]
; EXPCHECK: [[XLT0_AND_YLT0:%.+]] = and <4 x i1> [[X_LT_0]], [[Y_LT_0]]
; EXPCHECK: [[SELECT_SUB_PI:%.+]] = select <4 x i1> [[XLT0_AND_YLT0]], <4 x float> [[SUB_PI]], <4 x float> [[SELECT_ADD_PI]]
; EXPCHECK: [[XEQ0_AND_YLT0:%.+]] = and <4 x i1> [[X_EQ_0]], [[Y_LT_0]]
; EXPCHECK: [[SELECT_NEGHPI:%.+]] = select <4 x i1> [[XEQ0_AND_YLT0]], <4 x float> splat (float 0xBFF921FB60000000), <4 x float> [[SELECT_SUB_PI]]
; EXPCHECK: [[XEQ0_AND_YGE0:%.+]] = and <4 x i1> [[X_EQ_0]], [[Y_GE_0]]
; EXPCHECK: [[SELECT_HPI:%.+]] = select <4 x i1> [[XEQ0_AND_YGE0]], <4 x float> splat (float 0x3FF921FB60000000), <4 x float> [[SELECT_NEGHPI]]
; EXPCHECK: ret <4 x float> [[SELECT_HPI]]

; Scalarization occurs after expansion, so atan scalarization is tested separately.
; Expansion, scalarization and lowering:
; Just make sure this expands to exactly 4 scalar DXIL atan (OpCode=17) calls.
; DOPCHECK-COUNT-4: call float @dx.op.unary.f32(i32 17, float %{{.*}})
; DOPCHECK-NOT: call float @dx.op.unary.f32(i32 17,

  %elt.atan2 = call <4 x float> @llvm.atan2.v4f32(<4 x float> %y, <4 x float> %x)
  ret <4 x float> %elt.atan2
}

declare half @llvm.atan2.f16(half, half)
declare float @llvm.atan2.f32(float, float)
declare <4 x float> @llvm.atan2.v4f32(<4 x float>, <4 x float>)
