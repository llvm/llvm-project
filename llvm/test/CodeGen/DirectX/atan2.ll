; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure correct dxil expansions for atan2 are generated for float and half.

define noundef float @atan2_float(float noundef %y, float noundef %x) {
entry:
; CHECK: [[DIV:%.+]] = fdiv float %y, %x
; CHECK: [[TAN:%.+]] = call float @dx.op.unary.f32(i32 17, float [[DIV]])
; CHECK-DAG: [[ADD_PI:%.+]] = fadd float [[TAN]], 0x400921FB60000000
; CHECK-DAG: [[SUB_PI:%.+]] = fsub float [[TAN]], 0x400921FB60000000
; CHECK-DAG: [[X_LT_0:%.+]] = fcmp olt float %x, 0.000000e+00
; CHECK-DAG: [[X_EQ_0:%.+]] = fcmp oeq float %x, 0.000000e+00 
; CHECK-DAG: [[Y_GE_0:%.+]] = fcmp oge float %y, 0.000000e+00 
; CHECK-DAG: [[Y_LT_0:%.+]] = fcmp olt float %y, 0.000000e+00
; CHECK: [[XLT0_AND_YGE0:%.+]] = and i1 [[X_LT_0]], [[Y_GE_0]]
; CHECK: [[SELECT_ADD_PI:%.+]] = select i1 [[XLT0_AND_YGE0]], float [[ADD_PI]], float [[TAN]]
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
; CHECK: [[TAN:%.+]] = call half @dx.op.unary.f16(i32 17, half [[DIV]])
; CHECK-DAG: [[ADD_PI:%.+]] = fadd half [[TAN]], 0xH4248
; CHECK-DAG: [[SUB_PI:%.+]] = fsub half [[TAN]], 0xH4248
; CHECK-DAG: [[X_LT_0:%.+]] = fcmp olt half %x, 0xH0000
; CHECK-DAG: [[X_EQ_0:%.+]] = fcmp oeq half %x, 0xH0000 
; CHECK-DAG: [[Y_GE_0:%.+]] = fcmp oge half %y, 0xH0000 
; CHECK-DAG: [[Y_LT_0:%.+]] = fcmp olt half %y, 0xH0000
; CHECK: [[XLT0_AND_YGE0:%.+]] = and i1 [[X_LT_0]], [[Y_GE_0]]
; CHECK: [[SELECT_ADD_PI:%.+]] = select i1 [[XLT0_AND_YGE0]], half [[ADD_PI]], half [[TAN]]
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

declare half @llvm.atan2.f16(half, half)
declare float @llvm.atan2.f32(float, float)
