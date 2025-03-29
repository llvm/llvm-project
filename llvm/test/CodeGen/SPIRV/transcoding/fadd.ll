; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV:     OpName %[[#r1:]] "r1"
; CHECK-SPIRV:     OpName %[[#r2:]] "r2"
; CHECK-SPIRV:     OpName %[[#r3:]] "r3"
; CHECK-SPIRV:     OpName %[[#r4:]] "r4"
; CHECK-SPIRV:     OpName %[[#r5:]] "r5"
; CHECK-SPIRV:     OpName %[[#r6:]] "r6"
; CHECK-SPIRV:     OpName %[[#r7:]] "r7"
; CHECK-SPIRV:     OpName %[[#r1d:]] "r1"
; CHECK-SPIRV:     OpName %[[#r2d:]] "r2"
; CHECK-SPIRV:     OpName %[[#r3d:]] "r3"
; CHECK-SPIRV:     OpName %[[#r4d:]] "r4"
; CHECK-SPIRV:     OpName %[[#r5d:]] "r5"
; CHECK-SPIRV:     OpName %[[#r6d:]] "r6"
; CHECK-SPIRV:     OpName %[[#r7d:]] "r7"
; CHECK-SPIRV-NOT: OpDecorate %[[#r1]] FPFastMathMode
; CHECK-SPIRV-DAG: OpDecorate %[[#r2]] FPFastMathMode NotNaN
; CHECK-SPIRV-DAG: OpDecorate %[[#r3]] FPFastMathMode NotInf
; CHECK-SPIRV-DAG: OpDecorate %[[#r4]] FPFastMathMode NSZ
; CHECK-SPIRV-DAG: OpDecorate %[[#r5]] FPFastMathMode AllowRecip
; CHECK-SPIRV-DAG: OpDecorate %[[#r6]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|Fast
; CHECK-SPIRV-DAG: OpDecorate %[[#r7]] FPFastMathMode NotNaN|NotInf
; CHECK-SPIRV-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: %[[#double:]] = OpTypeFloat 64

; CHECK-SPIRV:     %[[#r1]] = OpFAdd %[[#float]]
; CHECK-SPIRV:     %[[#r2]] = OpFAdd %[[#float]]
; CHECK-SPIRV:     %[[#r3]] = OpFAdd %[[#float]]
; CHECK-SPIRV:     %[[#r4]] = OpFAdd %[[#float]]
; CHECK-SPIRV:     %[[#r5]] = OpFAdd %[[#float]]
; CHECK-SPIRV:     %[[#r6]] = OpFAdd %[[#float]]
; CHECK-SPIRV:     %[[#r7]] = OpFAdd %[[#float]]
define spir_kernel void @testFAdd_float(float %a, float %b) {
entry:
  %r1 = fadd float %a, %b
  %r2 = fadd nnan float %a, %b
  %r3 = fadd ninf float %a, %b
  %r4 = fadd nsz float %a, %b
  %r5 = fadd arcp float %a, %b
  %r6 = fadd fast float %a, %b
  %r7 = fadd nnan ninf float %a, %b
  ret void
}

; CHECK-SPIRV:     %[[#r1d]] = OpFAdd %[[#double]]
; CHECK-SPIRV:     %[[#r2d]] = OpFAdd %[[#double]]
; CHECK-SPIRV:     %[[#r3d]] = OpFAdd %[[#double]]
; CHECK-SPIRV:     %[[#r4d]] = OpFAdd %[[#double]]
; CHECK-SPIRV:     %[[#r5d]] = OpFAdd %[[#double]]
; CHECK-SPIRV:     %[[#r6d]] = OpFAdd %[[#double]]
; CHECK-SPIRV:     %[[#r7d]] = OpFAdd %[[#double]]
define spir_kernel void @testFAdd_double(double %a, double %b) {
entry:
  %r1 = fadd double %a, %b
  %r2 = fadd nnan double %a, %b
  %r3 = fadd ninf double %a, %b
  %r4 = fadd nsz double %a, %b
  %r5 = fadd arcp double %a, %b
  %r6 = fadd fast double %a, %b
  %r7 = fadd nnan ninf double %a, %b
  ret void
}
