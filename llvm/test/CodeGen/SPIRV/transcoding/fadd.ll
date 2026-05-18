; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-NOT: OpDecorate %[[#]] FPFastMathMode
; CHECK-SPIRV-DAG: OpDecorate %[[#r2:]] FPFastMathMode NotNaN
; CHECK-SPIRV-DAG: OpDecorate %[[#r3:]] FPFastMathMode NotInf
; CHECK-SPIRV-DAG: OpDecorate %[[#r4:]] FPFastMathMode NSZ
; CHECK-SPIRV-DAG: OpDecorate %[[#r5:]] FPFastMathMode AllowRecip
; CHECK-SPIRV-DAG: OpDecorate %[[#r6:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|Fast
; CHECK-SPIRV-DAG: OpDecorate %[[#r7:]] FPFastMathMode NotNaN|NotInf
; CHECK-SPIRV-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: %[[#double:]] = OpTypeFloat 64

; CHECK-SPIRV:     %[[#r1:]] = OpFAdd %[[#float]]
; CHECK-SPIRV:     %[[#r2]] = OpFAdd %[[#float]]
; CHECK-SPIRV:     %[[#r3]] = OpFAdd %[[#float]]
; CHECK-SPIRV:     %[[#r4]] = OpFAdd %[[#float]]
; CHECK-SPIRV:     %[[#r5]] = OpFAdd %[[#float]]
; CHECK-SPIRV:     %[[#r6]] = OpFAdd %[[#float]]
; CHECK-SPIRV:     %[[#r7]] = OpFAdd %[[#float]]
define spir_kernel void @testFAdd_float(float %a, float %b, ptr addrspace(1) %out) {
entry:
  %r1 = fadd float %a, %b
  store volatile float %r1, ptr addrspace(1) %out
  %r2 = fadd nnan float %a, %b
  store volatile float %r2, ptr addrspace(1) %out
  %r3 = fadd ninf float %a, %b
  store volatile float %r3, ptr addrspace(1) %out
  %r4 = fadd nsz float %a, %b
  store volatile float %r4, ptr addrspace(1) %out
  %r5 = fadd arcp float %a, %b
  store volatile float %r5, ptr addrspace(1) %out
  %r6 = fadd fast float %a, %b
  store volatile float %r6, ptr addrspace(1) %out
  %r7 = fadd nnan ninf float %a, %b
  store volatile float %r7, ptr addrspace(1) %out
  ret void
}

; CHECK-SPIRV:     %[[#]] = OpFAdd %[[#double]]
; CHECK-SPIRV:     %[[#]] = OpFAdd %[[#double]]
; CHECK-SPIRV:     %[[#]] = OpFAdd %[[#double]]
; CHECK-SPIRV:     %[[#]] = OpFAdd %[[#double]]
; CHECK-SPIRV:     %[[#]] = OpFAdd %[[#double]]
; CHECK-SPIRV:     %[[#]] = OpFAdd %[[#double]]
; CHECK-SPIRV:     %[[#]] = OpFAdd %[[#double]]

define spir_kernel void @testFAdd_double(double %a, double %b, ptr addrspace(1) %out) local_unnamed_addr {
entry:
  %r11 = fadd double %a, %b
  store volatile double %r11, ptr addrspace(1) %out
  %r12 = fadd nnan double %a, %b
  store volatile double %r12, ptr addrspace(1) %out
  %r13 = fadd ninf double %a, %b
  store volatile double %r13, ptr addrspace(1) %out
  %r14 = fadd nsz double %a, %b
  store volatile double %r14, ptr addrspace(1) %out
  %r15 = fadd arcp double %a, %b
  store volatile double %r15, ptr addrspace(1) %out
  %r16 = fadd fast double %a, %b
  store volatile double %r16, ptr addrspace(1) %out
  %r17 = fadd nnan ninf double %a, %b
  store volatile double %r17, ptr addrspace(1) %out
  ret void
}
