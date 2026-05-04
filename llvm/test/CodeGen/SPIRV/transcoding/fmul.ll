; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-NOT: OpDecorate %[[#]] FPFastMathMode
; CHECK-SPIRV-DAG: OpDecorate %[[#r2:]] FPFastMathMode {{NotNaN(\||$)}}
; CHECK-SPIRV-DAG: OpDecorate %[[#r3:]] FPFastMathMode {{NotInf(\||$)}}
; CHECK-SPIRV-DAG: OpDecorate %[[#r4:]] FPFastMathMode {{NSZ(\||$)}}
; CHECK-SPIRV-DAG: OpDecorate %[[#r5:]] FPFastMathMode {{AllowRecip(\||$)}}
; CHECK-SPIRV-DAG: OpDecorate %[[#r6:]] FPFastMathMode {{NotNaN\|NotInf\|NSZ\|AllowRecip\|Fast(\||$)}}
; CHECK-SPIRV-DAG: OpDecorate %[[#r7:]] FPFastMathMode {{NotNaN\|NotInf(\||$)}}
; CHECK-SPIRV:     %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV:     %[[#r1:]] = OpFMul %[[#float]]
; CHECK-SPIRV:     %[[#r2]] = OpFMul %[[#float]]
; CHECK-SPIRV:     %[[#r3]] = OpFMul %[[#float]]
; CHECK-SPIRV:     %[[#r4]] = OpFMul %[[#float]]
; CHECK-SPIRV:     %[[#r5]] = OpFMul %[[#float]]
; CHECK-SPIRV:     %[[#r6]] = OpFMul %[[#float]]
; CHECK-SPIRV:     %[[#r7]] = OpFMul %[[#float]]

define spir_kernel void @testFMul(float %a, float %b, ptr addrspace(1) %out) local_unnamed_addr {
entry:
  %r1 = fmul float %a, %b
  store volatile float %r1, ptr addrspace(1) %out
  %r2 = fmul nnan float %a, %b
  store volatile float %r2, ptr addrspace(1) %out
  %r3 = fmul ninf float %a, %b
  store volatile float %r3, ptr addrspace(1) %out
  %r4 = fmul nsz float %a, %b
  store volatile float %r4, ptr addrspace(1) %out
  %r5 = fmul arcp float %a, %b
  store volatile float %r5, ptr addrspace(1) %out
  %r6 = fmul fast float %a, %b
  store volatile float %r6, ptr addrspace(1) %out
  %r7 = fmul nnan ninf float %a, %b
  store volatile float %r7, ptr addrspace(1) %out
  ret void
}
