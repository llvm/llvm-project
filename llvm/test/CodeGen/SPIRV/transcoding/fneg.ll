; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-NOT: OpDecorate %{{.*}} FPFastMathMode
; CHECK-SPIRV: %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV: %[[#r1:]] = OpFNegate %[[#float]]
; CHECK-SPIRV: %[[#r2:]] = OpFNegate %[[#float]]
; CHECK-SPIRV: %[[#r3:]] = OpFNegate %[[#float]]
; CHECK-SPIRV: %[[#r4:]] = OpFNegate %[[#float]]
; CHECK-SPIRV: %[[#r5:]] = OpFNegate %[[#float]]
; CHECK-SPIRV: %[[#r6:]] = OpFNegate %[[#float]]
; CHECK-SPIRV: %[[#r7:]] = OpFNegate %[[#float]]

define spir_kernel void @testFNeg(float %a, ptr addrspace(1) %out) local_unnamed_addr {
entry:
  %r1 = fneg float %a
  store volatile float %r1, ptr addrspace(1) %out
  %r2 = fneg nnan float %a
  store volatile float %r2, ptr addrspace(1) %out
  %r3 = fneg ninf float %a
  store volatile float %r3, ptr addrspace(1) %out
  %r4 = fneg nsz float %a
  store volatile float %r4, ptr addrspace(1) %out
  %r5 = fneg arcp float %a
  store volatile float %r5, ptr addrspace(1) %out
  %r6 = fneg fast float %a
  store volatile float %r6, ptr addrspace(1) %out
  %r7 = fneg nnan ninf float %a
  store volatile float %r7, ptr addrspace(1) %out
  ret void
}
