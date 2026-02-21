; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-NOT: OpDecorate {{.*}} FPFastMathMode
; CHECK-SPIRV-DAG: OpDecorate %[[#r2:]] FPFastMathMode {{NotNaN(\||$)}}
; CHECK-SPIRV-DAG: OpDecorate %[[#r3:]] FPFastMathMode {{NotInf(\||$)}}
; CHECK-SPIRV-DAG: OpDecorate %[[#r4:]] FPFastMathMode {{NSZ(\||$)}}
; CHECK-SPIRV-DAG: OpDecorate %[[#r5:]] FPFastMathMode {{AllowRecip(\||$)}}
; CHECK-SPIRV-DAG: OpDecorate %[[#r6:]] FPFastMathMode {{NotNaN\|NotInf\|NSZ\|AllowRecip\|Fast(\||$)}}
; CHECK-SPIRV-DAG: OpDecorate %[[#r7:]] FPFastMathMode {{NotNaN\|NotInf(\||$)}}
; CHECK-SPIRV-NOT: OpDecorate {{.*}} FPFastMathMode
; CHECK-SPIRV:     %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV:     %[[#r1:]] = OpFSub %[[#float]]
; CHECK-SPIRV:     %[[#r2]] = OpFSub %[[#float]]
; CHECK-SPIRV:     %[[#r3]] = OpFSub %[[#float]]
; CHECK-SPIRV:     %[[#r4]] = OpFSub %[[#float]]
; CHECK-SPIRV:     %[[#r5]] = OpFSub %[[#float]]
; CHECK-SPIRV:     %[[#r6]] = OpFSub %[[#float]]
; CHECK-SPIRV:     %[[#r7]] = OpFSub %[[#float]]

define spir_kernel void @testFSub(float %a, float %b, ptr addrspace(1) %out) local_unnamed_addr {
entry:
  %r1 = fsub float %a, %b
  store volatile float %r1, ptr addrspace(1) %out
  %r2 = fsub nnan float %a, %b
  store volatile float %r2, ptr addrspace(1) %out
  %r3 = fsub ninf float %a, %b
  store volatile float %r3, ptr addrspace(1) %out
  %r4 = fsub nsz float %a, %b
  store volatile float %r4, ptr addrspace(1) %out
  %r5 = fsub arcp float %a, %b
  store volatile float %r5, ptr addrspace(1) %out
  %r6 = fsub fast float %a, %b
  store volatile float %r6, ptr addrspace(1) %out
  %r7 = fsub nnan ninf float %a, %b
  store volatile float %r7, ptr addrspace(1) %out
  ret void
}
