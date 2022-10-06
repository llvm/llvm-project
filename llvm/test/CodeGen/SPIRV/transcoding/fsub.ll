; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV:     OpName %[[#r1:]] "r1"
; CHECK-SPIRV:     OpName %[[#r2:]] "r2"
; CHECK-SPIRV:     OpName %[[#r3:]] "r3"
; CHECK-SPIRV:     OpName %[[#r4:]] "r4"
; CHECK-SPIRV:     OpName %[[#r5:]] "r5"
; CHECK-SPIRV:     OpName %[[#r6:]] "r6"
; CHECK-SPIRV:     OpName %[[#r7:]] "r7"
; CHECK-SPIRV-NOT: OpDecorate %[[#r1]] FPFastMathMode
; CHECK-SPIRV-DAG: OpDecorate %[[#r2]] FPFastMathMode NotNaN
; CHECK-SPIRV-DAG: OpDecorate %[[#r3]] FPFastMathMode NotInf
; CHECK-SPIRV-DAG: OpDecorate %[[#r4]] FPFastMathMode NSZ
; CHECK-SPIRV-DAG: OpDecorate %[[#r5]] FPFastMathMode AllowRecip
; CHECK-SPIRV-DAG: OpDecorate %[[#r6]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|Fast
; CHECK-SPIRV-DAG: OpDecorate %[[#r7]] FPFastMathMode NotNaN|NotInf
; CHECK-SPIRV:     %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV:     %[[#r1]] = OpFSub %[[#float]]
; CHECK-SPIRV:     %[[#r2]] = OpFSub %[[#float]]
; CHECK-SPIRV:     %[[#r3]] = OpFSub %[[#float]]
; CHECK-SPIRV:     %[[#r4]] = OpFSub %[[#float]]
; CHECK-SPIRV:     %[[#r5]] = OpFSub %[[#float]]
; CHECK-SPIRV:     %[[#r6]] = OpFSub %[[#float]]
; CHECK-SPIRV:     %[[#r7]] = OpFSub %[[#float]]

define spir_kernel void @testFSub(float %a, float %b) local_unnamed_addr {
entry:
  %r1 = fsub float %a, %b
  %r2 = fsub nnan float %a, %b
  %r3 = fsub ninf float %a, %b
  %r4 = fsub nsz float %a, %b
  %r5 = fsub arcp float %a, %b
  %r6 = fsub fast float %a, %b
  %r7 = fsub nnan ninf float %a, %b
  ret void
}
