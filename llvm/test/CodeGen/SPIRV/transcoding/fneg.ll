; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpName %[[#r1:]] "r1"
; CHECK-SPIRV: OpName %[[#r2:]] "r2"
; CHECK-SPIRV: OpName %[[#r3:]] "r3"
; CHECK-SPIRV: OpName %[[#r4:]] "r4"
; CHECK-SPIRV: OpName %[[#r5:]] "r5"
; CHECK-SPIRV: OpName %[[#r6:]] "r6"
; CHECK-SPIRV: OpName %[[#r7:]] "r7"
; CHECK-SPIRV-NOT: OpDecorate %{{.*}} FPFastMathMode
; CHECK-SPIRV: %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV: %[[#r1]] = OpFNegate %[[#float]]
; CHECK-SPIRV: %[[#r2]] = OpFNegate %[[#float]]
; CHECK-SPIRV: %[[#r3]] = OpFNegate %[[#float]]
; CHECK-SPIRV: %[[#r4]] = OpFNegate %[[#float]]
; CHECK-SPIRV: %[[#r5]] = OpFNegate %[[#float]]
; CHECK-SPIRV: %[[#r6]] = OpFNegate %[[#float]]
; CHECK-SPIRV: %[[#r7]] = OpFNegate %[[#float]]

define spir_kernel void @testFNeg(float %a) local_unnamed_addr {
entry:
  %r1 = fneg float %a
  %r2 = fneg nnan float %a
  %r3 = fneg ninf float %a
  %r4 = fneg nsz float %a
  %r5 = fneg arcp float %a
  %r6 = fneg fast float %a
  %r7 = fneg nnan ninf float %a
  ret void
}
