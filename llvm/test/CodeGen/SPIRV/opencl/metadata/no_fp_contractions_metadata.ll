; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: OpEntryPoint Kernel %[[#ENTRY:]] "foo"
; CHECK: OpExecutionMode %[[#ENTRY]] FPFastMathDefault %[[#FP16:]] 0
; CHECK: OpExecutionMode %[[#ENTRY]] FPFastMathDefault %[[#FP32:]] 0
; CHECK: OpExecutionMode %[[#ENTRY]] FPFastMathDefault %[[#FP64:]] 0
; CHECK: OpExecutionMode %[[#ENTRY]] FPFastMathDefault %[[#FP128:]] 0
; CHECK: %[[#FP16]] = OpTypeFloat 16
; CHECK: %[[#FP32]] = OpTypeFloat 32
; CHECK: %[[#FP64]] = OpTypeFloat 64
; CHECK: %[[#FP128]] = OpTypeFloat 128
define spir_kernel void @foo(half %h, float %f, double %d, fp128 %fp) {
entry:
  ret void
}
