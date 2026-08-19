; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - -filetype=obj | spirv-val %}

; The entry point already requests ArithmeticPoisonKHR through
; !spirv.ExecutionMode. The capability-driven path must not add a second,
; duplicate execution mode for it.

; CHECK-DAG: OpCapability PoisonFreezeKHR
; CHECK-DAG: OpExtension "SPV_KHR_poison_freeze"
; CHECK: OpEntryPoint Kernel %[[#ENTRY:]] "kernel_with_poison"
; CHECK-COUNT-1: OpExecutionMode %[[#ENTRY]] ArithmeticPoisonKHR
; CHECK-NOT: OpExecutionMode %[[#ENTRY]] ArithmeticPoisonKHR

define spir_kernel void @kernel_with_poison(ptr %dst) {
  store i32 poison, ptr %dst
  ret void
}

!spirv.ExecutionMode = !{!0}
!0 = !{ptr @kernel_with_poison, i32 5157}
