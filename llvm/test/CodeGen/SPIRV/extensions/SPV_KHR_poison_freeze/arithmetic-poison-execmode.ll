; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - -filetype=obj | spirv-val %}

; Verify that the ArithmeticPoisonKHR execution mode is attached to kernel
; entry points when SPV_KHR_poison_freeze is enabled and poison present in the
; module.

; CHECK-DAG:     OpCapability PoisonFreezeKHR
; CHECK-DAG:     OpExtension "SPV_KHR_poison_freeze"
; CHECK-COUNT-2: OpExecutionMode %[[#]] ArithmeticPoisonKHR
; CHECK-NOT:     OpExecutionMode %[[#]] ArithmeticPoisonKHR

define void @poison_helper(ptr %dst) {
  store i32 poison, ptr %dst
  ret void
}

define spir_kernel void @entry_a(ptr %dst) {
  call void @poison_helper(ptr %dst)
  ret void
}

define spir_kernel void @entry_b(ptr %dst) {
  store i32 0, ptr %dst
  ret void
}
