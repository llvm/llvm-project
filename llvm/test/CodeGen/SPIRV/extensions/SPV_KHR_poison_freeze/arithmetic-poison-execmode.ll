; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - -filetype=obj | spirv-val %}

; Poison only appears in poison_helper, which is not an entry point and is not
; reachable from one. The PoisonFreezeKHR capability is nonetheless module-wide,
; so every entry point must carry the ArithmeticPoisonKHR execution mode, while
; the non-entry helper must not.

; CHECK-DAG: OpCapability PoisonFreezeKHR
; CHECK-DAG: OpExtension "SPV_KHR_poison_freeze"

; CHECK: OpEntryPoint Kernel %[[#ENTRY_A:]] "entry_a"
; CHECK: OpEntryPoint Kernel %[[#ENTRY_B:]] "entry_b"
; CHECK: OpExecutionMode %[[#ENTRY_A]] ArithmeticPoisonKHR
; CHECK: OpExecutionMode %[[#ENTRY_B]] ArithmeticPoisonKHR
; CHECK-NOT: OpExecutionMode %[[#]] ArithmeticPoisonKHR

define void @poison_helper(ptr %dst) {
  store i32 poison, ptr %dst
  ret void
}

define spir_kernel void @entry_a(ptr %dst) {
  store i32 0, ptr %dst
  ret void
}

define spir_kernel void @entry_b(ptr %dst) {
  store i32 0, ptr %dst
  ret void
}
