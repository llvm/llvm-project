; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpCapability PoisonFreezeKHR
; CHECK-NOT: OpExtension "SPV_KHR_poison_freeze"
; CHECK-NOT: ArithmeticPoisonKHR

define spir_kernel void @kernel_no_poison(ptr %dst, i32 %v) {
  store i32 %v, ptr %dst
  ret void
}
